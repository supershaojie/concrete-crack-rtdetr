"""One bounded native CUDA lifecycle, at most 16 training batches total."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import tempfile
import time
import traceback

import torch
from rdm_common import (ROOT, MAIN, NEW, require, paths, recipe, runtime, identity, verify_model,
                        training_rebuild, write_json, sha256, torch_load)
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import load_checkpoint


class RDMTrainer(RTDETRTrainer):
    """Keep native loading, optimization and training; audit common initialization."""
    variant = "cbr_lif_rdm_v1"

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(weights is not None, "RDM start/resume requires explicit controlled/native weights")
        model, self.rdm_loading = training_rebuild(cfg, weights, self.data, self.variant, fresh=not bool(self.args.resume))
        return model

    def _setup_train(self):
        # Locate existing mother's AMP probe assets. Native check_amp is unchanged;
        # only its resource directory and cwd are scoped to owned temporary links.
        with tempfile.TemporaryDirectory(prefix="rdm-v1-amp-", dir=self.save_dir) as folder:
            with amp_resources(Path(folder)):
                return super()._setup_train()


class BoundedStop(Exception):
    pass


class BudgetPending(Exception):
    pass


def find_amp_resources():
    candidates = {
        "yolo26n.pt": [MAIN / "yolo26n.pt", MAIN / "weights/yolo26n.pt", ROOT / "yolo26n.pt"],
        "bus.jpg": [MAIN / "ultralytics-main/ultralytics/assets/bus.jpg", MAIN / "bus.jpg", ROOT / "ultralytics-main/ultralytics/assets/bus.jpg"],
    }
    result = {name: next((p.resolve() for p in choices if p.is_file()), None) for name, choices in candidates.items()}
    missing = [name for name, p in result.items() if p is None]
    if missing:
        raise BudgetPending("Existing native AMP probe resources missing: " + ", ".join(missing) + "; reuse mother's copies in RDM_MAIN, no automatic download")
    return result


@contextmanager
def amp_resources(folder):
    from unittest.mock import patch
    from ultralytics.utils import checks
    resources = find_amp_resources()
    for name, source in resources.items():
        (folder / name).symlink_to(source)
    previous = Path.cwd()
    try:
        os.chdir(folder)
        with patch.object(checks, "ASSETS", folder):
            yield
    finally:
        os.chdir(previous)


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return True


def same_saved(a, b):
    """Native save/load dtype casting only; no pre-quantization equivalence claim."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a.cpu().to(b.dtype), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_saved(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_saved(x, y) for x, y in zip(a, b))
    return a == b


def optimizer_inventory(trainer):
    params = dict(trainer.model.named_parameters())
    flat = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    require(all(p.requires_grad for p in params.values()), "Unexpected frozen model parameter")
    require(len(flat) == len({id(p) for p in flat}) == len(params), "Native optimizer missing/duplicate parameter")
    require({id(p) for p in flat} == {id(p) for p in params.values()}, "Optimizer parameter coverage differs")
    return dict(all_parameters_once=True, all_parameters_trainable=True, groups=len(trainer.optimizer.param_groups),
                rdm={n: next(i for i, g in enumerate(trainer.optimizer.param_groups) if any(p is params[n] for p in g["params"])) for n in sorted(NEW)})


def shutdown_loaders(trainer):
    # Ensure native persistent workers release the owned temporary directory.
    for name in ("train_loader", "test_loader"):
        loader = getattr(trainer, name, None)
        iterator = getattr(loader, "iterator", None)
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def bounded_training(args, variant, report, stage, update_target, batch_limit, persist, checkpoint=None):
    row = report[stage] = dict(status="RUNNING", batches=0, effective_updates=0, scaler_skips=0, steps=[], upstream_active={},
                              rdm_calls=0, dn_batches=0, consecutive_nonfinite_steps=0)
    handles = []
    persist(stage + ".setup")

    class ProbeTrainer(RDMTrainer):
        def optimizer_step(self):
            params = dict(self.model.named_parameters())
            wo = params["model.5.rdm.Wo.weight"]
            old = wo.detach().clone()
            before_step = float(self.optimizer.state.get(wo, {}).get("step", 0))
            scale = self.scaler.get_scale()
            all_gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values())
            gradients = {n: dict(finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
                                  nonzero=p.grad is not None and bool(torch.count_nonzero(p.grad)),
                                  max_abs_unscaled=float(p.grad.abs().max()/scale) if p.grad is not None and torch.isfinite(p.grad).all() else None)
                         for n, p in params.items() if n in NEW}
            super().optimizer_step()  # Native unscale, clip, scaler step/update, zero_grad, EMA.
            after_step = float(self.optimizer.state.get(wo, {}).get("step", 0))
            skipped = after_step == before_step
            delta = float((wo.detach()-old).abs().max())
            effective = not skipped and delta > 0 and gradients["model.5.rdm.Wo.weight"]["finite"] and gradients["model.5.rdm.Wo.weight"]["nonzero"]
            require(torch.isfinite(wo).all(), "Nonfinite updated Wo")
            row["scaler_skips"] += int(skipped)
            row["consecutive_nonfinite_steps"] = row["consecutive_nonfinite_steps"] + 1 if not all_gradients_finite else 0
            row["effective_updates"] += int(effective)
            if effective:
                require(all(g["finite"] for g in gradients.values()), "Nonfinite effective RDM gradient")
                for n, g in gradients.items():
                    row["upstream_active"][n] = row["upstream_active"].get(n, False) or g["nonzero"]
            row["steps"].append(dict(batch=row["batches"], accumulate=self.accumulate, optimizer_step_before=before_step,
                                     optimizer_step_after=after_step, scale_before=scale, scale_after=self.scaler.get_scale(),
                                     skipped=skipped, effective=effective, Wo_max_change=delta, gradients=gradients))
            require(row["consecutive_nonfinite_steps"] < 3, "Three consecutive nonfinite optimizer attempts; no retry or AMP fallback")

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            require(batch["img"].shape == (16, 3, 640, 640), "B16/640 contract changed")
            require(batch["cls"].numel() > 0 and finite(batch["bboxes"]), "Real batch needs valid GT")
            row["last_GT_count"] = batch["cls"].numel()
            return batch

        def final_eval(self):
            raise RuntimeError("Preflight must never enter final_eval")

        def validate(self):
            raise RuntimeError("Preflight must escape before automatic epoch validation")

    ProbeTrainer.variant = variant
    trainer = ProbeTrainer(overrides=args)
    def setup(t):
        require(t.amp and t.args.batch == 16 and t.args.imgsz == 640, "Native AMP capacity settings changed")
        require(t.model.model[26].num_denoising > 0, "Native DN missing")
        row["optimizer"] = optimizer_inventory(t)
        row["trainer_loading"] = t.rdm_loading
        row["native_amp"] = True
        row["native_DN"] = t.model.model[26].num_denoising
        def rdm_executed(m, a, out):
            require(finite(out), "Nonfinite native AMP RDM output")
            row["rdm_calls"] += 1
        def dn_executed(m, a, out):
            require(m.training and out[4] is not None and out[4]["dn_num_split"][0] > 0, "Real native DN not exercised")
            require(finite(out), "Nonfinite native decoder output")
            row["dn_batches"] += 1
        handles.extend([t.model.model[5].rdm.register_forward_hook(rdm_executed), t.model.model[26].register_forward_hook(dn_executed)])
        if checkpoint is not None:
            saved = torch_load(checkpoint, map_location="cpu")
            selected = saved["ema"] if saved.get("ema") is not None else saved["model"]
            require(all(same_saved(v, t.model.state_dict()[k]) for k, v in selected.state_dict().items()), "Resume did not load selected quantized EMA/model")
            require(same_saved(saved["optimizer"], t.optimizer.state_dict()), "Resume optimizer differs from native saved state")
            require(same_saved(saved["scaler"], t.scaler.state_dict()), "Resume scaler differs")
            require(t.ema.updates == saved["updates"] and t.start_epoch == saved["epoch"] + 1, "Native resume epoch/EMA count differs")
            require(all(same_saved(v, t.ema.ema.state_dict()[k]) for k, v in selected.state_dict().items()), "Resume EMA weights differ")
            row["native_resume_state_exact"] = True
        persist(stage + ".training")

    def before_batch(t):
        t._oom_retries = 3  # Preserve B16: no native automatic shrinking or retry.
        if row["batches"] >= batch_limit or report["training_batches"] >= 16:
            raise BudgetPending("Training batch budget exhausted")
        row["batches"] += 1
        report["training_batches"] += 1
        persist(stage + ".batch")

    def after_batch(t):
        require(finite(t.loss) and finite(t.loss_items), "Nonfinite native training loss")
        require(row["rdm_calls"] == row["dn_batches"] == row["batches"], "Missing/duplicated RDM or DN execution")
        row["last_loss"] = float(t.loss.detach())
        persist(stage + ".batch_completed")
        if row["effective_updates"] >= update_target and all(row["upstream_active"].get(n, False) for n in NEW):
            row["status"] = "PASSED"
            raise BoundedStop()
        if row["batches"] >= batch_limit or report["training_batches"] >= 16:
            raise BudgetPending("Effective updates/branch gradients not reached within fixed batch budget")

    trainer.add_callback("on_pretrain_routine_end", setup)
    trainer.add_callback("on_train_batch_start", before_batch)
    trainer.add_callback("on_train_batch_end", after_batch)
    try:
        trainer.train()
        raise RuntimeError("Native loop returned without bounded stop")
    except BoundedStop:
        return trainer
    except BudgetPending:
        row["status"] = "PENDING"
        shutdown_loaders(trainer)
        raise
    except BaseException:
        row["status"] = "FAILED"
        shutdown_loaders(trainer)
        raise
    finally:
        for handle in handles:
            handle.remove()


class OneBatch:
    def __init__(self, loader):
        self.loader, self.dataset = loader, loader.dataset

    def __len__(self):
        return 1

    def __iter__(self):
        yield next(iter(self.loader))


def server_preflight(variant, destination):
    started = time.monotonic()
    report = dict(status="RUNNING", phase="environment", variant=variant, training_batches=0,
                  start={"status": "NOT_RUN"}, save_reload={"status": "NOT_RUN"}, resume={"status": "NOT_RUN"},
                  half_ema_val={"status": "NOT_RUN"}, backend_fp32={"status": "NOT_RUN"}, backend_half={"status": "NOT_RUN"},
                  formal_training="NOT_STARTED", final_test="NOT_RUN", temporary_files_cleaned=False)
    def persist(phase):
        report["phase"] = phase
        write_json(destination, report)
    owner = None
    trainers = []
    try:
        persist("environment")
        report["environment"] = runtime()
        info = report["environment"]
        p = paths(variant)
        report["identity"] = identity(variant, include_data=p["data"].is_file())
        missing = [str(p[k]) for k in ("source", "init", "data") if not p[k].is_file()]
        if missing:
            raise BudgetPending("Required input missing: " + ", ".join(missing))
        if not torch.cuda.is_available() or str(torch.__version__) != "2.1.2+cu121" or "4090" not in (info["gpu"] or ""):
            raise BudgetPending("Requires existing server RTX4090 / torch2.1.2+cu121; no dependency upgrades or smaller batch")
        report["amp_resources"] = {n: dict(path=str(p), sha256=sha256(p)) for n, p in find_amp_resources().items()}
        require(report["identity"]["source_sha256"] == __import__("rdm_common").SOURCE_SHA256, "Source hash changed")
        local = __import__("json").loads((p["output"] / "local.json").read_text())
        require(local["status"] == "PASSED" and local["code"] == report["identity"]["code"], "Run current local checks first")
        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory(prefix="rdm-v1-preflight-", dir=p["output"]) as folder:
            owner = Path(folder)
            write_json(owner / "owner.json", dict(tool="preflight_rdm", pid=__import__("os").getpid(), variant=variant))
            args, _ = recipe(variant)
            args.pop("save_dir", None)
            args.update(project=str(owner), name="native", plots=False)
            # Native AMP probing may use its standard resources. It does not change RDM dtype.
            # No source/model monkeypatch, no disable-AMP fallback, no library upgrade.
            first = bounded_training(args, variant, report, "start", 2, 8, persist)
            trainers.append(first)
            report["save_reload"] = dict(status="RUNNING")
            persist("save_reload")
            require(torch.count_nonzero(first.model.model[5].rdm.Wo.weight) > 0, "Zero-Wo cannot satisfy lifecycle")
            first.ema.update_attr(first.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            first.save_model()  # Unmodified native format, including optional last/best duplication.
            last = first.last
            saved = torch_load(last, map_location="cpu")
            selected = saved["ema"] if saved.get("ema") is not None else saved["model"]
            require(selected is not None and torch.count_nonzero(selected.model[5].rdm.Wo.weight) > 0, "Saved EMA lost nonzero RDM")
            require(all(same_saved(v, selected.state_dict()[k]) for k, v in deepcopy(first.ema.ema).half().state_dict().items()), "Native half-save source mismatch")
            restored, _ = load_checkpoint(last, device="cpu", fuse=False)
            require(all(same_saved(v, restored.state_dict()[k]) for k, v in selected.state_dict().items()), "Native reload mismatch")
            report["save_reload"] = dict(status="PASSED", fields=sorted(saved), selected="ema" if saved.get("ema") is not None else "model",
                                         sha256=sha256(last), bytes=last.stat().st_size, same_quantized_source_exact=True, temporary=True)
            if first.best.exists():
                first.best.unlink()  # Owned temporary duplicate only; preserve last for actual resume.
            shutdown_loaders(first)
            trainers.remove(first)
            del first, saved, selected, restored
            gc.collect(); torch.cuda.empty_cache()
            second = bounded_training(dict(args, model=str(last), resume=str(last), exist_ok=True), variant, report, "resume", 1,
                                      16-report["training_batches"], persist, checkpoint=last)
            trainers.append(second)
            persist("half_ema_val")
            second.ema.update_attr(second.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            val_row = report["half_ema_val"] = dict(status="RUNNING", batches=0, rdm_calls=0)
            def val_rdm(m, a, out):
                require(out.dtype == torch.float16 and finite(out), "Half EMA RDM output invalid")
                require(torch.count_nonzero(m.Wo.weight) > 0, "Half EMA RDM zero")
                val_row["rdm_calls"] += 1
            def val_batch(v):
                val_row["batches"] += 1
                require(val_row["batches"] <= 1 and v.training and v.args.half, "Expected one native half EMA validation batch")
            handle = second.ema.ema.model[5].rdm.register_forward_hook(val_rdm)
            def val_output(m, a, out):
                require(finite(out), "Nonfinite native half EMA prediction")
                val_row["prediction_shape"] = list(out[0].shape)
            output_handle = second.ema.ema.register_forward_hook(val_output)
            second.validator.dataloader = OneBatch(second.test_loader)
            second.validator.add_callback("on_val_batch_end", val_batch)
            values = RTDETRTrainer.validate(second)
            handle.remove()
            output_handle.remove()
            require(val_row["batches"] == 1 and val_row["rdm_calls"] == 1 and all(__import__("math").isfinite(v) for v in values[0].values()), "Native half EMA val failed")
            val_row.update(status="PASSED", finite=True, scope="one real val batch; metrics are not full validation results")
            shutdown_loaders(second)
            trainers.remove(second)
            del second
            gc.collect(); torch.cuda.empty_cache()
            for half in (False, True):
                key = "backend_half" if half else "backend_fp32"
                persist(key)
                row = report[key] = dict(status="RUNNING", rdm_calls=0, checkpoint_sha256=sha256(last))
                backend = AutoBackend(model=str(last), device=torch.device("cuda:0"), fp16=half, fuse=True, verbose=False)
                verify_model(backend.model, variant)
                def executed(m, a, out):
                    require(finite(out) and torch.count_nonzero(m.Wo.weight) > 0, "AutoBackend lost nonzero RDM")
                    row["rdm_calls"] += 1
                handle = backend.model.model[5].rdm.register_forward_hook(executed)
                with torch.no_grad():
                    result = backend(torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16 if half else torch.float32))
                handle.remove()
                require(finite(result) and row["rdm_calls"] == 1, "AutoBackend nonfinite/lost branch")
                prediction = result[0] if isinstance(result, (tuple, list)) else result
                require(prediction.shape == (1, 300, 5), "AutoBackend output shape differs")
                row.update(status="PASSED", shape=list(prediction.shape), dtype=str(prediction.dtype), finite=True)
                del result, prediction, backend
                gc.collect(); torch.cuda.empty_cache()
        report["status"] = "PASSED"
    except BudgetPending as error:
        report.update(status="PENDING", error=str(error))
        active = report.get(report["phase"].split(".")[0])
        if isinstance(active, dict) and active.get("status") == "RUNNING":
            active["status"] = "PENDING"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(limit=8))
        active = report.get(report["phase"].split(".")[0])
        if isinstance(active, dict) and active.get("status") == "RUNNING":
            active["status"] = "FAILED"
        raise
    finally:
        for trainer in trainers:
            shutdown_loaders(trainer)
        gc.collect()
        report.update(temporary_files_cleaned=owner is None or not owner.exists(), seconds=time.monotonic()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                      cleanup_note="TemporaryDirectory handles normal exit, exceptions and catchable interruption; SIGKILL/power loss cannot guarantee cleanup")
        write_json(destination, report)
    return report
