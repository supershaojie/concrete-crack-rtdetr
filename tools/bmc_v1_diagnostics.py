"""Bounded capacity and read-only trained-mother opportunity diagnostics."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import time

from bmc_v1_common import ROOT, OUT, INIT, MODEL, MAIN, require, read, write, sha, runtime, research, utc
import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import autocast
from ultralytics.models.utils.bmc import bounded_consensus
from ultralytics.models.utils.ops import HungarianMatcher
from bmc_v1_training import BMCTrainer, EpochCollector


def step_numbers(optimizer):
    return {id(p): float(optimizer.state[p].get("step", 0)) for group in optimizer.param_groups for p in group["params"]}


def capacity(report_path):
    """Real B16/640 at epoch20, native groups/accumulation/AMP, <=16 microbatches/2 updates."""
    started = time.monotonic()
    report = dict(status="FAIL", scope="B16/640 server capacity", runtime=runtime(), diagnostic_only_scale=128,
                  max_micro_batches=16, max_seconds=900, effective_updates=0, overflow_skips=0, batches=[])
    try:
        require(torch.cuda.is_available(), "CUDA required for formal capacity check")
        require(torch.cuda.get_device_properties(0).total_memory >= 16 * 1024**3,
                "Server capacity remains PENDING on this small local GPU; do not reduce B16/640")
        args = YAML.load(OUT / "train_args.yaml")
        with tempfile.TemporaryDirectory(prefix="capacity_", dir=OUT) as temporary:
            args.update(project=temporary, name="isolated_capacity", save_dir=str(Path(temporary)/"isolated_capacity"),
                        plots=False, val=False, save=False)
            trainer = BMCTrainer(overrides=args)
            trainer.audit_folder = Path(temporary)
            trainer._setup_train()  # includes real native check_amp and native AdamW groups
            require(trainer.amp and trainer.batch_size == 16 and trainer.args.imgsz == 640, "Formal capacity recipe changed")
            require(isinstance(trainer.optimizer, torch.optim.AdamW) and trainer.accumulate == 4, "Optimizer/accumulation changed")
            trainer.scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128)
            trainer.epoch = 20
            trainer._model_train()
            trainer._bmc_epoch_start(trainer)
            collector = EpochCollector()
            trainer.model.criterion.collector = collector
            for group in trainer.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * trainer.lf(20)
            trainer.optimizer.zero_grad()
            selected = {n: p for n, p in trainer.model.named_parameters() if n in (
                "model.0.conv.weight", "model.20.O_proj.weight", "model.26.dec_bbox_head.1.layers.2.weight", "model.26.cbr.offset_out.weight")}
            initial = {n: p.detach().clone() for n, p in selected.items()}
            shapes = {}
            handles = [trainer.model.model[i].register_forward_hook(
                lambda module, inputs, output, i=i: shapes.update({str(i): list(output.shape)})) for i in (19, 20, 22, 25)]
            torch.cuda.reset_peak_memory_stats()
            iterator = iter(trainer.train_loader)
            last_opt_step = -1
            for step in range(16):
                if time.monotonic()-started >= 900:
                    break
                batch = trainer.preprocess_batch(next(iterator))
                require(list(batch["img"].shape) == [16, 3, 640, 640], "Unexpected real batch shape")
                with autocast(trainer.amp):
                    loss, main = trainer.model(batch)
                record = dict(step=step, shape=list(batch["img"].shape), loss=float(loss.detach()),
                              main_losses=main.detach().cpu().tolist(), scale_before=trainer.scaler.get_scale())
                report["batches"].append(record)
                require(bool(torch.isfinite(loss)), "Nonfinite diagnostic loss")
                trainer.scaler.scale(loss).backward()
                if step - last_opt_step >= trainer.accumulate:
                    before = step_numbers(trainer.optimizer)
                    trainer.scaler.unscale_(trainer.optimizer)
                    gradients = {n: dict(finite=bool(torch.isfinite(p.grad).all()), norm=float(p.grad.float().norm()))
                                 for n, p in trainer.model.named_parameters() if p.grad is not None}
                    record["gradients"] = {n: gradients.get(n) for n in selected}
                    finite = all(r["finite"] for r in gradients.values())
                    require(all(n in gradients for n in selected), "Related original parameters have no gradients")
                    # Native clipping/scaler/EMA sequence; count Adam step advancement, never scaler.step invocation.
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10.)
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                    after = step_numbers(trainer.optimizer)
                    advanced = any(after[k] > before[k] for k in before)
                    if advanced:
                        require(finite, "Nonfinite gradient produced an optimizer update")
                        require(all(gradients[n]["norm"] > 0 for n in selected), "Zero relevant gradient")
                        report["effective_updates"] += 1
                    else:
                        report["overflow_skips"] += 1
                    record.update(effective_update=advanced, all_gradients_finite=finite, scale_after=trainer.scaler.get_scale())
                    trainer.optimizer.zero_grad()
                    trainer.ema.update(trainer.model)
                    last_opt_step = step
                if report["effective_updates"] >= 2:
                    break
            for handle in handles:
                handle.remove()
            changes = {n: float((p.detach()-initial[n]).abs().max()) for n, p in selected.items()}
            require(report["effective_updates"] >= 1 and all(v > 0 for v in changes.values()), "No verified effective update/parameter change")
            require(shapes["19"] == [16, 256, 80, 80], "Neck P3 contract changed")
            report.update(status="PASS", amp=trainer.amp, optimizer=type(trainer.optimizer).__name__, accumulate=trainer.accumulate,
                          parameters=sum(p.numel() for p in trainer.model.parameters()), states=len(trainer.model.state_dict()),
                          shapes=shapes, parameter_max_changes=changes,
                          peak_memory_bytes=torch.cuda.max_memory_allocated(), mechanism=collector.summary(),
                          initialization=trainer.initialization_audit,
                          diagnostic_state_discarded=True, native_initial_scale_adaptation="NOT_CLAIMED")
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        write(report_path, report)
    return report


@torch.no_grad()
def diagnostic_predictions(model, images):
    """Isolated model in eval; only head/decoder dispatch flags expose all native decoder outputs.

    All BN/dropout children stay eval. No duplicate forward, DN, optimizer, or caller mutation.
    """
    model.eval()
    head = model.model[-1]
    head.training = True
    head.decoder.training = True
    saved, x = [], images
    for module in model.model[:-1]:
        if module.f != -1:
            x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
        x = module(x)
        saved.append(x if module.i in model.save else None)
    raw, details = head.forward_with_diagnostics([saved[j] for j in head.f], batch=None)
    require(raw[0].shape[0] == 3 and raw[0].shape[2] == 300 and raw[4] is None, "Probe decoder/DN topology changed")
    require(torch.equal(raw[0][-1], details["after"]), "Probe did not use final CBR boxes")
    return raw, details


def probe(best=None, count=64, device="0"):
    import cv2
    torch.set_num_threads(4)
    from init_c19_lif_v1 import verify_model
    require(1 <= count <= 64, "Probe is bounded to 64 train images")
    best = Path(best) if best else MAIN / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt"
    report = dict(status="PENDING", scope="read-only trained-mother opportunity; not engineering PASS or performance",
                  weights=str(best), requested_images=count, split="train", imgsz=640, augmentation=False,
                  runtime=runtime(), device=device, precision="FP32", preprocessing="cv2 linear stretch 640, BGR to RGB, /255")
    path = OUT / "probe.json"
    if not best.is_file():
        report["reason"] = "Trained mother best not available; no random-weight substitute"
        write(path, report)
        return report
    require((OUT / "prepare.json").is_file(), "Run prepare first")
    prepared = read(OUT / "prepare.json")
    dataset = Path(prepared["data"]["root"])
    expected_sha = read(ROOT / "docs/bmc_v1/parent_assets.json")["best_sha256"]
    require(sha(best) == expected_sha, "Probe checkpoint is not the verified trained mother best")
    chosen = [r for r in prepared["data"]["images"] if r["path"].startswith("images/train/")][:count]
    require(len(chosen) == count, "Missing fixed train images")
    torch_device = torch.device("cpu" if device == "cpu" else "cuda:" + str(device).replace("cuda:", ""))
    # A new loader object owns this diagnostic copy; it cannot change a training model.
    model = RTDETR(str(best)).model.float().to(torch_device).eval()
    verify_model(model)
    before = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
    collector = EpochCollector()
    collector.reset(20)
    records, cbr_nonzero = [], 0
    started = time.monotonic()
    try:
        for record in chosen:
            image_path = dataset / record["path"]
            require(sha(image_path) == record["sha256"], "Probe image content changed")
            label = dataset / record["path"].replace("images/", "labels/", 1)
            label = label.with_suffix(".txt")
            rows = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            image = cv2.imread(str(image_path))
            require(image is not None, "Unreadable train image")
            tensor = torch.from_numpy(cv2.resize(image, (640, 640))[:, :, ::-1].copy()).permute(2, 0, 1)[None].to(torch_device).float()/255
            gt = torch.tensor([r[1:] for r in rows], device=torch_device).reshape(-1, 4)
            cls = torch.tensor([int(r[0]) for r in rows], device=torch_device, dtype=torch.long)
            raw, detail = diagnostic_predictions(model, tensor)
            a2, costs = matcher(raw[0][1], raw[1][1], gt, cls, [len(rows)], return_costs=True)
            a3, _ = matcher(raw[0][2], raw[1][2], gt, cls, [len(rows)], return_costs=True)
            _, mechanism = bounded_consensus(a2, a3, costs, [len(rows)], research(), detail=True)
            mechanism["state"] = "ENABLED"
            collector.record(20, mechanism, {})
            cbr_nonzero += int(not torch.equal(detail["before"], detail["after"]))
            records.append(dict(**record, label_sha256=sha(label), gt=len(rows),
                                mechanism=mechanism["images"][0]))
        require(all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items()), "Probe updated BN/model state")
        require(sha(best) == expected_sha, "Probe modified source weights")
        summary = collector.summary()
        report.update(status="COMPLETED", weights_sha256=expected_sha, images=records, summary=summary,
            state_unchanged=True, final_cbr_nonzero_images=cbr_nonzero, elapsed_seconds=time.monotonic()-started,
            conclusion="本次探针未发现活跃机制" if summary["changed"] == 0 else "本次探针观察到实际替换；不构成 AP 结论")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), images=records)
        raise
    finally:
        write(path, report)
    return report
