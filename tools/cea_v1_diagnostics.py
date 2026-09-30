"""Bounded isolated real-data preflight and mother-checkpoint CEA diagnosis."""
from __future__ import annotations

from copy import deepcopy
import gc
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from cea_v1_common import (ROOT, OUT, RUN, MAIN, MODEL, INIT, algorithm, runtime, sha256, require,
                           read_json, write_json, prepared_identity)
from ultralytics.models.rtdetr.cea import CEADetectionModel, summarize
from ultralytics.models.rtdetr.cea_trainer import CEATrainer, CEABudgetStop, strict_load
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import init_seeds


def finite_parameters(model):
    return all(torch.isfinite(p).all() for p in model.parameters())


def preflight(folder, max_batches=16, seconds=900):
    require(1 <= max_batches <= 16 and 0 < seconds <= 900, "Preflight bound exceeded")
    started = time.monotonic()
    prepared, identity = prepared_identity()
    report = dict(status="PENDING", identity=identity, runtime=runtime(), max_microbatches=max_batches,
                  budget_seconds=seconds, epoch_context="CEA only e=20; optimizer/warmup real epoch0", folder=str(folder))
    trainer = None
    try:
        require(torch.cuda.is_available(), "PENDING: real B16/640/AMP preflight requires CUDA")
        # CPU exact whole-model gradient/update equivalence and CUDA local contracts.
        for label, device, options in (("cpu", "cpu", []), ("cuda", "cuda:0", ["--core-only"])):
            output = folder / f"{label}_checks.json"
            command = [sys.executable, str(ROOT / "tools/check_cea_v1.py"), "--device", device, "--output", str(output), *options]
            remaining = seconds - (time.monotonic() - started)
            require(remaining > 0, "PENDING: budget exhausted before integration checks")
            # Native CPU select_device mutates CUDA_VISIBLE_DEVICES. Keep its
            # lifecycle fixture in a child process so real AMP training is untouched.
            with (folder / f"{label}_checks.log").open("x", encoding="utf-8") as log:
                try: result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                except subprocess.TimeoutExpired: raise RuntimeError("PENDING: integration-check time budget exhausted")
            report[label + "_checks"] = read_json(output) if output.exists() else {"status": "FAIL", "reason": "missing report"}
            require(result.returncode == 0 and report[label + "_checks"]["status"] == "PASS", label + " integration contract failure")
        gc.collect(); torch.cuda.empty_cache()
        cfg = deepcopy(prepared["recipe"])
        cfg.update(project=str(folder), name="isolated_run", save_dir=str(folder / "isolated_run"))
        trainer = CEATrainer(overrides=cfg, cea_config=algorithm(), cea_identity=identity,
                             cea_budget=dict(batches=max_batches, seconds=max(1, seconds - (time.monotonic() - started))))
        torch.cuda.reset_peak_memory_stats()
        try: trainer.train()
        except CEABudgetStop as stop: report["stop_reason"] = str(stop)
        require(trainer.batch_size == 16 and trainer.args.imgsz == 640 and trainer.args.nbs == 64, "Formal preflight dimensions changed")
        require(trainer.amp and trainer.scaler.is_enabled(), "AMP silently disabled; formal protocol requires AMP")
        report.update(microbatches=trainer.cea_batches, steps=trainer.cea_steps,
                      effective_updates=sum(s["effective_update"] for s in trainer.cea_steps),
                      skipped_updates=sum(s["skipped"] for s in trainer.cea_steps),
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(), amp=trainer.amp,
                      actual_args=vars(trainer.args), ema_updates=trainer.ema.updates)
        path = trainer.save_dir / "cea_batches.jsonl"
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        report["cea_activation"] = dict(batches=len(rows), coefficients=sorted({r["coefficient"] for r in rows}),
                                        raw_loss=[r["raw_loss"] for r in rows], weighted_loss=[r["weighted_loss"] for r in rows],
                                        zero_head_start="expected; no initialization altered")
        valid = report["effective_updates"] > 0 and bool(rows) and all(r["coefficient"] == .1 for r in rows)
        valid &= all(s["parameters_finite"] and (s["gradients_finite"] or s["skipped"]) for s in trainer.cea_steps)
        # Native checkpoint/resume exercise with the REAL CUDA GradScaler state.
        # This partial-epoch snapshot stays inside isolated_run; it is a serialization
        # fixture, never a claim of epoch completion or a legal formal initialization.
        require(time.monotonic() - started < seconds, "PENDING: budget exhausted before native CUDA resume check")
        trainer.save_model()
        snapshot_path = trainer.last
        saved = torch_load(snapshot_path, map_location="cpu")
        expected_epoch, expected_updates = saved["epoch"] + 1, saved["updates"]
        del trainer
        trainer = None
        gc.collect(); torch.cuda.empty_cache()
        resumed = CEATrainer(overrides=dict(model=str(snapshot_path), resume=str(snapshot_path)),
                             cea_config=algorithm(), cea_identity=identity)
        resumed._setup_train()
        require(resumed.start_epoch == expected_epoch and resumed.ema.updates == expected_updates, "Native resume epoch/EMA mismatch")
        require(resumed.scaler.state_dict() == saved["scaler"] and bool(saved["scaler"]), "CUDA GradScaler state was not restored")
        state = resumed.optimizer.state_dict()
        require(state["param_groups"] == saved["optimizer"]["param_groups"], "Native optimizer groups changed on resume")
        for parameter_id, values in saved["optimizer"]["state"].items():
            for name, value in values.items():
                restored = state["state"][parameter_id][name]
                require(torch.equal(restored.cpu(), value.to(restored.dtype)) if torch.is_tensor(value) else restored == value,
                        "Native optimizer state changed beyond its stored precision")
        expected_state = saved["ema"].float().state_dict()
        require(set(expected_state) == set(resumed.model.state_dict()), "Resume parameter keys changed")
        require(all(torch.equal(v.cpu(), expected_state[k]) for k, v in resumed.model.state_dict().items()), "Native EMA-to-model restore changed values")
        resumed.epoch = resumed.start_epoch; resumed._cea_epoch_start(resumed)
        require(resumed.model.cea_epoch == expected_epoch, "CEA schedule did not restore actual epoch")
        report["native_cuda_resume"] = dict(status="PASS", saved_epoch=saved["epoch"], restored_epoch=expected_epoch,
                cea_epoch=resumed.model.cea_epoch, scaler=saved["scaler"], ema_updates=expected_updates,
                optimizer_exact_at_stored_precision=True, model_from_saved_ema_exact=True,
                partial_epoch_snapshot="serialization fixture only; native epoch-granularity semantics, not exact microbatch resume")
        del resumed, saved
        gc.collect(); torch.cuda.empty_cache()
        valid &= time.monotonic() - started <= seconds
        report["status"] = "TECHNICAL_PASS" if valid else "PENDING"
        if not valid: report["reason"] = "Budget ended without all required effective-update/activation/finite evidence"
        # Never save a preflight weight as the formal init or formal run checkpoint.
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="RESOURCE_ERROR", reason=repr(error))
    except BaseException as error:
        report.update(status="PENDING" if "PENDING:" in str(error) else "FAIL", reason=repr(error))
        if trainer is not None:
            report.update(microbatches=trainer.cea_batches, steps=trainer.cea_steps)
    report["seconds"] = time.monotonic() - started
    write_json(folder / "report.json", report); write_json(OUT / "preflight.json", report)
    return report


def locate_mother_best(main=MAIN, explicit=None):
    expected = read_json(ROOT / "docs/cea_v1/mother_evidence.json")["best_sha256"]
    path = Path(explicit) if explicit else main / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt"
    require(path.is_file(), f"PENDING: verified mother best missing: {path}")
    require(sha256(path) == expected, "Mother best differs from locally verified full-split val/test archive")
    return path, expected


def diagnose(folder, main=MAIN, mother_best=None, seconds=900, max_val=64, max_train=8):
    require(1 <= max_val <= 64 and 0 <= max_train <= 8 and 0 < seconds <= 900, "Diagnosis bounds exceeded")
    started = time.monotonic()
    prepared, identity = prepared_identity()
    report = dict(status="PENDING", identity=identity, runtime=runtime(), budget_seconds=seconds,
                  max_val_images=max_val, max_train_batches=max_train, test_used=False, optimizer_updates=0,
                  splits={}, gradient_diagnostics=[], folder=str(folder))
    try:
        path, expected = locate_mother_best(main, mother_best)
        report.update(mother_best=str(path), mother_best_sha256=expected)
        require(torch.cuda.is_available(), "PENDING: default B16/640/AMP diagnosis requires CUDA")
        original = torch_load(path, map_location="cpu")
        weights = (original.get("ema") or original["model"]).float()
        require(weights.model[-1].nc == 1, "Mother best class count mismatch")
        report["zero_offset_head"] = all(int(torch.count_nonzero(p)) == 0 for p in weights.model[-1].cbr.offset_out.parameters())
        # Each split gets a fresh model; train BN/gradients cannot contaminate val or formal state.
        for split in ("val", "train"):
            cfg = deepcopy(prepared["recipe"])
            cfg.update(project=str(folder), name="loader_" + split, save_dir=str(folder / ("loader_" + split)),
                       workers=0 if split == "val" else prepared["recipe"]["workers"])
            init_seeds(42, deterministic=True)
            trainer = CEATrainer(overrides=cfg, cea_config=algorithm(), cea_identity=identity)
            model = CEADetectionModel(str(MODEL), nc=1, verbose=False, cea_config=algorithm())
            strict_load(model, weights)
            model.nc = 1; model.cea_epoch = 20; model.to(trainer.device).train(split == "train")
            trainer.model = model
            loader = trainer.get_dataloader(trainer.data[split], batch_size=16, rank=-1, mode=split)
            arrays = {k: [] for k in ("advantage", "improvement", "kl", "weighted_kl", "choice")}
            rows, images = [], 0
            batches = (max_val + 15) // 16 if split == "val" else max_train
            for index, raw in enumerate(itertools.islice(loader, batches)):
                if time.monotonic() - started >= seconds:
                    break
                if split == "val" and images + len(raw["img"]) > max_val:
                    keep = max_val - images
                    mask = raw["batch_idx"] < keep
                    raw = {k: (v[mask] if k in ("cls", "bboxes", "batch_idx") else v[:keep]
                               if k == "img" or isinstance(v, (list, tuple)) else v) for k, v in raw.items()}
                batch = trainer.preprocess_batch(raw)
                grad_enabled = split == "train" and index < 2
                # Compare memory/time with the same main forward+L0 and RNG/BN state.
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                buffers = {n: v.clone() for n, v in model.named_buffers()}
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.monotonic()
                with torch.set_grad_enabled(grad_enabled), torch.autocast("cuda", enabled=split == "train"):
                    model.cea_epoch = 0
                    base_loss, _ = model.loss(batch)
                torch.cuda.synchronize(); baseline_time = time.monotonic() - t0
                baseline_peak = torch.cuda.max_memory_allocated()
                del base_loss
                with torch.no_grad():
                    for n, v in model.named_buffers(): v.copy_(buffers[n])
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
                model.cea_epoch = 20
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.monotonic()
                with torch.set_grad_enabled(grad_enabled), torch.autocast("cuda", enabled=split == "train"):
                    losses, detail, matches = model.cea_components(batch)
                torch.cuda.synchronize(); active_time = time.monotonic() - t0
                active_peak = torch.cuda.max_memory_allocated()
                row = dict(batch=index, images=len(batch["img"]), **summarize(detail),
                           matched_images=detail["matched_images"], improved_images=detail["improved_images"],
                           raw_loss=detail["raw_loss"], weighted_loss=detail["weighted_loss"],
                           baseline_seconds=baseline_time, active_seconds=active_time,
                           extra_seconds=active_time-baseline_time, baseline_peak_bytes=baseline_peak,
                           active_peak_bytes=active_peak, extra_peak_bytes=active_peak-baseline_peak,
                           images_used=list(batch["im_file"]))
                if grad_enabled:
                    weight = model.model[-1].cbr.score.weight
                    l0 = sum(v for k, v in losses.items() if k != "loss_cea")
                    g0, = torch.autograd.grad(l0, weight, retain_graph=True)
                    gc_, = torch.autograd.grad(losses["loss_cea"], weight)
                    n0, nc = float(g0.norm()), float(gc_.norm())
                    gradient = dict(split=split, batch=index, l0_score_norm=n0, weighted_cea_score_norm=nc,
                                    ratio=nc/n0 if n0 else None,
                                    cosine=float((g0.double() * gc_.double()).sum() / (g0.double().norm() * gc_.double().norm())) if n0 and nc else None,
                                    ratio_status="defined" if n0 else "undefined: zero L0 gradient",
                                    cosine_status="defined" if n0 and nc else "undefined: zero gradient norm")
                    report["gradient_diagnostics"].append(gradient)
                for key in arrays:
                    if key in detail: arrays[key].append(detail[key].detach().cpu().numpy())
                rows.append(row); images += len(batch["img"])
                write_json(folder / f"{split}_batches.json", rows)
                del losses, detail, matches, buffers
                if grad_enabled: del g0, gc_, l0
            merged = {k: np.concatenate(v, axis=0) if v else np.empty((0, 4)) for k, v in arrays.items()}
            np.savez_compressed(folder / f"{split}_raw_stats.npz", **merged)
            totals = {k: sum(r.get(k, 0) for r in rows) for k in ("matches", "edges", "improved_edges", "improved_gt", "matched_images", "improved_images")}
            totals.update(images=images, batches=len(rows), improved_edge_fraction=totals["improved_edges"] / totals["edges"] if totals["edges"] else None,
                          improved_gt_fraction=totals["improved_gt"] / totals["matches"] if totals["matches"] else None,
                          improved_image_fraction=totals["improved_images"] / images if images else None,
                          quantile_levels=[0, .25, .5, .75, .9, .99, 1],
                          quantiles={k: np.quantile(v, [0, .25, .5, .75, .9, .99, 1]).tolist() if v.size else []
                                     for k, v in merged.items() if k != "choice"},
                          choice_counts=np.bincount(merged["choice"].astype(int).flatten(), minlength=4).tolist(),
                          anchor_main_max_abs=max((r["anchor_main_max_abs"] for r in rows), default=None))
            report["splits"][split] = totals
            del trainer, model, loader
            gc.collect(); torch.cuda.empty_cache()
        done = report["splits"]["val"]["images"] == max_val and report["splits"]["train"]["batches"] == max_train
        report["status"] = "COMPLETED" if done else "PENDING"
        improved = sum(v["improved_edges"] for v in report["splits"].values())
        report["mechanism"] = ("NO_CANDIDATE_IMPROVEMENT: no support for long training" if done and improved == 0 else
                               "OBSERVED_SIGNAL: inspect raw magnitude/coverage and gradients; no AP evidence" if improved else
                               "PENDING: incomplete diagnosis; no mechanism conclusion")
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="RESOURCE_ERROR", reason=repr(error))
    except BaseException as error:
        report.update(status="PENDING" if "PENDING:" in str(error) else "FAIL", reason=repr(error))
    report["seconds"] = time.monotonic() - started
    write_json(folder / "report.json", report); write_json(OUT / "diagnose.json", report)
    return report
