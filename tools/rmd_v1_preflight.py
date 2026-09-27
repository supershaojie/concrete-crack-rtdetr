"""A single 900-second / 16 real B16 micro-batch diagnostic budget, isolated from the formal run."""
from __future__ import annotations

from copy import deepcopy
import gc
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch
from rmd_v1_common import (ROOT, OUT, require, config, write_json, read_json, append_json, now, runtime, sha256,
    unique_name, verify_prepared, offline_amp_resources, parent_identity, optimizer_groups, native_amp_guard,
    aggregate, verify_pair, digest)
from ultralytics.models.rtdetr.rmd_v1 import RMDTrainer, promote, sync_epoch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.patches import torch_load


class Budget:
    def __init__(self, folder, seconds=900):
        self.start = time.monotonic(); self.seconds = seconds; self.used = 0; self.folder = Path(folder)
    def stage(self, name):
        elapsed = time.monotonic()-self.start
        report = dict(stage=name, elapsed_seconds=elapsed, remaining_seconds=max(0, self.seconds-elapsed),
                      micro_batches=self.used, remaining_micro_batches=16-self.used)
        write_json(self.folder/"progress.json", report)
        print("RMD_PREFLIGHT", report, flush=True)
        if elapsed >= self.seconds:
            raise TimeoutError("Shared preflight time budget exhausted")
    def consume(self, name):
        require(self.used < 16, "Shared real-training micro-batch limit exceeded")
        self.used += 1; self.stage(name)


class DiagnosticTrainer(RMDTrainer):
    def _setup_train(self):
        with native_amp_guard() as messages:
            super()._setup_train()
        self.rmd_amp_evidence = messages
        require(self.amp and self.batch_size == 16 and self.args.imgsz == 640, "Native AMP/B16/640 recipe changed")


def audit_optimizer(trainer):
    original = deepcopy(trainer.model).cpu()
    original.__class__ = __import__("ultralytics.nn.tasks", fromlist=["RTDETRDetectionModel"]).RTDETRDetectionModel
    native = trainer.build_optimizer(original, name="AdamW", lr=.0005, momentum=.937, decay=.0001,
                                     iterations=math.ceil(len(trainer.train_loader.dataset)/64)*200)
    expected = optimizer_groups(original, native)
    actual = optimizer_groups(trainer.model, trainer.optimizer)
    require(actual == expected, "Optimizer groups differ from mother")
    ids = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    require(all(ids.count(id(p)) == 1 for p in trainer.model.parameters()), "Optimizer parameter coverage differs")
    return dict(status="PASS", rebuild=trainer.rmd_rebuild_audit, native_groups_exact=True, groups=actual,
                trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))


def capacity(trainer, budget, folder):
    trainer.epoch = 20; sync_epoch(trainer); trainer.model.train()
    trainer.accumulate = max(round(trainer.args.nbs/trainer.batch_size), 1)
    require(trainer.accumulate == 4, "Original e20 accumulation changed")
    for group in trainer.optimizer.param_groups:
        group["lr"] = group["initial_lr"] * trainer.lf(20)
    trainer.optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    updates, observations = [], []
    def stepped(optimizer, args, kwargs):
        gradients = {n: float(v.grad.float().norm()) for n, v in trainer.model.named_parameters()
                     if v.grad is not None and (n.startswith("model.20.") or ".cbr." in n or "dec_bbox_head.2" in n)}
        finite = all(bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters() if p.grad is not None)
        require(finite, "Actual optimizer update has nonfinite gradients")
        updates.append(dict(key_gradients_after_native_clip=gradients, finite=True))
    hook = trainer.optimizer.register_step_post_hook(stepped)
    fallback = False
    effective = 0
    last_loss = None
    try:
        iterator = iter(trainer.train_loader)
        for i in range(4):
            budget.consume(f"capacity micro-batch {i+1}/4, e=20, accumulate=4")
            batch = trainer.preprocess_batch(next(iterator))
            require(tuple(batch["img"].shape) == (16, 3, 640, 640), "Capacity is not real B16/640")
            begin = time.monotonic()
            with autocast(trainer.amp):
                loss, trainer.loss_items = trainer.model(batch)
                trainer.loss = loss.sum()
            require(bool(torch.isfinite(trainer.loss)), "Nonfinite diagnostic loss; cannot lower scale to hide it")
            trainer.scaler.scale(trainer.loss).backward()
            last_loss = float(trainer.loss.detach())
            row = dict(trainer.model.criterion.statistics, micro_batch=i, loss=last_loss,
                       scaler_scale=trainer.scaler.get_scale(), reduced_diagnostic_init_scale=fallback)
            if (i+1) % trainer.accumulate == 0:
                selected = {n: p.detach().clone() for n, p in trainer.model.named_parameters()
                            if n.endswith("cbr.offset_out.weight") or n.endswith("O_proj.weight") or n.endswith("dec_bbox_head.2.layers.2.weight")}
                before = len(updates); scale = trainer.scaler.get_scale()
                trainer.optimizer_step()  # EXACT original unscale -> clip(10) -> scaler.step -> EMA
                changed = {n: float((dict(trainer.model.named_parameters())[n].detach()-p).abs().max()) for n, p in selected.items()}
                successful = len(updates) > before and any(v > 0 for v in changed.values())
                effective += int(successful)
                row.update(scale_before=scale, scale_after=trainer.scaler.get_scale(), optimizer_step_invoked=len(updates)>before,
                           overflow_skipped=len(updates)==before, effective_update=successful, parameter_max_changes=changed)
                if successful:
                    row["update"] = updates[-1]
                elif i == 3:
                    # All these objects belong solely to this diagnostic process.
                    fallback = True
                    trainer.scaler = torch.cuda.amp.GradScaler(init_scale=128, enabled=True)
                    trainer.optimizer.zero_grad()
            row["elapsed_seconds"] = time.monotonic()-begin
            observations.append(row); append_json(folder/"capacity_batches.jsonl", row)
            del loss, batch
            if effective:
                break
        trainer.fitness = trainer.best_fitness = 0.0
        trainer.model.rmd_identity = trainer.ema.ema.rmd_identity = {"scope": "diagnostic", "formula": config()["formula_version"]}
        trainer.save_model()  # native EMA/optimizer/scaler checkpoint, under diagnostic output only
        report = dict(status="PASS" if effective else "PENDING", effective_updates=effective, optimizer_calls=len(updates),
                      used_micro_batches=len(observations), batch=16, imgsz=640, amp=True, accumulate=4, epoch=20,
                      fallback_init_scale_128=fallback, native_initial_scale_adaptation="not fully verified" if fallback else "effective update observed",
                      loss=last_loss, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), checkpoint=str(trainer.last),
                      checkpoint_sha256=sha256(trainer.last), observations=observations)
        write_json(folder/"capacity.json", report)
        return report
    finally:
        hook.remove()


def probe_model(model, trainer, budget, folder, count):
    model = promote(model).to(trainer.device).train()
    model.nc = 1; model.rmd_epoch = 20
    rows = []; iterator = iter(trainer.train_loader)
    with torch.no_grad():
        for i in range(count):
            budget.consume(f"trained-mother probe {i+1}/{count}; original augmented B16/640")
            batch = trainer.preprocess_batch(next(iterator))
            require(tuple(batch["img"].shape) == (16, 3, 640, 640), "Probe batch recipe changed")
            begin = time.monotonic()
            with autocast(trainer.amp):
                loss, _ = model(batch)
            require(bool(torch.isfinite(loss)), "Nonfinite trained-mother diagnostic loss")
            row = dict(model.criterion.statistics, micro_batch=i, elapsed_seconds=time.monotonic()-begin)
            rows.append(row); append_json(folder/"probe_batches.jsonl", row)
            del loss, batch
    return aggregate(rows)


def applicability(summary, parent):
    result = dict(summary, trained_mother=parent, gates=config())
    if summary["valid_batches"] < 8:
        return dict(result, status="APPLICABILITY_PENDING", reason="Fewer than eight valid original-recipe batches")
    failures = []
    if summary["gt_coverage_k_ge_2"] < .5:
        failures.append("GT-occurrence coverage with K>=2 is below 50%")
    if summary["mean_abs_weight_minus_one"] < .01:
        failures.append("Trained-mother mean(abs(w-1)) is below 0.01")
    return dict(result, status="NOT_APPLICABLE" if failures else "APPLICABILITY_PASS",
                reason="; ".join(failures) if failures else "Both fixed engineering gates passed; not an accuracy prediction")


def child(p, folder, probe_only=False, local=False, parent_path=None):
    folder = Path(folder); budget = Budget(folder)
    report = dict(status="PENDING", started=now(), runtime=runtime(), correctness={"status": "PENDING"},
                  capacity={"status": "PENDING"}, lifecycle={"status": "PENDING"}, resume={"status": "PENDING"},
                  applicability={"status": "APPLICABILITY_PENDING"}, incremental_training_overhead="PENDING: no paired timing measurement")
    try:
        budget.stage("verify prepared code/data/source binding")
        prepared = verify_prepared(p, diagnostic_splits=("train",) if probe_only else ("train", "val"))
        report["binding"] = prepared["binding"]
        if not probe_only:
            budget.stage("CPU correctness and original-forward equivalence")
            from check_rmd_v1 import run
            ok = run(folder/"correctness.json")
            report["correctness"] = read_json(folder/"correctness.json")
            require(ok, "CPU correctness failed")
        if local or not torch.cuda.is_available():
            report["capacity"]["reason"] = "Server B16/640 AMP capacity required; local checks never substitute a smaller batch"
            report["lifecycle"]["reason"] = "Run the separate real-val lifecycle or server preflight"
            report["applicability"]["reason"] = "Trained original mother checkpoint and >=8 real B16 batches required"
            return report
        budget.stage("offline original AMP resources and native training setup")
        report["amp_resources"] = offline_amp_resources(p)
        args = dict(prepared["args"], project=str(folder), name="native_diagnostic", save_dir=str(folder/"native_diagnostic"), plots=False)
        trainer = DiagnosticTrainer(overrides=args)
        trainer._setup_train()
        report["optimizer_audit"] = audit_optimizer(trainer)
        report["actual_diagnostic_args"] = vars(trainer.args).copy()
        report["amp_check"] = trainer.rmd_amp_evidence
        if not probe_only:
            report["capacity"] = capacity(trainer, budget, folder)
            # Four additional training forwards are reserved for the fresh-process
            # native resume/update proof, leaving exactly eight for applicability.
            for _ in range(4):
                budget.consume("reserve fresh-process resumed B16/640 micro-batch (4 total)")
            budget.stage("new Python: native checkpoint reload, full resume and four actual training micro-batches")
            command = [sys.executable, "-u", str(ROOT/"tools/rmd_v1.py"), "_resume-check", "--checkpoint",
                       report["capacity"]["checkpoint"], "--output", str(folder/"resume.json")]
            subprocess.run(command, cwd=ROOT, check=True, timeout=max(1, 900-(time.monotonic()-budget.start)))
            report["resume"] = read_json(folder/"resume.json")
            combined_updates = report["capacity"]["effective_updates"] + report["resume"]["effective_updates"]
            report["capacity"].update(status="PASS" if combined_updates else "PENDING",
                effective_updates=combined_updates, initial_process_updates=report["capacity"]["effective_updates"],
                fresh_process_updates=report["resume"]["effective_updates"], used_micro_batches=8,
                checkpoint_sha256=sha256(report["capacity"]["checkpoint"]), resumed_update=report["resume"],
                peak_allocated_bytes=max(report["capacity"]["peak_allocated_bytes"], report["resume"]["peak_allocated_bytes"]))
            write_json(folder/"capacity.json", report["capacity"])
            budget.stage("new Python: FP32 AutoBackend/fuse/zeros warmup and one actual val batch")
            command = [sys.executable, "-u", str(ROOT/"tools/rmd_v1.py"), "_lifecycle", "--checkpoint",
                       report["capacity"]["checkpoint"], "--data", str(p["data"]), "--output", str(folder/"lifecycle")]
            subprocess.run(command, cwd=ROOT, check=True, timeout=max(1, 900-(time.monotonic()-budget.start)))
            report["lifecycle"] = read_json(folder/"lifecycle/metrics.json")
        budget.stage("locate and identify trained original mother")
        parent_file = Path(parent_path or p["parent"])
        if not parent_file.is_file():
            report["applicability"] = dict(status="APPLICABILITY_PENDING", missing_path=str(parent_file),
                reason="Trained mother absent; random initialization is not evidence of inapplicability")
        else:
            parent, parent_record = parent_identity(parent_file)
            report["applicability"] = applicability(probe_model(parent, trainer, budget, folder, 16-budget.used), parent_record)
            require(sha256(parent_file) == parent_record["sha256"], "Read-only probe changed mother checkpoint")
            write_json(folder/"applicability.json", report["applicability"])
        if probe_only:
            report["status"] = report["applicability"]["status"]
        elif all(report[k]["status"] == "PASS" for k in ("correctness", "capacity", "resume", "lifecycle")) and report["applicability"]["status"] == "APPLICABILITY_PASS":
            report["status"] = "PASS"
        elif report["applicability"]["status"] == "NOT_APPLICABLE":
            report["status"] = "NOT_APPLICABLE"
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        report.update(status="PENDING", error=repr(error), traceback=traceback.format_exc())
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        actual = 0
        for name in ("capacity_batches.jsonl", "resume_batches.jsonl", "probe_batches.jsonl"):
            path = folder/name
            if path.is_file():
                actual += len(path.read_text(encoding="utf-8").splitlines())
        report.update(finished=now(), elapsed_seconds=time.monotonic()-budget.start,
                      micro_batches=actual, reserved_micro_batch_slots=budget.used)
        if report["elapsed_seconds"] > 900:
            report.update(status="PENDING", reason="Shared time budget exhausted")
        write_json(folder/"report.json", report)
    return report


def run_bounded(p, probe_only=False, local=False, parent_path=None):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT/"check.lock"
    lock.mkdir(exist_ok=False)
    try:
        return _run_bounded(p, probe_only, local, parent_path)
    finally:
        lock.rmdir()


def _run_bounded(p, probe_only=False, local=False, parent_path=None):
    folder = OUT/"checks"/unique_name(); folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(ROOT/"tools/rmd_v1.py"), "_check-worker", "--main", str(p["main"]),
               "--data", str(p["data"]), "--c2-args", str(p["c2_args"]), "--output", str(folder)]
    if local: command.append("--local")
    if probe_only: command.append("--probe-only")
    if parent_path: command.extend(["--probe-weights", str(parent_path)])
    started = time.monotonic(); deadline = started+900
    with (folder/"console.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        write_json(folder/"dispatch.json", dict(command=command, pid=process.pid, started=now(), max_seconds=900, max_micro_batches=16))
        next_update = 0
        while process.poll() is None and time.monotonic() < deadline:
            if time.monotonic() >= next_update:
                print("RMD bounded check", read_json(folder/"progress.json", {"stage": "starting"}), "log:", folder/"console.log", flush=True)
                next_update = time.monotonic()+15
            time.sleep(.5)
        timed_out = process.poll() is None
        if timed_out:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)  # only our freshly created diagnostic process group
            else:
                process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt": os.killpg(process.pid, signal.SIGKILL)
                else: process.kill()
                process.wait(timeout=5)
    report = read_json(folder/"report.json", {"status": "PENDING", "reason": "Diagnostic process ended before its final report"})
    if timed_out:
        report.update(status="PENDING", timeout=True, reason="Hard shared 900-second deadline reached")
    write_json(folder/"report.json", report)
    pointer = dict(report=str(folder/"report.json"), report_sha256=sha256(folder/"report.json"),
                   status=report["status"], process_exit_code=process.returncode, timed_out=timed_out)
    write_json(OUT/("probe.json" if probe_only else "preflight.json"), pointer)
    if report.get("applicability", {}).get("status") == "NOT_APPLICABLE":
        write_json(OUT/"not_applicable.json", dict(binding=report["binding"],
                   parent_sha256=report["applicability"]["trained_mother"]["sha256"], evidence=pointer))
    print(pointer, flush=True)
    return report


def resume_check(checkpoint, output):
    """Native state restoration followed by exactly four real micro-batches and one native update attempt."""
    import warnings
    input_hash = sha256(checkpoint)
    ckpt = torch_load(checkpoint, map_location="cpu")
    require(ckpt.get("optimizer") is not None and ckpt.get("scaler") is not None, "Diagnostic resume state missing")
    trainer = DiagnosticTrainer(overrides=dict(resume=str(checkpoint), model=str(checkpoint)))
    trainer._setup_train()
    require(trainer.start_epoch == ckpt["epoch"]+1, "Resume epoch lost")
    require(trainer.scaler.state_dict() == ckpt["scaler"], "Native scaler not restored")
    state = trainer.optimizer.state_dict()
    expected = ckpt["optimizer"]
    require(state["param_groups"] == expected["param_groups"], "Resume optimizer groups changed")
    for key, row in expected["state"].items():
        for field, value in row.items():
            actual = state["state"][key][field]
            if isinstance(value, torch.Tensor):
                require(torch.equal(actual.cpu(), value.to(dtype=actual.dtype).cpu()), "Resume optimizer state differs")
            else:
                require(actual == value, "Resume optimizer scalar differs")
    for key, value in ckpt["ema"].float().state_dict().items():
        require(torch.equal(trainer.ema.ema.state_dict()[key].cpu(), value), "Resume EMA differs")
    require(trainer.ema.updates == ckpt["updates"], "Resume EMA update count differs")
    trainer.epoch = trainer.start_epoch; sync_epoch(trainer)
    require(trainer.model.rmd_epoch == trainer.start_epoch, "RMD resume schedule stuck at saved value")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.scheduler.step()
    require(trainer.accumulate == 4, "Native resumed accumulation changed")
    trainer.model.train(); trainer.optimizer.zero_grad()
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()
              if n.endswith("O_proj.weight") or n.endswith("cbr.offset_out.weight")}
    updates = []
    def observed(optimizer, args, kwargs):
        finite = all(bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters() if p.grad is not None)
        require(finite, "Nonfinite gradients at resumed optimizer update")
        updates.append({n: float(p.grad.float().norm()) for n, p in trainer.model.named_parameters()
                        if p.grad is not None and (".cbr." in n or "O_proj" in n or "dec_bbox_head.2" in n)})
    hook = trainer.optimizer.register_step_post_hook(observed)
    rows = []; scale_before = trainer.scaler.get_scale()
    torch.cuda.reset_peak_memory_stats()
    try:
        iterator = iter(trainer.train_loader)
        for index in range(4):
            print(f"RMD native resume micro-batch {index+1}/4, e={trainer.epoch}, AMP, B16/640", flush=True)
            batch = trainer.preprocess_batch(next(iterator))
            require(tuple(batch["img"].shape) == (16, 3, 640, 640), "Resumed capacity batch changed")
            with autocast(trainer.amp):
                loss, trainer.loss_items = trainer.model(batch)
                trainer.loss = loss.sum()
            require(bool(torch.isfinite(trainer.loss)), "Nonfinite resumed RMD loss")
            trainer.scaler.scale(trainer.loss).backward()
            rows.append(dict(trainer.model.criterion.statistics, loss=float(trainer.loss.detach())))
            append_json(Path(output).with_name("resume_batches.jsonl"), rows[-1])
            del loss, batch
        trainer.optimizer_step()
    finally:
        hook.remove()
    changed = {n: float((dict(trainer.model.named_parameters())[n].detach()-p).abs().max()) for n, p in before.items()}
    effective = int(bool(updates) and any(v > 0 for v in changed.values()))
    trainer.fitness = trainer.best_fitness = 0.0
    trainer.model.rmd_identity = trainer.ema.ema.rmd_identity = deepcopy(getattr(ckpt["ema"], "rmd_identity", {"scope": "diagnostic"}))
    trainer.save_model()
    write_json(output, dict(status="PASS" if effective else "PENDING", epoch=trainer.start_epoch,
        optimizer_exact=True, scaler_exact=True, ema_exact=True, ema_updates=trainer.ema.updates,
        rmd_epoch=trainer.model.rmd_epoch, effective_updates=effective, parameter_max_changes=changed,
        scale_before=scale_before, scale_after=trainer.scaler.get_scale(), overflow_skipped=not updates,
        gradients_after_native_clip=updates, micro_batches=4, observations=rows,
        input_checkpoint_sha256=input_hash, output_checkpoint_sha256=sha256(trainer.last),
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
