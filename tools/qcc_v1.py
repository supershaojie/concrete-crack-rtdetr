"""QCC independent lifecycle; safeguards ported from ef9cb7e, evaluation rewritten."""
from __future__ import annotations

import argparse

from contextlib import contextmanager

from copy import deepcopy

import gzip

import io

import json

import os

from pathlib import Path

import shlex

import shutil

import signal

import subprocess

import sys

import tarfile

import time

import traceback

from qcc_v1_common import (ROOT, MAIN, OUT, RUN, INIT, SOURCE, SESSION, BRANCH, BASE, SERVER_PYTHON,
    FORMULA, now, git, read_json, digest, runtime, code_identity, data_config, inventory, recipe,
    binding, prepare, sha256, require, write_json, cached_data, COUNTS)

def process_identity(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return dict(pid=pid, created=p.create_time(), command=p.cmdline())
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as error:
        raise RuntimeError(f"Cannot verify PID {pid}; preserve the existing lock/process") from error

def active_workers():
    import psutil
    found = []
    for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
        cmd = p.info["cmdline"] or []
        if "_worker" in cmd and any(str(ROOT / "tools/qcc_v1.py").replace("\\", "/") == s.replace("\\", "/") for s in cmd):
            found.append(p.info)
    return found

def has_tmux():
    return bool(shutil.which("tmux") and subprocess.run(["tmux", "has-session", "-t", "=" + SESSION],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0)

@contextmanager
def operation_lock():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "operation.lock"
    if path.exists():
        old = read_json(path)
        live = process_identity(old["pid"])
        require(not live or live["created"] != old["created"], f"QCC operation is active: {old}")
        path.unlink()  # only this experiment's verified stale lock
    with path.open("x", encoding="utf-8") as f:
        json.dump(process_identity(os.getpid()), f)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)

def strict_amp_resources():
    """No download or implicit skipped AMP check when offline assets are absent."""
    from ultralytics.utils import ASSETS
    resources = []
    for name, target, candidates in (
        ("bus.jpg", ASSETS / "bus.jpg", [MAIN / "ultralytics-main/ultralytics/assets/bus.jpg", MAIN / "bus.jpg"]),
        ("yolo26n.pt", ROOT / "yolo26n.pt", [MAIN / "yolo26n.pt", MAIN / "weights/yolo26n.pt"]),
    ):
        if not target.exists():
            source = next((p for p in candidates if p.is_file()), None)
            require(source, f"Offline AMP resource missing: {name}; supply the mother's existing resource before preflight")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as a, target.open("xb") as b:
                shutil.copyfileobj(a, b)
        resources.append(dict(name=name, path=str(target), sha256=sha256(target)))
    return resources

def status():
    dispatches = sorted((OUT / "dispatches").glob("*/dispatch.json"))
    latest = dispatches[-1].parent if dispatches else None
    completed, final = read_json(OUT / "training_completed.json"), read_json(OUT / "final_eval.json")
    return dict(active_workers=active_workers(), tmux=dict(name=SESSION, exists=has_tmux()),
        dispatch=read_json(latest / "dispatch.json") if latest else None,
        worker=read_json(latest / "worker.json") if latest else None,
        exit=read_json(latest / "exit.json") if latest else None,
        shell_python_exit=(latest / "python_exit_code.txt").read_text().strip() if latest and (latest / "python_exit_code.txt").exists() else None,
        progress=read_json(OUT / "progress.json"), training=completed or "NOT_COMPLETED", final_eval=final or "NOT_RUN",
        best=file_info(RUN / "weights/best.pt", hash_file=False), last=file_info(RUN / "weights/last.pt", hash_file=False),
        logs=[str(p) for p in (OUT / "dispatches").glob("*/console.log")], preflight=read_json(OUT / "preflight.json", {}).get("status", "NOT_RUN"))

def file_info(path, hash_file=True):
    path = Path(path)
    return dict(path=str(path), exists=path.exists(), bytes=path.stat().st_size if path.exists() else None,
                sha256=sha256(path) if hash_file and path.is_file() else None)

def run_bounded(command, timeout, label):
    """One process group owned by this invocation; no retry and no global kill."""
    require(timeout > 0, f"No time remains for {label}")
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=ROOT, start_new_session=os.name != "nt")
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(f"{label} exceeded remaining boundary {timeout:.1f}s")
            try:
                code = proc.wait(timeout=min(20, remaining))
                return code
            except subprocess.TimeoutExpired:
                print(f"QCC stage={label} elapsed={time.monotonic()-started:.1f}s remaining={remaining:.1f}s", flush=True)
    except BaseException:
        if proc.poll() is None:
            import psutil
            # Only descendants of this exact spawned PID; nested lifecycle checks
            # may have their own process groups. Never inspect/kill global Python.
            descendants = psutil.Process(proc.pid).children(recursive=True)
            for child in reversed(descendants):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=5)
            _, alive = psutil.wait_procs(descendants, timeout=1)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        raise

def preflight(seconds=900, micro_batches=16):
    import torch
    require(1 <= seconds <= 900 and 4 <= micro_batches <= 16, "Preflight boundary: <=900s and <=16 micro-batches")
    folder = OUT / "preflights" / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="PENDING", boundary=dict(seconds=seconds, micro_batches=micro_batches), started=now(),
                  checks={k: dict(status="PENDING") for k in ("cpu", "cuda_b16_amp", "mechanism", "native_scale", "new_process_val", "resume")})
    write_json(OUT / "preflight.json", report)
    def remaining():
        return seconds - (time.monotonic() - started)
    try:
        print("QCC stage=identity elapsed=0s", flush=True)
        report["binding"] = binding()
        require(remaining() > 0, "Preflight time boundary exhausted during identity checks")
        code = run_bounded([sys.executable, str(ROOT / "tools/check_qcc_v1.py"), "--math-only", "--output", str(folder / "cpu")], remaining(), "CPU math/loss routing")
        require(code == 0, "CPU tests failed")
        report["checks"]["cpu"] = dict(status="PASS", report=str(folder / "cpu/checks.json"))
        write_json(OUT / "preflight.json", report)
        if os.name != "posix" or not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
            report["reason"] = "Formal CUDA B16/640 check requires Linux server and sufficient memory; local small tests do not qualify"
            return report
        report["amp_resources"] = strict_amp_resources()
        require(Path(sys.executable).resolve() == Path(SERVER_PYTHON).resolve(), "Use the fixed server Python")
        write_json(folder / "request.json", dict(binding=report["binding"], seconds=remaining(), micro_batches=micro_batches))
        code = run_bounded([sys.executable, str(ROOT / "tools/qcc_v1_preflight.py"), "--folder", str(folder)], remaining(), "CUDA B16/640 lifecycle")
        gpu = read_json(folder / "gpu.json", {})
        report["gpu"] = gpu
        report["checks"].update(gpu.get("checks", {}))
        require(code == 0, f"GPU preflight exited {code}; see {folder}")
        statuses = [v["status"] for v in report["checks"].values()]
        report["status"] = "PASS" if all(s == "PASS" for s in statuses) else "PENDING"
        # Native high-scale adaptation may be unresolved; never disguise it with fallback.
        required = ("cpu", "cuda_b16_amp", "mechanism", "new_process_val", "resume")
        report["start_eligible"] = all(report["checks"][k]["status"] == "PASS" for k in required)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), start_eligible=False)
        gpu = read_json(folder / "gpu.json", {})
        report["gpu"] = gpu
        report["checks"].update(gpu.get("checks", {}))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["folder"] = str(folder)
        write_json(folder / "preflight.json", report)
        write_json(OUT / "preflight.json", report)
    return report

def guarded_preflight(seconds, micro_batches):
    require(1 <= seconds <= 900 and 4 <= micro_batches <= 16, "Preflight boundary: <=900s and <=16 micro-batches")
    try:
        code = run_bounded([sys.executable, str(ROOT / "tools/qcc_v1.py"), "_preflight", "--seconds", str(seconds),
                            "--micro-batches", str(micro_batches)], seconds, "entire preflight including identity")
        result = read_json(OUT / "preflight.json", {})
        require(code in (0, 2) and result, "Bounded preflight process failed")
        return result
    except BaseException as error:
        result = read_json(OUT / "preflight.json", {})
        result.update(status="FAIL", start_eligible=False, boundary_error=repr(error))
        write_json(OUT / "preflight.json", result)
        raise

def checkpoint_identity(path, training, resume=False):
    from ultralytics.utils.patches import torch_load
    ckpt = torch_load(path, map_location="cpu")
    model = ckpt.get("ema") or ckpt.get("model")
    require(getattr(model, "qcc_identity", None) == training, "Checkpoint is not bound to this QCC training identity")
    if resume:
        require(0 <= ckpt.get("epoch", -1) < 199 and ckpt.get("optimizer") is not None and ckpt.get("scaler"), "last lacks valid resumable epoch/optimizer/scaler state")
    return ckpt

def verify_preflight(pre, current):
    require(pre.get("status") in ("PASS", "PENDING") and pre.get("start_eligible"), "Preflight is not eligible for start")
    require(pre.get("binding") == current, "Preflight code/data/init/recipe binding changed")
    checks = pre.get("checks", {})
    for key in ("cpu", "cuda_b16_amp", "mechanism", "new_process_val", "resume"):
        require(checks.get(key, {}).get("status") == "PASS", f"Required check is not PASS: {key}")
    gpu = pre.get("gpu", {})
    capacity = checks["cuda_b16_amp"]
    require(0 < gpu.get("micro_batches", 0) <= 16 and capacity.get("effective_updates", 0) > 0, "No bounded actual optimizer update evidence")
    require(capacity.get("batch") == 16 and capacity.get("imgsz") == 640 and capacity.get("amp") is True, "Capacity evidence is not B16/640 AMP")
    if checks.get("native_scale", {}).get("status") != "PASS":
        require(checks.get("fallback_scale", {}).get("status") == "PASS" and gpu.get("effective_update_arm") == "diagnostic_init_scale_128",
                "Native scale pending without isolated scale128 evidence")
        print("QCC: native initial-scale adaptation is PENDING; isolated scale128 produced a real update. Formal scaler remains native.", flush=True)

def dispatch(resume=False):
    require(os.name == "posix" and Path(sys.executable).resolve() == Path(SERVER_PYTHON).resolve(), "Formal dispatch requires fixed server Python/Linux")
    require(shutil.which("tmux"), "tmux is missing; no dispatch/run identity was created")
    require(not active_workers() and not has_tmux(), "QCC worker/tmux already exists; status only, no replacement")
    current = binding()
    pre = read_json(OUT / "preflight.json", {})
    verify_preflight(pre, current)
    strict_amp_resources()
    completed = read_json(OUT / "training_completed.json")
    require(not completed, "Training already completed; use val for final-eval recovery, never resume")
    train_identity = read_json(OUT / "training_identity.json")
    if resume:
        require(train_identity and train_identity["binding"] == current, "Resume must retain original training SHA/data/recipe")
        checkpoint_identity(RUN / "weights/last.pt", train_identity, resume=True)
    else:
        require(not RUN.exists() and not train_identity, "Existing run/dispatch preserved; start is one-shot")
        train_identity = dict(binding=current, run=str(RUN), session=SESSION, created=now())
        write_json(OUT / "training_identity.json", train_identity)
    identifier = time.strftime("%Y%m%d_%H%M%S") + ("_resume" if resume else "_start")
    folder = OUT / "dispatches" / identifier
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(ROOT / "tools/qcc_v1.py"), "_worker", "--dispatch", identifier]
    record = dict(created=now(), resume=resume, command=command, cwd=str(ROOT), identity=train_identity, status="DISPATCHED")
    write_json(folder / "dispatch.json", record)
    environment = dict(PYTHONPATH=str(ROOT / "ultralytics-main") + os.pathsep + str(ROOT / "tools"),
                       QCC_V1_MAIN=str(MAIN), PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="false")
    for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"):
        environment[name] = os.environ.get(name)
    exports = "".join(("unset " + k if v is None else "export " + k + "=" + shlex.quote(v)) + "\n" for k, v in environment.items())
    shell = ("#!/usr/bin/env bash\nset -uo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n" + exports +
             shlex.join(command) + " 2>&1 | tee -a " + shlex.quote(str(folder / "console.log")) + "\n" +
             "codes=(\"${PIPESTATUS[@]}\")\nprintf '%s\\n' \"${codes[0]}\" > " + shlex.quote(str(folder / "python_exit_code.txt")) + "\n" +
             "printf '%s\\n' \"${codes[1]}\" > " + shlex.quote(str(folder / "tee_exit_code.txt")) + "\nexit \"${codes[0]}\"\n")
    with (folder / "worker.sh").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(shell)
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "bash " + shlex.quote(str(folder / "worker.sh"))], check=True, timeout=15)
    return dict(status="DISPATCHED", session=SESSION, log=str(folder / "console.log"), worker_command=command)

def worker(identifier):
    from ultralytics.models.rtdetr.qcc_model import QCCTrainer
    require(Path(identifier).name == identifier, "Invalid dispatch ID")
    folder = OUT / "dispatches" / identifier
    record = read_json(folder / "dispatch.json")
    require(record, "Worker requires an existing reviewed dispatch record")
    require(os.environ.get("TMUX"), "Formal worker must run inside QCC tmux")
    session = subprocess.check_output(["tmux", "display-message", "-p", "-t", os.environ["TMUX_PANE"], "#S"], text=True, timeout=10).strip()
    require(session == SESSION, "Wrong tmux session")
    with (folder / "worker.claim").open("x") as f:
        f.write(str(os.getpid()))
    write_json(folder / "worker.json", dict(process_identity(os.getpid()), started=now(), command=record["command"]))
    exit_record = dict(status="FAILED", exit_code=1, started=now())
    try:
        require(binding() == record["identity"]["binding"], "Worker identity changed after dispatch")
        strict_amp_resources()
        args = deepcopy(read_json(OUT / "prepare.json")["args"])
        if record["resume"]:
            checkpoint_identity(RUN / "weights/last.pt", record["identity"], resume=True)
            args["model"] = args["resume"] = str(RUN / "weights/last.pt")
        else:
            require(not RUN.exists(), "Run created between dispatch and worker; preserved")
        trainer = QCCTrainer(overrides=args)
        trainer.qcc_output = OUT
        def attach_identity(t):
            t.model.qcc_identity = deepcopy(record["identity"])
            t.ema.ema.qcc_identity = deepcopy(record["identity"])
            actual = vars(t.args)
            allowed = {"model", "resume"} if record["resume"] else set()
            require(all(type(actual[k]) is type(v) and actual[k] == v for k, v in args.items() if k not in allowed), "Effective training recipe drift")
        trainer.add_callback("on_pretrain_routine_end", attach_identity)
        trainer.train()
        exit_record.update(status="TRAINING_COMPLETED", exit_code=0)
    except BaseException as error:
        exit_record.update(error=repr(error), traceback=traceback.format_exc(), geometry=getattr(error, "details", None))
        if (OUT / "training_completed.json").exists():
            exit_record["status"] = "TRAINING_COMPLETED+FINAL_EVAL_FAILED"
        raise
    finally:
        exit_record.update(ended=now(), best=file_info(RUN / "weights/best.pt"), last=file_info(RUN / "weights/last.pt"))
        write_json(folder / "exit.json", exit_record)

def evaluation_source(training):
    current = code_identity(clean=True)
    original = training["binding"]["code"]
    # Evaluation-only repairs have a separate identity, never rewrite training SHA.
    changed = set(p for p in set(current["files"]) | set(original["files"]) if current["files"].get(p) != original["files"].get(p))
    allowed = {"ultralytics-main/ultralytics/nn/autobackend.py", "ultralytics-main/ultralytics/models/rtdetr/qcc_val.py"}
    require(changed <= allowed, f"Evaluation code differs outside allowed eval-only files: {sorted(changed)}")
    return dict(training_commit=original["commit"], eval_commit=current["commit"], source=current, allowed_differences=sorted(changed))


def export_complete(lock, verify_hash=False):
    if not lock or lock.get("status") != "PASS" or not lock.get("predictions"):
        return False
    path = Path(lock["predictions"]["path"])
    marker = read_json(Path(str(path) + ".complete.json"), {})
    return bool(path.is_file() and marker.get("status") == "PASS" and marker.get("scope") == "full_split"
        and marker.get("predictions_sha256") == lock["predictions"].get("sha256")
        and marker.get("images") == COUNTS[lock["split"]][0] and marker.get("gt") == COUNTS[lock["split"]][1]
        and marker.get("identity", {}).get("checkpoint_sha256") == lock.get("checkpoint_sha256")
        and (not verify_hash or sha256(path) == marker["predictions_sha256"])
        and (path.parent / "evaluator_stats.npz").is_file() and (path.parent / "curves.json").is_file())


def finalize_weights():
    """Evaluation recovery after the loop; never restart completed training."""
    from ultralytics.utils.patches import torch_load
    from ultralytics.utils.torch_utils import strip_optimizer
    require(read_json(OUT / "training_completed.json", {}).get("status") == "TRAINING_COMPLETED", "Training has not completed")
    training = read_json(OUT / "training_identity.json")
    best, last = RUN / "weights/best.pt", RUN / "weights/last.pt"
    ckpt = checkpoint_identity(best, training)
    if ckpt["epoch"] >= 0:
        require(not read_json(OUT / "val_lock.json"), "Unstripped checkpoint conflicts with existing val lock")
        write_json(OUT / "best_selection.json", dict(epoch=ckpt["epoch"], fitness=ckpt["best_fitness"],
            rule="native val mAP50-95; save on best_fitness == fitness (latest tie)", pre_strip_sha256=sha256(best)))
        last_ckpt = torch_load(last, map_location="cpu") if last.is_file() else {}
        if last_ckpt.get("epoch", -1) >= 0:
            last_ckpt = strip_optimizer(last)
        strip_optimizer(best, updates={"train_results": last_ckpt.get("train_results")})
    write_json(OUT / "checkpoint_finalization.json", dict(best=file_info(best), last=file_info(last),
        resume="Training complete; native stripped checkpoints are weights only. Use finish/val recovery."))


def evaluate(split, in_worker=False, recover_export=False):
    import numpy as np
    from ultralytics.models.rtdetr.qcc_val import QCCValidator, EVAL, POLICY, SCHEMA
    from ultralytics.utils.torch_utils import init_seeds
    require(split in ("val", "test"), "Invalid evaluation split")
    if not in_worker:
        require(not active_workers() and not has_tmux(), "Wait for QCC worker/session before evaluation")
    else:
        require(split == "val", "Native final_eval can only run val")
    completion = read_json(OUT / "training_completed.json")
    require(completion and completion["status"] == "TRAINING_COMPLETED", "No native training-completion evidence")
    training = read_json(OUT / "training_identity.json")
    identity = evaluation_source(training)
    finalize_weights()
    best = RUN / "weights/best.pt"
    checkpoint_identity(best, training)
    weights_hash = sha256(best)
    path, data = data_config(), cached_data()
    require(data == training["binding"]["data"], "Prepared data identity differs from training")
    lock_path = OUT / f"{split}_lock.json"
    lock = read_json(lock_path)
    def same(record):
        return (record["checkpoint_sha256"] == weights_hash and record["data"] == data
                and record["eval_identity"] == identity and record["settings"] == EVAL and record["policy"] == POLICY)
    if lock:
        require(same(lock), "Existing evaluation lock has another checkpoint/data/code/protocol; preserved")
        if export_complete(lock, verify_hash=True):
            return dict(lock, reused=True)
        require(recover_export, f"{split} succeeded but full export is missing/corrupt; no implicit inference. Explicit recovery: qcc_v1.py {split} --recover-export")
        write_json(OUT / "history" / f"{split}_lock_{time.time_ns()}.json", lock)
    if split == "test":
        val = read_json(OUT / "val_lock.json")
        require(val and same(val) and export_complete(val, verify_hash=True), "Test requires complete same-best formal val lock")
    folder = OUT / "evaluations" / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}_{split}"
    folder.mkdir(parents=True, exist_ok=False)
    settings = dict(EVAL, model=str(best), data=str(path), split=split, device="0", plots=True,
                    save_json=False, save_txt=False, project=str(folder), name="plots", exist_ok=False)
    report = dict(status="FAIL", split=split, policy=POLICY, checkpoint_sha256=weights_hash, data=data,
        eval_identity=identity, settings=EVAL, actual_settings=settings, started=now(), scope="full_split", report=str(folder / "metrics.json"))
    validator = QCCValidator(args=settings, save_dir=folder / "plots")
    validator.export_path = folder / f"{split}_queries_gt.jsonl.gz"
    validator.export_identity = dict(checkpoint_sha256=weights_hash, data_content_sha256=data["content_sha256"],
        split=split, eval_commit=identity["eval_commit"], actual_requested_args=settings, schema=SCHEMA, policy=POLICY)
    validator.expected_images, validator.expected_gt = COUNTS[split]
    try:
        init_seeds(42, deterministic=True)
        metrics = validator(model=str(best))
        require(sha256(best) == weights_hash, "Checkpoint changed during evaluation")
        require(all(type(validator.actual_settings[k]) is type(v) and validator.actual_settings[k] == v for k, v in EVAL.items()), "Actual eval protocol differs")
        report.update(status="PASS", metrics=metrics, detailed_metrics=validator.qcc_metrics,
            images=len(validator.qcc_seen), actual_settings=validator.actual_settings, predictions=file_info(validator.export_path))
        require(export_complete(report, verify_hash=True), "Full predictions/GT export is incomplete")
        write_json(folder / "metrics.json", dict(report, ended=now()))
        write_json(lock_path, report)
        if split == "val" and read_json(OUT / "final_eval.json", {}).get("status") == "FINAL_EVAL_FAILED":
            write_json(folder / "evaluation_recovery.json", dict(status="EVALUATION_RECOVERED", recovered_at=now(),
                original_final_eval=read_json(OUT / "final_eval.json"), training_completion=completion, checkpoint_sha256=weights_hash))
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["ended"] = now()
        write_json(folder / "metrics.json", report)
    return report


def offline_analysis():
    """Recompute a val-locked operating point from exports, without inference."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.ops import xywh2xyxy
    locks = {split: read_json(OUT / f"{split}_lock.json") for split in ("val", "test")}
    if not all(export_complete(x, verify_hash=True) for x in locks.values()):
        return dict(status="PENDING", reason="Complete val/test locks and exports required")
    key = digest(dict(exports={s: r["predictions"]["sha256"] for s, r in locks.items()}, analysis_source=sha256(__file__)))
    prior = read_json(OUT / "offline_analysis.json")
    if prior and prior.get("exports_sha256") == key:
        return prior
    threshold = locks["val"]["detailed_metrics"]["reported_workpoint"]["confidence"]
    result = dict(status="PASS", exports_sha256=key, fixed_threshold_from="val only: native smoothed F1 workpoint",
                  confidence=threshold, IoU=0.5, matching="native IoU-greedy matcher recomputed AFTER fixed confidence filtering on saved raw queries; not QCC Hungarian",
                  class_average="single crack class; exact integer TP/FP/FN at the fixed threshold", splits={})
    for split, lock in locks.items():
        tp = fp = fn = 0
        score_above, gt_count, queries, images = 0, 0, 0, 0
        with gzip.open(lock["predictions"]["path"], "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                images += 1; queries += len(row["query_indices"]); gt_count += len(row["gt_classes"])
                scores = torch.tensor(row["scores"], dtype=torch.float32)
                order = scores.argsort(descending=True)
                selected = order[scores[order] >= threshold]
                score_above += len(selected)
                boxes = xywh2xyxy(torch.tensor(row["boxes_normalized_cxcywh"], dtype=torch.float32).reshape(-1,4))[selected] * 640
                gt = xywh2xyxy(torch.tensor(row["gt_boxes_normalized_cxcywh"], dtype=torch.float32).reshape(-1,4)) * 640
                classes = torch.tensor(row["classes"], dtype=torch.float32)[selected]
                gt_classes = torch.tensor(row["gt_classes"], dtype=torch.float32)
                hits = BaseValidator.match_predictions(SimpleNamespace(iouv=torch.tensor([.5])), classes, gt_classes, box_iou(gt, boxes))
                matched = int(hits.sum())
                tp += matched; fp += len(selected)-matched; fn += len(gt)-matched
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        result["splits"][split] = dict(TP=tp, FP=fp, FN=fn, Precision=precision, Recall=recall,
            F1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
            raw_export=dict(images=images, gt=gt_count, queries=queries, scores_ge_val_threshold=score_above))
    write_json(OUT / "offline_analysis.json", result)
    return result


def completeness():
    """Only inspect existing products. No imports that instantiate a detector."""
    required = ["prepare.json", "data_identity.json", "initialization.json", "recipe_diff.json", "train_args.yaml",
        "preflight.json", "training_identity.json", "training_completed.json", "best_selection.json", "epochs.jsonl",
        "val_lock.json", "test_lock.json", "source_snapshot.tar.gz"]
    missing = [p for p in required if not (OUT / p).is_file()]
    for split in ("val", "test"):
        lock = read_json(OUT / f"{split}_lock.json")
        if not export_complete(lock):
            missing.append(f"complete {split} query/GT export (explicit {split} --recover-export if an old successful lock lacks export)")
    if read_json(OUT / "training_completed.json", {}).get("status") != "TRAINING_COMPLETED":
        missing.append("verified native training completion")
    if not read_json(OUT / "preflight.json", {}).get("start_eligible"):
        missing.append("eligible preflight with mandatory gates PASS")
    val, test = read_json(OUT / "val_lock.json", {}), read_json(OUT / "test_lock.json", {})
    if val and test and any(val.get(k) != test.get(k) for k in ("checkpoint_sha256", "data", "eval_identity", "policy", "settings")):
        missing.append("val/test lock identities disagree")
    for p in (RUN / "results.csv", RUN / "args.yaml", RUN / "weights/best.pt", RUN / "weights/last.pt"):
        if not p.is_file(): missing.append(str(p))
    dispatches = list((OUT / "dispatches").glob("*/exit.json"))
    if not dispatches: missing.append("worker exit evidence")
    return dict(status="COMPLETE" if not missing else "INCOMPLETE", missing=missing)


def pack():
    """One analysis package, including all available exports, errors and sources."""
    import hashlib
    OUT.mkdir(parents=True, exist_ok=True)
    analysis = offline_analysis()
    complete = completeness()
    complete["offline_analysis"] = analysis.get("status")
    if analysis.get("status") != "PASS": complete["status"] = "INCOMPLETE"
    best = RUN / "weights/best.pt"
    val = read_json(OUT / "val_lock.json", {})
    if val and best.is_file() and sha256(best) != val.get("checkpoint_sha256"):
        complete["status"] = "INCOMPLETE"
        complete["missing"].append("current best bytes differ from locked evaluation checkpoint")
    filename = OUT / f"QCC_v1_ANALYSIS_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.tar.gz"
    partial = filename.with_suffix(filename.suffix + ".partial")
    manifest = []
    with tarfile.open(partial, "w:gz") as archive:
        def add_bytes(name, content, include_manifest=True):
            entry = tarfile.TarInfo(name); entry.size = len(content); entry.mtime = int(time.time())
            archive.addfile(entry, io.BytesIO(content))
            if include_manifest: manifest.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        def add_file(name, path):
            archive.add(path, arcname=name, recursive=False)
            manifest.append(dict(path=name, bytes=path.stat().st_size, sha256=sha256(path)))
        add_bytes("README.md", ("# QCC-v1-capK3 analysis\nStatus: " + complete["status"] + "\n"
            "Weights and original image/label datasets are excluded. Their paths, sizes and SHA256 identities are in summary/evidence.\n"
            "All available query+GT exports, curves, logs, errors, resume history and source are included in this single package.\n"
            "Missing: " + json.dumps(complete["missing"], ensure_ascii=False) + "\n"
            "PASS/PENDING/FAIL apply to recorded scopes only; no claim of AP gain. DDP unverified.\n").encode())
        for name in git("ls-files").splitlines():
            add_file("source/" + name, ROOT / name)
        for prefix, folder in (("evidence", OUT), ("training", RUN)):
            if not folder.exists(): continue
            for p in sorted(folder.rglob("*")):
                if not p.is_file() or p in (filename, partial) or p.suffix in {".pt", ".pth", ".tmp", ".partial"} or p.name.startswith("QCC_v1_ANALYSIS_"):
                    continue
                if p.name == "operation.lock": continue
                add_file(prefix + "/" + p.relative_to(folder).as_posix(), p)
        summary = dict(completeness=complete, lifecycle=status(), checkpoint_inventory=[file_info(RUN / "weights" / p) for p in ("best.pt", "last.pt")],
                       includes_weights=False, current_code=code_identity(), offline_analysis=analysis)
        add_bytes("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False, default=str).encode())
        add_bytes("manifest.json", json.dumps(manifest, indent=2).encode(), include_manifest=False)
    with tarfile.open(partial) as archive:
        recorded = json.load(archive.extractfile("manifest.json"))
        for row in recorded:
            with archive.extractfile(row["path"]) as stream:
                h = hashlib.sha256(); size = 0
                for block in iter(lambda: stream.read(1024*1024), b""):
                    h.update(block); size += len(block)
            require(size == row["bytes"] and h.hexdigest() == row["sha256"], "Package integrity failure")
    os.replace(partial, filename)
    result = dict(file_info(filename), completeness=complete)
    write_json(OUT / "last_package.json", result)
    return result


def finish():
    require(not active_workers() and not has_tmux(), "Wait for training worker to exit")
    require(read_json(OUT / "training_completed.json", {}).get("status") == "TRAINING_COMPLETED", "Training has not completed")
    evaluate("val")
    evaluate("test")
    offline_analysis()
    prior = read_json(OUT / "finish.json")
    if prior and Path(prior["package"]["path"]).is_file() and sha256(prior["package"]["path"]) == prior["package"]["sha256"]:
        return dict(prior, reused=True)
    result = dict(status="PASS", completed_at=now(), package=pack())
    require(result["package"]["completeness"]["status"] == "COMPLETE", "Finish package is incomplete; inspect missing entries")
    write_json(OUT / "finish.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Reuse fixed snapshot or inventory once; initialize from public weights")
    p.add_argument("--reuse-data", type=Path)
    p.add_argument("--recheck-data", action="store_true")
    for name in ("status", "start", "resume", "finish", "pack"):
        sub.add_parser(name)
    for name in ("preflight", "_preflight"):
        p = sub.add_parser(name)
        p.add_argument("--seconds", type=int, default=900)
        p.add_argument("--micro-batches", type=int, default=16)
    for name in ("val", "test"):
        sub.add_parser(name).add_argument("--recover-export", action="store_true", help="Explicit repeat only for an old successful evaluation missing its export")
    sub.add_parser("_worker", help=argparse.SUPPRESS).add_argument("--dispatch", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "status":
        result = dict(status(), materials=completeness())
    elif args.command == "_worker":
        worker(args.dispatch); return
    elif args.command == "_preflight":
        result = preflight(args.seconds, args.micro_batches)
    else:
        with operation_lock():
            if args.command == "prepare": result = prepare(args.reuse_data, args.recheck_data)
            elif args.command == "preflight": result = guarded_preflight(args.seconds, args.micro_batches)
            elif args.command == "start": result = dispatch(False)
            elif args.command == "resume": result = dispatch(True)
            elif args.command in ("val", "test"): result = evaluate(args.command, recover_export=args.recover_export)
            elif args.command == "finish": result = finish()
            elif args.command == "pack": result = pack()
    def compact(value):
        if isinstance(value, dict):
            return {k: (f"{len(v)} entries; see saved JSON" if k in ("files", "optimizer_groups", "groups") else compact(v)) for k, v in value.items()}
        if isinstance(value, list): return [compact(v) for v in value]
        return value
    print(json.dumps(compact(result), ensure_ascii=False, indent=2, allow_nan=False, default=str), flush=True)
    if args.command in ("preflight", "_preflight") and not result.get("start_eligible"):
        sys.exit(2)


if __name__ == "__main__":
    main()
