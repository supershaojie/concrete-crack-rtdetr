"""ARG v1 independent lifecycle. Explicit commands never start training implicitly."""
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

from arg_v1_common import (ROOT, MAIN, OUT, RUN, INIT, SOURCE, SESSION, BRANCH, BASE, SERVER_PYTHON,
    FORMULA, now, git, read_json, digest, runtime, code_identity, data_config, inventory, recipe,
    binding, prepare, sha256, require, write_json)


def process_identity(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return dict(pid=pid, created=p.create_time(), command=p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def active_workers():
    import psutil
    found = []
    for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
        cmd = p.info["cmdline"] or []
        if "_worker" in cmd and any(str(ROOT / "tools/arg_v1.py").replace("\\", "/") == s.replace("\\", "/") for s in cmd):
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
        require(not live or live["created"] != old["created"], f"ARG operation is active: {old}")
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
                print(f"ARG stage={label} elapsed={time.monotonic()-started:.1f}s remaining={remaining:.1f}s", flush=True)
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
        print("ARG stage=identity elapsed=0s", flush=True)
        report["binding"] = binding()
        require(remaining() > 0, "Preflight time boundary exhausted during identity checks")
        code = run_bounded([sys.executable, str(ROOT / "tools/check_arg_v1.py"), "--math-only", "--output", str(folder / "cpu")], remaining(), "CPU math/loss routing")
        require(code == 0, "CPU tests failed")
        report["checks"]["cpu"] = dict(status="PASS", report=str(folder / "cpu/checks.json"))
        write_json(OUT / "preflight.json", report)
        if os.name != "posix" or not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
            report["reason"] = "Formal CUDA B16/640 check requires Linux server and sufficient memory; local small tests do not qualify"
            return report
        report["amp_resources"] = strict_amp_resources()
        require(Path(sys.executable).resolve() == Path(SERVER_PYTHON).resolve(), "Use the fixed server Python")
        write_json(folder / "request.json", dict(binding=report["binding"], seconds=remaining(), micro_batches=micro_batches))
        code = run_bounded([sys.executable, str(ROOT / "tools/arg_v1_preflight.py"), "--folder", str(folder)], remaining(), "CUDA B16/640 lifecycle")
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
        code = run_bounded([sys.executable, str(ROOT / "tools/arg_v1.py"), "_preflight", "--seconds", str(seconds),
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
    require(getattr(model, "arg_identity", None) == training, "Checkpoint is not bound to this ARG training identity")
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
        print("ARG: native initial-scale adaptation is PENDING; isolated scale128 produced a real update. Formal scaler remains native.", flush=True)


def dispatch(resume=False):
    require(os.name == "posix" and Path(sys.executable).resolve() == Path(SERVER_PYTHON).resolve(), "Formal dispatch requires fixed server Python/Linux")
    require(not active_workers() and not has_tmux(), "ARG worker/tmux already exists; status only, no replacement")
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
    command = [sys.executable, "-u", str(ROOT / "tools/arg_v1.py"), "_worker", "--dispatch", identifier]
    record = dict(created=now(), resume=resume, command=command, cwd=str(ROOT), identity=train_identity, status="DISPATCHED")
    write_json(folder / "dispatch.json", record)
    environment = dict(PYTHONPATH=str(ROOT / "ultralytics-main") + os.pathsep + str(ROOT / "tools"),
                       ARG_V1_MAIN=str(MAIN), PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="false")
    for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"):
        environment[name] = os.environ.get(name)
    exports = "".join(("unset " + k if v is None else "export " + k + "=" + shlex.quote(v)) + "\n" for k, v in environment.items())
    shell = ("#!/usr/bin/env bash\nset -uo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n" + exports +
             shlex.join(command) + " 2>&1 | tee -a " + shlex.quote(str(folder / "console.log")) + "\n" +
             "codes=(\"${PIPESTATUS[@]}\")\nprintf '%s\\n' \"${codes[0]}\" > " + shlex.quote(str(folder / "python_exit_code.txt")) + "\n" +
             "printf '%s\\n' \"${codes[1]}\" > " + shlex.quote(str(folder / "tee_exit_code.txt")) + "\nexit \"${codes[0]}\"\n")
    (folder / "worker.sh").write_text(shell, encoding="utf-8", newline="\n")
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "bash " + shlex.quote(str(folder / "worker.sh"))], check=True, timeout=15)
    return dict(status="DISPATCHED", session=SESSION, log=str(folder / "console.log"), worker_command=command)


def worker(identifier):
    from ultralytics.models.rtdetr.arg_model import ARGTrainer
    require(Path(identifier).name == identifier, "Invalid dispatch ID")
    folder = OUT / "dispatches" / identifier
    record = read_json(folder / "dispatch.json")
    require(record, "Worker requires an existing reviewed dispatch record")
    require(os.environ.get("TMUX"), "Formal worker must run inside ARG tmux")
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
        trainer = ARGTrainer(overrides=args)
        trainer.arg_output = OUT
        def attach_identity(t):
            t.model.arg_identity = deepcopy(record["identity"])
            t.ema.ema.arg_identity = deepcopy(record["identity"])
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
    allowed = {"ultralytics-main/ultralytics/nn/autobackend.py", "ultralytics-main/ultralytics/models/rtdetr/arg_val.py"}
    require(changed <= allowed, f"Evaluation code differs outside allowed eval-only files: {sorted(changed)}")
    return dict(training_commit=original["commit"], eval_commit=current["commit"], source=current, allowed_differences=sorted(changed))


def evaluate(split, export=False):
    import numpy as np
    from ultralytics.models.rtdetr.arg_val import ARGValidator, EVAL, POLICY
    from ultralytics.utils.torch_utils import init_seeds
    require(not active_workers() and not has_tmux(), "Wait for ARG worker/session to finish before independent evaluation")
    completion = read_json(OUT / "training_completed.json")
    require(completion and completion["status"] == "TRAINING_COMPLETED", "No native training-completion evidence")
    training = read_json(OUT / "training_identity.json")
    identity = evaluation_source(training)
    best = RUN / "weights/best.pt"
    checkpoint_identity(best, training)
    weights_hash = sha256(best)
    data_path = data_config(); data = inventory(data_path)
    require(data == training["binding"]["data"], "Training/evaluation data identity differs")
    lock = read_json(OUT / "val_lock.json")
    if split == "test":
        require(lock and lock["checkpoint_sha256"] == weights_hash and lock["data"] == data
                and lock["eval_identity"] == identity and lock["settings"] == EVAL and lock["policy"] == POLICY, "Test requires same best, val lock, code, data and protocol")
        require(not (OUT / "test_lock.json").exists(), "Final test already exists; use status/pack, no repeat selection")
    folder = OUT / "evaluations" / (time.strftime("%Y%m%d_%H%M%S") + "_" + split)
    folder.mkdir(parents=True, exist_ok=False)
    settings = dict(EVAL, model=str(best), data=str(data_path), split=split, device="0", plots=True,
                    save_json=False, save_txt=False, project=str(folder), name="plots", exist_ok=False)
    report = dict(status="FAIL", split=split, policy=POLICY, checkpoint_sha256=weights_hash, data=data,
                  eval_identity=identity, settings=EVAL, actual_settings=settings, started=now(), scope="full_split")
    validator = ARGValidator(args=settings, save_dir=folder / "plots")
    if export:
        validator.export_path = folder / f"{split}_predictions_gt.jsonl.gz"
        validator.export_identity = dict(checkpoint_sha256=weights_hash, split=split, eval_commit=identity["eval_commit"], config_sha256=digest(settings))
    try:
        init_seeds(42, deterministic=True)
        metrics = validator(model=str(best))
        ap = np.asarray(validator.metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Expected finite one-class ten-IoU AP")
        require(sha256(best) == weights_hash, "Checkpoint changed during evaluation")
        require(all(type(validator.actual_settings[k]) is type(v) and validator.actual_settings[k] == v for k, v in EVAL.items()), "Effective independent eval protocol differs")
        report.update(status="PASS", metrics=metrics, AP50=float(ap[0, 0]), AP50_95=float(ap.mean()),
                      ap_by_class=ap.tolist(), ap_iou_thresholds=[round(.5 + .05 * i, 2) for i in range(10)],
                      images=len(validator.arg_seen), actual_settings=validator.actual_settings,
                      predictions=file_info(validator.export_path) if export else None)
        write_json(OUT / ("val_lock.json" if split == "val" else "test_lock.json"),
                   dict(report, report=str(folder / "metrics.json")))
        # Append recovery evidence, preserving original final_eval failure and worker exit1.
        if split == "val" and read_json(OUT / "final_eval.json", {}).get("status") == "FINAL_EVAL_FAILED":
            write_json(folder / "evaluation_recovery.json", dict(status="EVALUATION_RECOVERED", original_final_eval=read_json(OUT / "final_eval.json"),
                       training_completion=completion, eval_identity=identity, checkpoint_sha256=weights_hash))
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["ended"] = now(); write_json(folder / "metrics.json", report)
    return report


def pack(include_predictions=False):
    """Failure evidence is always packageable; no implicit training, val or test."""
    OUT.mkdir(parents=True, exist_ok=True)
    if include_predictions and read_json(OUT / "val_lock.json"):
        lock = read_json(OUT / "val_lock.json")
        if not lock.get("predictions"):
            export_locked_val(lock)
    filename = OUT / ("ARG_v1_LIGHT_" + time.strftime("%Y%m%d_%H%M%S") + ".tar.gz")
    require(not filename.exists(), "Package exists; preserved")
    manifest = []
    with tarfile.open(filename, "w:gz") as archive:
        def add_bytes(name, content):
            entry = tarfile.TarInfo(name); entry.size = len(content); entry.mtime = int(time.time())
            archive.addfile(entry, io.BytesIO(content))
            manifest.append(dict(path=name, bytes=len(content), sha256=__import__("hashlib").sha256(content).hexdigest()))
        add_bytes("README.md", ("# ARG v1 LIGHT\nNo datasets or best/last weights are included.\n"
            "PASS/PENDING/FAIL refer only to each recorded scope. Missing training/test is not success.\n"
            "See source/docs/arg_v1 and evidence for exact training/evaluation identities.\n").encode())
        for name in git("ls-files").splitlines():
            p = ROOT / name
            add_bytes("source/" + name, p.read_bytes())
        for prefix, folder in (("evidence", OUT), ("training", RUN)):
            if not folder.exists():
                continue
            for p in sorted(folder.rglob("*")):
                if not p.is_file() or p == filename or p.suffix in {".pt", ".pth", ".tmp"} or "LIGHT_" in p.name:
                    continue
                if "predictions_gt" in p.name and not include_predictions:
                    continue
                if p.name == "source_snapshot.tar.gz":
                    continue  # current source below + original commit snapshot separately if source differs
                add_bytes(prefix + "/" + p.relative_to(folder).as_posix(), p.read_bytes())
        training = read_json(OUT / "training_identity.json")
        if training:
            commit = training["binding"]["code"]["commit"]
            original = subprocess.check_output(["git", "archive", "--format=tar.gz", commit], cwd=ROOT, timeout=60)
            add_bytes("training_source_snapshot.tar.gz", original)
        summary = dict(status=status(), checkpoint_inventory=[file_info(RUN / "weights" / name) for name in ("best.pt", "last.pt")],
                       includes_weights=False, include_predictions=include_predictions, current_code=code_identity())
        add_bytes("summary.json", json.dumps(summary, ensure_ascii=False, allow_nan=False, default=str, indent=2).encode())
        add_bytes("manifest.json", json.dumps(manifest, indent=2).encode())
    with tarfile.open(filename) as archive:
        recorded = json.load(archive.extractfile("manifest.json"))
        for row in recorded:
            content = archive.extractfile(row["path"]).read()
            require(len(content) == row["bytes"] and __import__("hashlib").sha256(content).hexdigest() == row["sha256"], "Package integrity failure")
    return file_info(filename)


def export_locked_val(lock):
    """Export all queries of the already locked best; do not perform test or selection."""
    from ultralytics.models.rtdetr.arg_val import ARGValidator, EVAL, POLICY
    require(not active_workers(), "Wait for training to finish")
    training = read_json(OUT / "training_identity.json")
    require(evaluation_source(training) == lock["eval_identity"], "Export source differs from val lock")
    best = RUN / "weights/best.pt"
    require(sha256(best) == lock["checkpoint_sha256"], "Export checkpoint differs from val lock")
    path = data_config(); require(inventory(path) == lock["data"], "Export dataset changed")
    folder = OUT / "exports" / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    validator = ARGValidator(args=dict(EVAL, data=str(path), model=str(best), split="val", device="0", plots=False), save_dir=folder)
    validator.export_path = folder / "val_predictions_gt.jsonl.gz"
    validator.export_identity = dict(checkpoint_sha256=lock["checkpoint_sha256"], policy=POLICY, split="val", val_lock_sha256=sha256(OUT / "val_lock.json"))
    try:
        validator(model=str(best))
        write_json(folder / "identity.json", dict(status="PASS", val_lock=lock, predictions=file_info(validator.export_path)))
    except BaseException as error:
        write_json(folder / "identity.json", dict(status="FAIL", error=repr(error))); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "status", "start", "resume", "probe"):
        sub.add_parser(name)
    for name in ("preflight", "_preflight"):
        pre = sub.add_parser(name)
        pre.add_argument("--seconds", type=int, default=900); pre.add_argument("--micro-batches", type=int, default=16)
    for name in ("val", "test"):
        sub.add_parser(name).add_argument("--include-predictions", action="store_true")
    sub.add_parser("pack").add_argument("--include-predictions", action="store_true")
    sub.add_parser("_worker", help=argparse.SUPPRESS).add_argument("--dispatch", required=True)
    args = parser.parse_args()
    if args.command == "status":
        result = status()
    elif args.command == "_worker":
        worker(args.dispatch); return
    elif args.command == "_preflight":
        result = preflight(args.seconds, args.micro_batches)
    else:
        with operation_lock():
            if args.command == "prepare": result = prepare()
            elif args.command == "preflight": result = guarded_preflight(args.seconds, args.micro_batches)
            elif args.command == "start": result = dispatch(False)
            elif args.command == "resume": result = dispatch(True)
            elif args.command in ("val", "test"): result = evaluate(args.command, args.include_predictions)
            elif args.command == "pack": result = pack(args.include_predictions)
            elif args.command == "probe":
                code = run_bounded([sys.executable, str(ROOT / "tools/check_arg_v1.py"), "--math-only"], 120, "ARG mechanism CPU")
                require(code == 0, "ARG CPU probe failed")
                result = dict(status="PASS", scope="CPU formula/routing; CUDA effect belongs to preflight")
    def compact(value):
        if isinstance(value, dict):
            return {k: (f"{len(v)} entries; see saved JSON" if k in ("files", "optimizer_groups", "groups") else compact(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    print(json.dumps(compact(result), ensure_ascii=False, indent=2, allow_nan=False, default=str), flush=True)
    if args.command in ("preflight", "_preflight") and not result.get("start_eligible"):
        sys.exit(2)


if __name__ == "__main__":
    main()
