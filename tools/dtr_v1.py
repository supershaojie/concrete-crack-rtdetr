"""Fixed DTR v1 lifecycle. start/finish/test are always explicit commands."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from dtr_v1_common import *
from experiment_runtime import operation_lock, process_identity, run_bounded


def active_workers():
    import psutil
    return [p.info for p in psutil.process_iter(["pid", "cmdline", "create_time"])
            if "_worker" in (p.info["cmdline"] or []) and
            any(str(ROOT / "tools/dtr_v1.py").replace("\\", "/") == s.replace("\\", "/")
                for s in (p.info["cmdline"] or []))]


def has_tmux():
    return bool(shutil.which("tmux") and subprocess.run(["tmux", "has-session", "-t", "=" + SESSION],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0)


def status():
    dispatches = sorted((OUT / "dispatches").glob("*/dispatch.json"))
    latest = dispatches[-1].parent if dispatches else None
    pre = read_json(OUT / "preflight.json", {})
    return dict(active_workers=active_workers(), tmux=dict(name=SESSION, exists=has_tmux()),
                dispatch=read_json(latest / "dispatch.json") if latest else None,
                worker=read_json(latest / "worker.json") if latest else None,
                exit=read_json(latest / "exit.json") if latest else None,
                python_exit_code=(latest / "python_exit_code.txt").read_text().strip() if latest and (latest / "python_exit_code.txt").exists() else None,
                progress=read_json(OUT / "progress.json"),
                training=read_json(OUT / "training_completed.json", {"status": "NOT_COMPLETED"}),
                preflight=dict(status=pre.get("status", "NOT_RUN"), folder=pre.get("folder"),
                    reason=pre.get("reason") or pre.get("error") or pre.get("gpu", {}).get("reason"),
                    checks={k: v.get("status") for k, v in pre.get("checks", {}).items()},
                    micro_batches=pre.get("gpu", {}).get("micro_batches"), scale_updates=pre.get("gpu", {}).get("updates", [])),
                formal_evaluations={s: read_json(OUT / f"{s}_lock.json", {}).get("status", "NOT_RUN") for s in ("val", "test")},
                best=file_info(RUN / "weights/best.pt", False), last=file_info(RUN / "weights/last.pt", False))


def server_check():
    import torch
    require(os.name == "posix" and Path(sys.executable).resolve() == Path(SERVER_PYTHON).resolve(),
            f"Server operations require {SERVER_PYTHON} on Linux; actual {sys.executable}")
    require(torch.cuda.is_available(), "CUDA device 0 required")
    require(shutil.which("nvidia-smi"), "nvidia-smi required to check other experiments")
    query = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                                     "--format=csv,noheader,nounits"], text=True, timeout=15)
    others = [int(s.strip()) for s in query.splitlines() if s.strip().isdigit() and int(s.strip()) != os.getpid()]
    require(not others, f"GPU 0 has other processes {others}; wait for them, no killing or batch reduction")


def preflight(seconds, micro_batches):
    import torch
    require(1 <= seconds <= 900 and 4 <= micro_batches <= 16, "Limits: 1..900 seconds, 4..16 micro-batches")
    require(not active_workers() and not has_tmux(), "DTR worker/session active")
    folder = OUT / "preflights" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status="PENDING", start_eligible=False, started=now(), folder=str(folder),
                  boundary=dict(seconds=seconds, micro_batches=micro_batches),
                  checks={k: dict(status="PENDING") for k in ("cpu", "cuda_b16_amp", "native_scale", "mechanism", "resume", "new_process_val")})
    write_json(folder / "preflight.json", report)
    write_json(OUT / "preflight.json", report)
    started = time.monotonic()
    def remaining():
        return seconds - (time.monotonic() - started)
    try:
        report["binding"] = binding()
        print("DTR stage=CPU formula/routing", flush=True)
        code = run_bounded([sys.executable, str(ROOT / "tools/check_dtr_v1.py"), "--math-only",
                            "--output", str(folder / "cpu")], remaining(), "CPU", ROOT)
        require(code == 0, "CPU formula/routing failed")
        report["checks"]["cpu"] = dict(status="PASS", path=str(folder / "cpu/checks.json"))
        if os.name != "posix" or not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
            report["reason"] = "Formal Linux B16/640 AMP lifecycle not run on this machine; no smaller batch substituted"
            return report
        server_check()
        report["amp_resources"] = strict_amp_resources()
        write_json(folder / "request.json", dict(seconds=remaining(), micro_batches=micro_batches, binding=report["binding"]))
        code = run_bounded([sys.executable, str(ROOT / "tools/dtr_v1_preflight.py"), "--folder", str(folder)],
                           remaining(), "native CUDA lifecycle", ROOT)
        gpu = read_json(folder / "gpu.json", {})
        report["gpu"] = gpu
        report["checks"].update(gpu.get("checks", {}))
        require(code in (0, 2) and gpu, "GPU preflight failed")
        report["start_eligible"] = all(report["checks"].get(k, {}).get("status") == "PASS"
            for k in ("cpu", "cuda_b16_amp", "native_scale", "mechanism", "resume", "new_process_val"))
        report["status"] = "PASS" if report["start_eligible"] else "PENDING"
    except BaseException as error:
        report.update(status="FAIL", start_eligible=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(folder / "preflight.json", report)
        write_json(OUT / "preflight.json", report)
    return report


def guarded_preflight(seconds, micro_batches):
    require(1 <= seconds <= 900 and 4 <= micro_batches <= 16, "Limits: 1..900 seconds, 4..16 micro-batches")
    try:
        result = run_bounded([sys.executable, str(ROOT / "tools/dtr_v1.py"), "_preflight",
                              "--seconds", str(seconds), "--micro-batches", str(micro_batches)],
                             seconds, "entire preflight", ROOT)
        require(result in (0, 2), f"Preflight exited {result}")
        return read_json(OUT / "preflight.json")
    except BaseException as error:
        report = read_json(OUT / "preflight.json", {})
        report.update(status="FAIL", start_eligible=False, boundary_error=repr(error))
        write_json(OUT / "preflight.json", report)
        if report.get("folder"):
            folder = Path(report["folder"]).resolve()
            if folder.parent == (OUT / "preflights").resolve():
                write_json(folder / "preflight.json", report)
        raise


def checkpoint_identity(path, training, resume=False):
    from ultralytics.utils.patches import torch_load
    ckpt = torch_load(path, map_location="cpu")
    model = ckpt.get("ema") or ckpt.get("model")
    require(getattr(model, "dtr_identity", None) == training["id"], "Checkpoint is not this DTR run")
    if resume:
        require(0 <= ckpt.get("epoch", -1) < 199 and ckpt.get("optimizer") is not None and ckpt.get("scaler"),
                "last is stripped/completed or lacks optimizer/scaler: training resume impossible; use finish for evaluation recovery")
    return ckpt


def verify_preflight(current):
    pre = read_json(OUT / "preflight.json", {})
    require(pre.get("status") == "PASS" and pre.get("start_eligible"), "Valid formal preflight required; PENDING does not authorize start")
    require(pre.get("binding") == current, "Preflight code/data/init/recipe identity changed")
    for key in ("cpu", "cuda_b16_amp", "native_scale", "mechanism", "resume", "new_process_val"):
        require(pre.get("checks", {}).get(key, {}).get("status") == "PASS", f"Preflight check is not PASS: {key}")
    capacity = pre["checks"]["cuda_b16_amp"]
    require(capacity["effective_updates"] > 0 and capacity["batch"] == 16 and capacity["imgsz"] == 640 and capacity["amp"],
            "No actual native-scaler B16/640 optimizer update evidence")
    return pre


def dispatch(resume=False):
    require(shutil.which("tmux"), "tmux is required")
    require(not active_workers() and not has_tmux(), "DTR already active; use status")
    server_check()
    current = binding()
    verify_preflight(current)
    training = read_json(OUT / "training_identity.json")
    if resume:
        require(training and training["binding"] == current, "Resume requires original immutable run binding")
        require(not (OUT / "training_completed.json").exists(), "Training completed; use finish/val to recover evaluation")
        checkpoint_identity(RUN / "weights/last.pt", training, True)
    else:
        require(training is None and not RUN.exists(), "Existing DTR run preserved; use status/resume")
        training = dict(id=uuid.uuid4().hex, started=now(), binding=current, args=read_json(OUT / "prepare.json")["args"])
        write_json(OUT / "training_identity.json", training)
        subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(OUT / "source_snapshot.tar.gz"),
                        current["code"]["commit"]], cwd=ROOT, check=True, timeout=60)
        training["source_snapshot"] = file_info(OUT / "source_snapshot.tar.gz")
        write_json(OUT / "training_identity.json", training)
    folder = OUT / "dispatches" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(ROOT / "tools/dtr_v1.py"), "_worker", "--dispatch", str(folder)]
    record = dict(created=now(), resume=resume, run_id=training["id"], command=command, session=SESSION)
    write_json(folder / "dispatch.json", record)
    shell = ("#!/usr/bin/env bash\nset -uo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n"
             + shlex.join(command) + " 2>&1 | tee " + shlex.quote(str(folder / "console.log")) + "\n"
             + "codes=(\"${PIPESTATUS[@]}\")\nprintf '%s\\n' \"${codes[0]}\" > " + shlex.quote(str(folder / "python_exit_code.txt")) + "\n"
             + "printf '%s\\n' \"${codes[1]}\" > " + shlex.quote(str(folder / "tee_exit_code.txt")) + "\nexit \"${codes[0]}\"\n")
    with (folder / "worker.sh").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(shell)
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "bash " + shlex.quote(str(folder / "worker.sh"))],
                   check=True, timeout=15)
    return dict(record, folder=str(folder), run=str(RUN))


def worker(folder):
    from ultralytics.models.rtdetr.dtr_trainer import DTRTrainer
    folder = Path(folder).resolve()
    require(folder.parent == (OUT / "dispatches").resolve(), "Dispatch outside this experiment")
    dispatch_info = read_json(folder / "dispatch.json")
    training = read_json(OUT / "training_identity.json")
    code = 1
    write_json(folder / "worker.json", dict(process_identity(os.getpid()), started=now(), run_id=training["id"]))
    try:
        require(binding() == training["binding"], "Binding changed between launch and worker")
        require(dispatch_info["run_id"] == training["id"], "Wrong dispatch run")
        args = dict(training["args"])
        if dispatch_info["resume"]:
            checkpoint_identity(RUN / "weights/last.pt", training, True)
            args.update(model=str(RUN / "weights/last.pt"), resume=str(RUN / "weights/last.pt"))
        trainer = DTRTrainer(overrides=args)
        trainer.dtr_output, trainer.dtr_identity = str(OUT), training["id"]
        trainer.train()
        require(read_json(OUT / "training_completed.json", {}).get("status") == "TRAINING_COMPLETED",
                "Trainer returned without native completion evidence")
        code = 0
    except BaseException as error:
        write_json(folder / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(), time=now()))
        raise
    finally:
        write_json(folder / "exit.json", dict(python_exit_code=code, ended=now(), run_id=training["id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--refresh-data", action="store_true")
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--manifest", type=Path)
    for name in ("status", "start", "resume", "val", "test", "finish", "pack"):
        sub.add_parser(name)
    for name in ("preflight", "_preflight"):
        p = sub.add_parser(name)
        p.add_argument("--seconds", type=int, default=900)
        p.add_argument("--micro-batches", type=int, default=16)
    sub.add_parser("_worker").add_argument("--dispatch", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(4)
    if args.command == "status":
        result = status()
    elif args.command == "_worker":
        worker(args.dispatch)
        return
    elif args.command == "_preflight":
        result = preflight(args.seconds, args.micro_batches)
    else:
        with operation_lock(OUT):
            if args.command == "prepare":
                result = prepare(args.refresh_data, args.snapshot, args.manifest)
            elif args.command == "preflight":
                result = guarded_preflight(args.seconds, args.micro_batches)
            elif args.command in ("start", "resume"):
                result = dispatch(args.command == "resume")
            else:
                from dtr_v1_eval import evaluate, finish, package
                require(not active_workers() and not has_tmux(), "Wait for DTR worker/session before finish/eval/pack")
                result = finish() if args.command == "finish" else package() if args.command == "pack" else evaluate(args.command)
    def compact(value):
        if isinstance(value, dict):
            return {k: (f"{len(v)} entries; see saved evidence" if k in ("files", "binding", "args", "data") else compact(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    import json
    print(json.dumps(compact(result), ensure_ascii=False, indent=2, allow_nan=False, default=str))
    if args.command in ("preflight", "_preflight") and not result.get("start_eligible"):
        sys.exit(2)


if __name__ == "__main__":
    main()
