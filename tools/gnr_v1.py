"""GNR v1 lifecycle. Only explicit start/resume launches formal training.

Shared GPU0 is supported. Locks protect this experiment's run, never the GPU.
status/pack import no torch/model code and never scan the original dataset.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

from gnr_v1_common import (ROOT, OUT, RUN, INIT, SERVER_MAIN, SERVER_WT, SERVER_PY, BRANCH,
    BASE, COUNTS, now, require, read_json, write_json, sha256, binding, source_identity, prepare)


def process_identity(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return {"pid": pid, "created": p.create_time(), "command": p.cmdline(), "cwd": p.cwd()}
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
        return None


def active_workers():
    import psutil
    result = []
    script = str(ROOT / "tools/gnr_v1.py").replace("\\", "/").lower()
    for p in psutil.process_iter():
        try:
            cmd = p.cmdline() or []
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        if any(x.replace("\\", "/").lower() == script for x in cmd) and any(x in cmd for x in ("_worker", "_preflight", "_diagnose")):
            info = process_identity(p.pid)
            if info and Path(info["cwd"]).resolve() == ROOT:
                result.append(info)
    return result


@contextmanager
def operation_lock(wait=False):
    """Kernel lock is race-safe and released on death; stale pid files are not locks."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "run.lock"
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            deadline = time.monotonic() + (30 if wait else 0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("This GNR run already has an active writer") from error
                    time.sleep(.05)
        else:
            import fcntl
            deadline = time.monotonic() + (30 if wait else 0)
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as error:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("This GNR run already has an active writer") from error
                    time.sleep(.05)
        write_json(OUT / "lock_owner.json", process_identity(os.getpid()))
        try:
            yield
        finally:
            (OUT / "lock_owner.json").unlink(missing_ok=True)
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def status():
    plan = read_json(OUT / "prepare.json")
    training = read_json(OUT / "training.json")
    if training and training.get("status") == "RUNNING":
        saved = training.get("process", {})
        actual = process_identity(saved.get("pid", -1))
        if not actual or any(actual.get(k) != saved.get(k) for k in ("created", "command", "cwd")):
            training = dict(training, status="INACTIVE_INTERRUPTED", note="Stored RUNNING process identity is no longer live; no automatic recovery")
    def brief(name):
        value = read_json(OUT / name)
        return {k: value[k] for k in ("status", "reason", "coverage_complete", "folder", "exit_code", "effective_updates", "microbatches", "precision", "recall", "F1", "AP50", "AP75", "mAP50_95", "mother_delta_pp", "best_sha256") if k in value} if value else None
    return {"run": str(RUN), "out": str(OUT), "active_workers": active_workers(),
            "prepare": "PREPARED" if plan else "MISSING", "diagnose": brief("diagnose.json"),
            "preflight": brief("preflight.json"), "training": training,
            "completed": read_json(OUT / "training_completed.json"), "progress": read_json(OUT / "progress.json"),
            "latest_training_val": read_json(OUT / "checkpoint_state.json"),
            "val": brief("evaluation_val.json"), "test": brief("evaluation_test.json"),
            "package": read_json(OUT / "package.json"), "latest_dispatch": read_json(OUT / "latest_dispatch.json"),
            "exit_records": [read_json(p) for p in sorted((OUT / "dispatches").glob("*/exit.json"))],
            "missing": [name for name in ("prepare.json", "diagnose.json", "preflight.json", "training_completed.json", "evaluation_val.json", "evaluation_test.json", "threshold_analysis.json") if not (OUT / name).is_file()]}


def ready(job):
    current = binding()
    plan = read_json(OUT / "prepare.json")
    require(not plan["development"], "Development preparation is not a formal server run; run server prepare")
    require(plan["recipe_audit"]["actual_status"] == "CHECKED", "Actual mother args comparison still PENDING; prepare with --mother-args")
    require(sys.platform.startswith("linux"), "Formal tmux worker requires the server Linux environment")
    technical = read_json(OUT / "preflight.json")
    require(technical and technical.get("status") == "TECHNICAL_PASS" and technical["binding"] == current,
            "Missing/stale TECHNICAL_PASS; run bounded preflight")
    diagnostic = read_json(OUT / "diagnose.json")
    require(diagnostic and diagnostic.get("binding") == current, "Mother diagnosis is missing/stale; run diagnose")
    require(diagnostic.get("status") == "REVIEW", f"Diagnosis {diagnostic.get('status')}: {diagnostic.get('reason')}; start refused")
    print("Mother diagnostic conclusion:", diagnostic.get("status"), "coverage_complete=", diagnostic.get("coverage_complete"), flush=True)
    require(not (OUT / "training_completed.json").exists(), "Training already completed; use finish, never resume/retrain")
    if job == "start":
        require(not RUN.exists(), f"Existing run preserved: {RUN}; use resume for an interrupted run")
    else:
        from ultralytics.utils.patches import torch_load
        last = RUN / "weights/last.pt"
        require(last.is_file(), "No same-run full last checkpoint to resume")
        ckpt = torch_load(last, map_location="cpu")
        require(0 <= ckpt.get("epoch", -1) < 199 and ckpt.get("optimizer") is not None and ckpt.get("scaler") is not None and ckpt.get("ema") is not None,
                "Checkpoint completed/stripped/incomplete; true resume is unavailable")
        require(getattr(ckpt["ema"], "gnr_identity", None) == current, "Resume checkpoint binding differs")
        actual = ckpt["train_args"]
        expected = plan["args"]
        require(all(actual.get(k) == v and type(actual.get(k)) is type(v) for k, v in expected.items() if k not in ("model", "resume")), "Resume recipe differs")
    return current


def run_bounded(action, args):
    current = binding(clean=not args.development)
    folder = OUT / action / now()
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", {"binding": current, "seconds": args.seconds, "max_batches": args.max_batches})
    command = [sys.executable, "-u", str(ROOT / "tools/gnr_v1.py"), "_" + action, "--folder", str(folder),
               "--seconds", str(args.seconds), "--max-batches", str(args.max_batches), "--val-images", str(args.val_images), "--train-batches", str(args.train_batches)]
    if args.mother_best:
        command += ["--mother-best", str(Path(args.mother_best).resolve())]
    if args.development:
        command.append("--development")
    started = time.monotonic()
    with (folder / "console.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        write_json(folder / "process.json", process_identity(child.pid))
        expired = False
        try:
            code = child.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            expired = True
            import psutil
            # Only descendants of the child we just created, never other experiments.
            owned = psutil.Process(child.pid).children(recursive=True)
            for p in reversed(owned):
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            _, alive = psutil.wait_procs(owned, timeout=1)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            code = 2
    record = read_json(folder / f"{action}.json", {})
    if expired:
        record.update(status="PENDING", reason="Total time budget exhausted; completed coverage retained", coverage_complete=False)
    elif code not in (0, 2) and record.get("status") not in ("RESOURCE_ERROR", "IMPLEMENTATION_ERROR"):
        record.update(status="IMPLEMENTATION_ERROR", reason=f"Child exited {code}; inspect console.log")
    record.update(binding=current, exit_code=code, elapsed_seconds=time.monotonic() - started, folder=str(folder))
    if source_identity(clean=False)["functional_sha256"] != current["identity"]["functional_sha256"]:
        record.update(status="STALE", reason="Functional source changed while the bounded process ran; evidence is not reusable")
        code = 2
        record["exit_code"] = code
    write_json(folder / f"{action}.json", record)
    write_json(OUT / f"{action}.json", record)
    print((folder / "console.log").read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
    return record, code


def tmux_exists(name):
    return shutil.which("tmux") and subprocess.run(["tmux", "has-session", "-t", "=" + name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0


def dispatch(job):
    require(shutil.which("tmux"), "tmux is required for persistent server workers")
    require(not active_workers(), "This GNR worktree already has an active worker")
    if job in ("start", "resume"):
        current = ready(job)
    else:
        current = binding()
        require(read_json(OUT / "training_completed.json"), "Training is unfinished; finish cannot train/resume")
    session = "gnr-v1-finish" if job == "finish" else "gnr-v1-training"
    require(not tmux_exists(session), f"Existing session {session} preserved; inspect its actual command")
    folder = OUT / "dispatches" / now()
    folder.mkdir(parents=True)
    record = {"job": job, "folder": str(folder), "binding": current, "session": session, "time": now()}
    write_json(folder / "dispatch.json", record)
    write_json(OUT / "latest_dispatch.json", record)
    q = shlex.quote
    tool = str(ROOT / "tools/gnr_v1.py")
    shell = ["#!/usr/bin/env bash", "set +e", f"cd {q(str(ROOT))} || exit 90",
             f"{q(sys.executable)} -u {q(tool)} _worker --folder {q(str(folder))} 2>&1 | tee -a {q(str(folder / 'console.log'))} {q(str(OUT / 'view.log'))}",
             'codes=("${PIPESTATUS[@]}")',
             f'{q(sys.executable)} {q(tool)} _record-exit --folder {q(str(folder))} --python-code "${{codes[0]}}" --tee-code "${{codes[1]}}"',
             'result="${codes[0]}"', 'if [[ "$result" == 0 && "${codes[1]}" != 0 ]]; then result="${codes[1]}"; fi']
    if job == "finish":
        shell += ['if [[ "${codes[0]}" == 0 && "${codes[1]}" == 0 ]]; then',
                  f"  {q(sys.executable)} -u {q(tool)} pack 2>&1 | tee -a {q(str(folder / 'pack.log'))} {q(str(OUT / 'view.log'))}",
                  '  pack_codes=("${PIPESTATUS[@]}")',
                  f'  {q(sys.executable)} {q(tool)} _record-exit --folder {q(str(folder))} --python-code "${{pack_codes[0]}}" --tee-code "${{pack_codes[1]}}" --pack-exit',
                  '  result="${pack_codes[0]}"', '  if [[ "$result" == 0 && "${pack_codes[1]}" != 0 ]]; then result="${pack_codes[1]}"; fi', 'fi']
    shell += [f"{q(sys.executable)} {q(tool)} _display | tee -a {q(str(OUT / 'view.log'))}", 'exit "$result"']
    script = folder / "worker.sh"
    script.write_text("\n".join(shell) + "\n", encoding="utf-8")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash " + q(str(script))], check=True, timeout=15)
    (OUT / "view.log").touch(exist_ok=True)
    if not tmux_exists("gnr-v1-view"):
        viewer = f"tail -n 100 -F {q(str(OUT / 'view.log'))}; exec bash -i"
        subprocess.run(["tmux", "new-session", "-d", "-s", "gnr-v1-view", "bash -c " + q(viewer)], check=True, timeout=15)
    return {"status": "DISPATCHED", "session": session, "viewer": "tmux attach -t gnr-v1-view", "folder": str(folder)}


def worker(folder):
    record = read_json(folder / "dispatch.json")
    require(record and record["binding"] == binding(), "Dispatch identity stale")
    write_json(folder / "worker.json", process_identity(os.getpid()))
    job = record["job"]
    if job == "finish":
        from gnr_v1_eval import evaluate, offline_analysis
        for split in ("val", "test"):
            evaluate(split)
        return offline_analysis()
    current = ready(job)
    from gnr_v1_training import GNRTrainer, amp_assets
    plan = read_json(OUT / "prepare.json")
    amp_assets(Path(plan["main"]))
    args = dict(plan["args"])
    if job == "resume":
        args.update(model=str(RUN / "weights/last.pt"), resume=str(RUN / "weights/last.pt"))
    state = {"status": "RUNNING", "job": job, "binding": current, "process": process_identity(os.getpid()), "time": now()}
    write_json(OUT / "training.json", state)
    try:
        trainer = GNRTrainer(overrides=args, gnr_binding=current)
        trainer.train()
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="COMPLETE_EVAL_PENDING" if (OUT / "training_completed.json").exists() else "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=repr(error))
        raise
    finally:
        write_json(OUT / "training.json", state)
    return state


def package():
    """Pure offline: existing artifacts only. No torch, dataset, inference or recovery."""
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    plan = read_json(OUT / "prepare.json")
    tech = read_json(OUT / "preflight.json", {})
    diagnosis = read_json(OUT / "diagnose.json", {})
    completed = read_json(OUT / "training_completed.json", {})
    for name in ("prepare.json", "initialization.json", "nc1_loading.json", "structure.json", "data_snapshot.json", "recipe_audit.json",
                 "source_snapshot.tar.gz", "source_from_mother.patch", "training_setup.json", "training_completed.json", "threshold_analysis.json"):
        if not (OUT / name).is_file():
            missing.append(name)
    if (tech.get("status") != "TECHNICAL_PASS" or tech.get("effective_updates", 0) < 1
            or not any(s.get("updated") for s in tech.get("step_attempts", []))):
        missing.append("valid TECHNICAL_PASS")
    if diagnosis.get("status") != "REVIEW":
        missing.append("mother diagnostic evidence")
    if completed and (tech.get("binding") != completed.get("binding") or diagnosis.get("binding") != completed.get("binding")):
        missing.append("same training/preflight/diagnosis identity")
    if completed:
        for name in ("best", "last"):
            saved = completed.get("weights", {}).get(name, {})
            path = RUN / "weights" / (name + ".pt")
            if not path.is_file() or sha256(path) != saved.get("sha256"):
                missing.append(f"unchanged {name} file/hash")
    if plan and source_identity(clean=False)["functional_sha256"] != plan["code"]["functional_sha256"]:
        missing.append("unchanged functional source")
    if plan:
        if plan.get("development") or plan.get("recipe_audit", {}).get("actual_status") != "CHECKED":
            missing.append("formal server preparation / actual mother recipe audit")
        if not plan.get("artifacts"):
            missing.append("prepared artifact hashes")
        for name, expected_hash in plan.get("artifacts", {}).items():
            path = OUT / name
            if not path.is_file() or sha256(path) != expected_hash:
                missing.append("prepared artifact " + name)
        if completed.get("binding", {}).get("identity", {}).get("functional_sha256") != plan["code"]["functional_sha256"]:
            missing.append("same prepared/training source identity")
    for name in ("local_validation.json", "operations_validation.json", "mother_contract.json"):
        check = read_json(ROOT / "docs/gnr_v1" / name, {})
        if check.get("status") not in ("PASS", "CONTRACT_MATCH"):
            missing.append("reviewed check " + name)
    evaluations = {}
    for split in ("val", "test"):
        report = read_json(OUT / f"evaluation_{split}.json", {})
        evaluations[split] = report
        if report.get("status") != "COMPLETE" or report.get("exit_code") != 0 or not report.get("folder"):
            missing.append(f"{split} evaluation")
            continue
        if report.get("best_sha256") != completed.get("weights", {}).get("best", {}).get("sha256"):
            missing.append(f"{split} selected best identity")
        if report.get("identity", {}).get("binding") != completed.get("binding"):
            missing.append(f"{split} code/data/recipe binding")
        if (report.get("images"), report.get("ground_truths")) != COUNTS[split]:
            missing.append(f"{split} complete image/GT coverage")
        if report != read_json(Path(report["folder"]) / "metrics.json"):
            missing.append(f"{split} matching original metrics record")
        if not {"queries_gt.jsonl.gz", "raw_stats.npz", "curves.npz"} <= set(report.get("artifacts", {})):
            missing.append(f"{split} complete export")
        for relative, info in report.get("artifacts", {}).items():
            path = Path(report["folder"]) / relative
            if not path.is_file() or sha256(path) != info["sha256"]:
                missing.append(f"{split} artifact {relative}")
    analysis = read_json(OUT / "threshold_analysis.json", {})
    if (analysis.get("status") != "COMPLETE" or analysis.get("best_sha256") != completed.get("weights", {}).get("best", {}).get("sha256")
            or analysis.get("threshold") != evaluations.get("val", {}).get("native_max_f1_threshold")):
        missing.append("complete val-selected threshold analysis / same best")
    for split, report in evaluations.items():
        if analysis.get("source_sha256", {}).get(split) != report.get("artifacts", {}).get("queries_gt.jsonl.gz", {}).get("sha256"):
            missing.append(f"{split} offline analysis source binding")
    for name in ("args.yaml", "results.csv"):
        if not (RUN / name).is_file():
            missing.append("training/" + name)
    exits = list((OUT / "dispatches").glob("*/exit.json"))
    if not exits:
        missing.append("actual Python/tee exit records")
    stages = {}
    for path in sorted(exits):
        record = read_json(path, {})
        dispatch_record = read_json(path.parent / "dispatch.json", {})
        if dispatch_record.get("binding") == completed.get("binding") and dispatch_record.get("binding") is not None:
            if not all(type(record.get(k)) is int for k in ("python", "tee")) or record.get("binding") != dispatch_record["binding"]:
                missing.append("valid exit record " + str(path))
            stages[record.get("job")] = record
    training_exit = stages.get("resume") or stages.get("start")
    recovered = stages.get("finish", {}).get("python") == stages.get("finish", {}).get("tee") == 0
    training = read_json(OUT / "training.json", {})
    if (not training_exit or training_exit.get("tee") != 0 or
            (training_exit.get("python") != 0 and not (recovered and training.get("status") == "COMPLETE_EVAL_PENDING"))):
        missing.append("successful training exit or documented completed-training evaluation recovery")
    status_name = "INCOMPLETE" if missing else "COMPLETE"
    inventory = []
    files = [(p, "outputs/" + p.relative_to(OUT).as_posix()) for p in OUT.rglob("*") if p.is_file()
             and p.suffix not in (".pt", ".pth", ".tmp") and not p.name.startswith("GNR_v1_") and p.name not in ("run.lock", "lock_owner.json", "package.json")]
    files += [(p, "training/" + p.relative_to(RUN).as_posix()) for p in RUN.rglob("*") if p.is_file() and p.suffix not in (".pt", ".pth", ".jpg")]
    files += [(p, "docs/" + p.relative_to(ROOT / "docs/gnr_v1").as_posix()) for p in (ROOT / "docs/gnr_v1").rglob("*") if p.is_file()]
    destination = OUT / f"GNR_v1_{status_name}_{now()}.tar.gz"
    summary = {"status": status_name, "missing": missing, "code": source_identity(clean=False), "time": now(),
               "weights_included": False, "raw_images_included": False}
    with tarfile.open(destination, "w:gz") as archive:
        for path, relative in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            inventory.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        for name, value in (("PACKAGE_STATUS.json", summary),):
            data = json.dumps(value, ensure_ascii=False, indent=2).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            inventory.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        data = json.dumps(inventory, ensure_ascii=False, indent=2).encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, "r:gz") as archive:
        require(set(archive.getnames()) == {r["path"] for r in inventory} | {"MANIFEST.json"}, "Archive inventory mismatch")
        require(len(archive.getnames()) == len(inventory) + 1, "Duplicate archive names")
        for row in inventory:
            data = archive.extractfile(row["path"]).read()
            require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], f"Archive verification failed: {row['path']}")
    result = {"status": status_name, "path": str(destination), "sha256": sha256(destination), "payload_files": len(inventory),
              "missing": missing, "archive_readback": "PASS", "manifest_scope": "all payload files; MANIFEST itself is covered by external archive SHA256"}
    write_json(OUT / "package.json", result)
    return result


def display():
    rows = ["GNR v1 independent FP32 results (percent; delta in percentage points)",
            "split       P          R          F1         AP50       AP75       mAP50-95   delta_pp"]
    for split in ("val", "test"):
        r = read_json(OUT / f"evaluation_{split}.json")
        rows.append(split + "  " + "  ".join(f"{100*r[k]:.6f}" for k in ("precision", "recall", "F1", "AP50", "AP75", "mAP50_95")) + f"  {r['mother_delta_pp']:+.6f}" if r and r.get("status") == "COMPLETE" else split + " PENDING")
    complete = read_json(OUT / "training_completed.json", {})
    rows += ["best: " + json.dumps(complete.get("weights", {}).get("best"), ensure_ascii=False),
             "training state: " + str(complete.get("completion", "PENDING")),
             "package: " + json.dumps(read_json(OUT / "package.json"), ensure_ascii=False)]
    for path in sorted((OUT / "dispatches").glob("*/*exit.json")):
        rows.append("exit: " + json.dumps(read_json(path), ensure_ascii=False))
    text = "\n".join(rows)
    print(text, flush=True)
    return {"displayed": True}


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Server WT={SERVER_WT}\nRUN={SERVER_WT}/runs/c_series/{RUN.name}\nOUT={SERVER_WT}/outputs/gnr_v1\nPython={SERVER_PY}", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["prepare", "diagnose", "preflight", "status", "start", "resume", "val", "test", "finish", "pack", "_worker", "_preflight", "_diagnose", "_record-exit", "_display"])
    parser.add_argument("--main", default=SERVER_MAIN)
    parser.add_argument("--data", help="Original data YAML; default MAIN/configs/crack_autodl.yaml")
    parser.add_argument("--mother-args", help="Actual mother's args.yaml (exact documented model-path alias accepted)")
    parser.add_argument("--mother-best", help="Successful mother best.pt; fixed archived SHA256 is required")
    parser.add_argument("--seconds", type=int, default=900, help="diagnose/preflight total budget, at most 900s")
    parser.add_argument("--max-batches", type=int, default=16, help="preflight microbatch bound, at most 16")
    parser.add_argument("--val-images", type=int, default=64, help="diagnose val bound, at most 64")
    parser.add_argument("--train-batches", type=int, default=8, help="diagnose original augmented B16 bound, at most 8")
    parser.add_argument("--refresh-data", action="store_true", help="Explicitly rebuild invalid prepare snapshot before formal run")
    parser.add_argument("--development", action="store_true", help="Allow uncommitted local preparation/checks; cannot start formal training")
    parser.add_argument("--folder", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--python-code", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--tee-code", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--pack-exit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(0 < args.seconds <= 900 and 0 < args.max_batches <= 16, "Bounded budget exceeded")
    code = 0
    if args.action == "status":
        result = status()
    elif args.action == "_display":
        result = display()
    elif args.action == "_record-exit":
        require(args.folder and args.python_code is not None and args.tee_code is not None, "Exit arguments missing")
        dispatch_record = read_json(args.folder / "dispatch.json")
        result = {"job": dispatch_record["job"], "binding": dispatch_record["binding"],
                  "python": args.python_code, "tee": args.tee_code, "time": now()}
        write_json(args.folder / ("pack_exit.json" if args.pack_exit else "exit.json"), result)
    elif args.action in ("_preflight", "_diagnose"):
        request = read_json(args.folder / "request.json")
        if args.action == "_preflight":
            from gnr_v1_training import run_preflight
            result = run_preflight(request["binding"], args.folder, args.seconds, args.max_batches)
            code = 0 if result["status"] == "TECHNICAL_PASS" else 2
        else:
            from gnr_v1_diagnose import diagnose
            plan = read_json(OUT / "prepare.json")
            best = args.mother_best or str(Path(plan["main"]) / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt")
            result = diagnose(request["binding"], args.folder, best, args.seconds, args.val_images, args.train_batches)
            code = 0 if result["status"] in ("REVIEW", "ZERO_INTERVENTION") else 2
    else:
        with operation_lock(wait=args.action == "_worker"):
            if args.action == "prepare":
                result = prepare(args)
            elif args.action in ("diagnose", "preflight"):
                result, code = run_bounded(args.action, args)
            elif args.action in ("start", "resume", "finish"):
                result = dispatch(args.action)
            elif args.action == "_worker":
                result = worker(args.folder)
            elif args.action in ("val", "test"):
                from gnr_v1_eval import evaluate
                result = evaluate(args.action)
            else:
                result = package()
                code = 0 if result["status"] == "COMPLETE" else 2
                display()
    if args.action != "_display":
        visible = result
        if args.action in ("diagnose", "_diagnose", "preflight", "_preflight"):
            keys = ("status", "reason", "folder", "coverage_complete", "elapsed_seconds", "exit_code", "microbatches", "effective_updates", "scaler_skips", "peak_cuda_bytes")
            visible = {k: result[k] for k in keys if k in result}
            if "splits" in result:
                visible["splits"] = {k: {"images": len(v["images"]), "batches": len(v["batches"]),
                    "changed": v.get("summary", {}).get("changed"), "removed_negative_strength_fraction": v.get("summary", {}).get("removed_negative_strength_fraction")}
                    for k, v in result["splits"].items()}
        print(json.dumps(visible, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        traceback.print_exc()
        print(f"GNR action failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
