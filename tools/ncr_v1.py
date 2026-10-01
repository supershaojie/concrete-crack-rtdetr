"""NCR v1 server lifecycle. Every expensive action is explicit and bounded."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
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
import tempfile

from ncr_v1_common import (BRANCH, MAIN, NCR, PARENT, ROOT, SOURCE_SHA, digest,
                           git, identity, now, paths, read, recipe, require, sha, source_identity, write)


def initialize(source, destination, report):
    from init_c19_lif_v1 import initialize as parent_initialize
    write(report, parent_initialize(source, destination))


def verify_server(info):
    require(info["python"].startswith("3.10.") and info["torch"] == "2.1.2+cu121" and info["numpy"] == "1.26.4",
            "Server environment differs from Python3.10/torch2.1.2+cu121/NumPy1.26.4; no automatic upgrades")
    require(info["cuda_available"], "CUDA unavailable; server B16/640 smoke is NOT_RUN")


def smoke(initialized, dataset, output):
    """One real B16 batch, one FP32 step + one AMP step on separate temporary models."""
    import gc
    from types import SimpleNamespace
    import torch
    from c19_lif_v1_data import real_batch
    from init_c19_lif_v1 import build_training_model
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.models.utils.ncr import configure_ncr
    from ultralytics.utils.torch_utils import init_seeds
    report = dict(status="NOT_RUN", batch=16, imgsz=640, max_optimizer_steps=2, steps=[],
                  smoke_scaler_init_scale=128., scaler_source="parent tools/check_c19_lif_v1.py loss smoke",
                  formal_scaler="native trainer default; this bounded check does not alter formal training")
    write(output, report)
    if not torch.cuda.is_available():
        report["reason"] = "CUDA unavailable"
        write(output, report); return report
    try:
        batch, records = real_batch(dataset, size=640, count=16)
        report["samples"] = records
        for amp in (False, True):
            init_seeds(42, deterministic=True)
            public = RTDETR(str(initialized)).model
            model, loading = build_training_model(public.yaml, public, dict(nc=1, channels=3))
            del public
            model = configure_ncr(model, NCR).cuda().train()
            model.nc = 1; model.ncr_epoch = 19; model.ncr_collect_diagnostics = True
            args, _ = recipe()
            trainer = RTDETRTrainer.__new__(RTDETRTrainer); trainer.args = SimpleNamespace(**args)
            optimizer = trainer.build_optimizer(model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
            scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
            current = {k: v.cuda() for k,v in batch.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", enabled=amp):
                loss, items = model.loss(current)
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(items).all()), "Nonfinite smoke loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nonfinite = [name for name,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            if nonfinite:
                report["failed_step"] = dict(amp=amp, loss=float(loss.detach()), diagnostics=model.criterion.last_diagnostics,
                    nonfinite_parameters=nonfinite, optimizer_steps=0, scaler_scale=scaler.get_scale())
            require(not nonfinite, "Nonfinite smoke gradient")
            scaler.step(optimizer); scaler.update()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite smoke parameters")
            report["steps"].append(dict(amp=amp, loss=float(loss.detach()), items=items.tolist(), epoch=19,
                diagnostics=model.criterion.last_diagnostics, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                optimizer_steps=1, class_adaptation=loading["ALLOWED_CLASS_ADAPTATION"]))
            write(output, report)
            del model, optimizer, scaler, current, loss, items
            gc.collect(); torch.cuda.empty_cache()
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error),
                      remaining="NOT_RUN; no smaller batch/resolution, AMP change, or automatic retry")
        raise
    finally:
        write(output, report)
    return report


def amp_resources(p):
    # Parent native check_amp needs these; copy existing local resources only, never download.
    from ultralytics.utils import ASSETS
    rows = []
    for name, target, sources in (
        ("bus.jpg", ASSETS/"bus.jpg", [p["main"]/"ultralytics-main/ultralytics/assets/bus.jpg"]),
        ("yolo26n.pt", ROOT/"yolo26n.pt", [p["main"]/"yolo26n.pt", p["main"]/"weights/yolo26n.pt"]),
    ):
        if not target.is_file():
            source = next((s for s in sources if s.is_file()), None)
            require(source is not None, "Native AMP resource absent; provide existing local file: "+name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst: shutil.copyfileobj(src, dst)
        rows.append(dict(name=name, path=str(target), sha256=sha(target)))
    return rows


def preflight_worker(args):
    import torch
    torch.set_num_threads(4)
    p = paths(args.main); folder = p["output"]/args.evidence
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(status="FAILED", started=now(), scope="local" if args.local else "server", steps_max=2)
    try:
        ident = identity(args.main, args.data)
        report["identity"] = ident
        write(folder/"identity.json", ident)
        if not args.local:
            verify_server(ident["environment"])
            write(folder/"amp_resources.json", amp_resources(p))
        subprocess.run([sys.executable, str(ROOT/"tools/check_ncr_v1.py"), "--output", str(folder/"unit_tests.json")],
                       cwd=ROOT, check=True, timeout=180)
        subprocess.run([sys.executable, str(ROOT/"tools/check_ncr_v1_ops.py"), "--output", str(folder/"ops_tests.json")],
                       cwd=ROOT, check=True, timeout=180)
        # Parent Ultralytics checkpoint path sanitization strips apostrophes on Windows.
        # Keep its loader unchanged; isolate local temporary checkpoints under the main repo's ignored outputs.
        temporary_root = p["main"]/"outputs/ncr_v1_temporary_models" if args.local and "'" in str(folder) else folder
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ncr_preflight_", dir=temporary_root) as temp:
            init = Path(temp)/"init.pt"
            initialize(p["source"], init, folder/"initialization.json")
            result = smoke(init, Path(ident["data"]["config"]["path"]), folder/"smoke.json")
            require(result["status"] == "PASSED", "Real CUDA B16 smoke NOT_RUN; see smoke.json")
        report.update(status="PASSED", finished=now(), identity_sha256=digest(ident),
                      evidence_sha256={name:sha(folder/name) for name in ("unit_tests.json", "ops_tests.json", "smoke.json", "initialization.json")})
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write(folder/"preflight.json", report)


def preflight(args):
    p = paths(args.main); folder = p["output"]/args.evidence
    folder.mkdir(parents=True, exist_ok=True)
    previous = read(folder/"preflight.json", {})
    if previous.get("status") == "PASSED" and previous.get("scope") == ("local" if args.local else "server"):
        if previous.get("identity") == identity(args.main, args.data):
            print("Reused matching successful bounded preflight:", folder, flush=True); return
    if (folder/"preflight.json").exists():
        # Preserve every previous attempt; do not loop or replay silently.
        archive = folder.with_name(folder.name+"_prior_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        folder.rename(archive); folder.mkdir()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "_preflight", "--main", str(args.main), "--evidence", args.evidence]
    if args.local: command.append("--local")
    if args.data: command.extend(["--data", str(args.data)])
    log = folder/"console.log"
    print("Bounded preflight (900 seconds maximum):", log, flush=True)
    with log.open("x", encoding="utf-8") as stream:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            code = proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False, capture_output=True)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
            write(folder/"preflight.json", dict(status="FAILED", reason="TIMEOUT 900 seconds", finished=now()))
            code = 124
    print(log.read_text(encoding="utf-8", errors="replace")[-7000:])
    require(code == 0, f"Bounded preflight exit={code}; evidence retained at {folder}")


def require_preflight(p, ident):
    folder = p["output"]/"preflight"
    report = read(folder/"preflight.json", {})
    require(report.get("status") == "PASSED" and report.get("scope") == "server" and report.get("identity") == ident,
            "Matching successful server preflight missing; run preflight once on this delivery")
    for file in ("unit_tests.json", "ops_tests.json", "smoke.json"):
        require(read(folder/file, {}).get("status") == "PASSED", "Preflight evidence incomplete: "+file)
        require(sha(folder/file) == report.get("evidence_sha256", {}).get(file), "Preflight evidence checksum differs: "+file)
    return sha(folder/"preflight.json")


def source_archive(output):
    subprocess.run(["git", "archive", "--format=tar.gz", "--output="+str(output/"source.tar.gz"), "HEAD", "tools",
                    "ultralytics-main/ultralytics", "configs", "docs/ncr_v1", "docs/c19_lif_v1", ".gitattributes", ".gitignore"], cwd=ROOT, check=True)
    (output/"source_from_parent.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", PARENT, "HEAD"], cwd=ROOT))
    (output/"pip_freeze.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))


@contextlib.contextmanager
def own_lock(p, action):
    import fcntl  # server Linux only; lock is experiment-specific, never GPU-wide
    lock = p["run"].parent/(p["run"].name+"."+action+".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("This NCR action is already running: "+action)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def shell_worker(command, root, log, codefile, lock):
    """tee + Python failures both propagate; summary and dead pane remain visible."""
    return ("#!/usr/bin/env bash\nset -uo pipefail\ncd "+shlex.quote(str(root))+" || exit 90\n"
            "export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1\n"
            "finish_worker() { rc=$?; printf \"%s\\n\" \"$rc\" > "+shlex.quote(str(codefile))+"; "
            "printf \"\\nNCR process exit=%s\\nLog: %s\\nArtifacts: %s\\n\" \"$rc\" "+shlex.quote(str(log))+" "+shlex.quote(str(log.parent))+"; }\ntrap finish_worker EXIT\n"
            "exec 9>"+shlex.quote(str(lock))+" || exit 91\nflock -n 9 || exit 91\n"
            +shlex.join(command)+" 2>&1 | tee -a "+shlex.quote(str(log))+"\n"
            "codes=(\"${PIPESTATUS[@]}\")\nprintf 'Python exit=%s; tee exit=%s\\n' \"${codes[0]}\" \"${codes[1]}\"\n"
            "rc=${codes[0]}; if (( rc == 0 && codes[1] != 0 )); then rc=${codes[1]}; fi\nexit \"$rc\"\n")


def session_state(session):
    result = subprocess.run(["tmux", "list-panes", "-t", "="+session, "-F", "#{pane_dead} #{pane_dead_status}"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def dispatch(action, resume=False):
    p = paths(); require(os.name != "nt" and shutil.which("tmux"), "Server Linux + tmux required")
    require(ROOT != p["main"] and (ROOT/".git").is_file(), "Use the dedicated linked worktree")
    require(not git("status", "--porcelain", "--untracked-files=no"), "Commit tracked source changes before dispatch")
    require(not git("status", "--porcelain", "--", "tools", "ultralytics-main/ultralytics", "configs"), "Uncommitted/untracked executable source must be reviewed and committed before dispatch")
    require(git("branch", "--show-current") == BRANCH, "Wrong experiment branch")
    with own_lock(p, "dispatch"):
        ident = identity(); verify_server(ident["environment"])
        output = p["output"]; output.mkdir(parents=True, exist_ok=True)
        if action == "train":
            preflight_sha = require_preflight(p, ident)
            existing = read(output/"train_state.json", {})
            require(existing.get("status") != "SUCCEEDED", "Training already succeeded; use finish")
            if resume:
                require(p["run"].exists() and (p["run"]/"weights/last.pt").is_file(), "NCR last.pt missing for explicit resume")
                require(read(output/"manifest.json", {}).get("identity") == ident, "Resume run identity differs")
            else:
                require(not p["run"].exists() and not (output/"manifest.json").exists(), "Existing run protected; inspect status or explicitly resume")
                write(output/"manifest.json", dict(identity=ident, started=now(), command=sys.argv,
                      training_command=[sys.executable, "-u", str(ROOT/"tools/ncr_v1.py"), "_train"],
                      seed=42, ramp="0.25 * clip((actual zero-based trainer epoch - 4)/15, 0, 1)", preflight_sha256=preflight_sha))
                source_archive(output)
                manifest = read(output/"manifest.json")
                manifest["source_archive_sha256"] = sha(output/"source.tar.gz")
                manifest["source_patch_sha256"] = sha(output/"source_from_parent.patch")
                write(output/"manifest.json", manifest)
        else:
            require_training_success(p, ident)
        session = "ncr-v1-training" if action == "train" else "ncr-v1-finish"
        pane = session_state(session)
        if pane is not None:
            require(pane.startswith("1 "), "This experiment tmux pane is still running: "+session)
            owner = subprocess.check_output(["tmux", "show-options", "-t", "="+session, "-v", "@ncr_worktree"], text=True).strip()
            require(owner == str(ROOT), "Existing tmux session belongs to another worktree")
        codefile = output/(action+"_exit_code.txt")
        if codefile.exists(): codefile.rename(output/(action+"_exit_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".txt"))
        command = [sys.executable, "-u", str(ROOT/"tools/ncr_v1.py"), "_"+action]
        if resume: command.append("--resume")
        worker = output/(action+"_worker.sh")
        worker.write_text(shell_worker(command, ROOT, output/(action+".log"), codefile,
                          p["run"].parent/(p["run"].name+"."+action+".lock")), encoding="utf-8", newline="\n")
        if pane is None:
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT)], check=True)
            subprocess.run(["tmux", "set-option", "-t", "="+session, "@ncr_worktree", str(ROOT)], check=True)
        subprocess.run(["tmux", "set-option", "-w", "-t", session+":0", "remain-on-exit", "on"], check=True)
        write(output/(action+"_state.json"), dict(status="DISPATCHED", session=session, command=command, started=now()))
        # Only a new session's placeholder shell is replaced. Existing panes must already be dead.
        respawn = ["tmux", "respawn-pane"] + (["-k"] if pane is None else [])
        subprocess.run(respawn+["-t", session+":0.0", "bash "+shlex.quote(str(worker))], check=True)
        print("Dispatched:", session, "log:", output/(action+".log"), flush=True)


def train(resume=False):
    import torch
    from ultralytics import RTDETR
    from ultralytics.utils import YAML
    from ncr_v1_training import NCRTrainer
    p = paths(); output = p["output"]
    frozen = read(output/"manifest.json"); current = identity()
    require(current == frozen["identity"], "Manifest identity changed after dispatch")
    require_preflight(p, current)
    amp_resources(p)
    config, differences = recipe()
    YAML.save(output/"resolved_train_args.yaml", config)
    write(output/"recipe_diff.json", dict(changes=differences, ncr=NCR, other_changes=[]))
    shutil.copyfile(p["data"], output/"data.yaml")
    if resume:
        checkpoint = p["run"]/"weights/last.pt"
        # Native resume owns optimizer/scaler/scheduler/epoch restoration.
        require(read(output/"manifest.json")["identity"] == current, "Resume provenance differs")
        model = RTDETR(str(checkpoint))
        model.train(trainer=NCRTrainer, resume=True)
    else:
        require(not p["run"].exists(), "Training output already exists")
        require(not p["init"].exists(), "Existing formal init protected; inspect incomplete launch before recovery")
        initialize(p["source"], p["init"], output/"initialization.json")
        model = RTDETR(str(p["init"]))
        model.train(trainer=NCRTrainer, **config)
    require((p["run"]/"weights/best.pt").is_file(), "Training returned without selected best.pt")
    write(output/"training_complete.json", dict(status="SUCCEEDED", best=str(p["run"]/"weights/best.pt"),
          best_sha256=sha(p["run"]/"weights/best.pt"), identity_sha256=digest(current), finished=now()))


def require_training_success(p, ident=None):
    require(read(p["output"]/"train_state.json", {}).get("status") == "SUCCEEDED", "Training has not successfully ended")
    require((p["output"]/"train_exit_code.txt").is_file() and (p["output"]/"train_exit_code.txt").read_text().strip() == "0",
            "Training shell/Python exit evidence is missing or nonzero")
    completed = read(p["output"]/"training_complete.json", {})
    require(completed.get("status") == "SUCCEEDED" and completed.get("best_sha256") == sha(p["run"]/"weights/best.pt"),
            "Frozen best checkpoint identity differs")
    if ident is not None:
        require(read(p["output"]/"manifest.json", {}).get("identity") == ident, "Runtime/code/config/data differs from training manifest")
    return completed


def finish():
    from ncr_v1_results import analyze, evaluate, metrics_page
    p = paths(); ident = identity()
    require_training_success(p, ident)
    for split in ("val", "test"):
        folder = p["output"]/("evaluation_"+split)
        report = evaluate(p["run"]/"weights/best.pt", p["output"]/"data.yaml", split, folder, ident)
        analyze(folder)
        metrics_page(p["output"])
    package = pack(p)
    require(package["status"] == "COMPLETE", "Finish produced PARTIAL evidence; see package.json missing list")


def status(p):
    print("NCR branch:", git("branch", "--show-current"), "HEAD:", git("rev-parse", "HEAD"))
    print("run:", p["run"], "evidence:", p["output"])
    for action, session in (("train", "ncr-v1-training"), ("finish", "ncr-v1-finish")):
        row = read(p["output"]/(action+"_state.json"), {})
        code = p["output"]/(action+"_exit_code.txt")
        pane = session_state(session) if shutil.which("tmux") else None
        if code.is_file():
            state = "SUCCEEDED" if code.read_text().strip() == "0" and row.get("status") == "SUCCEEDED" else "FAILED"
        elif row:
            state = "RUNNING" if pane is not None and pane.startswith("0") else "INTERRUPTED_OR_FAILED"
        else: state = "NOT_STARTED"
        print(action, state, "python:", row, "shell exit:", code.read_text().strip() if code.exists() else None, "pane:", pane)
        log = p["output"]/(action+".log")
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size-3000)); print(stream.read().decode("utf-8", errors="replace"))
    for file in ("preflight/preflight.json", "package.json"):
        value = read(p["output"]/file, {})
        print(file, {k: value[k] for k in ("status", "error", "archive", "missing") if k in value})
    for split in ("val", "test"):
        value = read(p["output"]/("evaluation_"+split)/"metrics.json", {})
        print(split, value.get("status", "NOT_RUN"), value.get("percent", {}), "exit:", value.get("exit_code"))
    for file in (p["run"]/"weights/best.pt", p["run"]/"results.csv", p["output"]/"metrics.md"):
        print("artifact:", file, "bytes:", file.stat().st_size if file.is_file() else "MISSING")


def pack(p):
    """Offline archive only. COMPLETE requires consistent existing evidence, never inference."""
    output = p["output"]; output.mkdir(parents=True, exist_ok=True)
    missing = []
    required = ["manifest.json", "initialization.json", "nc1_loading.json", "training_setup.json", "actual_train_args.yaml",
                "resolved_train_args.yaml", "recipe_diff.json", "data.yaml", "training_diagnostics.jsonl", "source.tar.gz",
                "source_from_parent.patch", "pip_freeze.txt", "train.log", "train_exit_code.txt", "training_complete.json",
                "preflight/preflight.json", "preflight/unit_tests.json", "preflight/ops_tests.json", "preflight/smoke.json", "metrics.md"]
    missing += [name for name in required if not (output/name).is_file()]
    missing += ["training/"+name for name in ("weights/best.pt", "args.yaml", "results.csv", "results.png") if not (p["run"]/name).is_file()]
    try:
        require_training_success(p)
        frozen = read(output/"manifest.json")["identity"]
        require(source_identity() == frozen["source"], "Current source differs from recorded training")
        require_preflight(p, frozen)
        initialization = read(output/"initialization.json", {})
        require(initialization.get("status") == "passed" and initialization.get("source_sha256") == SOURCE_SHA, "Initialization evidence differs")
        require(read(output/"training_setup.json", {}).get("status") == "PASSED", "Training setup audit missing/failed")
        import yaml
        actual = yaml.safe_load((output/"actual_train_args.yaml").read_text(encoding="utf-8"))
        allowed_resume = {"model", "resume"} if actual.get("resume") else set()
        require(all(actual.get(k) == v for k,v in frozen["training_args"].items() if k not in allowed_resume), "Actual train args differ from manifest")
        manifest = read(output/"manifest.json")
        require(sha(output/"source.tar.gz") == manifest.get("source_archive_sha256") and
                sha(output/"source_from_parent.patch") == manifest.get("source_patch_sha256"), "Source archive/patch checksum differs")
        from ncr_v1_results import eval_identity, verify_eval
        for split in ("val", "test"):
            folder = output/("evaluation_"+split)
            expected = eval_identity(p["run"]/"weights/best.pt", output/"data.yaml", split, folder, frozen)
            verify_eval(read(folder/"metrics.json", {}), expected, folder)
            analysis = read(folder/"analysis.json", {})
            require(analysis.get("status") == "PASSED" and analysis.get("export_sha256") == sha(folder/"queries_gt.jsonl.gz"), "Missing/stale offline analysis")
    except (RuntimeError, OSError, KeyError, TypeError) as error:
        missing.append(str(error))
    files = []
    for base, prefix in ((output, "evidence"), (p["run"], "training")):
        if base.exists():
            for file in sorted(base.rglob("*")):
                rel = file.relative_to(base)
                if not file.is_file() or file.is_symlink() or "packages" in rel.parts or "local" in rel.parts or file.name == "package.json": continue
                if prefix == "evidence" and file.name.startswith("finish"): continue  # packing must not archive its own growing log/state
                # No other experiment/data/env; failed preflight attempts retained on disk, not duplicated in COMPLETE.
                if any("_prior_" in part or "_failed_" in part for part in rel.parts): continue
                if prefix == "training" and file.name == "last.pt": continue
                files.append((file, prefix+"/"+rel.as_posix()))
    inventory = [dict(path=name, bytes=file.stat().st_size, sha256=sha(file)) for file,name in files]
    state = "PARTIAL" if missing else "COMPLETE"
    content_id = digest(dict(inventory=inventory, status=state, missing=missing))
    previous = read(output/"package.json", {})
    if previous.get("content_id") == content_id and Path(previous.get("archive", "")).is_file() and sha(previous["archive"]) == previous.get("sha256"):
        print("Reused", state, previous["archive"]); return previous
    packages = output/"packages"; packages.mkdir(exist_ok=True)
    destination = packages/("NCR_v1_"+state+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".tar.gz")
    manifest = dict(status=state, missing=missing, files=inventory, content_id=content_id, created=now())
    with tarfile.open(destination, "x:gz") as archive:
        for file, name in files: archive.add(file, arcname=name, recursive=False)
        payload = (json.dumps(manifest, indent=2, ensure_ascii=False)+"\n").encode()
        item = tarfile.TarInfo("MANIFEST.json"); item.size=len(payload); archive.addfile(item, io.BytesIO(payload))
    # Stream readback checks every member, not merely tar headers.
    with tarfile.open(destination, "r:gz") as archive:
        require(len(archive.getmembers()) == len(inventory)+1, "Archive member count differs")
        import hashlib
        for row in inventory:
            check = hashlib.sha256()
            with archive.extractfile(row["path"]) as stream:
                for block in iter(lambda: stream.read(1024*1024), b""): check.update(block)
            require(check.hexdigest() == row["sha256"], "Archive changed during packing: "+row["path"])
    checksum = sha(destination)
    Path(str(destination)+".sha256").write_text(checksum+"  "+destination.name+"\n", encoding="utf-8")
    write(Path(str(destination)+".manifest.json"), manifest)
    result = dict(status=state, missing=missing, archive=str(destination), sha256=checksum, content_id=content_id)
    write(output/"package.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "start", "status", "attach", "finish", "pack", "_preflight", "_train", "_finish", "config"))
    parser.add_argument("--main", type=Path, default=MAIN)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--evidence", default="preflight")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); os.environ["YOLO_AUTOINSTALL"] = "false"
    require(Path(args.evidence).name == args.evidence, "Evidence name must be one directory component")
    p = paths(args.main)
    if args.command in ("_train", "_finish"):
        action = args.command[1:]; code = 1
        try:
            write(p["output"]/(action+"_state.json"), dict(status="RUNNING", pid=os.getpid(), started=now()))
            train(args.resume) if action == "train" else finish()
            code = 0
        finally:
            write(p["output"]/(action+"_state.json"), dict(status="SUCCEEDED" if code == 0 else "FAILED", exit_code=code, finished=now()))
        return
    if args.command == "preflight": preflight(args)
    elif args.command == "_preflight": preflight_worker(args)
    elif args.command == "config":
        config, changes = recipe(args.main); print(json.dumps(dict(args=config, changes=changes, ncr=NCR), indent=2))
    elif args.command == "start": dispatch("train", args.resume)
    elif args.command == "finish": dispatch("finish")
    elif args.command == "status": status(p)
    elif args.command == "attach": subprocess.run(["tmux", "attach-session", "-t", "=ncr-v1-training"], check=True)
    elif args.command == "pack": pack(p)


if __name__ == "__main__":
    main()
