"""Run-scoped locking, verifiable workers and retained tmux panes. No GPU polling."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from cea_v1_common import ROOT, OUT, RUN, require, read_json, write_json


def process_identity(pid):
    """Linux PID, kernel start time, executable argv and cwd, not a pid-file guess."""
    try:
        proc = Path("/proc") / str(pid)
        raw = (proc / "stat").read_text()
        stat = raw[raw.rfind(")") + 2:].split()
        if stat[0] == "Z": return None
        return dict(pid=int(pid), start_ticks=stat[19], cwd=str((proc / "cwd").resolve()),
                    argv=(proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0"))
    except (OSError, ValueError):
        return None


def live_record(row):
    proc = process_identity(row.get("pid", -1))
    return bool(proc and proc.get("start_ticks") == row.get("start_ticks") and
                proc["cwd"] == str(ROOT) and proc["argv"] == row.get("argv") and
                row.get("run_id") in proc["argv"] and row.get("run") == str(RUN))


def active_workers():
    return [row for p in (OUT / "workers").glob("*/process.json")
            for row in [read_json(p)] if live_record(row)]


@contextmanager
def run_lock():
    require(os.name == "posix", "Server process control requires Linux; local formula checks remain available")
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.lock").open("a+") as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("This CEA run already has an active writer (run-scoped lock)")
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def stage_directory(action):
    run_id = uuid.uuid4().hex
    folder = OUT / "workers" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{action}_{run_id[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    return run_id, folder


def worker_command(action, run_id, folder, extra=()):
    return [sys.executable, "-u", str(ROOT / "tools/cea_v1.py"), action, "--worker", "--run-id", run_id,
            "--stage-dir", str(folder), *map(str, extra)]


def mark_worker(action, run_id, folder):
    row = process_identity(os.getpid())
    require(row is not None, "Cannot verify this worker process")
    row.update(action=action, run_id=run_id, run=str(RUN), started_utc=datetime.now(timezone.utc).isoformat())
    write_json(folder / "process.json", row)


def bounded(action, seconds=900, extra=()):
    require(0 < seconds <= 900, "Bounded action budget must be <=900 seconds")
    with run_lock():
        require(not active_workers(), "This CEA run has an active worker")
        run_id, folder = stage_directory(action)
        command = worker_command(action, run_id, folder, extra)
        write_json(folder / "launch.json", dict(action=action, run_id=run_id, command=command, state="DISPATCHED", time=time.time()))
    print(f"{action}: at most {seconds}s; log {folder / 'console.log'}", flush=True)
    started = time.monotonic()
    with (folder / "console.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            # Only the process group this invocation created, including its own loaders.
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
            code = 124
            report = dict(status="PENDING", reason="hard wall-clock budget exhausted", seconds=time.monotonic() - started,
                          log=str(folder / "console.log"), worker_returncode=child.returncode)
            write_json(folder / "report.json", report)
            write_json(OUT / f"{action}.json", report)
    write_json(folder / "exit_codes.json", dict(python=code, tee=None, seconds=time.monotonic() - started))
    path = folder / "report.json"
    if path.exists(): print(json.dumps(read_json(path), ensure_ascii=False, indent=2))
    else: print(f"Worker exited {code}; see {folder / 'console.log'}")
    return code


def launch(action, extra=()):
    require(os.name == "posix", "tmux server launch requires Linux")
    session_base = "cea-v1-training" if action in ("start", "resume") else f"cea-v1-{action}"
    session = session_base if action != "resume" else session_base + "-resume-" + datetime.now().strftime("%m%d-%H%M%S")
    with run_lock():
        require(not active_workers(), "This CEA run has an active worker")
        require(subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0,
                f"tmux session already exists, preserved: {session}. See status/attach; do not relaunch blindly")
        run_id, folder = stage_directory(action)
        command = worker_command(action, run_id, folder, extra)
        worker = folder / "worker.sh"
        script = "#!/usr/bin/env bash\nset +e\nset -o pipefail\ncd " + shlex.quote(str(ROOT)) + " || exit 70\n"
        script += shlex.join(command) + " 2>&1 | tee " + shlex.quote(str(folder / "console.log")) + "\n"
        script += 'codes=("${PIPESTATUS[@]}")\n'
        script += "printf '{\"python\":%s,\"tee\":%s}\\n' \"${codes[0]}\" \"${codes[1]}\" > " + shlex.quote(str(folder / "exit_codes.json")) + "\n"
        script += "printf '\\nCEA worker exited: Python=%s tee=%s. Pane retained.\\n' \"${codes[0]}\" \"${codes[1]}\"\n"
        script += 'if (( codes[0] != 0 )); then exit "${codes[0]}"; fi\nexit "${codes[1]}"\n'
        worker.write_text(script, encoding="utf-8")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT)], check=True)
        subprocess.run(["tmux", "set-option", "-w", "-t", session + ":0", "remain-on-exit", "on"], check=True)
        row = dict(action=action, run_id=run_id, session=session, folder=str(folder), worker=str(worker),
                   command=command, run=str(RUN), state="DISPATCHED", time=time.time())
        write_json(folder / "launch.json", row)
        write_json(OUT / "latest_launch.json", row)
    # Release this run's launch lock before the worker acquires the same lock.
    subprocess.run(["tmux", "send-keys", "-t", session + ":0", "exec bash " + shlex.quote(str(worker)), "C-m"], check=True)
    print(f"DISPATCHED {action}: tmux attach -t {shlex.quote(session)}\nLog: {folder / 'console.log'}")
    return row


def status():
    alive = active_workers()
    print(json.dumps(dict(active_workers=alive, state="RUNNING" if alive else "NO_ACTIVE_WORKER",
                          note="A retained tmux pane is not a live training process", run=str(RUN), out=str(OUT)), indent=2))
    for name in ("prepare", "preflight", "diagnose", "latest_launch", "evaluation_val", "evaluation_test"):
        p = OUT / f"{name}.json"
        if p.is_file():
            row = read_json(p)
            if name == "latest_launch":
                stage = Path(row["folder"])
                proc = stage / "process.json"
                exits = stage / "exit_codes.json"
                row["state"] = ("RUNNING" if proc.exists() and live_record(read_json(proc)) else
                                "EXITED" if exits.exists() else
                                "DISPATCHED" if time.time() - row.get("time", 0) < 60 else "NO_ACTIVE_WORKER")
            print(name + ": " + json.dumps({k: row[k] for k in ("status", "state", "mechanism", "reason", "session", "folder", "metrics", "seconds") if k in row}, ensure_ascii=False))
        else: print(name + ": PENDING")
    for name in ("best_selection.json", "latest_epoch.json", "training_complete.json"):
        p = RUN / name
        if p.is_file():
            row = read_json(p); row.pop("cea_identity", None)
            print(name + ": " + json.dumps(row, ensure_ascii=False))
    for p in sorted((OUT / "workers").glob("*/exit_codes.json"))[-5:]: print(str(p) + ": " + p.read_text())
    p = RUN / "results.csv"
    if p.exists():
        rows = p.read_text().splitlines(); print("Recent training metrics:\n" + "\n".join(rows[:1] + rows[-3:]))


def attach():
    row = read_json(OUT / "latest_launch.json")
    return subprocess.call(["tmux", "attach", "-t", row["session"]])
