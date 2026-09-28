"""Small shared reliability utilities, extracted/adapted from ARG ef9cb7e.

Contains no experiment algorithm, model initialization or training state.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_info(path, hash_file=True):
    path = Path(path)
    return dict(path=str(path), exists=path.is_file(), bytes=path.stat().st_size if path.is_file() else None,
                sha256=sha256(path) if hash_file and path.is_file() else None)


def process_identity(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return dict(pid=pid, created=p.create_time(), command=p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


@contextmanager
def operation_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operation.lock"
    if path.exists():
        old = read_json(path)
        live = process_identity(old["pid"])
        require(not live or live["created"] != old["created"], f"Operation is active: {old}")
        path.unlink()  # verified stale lock, only in this experiment
    with path.open("x", encoding="utf-8") as stream:
        json.dump(process_identity(os.getpid()), stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def run_bounded(command, timeout, label, cwd):
    require(timeout > 0, f"No time remains for {label}")
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=cwd, start_new_session=os.name != "nt")
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            require(remaining > 0, f"{label}: time boundary exhausted")
            try:
                return proc.wait(timeout=min(20, remaining))
            except subprocess.TimeoutExpired:
                print(f"DTR stage={label} elapsed={time.monotonic()-started:.1f}s remaining={remaining:.1f}s", flush=True)
    finally:
        if proc.poll() is None:
            import psutil
            children = psutil.Process(proc.pid).children(recursive=True)
            for child in reversed(children):
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
                proc.kill()
                proc.wait(timeout=5)
            _, alive = psutil.wait_procs(children, timeout=1)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
