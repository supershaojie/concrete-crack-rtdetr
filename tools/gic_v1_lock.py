"""Per-experiment kernel locking and process identity. No GPU exclusion."""
from contextlib import contextmanager
import os
import time
from gic_v1_common import OUT, write_json

def process_identity(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return {"pid": pid, "created": p.create_time(), "command": p.cmdline(), "cwd": p.cwd()}
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
        return None


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
                        raise RuntimeError("This GIC run already has an active writer") from error
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
                        raise RuntimeError("This GIC run already has an active writer") from error
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
