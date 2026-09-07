"""Standard-library supervisor retaining C20 bootstrap failures and the actual child exit code."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback
import threading
import time


def mirror_console(report_dir, done):
    """Stream the native trainer log into the retained tmux pane without a shell pipeline."""
    path = report_dir / "console.log"
    position = 0
    while True:
        if path.is_file():
            with path.open("rb") as file:
                file.seek(position)
                chunk = file.read()
                position = file.tell()
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        if done.is_set():
            return
        done.wait(0.2)


def supervise(report_dir, command):
    report_dir = Path(report_dir)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": command, "exit_code": 1}
    done = threading.Event()
    mirror = threading.Thread(target=mirror_console, args=(report_dir, done), daemon=True)
    mirror.start()
    try:
        with (report_dir / "bootstrap.log").open("x", encoding="utf-8") as log:
            try:
                result["exit_code"] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            except BaseException:
                traceback.print_exc(file=log)
                raise
    finally:
        done.set()
        mirror.join(timeout=3)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(result, indent=2) + "\n"
        (report_dir / "process_exit_code.json").write_text(payload, encoding="utf-8")
        if not (report_dir / "exit_code.json").exists():
            (report_dir / "exit_code.json").write_text(payload, encoding="utf-8")
        print(payload, flush=True)
        bootstrap = report_dir / "bootstrap.log"
        if bootstrap.is_file():
            with bootstrap.open("rb") as file:
                file.seek(max(0, bootstrap.stat().st_size - 16000))
                sys.stdout.buffer.write(file.read())
                sys.stdout.buffer.flush()
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(supervise(sys.argv[1], sys.argv[2:]))
