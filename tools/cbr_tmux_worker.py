"""Standard-library supervisor retaining C19 bootstrap failures and the actual child exit code."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback


def supervise(report_dir, command):
    report_dir = Path(report_dir)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": command, "exit_code": 1}
    try:
        with (report_dir / "bootstrap.log").open("x", encoding="utf-8") as log:
            try:
                result["exit_code"] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            except BaseException:
                traceback.print_exc(file=log)
                raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(result, indent=2) + "\n"
        (report_dir / "process_exit_code.json").write_text(payload, encoding="utf-8")
        if not (report_dir / "exit_code.json").exists():
            (report_dir / "exit_code.json").write_text(payload, encoding="utf-8")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(supervise(sys.argv[1], sys.argv[2:]))
