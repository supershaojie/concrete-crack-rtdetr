"""Synthetic lock recovery regression; no model, GPU, dataset or training lifecycle."""
import argparse
import json
from pathlib import Path
import socket
import tempfile

from bfr_p4_lifecycle import reserve_lock, require, write_json


def run():
    variant, commit = "SYNTHETIC_bfr", "0" * 40
    cases = []
    with tempfile.TemporaryDirectory(prefix="bfr-p4-synthetic-lock-") as temporary:
        root = Path(temporary).resolve()
        lock = root / "synthetic-run.bfr-active.lock"
        owner = {"pid": 123456789, "hostname": socket.gethostname(), "variant": variant, "commit": commit}
        lock.mkdir()
        write_json(lock / "owner.json", owner)
        def rejected(label, resume, predicate, replacement=None):
            if replacement is not None:
                write_json(lock / "owner.json", replacement)
            before = (lock / "owner.json").read_bytes()
            try:
                reserve_lock(lock, variant, commit, resume=resume, absent_check=predicate)
            except RuntimeError:
                require(lock.is_dir() and (lock / "owner.json").read_bytes() == before, "Rejected owner was modified")
                cases.append(label)
            else:
                raise AssertionError("Unexpected lock recovery: " + label)
        rejected("fresh_start_cannot_recover", False, lambda pid: True)
        rejected("live_or_reused_pid_protected", True, lambda pid: False)
        rejected("foreign_host_protected", True, lambda pid: True, dict(owner, hostname="SYNTHETIC_other_host"))
        rejected("wrong_variant_protected", True, lambda pid: True, dict(owner, variant="SYNTHETIC_other_variant"))
        rejected("wrong_commit_protected", True, lambda pid: True, dict(owner, commit="f" * 40))
        write_json(lock / "owner.json", owner)
        original = (lock / "owner.json").read_bytes()
        reserve_lock(lock, variant, commit, resume=True, absent_check=lambda pid: pid == owner["pid"])
        archived = list(root.glob("synthetic-run.bfr-active.lock.stale.*"))
        require(len(archived) == 1 and (archived[0] / "owner.json").read_bytes() == original,
                "Stale owner evidence was not preserved exactly")
        new = json.loads((lock / "owner.json").read_text())
        require(new["hostname"] == socket.gethostname() and new["pid"] != owner["pid"], "New ownership not recorded")
        cases.append("explicit_resume_archives_confirmed_absent_owner")
        # TemporaryDirectory owns only this freshly created, resolved synthetic subtree.
        require(root.parent == Path(tempfile.gettempdir()).resolve(), "Synthetic cleanup path escaped temp root")
    return {"status": "PASSED", "scope": "Synthetic temporary lock metadata only; process absence mocked", "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
