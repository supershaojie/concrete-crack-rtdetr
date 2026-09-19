"""Fault-injection checks for BLC evidence gates; never calls train/val/test."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from blc_common import ROOT, require, write_json, stamp
import blc_server


def check():
    context = dict(variant="cbr_lif_blc_v1", code={"sha256": "current"}, init_sha256="init", source_sha256="source",
                   data={"labels": "current"}, recipe={"batch": 16},
                   runtime=dict(python="3.10", torch="2.1.2+cu121", cuda="12.1", gpu="4090", ultralytics="worktree"))
    valid = dict(status="PASSED", context=context, capacity=dict(status="PASSED", batch=16, imgsz=640, AMP=True, effective_updates=2),
                 lifecycle=dict(status="PASSED"), native_half_ema_epoch_val=dict(status="PASSED"), native_resume=dict(status="PASSED"))
    outcomes = {}
    folder = ROOT/"outputs/blc_v1/temporary"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gate-faults-", dir=folder) as tmp:
        report_path = Path(tmp)/"preflight-01/report.json"
        report_path.parent.mkdir()
        cases = {"valid": valid}
        for status in ("FAILED", "PENDING"):
            cases[status] = {**valid, "status": status}
        for key in ("variant", "code", "init_sha256", "source_sha256", "data", "recipe"):
            row = deepcopy(valid); row["context"][key] = "changed"; cases["stale_"+key] = row
        for key in ("python", "torch", "cuda", "gpu", "ultralytics"):
            row = deepcopy(valid); row["context"]["runtime"][key] = "changed"; cases["environment_"+key] = row
        for key, value in (("batch", 8), ("imgsz", 320), ("AMP", False), ("effective_updates", 1)):
            row = deepcopy(valid); row["capacity"][key] = value; cases["capacity_"+key] = row
        for key in ("lifecycle", "native_half_ema_epoch_val", "native_resume"):
            row = deepcopy(valid); row[key]["status"] = "PENDING"; cases[key] = row
        with patch.object(blc_server, "evidence_context", return_value=context), \
             patch.object(blc_server, "paths", return_value={"evidence": Path(tmp)}):
            for name, row in cases.items():
                # This exclusively owned fixture is mutable; actual reports are exclusive-write.
                report_path.write_text(json.dumps(row), encoding="utf-8")
                try:
                    blc_server.require_evidence("cbr_lif_blc_v1", "preflight")
                    accepted = True
                except RuntimeError:
                    accepted = False
                require(accepted == (name == "valid"), "Evidence gate regression: "+name)
                outcomes[name] = "accepted" if accepted else "rejected"
    report = dict(status="PASSED", scope="isolated evidence-gate fixtures, no training", cases=outcomes)
    path = ROOT/"outputs/blc_v1"/("ops-gates-"+stamp()+".json")
    write_json(path, report)
    print(path, "PASSED")


if __name__ == "__main__":
    check()
