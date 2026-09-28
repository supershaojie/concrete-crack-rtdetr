"""Fresh-process CLI and offline lifecycle fixtures; never starts training/test."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch

from dtr_v1_common import *
from experiment_runtime import operation_lock


def check(output):
    cli = []
    scripts = {"tools/dtr_v1.py": ["", "prepare", "preflight", "status", "start", "resume", "val", "test", "finish", "pack", "_worker", "_preflight"],
               "tools/dtr_v1_preflight.py": [""], "tools/check_dtr_v1.py": [""], "tools/check_dtr_v1_ops.py": [""]}
    for script, actions in scripts.items():
        for action in actions:
            command = [sys.executable, str(ROOT / script), *([action] if action else []), "--help"]
            proc = subprocess.run(command, cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=60)
            require(proc.returncode == 0 and "usage:" in proc.stdout.lower(), f"CLI help failed: {command}\n{proc.stderr}")
            cli.append(dict(command=command, status="PASS"))
            print("PASS fresh help:", script, action, flush=True)
    import dtr_v1 as lifecycle
    import dtr_v1_eval as ev
    # Read-only status has no paths into data preparation/evaluation.
    with patch.object(lifecycle, "binding", side_effect=AssertionError("status must not bind/scan")), \
         patch.object(lifecycle, "snapshot", side_effect=AssertionError("status must not scan")), \
         patch.object(ev, "evaluate", side_effect=AssertionError("status must not infer")):
        lifecycle.status()
    from ultralytics.models.rtdetr.dtr_val import postprocess
    import torch
    raw = torch.tensor([[[.5, .5, .2, .2, .0001], [.4, .4, .1, .1, .8], [.6, .6, .1, .1, .002]]])
    before = raw.clone()
    pred = postprocess(raw, 640, .001)[0]
    require(torch.equal(raw, before) and pred["conf"].tolist() == raw[0, [1, 2], 4].tolist(), "Sorted-confidence mask failure")
    with tempfile.TemporaryDirectory(prefix="dtr_ops_") as temp:
        folder = Path(temp)
        evidence, run = folder / "evidence", folder / "run"
        evidence.mkdir()
        export = evidence / "fixture_predictions_gt.jsonl.gz"
        rows = [
            dict(image_id="a", boxes=[[0., 0., 10., 10.]], scores=[.9], classes=[0], gt_boxes=[[0., 0., 10., 10.]], gt_classes=[0]),
            dict(image_id="b", boxes=[], scores=[], classes=[], gt_boxes=[], gt_classes=[]),
            dict(image_id="c", boxes=[[0., 0., 10., 10.]], scores=[.8], classes=[0], gt_boxes=[], gt_classes=[]),
        ]
        with gzip.open(export, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        metrics = ev.counts_at_threshold(export, .5)
        require((metrics["TP"], metrics["FP"], metrics["FN"]) == (1, 1, 0), "Offline counts/empty images failure")
        # Reuse validation succeeds for matching identities; damaged exports fail.
        identity = dict(example="same checkpoint/data/code/protocol", checkpoint_sha256="fixture")
        report = dict(status="PASS", identity=identity, split="val", images=3, gt_count=1, query_count=900,
                      artifacts=dict(predictions=file_info(export)))
        with patch.object(ev, "COUNTS", dict(val=(3, 1), test=(3, 1))):
            require(ev.verified_result(report, identity, "val"), "Lock should be reusable")
            require(not ev.verified_result(report, dict(example="changed"), "val"), "Changed identity must not reuse")
            broken = deepcopy(report); broken["artifacts"]["predictions"]["sha256"] = "0" * 64
            try:
                ev.verified_result(broken, identity, "val")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Tampered artifact accepted")
            from types import SimpleNamespace
            saved = evidence / "evaluations/fixture_val/metrics.json"
            report["report"] = str(saved)
            write_json(saved, report)
            write_json(evidence / "training_completed.json", dict(status="TRAINING_COMPLETED"))
            write_json(evidence / "training_identity.json", dict(id="fixture", binding=dict(data={})))
            best = run / "weights/best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"fixture identity only, not a checkpoint")
            with patch.object(ev, "OUT", evidence), patch.object(ev, "RUN", run), \
                 patch.object(ev, "data_config", return_value=folder / "unused.yaml"), \
                 patch.object(ev, "snapshot", return_value={}), patch.object(ev, "eval_identity", return_value=identity), \
                 patch.object(lifecycle, "active_workers", return_value=[]), patch.object(lifecycle, "has_tmux", return_value=False), \
                 patch("ultralytics.utils.patches.torch_load", return_value=dict(model=SimpleNamespace(dtr_identity="fixture"), epoch=3)), \
                 patch.object(ev, "DTRValidator", side_effect=AssertionError("Complete report must not infer again")):
                restored = ev.evaluate("val")
                require(restored["repaired_lock"], "Missing success lock should be repaired")
                require(ev.evaluate("val")["reused"], "Valid lock must reuse")
        with patch.object(ev, "OUT", evidence), patch.object(ev, "RUN", run), \
             patch.object(ev, "evaluate", side_effect=AssertionError("pack must not infer")), \
             patch.object(ev, "snapshot", side_effect=AssertionError("pack must not scan")):
            package = ev.package()
            require(package["completeness"] == "INCOMPLETE" and package["missing"], "Failure package disguised as complete")
            with tarfile.open(package["path"]) as archive:
                manifest = json.load(archive.extractfile("manifest.json"))
                require(all(row["path"] != "manifest.json" for row in manifest), "Self-hashing manifest")
                require("evidence/fixture_predictions_gt.jsonl.gz" in archive.getnames(), "Predictions absent from default archive")
        with operation_lock(evidence):
            try:
                with operation_lock(evidence):
                    raise AssertionError("Active operation lock was ignored")
            except RuntimeError:
                pass
        # Exercise dispatch writing/quoting without tmux, GPU or training.
        launch_out, launch_run = folder / "launch", folder / "launch_run"
        write_json(launch_out / "prepare.json", dict(args={"epochs": 200}))
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            if command[:2] == ["git", "archive"]:
                target = next(s.split("=", 1)[1] for s in command if s.startswith("--output="))
                Path(target).write_bytes(b"fixture source archive")
            return subprocess.CompletedProcess(command, 0)
        with patch.object(lifecycle, "OUT", launch_out), patch.object(lifecycle, "RUN", launch_run), \
             patch.object(lifecycle, "active_workers", return_value=[]), patch.object(lifecycle, "has_tmux", return_value=False), \
             patch.object(lifecycle, "server_check"), patch.object(lifecycle, "verify_preflight"), \
             patch.object(lifecycle, "binding", return_value=dict(code=dict(commit="a" * 40))), \
             patch.object(lifecycle.shutil, "which", return_value="tmux"), patch.object(lifecycle.subprocess, "run", side_effect=fake_run):
            dispatched = lifecycle.dispatch()
        shell_path = Path(dispatched["folder"]) / "worker.sh"
        shell = shell_path.read_text(encoding="utf-8")
        require("PIPESTATUS" in shell and "python_exit_code.txt" in shell and "pipefail" in shell, "Worker lost real pipeline exit capture")
        bash = "C:/Program Files/Git/bin/bash.exe" if sys.platform == "win32" else "bash"
        proc = subprocess.run([bash, "-n"], input=shell, capture_output=True, text=True, timeout=15)
        require(proc.returncode == 0, f"Generated worker shell invalid: {proc.stderr}")
    write_json(output, dict(status="PASS", CLI=cli, offline_counts=metrics, no_inference_status_pack=True,
                            incomplete_archive=True, checksum_tampering_rejected=True, active_lock_rejected=True,
                            corrected_sorted_conf_mask=True, missing_lock_repaired_without_inference=True,
                            dispatch_fixture_and_shell_syntax=True,
                            current_commit=git("rev-parse", "HEAD")))
    print("PASS offline lifecycle fixtures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "local_ops.json")
    args = parser.parse_args()
    check(args.output)
