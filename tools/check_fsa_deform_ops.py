"""Lifecycle guard tests with explicit fixtures/mocks; never launch tmux or train."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from init_fsa_deform import ROOT, require, runtime, sha256, write_json
import train_fsa_deform as train
import fsa_deform_results as results


def rejected(action):
    try:
        action()
    except (RuntimeError, FileExistsError):
        return True
    raise AssertionError("Expected protective rejection")


def run(output):
    info = runtime()
    report = dict(runtime=info, scope="MOCKED lifecycle controls and real archive IO; no real dispatch/training", full_server_preflight="NOT_RUN")
    # Threshold-order stress case: explicitly preserve the archived C2 validator output.
    raw = torch.tensor([[[.5,.5,.1,.1,.0001],[.3,.3,.2,.2,.9],[.2,.7,.1,.1,.4]]])
    untouched = raw.clone()
    validator = object.__new__(results.RTDETRValidator)
    validator.args = SimpleNamespace(imgsz=640, conf=.001)
    expected = validator.postprocess(raw.clone())
    selected, full, affected = results.native_predictions(raw, 640, .001)
    require(all(torch.equal(expected[0][k], selected[0][k]) for k in expected[0]), "Changed C2 metric selection")
    require(torch.equal(raw, untouched) and len(full[0]["conf"]) == 3 and affected == 1, "Diagnostic export changed input/selection")
    report["native_C2_postprocess"] = "exact on unsorted threshold stress case; historical behavior retained"
    (ROOT / "outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fsa_deform_ops_", dir=ROOT / "outputs") as d:
        temporary = Path(d)
        main, root = temporary / "main", temporary / "worktree"
        root.mkdir(); (main / "configs").mkdir(parents=True)
        (root / "docs/fsa_deform").mkdir(parents=True)
        (root / "docs/fsa_deform/c2_args.yaml").write_bytes((ROOT / "docs/fsa_deform/c2_args.yaml").read_bytes())
        (root / "docs/fsa_deform/validation.json").write_text('{"fixture": true}')
        model_yaml = root / "ultralytics-main/ultralytics/cfg/models/rt-detr" / train.VARIANTS["fsa_deform"][0]
        model_yaml.parent.mkdir(parents=True)
        model_yaml.write_text("# MOCK MODEL YAML")
        data = main / "configs/crack_autodl.yaml"; data.write_text("names: [crack]\n")
        source = temporary / "source.pt"; source.write_bytes(b"TEST FIXTURE, NOT A CHECKPOINT")
        p = dict(name="fixture", run=main / "runs/c_series" / train.VARIANTS["fsa_deform"][2], launch=root / "outputs/fsa_deform", init=root / "init.pt", source=source,
                 c2_args=ROOT / "docs/fsa_deform/c2_args.yaml")
        calls = []
        def dispatch(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 1 if "has-session" in command and not any("new-session" in c for c in calls) else 0)
        def initialization(src, dest, variant):
            dest.write_bytes(b"MOCK INITIALIZATION; NEVER TRAIN")
            return {"source_sha256": sha256(src), "status": "MOCKED"}
        def snapshot(folder, c2, config):
            write_json(folder / "source_record.json", dict(data_sha256=sha256(config)))
        with ExitStack() as stack:
            for obj, name, value in ((train, "ROOT", root), (train, "MAIN", main), (train, "paths", lambda _: p),
                                     (train, "runtime", lambda: info), (train, "initialize", initialization),
                                     (train, "record_source", snapshot), (train, "ensure_amp_resources", lambda *a: None),
                                     (train, "verify_server_environment", lambda *a: None),
                                     (train, "check_det_dataset", lambda *a, **k: dict(nc=1, train=str(main), val=str(main), test=str(main)))):
                stack.enter_context(patch.object(obj, name, value))
            stack.enter_context(patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "rtdetr"}))
            stack.enter_context(patch.object(train.torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(train.shutil, "which", return_value="mock-tmux"))
            stack.enter_context(patch.object(train.subprocess, "check_output", return_value=""))
            stack.enter_context(patch.object(train.subprocess, "run", side_effect=dispatch))
            train.start_direct("fsa_deform")
            plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
            require(plan["args"]["batch"] == 16 and plan["args"]["epochs"] == 200 and plan["full_server_preflight"] == "NOT_RUN", "Direct recipe changed")
            require(len([c for c in calls if "new-session" in c]) == 1, "Dispatch count")
            require(not (p["launch"] / "audit.json").exists(), "Unexpected audit gate")
            report["direct_without_preflight_marker"] = "passed (mocked dispatch)"
            report["duplicate_reservation_rejected"] = rejected(lambda: train.start_direct("fsa_deform"))
            require(train.run_state(p) == "DISPATCHED", "Dispatch state")
            class FailingModel:
                def __init__(self, *a): pass
                def add_callback(self, *a): pass
                def train(self, **kwargs): raise train.torch.cuda.OutOfMemoryError("MOCK OOM")
            stack.enter_context(patch.object(train, "RTDETR", FailingModel))
            stack.enter_context(patch.object(train, "process_token", return_value="fixture-start-token"))
            try: train.worker("fsa_deform")
            except train.torch.cuda.OutOfMemoryError: pass
            else: raise AssertionError("OOM swallowed")
            require(train.run_state(p) == "FAILED", "OOM not failed")
            require(plan["args"]["batch"] == 16, "OOM changed batch")
            report["worker_oom_exit"] = "passed: mocked OOM propagates, exit_code=1, batch stays 16"
        # Explicit state fixtures, not actual training evidence.
        q = dict(launch=temporary / "state", run=temporary / "state_run")
        q["launch"].mkdir()
        states = [train.run_state(q)]
        write_json(q["launch"] / "process.json", dict(pid=123, process_token="abc"))
        with patch.object(train, "process_token", return_value="abc"): states.append(train.run_state(q))
        with patch.object(train, "process_token", return_value="different"): states.append(train.run_state(q))
        write_json(q["launch"] / "exit_code.json", dict(exit_code=0))
        (q["launch"] / "process_exit_code.txt").write_text("0\n")
        states.append(train.run_state(q))
        for name in ("weights/best.pt", "weights/last.pt", "results.csv"):
            path = q["run"] / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture")
        states.append(train.run_state(q))
        require(states == ["NOT_STARTED", "RUNNING", "FAILED", "FAILED", "SUCCESS"], "State classification")
        report["states"] = states
        # Partial pack is permitted and must clearly enumerate absent evidence.
        p["run"].mkdir(parents=True)
        (p["run"] / "test_fixture_over_20MiB.bin").write_bytes(os.urandom(21 * 1024 * 1024))
        with patch.object(results, "paths", return_value=p):
            report["incomplete_complete_pack_rejected"] = rejected(lambda: results.package(temporary / "must_not_exist.tar.gz"))
            archive = results.package(temporary / "fixture.tar.gz", allow_incomplete=True)
            verification = json.loads(Path(str(archive) + ".verification.json").read_text(encoding="utf-8"))
            require(not verification["evidence_complete"] and verification["missing_evidence"], "Incomplete pack claimed complete")
            results.verify_archive(archive)
            require(archive.stat().st_size > 20 * 1024 * 1024, "Large archive test did not exceed 20 MiB")
            report["partial_archive"] = dict(integrity="passed", missing_count=len(verification["missing_evidence"]),
                                              evidence_complete=False, bytes=archive.stat().st_size,
                                              overwrite_rejected=rejected(lambda: results.package(archive)))
        a = type("Trainer", (), {})(); a._oom_retries = 0
        train.disable_oom_retry(a); require(a._oom_retries == 3, "Native OOM guard not set")
    report["status"] = "passed"
    write_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
