"""Fast mocked control-flow regression; explicitly no detector training or final test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import train_sdb_p3 as lifecycle
from eval_sdb_p3 import postprocess


def exercise_worker(folder, mode, start_epoch, final_epoch, final_error=False):
    """Exercise real worker callbacks/state writes with a mocked learning loop."""
    p = dict(run=folder / "run", launch=folder / "launch")
    attempt = p["launch"] / "attempts/0001"
    attempt.mkdir(parents=True)
    checkpoint = folder / "checkpoint.pt"
    checkpoint.write_bytes(b"MOCK: no serialized detector")
    native_args = dict(model=str(checkpoint), resume=False, amp=True, save_dir=str(p["run"]))
    plan = dict(args=native_args, initialized=str(checkpoint))
    lifecycle.write_json(p["launch"] / "plan.json", plan)
    lifecycle.write_json(attempt / "dispatch.json", dict(mode=mode, checkpoint=str(checkpoint) if mode == "resume" else None,
                         checkpoint_sha256=lifecycle.sha256(checkpoint) if mode == "resume" else None))
    observed = {}

    class FakeModel:
        def __init__(self, checkpoint):
            self.callbacks = {}
        def add_callback(self, name, function):
            self.callbacks.setdefault(name, []).append(function)
        def train(self, trainer, **kwargs):
            # No epoch attribute exists here, matching the native start lifecycle.
            t = trainer.__new__(trainer)
            t.start_epoch, t.amp, t.save_dir = start_epoch, True, p["run"]
            effective = dict(native_args)
            if mode == "resume":
                effective.update(model=str(checkpoint), resume=str(checkpoint))
            t.args = SimpleNamespace(**effective)
            parameter = torch.nn.Parameter(torch.ones(1))
            t.model = SimpleNamespace(parameters=lambda: iter([parameter]),
                                      named_parameters=lambda: iter([("model.19.sdb.W_o.weight", parameter)]))
            t.optimizer = torch.optim.AdamW([parameter])
            assert not hasattr(t, "epoch")
            for callback in self.callbacks["on_train_start"]:
                callback(t)
            observed["recorded_start_epoch"] = json.loads((attempt / "state.json").read_text())["start_epoch"]
            for callback in self.callbacks["on_train_batch_start"]:
                callback(t)
            assert t._oom_retries == 3
            observed["no_oom_batch_retry"] = True
            t.epoch = final_epoch
            for callback in self.callbacks["on_fit_epoch_end"]:
                callback(t)
            t.final_eval()

    def fake_native_final_eval(trainer):
        if final_error:
            raise RuntimeError("mock final-validation failure")

    with patch.object(lifecycle, "paths", return_value=p), patch.object(lifecycle, "validate_plan"), \
         patch.object(lifecycle, "verify_model"), patch.object(lifecycle, "RTDETR", FakeModel), \
         patch.object(lifecycle.RTDETRTrainer, "final_eval", fake_native_final_eval), \
         patch.object(lifecycle.signal, "signal"):
        if final_error:
            try:
                lifecycle.worker(SimpleNamespace(variant=lifecycle.DEFAULT_VARIANT, attempt=attempt))
            except RuntimeError as error:
                assert str(error) == "mock final-validation failure"
            else:
                raise AssertionError("Worker swallowed final-validation exception")
        else:
            lifecycle.worker(SimpleNamespace(variant=lifecycle.DEFAULT_VARIANT, attempt=attempt))
    state = json.loads((attempt / "state.json").read_text())
    assert observed["recorded_start_epoch"] == start_epoch
    assert state["completed_epochs"] == final_epoch + 1
    expected = "FAILED" if final_error else ("COMPLETED_200" if final_epoch == 199 else "EARLY_STOPPED")
    assert state["status"] == expected
    assert state["final_validation"] == ("FAILED" if final_error else "PASSED")
    return dict(**observed, status=state["status"], completed_epochs=state["completed_epochs"], final_validation=state["final_validation"])


def check():
    results = {}
    with tempfile.TemporaryDirectory(prefix="sdb-lifecycle-") as temp:
        root = Path(temp)
        for name, arguments in {
            "fresh_start": ("start", 0, 199, False),
            "resumed_early_stop": ("resume", 17, 19, False),
            "final_validation_failure_after_200": ("start", 0, 199, True),
        }.items():
            results[name] = exercise_worker(root / name, *arguments)
        assert lifecycle.run_state(dict(launch=root / "absent", run=root / "absent_run"))["status"] == "NOT_STARTED"
        # Local/PENDING evidence must fail before any training dispatch/data access.
        report = root / "local_checks.json"
        lifecycle.write_json(report, dict(scope="local", status="PENDING"))
        try:
            lifecycle.require_preflight(report, root / "no_init.pt", root / "no_data.yaml")
        except RuntimeError as error:
            assert "PASSED server preflight required" in str(error)
        else:
            raise AssertionError("PENDING/local preflight was accepted")
    predictions = torch.tensor([[[.5, .5, .1, .1, .1], [.5, .5, .2, .2, .9], [.5, .5, .3, .3, .4]]])
    assert torch.equal(postprocess(predictions, 640, .2)[0]["conf"], torch.tensor([.9, .4]))
    return dict(status="PASSED", scope="mocked lifecycle control flow only; not server capacity or detector validation",
                checks=results, pending_local_gate_rejected=True, sorted_confidence_mask=True,
                formal_training="NOT_STARTED", test="NOT_RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        lifecycle.write_json(args.output, result)
    print(json.dumps(result, indent=2))
