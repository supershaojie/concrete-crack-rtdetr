"""Tiny CPU fixtures only: no real RDM, dataset, checkpoint, training or server validation."""
import ast
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import rdm_half_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[1]
OLD = "f01878ebd944c793257f058031fe13d71ee2fc46"


class Overflow(nn.Module):
    def forward(self, x):
        return x * 65504


class Fixture(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.tensor([70000. if mode == "conversion" else 1.]))
        self.register_buffer("running", torch.tensor([70000. if mode == "conversion" else 1.]))
        self.register_buffer("counter", torch.tensor([2**62], dtype=torch.int64))
        self.first = nn.Identity()
        self.overflow = Overflow() if mode == "boundary" else nn.Identity()

    def forward(self, x):
        x = self.overflow(self.first(x))
        y = x.clone()
        raw = [x.clone() for _ in range(4)] + [None]
        if self.mode == "y":
            y.flatten()[0] = float("nan")
        if self.mode == "raw":
            raw[3].flatten()[0] = float("inf")
        if self.mode == "native_error":
            raise RuntimeError("fixture original native exception")
        return y, tuple(raw)


class Validator:
    def __init__(self):
        self.training = True
        self.args = SimpleNamespace(half=True)
        self.callbacks = defaultdict(list)

    def add_callback(self, event, function):
        self.callbacks[event].append(function)


def finite(value):
    return all(bool(torch.isfinite(t).all()) for _, t in diagnostic.tensors(value, "out"))


def execute(mode, broken_diagnostic=False, preexisting=False):
    model, validator, row = Fixture(mode), Validator(), {}
    if preexisting:
        model.running.fill_(float("inf"))
    expected = deepcopy(model).half().eval()
    rng = torch.random.get_rng_state().clone()
    image = torch.full((2, 3, 2, 2), 2., dtype=torch.float16)
    image_before = image.clone()
    native_error = None
    observer = diagnostic.HalfEMADiagnostic(model, validator, row, amp=True, ema_updates=17)
    def original_output_check(m, a, out):
        if not finite(out):
            raise RuntimeError("Nonfinite native half EMA prediction")
    try:
        with observer:
            handle = model.register_forward_hook(original_output_check)
            try:
                # Mimic ONLY the native conversion/callback/forward ordering with a tiny CPU fixture.
                model.half().eval()
                for callback in validator.callbacks["on_val_start"]:
                    callback(validator)
                if broken_diagnostic:
                    with patch.object(diagnostic, "output_stats", side_effect=ValueError("fixture diagnostic error")):
                        model(image)
                else:
                    output = model(image)
                    if mode == "finite":
                        assert torch.equal(output[0], image) and all(torch.equal(t, image) for t in output[1][:4])
            finally:
                handle.remove()
    except RuntimeError as error:
        native_error = str(error)
    assert torch.equal(rng, torch.random.get_rng_state()), "Observer changed RNG"
    assert torch.equal(image, image_before), "Observer changed input"
    assert all(torch.equal(v, expected.state_dict()[k]) for k, v in model.state_dict().items()), "Observer changed state beyond native half conversion"
    assert not validator.callbacks["on_val_start"]
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    d = row["diagnostic"]
    assert d["hooks_removed"] and d["forward_calls"] == 1
    assert d["input"]["actual_batch_size"] == 2 and d["input"]["dtype"] == "torch.float16"
    assert not d["input"]["model_training"] and d["ema_updates"] == 17
    json.dumps(row, allow_nan=False)
    return d, native_error


def run():
    cases = {}
    for mode, path in (("y", "y"), ("raw", "raw.enc_scores")):
        d, error = execute(mode)
        assert error == "Nonfinite native half EMA prediction"
        assert d["outputs"]["nonfinite_paths"] == [path]
        cases[mode] = dict(nonfinite_paths=d["outputs"]["nonfinite_paths"], original_error=error, hooks_removed=d["hooks_removed"])
    d, error = execute("conversion")
    assert "Nonfinite or unobserved EMA" in error
    assert d["ema_before_half"]["finite"] and not d["ema_after_native_half"]["finite"]
    assert set(d["ema_after_native_half"]["new_nonfinite_names"]) == {"parameters.weight", "buffers.running"}
    assert d["ema_after_native_half"]["buffers"]["integer_items"] == 1
    cases["conversion"] = {k: d[k] for k in ("ema_before_half", "ema_after_native_half")}
    d, error = execute("finite", preexisting=True)
    assert not d["ema_before_half"]["finite"] and d["ema_after_native_half"]["new_nonfinite_items"] == 0
    cases["preexisting_nonfinite_buffer"] = True
    d, error = execute("boundary")
    first = d["first_nonfinite"]
    assert first["module"] == "overflow" and first["side"] == "output" and first["input_finite"] is True
    assert first["root_cause"] == "UNDETERMINED" and first["preceding_finite"]
    cases["finite_input_bad_output"] = first
    d, error = execute("native_error")
    assert error == "fixture original native exception" and d["hooks_removed"]
    cases["native_exception_cleanup"] = True
    d, error = execute("raw", broken_diagnostic=True)
    assert error == "Nonfinite native half EMA prediction" and d["diagnostic_errors"]
    cases["diagnostic_error_preserves_native_error"] = True
    d, error = execute("finite", broken_diagnostic=True)
    assert "diagnostic incomplete" in error
    cases["missing_diagnostics_blocks_pass"] = True
    d, error = execute("finite")
    assert error is None and not d["outputs"]["nonfinite_paths"] and d["first_nonfinite"] is None
    cases["finite_result_unchanged"] = True
    all_bad = diagnostic.tensor_stats(torch.tensor([float("nan"), float("inf"), -float("inf")]), "fixture")
    assert (all_bad["nan"], all_bad["posinf"], all_bad["neginf"], all_bad["finite_max_abs"]) == (1, 1, 1, None)
    cases["standard_json_nonfinite_counts"] = all_bad

    # Explicit -Inf attention masks are labelled separately; Inf in data remains an anomaly.
    row, v = {}, Validator()
    m = nn.MultiheadAttention(4, 2, dropout=0.)
    observer = diagnostic.HalfEMADiagnostic(m, v, row, amp=True, ema_updates=0)
    mask = torch.tensor([[0., -float("inf")], [-float("inf"), 0.]])
    observer.boundary("attention", m, "input", dict(kwargs=dict(attn_mask=mask)))
    assert row["diagnostic"]["semantic_masks"] and row["diagnostic"]["first_nonfinite"] is None
    observer.boundary("attention", m, "input", dict(args=(mask,)))
    assert row["diagnostic"]["first_nonfinite"] is not None
    for _ in range(30):
        observer.boundary("attention", m, "output", mask, True)
    assert len(row["diagnostic"]["anomalies"]) == diagnostic.EVENT_LIMIT
    assert row["diagnostic"]["nonfinite_events"] == 31
    cases["semantic_mask_and_event_cap"] = True
    head = SimpleNamespace(anchors=torch.tensor([[[float("inf"), float("inf")], [0., 0.]]]),
                           valid_mask=torch.tensor([[[False], [True]]]))
    observer.model = SimpleNamespace(model=[head])
    observer.anchors()
    assert row["diagnostic"]["anchor_cache"]["positive_inf_matches_invalid_mask"] is True
    assert row["diagnostic"]["anchor_cache"]["invalid_anchor_elements"] == 2
    cases["anchor_cache_semantics_only"] = True

    old = subprocess.check_output(["git", "show", OLD+":tools/preflight_rdm.py"], cwd=ROOT).decode()
    new = (ROOT/"tools/preflight_rdm.py").read_text(encoding="utf-8")
    def source_part(source, name):
        return ast.get_source_segment(source, next(n for n in ast.parse(source).body if getattr(n, "name", None) == name))
    for name in ("RDMTrainer", "classify_amp_attempt", "observe_amp_attempt", "check_training_progress", "bounded_training", "OneBatch"):
        assert source_part(old, name) == source_part(new, name), name + " changed"
    protected = ["ultralytics-main", "configs", "tools/rdm.py", "tools/rdm_common.py", "tools/rdm_server.sh", "tools/sync_rdm.sh",
                 "docs/rdm_v1/parent_args.yaml", "docs/rdm_v1/data_identity.json", "docs/rdm_v1/cbr_lif_rdm_v1_recipe.yaml", "docs/rdm_v1/rdm_v1_recipe.yaml"]
    assert not subprocess.check_output(["git", "diff", OLD, "--", *protected], cwd=ROOT)
    init = ROOT/"weights/cbr_lif_rdm_v1_controlled_init.pt"
    digest = hashlib.sha256(init.read_bytes()).hexdigest()
    assert digest == "5dac7ce5dae8ad6f94890c9d5f70efa9e37d4fac855dae0d2fe5d25bc070a443"
    return dict(status="PASSED", scope="tiny CPU fixture only, NOT real server half EMA validation", cases=cases,
                production_and_amp_budget_unchanged=True, existing_local_init_sha256=digest,
                server_validation="PENDING", numerical_failure="NOT_FIXED; NEEDS_SERVER_DIAGNOSTIC", formal_training="NOT_STARTED",
                torch=str(torch.__version__), checkpoints_created=0)


if __name__ == "__main__":
    report = run()
    destination = ROOT/"docs/rdm_v1/half_ema_diagnostic/fixtures.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], scope=report["scope"], cases=len(report["cases"]),
                          report=str(destination), bytes=destination.stat().st_size), indent=2))
