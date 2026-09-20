"""Tiny CPU fixtures for preflight classification/hooks/budgets, NOT server AMP validation."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
from preflight_rdm import (observe_amp_attempt, classify_amp_attempt, check_training_progress,
                           BudgetPending, BoundedStop)
from rdm_common import NEW, ROOT, write_json, sha256, lf_hash
from rdm import preflight_failures


class FixtureScaler:
    """Explicit simulation of scaler decisions; never label this real GradScaler evidence."""
    def __init__(self):
        self.scale = 65536.

    def get_scale(self):
        return self.scale

    def get_backoff_factor(self):
        return .5

    def is_enabled(self):
        return True


def fixture():
    params = {n: torch.nn.Parameter(torch.zeros(1)) for n in sorted(NEW)}
    params.update({"fixture.overflow": torch.nn.Parameter(torch.ones(1)), "fixture.unused": torch.nn.Parameter(torch.ones(1))})
    t = SimpleNamespace(model=SimpleNamespace(named_parameters=lambda: params.items()),
                        optimizer=torch.optim.AdamW(params.values(), lr=.001), scaler=FixtureScaler(),
                        amp=True, accumulate=1, loss=torch.tensor(37.82496), loss_items=torch.tensor([1., 2., 3.]))
    row = dict(status="RUNNING", batches=0, effective_updates=0, scaler_skips=0, steps=[], upstream_active={}, consecutive_nonfinite_steps=0)
    return t, row, params


def attempt(t, row, params, overflow=False, mode="native", mutate=None):
    row["batches"] += 1
    row["current_forward"] = dict(batch=row["batches"], rdm=True, decoder=True)
    for n, p in params.items():
        p.grad = None if n == "fixture.unused" else torch.ones_like(p) * t.scaler.scale
    if overflow:
        params["fixture.overflow"].grad.fill_(float("inf"))
    if mutate:
        mutate(t, row, params)
    original_hooks = len(t.optimizer._optimizer_step_pre_hooks)
    try:
        with observe_amp_attempt(t, row):
            if mode == "raise":
                raise RuntimeError("fixture native exception")
            if mode == "skip_without_backoff":
                pass
            elif overflow and mode == "native":
                t.scaler.scale *= .5  # simulated native skip, no optimizer invocation
            else:
                for p in params.values():
                    if p.grad is not None:
                        p.grad.div_(t.scaler.scale)
                if mode == "zero_applied_Wo":
                    params["model.5.rdm.Wo.weight"].grad.zero_()
                assert t.optimizer.step() is None  # Return None does NOT mean skipped.
            t.optimizer.zero_grad()
    finally:
        assert len(t.optimizer._optimizer_step_pre_hooks) == original_hooks


def run():
    torch.set_num_threads(1)
    t, row, params = fixture()
    for _ in range(3):
        attempt(t, row, params, overflow=True)
        check_training_progress(row, row["batches"], 2, 8)
    assert row["effective_updates"] == 0 and row["scaler_skips"] == 3 and row["consecutive_nonfinite_steps"] == 3
    assert [r["classification"] for r in row["steps"]] == ["AMP_BACKOFF"] * 3
    assert [r["scale_after"] for r in row["steps"]] == [32768., 16384., 8192.]
    assert all(set(r["gradients"]) == NEW and all(g["finite"] for g in r["gradients"].values()) for r in row["steps"])
    assert all(r["nonfinite_scaled_gradient_names"] == ["fixture.overflow"] and r["none_gradient_parameter_items"] == 1 for r in row["steps"])
    attempt(t, row, params)
    attempt(t, row, params)
    assert row["effective_updates"] == 2 and row["steps"][-1]["optimizer_step_calls_delta"] == 1
    try:
        check_training_progress(row, row["batches"], 2, 8)
    except BoundedStop:
        pass
    else:
        raise AssertionError("Valid effective updates not accepted")

    failures = {}
    mutations = {
        "current_loss_nan": lambda t,r,p: setattr(t, "loss", torch.tensor(float("nan"))),
        "loss_items_inf": lambda t,r,p: setattr(t, "loss_items", torch.tensor([float("inf")])),
        "current_output_nonfinite": lambda t,r,p: r["current_forward"].update(decoder=False),
        "previous_batch_output_only": lambda t,r,p: r["current_forward"].update(batch=0),
        "missing_output_evidence": lambda t,r,p: r["current_forward"].pop("rdm"),
        "parameter_nonfinite": lambda t,r,p: p["fixture.overflow"].detach().fill_(float("inf")),
        "missing_RDM_gradient": lambda t,r,p: setattr(p["model.5.rdm.Dc.weight"], "grad", None),
    }
    for name, mutation in mutations.items():
        a, b, c = fixture()
        try:
            attempt(a, b, c, overflow=True, mutate=mutation)
        except RuntimeError:
            assert b["steps"][-1]["classification"] == "FAILED"
            failures[name] = True
        else:
            raise AssertionError(name + " incorrectly accepted")
    for mode in ("skip_without_backoff", "apply_nonfinite", "raise"):
        a, b, c = fixture()
        try:
            attempt(a, b, c, overflow=True, mode=mode)
        except RuntimeError:
            assert b["steps"][-1]["classification"] == "FAILED"
            failures[mode] = True
        else:
            raise AssertionError(mode + " incorrectly accepted")
    for name, edit in {
        "missing_step_call_evidence": lambda r: r.pop("optimizer_step_calls_delta"),
        "step_called_despite_overflow": lambda r: r.update(optimizer_step_calls_delta=1),
        "wrong_backoff_factor": lambda r: r.update(scale_after=16000.),
        "missing_one_of_seven_gradient_records": lambda r: r["gradients"].pop(next(iter(NEW))),
        "no_overflow_but_skip": lambda r: r.update(all_scaled_gradients_finite=True),
        "nonfinite_parameters_after": lambda r: r.update(parameters_finite_after=False),
    }.items():
        r = deepcopy(row["steps"][0]); edit(r)
        try:
            classify_amp_attempt(r)
        except (RuntimeError, KeyError):
            failures[name] = True
        else:
            raise AssertionError(name + " incorrectly accepted")

    # A real optimizer invocation with zero applied Wo gradient is not an effective update.
    a, b, c = fixture()
    c["model.5.rdm.Wo.weight"].detach().fill_(1.)
    attempt(a, b, c, mode="zero_applied_Wo")
    assert b["steps"][-1]["Wo_max_change"] > 0 and b["steps"][-1]["optimizer_step_calls_delta"] == 1
    assert b["effective_updates"] == 0 and b["steps"][-1]["classification"] == "NO_EFFECTIVE_UPDATE"

    budgets = {}
    for stage, prior, target, limit in (("start", 0, 2, 8), ("resume", 8, 1, 8)):
        a, b, c = fixture()
        for batch in range(1, 9):
            attempt(a, b, c, overflow=True)
            try:
                check_training_progress(b, prior+batch, target, limit)
            except BudgetPending:
                assert batch == 8 and b["effective_updates"] == 0
                b["status"] = "PENDING"  # same exception-to-status mapping in bounded_training
        assert b["status"] == "PENDING"
        assert preflight_failures(dict(status="PENDING", **{stage: b}), {})
        budgets[stage] = dict(status="PENDING", actual_batches=8, total_batches=prior+8, effective_updates=0, start_blocked=True)

    old = "d6e4aa96db7ffdb749ac433c4bb8faa2c39a537d"
    old_source = subprocess.check_output(["git", "show", old+":tools/preflight_rdm.py"], cwd=ROOT).decode()
    new_source = (ROOT/"tools/preflight_rdm.py").read_text(encoding="utf8")
    def formal_class(source):
        return ast.get_source_segment(source, next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RDMTrainer"))
    assert formal_class(old_source) == formal_class(new_source)
    protected = ["ultralytics-main/ultralytics", "tools/rdm_common.py", "tools/rdm.py", "tools/rdm_server.sh", "tools/sync_rdm.sh",
                 "docs/rdm_v1/parent_args.yaml", "docs/rdm_v1/data_identity.json", "docs/rdm_v1/cbr_lif_rdm_v1_recipe.yaml", "docs/rdm_v1/rdm_v1_recipe.yaml"]
    assert not subprocess.check_output(["git", "diff", old, "--", *protected], cwd=ROOT)
    init = ROOT/"weights/cbr_lif_rdm_v1_controlled_init.pt"
    assert sha256(init) == "5dac7ce5dae8ad6f94890c9d5f70efa9e37d4fac855dae0d2fe5d25bc070a443"
    return dict(status="PASSED", scope="CPU fixture, simulated scaler decisions + real tiny AdamW step hooks; NOT server AMP validation",
                three_backoffs_then_updates=row["steps"], failures_rejected=failures, budgets=budgets,
                optimizer_hooks_removed=True, formal_RDMTrainer_unchanged=True, protected_model_recipe_files_unchanged=True,
                existing_init_sha256=sha256(init), changed_source_lf_sha256=lf_hash(ROOT/"tools/preflight_rdm.py"),
                server_preflight="PENDING", formal_training="NOT_STARTED", full_val="NOT_RUN", final_test="NOT_RUN",
                checkpoints_created=0, old_reports_modified=False, torch=str(torch.__version__))


if __name__ == "__main__":
    report = run()
    destination = ROOT/"docs/rdm_v1/amp_backoff_fix/fixtures.json"
    write_json(destination, report)
    print(json.dumps(dict(status=report["status"], scope=report["scope"], failures_rejected=len(report["failures_rejected"]), report=str(destination)), indent=2))
