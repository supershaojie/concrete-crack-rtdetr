"""R1 fail-closed policy and actual native update boundary regression tests."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from init_dpr import ROOT
from dpr_acceptance import (validate_b, validate_half, validate_supplement, validate_scope, validate_full_assessments, scope, table, raw_metric,
                            CONTRACT, FULL_SCHEMA, policy)
from dpr_r1_evidence import native_update_capture
from preflight_dpr import snapshot, compare
from train_dpr import atomic_json
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import ModelEMA


class TinyTrainer(SimpleNamespace):
    optimizer_step = BaseTrainer.optimizer_step


def rejects(function, *args, **kwargs):
    try: function(*args, **kwargs)
    except (RuntimeError, KeyError, TypeError): return
    raise AssertionError("Malformed/incomplete evidence was accepted")


def native_replay_unit(device, amp, root):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 1)).to(device)
    trainer = TinyTrainer(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=.0005),
        scaler=torch.cuda.amp.GradScaler(enabled=amp), ema=ModelEMA(model), updates=0, epoch=0,
        start_epoch=0, epochs=200, best_fitness=0., device=torch.device(device))
    records, states = [], []
    for i in range(2):
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss = model(torch.randn(4, 8, device=device)).square().sum() * .0001
        trainer.scaler.scale(loss).backward()
        with native_update_capture(trainer, root / (device+str(i)+".pt"), records, states): trainer.optimizer_step()
    assert all(x["post_comparison"]["exact"] and x["pre_state_exact"] and x["scaled_gradients_exact"]
               and x["storage_independent"] for x in records)
    assert "optimizer_step" not in trainer.__dict__
    try:
        with native_update_capture(trainer, root / "unused.pt", [], []): raise ValueError("cleanup fault")
    except ValueError: pass
    assert "optimizer_step" not in trainer.__dict__
    return dict(native_replays=2, actual_amp=trainer.scaler.is_enabled(), exact=True, exception_cleanup=True)


def fault_real_evidence(report):
    """Use collected raw tensors/metrics; never manufacture a positive report."""
    checks = {}
    for mode, original in report.get("modes", {}).items():
        if "B_evidence" not in original: continue
        # Existing genuine failures must not be bypassed by rewriting statuses.
        changed = deepcopy(original); changed["status"] = "PASSED"
        changed["assessment"] = {"status":"EXPLAINED_BACKWARD_VARIATION"}
        try: baseline = validate_b(original, mode, files=False)
        except RuntimeError:
            rejects(validate_b, changed, mode, files=False); checks[mode+"_summary_cannot_override_failure"] = True
        # A clearly synthetic unit fixture isolates guard behavior after B7.
        # Never write it as collected evidence or relabel the input report.
        positive = deepcopy(original)
        for control in ("parent", "candidate"):
            # Old development records predate the initial-table field. This
            # synthetic exact table is only a unit fixture, never real evidence.
            extra = positive["B_evidence"][control]
            extra.setdefault("initial_comparison", deepcopy(extra["native_replays"][0]["post_comparison"]))
        trace = positive["B_evidence"]["updated_function"]
        for row in trace["comparisons"].values():
            row.update(raw_allclose=True, max_abs=0., relative_L2=0., exceed_fraction=0.)
        validate_b(positive, mode, files=False)
        checks[mode+"_synthetic_guard_fixture"] = "UNIT_ONLY_NOT_OBSERVED"
        for label, mutate in (
            ("A_missing", lambda x:x.pop("A")),
            ("B2_rng", lambda x:x["B_evidence"]["candidate"].pop("starts")),
            ("B2_initial", lambda x:x["B_evidence"]["candidate"].pop("initial_comparison")),
            ("B3_moment", lambda x:x["B"]["candidate"]["state"]["tensors"].pop("state.optimizer.model.5.blocks.1.branch2b.dpr_cd.exp_avg")),
            ("B5_maps", lambda x:x["B_evidence"]["mapping"].pop("difference_propagation")),
            ("B5_scale", lambda x:x["B_evidence"]["mapping"]["runs"][0].update(scale=.5)),
            ("B5_artifact", lambda x:x["B_evidence"]["mapping"]["runs"][0].pop("artifact")),
            ("B6_replay", lambda x:x["B_evidence"]["candidate"].pop("native_replays")),
            ("B7_function", lambda x:x["B_evidence"].pop("updated_function")),
            ("B7_actual_indices", lambda x:x["B_evidence"]["updated_function"]["selection"].update(changed_positions=-1)),
            ("policy", lambda x:x["policy"].update(contract="dpr_acceptance_v1"))):
            changed = deepcopy(positive); mutate(changed); rejects(validate_b, changed, mode, files=False)
            checks[mode+"_"+label] = True
    if report.get("half"):
        original = report["half"]
        validate_half(original, files=False)  # only a genuine passing fixture can test all downstream mutations
        for label, mutate in (
            ("H0_true_half", lambda x:x["H0"].update(input_dtypes=["torch.float32"])),
            ("H1_state", lambda x:x["H1"].update(state_bytes_exact=False)),
            ("H2_final", lambda x:x["H2"].pop("final_eval_half")),
            ("H2_source", lambda x:x["H2"]["native_ema_half"]["audit"]["provenance"].update(selected="model")),
            ("H3_scope", lambda x:x["H3"].update(capability="PASSED")),
            ("H4_raw", lambda x:x["H4"].pop("raw"))):
            changed=deepcopy(original); mutate(changed); rejects(validate_half, changed, files=False); checks[label]=True
    return checks


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--evidence", type=Path, nargs="*", default=[])
    args=parser.parse_args(); torch.set_num_threads(4)
    require_new = not args.output.exists(); assert require_new, "Preserve old unit report"
    result=dict(status="FAILED", scope="unit only; never admission", contract=CONTRACT, checks={})
    with tempfile.TemporaryDirectory(prefix="r1_unit_", dir=ROOT / "outputs/dpr") as root:
        result["checks"]["cpu_native_replay"] = native_replay_unit("cpu", False, Path(root))
        if torch.cuda.is_available(): result["checks"]["amp_native_replay"] = native_replay_unit("cuda", True, Path(root))
        rejects(validate_supplement, {"status":"PASSED", "report_schema":FULL_SCHEMA}, files=False)
        rejects(validate_full_assessments, dict(contract=CONTRACT, report_schema=FULL_SCHEMA, policy=policy(), capability_scope=scope()), files=False)
        rejects(validate_half, {"status":"PASSED"}, files=False)
        rejects(validate_b, {"status":"PASSED"}, "cuda_fp32", files=False)
        changed=scope(); changed["fp16_cross_precision_H3"]="PASSED"; rejects(validate_scope, changed)
        row=dict(finite=True, raw_allclose=True, atol=2e-5, rtol=2e-4, max_abs=1., relative_L2=1., exceed_fraction=.1)
        rejects(raw_metric,row)
        table(compare({"v":torch.ones(1)}, {"v":torch.ones(1)}), exact=True)
        result["checks"]["missing_stale_scope_inconsistent_raw_rejected"]=True
        for source in args.evidence:
            result["checks"][str(source)] = fault_real_evidence(json.loads(source.read_text(encoding="utf-8")))
    result["status"]="PASSED"; atomic_json(args.output,result); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
