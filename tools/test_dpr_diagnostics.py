"""Bounded fault-oriented regression for DPR observations; no detector training."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import tempfile

import torch
from init_dpr import ROOT, build
from check_dpr import comparison, _capture, FP64_ATOL, FP64_RTOL
from dpr_diagnostics import (TARGET, backend_audit, capture_trace, compare_traces, decoder_selection,
                             transpose_maps, validate_backend_record, GradientObserver)


def run():
    torch.set_num_threads(4)
    torch.manual_seed(721)
    result = dict(scope="unit/synthetic evidence only; no admission", checks={})
    sample = comparison(torch.tensor([1.]), torch.tensor([1.21]), 0., .2)
    assert sample["raw_allclose"] and sample["exceed_fraction"] == 0
    assert not comparison(torch.tensor([float("nan")]), torch.zeros(1))["finite"]
    result["checks"]["right_reference_and_nonfinite"] = sample
    model = build("dpr_v1", nc=1).eval()
    target = model.get_submodule(TARGET)
    for name in target.parameter_names:
        with torch.no_grad():
            getattr(target, name).normal_(0, .003)
    local = deepcopy(target).double()
    gradients = []
    for _ in range(2):
        upstream = torch.randn_like(local.conv.weight)
        actual = torch.autograd.grad((local.get_equivalent_kernel()*upstream).sum(),
                                    [getattr(local, name) for name in local.parameter_names])
        expected = transpose_maps(upstream)
        for name, value in zip(local.parameter_names, actual):
            torch.testing.assert_close(value, expected[name], atol=FP64_ATOL, rtol=FP64_RTOL)
        gradients.append((upstream, dict(zip(local.parameter_names, actual))))
    expected_delta = transpose_maps(gradients[0][0]-gradients[1][0])
    for name in local.parameter_names:
        torch.testing.assert_close(gradients[0][1][name]-gradients[1][1][name], expected_delta[name], atol=FP64_ATOL, rtol=FP64_RTOL)
    assert not torch.allclose(expected["dpr_hd"], -expected["dpr_hd"], atol=FP64_ATOL, rtol=FP64_RTOL)
    assert not torch.allclose(expected["dpr_ad"], expected["dpr_ad"]*2, atol=FP64_ATOL, rtol=FP64_RTOL)
    result["checks"]["independent_adjoint_fp64_and_direction_scale_faults"] = "PASSED"
    class LocalContainer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__(); self.layer = layer
        def get_submodule(self, name):
            assert name == TARGET
            return self.layer
        def forward(self, x):
            return self.layer(x)
    a, b = LocalContainer(deepcopy(local)), LocalContainer(deepcopy(local))
    x = torch.randn(1,128,4,5,dtype=torch.float64)
    upstream = torch.randn_like(x)
    ya = a(x);ya.backward(upstream)
    with GradientObserver(b) as observer:
        yb=b(x);yb.backward(upstream)
    assert torch.equal(ya,yb)
    assert all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(a.parameters(),b.parameters()))
    assert 'get_equivalent_kernel' not in b.layer.__dict__
    try:
        with GradientObserver(b):
            b(x)
            raise RuntimeError('intentional gradient observation exception')
    except RuntimeError:
        pass
    assert 'get_equivalent_kernel' not in b.layer.__dict__
    assert not b.layer._forward_hooks and not b.layer._forward_pre_hooks
    result['checks']['gradient_observer_exact_forward_backward_and_exception_cleanup']='PASSED'
    image = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        original = model(image)[0].detach().clone()
    snapshot_hooks = lambda: {name: (dict(module._forward_hooks),dict(module._forward_pre_hooks)) for name,module in model.named_modules()}
    hooks = snapshot_hooks()
    native_topk = torch.topk
    before, selection = capture_trace(model, image)
    assert torch.equal(original.cpu(), before["output"])
    assert torch.topk is native_topk and len(selection["calls"]) == 1
    for fail_at in ("invocation", "selection"):
        try:
            if fail_at == "invocation":
                capture_trace(model, image, invoke=lambda x: (_ for _ in ()).throw(RuntimeError("intentional")))
            else:
                capture_trace(model, image, fixed_indices=torch.zeros((1, 300), dtype=torch.long))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected deliberate failure")
        assert "_get_decoder_input" not in model.model[-1].__dict__
        assert snapshot_hooks() == hooks
        assert 'get_equivalent_kernel' not in target.__dict__
        assert torch.topk is native_topk
    reversed_indices = before["candidate_indices"].flip(1)
    after, _ = capture_trace(model, image, fixed_indices=reversed_indices)
    candidate = compare_traces(before, after, model, image)
    assert candidate["candidate_sensitivity_complete"] and candidate["selection"]["batches"][0]["same_set"]
    broken = deepcopy(after)
    broken["P5"] += 1
    continuous = compare_traces(before, broken, model, image)
    assert not continuous["candidate_sensitivity_complete"] and continuous["first_exceeded"] == "P5"
    def wrong_output(x):
        out = model(x)
        return out[0]+1, out[1]
    replay = compare_traces(before, after, model, image, invoke=wrong_output)
    assert not replay["candidate_sensitivity_complete"] and not replay["fixed_candidate_replay_output"]["raw_allclose"]
    result["checks"]["actual_selection_observer_cleanup_and_replay_faults"] = dict(candidate=candidate, continuous=continuous, replay=replay)
    with tempfile.TemporaryDirectory(prefix="diag_unit_", dir=ROOT/"outputs/dpr") as tmp:
        checkpoint = Path(tmp)/"fixture.pt"
        torch.save(dict(model=model, ema=None, epoch=0, train_args={"task":"detect"}), checkpoint)
        fused = deepcopy(model).fuse(verbose=False)
        record = backend_audit(checkpoint, model, image, _capture(fused, image), legacy_model=fused)
        validate_backend_record(record, 2e-5, 2e-4)
        for fault in ("comparison", "nonfinite", "deployed", "identity", "missing_schema", "precision_note", "checkpoint_sha", "wrong_weights"):
            bad = deepcopy(record)
            bad["details"] = {"output": dict(raw_allclose=True)}
            if fault == "comparison": bad["comparison"]["raw_allclose"] = False
            elif fault == "nonfinite": bad["comparison"]["finite"] = False
            elif fault == "deployed": bad["deployed"] = False
            elif fault == "identity": bad["backend_identity"]["input"]["sha256"] = "0"*64
            elif fault == "missing_schema": bad.pop("backend_schema")
            elif fault == "precision_note": bad["status"] = "PRECISION_NOTE"
            elif fault == "checkpoint_sha": bad['provenance']['checkpoint_sha256']='0'*64
            else: bad['source_state_sha256']='0'*64
            try:
                validate_backend_record(bad, 2e-5, 2e-4)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Backend gate accepted fault: "+fault)
        result["checks"]["real_file_backend_and_gate_faults"] = record
        from train_dpr import strict_gate
        diagnostic=Path(tmp)/'diagnostic.json'
        diagnostic.write_text(json.dumps(dict(report_kind='dpr_failure_diagnostic_v1',status='PASSED',admission_eligible=False)))
        try:
            strict_gate('dpr_v1',checkpoint,checkpoint,checkpoint,diagnostic)
        except RuntimeError as error:
            assert 'complete DPR preflight contract' in str(error)
        else:
            raise AssertionError('A diagnostic report entered formal admission')
        result['checks']['diagnostic_cannot_enter_formal_gate']='PASSED'
    result["status"] = "PASSED"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Existing test report preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run()
    except Exception:
        import traceback
        report = dict(status="FAILED", traceback=traceback.format_exc())
    args.output.write_bytes((json.dumps(report, indent=2)+"\n").encode())
    print(report["status"], args.output)
    if report["status"] == "FAILED":
        print(report["traceback"])
        raise SystemExit(1)
