"""Focused PBI fusion precision regression probes; never train or evaluate a dataset."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback
from types import MethodType
from unittest.mock import patch

import torch

from check_pbi_math import (
    FusionAuditError, precision_settings, strict_fp32_reference, wrapper_fusion_audit,
)
from pbi_common import require, runtime, sha256, write_json
from ultralytics.nn.modules import PBI, PBIConv
from ultralytics.utils.torch_utils import fuse_conv_and_bn


class RestorationProbe(RuntimeError):
    """Sentinel deliberately raised inside the controlled precision context."""


def restore_settings(settings):
    # The precision setter must be last: setting allow_tf32=True selects "high".
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(settings["float32_matmul_precision"])


def context_cases():
    """Exercise every effective legacy TF32/float32-precision combination."""
    original = precision_settings()
    rows = []
    try:
        for precision in ("highest", "high", "medium"):
            for cudnn_tf32 in (False, True):
                for exceptional in (False, True):
                    restore_settings(dict(matmul_allow_tf32=precision != "highest",
                                          cudnn_allow_tf32=cudnn_tf32,
                                          float32_matmul_precision=precision))
                    before = precision_settings()
                    require(before["float32_matmul_precision"] == precision and
                            before["cudnn_allow_tf32"] == cudnn_tf32 and
                            before["matmul_allow_tf32"] == (precision != "highest"),
                            "Unable to establish precision restoration test input")
                    evidence = {}
                    raised = False
                    try:
                        with strict_fp32_reference(evidence):
                            require(precision_settings() == dict(matmul_allow_tf32=False,
                                    cudnn_allow_tf32=False, float32_matmul_precision="highest"),
                                    "Strict reference did not disable both TF32 paths")
                            if exceptional:
                                raise RestorationProbe("deliberate exceptional exit")
                    except RestorationProbe:
                        raised = True
                    after = precision_settings()
                    require(raised == exceptional, "Controlled context swallowed the test exception")
                    require(after == before, "Controlled context leaked a runtime precision setting")
                    require(evidence.get("before") == before and evidence.get("after") == after and
                            evidence.get("restored") is True and
                            evidence.get("exited_via_exception") == exceptional,
                            "Restoration evidence does not describe the observed exit")
                    rows.append(dict(status="PASSED",exceptional_exit=exceptional,
                                     before=before,after=after,evidence=evidence))
    finally:
        restore_settings(original)
        require(precision_settings() == original, "Regression probe leaked runtime precision settings")
    return dict(status="PASSED",cases=rows,
                effective_combinations=6,normal_exits=6,exceptional_exits=6,
                note="Legacy matmul allow_tf32 and float32_matmul_precision are coupled; "
                     "highest=False, high/medium=True cover all effective combinations.")


def learned_wrapper(device):
    wrapped = PBIConv(128, 256, 1, 1, None, 1, 1, False, 32).eval().to(device)
    with torch.no_grad():
        wrapped.pbi.Wo.weight.normal_(std=.005)
        wrapped.bn.running_mean.normal_(std=.1)
        wrapped.bn.running_var.uniform_(.5, 1.5)
        wrapped.bn.weight.uniform_(.8, 1.2)
        wrapped.bn.bias.normal_(std=.03)
    return wrapped, torch.randn(2, 128, 9, 11, device=device)


def fuse_wrapper(wrapped):
    # Fault fixtures isolate the deliberate fault from native TF32 quantization
    # in Conv/BN fusion itself; production audits create each regime separately.
    with strict_fp32_reference({}):
        fused = deepcopy(wrapped)
        fused.conv = fuse_conv_and_bn(fused.conv, fused.bn)
        delattr(fused, "bn")
        fused.forward = fused.forward_fuse
    return fused


def omit_pbi(self, image):
    return self.act(self.conv(image))


def duplicate_pbi(self, image):
    return self.pbi(self.pbi(self.act(self.conv(image))))


def nonfinite_pbi(self, image):
    return PBI.forward(self, image) * float("nan")


def shifted_pbi(self, image):
    return PBI.forward(self, image) + .1


def simulated_native_rounding(self, image):
    result = PBI.forward(self, image)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        result = result + .1
    return result


def native_diagnostic_case(wrapped, image):
    """A synthetic precision-dependent offset tests preservation of native failure."""
    original = precision_settings()
    try:
        restore_settings(dict(matmul_allow_tf32=True,cudnn_allow_tf32=True,
                              float32_matmul_precision="high"))
        fused = fuse_wrapper(wrapped)
        fused.pbi.forward = MethodType(simulated_native_rounding, fused.pbi)
        evidence = wrapper_fusion_audit(wrapped, image, fused=fused)
        native, strict = evidence.get("native", {}), evidence.get("strict_fp32", {})
        require(native.get("status") == "PRECISION_NOTE" and
                native.get("after_pbi", {}).get("raw_allclose") is False and
                native.get("after_pbi", {}).get("finite") is True and
                native.get("after_pbi", {}).get("max_abs", 0) > .01,
                "Native diagnostic failure was discarded or relabeled as native success")
        require(strict.get("status") == "PASSED" and
                strict.get("after_pbi", {}).get("raw_allclose") is True,
                "Controlled reference did not pass after the synthetic native-only offset")
        return dict(status="PASSED",kind="SYNTHETIC_PRECISION_POLICY_PROBE",
                    note="Injected a native-only 0.1 offset; this is not a measurement of hardware TF32 error.",
                    evidence=evidence)
    finally:
        restore_settings(original)
        require(precision_settings() == original, "Synthetic native policy probe leaked precision settings")


def require_stored_diagnostics(evidence):
    require(isinstance(evidence, dict) and evidence.get("settings_before") and
            evidence.get("settings_after"), "Failed fusion lost runtime precision settings")
    for mode in ("native", "strict_fp32"):
        row = evidence.get(mode, {})
        require(row.get("precision") and isinstance(row.get("pbi_calls"), dict) and
                "pbi_state_equal" in row and "before_pbi" in row and "after_pbi" in row,
                "Failed fusion lost staged numerical/call/state diagnostics: " + mode)
        for stage in ("before_pbi", "after_pbi"):
            require(isinstance(row[stage].get("raw_allclose"), bool) and
                    isinstance(row[stage].get("finite"), bool) and
                    "max_abs" in row[stage] and "nonfinite_a" in row[stage] and
                    "nonfinite_b" in row[stage], "Failure dropped concrete comparison evidence")
    encoded = json.dumps(evidence, allow_nan=False)
    restored = json.loads(encoded)
    require(restored == evidence, "Failure evidence is not preserved by a standards-compliant JSON round trip")
    return restored


def device_cases(device):
    wrapped, image = learned_wrapper(device)
    original = precision_settings()
    valid = wrapper_fusion_audit(wrapped, image)
    require(precision_settings() == original, "Valid fusion changed process precision settings")
    strict = valid.get("strict_fp32", {})
    require(strict.get("status") == "PASSED" and
            all(strict.get(stage, {}).get("raw_allclose") is True and
                strict.get(stage, {}).get("finite") is True for stage in ("before_pbi", "after_pbi")),
            "Correct nonzero fusion did not pass the original strict FP32 tolerance")
    require(strict.get("pbi_calls") == {"unfused": 1, "fused": 1} and
            strict.get("pbi_state_equal") is True and
            all(v > 0 for v in strict.get("residual_abs_max", {}).values()),
            "Successful reference lacks an active nonzero PBI branch")
    cases = []
    for name, expected in (("omitted_pbi", "PBI_CALL_COUNT"),
                           ("duplicate_pbi", "PBI_CALL_COUNT"),
                           ("nonfinite_output", "NONFINITE"),
                           ("strict_tolerance", "TOLERANCE")):
        fused = fuse_wrapper(wrapped)
        if name == "omitted_pbi":
            fused.forward = MethodType(omit_pbi, fused)
        elif name == "duplicate_pbi":
            fused.forward = MethodType(duplicate_pbi, fused)
        elif name == "nonfinite_output":
            fused.pbi.forward = MethodType(nonfinite_pbi, fused.pbi)
        else:
            fused.pbi.forward = MethodType(shifted_pbi, fused.pbi)
        rejected = False
        try:
            wrapper_fusion_audit(wrapped, image, fused=fused)
        except FusionAuditError as exc:
            rejected = True
            require(exc.reason == expected, "Fault got the wrong reason: " + name + ": " + str(exc.reason))
            saved = require_stored_diagnostics(exc.evidence)
            require(any(error.get("reason") == exc.reason for mode in ("native", "strict_fp32")
                        for error in saved[mode].get("errors", [])),
                    "Failure reason was not preserved in the saved numerical evidence")
            if name == "strict_tolerance":
                strict_failure = exc.evidence.get("strict_fp32", {})
                require(strict_failure.get("after_pbi", {}).get("raw_allclose") is False and
                        strict_failure.get("after_pbi", {}).get("finite") is True and
                        strict_failure.get("after_pbi", {}).get("max_abs", 0) > .01,
                        "Numerical fault did not retain strict comparison values")
                require(exc.evidence.get("precision_context", {}).get("restored") is True,
                        "A strict fusion failure did not restore precision settings")
            if name == "nonfinite_output":
                require(exc.evidence["native"]["after_pbi"].get("finite") is False,
                        "Nonfinite fault lost its numerical diagnosis")
            cases.append(dict(case=name,status="PASSED",rejected=True,
                              reason=exc.reason,json_roundtrip_verified=True,evidence=saved))
        finally:
            require(precision_settings() == original, "Rejected fusion changed runtime precision settings: " + name)
        require(rejected, "Fusion audit accepted deliberately defective PBI: " + name)
    diagnostic = native_diagnostic_case(wrapped, image)
    return dict(device=device,status="PASSED",correct_nonzero_fusion=valid,
                native_diagnostic_preservation=diagnostic,faults=cases)


def math_report_cases(path):
    """Mutate a real passing report only in memory; these fixtures grant no permission."""
    from train_pbi import _math_pass
    source = json.loads(path.read_text(encoding="utf-8"))
    _math_pass(source)
    cpu = next(index for index, row in enumerate(source["devices"]) if row["device"] == "cpu")
    cuda = next(index for index, row in enumerate(source["devices"]) if row["device"] == "cuda")
    fusion = ("devices", cpu, "nonzero_wrapper_fusion")
    native, strict = fusion + ("native",), fusion + ("strict_fp32",)
    first_source = next(iter(source["audit_source"]))
    remove = object()
    mutations = (
        ("old_math_evidence_version", [(('math_evidence_version',), 'pbi_math_precision_v1')]),
        ("missing_precision", [(native + ('precision',), remove)]),
        ("mismatched_audit_source", [(('audit_source', first_source), '0' * 64)]),
        ("native_false_mislabelled_passed", [
            (native + ('after_pbi', 'raw_allclose'), False),
            (native + ('after_pbi', 'max_abs'), .1),
            (native + ('after_pbi', 'finite_max_abs'), .1),
            (native + ('raw_allclose',), False), (native + ('status',), 'PASSED'),
            (fusion + ('native_precision_status',), 'PASSED')]),
        ("strict_tolerance_failed", [(strict + ('after_pbi', 'raw_allclose'), False),
                                     (strict + ('after_pbi', 'max_abs'), .1),
                                     (strict + ('after_pbi', 'finite_max_abs'), .1),
                                     (strict + ('raw_allclose',), False)]),
        ("missing_pbi_call", [(strict + ('pbi_calls', 'fused'), 0)]),
        ("nonfinite_fusion", [(native + ('after_pbi', 'finite'), False),
                              (native + ('after_pbi', 'max_abs'), None),
                              (native + ('after_pbi', 'nonfinite_b'), 1),
                              (native + ('finite',), False)]),
        ("precision_not_restored", [(fusion + ('precision_context', 'restored'), False)]),
        ("native_amp_still_rejected", [(('devices', cuda, 'native_amp', 'raw_allclose'), False)]),
        ("cuda_half_still_rejected", [(('devices', cuda, 'cuda_half', 'raw_allclose'), False)]),
        ("native_gradient_still_rejected", [
            (('devices', cuda, 'nonzero_gradients', 'Wo.weight', 'raw_allclose'), False)]),
    )
    cases = []
    for name, changes in mutations:
        fixture = deepcopy(source)
        for fields, value in changes:
            target = fixture
            for field in fields[:-1]:
                target = target[field]
            if value is remove:
                del target[fields[-1]]
            else:
                target[fields[-1]] = value
        rejection = None
        try:
            _math_pass(fixture)
        except RuntimeError as exc:
            rejection = str(exc)
        require(bool(rejection), "Math validator accepted deliberately invalid report fixture: " + name)
        cases.append(dict(case=name,status="PASSED",rejected=True,reason=rejection,
                          changed_fields=[".".join(map(str, fields)) for fields, _ in changes]))
    require(json.loads(path.read_text(encoding="utf-8")) == source, "Fixture probe changed the source math report")
    return dict(status="PASSED",evidence_kind="TEST_FIXTURE_NOT_PERMISSION",
                source_report=str(path.resolve()),source_sha256=sha256(path),
                real_source_passed_math_validator=True,source_report_unchanged=True,cases=cases,
                note="In-memory admission fault fixtures only; this does not authorize training, "
                     "replace full native checks, or verify server capacity.")


def cli_failure_case(directory):
    """Run the actual math CLI through formula_check and verify durable failure evidence."""
    import check_pbi_math as audit
    directory.mkdir(parents=True, exist_ok=False)
    failed_path = directory / "omitted_pbi_math_FAILED.json"
    original = precision_settings()
    real_audit = audit.wrapper_fusion_audit
    caught = None

    def omit_branch(wrapped, image, fused=None):
        faulty = fuse_wrapper(wrapped)
        faulty.forward = MethodType(omit_pbi, faulty)
        return real_audit(wrapped, image, fused=faulty)

    with patch.object(audit, "wrapper_fusion_audit", side_effect=omit_branch), \
            patch.object(audit, "full_structure", side_effect=AssertionError("must stop at fusion failure")) as later, \
            patch.object(sys, "argv", ["check_pbi_math.py", "--device", "cpu", "--output", str(failed_path)]):
        try:
            audit.main()
        except FusionAuditError as exc:
            caught = exc
        require(not later.called, "Math CLI continued after a fusion failure")
    require(caught is not None and caught.reason == "PBI_CALL_COUNT", "Math CLI swallowed/misidentified a fusion fault")
    require(failed_path.is_file(), "Math CLI did not persist the failed report")

    def reject_constant(value):
        raise ValueError("Nonstandard numeric token in failure report: " + value)

    persisted = json.loads(failed_path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    require(persisted.get("status") == "FAILED" and persisted.get("error", "").startswith("PBI_CALL_COUNT:"),
            "Math CLI did not preserve the failure status/reason")
    require(persisted.get("precision_restored") is True and
            persisted.get("precision_before") == persisted.get("precision_after") == original and
            precision_settings() == original, "Math CLI failure leaked precision settings")
    rows = persisted.get("devices", [])
    require(len(rows) == 1 and rows[0].get("device") == "cpu" and rows[0].get("status") == "FAILED",
            "Math CLI lost its failing device record")
    saved = require_stored_diagnostics(rows[0].get("nonzero_wrapper_fusion"))
    require(saved == caught.evidence and saved["native"]["pbi_calls"]["fused"] == 0 and
            saved["strict_fp32"]["pbi_calls"]["fused"] == 0,
            "Persisted CLI diagnostics differ from the actual failing fusion evidence")
    require(persisted.get("code_identity_unchanged_during_run") is True,
            "CLI fixture source changed during verification")
    return dict(status="PASSED",evidence_kind="TEST_FIXTURE_NOT_PERMISSION",rejected=True,
                reason=caught.reason,failed_report=str(failed_path.resolve()),failed_report_sha256=sha256(failed_path),
                json_roundtrip_verified=True,full_structure_not_called=True,precision_restored=True,
                evidence=saved,note="An intentional omitted-PBI fixture exercised the real math CLI failure path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("all", "cpu", "cuda"), default="all")
    parser.add_argument("--math-report", type=Path,
                        help="Validate an actual new math report, then reject altered copies in memory")
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing precision regression report")
    report = dict(status="FAILED",report_kind="pbi_math_precision_regression",
                  evidence_version="pbi_fusion_precision_regression_v1",runtime=runtime(),
                  evidence_kind="TEST_FIXTURE_NOT_PERMISSION",
                  formal_training="NOT_STARTED",final_test="NOT_RUN",
                  server_rtx4090_torch212="PENDING",devices=[])
    original = precision_settings()
    try:
        torch.set_num_threads(4)
        report["context_restoration"] = context_cases()
        devices = ("cpu", "cuda") if args.device == "all" else (args.device,)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(42)
            for device in devices:
                if device == "cuda" and not torch.cuda.is_available():
                    report["devices"].append(dict(device=device,status="PENDING",reason="CUDA unavailable"))
                else:
                    report["devices"].append(device_cases(device))
        report["math_report_validation"] = math_report_cases(args.math_report) if args.math_report else dict(
            status="PENDING",reason="No --math-report supplied",evidence_kind="TEST_FIXTURE_NOT_PERMISSION")
        report["cli_failure_persistence"] = cli_failure_case(
            args.output.parent / (args.output.stem + "_cli_failure"))
        report["status"] = "PASSED" if (all(row["status"] == "PASSED" for row in report["devices"]) and
                                          report["math_report_validation"]["status"] == "PASSED") else "PENDING"
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["settings_before"] = original
        report["settings_after"] = precision_settings()
        report["settings_restored"] = report["settings_after"] == original
        if not report["settings_restored"]:
            report["status"] = "FAILED"
        from train_pbi import code_identity
        report["code_identity"] = code_identity()
        write_json(args.output, report)
    print(json.dumps(dict(status=report["status"],output=str(args.output.resolve()))))
    require(report["status"] == "PASSED", "Precision regression incomplete/failed; inspect the saved report")


if __name__ == "__main__":
    main()
