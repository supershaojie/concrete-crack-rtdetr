"""Finite local fusion regression; no server-capacity or AP claim."""
from __future__ import annotations

import argparse
from copy import deepcopy
from unittest.mock import patch

from tcr_v1_core import *
from tcr_v1_fusion import (
    diagnose, precision_scope, precision_state, model_identity, pair_invariants,
    early_trace, _one_comparison, rng_state, rng_hash, VERSION,
)
from ultralytics.nn.modules import Conv
from ultralytics.utils.torch_utils import init_seeds


def context_checks():
    original = precision_state()
    results = []
    try:
        for setting in ("highest", "high", "medium"):
            torch.set_float32_matmul_precision(setting)
            torch.backends.cudnn.allow_tf32 = True
            for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                dtype = torch.bfloat16 if device == "cpu" else torch.float16
                with torch.autocast(device_type=device, dtype=dtype):
                    before = precision_state()
                    for raises in (False, True):
                        try:
                            with precision_scope(strict=True) as inside:
                                require(not any(inside[k] for k in ("matmul_tf32", "cudnn_tf32", "autocast_cuda", "autocast_cpu")), "Strict flags not effective")
                                require(inside["float32_matmul_precision"] == "highest", "Strict matmul precision")
                                x = torch.ones(4, 4, device=device)
                                require((x @ x).dtype == torch.float32, "Actual operation still under autocast")
                                if raises:
                                    raise ValueError("injected scoped diagnostic failure")
                        except ValueError as error:
                            require(raises and "injected" in str(error), "Unexpected test error")
                        require(precision_state() == before, "Exception/success failed to restore precision")
                    require((x @ x).dtype == dtype, "Outer autocast was not restored")
                    with precision_scope(recorded=dict(tf32=False, cudnn_tf32=False)) as inside:
                        require(not inside["matmul_tf32"] and not inside["cudnn_tf32"], "Recorded TF32 flags not restored for replay")
                        require(inside["autocast_"+device] == before["autocast_"+device], "Runtime replay changed autocast")
                    require(precision_state() == before, "Recorded replay leaked precision settings")
                    results.append(dict(entry_matmul_precision=setting, device=device, status="PASS", success_and_exception=True))
    finally:
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])
        if torch.backends.cuda.matmul.allow_tf32 != original["matmul_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = original["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = original["cudnn_tf32"]
    require(precision_state() == original, "Context regression leaked settings")
    return results


def rejection_checks(model, image, folder):
    from tcr_v1_ops import preflight_accepted
    results = {}
    with precision_scope(strict=True):
        fused = deepcopy(model).fuse(verbose=False).eval().float()
        pair_invariants(model, fused)
        saved = fused.model[17].tcr.P.weight.detach().clone()
        with torch.no_grad():
            fused.model[17].tcr.P.weight.add_(.1)
        try:
            pair_invariants(model, fused)
        except RuntimeError:
            results["changed_P_rejected"] = True
        finally:
            with torch.no_grad():
                fused.model[17].tcr.P.weight.copy_(saved)
        saved_o = fused.model[17].tcr.O.weight.detach().clone()
        with torch.no_grad():
            fused.model[17].tcr.O.weight.add_(.1)
        try:
            pair_invariants(model, fused)
        except RuntimeError:
            results["changed_O_rejected"] = True
        finally:
            with torch.no_grad():
                fused.model[17].tcr.O.weight.copy_(saved_o)
        forward = fused.model[17].forward
        fused.model[17].forward = Conv.forward_fuse.__get__(fused.model[17])
        try:
            pair_invariants(model, fused)
        except RuntimeError:
            results["missing_TCR_forward_fuse_rejected"] = True
        try:
            early_trace(fused, image, rng_state())
        except RuntimeError:
            results["missing_TCR_execution_rejected"] = True
        finally:
            fused.model[17].forward = forward
        saved_bn = fused.model[20].bn
        del fused.model[20].bn
        try:
            pair_invariants(model, fused)
        except RuntimeError:
            results["missing_LIF_BN_rejected"] = True
        finally:
            fused.model[20].bn = saved_bn
        # An actual upstream perturbation must remain a real mismatch. The
        # ordered trace must locate node 0 before the original lif_input failure.
        with torch.no_grad():
            fused.model[0].conv.weight.add_(.01)
        wrong = _one_comparison(model, fused, image, folder / "upstream_error", rng_state())
        require(not wrong["accepted"], "Real numerical mismatch accepted")
        require(wrong["early"]["first_tolerance_failure"] == "node0.output", "Earliest upstream error misattributed")
        require(wrong["full"]["failure"]["key"] == "lif_input", "Mother lif_input check was bypassed")
        results["real_upstream_error_retained"] = dict(first=wrong["early"]["first_tolerance_failure"],
                                                       full=wrong["full"], accepted=False)

    # Admission explicitly separates strict-only from runtime certification.
    gate = dict(status="PASS", capacity_status="PASS", effective_updates=2, post_O_P_gradient=True,
                original_gradients=True, fusion_trainer_unchanged=True,
                fusion=dict(strict_accepted=True, runtime_accepted=True))
    require(preflight_accepted(gate), "Completed strict/runtime capacity fixture rejected")
    gate["fusion"]["runtime_accepted"] = False
    require(not preflight_accepted(gate), "Runtime failure mislabeled PASS")
    gate["status"] = "PASS_STRICT_FP32_ONLY"
    require(preflight_accepted(gate), "Strict-only status scope incorrect")
    gate["fusion"]["strict_accepted"] = False
    require(not preflight_accepted(gate), "Strict failure admitted")
    gate["fusion"]["strict_accepted"] = True
    gate["effective_updates"] = 1
    require(not preflight_accepted(gate), "Incomplete real capacity admitted")
    results["gate_scope_and_strict_failure_rejection"] = True

    before, rng = precision_state(), rng_hash(rng_state())
    with patch("tcr_v1_fusion.fusion_protocol", side_effect=RuntimeError("injected protocol error")):
        try:
            diagnose(model, image, folder / "exception_cleanup")
        except RuntimeError as error:
            require("injected protocol error" in str(error), "Unexpected cleanup failure")
            results["protocol_exception_propagated"] = True
    require(precision_state() == before and rng_hash(rng_state()) == rng, "Protocol failure leaked precision or RNG")
    require(all(results.get(k) for k in ("changed_P_rejected", "changed_O_rejected", "missing_TCR_forward_fuse_rejected",
            "missing_TCR_execution_rejected", "missing_LIF_BN_rejected", "protocol_exception_propagated")), "Negative regression incomplete")
    results["exception_precision_rng_restored"] = True
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--negative", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    report = dict(version=VERSION, status="FAILED", runtime=runtime(), source=source_identity(),
                  scope=f"local B1/{args.size} inference fusion, no B16/640 server capacity or formal training",
                  server_preflight="PENDING", formal_training="NOT_RUN")
    try:
        report["context"] = context_checks()
        initialized = torch_load(INIT, map_location="cpu")["model"]
        model, loading = native_rebuild(initialized.yaml, initialized, dict(nc=1, channels=3))
        report["loading"] = loading
        model = model.to(args.device)
        model.nc = 1
        # Exercise nontrivial BN statistics with two finite, no-gradient forwards.
        model.train()
        with torch.no_grad():
            model(torch.rand(2, 3, 160, 160, device=args.device))
            model(torch.rand(2, 3, 160, 160, device=args.device))
            model.model[17].tcr.O.weight.normal_(0, .01)
        model.eval().float()
        image = torch.rand(1, 3, args.size, args.size, device=args.device)
        source, before_rng = model_identity(model), rng_hash(rng_state())
        report["fusion"] = diagnose(model, image, args.output / "fusion")
        require(report["fusion"]["strict_accepted"], "Strict FP32 fusion regression failed")
        require(model_identity(model) == source and rng_hash(rng_state()) == before_rng, "Caller weights/BN/RNG mutated")
        if args.negative:
            report["negative"] = rejection_checks(model, image, args.output)
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        report["tf32_hardware"] = dict(capability=capability,
            locally_exercises_tf32=bool(args.device.startswith("cuda") and capability and capability[0] >= 8),
            server_causal_confirmation="PENDING; replay saved server fixture on RTX4090")
        report["status"] = "PASS"
    finally:
        write_json(args.output / "checks.json", report)
    print(json.dumps(dict(status=report["status"], fusion=report["fusion"]["status"], output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
