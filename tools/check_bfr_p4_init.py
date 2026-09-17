"""Finite negative tests for initialization identity; no inference or training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from init_bfr_p4 import VARIANTS, audit_checkpoint, require, sha256, write_json
from ultralytics.utils.patches import torch_load


def rejection(action, expected):
    try:
        action()
    except RuntimeError as error:
        require(expected in str(error), f"Wrong rejection reason: {error}")
        return {"status": "PASSED", "rejected": str(error)}
    raise RuntimeError("Invalid initialization was accepted")


def run(variant, initialized, output):
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "disposable_tampered.pt"
    report = {"status": "RUNNING", "variant": variant, "initialized_sha256": sha256(initialized)}
    try:
        audit = audit_checkpoint(initialized, variant, require_untrained=True)
        report["valid_original"] = {"status": audit["status"], "all_recorded_tensor_hashes_checked":
                                    audit["all_recorded_tensor_hashes_checked"]}
        report["wrong_file_hash"] = rejection(
            lambda: audit_checkpoint(initialized, variant, "0" * 64, require_untrained=True), "SHA256 mismatch")
        other = next(v for v in VARIANTS if v != variant)
        report["wrong_variant"] = rejection(
            lambda: audit_checkpoint(initialized, other, require_untrained=True), "variant provenance mismatch")
        checkpoint = torch_load(initialized, map_location="cpu")
        state = checkpoint["model"].state_dict()
        key = next(row["name"] for row in checkpoint["bfr_p4_provenance"]["COMMON"]
                   if state[row["name"]].is_floating_point() and state[row["name"]].ndim > 1)
        # Preserve all other metadata/zero output states; the common-value audit
        # must independently detect a modified backbone tensor.
        value = state[key].reshape(-1)[0].clone()
        state[key].reshape(-1)[0] += .25
        torch.save(checkpoint, temporary)
        report["modified_common_tensor"] = rejection(
            lambda: audit_checkpoint(temporary, variant, require_untrained=True),
            "tensor values differ from initialization audit")
        report["modified_common_tensor"]["key"] = key
        state[key].reshape(-1)[0] = value
        checkpoint["epoch"] = 0
        torch.save(checkpoint, temporary)
        report["trained_epoch"] = rejection(
            lambda: audit_checkpoint(temporary, variant, require_untrained=True), "contains learned training state")
        require(sha256(initialized) == report["initialized_sha256"], "Original initialization was modified")
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        # Only this exact disposable file can be removed; no recursive cleanup.
        if temporary.is_file():
            temporary.unlink()
        write_json(output / "initialization_negative_checks.json", report)
    print(variant, report["status"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--initialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.variant, args.initialized, args.output)
