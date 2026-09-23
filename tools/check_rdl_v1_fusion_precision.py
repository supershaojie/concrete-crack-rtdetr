"""Bounded precision restoration and preflight boundary tests; no formal training."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from init_c19_lif_v1 import require, write_json
from rdl_v1_fusion import precision_settings, strict_fusion_precision


def scope_checks():
    original = precision_settings()
    rows = []
    try:
        for matmul in ("highest", "high", "medium"):
            for cudnn in (False, True):
                for amp in (False, True):
                    for raise_inside in (False, True):
                        torch.set_float32_matmul_precision(matmul)
                        torch.backends.cudnn.allow_tf32 = cudnn
                        with torch.autocast("cpu", enabled=amp, dtype=torch.bfloat16, cache_enabled=False), \
                             torch.autocast("cuda", enabled=amp and torch.cuda.is_available(), dtype=torch.float16, cache_enabled=False):
                            before = precision_settings()
                            evidence = {}
                            try:
                                with strict_fusion_precision(evidence):
                                    inside = precision_settings()
                                    require(not any(inside[k] for k in ("cudnn_tf32", "matmul_tf32", "cpu_autocast", "cuda_autocast")),
                                            "Strict scope did not disable reduced precision")
                                    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                                        a = torch.ones(8, 8, device=device)
                                        require((a @ a).dtype == torch.float32, "Strict GEMM was autocast")
                                    if raise_inside:
                                        raise ValueError("injected scope failure")
                            except ValueError as error:
                                require(raise_inside and str(error) == "injected scope failure", "Wrong exception")
                            else:
                                require(not raise_inside, "Context suppressed the exception")
                            require(precision_settings() == before == evidence["after"] and evidence["restored"],
                                    "Precision/dtype/cache failed to restore")
                            rows.append(dict(matmul=matmul, cudnn=cudnn, outer_amp=amp, exception=raise_inside,
                                             precision=evidence))
    finally:
        torch.backends.cudnn.allow_tf32 = original["cudnn_tf32"]
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])
    require(precision_settings() == original, "Test changed caller settings")
    return rows


def preflight_boundary():
    import check_rdl_v1
    import check_rdl_v1_ops
    import rdl_v1 as cli
    expected = precision_settings()
    scope = {}
    with strict_fusion_precision(scope):
        pass
    calls = []
    def capacity_probe(prepared):
        require(precision_settings() == expected, "Capacity entered with strict settings still active")
        # Exercise restored CUDA autocast, but explicitly do not label this B16/640 capacity.
        if torch.cuda.is_available():
            x = torch.ones(8, 8, device="cuda", requires_grad=True)
            with torch.cuda.amp.autocast():
                y = x @ x
                require(y.dtype == torch.float16, "Restored AMP did not run")
                inside = precision_settings()
                require(inside["matmul_tf32"] == expected["matmul_tf32"] and inside["cudnn_tf32"] == expected["cudnn_tf32"],
                        "AMP inherited strict TF32 flags")
            y.sum().backward()
            require(torch.isfinite(x.grad).all(), "Small AMP backward failed")
        require(precision_settings() == expected, "AMP probe changed caller settings")
        calls.append(True)
        return dict(status="PASS", scope="TEST_DOUBLE_WITH_SMALL_AMP_PROBE_NOT_B16_640")
    with TemporaryDirectory(prefix="rdl-precision-") as tmp:
        root = Path(tmp)
        write_json(root/"prepared.json", {})
        failed_bytes = b'{"status":"FAIL","error":"original default-TF32 evidence"}\n'
        (root/"preflight.json").write_bytes(failed_bytes)
        result = dict(real_model=dict(status="PASS", fusion_precision=scope))
        with patch.object(cli, "OUT", root), patch.object(cli, "prepared", return_value=dict(source="fixture.pt")), \
             patch.object(cli, "runtime", return_value={}), patch.object(check_rdl_v1, "run_checks", return_value=result), \
             patch.object(check_rdl_v1_ops, "run", return_value=dict(status="PASS")), patch.object(cli, "capacity", side_effect=capacity_probe):
            cli.preflight(SimpleNamespace(local=False, device="cuda:0"))
            report = cli.read(root/"preflight.json")
            preserved = Path(report["previous_report"]["path"])
            require(preserved.read_bytes() == failed_bytes and cli.sha256(preserved) == report["previous_report"]["sha256"],
                    "Default-precision FAIL evidence was overwritten")
            require(len(calls) == 1, "Capacity boundary was not exercised")
            # Restoration failure must stop before capacity, not merely log a warning.
            result["real_model"]["fusion_precision"] = dict(scope, restored=False)
            try:
                cli.preflight(SimpleNamespace(local=False, device="cuda:0"))
            except RuntimeError as error:
                require("restored" in str(error), "Unexpected preflight rejection")
            else:
                raise AssertionError("Preflight accepted failed precision restoration")
            require(len(calls) == 1 and cli.read(root/"preflight.json")["status"] == "FAIL", "Capacity ran after restoration failure")
    return dict(status="PASS", capacity_scope="stub + CUDA 8x8 AMP forward/backward; B16/640 NOT_RUN",
                old_failure_bytes_preserved=True, capacity_blocked_on_restore_failure=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new evidence file")
    rows = scope_checks()
    result = dict(status="PASS", cases=len(rows), scope_cases=rows, preflight_boundary=preflight_boundary())
    write_json(args.output, result)
    print(f"precision restoration PASS ({len(rows)} cases); preflight boundary PASS; formal capacity NOT_RUN")
