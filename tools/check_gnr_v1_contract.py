"""Reproduce the GNR document/mother VFL contract check without training.

Exit 2 means the corrected GNR v1 parameter contract differs from the mother.
The original conflict report is retained under docs/gnr_v1/history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
PACKAGE = ROOT / "ultralytics-main"
MODEL = "ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"
sys.path.insert(0, str(PACKAGE))


def audit():
    import torch
    import yaml
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import RTDETRDetectionModel
    from ultralytics.utils.loss import VarifocalLoss

    source_files = [
        "ultralytics/nn/tasks.py", "ultralytics/models/utils/loss.py",
        "ultralytics/utils/loss.py", "ultralytics/models/rtdetr/train.py",
        "ultralytics/models/yolo/detect/train.py", "ultralytics/nn/modules/cbr.py",
        "ultralytics/nn/modules/lif_down.py", "ultralytics/nn/modules/transformer.py", MODEL,
    ]
    hashes = {}
    for relative in source_files:
        path = PACKAGE / relative
        tracked = path.relative_to(ROOT).as_posix()
        pinned = subprocess.check_output(["git", "show", f"{BASE}:{tracked}"], cwd=ROOT, timeout=30)
        current = path.read_bytes().replace(b"\r\n", b"\n")
        if current != pinned.replace(b"\r\n", b"\n"):
            raise RuntimeError(f"Audit requires the unmodified mother source: {tracked}")
        hashes[tracked] = hashlib.sha256(current).hexdigest()

    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = RTDETRDetectionModel(str(PACKAGE / MODEL), nc=1, verbose=False)
        # Invoke the real Trainer method that normally attaches nc before loss.
        trainer_context = SimpleNamespace(model=model, data={"nc": 1, "names": {0: "crack"}}, args=SimpleNamespace())
        DetectionTrainer.set_model_attributes(trainer_context)
        criterion = model.init_criterion()
        head = model.model[-1]
        public_shape_model = RTDETRDetectionModel(str(PACKAGE / MODEL), nc=80, verbose=False)
    old, new = public_shape_model.state_dict(), model.state_dict()
    if set(old) != set(new):
        raise RuntimeError("Unexpected nc conversion state key difference")
    changes = [{"key": k, "nc80": list(old[k].shape), "nc1": list(new[k].shape)}
               for k in sorted(old) if old[k].shape != new[k].shape]

    # Actual mother's VFL autograd, including sigmoid modulation derivatives.
    z = torch.tensor([-20., -8., -2., 0., 2., 8., 20.], requires_grad=True)
    loss = criterion.vfl(z.view(1, -1, 1), torch.zeros(1, len(z), 1), torch.zeros(1, len(z), 1)) * len(z)
    actual = torch.autograd.grad(loss, z)[0]
    p = z.detach().sigmoid()
    softplus = torch.nn.functional.softplus(z.detach())

    def derivative(alpha, gamma):
        return alpha * p.pow(gamma) * (p + gamma * (1 - p) * softplus)

    reference = derivative(criterion.vfl.alpha, criterion.vfl.gamma)
    document = derivative(.75, 2.)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    if not torch.isfinite(actual).all():
        raise RuntimeError("Non-finite mother derivative")
    expected = {"alpha": .25, "gamma": 1.5}
    observed = {"alpha": criterion.vfl.alpha, "gamma": criterion.vfl.gamma}
    cfg = yaml.safe_load((ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml").read_text(encoding="utf-8"))
    return {
        "status": "CONTRACT_CONFLICT" if observed != expected else "CONTRACT_MATCH",
        "created_utc": datetime.now(timezone.utc).isoformat(), "mother_commit": BASE,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "execution_device": "cpu", "cuda_available": torch.cuda.is_available(), "cuda_checks_executed": False},
        "document_vfl": expected, "actual_model_criterion_vfl": observed,
        "standalone_class_defaults": {"alpha": VarifocalLoss().alpha, "gamma": VarifocalLoss().gamma},
        "criterion_constructor": str(inspect.signature(RTDETRDetectionLoss.__init__)),
        "criterion_source": Path(inspect.getfile(type(criterion))).relative_to(ROOT).as_posix(),
        "source_lf_sha256": hashes,
        "model": {"nc": model.nc, "ordinary_queries": head.num_queries, "decoder_layers": head.num_decoder_layers,
                  "parameters_unfused": sum(p.numel() for p in model.parameters()),
                  "last_classifier_type": type(head.dec_score_head[-1]).__name__,
                  "last_classifier_weight_shape": list(head.dec_score_head[-1].weight.shape),
                  "last_classifier_bias_shape": list(head.dec_score_head[-1].bias.shape),
                  "nc_shape_changes": changes},
        "archived_recipe_fields": len(cfg),
        "negative_vfl_derivative_check": {"status": "PASS_CPU", "logits": z.detach().tolist(),
            "actual_autograd": actual.tolist(), "actual_parameter_analytic": reference.tolist(),
            "legacy_document_parameter_analytic": document.tolist(),
            "max_abs_error_actual_parameters": float((actual - reference).abs().max()),
            "max_abs_error_legacy_document_parameters": float((actual - document).abs().max()),
            "rtol": 1e-6, "atol": 1e-7},
        "parameter_resolution": "User-authorized 2026-09-29 erratum: read actual criterion.vfl parameters; original losses unchanged",
        "scope": "mother source/parameter audit; see local_validation.json for GNR integration checks",
        "formal_training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON audit output; never writes model/checkpoint files")
    args = parser.parse_args()
    result = audit()
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 2 if result["status"] == "CONTRACT_CONFLICT" else 0


if __name__ == "__main__":
    sys.exit(main())
