"""Create a clean FP32 CSCEF-v5.1 initialization from the hash-verified C2 checkpoint (no training)."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics/cfg/models/rt-detr"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
V51_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v51.yaml"
DEFAULT_SOURCE = Path("/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt")
DEFAULT_OUTPUT = ROOT / "weights/rtdetr_r18_lite_cscef_v51_imagenet_backbone_init.pt"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
V5_TRAIN_COMMIT = "fad58b01eb813c5cea7f9fc2c36c93a5a4d388e7"
NEW_SUFFIXES = {
    "lateral_projection.weight", "semantic_projection.weight", "mix_projection.weight",
    "depthwise_conv.weight", "output_projection.weight", "scharr_x", "scharr_y",
}
CLASSIFICATION_KEYS = [
    "model.26.denoising_class_embed.weight", "model.26.enc_score_head.weight",
    *(f"model.26.dec_score_head.{i}.weight" for i in range(3)),
]
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics import RTDETR  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: E402
from ultralytics.nn.modules import CSCEFv51  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402

from init_rtdetr_r18_lite_cscef_v3_from_baseline import (  # noqa: E402
    build_strict_mapping, checkpoint_model, remap_baseline_key, sha256,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def runtime_info() -> dict:
    imported = Path(ultralytics.__file__).resolve()
    require(imported.is_relative_to(ULTRALYTICS_ROOT.resolve()), "Ultralytics imported outside this worktree.")
    audited_files = [
        "ultralytics-main/ultralytics/nn/modules/cscef_v51.py",
        "ultralytics-main/ultralytics/nn/modules/cscef_v5.py",
        "ultralytics-main/ultralytics/nn/modules/__init__.py",
        "ultralytics-main/ultralytics/nn/tasks.py",
        "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-v51.yaml",
        "tools/init_rtdetr_r18_lite_cscef_v51_controlled.py", "tools/audit_rtdetr_r18_lite_cscef_v51.py",
        "tools/train_rtdetr_r18_lite_cscef_v51.py",
        "tools/init_rtdetr_r18_lite_cscef_v3_from_baseline.py",
        "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite.yaml",
        "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-v5.yaml",
        "ultralytics-main/ultralytics/models/rtdetr/train.py",
        "ultralytics-main/ultralytics/engine/trainer.py",
        "ultralytics-main/ultralytics/engine/model.py",
        "ultralytics-main/ultralytics/cfg/__init__.py",
        "ultralytics-main/ultralytics/cfg/default.yaml",
    ]
    return {
        "ultralytics_file": str(imported), "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__, "python": sys.version, "cuda_available": torch.cuda.is_available(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                        for name in audited_files},
    }


def write_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_checkpoint(checkpoint: dict) -> bool:
    return checkpoint.get("epoch") == -1 and all(
        checkpoint.get(key) is None for key in ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results")
    )


def read_source(source: Path) -> tuple[dict, torch.nn.Module]:
    if not source.is_file():
        raise FileNotFoundError(f"Required original C2 initialization: {source}")
    require(sha256(source) == SOURCE_SHA256, "C2 initialization SHA256 mismatch; refusing an unverified source.")
    checkpoint = torch_load(source, map_location="cpu")
    require(clean_checkpoint(checkpoint), "C2 source is not a clean initialization checkpoint.")
    return checkpoint, checkpoint_model(checkpoint)


def tensor_rows(source: dict, target: dict, mapping: dict) -> list[dict]:
    rows = []
    for key, target_key in mapping.items():
        a, b = source[key].detach().cpu(), target[target_key].detach().cpu()
        same_shape = a.shape == b.shape
        rows.append({
            "baseline": key, "target": target_key, "baseline_shape": list(a.shape), "target_shape": list(b.shape),
            "equal": same_shape and torch.equal(a, b),
            "max_abs": float((a.double() - b.double()).abs().max()) if same_shape else None,
        })
    return rows


def verify_module(model: torch.nn.Module) -> None:
    require(type(model.model[18]) is CSCEFv51, "Expected CSCEFv51 at layer 18.")
    module = model.model[18]
    require(sum(p.numel() for p in module.parameters()) == 26912, "Unexpected V51 parameter count.")
    require(set(module.state_dict()) == NEW_SUFFIXES, "Unexpected V51 state keys.")
    require(len(list(module.parameters())) == 5 and len(list(module.buffers())) == 2,
            "Expected five trainable weights and two Scharr buffers.")
    require(torch.count_nonzero(module.output_projection.weight).item() == 0, "Output projection is not exactly zero.")


def verify_reloads(output: Path, expected: dict) -> dict:
    """Check raw checkpoint, direct API and YAML.load API with no storage quantization tolerance."""
    checkpoint = torch_load(output, map_location="cpu")
    require(clean_checkpoint(checkpoint), "Saved checkpoint retained training state.")
    raw = checkpoint_model(checkpoint)
    direct = RTDETR(str(output)).model
    yaml_loaded = RTDETR(str(V51_CFG)).load(str(output)).model
    report = {}
    for name, model in (("raw", raw), ("RTDETR(checkpoint)", direct), ("RTDETR(YAML).load(checkpoint)", yaml_loaded)):
        verify_module(model)
        state = model.state_dict()
        require(set(state) == set(expected), f"State keys changed on {name} reload.")
        differences = [key for key in expected if not torch.equal(expected[key], state[key])]
        require(not differences, f"Inexact {name} reload: {differences}")
        report[name] = {"states_exact": len(state), "differences": differences, "output_projection_zero": True}
    return report


def initialize(source: Path, output: Path) -> dict:
    """Strictly map nc=80 base states; keep new FP32 module states and verify the saved file."""
    source, output = source.resolve(), output.resolve()
    require(source != output and not output.exists(), f"Refusing to overwrite existing checkpoint: {output}")
    report = {"runtime": runtime_info(), "source": str(source), "source_sha256": SOURCE_SHA256,
              "base_commit": BASE_COMMIT, "v5_train_commit": V5_TRAIN_COMMIT, "seed": 42,
              "storage": "FP32; original C2 half values promoted exactly, new branch has no half quantization"}
    _, source_model = read_source(source)
    source_state = source_model.state_dict()
    # Control construction locally. The CLI, including random reload checks, must run outside training.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        baseline = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
        torch.random.default_generator.manual_seed(42)
        target = RTDETRDetectionModel(str(V51_CFG), ch=3, nc=80, verbose=False).eval()
    plan = build_strict_mapping(source_state, baseline.state_dict(), target.state_dict())
    require(len(plan.state) == 533, "Expected exactly 533 common base states.")
    require(set(plan.new_target_keys) == {f"model.18.{s}" for s in NEW_SUFFIXES}, "Unexpected unmapped target states.")
    complete = {**target.state_dict(), **plan.state}
    target.load_state_dict(complete, strict=True)
    verify_module(target)
    rows = tensor_rows(source_state, target.state_dict(), plan.source_to_target)
    require(all(row["equal"] for row in rows), "Mapped C2 values changed.")
    backbone = {k: p for k, p in target.named_parameters() if any(k.startswith(f"model.{i}.") for i in range(8))}
    require(len(backbone) == 69 and sum(p.numel() for p in backbone.values()) == 11199968,
            "Unexpected backbone parameter coverage.")
    require(set(backbone).issubset(plan.state), "Backbone mapping is incomplete.")
    target.args = {**DEFAULT_CFG_DICT, "model": str(V51_CFG), "task": "detect"}
    target.task = "detect"
    target.pt_path = str(output)
    checkpoint = {
        "epoch": -1, "best_fitness": None, "model": deepcopy(target).float(), "ema": None,
        "updates": None, "optimizer": None, "scaler": None, "train_args": target.args,
        "train_metrics": None, "train_results": None, "date": datetime.now(timezone.utc).isoformat(),
        "version": ultralytics.__version__, "license": "AGPL-3.0", "cscef_v51_provenance": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        torch.save(checkpoint, file)
    report.update({"output": str(output), "output_sha256": sha256(output), "mapped_states": len(rows),
                   "common_tensor_audit": rows, "new_states": 7, "new_parameters": 26912,
                   "backbone_parameter_keys": 69, "backbone_parameter_numel": 11199968,
                   "clean_training_state": True, "reloads": verify_reloads(output, target.state_dict()),
                   "status": "passed"})
    require(sha256(source) == SOURCE_SHA256, "Source checkpoint changed during initialization.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New path; existing files are never overwritten.")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/cscef_v51/initialization.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = initialize(args.source, args.output)
    write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "common_tensor_audit"}, indent=2))
    print(f"Full initialization audit: {args.report.resolve()}")


if __name__ == "__main__":
    main()
