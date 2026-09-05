"""Create a clean FP32 C16 initialization from the exact original C2 checkpoint; no training."""
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
V2_CFG = MODEL_DIR / "rtdetr-resnet18-lite-gsdr-aifi-v2.yaml"
SERVER_MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
DEFAULT_SOURCE = SERVER_MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_OUTPUT = ROOT / "weights/rtdetr_r18_lite_gsdr_aifi_v2_imagenet_backbone_init.pt"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
C14_COMMIT = "d8954374f7564dc6180424e9058b44403968308e"
SPARSE_PREFIX = "model.9.sparse_relation."
PROTECTION = ROOT / "docs/gsdr_aifi_v2_protected.json"
AMP_ASSET = ULTRALYTICS_ROOT / "ultralytics/assets/bus.jpg"
AMP_ASSET_SHA256 = "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import GSDRAIFIV2
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load
from init_rtdetr_r18_lite_gsdr_aifi_controlled import (
    build_strict_mapping, checkpoint_model, original_aifi_keys, sha256,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_hash(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_protected():
    manifest = json.loads(PROTECTION.read_text(encoding="utf-8"))
    actual = {name: canonical_hash((ROOT / name).read_bytes()) for name in manifest["files"]}
    require(actual == manifest["files"], "Protected C2/C14 source changed; restore the independent V2 checkout.")
    require(AMP_ASSET.is_file() and sha256(AMP_ASSET) == AMP_ASSET_SHA256, "Native AMP self-check bus.jpg is missing/changed.")
    return {"reference_commit": C14_COMMIT, "canonical_lf_sha256": actual, "status": "passed"}


def runtime_info():
    imported = Path(ultralytics.__file__).resolve()
    require(imported.is_relative_to(ULTRALYTICS_ROOT.resolve()), "Ultralytics imported outside this V2 worktree.")
    # Include the launcher, tests, old helper dependencies, and all actual package code/configuration.
    files = [p for folder in (ULTRALYTICS_ROOT / "ultralytics", ULTRALYTICS_ROOT / "tests", ROOT / "tools")
             for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml"}]
    files.append(PROTECTION)
    return {
        "ultralytics_file": str(imported), "ultralytics_version": ultralytics.__version__,
        "python": sys.version, "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "amp_asset_sha256": sha256(AMP_ASSET),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "code_sha256": {p.relative_to(ROOT).as_posix(): canonical_hash(p.read_bytes()) for p in sorted(files)},
    }


def clean_checkpoint(checkpoint):
    return checkpoint.get("epoch") == -1 and all(checkpoint.get(k) is None for k in
        ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results"))


def read_source(source):
    if not source.is_file():
        raise FileNotFoundError(f"Required original C2 initialization: {source}")
    require(sha256(source) == SOURCE_SHA256, "C2 source SHA256 mismatch; trained checkpoints are forbidden.")
    checkpoint = torch_load(source, map_location="cpu")
    require(clean_checkpoint(checkpoint), "C2 source is not a clean initialization.")
    return checkpoint, checkpoint_model(checkpoint)


def tensor_rows(source, target):
    rows = []
    for key, value in source.items():
        other = target.get(key)
        equal = other is not None and value.shape == other.shape and torch.equal(value.cpu(), other.cpu())
        rows.append({"key": key, "shape": list(value.shape), "equal": equal})
    return rows


def verify_module(model):
    module = model.model[9]
    require(type(module) is GSDRAIFIV2, "Expected GSDRAIFIV2 at layer 9.")
    relation = module.sparse_relation
    require((module.ma.embed_dim, module.fc1.out_features, module.ma.num_heads) == (256, 1024, 8),
            "Original AIFI dimensions changed.")
    require((relation.aux_dim, relation.aux_heads, relation.offset_groups, relation.sample_stride,
             relation.offset_range_factor, relation.offset_kernel) == (128, 4, 4, 2, 2.0, 3), "Sparse structure changed.")
    require(len(relation.state_dict()) == 31 and sum(p.numel() for p in relation.parameters()) == 117644,
            "Unexpected sparse state/parameter count.")
    require(not hasattr(relation, "relative_bias"), "Forbidden relative_bias module alias.")
    boundaries = [relation.output_proj, relation.offset_out, *(m.fc2 for m in relation.position_mlp)]
    require(all(torch.count_nonzero(p).item() == 0 for m in boundaries for p in m.parameters()),
            "A required zero boundary is nonzero.")


def verify_reloads(output, expected):
    checkpoint = torch_load(output, map_location="cpu")
    require(clean_checkpoint(checkpoint), "Checkpoint contains training state.")
    raw = checkpoint["model"]
    require(all(p.dtype == torch.float32 for p in raw.parameters()), "Saved parameters must be FP32.")
    result = {}
    for label, model in (("raw", raw), ("RTDETR(checkpoint)", RTDETR(str(output)).model),
                         ("RTDETR(YAML).load(checkpoint)", RTDETR(str(V2_CFG)).load(str(output)).model)):
        verify_module(model)
        require(set(model.state_dict()) == set(expected), f"{label}: state keys differ.")
        rows = tensor_rows(expected, model.state_dict())
        require(all(r["equal"] for r in rows), f"{label}: exact reload failed.")
        result[label] = {"states_exact": len(rows), "dtype": "torch.float32", "status": "passed"}
    return result


def initialize(source, output):
    source, output = source.resolve(), output.resolve()
    require(source != output and not output.exists(), f"Refusing checkpoint overwrite: {output}")
    verify_protected()
    _, source_model = read_source(source)
    # Seed once, then let the original AIFI and complete model constructors run normally.
    torch.manual_seed(42)
    baseline = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
    torch.manual_seed(42)
    target = RTDETRDetectionModel(str(V2_CFG), ch=3, nc=80, verbose=False).eval()
    plan = build_strict_mapping(source_model.state_dict(), baseline.state_dict(), target.state_dict())
    require(len(plan.state) == 533 and len(plan.new_target_keys) == 31, "Unexpected strict mapping coverage.")
    target.load_state_dict({**target.state_dict(), **plan.state}, strict=True)
    verify_module(target)
    rows = tensor_rows(source_model.state_dict(), target.state_dict())
    require(all(r["equal"] for r in rows), "C2 source tensor values changed.")
    aifi = original_aifi_keys(source_model.state_dict())
    backbone = [k for k, _ in target.named_parameters() if k.startswith(tuple(f"model.{i}." for i in range(8)))]
    require(len(aifi) == 12 and len(backbone) == 69 and set(aifi + backbone) <= set(plan.state),
            "Original AIFI or backbone mapping incomplete.")
    provenance = {"source": str(source), "source_sha256": SOURCE_SHA256, "runtime": runtime_info(),
                  "c2_commit": BASE_COMMIT, "c14_commit": C14_COMMIT, "seed": 42,
                  "storage": "FP32; C2 values exactly promoted; new states never half-quantized"}
    target.args = {**DEFAULT_CFG_DICT, "model": str(V2_CFG), "task": "detect"}
    target.task, target.pt_path = "detect", str(output)
    checkpoint = {
        "epoch": -1, "best_fitness": None, "model": deepcopy(target).float(), "ema": None, "updates": None,
        "optimizer": None, "scaler": None, "train_args": target.args, "train_metrics": None,
        "train_results": None, "date": datetime.now(timezone.utc).isoformat(), "version": ultralytics.__version__,
        "license": "AGPL-3.0", "gsdr_aifi_v2_provenance": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        torch.save(checkpoint, file)
    reloads = verify_reloads(output, target.state_dict())
    require(sha256(source) == SOURCE_SHA256, "Source changed during initialization.")
    return {**provenance, "output": str(output), "output_sha256": sha256(output), "common_tensor_audit": rows,
            "mapped_states": 533, "new_states": 31, "original_aifi_keys": aifi, "backbone_parameter_keys": backbone,
            "new_parameters": 117644, "clean_training_state": True, "reloads": reloads, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Exact original C2 checkpoint; SHA locked.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New FP32 checkpoint; never overwrite.")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/gsdr_aifi_v2/initialization.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = initialize(args.source, args.output)
    write_json(args.report, report)
    print(f"Initialization passed: {args.output.resolve()}\nSHA256: {report['output_sha256']}\nReport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
