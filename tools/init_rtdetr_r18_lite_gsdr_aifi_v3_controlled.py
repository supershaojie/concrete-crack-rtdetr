"""Create a clean FP32 C18 initialization from the exact original C2 checkpoint; no training."""
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
V3_CFG = MODEL_DIR / "rtdetr-resnet18-lite-gsdr-aifi-v3.yaml"
SERVER_MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
DEFAULT_SOURCE = SERVER_MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_OUTPUT = ROOT / "weights/rtdetr_r18_lite_gsdr_aifi_v3_imagenet_backbone_init.pt"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
C14_COMMIT = "d8954374f7564dc6180424e9058b44403968308e"
C16_COMMIT = "30f2da6e4bb91dcae5f761bcafd5993e263197da"
SPARSE_PREFIX = "model.9.sparse_relation."
PROTECTION = ROOT / "docs/gsdr_aifi_v3_protected.json"
AMP_ASSET = ULTRALYTICS_ROOT / "ultralytics/assets/bus.jpg"
AMP_ASSET_SHA256 = "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import GSDRAIFIV3
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
    changed = [name for name in actual if actual[name] != manifest["files"][name]]
    require(not changed, f"Protected C2/C14/C16 source changed: {changed}")
    require(AMP_ASSET.is_file() and sha256(AMP_ASSET) == AMP_ASSET_SHA256, "Native AMP self-check bus.jpg is missing/changed.")
    return {"reference_commit": C16_COMMIT, "canonical_lf_sha256": actual, "status": "passed"}


def runtime_info():
    imported = Path(ultralytics.__file__).resolve()
    require(imported.is_relative_to(ULTRALYTICS_ROOT.resolve()), "Ultralytics imported outside this V3 worktree.")
    # Include the launcher, tests, old helper dependencies, and all actual package code/configuration.
    files = [p for folder in (ULTRALYTICS_ROOT / "ultralytics", ULTRALYTICS_ROOT / "tests", ROOT / "tools")
             for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}]
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
        ("best_fitness", "ema", "optimizer", "scaler", "updates", "train_metrics", "train_results"))


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
        equal = (other is not None and value.shape == other.shape and value.dtype == other.dtype
                 and torch.equal(value.cpu(), other.cpu()))
        rows.append({"key": key, "shape": list(value.shape), "dtype": str(value.dtype), "equal": equal})
    return rows


def verify_module(model, require_zero=True):
    module = model.model[9]
    require(type(module) is GSDRAIFIV3, "Expected GSDRAIFIV3 at layer 9.")
    relation = module.sparse_relation
    require((module.ma.embed_dim, module.fc1.out_features, module.ma.num_heads) == (256, 1024, 8),
            "Original AIFI dimensions changed.")
    require((relation.aux_dim, relation.aux_heads, relation.n_points, relation.radius_cells) == (128, 4, 4, 2.0),
            "Local structure changed.")
    projections = ("input_proj", "value_proj", "offset_proj", "weight_proj", "output_proj")
    expected = {name + ".weight" for name in projections}
    require(set(dict(relation.named_parameters())) == expected, "Expected exactly five projection weights.")
    require(set(relation.state_dict()) == expected | {"anchors"}, "Unexpected local state keys.")
    require(sum(p.numel() for p in relation.parameters()) == 88064, "Unexpected local parameter count.")
    require(all(getattr(relation, name).bias is None for name in projections), "New biases are forbidden.")
    require(relation.token_norm.normalized_shape == (128,) and not relation.token_norm.elementwise_affine
            and relation.token_norm.eps == 1e-5, "Expected per-token non-affine channel LayerNorm.")
    require(torch.equal(relation.anchors.float().cpu(), torch.tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])),
            "Fixed anchors changed.")
    if require_zero:
        require(all(torch.count_nonzero(getattr(relation, name).weight).item() == 0
                    for name in ("output_proj", "offset_proj", "weight_proj")), "A required zero boundary is nonzero.")


def verify_reloads(output, expected):
    checkpoint = torch_load(output, map_location="cpu")
    require(clean_checkpoint(checkpoint), "Checkpoint contains training state.")
    raw = checkpoint["model"]
    require(all(p.dtype == torch.float32 for p in raw.parameters()), "Saved parameters must be FP32.")
    result = {}
    for label, model in (("raw", raw), ("RTDETR(checkpoint)", RTDETR(str(output)).model),
                         ("RTDETR(YAML).load(checkpoint)", RTDETR(str(V3_CFG)).load(str(output)).model)):
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
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    target = RTDETRDetectionModel(str(V3_CFG), ch=3, nc=80, verbose=False).eval()
    require(torch.equal(baseline_rng, torch.get_rng_state()), "Construction changed the CPU RNG.")
    require(all(r["equal"] for r in tensor_rows(baseline.state_dict(), target.state_dict())),
            "Unloaded constructors differ on common states.")
    plan = build_strict_mapping(source_model.state_dict(), baseline.state_dict(), target.state_dict())
    require(set(plan.state) == set(baseline.state_dict()), "Incomplete strict mapping coverage.")
    require(set(plan.new_target_keys) == {SPARSE_PREFIX + k for k in target.model[9].sparse_relation.state_dict()},
            "Unexpected new mapping keys.")
    target.load_state_dict({**target.state_dict(), **plan.state}, strict=True)
    verify_module(target)
    rows = tensor_rows(source_model.state_dict(), target.state_dict())
    require(all(r["equal"] for r in rows), "C2 source tensor values changed.")
    aifi = original_aifi_keys(source_model.state_dict())
    backbone = [k for k, _ in target.named_parameters() if k.startswith(tuple(f"model.{i}." for i in range(8)))]
    require(len(aifi) == 12 and len(backbone) == 69 and set(aifi + backbone) <= set(plan.state),
            "Original AIFI or backbone mapping incomplete.")
    provenance = {"source": str(source), "source_sha256": SOURCE_SHA256, "runtime": runtime_info(),
                  "c2_commit": BASE_COMMIT, "c14_commit": C14_COMMIT, "c16_commit": C16_COMMIT, "seed": 42,
                  "storage": "FP32; C2 values exactly promoted; new states never half-quantized"}
    target.args = {**DEFAULT_CFG_DICT, "model": str(V3_CFG), "task": "detect"}
    target.task, target.pt_path = "detect", str(output)
    checkpoint = {
        "epoch": -1, "best_fitness": None, "model": deepcopy(target).float(), "ema": None, "updates": None,
        "optimizer": None, "scaler": None, "train_args": target.args, "train_metrics": None,
        "train_results": None, "date": datetime.now(timezone.utc).isoformat(), "version": ultralytics.__version__,
        "license": "AGPL-3.0", "gsdr_aifi_v3_provenance": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        torch.save(checkpoint, file)
    reloads = verify_reloads(output, target.state_dict())
    require(sha256(source) == SOURCE_SHA256, "Source changed during initialization.")
    return {**provenance, "output": str(output), "output_sha256": sha256(output), "common_tensor_audit": rows,
            "mapped_states": len(plan.state), "new_states": len(plan.new_target_keys), "new_state_keys": plan.new_target_keys,
            "original_aifi_keys": aifi, "backbone_parameter_keys": backbone,
            "constructor_common_states_and_rng_equal": True,
            "new_parameters": sum(p.numel() for p in target.model[9].sparse_relation.parameters()),
            "clean_training_state": True, "reloads": reloads, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Exact original C2 checkpoint; SHA locked.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New FP32 checkpoint; never overwrite.")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/gsdr_aifi_v3/initialization.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = initialize(args.source, args.output)
    write_json(args.report, report)
    print(f"Initialization passed: {args.output.resolve()}\nSHA256: {report['output_sha256']}\nReport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
