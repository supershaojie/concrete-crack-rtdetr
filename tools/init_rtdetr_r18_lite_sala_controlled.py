"""Create clean FP32 C21 initialization from the SHA-locked original C2 source. No training."""
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
SALA_CFG = MODEL_DIR / "rtdetr-resnet18-lite-sala.yaml"
SERVER_MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
DEFAULT_SOURCE = SERVER_MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_OUTPUT = ROOT / "weights/rtdetr_r18_lite_sala_imagenet_backbone_init.pt"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
PROTECTION = ROOT / "docs/sala_protected.json"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import AIFI, RTDETRDecoderSALA, SALAMSDeformAttn
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, Path): return str(value)
        raise TypeError(f"Unsupported report value: {type(value)}")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n", encoding="utf-8")


def verify_protected():
    manifest = json.loads(PROTECTION.read_text(encoding="utf-8"))
    changed = [n for n, digest in manifest["files"].items()
               if (sha256(ROOT / n) if n.endswith(".jpg") else canonical_hash((ROOT / n).read_bytes())) != digest]
    require(not changed, f"Protected C2 sources changed: {changed}")
    return {"status": "passed", "reference_commit": BASE_COMMIT, "files": manifest["files"]}


def runtime_info():
    imported = Path(ultralytics.__file__).resolve()
    require(imported.is_relative_to(ULTRALYTICS_ROOT.resolve()), "Ultralytics imported outside C21 worktree.")
    files = [p for folder in (ULTRALYTICS_ROOT / "ultralytics", ULTRALYTICS_ROOT / "tests", ROOT / "tools")
             for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}]
    files.append(PROTECTION)
    return {"ultralytics_file": str(imported), "ultralytics_version": ultralytics.__version__,
            "python": sys.version, "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
            "code_sha256": {p.relative_to(ROOT).as_posix(): canonical_hash(p.read_bytes()) for p in sorted(files)}}


def clean_checkpoint(checkpoint):
    return checkpoint.get("epoch") == -1 and all(checkpoint.get(k) is None for k in
        ("best_fitness", "ema", "optimizer", "scaler", "updates", "train_metrics", "train_results"))


def read_source(source):
    require(source.is_file() and sha256(source) == SOURCE_SHA256,
            f"Original C2 initialization missing/SHA mismatch: {source}; trained checkpoints forbidden.")
    checkpoint = torch_load(source, map_location="cpu")
    require(clean_checkpoint(checkpoint), "C2 source contains training state.")
    return checkpoint, deepcopy(checkpoint["model"]).float()


def tensor_rows(source, target):
    return [{"key": k, "shape": list(v.shape), "dtype": str(v.dtype),
             "equal": k in target and v.shape == target[k].shape and v.dtype == target[k].dtype
             and torch.equal(v.cpu(), target[k].cpu())} for k, v in source.items()]


def new_state_keys(model):
    return {k for k in model.state_dict() if '.sala_' in k}


def common_rows(source, target):
    return tensor_rows(source, target)


def remap_key(key):
    return key


def branch_modules(model):
    return {n: m for n, m in model.named_modules() if type(m) is SALAMSDeformAttn}


def verify_module(model, require_zero=True):
    head = model.model[-1]
    require(type(head) is RTDETRDecoderSALA, 'Expected RTDETRDecoderSALA head.')
    require(head.num_decoder_layers == 3 and head.num_queries == 300 and head.decoder.eval_idx == 2,
            'C2 Lite decoder configuration changed.')
    require(sum(type(m) is AIFI for m in model.modules()) == 1, 'Original AIFI missing.')
    modules = branch_modules(model)
    require(len(modules) == 3, 'Exactly three SALA cross-attention modules required.')
    require(all(type(layer.cross_attn) is SALAMSDeformAttn for layer in head.decoder.layers), 'Cross-attention not replaced.')
    pointers = []
    for name, module in modules.items():
        require((module.d_model, module.n_levels, module.n_heads, module.n_points) == (256,3,8,4), 'C2 attention shape changed.')
        require(sum(p.numel() for p in module.parameters()) == 214280, 'Complete SALA parameter count differs.')
        added = [p for n,p in module.named_parameters() if n.startswith('sala_')]
        require(sum(p.numel() for p in added) == 8680, 'Added parameter count differs.')
        pointers.extend(p.data_ptr() for p in module.parameters())
        if require_zero:
            require(torch.count_nonzero(module.sala_level_head.weight) == 0
                    and torch.count_nonzero(module.sala_level_head.bias) == 0, 'Expected zero SALA output head.')
    require(len(pointers) == len(set(pointers)), 'Decoder attention parameters accidentally shared.')


def verify_reloads(output, expected):
    checkpoint = torch_load(output, map_location="cpu")
    require(clean_checkpoint(checkpoint), "Initialization contains trained state.")
    results = {}
    for name, model in (("raw", checkpoint["model"]), ("RTDETR(checkpoint)", RTDETR(str(output)).model),
                        ("RTDETR(YAML).load(checkpoint)", RTDETR(str(SALA_CFG)).load(str(output)).model)):
        verify_module(model)
        rows = tensor_rows(expected, model.state_dict())
        require(set(expected) == set(model.state_dict()) and all(r["equal"] for r in rows), f"{name}: reload inexact.")
        results[name] = {"status": "passed", "states_exact": len(rows)}
    return results


def initialize(source, output):
    source, output = source.resolve(), output.resolve()
    require(source != output and not output.exists(), f"Refusing overwrite: {output}")
    verify_protected()
    _, source_model = read_source(source)
    torch.manual_seed(42)
    baseline = RTDETRDetectionModel(str(BASE_CFG), nc=80, verbose=False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    target = RTDETRDetectionModel(str(SALA_CFG), nc=80, verbose=False).eval()
    require(torch.equal(rng, torch.get_rng_state()), "Constructor consumed extra CPU RNG.")
    require(all(r["equal"] for r in tensor_rows(baseline.state_dict(), target.state_dict())), "Fresh common states differ.")
    common = source_model.state_dict()
    require(set(common) == set(baseline.state_dict()), "Source is not the complete C2 model.")
    require(all(common[k].shape == baseline.state_dict()[k].shape for k in common), "C2 source shapes differ.")
    extra = set(target.state_dict()) - set(common)
    require(extra == new_state_keys(target), "Unexpected new states.")
    target.load_state_dict({**target.state_dict(), **common}, strict=True)
    rows = tensor_rows(common, target.state_dict())
    require(all(r["equal"] for r in rows), "C2 source values changed.")
    verify_module(target)
    provenance = {"source": str(source), "source_sha256": SOURCE_SHA256, "c2_commit": BASE_COMMIT,
                  "runtime": runtime_info(), "seed": 42, "storage": "FP32; original C2 half values exactly promoted"}
    target.args = {**DEFAULT_CFG_DICT, "model": str(SALA_CFG), "task": "detect"}
    target.task, target.pt_path = "detect", str(output)
    checkpoint = {"epoch": -1, "best_fitness": None, "model": deepcopy(target).float(), "ema": None,
                  "updates": None, "optimizer": None, "scaler": None, "train_args": target.args,
                  "train_metrics": None, "train_results": None, "date": datetime.now(timezone.utc).isoformat(),
                  "version": ultralytics.__version__, "license": "AGPL-3.0", "sala_provenance": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        torch.save(checkpoint, file)
    return {**provenance, "output": str(output), "output_sha256": sha256(output), "common_tensor_audit": rows,
            "new_state_keys": sorted(extra), "new_parameters": sum(p.numel() for n,p in target.named_parameters() if '.sala_' in n),
            "constructor_common_states_and_rng_equal": True, "reloads": verify_reloads(output, target.state_dict()),
            "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/sala/initialization.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"status": "failed"}
    try:
        report = initialize(args.source, args.output)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.report, report)
    print(f"Initialization passed: {args.output}\nSHA256: {report['output_sha256']}")


if __name__ == "__main__":
    main()
