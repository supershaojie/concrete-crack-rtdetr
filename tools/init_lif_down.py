"""Strict C2 public initialization with only LIF-Down added; no trained candidate weights."""
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
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import LIFDown
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load

MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
C2_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
VARIANTS = {"lif_down": ("rtdetr-resnet18-lite-lif-down.yaml", "rtdetr-resnet18-lite.yaml",
                           "lif_down_rtdetr_r18_lite_e200_b16_onlineaug")}


def is_added(key):
    return len(key.split('.')) > 3 and key.split('.')[2] in {'B_proj', 'P', 'U_mix', 'U_dw', 'O_proj'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def runtime():
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "ultralytics-main"), "Wrong ultralytics import")
    module_path = Path(__import__(LIFDown.__module__, fromlist=["__file__"]).__file__).resolve()
    require(module_path == ROOT / "ultralytics-main/ultralytics/nn/modules/lif_down.py", "Wrong LIFDown source")
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, lif_down_module=__import__(LIFDown.__module__, fromlist=["__file__"]).__file__, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())


def build(variant="lif_down", nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant="lif_down", zero=False):
    from ultralytics.utils import YAML
    from lif_down_topology import locate, variant_config
    from ultralytics.nn.modules import Conv
    require(variant == "lif_down", "LIF-Down-only entry")
    baseline = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected = variant_config(baseline)
    require(type(model) is RTDETRDetectionModel, "Native RTDETRDetectionModel required")
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "Only P3-to-P4 Conv may change")
    topology = locate(baseline)
    m = model.model[topology['p3_to_p4']['downsample']]
    require(type(m) is LIFDown and sum(isinstance(v, LIFDown) for v in model.modules()) == 1, "Exactly one LIFDown")
    require(type(model.model[topology['p4_to_p5']['downsample']]) is Conv, "P4-to-P5 must remain Conv")
    require((m.conv.kernel_size, m.conv.stride, m.conv.padding) == ((3,3),(2,2),(1,1)), "Align geometry changed")
    require(sum(p.numel() for name,p in model.named_parameters() if is_added(name)) == 21104, "Wrong LIF delta")
    if model.model[-1].nc == 1:
        require(sum(p.numel() for p in model.parameters()) == 20103876, "Wrong unfused nc1 count")
    if zero:
        require(torch.count_nonzero(m.O_proj.weight) == 0, "O must initialize zero")
        require(all(torch.count_nonzero(v.weight) > 0 for v in (m.B_proj,m.P,m.U_mix,m.U_dw)), "B/P/U must be nonzero")
    return topology


def controlled_models(source, variant="lif_down"):
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Missing or incorrect unified initialization SHA256")
    ckpt = torch_load(source, map_location="cpu")
    require(ckpt.get("epoch") == -1 and all(ckpt.get(k) is None for k in
        ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness")), "Source has training state")
    original = deepcopy(ckpt["model"]).float()
    reference, target = build(variant, baseline=True), build(variant)
    public, new = reference.state_dict(), target.state_dict()
    require(all(k in new and torch.equal(v, new[k]) for k, v in public.items()), "Constructor disturbed public initialization")
    source_state = original.state_dict()
    c2 = build("lif_down", baseline=True).state_dict()
    require(set(source_state) == set(c2), "Missing/unexpected C2 source states")
    require(all(v.shape == c2[k].shape for k, v in source_state.items()), "C2 source shape mismatch")
    mapping = {k: k for k in source_state}
    mapped = {mapping[k]: v for k, v in source_state.items()}
    expected_extra_public = set()
    require(set(public) - set(mapped) == expected_extra_public and not set(mapped) - set(public), "Unexpected C2 mapping gap")
    require(all(v.shape == public[k].shape for k, v in mapped.items()), "Mapped shape mismatch")
    reference.load_state_dict({**public, **mapped}, strict=True)
    expected_lif_down = {k for k in new if is_added(k)}
    require(set(new) - set(public) == expected_lif_down and not set(public) - set(new), "Unexpected LIF-Down mapping gap")
    target.load_state_dict({**new, **reference.state_dict()}, strict=True)
    rows = [{"source": k, "target": mapping[k], "shape": list(v.shape),
             "equal": torch.equal(v, target.state_dict()[mapping[k]])} for k, v in source_state.items()]
    require(all(r["equal"] for r in rows), "Public loading changed values")
    verify_model(target, variant, zero=True)
    report = dict(source=str(source.resolve()), source_sha256=SOURCE_SHA256, c2_commit=C2_COMMIT,
                  variant=variant, source_nc=original.model[-1].nc, target_nc=target.model[-1].nc, seed=42, source_to_target=rows, source_states=len(rows), missing_source=[],
                  unexpected_source=[], shape_mismatches=[], new_lif_down_states=sorted(expected_lif_down),
                  new_lif_down_parameters=21104, COMMON=sorted(source_state), NEW=sorted(expected_lif_down),
                  MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[], ALLOWED_CLASS_ADAPTATION=[],
                  new_initial_values={k: dict(shape=list(v.shape), min=float(v.min()), max=float(v.max()), nonzero=int(torch.count_nonzero(v))) for k,v in target.state_dict().items() if is_added(k)},
                  public_constructor_equal=True, storage="FP32, original half source values exactly promoted")
    return reference, target, report


def initialize(source, output, variant):
    output = Path(output)
    require(not output.exists(), f"Preserve existing initialization: {output}")
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    report["runtime"] = runtime()
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                      date=datetime.now(timezone.utc).isoformat(), version=ultralytics.__version__,
                      license="AGPL-3.0", lif_down_provenance=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        torch.save(checkpoint, f)
    reloaded = RTDETR(str(output)).model
    require(set(target.state_dict()) == set(reloaded.state_dict()) and
            all(torch.equal(v, reloaded.state_dict()[k]) for k, v in target.state_dict().items()), "Reload mismatch")
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True, status="passed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.report, initialize(args.source, args.output, args.variant))
