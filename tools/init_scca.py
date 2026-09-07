"""Strict, clean SCCA initialization. Never reads a trained C17 checkpoint."""
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
from ultralytics.nn.modules import SCCAAIFI, CSCEFv51, RTDETRDecoder
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load
from init_rtdetr_r18_lite_cscef_v3_from_baseline import remap_baseline_key

MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
C2_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
C17_COMMIT = "0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139"
VARIANTS = {
    "c24": ("rtdetr-resnet18-lite-scca.yaml", "rtdetr-resnet18-lite.yaml", "c24_rtdetr_r18_lite_scca_e200_b16_onlineaug"),
    "c25": ("rtdetr-resnet18-lite-cscef-scca.yaml", "rtdetr-resnet18-lite-cscef-v51.yaml", "c25_rtdetr_r18_lite_cscef_v51_scca_e200_b16_onlineaug"),
}


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
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())


def build(variant, nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant, zero=False):
    require(type(model.model[9]) is SCCAAIFI, "SCCAAIFI must replace layer 9")
    require(type(model.model[-1]) is RTDETRDecoder, "Other candidate decoder enabled")
    require(model.model[-1].f == ([19, 22, 25] if variant == "c24" else [20, 23, 26]), "Decoder inputs changed")
    module = model.model[9]
    require(module.ma.embed_dim == 256 and module.ma.num_heads == 8 and module.fc1.out_features == 1024, "AIFI recipe changed")
    require(sum(p.numel() for n, p in module.named_parameters() if n.startswith("scca_")) == 65540, "Wrong SCCA parameter increment")
    if variant == "c25":
        require(type(model.model[18]) is CSCEFv51 and model.model[18].f == [17, 16], "C17 CSCEF changed")
        require(model.model[19].f == [16, 18], "C17 Concat changed")
    if zero:
        require(torch.count_nonzero(module.scca_o.weight).item() == 0, "Initial O must be zero")
        require(torch.count_nonzero(module.scca_temperature_raw).item() == 0, "Initial temperature raw must be zero")
        require(all(torch.count_nonzero(m.weight).item() for m in (module.scca_q, module.scca_k, module.scca_v)), "Q/K/V cannot all be zero")


def controlled_models(source, variant):
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
    c2 = build("c24", baseline=True).state_dict()
    require(set(source_state) == set(c2), "Missing/unexpected C2 source states")
    require(all(v.shape == c2[k].shape for k, v in source_state.items()), "C2 source shape mismatch")
    mapping = {k: remap_baseline_key(k) if variant == "c25" else k for k in source_state}
    mapped = {mapping[k]: v for k, v in source_state.items()}
    expected_cscef = {"model.18." + k for k in reference.model[18].state_dict()} if variant == "c25" else set()
    require(set(public) - set(mapped) == expected_cscef and not set(mapped) - set(public), "Unexpected C17 mapping gap")
    require(all(v.shape == public[k].shape for k, v in mapped.items()), "Mapped shape mismatch")
    reference.load_state_dict({**public, **mapped}, strict=True)
    expected_scca = {k for k in new if ".scca_" in k}
    require(set(new) - set(public) == expected_scca and not set(public) - set(new), "Unexpected SCCA mapping gap")
    target.load_state_dict({**new, **reference.state_dict()}, strict=True)
    rows = [{"source": k, "target": mapping[k], "shape": list(v.shape),
             "equal": torch.equal(v, target.state_dict()[mapping[k]])} for k, v in source_state.items()]
    require(all(r["equal"] for r in rows), "Public loading changed values")
    verify_model(target, variant, zero=True)
    report = dict(source=str(source.resolve()), source_sha256=SOURCE_SHA256, c2_commit=C2_COMMIT, c17_commit=C17_COMMIT,
                  variant=variant, seed=42, source_to_target=rows, source_states=len(rows), missing_source=[],
                  unexpected_source=[], shape_mismatches=[], new_scca_states=sorted(expected_scca),
                  retained_c17_states=sorted(expected_cscef), new_scca_parameters=65540,
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
                      license="AGPL-3.0", scca_provenance=report)
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
