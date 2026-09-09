"""Strict C2 public initialization with only RSC-Head added; no trained candidate weights."""
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
from ultralytics.nn.modules import RSCHead, RTDETRDecoderRSC
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load

MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
C2_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
BASE_COMMIT = "67c3078e54a657fd96d65fee657a75fbb1dae0d6"
VARIANTS = {"rsc_head": ("rtdetr-resnet18-lite-rsc-head.yaml", "rtdetr-resnet18-lite.yaml",
                           "rsc_head_rtdetr_r18_lite_e200_b16_onlineaug")}


def is_added(key):
    return key.startswith(tuple("model.26.dec_score_head.2." + n + "." for n in ("sem_proj", "geo_proj", "mod_proj")))


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
                ultralytics=ultralytics.__file__, rsc_head_module=__import__(RSCHead.__module__, fromlist=["__file__"]).__file__, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())


def build(variant="rsc_head", nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant="rsc_head", zero=False):
    from ultralytics.utils import YAML
    require(variant == "rsc_head", "RSC-Head-only entry")
    expected = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected["head"][-1][2] = "RTDETRDecoderRSC"
    require(type(model) is RTDETRDetectionModel, "Native RTDETRDetectionModel required")
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "Only the final C2 Decoder container may change")
    require(sum(type(layer) is RSCHead for layer in model.modules()) == 1, "Exactly one RSCHead required")
    require(len(model.model) == 27, "C2 topology changed")
    head = model.model[-1]
    m = head.dec_score_head[2]
    require(type(head) is RTDETRDecoderRSC and head.f == [19, 22, 25], "Decoder changed")
    require(type(m) is RSCHead and all(type(h) is torch.nn.Linear for h in head.dec_score_head[:2]), "Only final classification head may change")
    require(head.decoder.num_layers == 3 and head.decoder.eval_idx == 2 and head.num_queries == 300, "Decoder semantics changed")
    require(head.num_denoising == 100 and head.label_noise_ratio == .5 and head.box_noise_scale == 1., "DN changed")
    require(sum(p.numel() for name, p in model.named_parameters() if is_added(name)) == 8736, "Wrong RSC-Head increment")
    if head.nc == 1:
        require(sum(p.numel() for p in model.parameters()) == 20091508, "Wrong unfused nc1 count")
    if zero:
        require(all(torch.count_nonzero(p) == 0 for p in m.mod_proj.parameters()), "mod_proj must initialize zero")
        require(all(torch.count_nonzero(h.weight) > 0 and torch.count_nonzero(h.bias) == 0 for h in (m.sem_proj, m.geo_proj)), "Conditional mapping initialization")


def controlled_models(source, variant="rsc_head"):
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
    c2 = build("rsc_head", baseline=True).state_dict()
    require(set(source_state) == set(c2), "Missing/unexpected C2 source states")
    require(all(v.shape == c2[k].shape for k, v in source_state.items()), "C2 source shape mismatch")
    mapping = {k: k for k in source_state}
    mapped = {mapping[k]: v for k, v in source_state.items()}
    expected_extra_public = set()
    require(set(public) - set(mapped) == expected_extra_public and not set(mapped) - set(public), "Unexpected C2 mapping gap")
    require(all(v.shape == public[k].shape for k, v in mapped.items()), "Mapped shape mismatch")
    reference.load_state_dict({**public, **mapped}, strict=True)
    expected_rsc_head = {k for k in new if is_added(k)}
    require(set(new) - set(public) == expected_rsc_head and not set(public) - set(new), "Unexpected RSC-Head mapping gap")
    target.load_state_dict({**new, **reference.state_dict()}, strict=True)
    rows = [{"source": k, "target": mapping[k], "shape": list(v.shape),
             "equal": torch.equal(v, target.state_dict()[mapping[k]])} for k, v in source_state.items()]
    require(all(r["equal"] for r in rows), "Public loading changed values")
    verify_model(target, variant, zero=True)
    report = dict(source=str(source.resolve()), source_sha256=SOURCE_SHA256, c2_commit=C2_COMMIT,
                  variant=variant, source_nc=original.model[-1].nc, target_nc=target.model[-1].nc, seed=42, source_to_target=rows, source_states=len(rows), missing_source=[],
                  unexpected_source=[], shape_mismatches=[], new_rsc_head_states=sorted(expected_rsc_head),
                  new_rsc_head_parameters=8736,
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
                      license="AGPL-3.0", rsc_head_provenance=report)
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
