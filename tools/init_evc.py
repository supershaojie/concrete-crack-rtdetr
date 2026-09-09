"""Auditable C2 identity mapping, with native nc=1 adaptation before serialization."""
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
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import AIFI, RTDETRDecoder, RTDETRDecoderEVC, EVCMSDeformAttn
from ultralytics.nn.modules.transformer import MSDeformAttn
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
SOURCE_SHA256 = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
BASE_COMMIT = "beedcfa307e250fb2de47587097c51f9c141123b"
VARIANTS = {"evc_deform": ("rtdetr-resnet18-lite-evc-deform.yaml", "rtdetr-resnet18-lite.yaml",
                            "evc_deform_rtdetr_r18_lite_e200_b16_onlineaug")}
PARAMETERS = {"c2": 20082772, "evc_deform": 20086884}


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


def is_added(key):
    return key.startswith("model.26.decoder.layers.2.cross_attn.evc_")


def build(variant="evc_deform", nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant="evc_deform", zero=False):
    require(variant == "evc_deform", "Only EVC single module is supported")
    expected = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected["head"][-1][2] = "RTDETRDecoderEVC"
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "Not pure C2 + EVC topology")
    head = model.model[-1]
    require(len(model.model) == 27 and type(model.model[9]) is AIFI and type(head) is RTDETRDecoderEVC, "Wrong backbone/AIFI/head")
    require(head.f == [19, 22, 25] and head.num_queries == 300, "Neck/regular queries changed")
    require(head.decoder.num_layers == 3 and head.decoder.eval_idx == 2 and not head.acr, "Wrong decoder setup")
    require(head.num_denoising == 100 and head.label_noise_ratio == .5 and head.box_noise_scale == 1., "DN changed")
    layers = head.decoder.layers
    require([type(l.cross_attn) for l in layers] == [MSDeformAttn, MSDeformAttn, EVCMSDeformAttn], "Replace last cross-attention only")
    require(all(type(l.self_attn) is torch.nn.MultiheadAttention for l in layers), "Self-attention changed")
    evc = layers[-1].cross_attn
    require((evc.d_model, evc.n_heads, evc.n_levels, evc.n_points) == (256, 8, 3, 4), "C2 sampling dimensions changed")
    require(sum(p.numel() for n, p in model.named_parameters() if is_added(n)) == 4112, "Wrong added count")
    require(not any(any(s in type(m).__name__ for s in ("CSCEF", "SCCA", "CBR", "CrackBoundary", "CoverageRelation", "DRA", "RCA"))
                    for m in model.modules()), "Unrequested module enabled")
    if head.nc == 1:
        require(sum(p.numel() for p in model.parameters()) == PARAMETERS[variant], "Wrong unfused nc1 count")
    if zero:
        require(not torch.count_nonzero(evc.evc_a_p) and not torch.count_nonzero(evc.evc_a_s), "Coefficients not zero")
        require(torch.count_nonzero(evc.evc_q) and torch.count_nonzero(evc.evc_v), "Scoring projections zero")


def rebuild_nc1(model):
    trainer = object.__new__(RTDETRTrainer)
    trainer.data = dict(nc=1, channels=3)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return trainer.get_model(cfg=model.yaml, weights=model, verbose=False)


def controlled_models(source, variant="evc_deform"):
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Missing/incorrect authoritative C2 source SHA256")
    ckpt = torch_load(source, map_location="cpu")
    require(ckpt.get("epoch") == -1 and all(ckpt.get(k) is None for k in
            ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness")), "Source contains training state")
    require(type(ckpt["model"].model[-1]) is RTDETRDecoder, "Source is not C2")
    original = ckpt["model"].float().state_dict()
    reference, target = build(baseline=True), build()
    public, new = reference.state_dict(), target.state_dict()
    require(set(original) == set(public), "Missing/unexpected C2 state")
    require(all(v.shape == public[k].shape for k, v in original.items()), "C2 shape mismatch")
    require(all(k in new and torch.equal(v, new[k]) for k, v in public.items()), "Public constructor state changed")
    extra = {k for k in new if is_added(k)}
    require(set(new) - set(public) == extra and not set(public) - set(new), "Unexpected EVC state gap")
    reference.load_state_dict(original, strict=True)
    target.load_state_dict({**new, **original}, strict=True)
    rows = [dict(source=k, target=k, shape=list(v.shape), equal=torch.equal(v, target.state_dict()[k])) for k, v in original.items()]
    require(all(r["equal"] for r in rows), "Public mapping changed values")
    from train_evc import rebuild_audit
    reference1, target1 = rebuild_nc1(reference), rebuild_nc1(target)
    adaptation = rebuild_audit(target, target1, variant)
    require(all(torch.equal(v, target1.state_dict()[k]) for k, v in reference1.state_dict().items()), "C2/EVC nc1 common states differ")
    verify_model(target1, zero=True)
    report = dict(source=str(source.resolve()), source_sha256=SOURCE_SHA256, variant=variant, seed=42,
                  source_to_target=rows, source_states=len(rows), missing_source=[], unexpected_source=[], shape_mismatches=[],
                  new_states=sorted(extra), new_parameters=4112, public_constructor_equal=True,
                  nc1_adaptation=adaptation, nc1_public_equal=True, storage="FP32, source exactly promoted; native nc=1 adaptation",
                  mapping="identity; all common decoder states retained", parameters_unfused=PARAMETERS)
    return reference1, target1, report


def initialize(source, output, variant="evc_deform"):
    output = Path(output)
    require(not output.exists(), f"Preserve existing initialization: {output}")
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    report["runtime"] = runtime()
    checkpoint = dict(epoch=-1, model=deepcopy(target).float(), train_args=target.args,
                      date=datetime.now(timezone.utc).isoformat(), version=ultralytics.__version__, license="AGPL-3.0",
                      evc_provenance=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        torch.save(checkpoint, f)
    reloaded = RTDETR(str(output)).model
    verify_model(reloaded, zero=True)
    require(set(target.state_dict()) == set(reloaded.state_dict()) and
            all(torch.equal(v, reloaded.state_dict()[k]) for k, v in target.state_dict().items()), "Reload mismatch")
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True, status="passed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.report, initialize(args.source, args.output))
