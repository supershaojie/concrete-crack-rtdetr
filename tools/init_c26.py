"""C26 only: exact C19 CBR + C24 SCCA, clean identity-mapped C2 initialization."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import torch
from init_scca import ROOT, MODEL_DIR, SOURCE_SHA256, require, runtime, sha256, write_json
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import SCCAAIFI, RTDETRDecoderCBR, CrackBoundaryRefinement
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

C19_COMMIT = "025997e3c51eaf6933534308a95da6ebf97bff53"
C24_COMMIT = "f6e9dfda765046ae7691302cf5ec89d3f76cec5d"
VARIANTS = {"c26": ("rtdetr-resnet18-lite-cbr-scca.yaml", "rtdetr-resnet18-lite.yaml",
                     "c26_rtdetr_r18_lite_cbr_scca_e200_b16_onlineaug")}
PARAMETERS = {"c2": 20082772, "c19": 20128661, "c24": 20148312, "c26": 20194201}
MODELS = dict(c2="rtdetr-resnet18-lite.yaml", c19="rtdetr-resnet18-lite-cbr.yaml",
              c24="rtdetr-resnet18-lite-scca.yaml", c26=VARIANTS["c26"][0])
MODULE_HASHES = {"cbr.py": "d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787",
                 "scca_aifi.py": "67b0d347d5305c16d48bb031cd10fd1d23b7ea447b58b2423166eb68b81dedca"}


def verify_sources():
    actual = {name: hashlib.sha256((ROOT / "ultralytics-main/ultralytics/nn/modules" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for name in MODULE_HASHES}
    require(actual == MODULE_HASHES, "C19/C24 original module source changed (canonical LF SHA256)")
    return actual


def is_added(key):
    return key.startswith("model.9.scca_") or key.startswith("model.26.cbr.")


def build(variant="c26", nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / MODELS["c2" if baseline else variant]), nc=nc, verbose=False)


def verify_model(model, variant="c26", zero=False):
    require(variant == "c26", "C26 validator only")
    expected = YAML.load(MODEL_DIR / MODELS["c2"])
    expected["head"][1][2] = "SCCAAIFI"
    expected["head"][-1][2] = "RTDETRDecoderCBR"
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "C26 YAML topology changed")
    require(len(model.model) == 27, "C26 must have no inserted layers")
    scca, head = model.model[9], model.model[-1]
    require(type(scca) is SCCAAIFI and type(head) is RTDETRDecoderCBR, "Wrong C26 modules")
    require(head.f == [19, 22, 25] and type(head.cbr) is CrackBoundaryRefinement, "CBR must use Neck P3/P4/P5")
    require(scca.ma.embed_dim == 256 and scca.ma.num_heads == 8 and scca.fc1.out_features == 1024, "AIFI arguments changed")
    require(head.decoder.num_layers == 3 and head.decoder.eval_idx == 2 and head.num_queries == 300,
            "Decoder layers/queries changed")
    require(head.num_denoising == 100 and head.label_noise_ratio == .5 and head.box_noise_scale == 1., "DN changed")
    require(head.cbr.rho == .1 and head.cbr.normal_fraction == .1 and head.cbr.p3_proj.in_channels == 256,
            "CBR original settings changed")
    require(not any(any(x in type(m).__name__ for x in ("CSCEF", "CoverageRelation", "SALA")) for m in model.modules()),
            "Unrequested candidate in C26")
    scca_count = sum(p.numel() for n, p in scca.named_parameters() if n.startswith("scca_"))
    cbr_count = sum(p.numel() for p in head.cbr.parameters())
    require((scca_count, cbr_count) == (65540, 45889), "Original module parameter increment changed")
    if head.nc == 1:
        require(sum(p.numel() for p in model.parameters()) == PARAMETERS["c26"], "C26 unfused nc1 parameter count changed")
    if zero:
        require(not torch.count_nonzero(scca.scca_o.weight) and not torch.count_nonzero(scca.scca_temperature_raw),
                "SCCA original zero initialization changed")
        require(all(torch.count_nonzero(m.weight) for m in (scca.scca_q, scca.scca_k, scca.scca_v)), "SCCA Q/K/V zero")
        require(not torch.count_nonzero(head.cbr.offset_out.weight) and not torch.count_nonzero(head.cbr.offset_out.bias),
                "CBR original zero initialization changed")


def controlled_models(source, variant="c26"):
    module_hashes = verify_sources()
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Missing/incorrect unified C2 source SHA256")
    ckpt = torch_load(source, map_location="cpu")
    require(ckpt.get("epoch") == -1 and all(ckpt.get(k) is None for k in
            ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness")), "Source has training state")
    original = ckpt["model"].float().state_dict()
    reference, target = build(baseline=True), build()
    public, new = reference.state_dict(), target.state_dict()
    require(set(original) == set(public), "Unexpected/missing C2 state")
    require(all(v.shape == public[k].shape for k, v in original.items()), "C2 shape mismatch")
    require(all(k in new and torch.equal(v, new[k]) for k, v in public.items()), "Constructor disturbed public RNG/state")
    extra = {k for k in new if is_added(k)}
    require(set(new) - set(public) == extra and not set(public) - set(new), "Unexpected C26 state gap")
    original_added = {k: new[k].clone() for k in extra}
    reference.load_state_dict(original, strict=True)
    target.load_state_dict({**new, **original}, strict=True)
    loaded = target.state_dict()
    require(all(torch.equal(loaded[k], v) for k, v in original_added.items()), "New module initialization changed during mapping")
    rows = [dict(source=k, target=k, shape=list(v.shape), equal=torch.equal(v, loaded[k])) for k, v in original.items()]
    require(all(r["equal"] for r in rows), "Identity mapping changed source values")
    verify_model(target, variant, zero=True)
    report = dict(source=str(source.resolve()), source_sha256=SOURCE_SHA256, c19_commit=C19_COMMIT, c24_commit=C24_COMMIT,
                  variant=variant, seed=42, module_source_sha256=module_hashes, source_to_target=rows, source_states=len(rows), total_states=len(new),
                  missing_source=[], unexpected_source=[], shape_mismatches=[], new_states=sorted(extra),
                  new_scca_states=sorted(k for k in extra if ".scca_" in k), new_cbr_states=sorted(k for k in extra if ".cbr." in k),
                  new_scca_parameters=65540, new_cbr_parameters=45889, public_constructor_equal=True,
                  new_initialization_preserved=True, mapping="identity, no C25 layer shift", storage="FP32; source values exactly promoted")
    return reference, target, report


def initialize(source, output, variant="c26"):
    output = Path(output)
    require(not output.exists(), f"Preserve existing initialization: {output}")
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / MODELS["c26"]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    report["runtime"] = runtime()
    checkpoint = dict(epoch=-1, model=deepcopy(target).float(), train_args=target.args,
                      date=datetime.now(timezone.utc).isoformat(), version=ultralytics.__version__, license="AGPL-3.0",
                      c26_provenance=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        torch.save(checkpoint, f)
    reloaded = RTDETR(str(output)).model
    verify_model(reloaded, zero=True)
    require(set(target.state_dict()) == set(reloaded.state_dict()) and
            all(torch.equal(v, reloaded.state_dict()[k]) for k, v in target.state_dict().items()), "Reload mismatch")
    raw = torch_load(output, map_location="cpu")
    require(not set(raw) & {"ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness"},
            "Initialization contains training state")
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
