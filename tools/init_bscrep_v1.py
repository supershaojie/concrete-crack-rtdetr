"""Strict BSC-Rep v1 initialization and audited native RT-DETR class adaptation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys

import torch

from init_c19_lif_v1 import (
    ROOT, MODEL_DIR, SOURCE_SHA256, require, sha256, write_json,
    controlled_models as original_pair_models, source_contract,
)
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import BSCRepC3, BilateralContextSupport, Conv, LIFDown, RepConv, RTDETRDecoder, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML

BASE_COMMIT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
DEFAULT_VARIANT = "cbr_lif_bscrep_v1"
VARIANTS = {
    DEFAULT_VARIANT: (
        "rtdetr-resnet18-lite-cbr-lif-bscrep-v1.yaml",
        "rtdetr-resnet18-lite-cbr-lif-down.yaml",
        "cbr_lif_bscrep_v1_rtdetr_r18_lite_e200_b16_onlineaug",
    ),
    "bscrep_v1": (
        "rtdetr-resnet18-lite-bscrep-v1.yaml",
        "rtdetr-resnet18-lite.yaml",
        "bscrep_v1_rtdetr_r18_lite_e200_b16_onlineaug",
    ),
}
BSC_KEYS = frozenset("model.19.bsc." + name for name in (
    "in_proj.weight", "gate.weight", "gate.bias", "out_proj.weight",
))
BSC_PARAMETERS = 11296


def is_added(key):
    return key in BSC_KEYS


def runtime():
    import ultralytics
    from ultralytics.nn.modules import bsc_rep, cbr, lif_down
    require(Path(ultralytics.__file__).resolve() == ROOT / "ultralytics-main/ultralytics/__init__.py", "Wrong ultralytics import")
    for module in (bsc_rep, cbr, lif_down):
        require(Path(module.__file__).resolve() == ROOT / "ultralytics-main/ultralytics/nn/modules" / (module.__name__.rsplit(".", 1)[-1] + ".py"), "Wrong module import")
    return dict(
        python=sys.version, executable=sys.executable, torch=str(torch.__version__),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        ultralytics=ultralytics.__file__, bsc_rep_module=bsc_rep.__file__,
        original_module_hashes=source_contract(), base_commit=BASE_COMMIT,
        commit=subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )


def build(variant=DEFAULT_VARIANT, nc=80, parent=False):
    require(variant in VARIANTS, "Unknown BSC variant")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(parent)]), nc=nc, verbose=False)


def verify_model(model, variant=DEFAULT_VARIANT, zero=False):
    """Check fixed topology, branch inventory and original CBR/LIF preservation; never reset state."""
    require(variant in VARIANTS, "Unknown BSC variant")
    source_contract()
    require(type(model) is RTDETRDetectionModel, "Native RTDETRDetectionModel required")
    parent_config = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected = deepcopy(parent_config)
    index = 19 - len(expected["backbone"])
    require(expected["head"][index] == [-1, 3, "RepC3", [256, 0.5]], "Parent P3 topology changed")
    expected["head"][index][2] = "BSCRepC3"
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "Only parent layer 19 may change")
    node = model.model[19]
    require(type(node) is BSCRepC3 and sum(type(m) is BSCRepC3 for m in model.modules()) == 1, "Exactly one BSCRepC3 required")
    require(type(node.bsc) is BilateralContextSupport and sum(type(m) is BilateralContextSupport for m in model.modules()) == 1, "Exactly one BSC branch required")
    require(len(node.m) == 3 and all(type(m) is RepConv for m in node.m), "Original three internal RepConv blocks required")
    require(node.cv1.conv.in_channels == 512 and node.cv1.conv.out_channels == 128 and node.cv3.conv.out_channels == 256, "P3 channel contract changed")
    require(set(node.bsc.state_dict()) == {k.removeprefix("model.19.bsc.") for k in BSC_KEYS}, "Unexpected BSC state/buffers")
    require(sum(p.numel() for p in node.bsc.parameters()) == BSC_PARAMETERS, "BSC parameter delta differs from 11296")
    require({k for k in model.state_dict() if ".bsc." in k} == BSC_KEYS, "Unexpected BSC placement")
    require(model.model[18].f == [-2, -1] and model.model[19].f == -1, "P3 input edges changed")
    require(type(model.model[23]) is Conv and model.model[26].f == [19, 22, 25], "PAN/Decoder edges changed")
    head = model.model[26]
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2, "Decoder dimensions changed")
    if variant == DEFAULT_VARIANT:
        lif = model.model[20]
        require(type(lif) is LIFDown and sum(isinstance(m, LIFDown) for m in model.modules()) == 1, "Original single LIFDown required")
        require(hasattr(lif, "bn") and lif.forward.__func__ is LIFDown.forward, "Original pre-BN LIF residual/fusion protection changed")
        require(type(head) is RTDETRDecoderCBR and head.cbr.rho == head.cbr.normal_fraction == 0.10, "Original CBR required")
        require(sum(p.numel() for p in head.cbr.parameters()) == 45889 and len(list(head.cbr.parameters())) == 14, "Original CBR parameters changed")
    else:
        require(type(model.model[20]) is Conv and type(head) is RTDETRDecoder, "Single BSC variant retained CBR/LIF")
        require(not any(isinstance(m, (LIFDown, RTDETRDecoderCBR)) for m in model.modules()), "Single BSC variant retained CBR/LIF")
    if zero:
        require(torch.count_nonzero(node.bsc.out_proj.weight).item() == 0, "Fresh BSC out projection must be zero")
        require(torch.count_nonzero(node.bsc.gate.bias).item() == 0, "Fresh gate bias must be zero")
        require(all(torch.count_nonzero(p).item() > 0 for p in (node.bsc.in_proj.weight, node.bsc.gate.weight)), "Fresh upstream projections must be nonzero")
    return dict(p3=19, downsample=20, pan=[22, 25], decoder=26, from_layers=[19, 22, 25], bsc_parameters=BSC_PARAMETERS)


def controlled_models(source, variant=DEFAULT_VARIANT):
    """Use the archived original pair initializer, then strict-load the complete selected parent."""
    require(variant in VARIANTS, "Unknown BSC variant")
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Fixed untrained source SHA256 mismatch")
    baseline, pair, original_report = original_pair_models(source)
    parent = pair if variant == DEFAULT_VARIANT else baseline
    target = build(variant)
    public, fresh = parent.state_dict(), target.state_dict()
    require(set(fresh) - set(public) == BSC_KEYS and not set(public) - set(fresh), "Unexpected common/new state inventory")
    require(all(public[k].shape == fresh[k].shape for k in public), "Parent state shape mismatch")
    # This strict merge is the proof of common initial values, independent of RNG claims.
    target.load_state_dict({**fresh, **public}, strict=True)
    require(all(torch.equal(v, target.state_dict()[k]) for k, v in public.items()), "Parent parameters or BN buffers changed")
    verify_model(target, variant, zero=True)
    report = dict(
        status="PASSED", variant=variant, base_commit=BASE_COMMIT,
        source=str(source.resolve()), source_sha256=SOURCE_SHA256, source_nc=80, target_nc=80,
        parent_yaml=VARIANTS[variant][1], target_yaml=VARIANTS[variant][0], seed=42,
        COMMON=[dict(name=k, shape=list(v.shape), equal=True) for k, v in public.items()],
        NEW_TRAINABLE=sorted(BSC_KEYS), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
        ALLOWED_CLASS_ADAPTATION=[], new_parameters=BSC_PARAMETERS, parent_public_exact=True,
        parent_state_count=len(public), target_state_count=len(fresh), formal_optimizer_steps=0,
        parent_initialization_source_sha256=original_report["source_sha256"],
        bsc_initial_sha256={k: hashlib.sha256(target.state_dict()[k].detach().cpu().numpy().tobytes()).hexdigest() for k in sorted(BSC_KEYS)},
    )
    return parent, target, report


def native_rebuild(cfg, weights, nc=1, channels=3):
    """Invoke the real Trainer construction/loading method, without allocating a run directory."""
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=nc, channels=channels)
    return RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=weights, verbose=False)


def build_training_model(cfg, weights, data, variant=DEFAULT_VARIANT, fresh=True):
    """Audit native nc=80->1 loading and learned-state resume without resetting any parameter."""
    require(weights is not None and isinstance(weights, RTDETRDetectionModel), "A complete controlled init or resume model is required")
    require(data["nc"] == 1 and data.get("channels", 3) == 3, "Original single-class RGB data required")
    verify_model(weights, variant, zero=fresh)
    # A fork restores precisely the RNG entering the real model construction. Both
    # paths therefore use identical native initialization for the nine class tensors.
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), deepcopy(weights), data["nc"], data.get("channels", 3))
    target = native_rebuild(cfg, weights, data["nc"], data.get("channels", 3))
    verify_model(target, variant, zero=fresh)
    before, after = weights.state_dict(), target.state_dict()
    require(set(before) == set(after), "Native Trainer silently lost or added state")
    allowed = {"model.26." + k for k in ("denoising_class_embed.weight", "enc_score_head.weight", "enc_score_head.bias")}
    allowed |= {f"model.26.dec_score_head.{i}.{s}" for i in range(3) for s in ("weight", "bias")}
    changed = {k for k in before if before[k].shape != after[k].shape}
    expected_changes = allowed if weights.model[26].nc != data["nc"] else set()
    require(changed == expected_changes, "Unexpected native class shape adaptation: " + repr(sorted(changed)))
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in changed), "Native loading changed or omitted initialized/learned values")
    require(all(torch.equal(v, after[k]) for k, v in parent.state_dict().items()), "Native nc=1 parent common state differs")
    # The native loader uses intersect_dicts/strict=False; the full inventory,
    # explicit shape allowlist and exact comparisons above close that loading gap.
    report = dict(
        status="PASSED", variant=variant, source_nc=weights.model[26].nc, target_nc=1,
        native_get_model=True, fresh=fresh, reset_learned_parameters=False,
        COMMON=sorted(k for k in before if k not in changed and k not in BSC_KEYS),
        NEW_TRAINABLE=sorted(BSC_KEYS), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
        ALLOWED_CLASS_ADAPTATION=[dict(name=k, source_shape=list(before[k].shape), target_shape=list(after[k].shape),
            rule="Native RTDETRTrainer class constructor; exact corresponding parent value") for k in sorted(changed)],
        loaded_exact=len(before) - len(changed), parent_nc1_public_exact=True,
        parent_nc1_public_states=len(parent.state_dict()), bsc_exact=True, formal_optimizer_steps=0,
    )
    return target, report


def initialize(source, output, variant=DEFAULT_VARIANT):
    output = Path(output)
    require(not output.exists(), "Existing initialization protected: " + str(output))
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    checkpoint = dict(
        epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
        optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
        date=datetime.now(timezone.utc).isoformat(), bscrep_v1_provenance=report,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    restored = RTDETR(str(output)).model
    require(set(restored.state_dict()) == set(target.state_dict()), "Serialized initialization inventory differs")
    require(all(torch.equal(v, restored.state_dict()[k]) for k, v in target.state_dict().items()), "Serialized initialization values differ")
    verify_model(restored, variant, zero=True)
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True, runtime=runtime())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.report, initialize(args.source, args.output, args.variant))
