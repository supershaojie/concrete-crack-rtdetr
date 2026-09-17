"""Strict public-source initialization and native Trainer state audits for BFR-P4."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys

import torch

from init_lif_down import ROOT, MODEL_DIR, SOURCE_SHA256, C2_COMMIT, require, sha256, write_json
import init_c19_lif_v1 as parent_init
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import BFRP4, BFRRepC3, Conv, LIFDown, RepC3, RTDETRDecoder, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_COMMIT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
MAIN_VARIANT = "cbr_lif_bfr_p4_v1"
VARIANTS = {
    MAIN_VARIANT: ("rtdetr-resnet18-lite-cbr-lif-bfr-p4-v1.yaml", "rtdetr-resnet18-lite-cbr-lif-down.yaml",
                   "cbr_lif_bfr_p4_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
    "bfr_p4_v1": ("rtdetr-resnet18-lite-bfr-p4-v1.yaml", "rtdetr-resnet18-lite.yaml",
                   "bfr_p4_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
}
BFR_PREFIX = "model.22.bfr."
PARAMETERS = {MAIN_VARIANT: (20170453, 19965653), "bfr_p4_v1": (20103460, 19898404)}
CLASS_ADAPTATION = {"model.26.denoising_class_embed.weight", "model.26.enc_score_head.weight",
                    "model.26.enc_score_head.bias"} | {
    f"model.26.dec_score_head.{i}.{suffix}" for i in range(3) for suffix in ("weight", "bias")
}


def is_added(key):
    return key.startswith(BFR_PREFIX)


def tensor_sha256(value):
    """Hash dtype, shape, and exact CPU tensor bytes (parameters and buffers)."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update((str(value.dtype) + ":" + str(tuple(value.shape)) + ":").encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_inventory(state):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_sha256(v)}
            for k, v in sorted(state.items())}


def runtime():
    import ultralytics
    import ultralytics.nn.modules.bfr_p4 as module
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "ultralytics-main"), "Wrong ultralytics import")
    expected = ROOT / "ultralytics-main/ultralytics/nn/modules/bfr_p4.py"
    require(Path(module.__file__).resolve() == expected, "Wrong BFR import")
    info = dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, module_hashes=parent_init.source_contract(),
                commit=subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
                bfr_module=str(expected), bfr_module_sha256=sha256(expected), base_commit=BASE_COMMIT)
    return info


def build(variant=MAIN_VARIANT, nc=80, baseline=False):
    require(variant in VARIANTS, "Unknown variant")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant=MAIN_VARIANT, zero=False):
    require(variant in VARIANTS, "Unknown variant")
    parent_init.source_contract()
    require(type(model) is RTDETRDetectionModel, "Native RTDETRDetectionModel required")
    reference = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected = deepcopy(reference)
    expected["head"][22 - len(reference["backbone"])] = [-1, 3, "BFRRepC3", [256, 0.5, 32]]
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")),
            "Only complete final P4 RepC3 may be wrapped")
    require(len(model.model) == 27, "Layer numbering changed")
    wrapper = model.model[22]
    require(type(wrapper) is BFRRepC3 and isinstance(wrapper, RepC3) and wrapper.f == -1,
            "P4 wrapper type/from changed")
    require(len(wrapper.m) == 3 and sum(isinstance(m, BFRRepC3) for m in model.modules()) == 1,
            "Exactly one wrapper and three internal RepConvs required")
    require(type(wrapper.bfr) is BFRP4 and sum(isinstance(m, BFRP4) for m in model.modules()) == 1,
            "Exactly one BFR required")
    require(sum(p.numel() for p in wrapper.bfr.parameters()) == 20688, "BFR parameter count changed")
    require(sum(p.numel() for k, p in model.named_parameters() if is_added(k)) == 20688,
            "Added trainable namespace changed")
    require(not list(wrapper.bfr.named_buffers()), "BFR constants must remain nonpersistent runtime state")
    for index in (19, 25):
        require(type(model.model[index]) is RepC3, f"Original RepC3 {index} changed")
    require(type(model.model[23]) is Conv and model.model[23].f == -1, "Original P4-to-P5 path changed")
    require(model.model[21].f == [-1, 15] and model.model[24].f == [-1, 10], "Original fusion wiring changed")
    head = model.model[26]
    require(head.f == [19, 22, 25] and head.hidden_dim == 256 and head.num_queries == 300,
            "Original decoder inputs/dimensions changed")
    require(len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2, "Original decoder depth changed")
    if variant == MAIN_VARIANT:
        require(type(model.model[20]) is LIFDown and type(head) is RTDETRDecoderCBR,
                "Original CBR/LIF required")
        require(head.cbr.rho == head.cbr.normal_fraction == .10, "Original CBR fraction changed")
        if zero:
            require(torch.count_nonzero(model.model[20].O_proj.weight) == 0, "Original LIF initialization changed")
            require(all(torch.count_nonzero(p) == 0 for p in head.cbr.offset_out.parameters()),
                    "Original CBR initialization changed")
    else:
        require(type(model.model[20]) is Conv and type(head) is RTDETRDecoder, "C2 ablation altered")
        require(not any(isinstance(m, LIFDown) for m in model.modules()), "Ablation unexpectedly has LIF")
    if head.nc == 1:
        require(sum(p.numel() for p in model.parameters()) == PARAMETERS[variant][int(model.is_fused())],
                "nc1 parameter count differs from contract")
    if zero:
        state = wrapper.bfr.state_dict()
        # The exact zero and nonzero contract is checked without relying on class internals.
        matrix_weights = {k: v for k, v in state.items() if k.endswith("weight") and v.ndim >= 2}
        zero_weights = [k for k, v in matrix_weights.items() if torch.count_nonzero(v) == 0]
        require(len(zero_weights) == 1 and tuple(matrix_weights[zero_weights[0]].shape) == (256, 32, 1, 1),
                "Only output projection weight may be zero")
        require(all(torch.count_nonzero(v) == 0 for k, v in state.items() if k.endswith("bias")),
                "BFR biases must initialize to zero")
        require(all(torch.equal(v, torch.ones_like(v)) for k, v in state.items()
                    if k.endswith("weight") and v.ndim == 1), "GN weight must initialize to one")
    return {"p4": 22, "decoder": 26, "new_parameters": 20688, "nodes": 27}


def controlled_models(source, variant=MAIN_VARIANT):
    """Reuse the successful parent's exact source and original CBR/LIF initialization."""
    require(variant in VARIANTS, "Unknown variant")
    c2, pair, original_audit = parent_init.controlled_models(source)
    parent = pair if variant == MAIN_VARIANT else c2
    target = build(variant)
    constructor_parent = build(variant, baseline=True).state_dict()
    fresh = target.state_dict()
    require(all(k in fresh and torch.equal(v, fresh[k]) for k, v in constructor_parent.items()),
            "BFR construction consumed parent RNG or changed shared state")
    before = parent.state_dict()
    extra = set(fresh) - set(before)
    missing = sorted(set(before) - set(fresh))
    mismatched = sorted(k for k in before if k in fresh and before[k].shape != fresh[k].shape)
    parameters = dict(target.named_parameters())
    require(not missing and not mismatched and extra == {k for k in fresh if is_added(k)},
            "BFR state inventory differs from parent")
    require(extra <= parameters.keys() and sum(parameters[k].numel() for k in extra) == 20688,
            "Unexpected new buffer/trainable parameter inventory")
    target.load_state_dict({**fresh, **before}, strict=True)
    after = target.state_dict()
    require(all(torch.equal(v, after[k]) for k, v in before.items()), "Shared values changed during initialization")
    verify_model(target, variant, zero=True)
    report = dict(variant=variant, base_commit=BASE_COMMIT, c2_commit=C2_COMMIT, source=str(Path(source).resolve()),
                  source_sha256=SOURCE_SHA256, source_nc=80, target_nc=80, seed=42,
                  parent_yaml=VARIANTS[variant][1], target_yaml=VARIANTS[variant][0],
                  COMMON=[dict(name=k, equal=True, **state_inventory({k: v})[k]) for k, v in sorted(before.items())],
                  NEW_TRAINABLE=sorted(extra), NEW_BUFFER=[], MISSING=missing, UNEXPECTED=[],
                  SHAPE_MISMATCH=mismatched, ALLOWED_CLASS_ADAPTATION=[], new_parameters=20688,
                  new_initial_values=state_inventory({k: after[k] for k in extra}),
                  parent_controlled_audit=original_audit, public_constructor_equal=True,
                  runtime_constants="Rebuilt FP32 constants; no persistent input-dependent tensors",
                  storage="FP32; source checkpoint quantization is preserved exactly")
    return parent, target, report


def native_rebuild(cfg, weights, nc=1, channels=3):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = {"nc": nc, "channels": channels}
    return RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=weights, verbose=False)


def build_training_model(cfg, weights, data, variant=MAIN_VARIANT):
    """Audit the actual native get_model path, including learned-state reload/resume.

    Parent reconstruction sees the identical incoming CPU RNG and is isolated;
    target construction is the only call that advances the caller's native RNG.
    """
    require(weights is not None, "Audited model reconstruction requires controlled or learned weights")
    with torch.random.fork_rng(devices=[]):
        reference = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, data["nc"], data["channels"])
    target = native_rebuild(cfg, weights, data["nc"], data["channels"])
    verify_model(target, variant)
    before, after, baseline = weights.state_dict(), target.state_dict(), reference.state_dict()
    require(set(before) == set(after), "Native Trainer state keys changed")
    changed = {k for k in before if before[k].shape != after[k].shape}
    expected = CLASS_ADAPTATION if weights.model[-1].nc != data["nc"] else set()
    require(changed == expected, "Unexpected native Trainer class adaptation")
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in changed),
            "Native Trainer overwrote initialized/learned state")
    require(set(baseline) == {k for k in after if not is_added(k)}, "Native parent state inventory changed")
    require(all(torch.equal(v, after[k]) for k, v in baseline.items()), "Native nc1 parent tensors differ")
    adaptations = [dict(name=k, source_shape=list(before[k].shape), target_shape=list(after[k].shape),
                        parent_equal=True, sha256=tensor_sha256(after[k]),
                        rule="Native get_model with identical CPU constructor RNG; exact own-parent nc1 value")
                   for k in sorted(changed)]
    report = dict(native_get_model=True, variant=variant, source_nc=weights.model[-1].nc,
                  target_nc=data["nc"], COMMON=sorted(k for k in baseline if k not in changed),
                  NEW_TRAINABLE=sorted(k for k in after if is_added(k)), NEW_BUFFER=[], MISSING=[],
                  UNEXPECTED=[], SHAPE_MISMATCH=[], ALLOWED_CLASS_ADAPTATION=adaptations,
                  loaded_exact=len(before) - len(changed), parent_nc1_shared_exact=True,
                  parent_nc1_shared_states=len(baseline), original_and_bfr_states_preserved=True,
                  new_state_hashes=state_inventory({k: v for k, v in after.items() if is_added(k)}),
                  parameters=sum(p.numel() for p in target.parameters()))
    return target, report


def audit_checkpoint(path, variant=MAIN_VARIANT, expected_sha256=None, require_untrained=False):
    path = Path(path)
    require(path.is_file(), f"Missing checkpoint: {path}")
    actual_sha = sha256(path)
    require(expected_sha256 is None or actual_sha == expected_sha256, "Checkpoint SHA256 mismatch")
    checkpoint = torch_load(path, map_location="cpu")
    provenance = checkpoint.get("bfr_p4_provenance", {})
    require(provenance.get("variant") == variant, "Checkpoint variant provenance mismatch")
    if require_untrained:
        require(checkpoint.get("epoch") == -1 and all(checkpoint.get(k) is None for k in
                    ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness")),
                "Initialization contains learned training state")
        require(provenance.get("source_sha256") == SOURCE_SHA256 and provenance.get("base_commit") == BASE_COMMIT,
                "Initialization source/base provenance mismatch")
    model = checkpoint.get("ema") or checkpoint.get("model")
    require(model is not None, "Checkpoint has no model")
    verify_model(model, variant, zero=require_untrained)
    if require_untrained:
        require(model.model[-1].nc == 80, "Controlled initialization must retain source nc80")
        state = model.state_dict()
        common = {row["name"]: row for row in provenance.get("COMMON", [])}
        added = provenance.get("new_initial_values", {})
        require(set(state) == set(common) | set(added) and not set(common) & set(added),
                "Recorded initialization key inventory differs from checkpoint")
        require(set(added) == {k for k in state if is_added(k)}, "Recorded new-state namespace changed")
        require(all(tensor_sha256(state[k]) == row.get("sha256") for k, row in {**common, **added}.items()),
                "Checkpoint tensor values differ from initialization audit")
    return dict(path=str(path.resolve()), sha256=actual_sha, variant=variant, epoch=checkpoint.get("epoch"),
                untrained_checked=require_untrained, all_recorded_tensor_hashes_checked=require_untrained,
                provenance=provenance, status="PASSED")


def initialize(source, output, variant=MAIN_VARIANT):
    output = Path(output)
    require(not output.exists(), f"Existing initialization preserved: {output}")
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        nc1, trainer_report = build_training_model(target.yaml, target, {"nc": 1, "channels": 3}, variant)
    del nc1
    other = next(v for v in VARIANTS if v != variant)
    comparison = build(other).state_dict()
    own = {k: v for k, v in target.state_dict().items() if is_added(k)}
    require(set(own) == {k for k in comparison if is_added(k)} and
            all(torch.equal(v, comparison[k]) for k, v in own.items()), "Cross-variant BFR initialization differs")
    report.update(trainer_nc1=trainer_report, cross_variant_new_initial_values_exact=True,
                  compared_variant=other, runtime=runtime())
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                      date=datetime.now(timezone.utc).isoformat(), bfr_p4_provenance=deepcopy(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    restored = RTDETR(str(output)).model
    require(set(restored.state_dict()) == set(target.state_dict()) and
            all(torch.equal(v, restored.state_dict()[k]) for k, v in target.state_dict().items()),
            "Serialized FP32 initialization differs")
    audit_checkpoint(output, variant, require_untrained=True)
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True, status="PASSED")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    for name in ("source", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.report, initialize(args.source, args.output, args.variant))
