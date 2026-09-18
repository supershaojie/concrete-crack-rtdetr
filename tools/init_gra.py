"""Controlled, untrained GRA initialization and strict native Trainer rebuild audits."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

import init_c19_lif_v1 as parent_init
from init_lif_down import ROOT, MODEL_DIR, SOURCE_SHA256, C2_COMMIT, require, sha256, write_json
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import Conv, GRAConcat, LIFDown, RTDETRDecoder, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_COMMIT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-gra-v1"
DEFAULT_VARIANT = "cbr_lif_gra_v1"
VARIANTS = {
    DEFAULT_VARIANT: (
        "rtdetr-resnet18-lite-cbr-lif-gra-v1.yaml",
        "rtdetr-resnet18-lite-cbr-lif-down.yaml",
        "cbr_lif_gra_v1_rtdetr_r18_lite_e200_b16_onlineaug",
    ),
    "gra_v1": (
        "rtdetr-resnet18-lite-gra-v1.yaml",
        "rtdetr-resnet18-lite.yaml",
        "gra_v1_rtdetr_r18_lite_e200_b16_onlineaug",
    ),
}
GRA_PARAMETERS = 8744
EXPECTED_PARAMETERS = {DEFAULT_VARIANT: 20158509, "gra_v1": 20091516}
CLASS_ADAPTATION_KEYS = {
    "model.26.denoising_class_embed.weight",
    "model.26.enc_score_head.weight",
    "model.26.enc_score_head.bias",
} | {f"model.26.dec_score_head.{i}.{kind}" for i in range(3) for kind in ("weight", "bias")}


def is_added(key):
    return key.startswith("model.18.")


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_summary(tensor):
    value = tensor.detach().cpu()
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha256(value),
                nonzero=int(torch.count_nonzero(value)), min=float(value.min()), max=float(value.max()),
                finite=bool(torch.isfinite(value).all()))


def runtime():
    result = parent_init.runtime()
    import ultralytics.nn.modules.gra as gra
    require(Path(gra.__file__).resolve() == ROOT / "ultralytics-main/ultralytics/nn/modules/gra.py", "Wrong GRA import")
    result.update(gra_module=gra.__file__, gra_sha256=sha256(gra.__file__), base_commit=BASE_COMMIT)
    return result


def build(variant=DEFAULT_VARIANT, nc=1, baseline=False):
    require(variant in VARIANTS, "Unknown GRA variant")
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU generator; fork_rng(devices=[]) does not preserve CUDA RNG.
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant=DEFAULT_VARIANT, zero=False):
    require(variant in VARIANTS, "Unknown GRA variant")
    parent_init.source_contract()
    expected = deepcopy(YAML.load(MODEL_DIR / VARIANTS[variant][1]))
    expected["head"][18 - len(expected["backbone"])] = [[15, 16, 17], 1, "GRAConcat", [16, 4, 0.25, 0.5]]
    require(type(model) is RTDETRDetectionModel, "Native RTDETRDetectionModel required")
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "Only node 18 may differ from the specified parent")
    require(len(model.model) == 27, "Original 27 node indices must remain stable")
    gra = model.model[18]
    require(type(gra) is GRAConcat and gra.f == [15, 16, 17], "GRA input order must be H/U/L from 15/16/17")
    require({15, 16, 17} <= set(model.save), "GRA input missing from savelist")
    require(sum(isinstance(m, GRAConcat) for m in model.modules()) == 1, "Exactly one GRAConcat required")
    require(sum(p.numel() for p in gra.parameters()) == GRA_PARAMETERS, "GRA parameter budget must be exactly 8744")
    require(not list(gra.named_buffers()), "GRA must not add serialized buffers")
    require(gra.groups == 4 and gra.max_offset == 0.25 and gra.alpha == 0.5, "GRA fixed constants changed")
    require(type(model.model[16]) is torch.nn.Upsample and model.model[16].mode == "nearest", "Original nearest node changed")
    require(model.model[17].f == 5 and model.model[19].f == -1 and model.model[20].f == -1, "Original P3 path changed")
    head = model.model[26]
    require(head.f == [19, 22, 25], "Decoder must use the original final neck P3/P4/P5")
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2,
            "Original Decoder dimensions changed")
    if variant == DEFAULT_VARIANT:
        require(type(model.model[20]) is LIFDown and type(head) is RTDETRDecoderCBR, "Original CBR/LIF pair required")
        require(head.cbr.rho == head.cbr.normal_fraction == 0.1, "Original CBR rho changed")
    else:
        require(type(model.model[20]) is Conv and type(head) is RTDETRDecoder, "GRA-only must retain original C2")
        require(not any(isinstance(m, (LIFDown, RTDETRDecoderCBR)) for m in model.modules()), "GRA-only contains CBR/LIF")
    if head.nc == 1 and not model.is_fused():
        require(sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS[variant], "Unexpected nc1 parameter count")
    if zero:
        require(all(torch.count_nonzero(p) == 0 for p in gra.offset.parameters()), "Only offset head must start zero")
        require(all(torch.count_nonzero(m.weight) > 0 for m in (gra.high_proj, gra.lateral_proj, gra.depthwise)),
                "Descriptor convolutions must retain default nonzero initialization")
        if variant == DEFAULT_VARIANT:
            require(torch.count_nonzero(model.model[20].O_proj.weight) == 0, "Original LIF initialization lost")
            require(all(torch.count_nonzero(p) == 0 for p in head.cbr.offset_out.parameters()), "Original CBR initialization lost")
    return dict(gra=18, inputs=[15, 16, 17], p3=19, p4=22, p5=25, decoder=26, savelist=list(model.save))


def state_audit(reference, target):
    """Audit common keys, shapes and values, permitting only the five GRA tensors."""
    before, after = reference.state_dict(), target.state_dict()
    parameters = dict(target.named_parameters())
    extra = set(after) - set(before)
    rows, mismatch = [], []
    for key in sorted(set(before) & set(after)):
        same_shape = before[key].shape == after[key].shape
        row = dict(source=key, target=key, source_shape=list(before[key].shape), target_shape=list(after[key].shape),
                   equal=bool(same_shape and torch.equal(before[key], after[key])), source_sha256=tensor_sha256(before[key]),
                   target_sha256=tensor_sha256(after[key]))
        rows.append(row)
        if not same_shape:
            mismatch.append(row)
    result = dict(COMMON=rows, NEW_TRAINABLE=sorted(extra & parameters.keys()), NEW_BUFFER=sorted(extra - parameters.keys()),
                  MISSING=sorted(set(before) - set(after)), UNEXPECTED=sorted(k for k in extra if not is_added(k)), SHAPE_MISMATCH=mismatch)
    require(not any(result[k] for k in ("MISSING", "UNEXPECTED", "SHAPE_MISMATCH", "NEW_BUFFER")), "Common state inventory/shape mismatch")
    require(all(row["equal"] for row in rows), "Common state value mismatch")
    require(extra == {k for k in after if is_added(k)} and len(extra) == 5, "Expected only five new GRA state tensors")
    return result


def native_rebuild(cfg, weights, nc=1, channels=3):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=nc, channels=channels)
    return RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=weights, verbose=False)


def build_training_model(cfg, weights, data, variant=DEFAULT_VARIANT):
    """Actual get_model path used for both start and learned-state native resume.

    The parent construction sees the exact same CPU RNG but restores it before the
    candidate construction. No learned parameter is reset or reinitialized here.
    """
    require(weights is not None and hasattr(weights, "state_dict"), "GRA start requires its controlled initialization; resume requires its checkpoint")
    require(data["nc"] == 1 and data.get("channels", 3) == 3, "GRA contract requires nc=1/RGB")
    with torch.random.fork_rng(devices=[]):
        baseline = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, nc=1, channels=3)
    target = native_rebuild(cfg, weights, nc=1, channels=3)
    verify_model(target, variant, zero=False)
    before, after = weights.state_dict(), target.state_dict()
    require(set(before) == set(after), "Training rebuild lost or introduced states")
    changed = {k for k in before if before[k].shape != after[k].shape}
    expected = CLASS_ADAPTATION_KEYS if weights.model[26].nc == 80 else set()
    require(weights.model[26].nc in (1, 80) and changed == expected, "Unexpected nc80 to nc1 adaptation")
    require(all(torch.equal(value, after[key]) for key, value in before.items() if key not in changed), "Training rebuild changed initialized/learned tensors")
    report = state_audit(baseline, target)
    report.update(native_get_model=True, source_nc=weights.model[26].nc, target_nc=1, loaded_exact=len(before) - len(changed),
                  learned_gra_state_preserved=True,
                  ALLOWED_CLASS_ADAPTATION=[dict(name=key, source_shape=list(before[key].shape), target_shape=list(after[key].shape),
                                                parent_target_equal=torch.equal(baseline.state_dict()[key], after[key]),
                                                rule="native RTDETRTrainer constructor with synchronized CPU RNG") for key in sorted(changed)])
    return target, report


def controlled_models(source, variant=DEFAULT_VARIANT):
    source = Path(source)
    require(variant in VARIANTS, "Unknown GRA variant")
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Missing/incorrect unified untrained source SHA256")
    checkpoint = torch_load(source, map_location="cpu")
    require(checkpoint.get("epoch") == -1 and all(checkpoint.get(k) is None for k in
            ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness")), "Public source contains trained state")
    if variant == DEFAULT_VARIANT:
        _, parent80, parent_report = parent_init.controlled_models(source)
    else:
        parent80 = build(variant, nc=80, baseline=True)
        original = deepcopy(checkpoint["model"]).float()
        require(original.model[26].nc == 80, "Public source must have nc=80")
        require(all(original.yaml[k] == parent80.yaml[k] for k in ("backbone", "head", "scales")), "Public source differs from original C2")
        parent80.load_state_dict(original.state_dict(), strict=True)
        parent_report = dict(source_sha256=SOURCE_SHA256, source_nc=80, source_states=len(original.state_dict()),
                             all_source_values_exact=all(torch.equal(v, parent80.state_dict()[k]) for k, v in original.state_dict().items()))
        require(parent_report["all_source_values_exact"], "C2 source values changed")
    candidate80, fresh_parent = build(variant, nc=80), build(variant, nc=80, baseline=True)
    constructor_audit = state_audit(fresh_parent, candidate80)
    candidate80.load_state_dict({**candidate80.state_dict(), **parent80.state_dict()}, strict=True)
    verify_model(candidate80, variant, zero=True)
    nc80_audit = state_audit(parent80, candidate80)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        target, trainer_report = build_training_model(str(MODEL_DIR / VARIANTS[variant][0]), candidate80, dict(nc=1, channels=3), variant)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), parent80, nc=1)
    verify_model(target, variant, zero=True)
    report = state_audit(parent, target)
    require(len(trainer_report["ALLOWED_CLASS_ADAPTATION"]) == 9, "Exactly nine nc80 to nc1 adaptations required")
    # A separately built other variant checks all five tensors, not only the zero head.
    other_variant = next(v for v in VARIANTS if v != variant)
    other = build(other_variant, nc=1)
    new_state = {k: v for k, v in target.state_dict().items() if is_added(k)}
    require(all(torch.equal(value, other.state_dict()[key]) for key, value in new_state.items()), "New states differ across GRA variants")
    report.update(status="PASSED", variant=variant, source=str(source.resolve()), source_sha256=SOURCE_SHA256,
                  base_commit=BASE_COMMIT, c2_commit=C2_COMMIT, seed=42, source_nc=80, target_nc=1,
                  new_parameters=sum(v.numel() for v in new_state.values()), public_constructor_equal=True,
                  constructor_common_states=len(constructor_audit["COMMON"]), nc80_common_audit=nc80_audit,
                  ALLOWED_CLASS_ADAPTATION=trainer_report["ALLOWED_CLASS_ADAPTATION"], native_trainer=trainer_report,
                  parent_initialization=parent_report, new_initial_values={k: tensor_summary(v) for k, v in new_state.items()},
                  variants_new_state_exact=True, storage="FP32; original half source values exactly promoted")
    return parent, target, report


class _SetupAuditTrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model, self.gra_rebuild_audit = build_training_model(cfg, weights, self.data, self.gra_variant)
        return model


def audit_training_setup(checkpoint_path, reference, variant):
    """Execute native BaseTrainer.setup_model checkpoint loading and get_model dispatch."""
    trainer = _SetupAuditTrainer.__new__(_SetupAuditTrainer)
    trainer.args = SimpleNamespace(pretrained=False)
    trainer.model = str(Path(checkpoint_path).resolve())
    trainer.data = dict(nc=1, channels=3)
    trainer.gra_variant = variant
    checkpoint = trainer.setup_model()
    require(checkpoint is not None, "Native setup_model did not load checkpoint")
    before, after = reference.state_dict(), trainer.model.state_dict()
    require(set(before) == set(after) and all(torch.equal(v, after[k]) for k, v in before.items()), "Native training setup changed states")
    verify_model(trainer.model, variant, zero=False)
    return dict(status="PASSED", native_setup_model=True, checkpoint_epoch=checkpoint.get("epoch"),
                all_state_values_exact=True, state_count=len(before), get_model=trainer.gra_rebuild_audit)


def initialize(source, output, variant=DEFAULT_VARIANT):
    output = Path(output)
    require(not output.exists(), f"Existing initialization preserved: {output}")
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    report["runtime"] = runtime()
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                      date=datetime.now(timezone.utc).isoformat(), gra_provenance=deepcopy(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    restored = RTDETR(str(output)).model
    before, after = target.state_dict(), restored.state_dict()
    require(set(before) == set(after) and all(torch.equal(v, after[k]) for k, v in before.items()), "Saved initialization reload differs")
    verify_model(restored, variant, zero=True)
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True,
                  reload_new_initial_values={k: tensor_summary(v) for k, v in after.items() if is_added(k)},
                  training_rebuild=audit_training_setup(output, target, variant), status="PASSED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Refuse before the failure-report handler so reruns preserve a prior audit.
    require(not args.report.exists(), f"Existing initialization audit preserved: {args.report}")
    try:
        result = initialize(args.source, args.output, args.variant)
    except Exception as exc:
        write_json(args.report, dict(status="FAILED", variant=args.variant, error=f"{type(exc).__name__}: {exc}"))
        raise
    write_json(args.report, result)
    print(f"PASSED {args.variant}: {result['new_parameters']} new parameters, "
          f"{len(result['COMMON'])} exact common states; init SHA256 {result['output_sha256']}")


if __name__ == "__main__":
    main()
