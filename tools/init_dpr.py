"""DPR-1 controlled public initialization and actual nc=1 Trainer rebuild audit.

This entry never optimizes a model. Its train-entry audit stops immediately after
the native RTDETR.train -> Trainer.get_model reconstruction, before data loading.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from init_lif_down import ROOT, MODEL_DIR, SOURCE_SHA256, C2_COMMIT, require, sha256, write_json
from init_c19_lif_v1 import source_contract, controlled_models as controlled_cbr_lif
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import Conv, LIFDown, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_COMMIT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
MAIN = "cbr_lif_dpr_v1"
VARIANTS = {
    MAIN: ("rtdetr-resnet18-lite-cbr-lif-dpr-v1.yaml", "rtdetr-resnet18-lite-cbr-lif-down.yaml",
           "cbr_lif_dpr_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
    "dpr_v1": ("rtdetr-resnet18-lite-dpr-v1.yaml", "rtdetr-resnet18-lite.yaml",
               "dpr_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
}
TARGET = "model.5.blocks.1.branch2b"
DPR_KEYS = tuple(TARGET + "." + name for name in ("dpr_cd", "dpr_hd", "dpr_vd", "dpr_ad"))
COUNTS = {MAIN: 20152837, "dpr_v1": 20085844}


def runtime():
    from init_c19_lif_v1 import runtime as parent_runtime
    from ultralytics.nn.modules import dpr
    info = parent_runtime()
    path = Path(dpr.__file__).resolve()
    require(path == ROOT / "ultralytics-main/ultralytics/nn/modules/dpr.py", "Wrong DPR import")
    info.update(dpr_module=str(path), dpr_sha256=sha256(path), base_commit=BASE_COMMIT)
    return info


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build(variant=MAIN, nc=80, baseline=False):
    require(variant in VARIANTS, "Unknown DPR variant")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant=MAIN, zero=False):
    """Verify both training representation and a correctly folded deploy copy."""
    from ultralytics.nn.modules.dpr import BlocksDPR, DPRConvNormLayer
    source_contract()
    require(variant in VARIANTS, "Unknown DPR variant")
    require(type(model) is RTDETRDetectionModel, "Native RTDETRDetectionModel required")
    parent = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected = deepcopy(parent)
    require(expected["backbone"][5] == [-1, 1, "Blocks", [128, "BasicBlock", 2, 3, "relu"]],
            "Fixed parent P3 structure changed")
    expected["backbone"][5][2] = "BlocksDPR"
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")),
            "Only parent node 5 Blocks -> BlocksDPR may change")
    require(len(model.model) == 27, "Original 27 top-level nodes required")
    require(type(model.model[5]) is BlocksDPR, "DPR stage registration lost")
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, DPRConvNormLayer)]
    require(len(targets) == 1 and targets[0][0] == TARGET, "Exactly one DPR at fixed branch2b required")
    module = targets[0][1]
    conv = module.conv
    require(type(conv) is torch.nn.Conv2d and conv.in_channels == conv.out_channels == 128,
            "DPR must retain the original dense 128 -> 128 Conv2d")
    require((conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias) ==
            ((3, 3), (1, 1), (1, 1), (1, 1), 1, None), "DPR target convolution geometry changed")
    require(type(module.norm) is torch.nn.BatchNorm2d and module.norm.num_features == 128,
            "Standard deployment must retain the original target BatchNorm")
    require(type(module.act) is torch.nn.Identity, "branch2b has original Identity, no added activation")
    require([node.f for node in model.model] == [row[0] for row in parent["backbone"] + parent["head"]],
            "Original from indices changed")
    head = model.model[-1]
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3 and
            head.decoder.eval_idx == 2, "Original Decoder dimensions changed")
    require(head.f == [19, 22, 25], "Original Decoder feature inputs changed")
    if variant == MAIN:
        require(type(head) is RTDETRDecoderCBR and head.cbr.rho == head.cbr.normal_fraction == .10,
                "Original CBR contract changed")
        require(type(model.model[20]) is LIFDown and sum(isinstance(m, LIFDown) for m in model.modules()) == 1,
                "Original LIF node changed")
        require(sum(p.numel() for p in head.cbr.parameters()) == 45889, "Original CBR parameter count changed")
    else:
        require(not any(isinstance(m, (LIFDown, RTDETRDecoderCBR)) for m in model.modules()),
                "C2 single-module ablation must not contain CBR or LIF")
        require(type(model.model[20]) is Conv, "C2 original downsampling must remain Conv")
    params = dict(model.named_parameters())
    present = set(DPR_KEYS) & params.keys()
    require(present in (set(), set(DPR_KEYS)), "DPR parameters were partially lost")
    training_representation = bool(present)
    if training_representation:
        require([tuple(params[key].shape) for key in DPR_KEYS] == [(128, 9), (128, 3), (128, 3), (128, 9)],
                "DPR parameter shapes changed")
        require(sum(params[key].numel() for key in DPR_KEYS) == 3072, "DPR must add exactly 3072 parameters")
        if model.training:
            require(all(params[key].requires_grad for key in DPR_KEYS), "Live training DPR parameters must remain trainable")
    else:
        require(getattr(module, "deployed", False) is True, "Missing DPR parameters without explicit deploy state")
    if zero:
        require(training_representation, "Zero initialization requires training representation")
        require(all(torch.count_nonzero(params[key]).item() == 0 for key in DPR_KEYS), "DPR must start exactly zero")
        if variant == MAIN:
            require(torch.count_nonzero(model.model[20].O_proj.weight).item() == 0, "Original LIF initialization changed")
            require(all(torch.count_nonzero(p).item() == 0 for p in head.cbr.offset_out.parameters()),
                    "Original CBR initialization changed")
    if head.nc == 1 and not model.is_fused():
        require(sum(p.numel() for p in model.parameters()) == COUNTS[variant] - (0 if training_representation else 3072),
                "Unfused nc1 parameter count changed")
    return dict(target=TARGET, variant=variant, training_representation=training_representation,
                new_parameters=3072 if training_representation else 0, top_level_nodes=27,
                parameters=sum(p.numel() for p in model.parameters()))


def controlled_models(source, variant=MAIN):
    source = Path(source)
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Missing/incorrect public initialization SHA256")
    require(variant in VARIANTS, "Unknown DPR variant")
    checkpoint = torch_load(source, map_location="cpu")
    require(checkpoint.get("epoch") == -1 and all(checkpoint.get(k) is None for k in
            ("ema", "optimizer", "scaler", "updates", "train_metrics", "train_results", "best_fitness")),
            "Public source contains trained state")
    original = deepcopy(checkpoint["model"]).float()
    require(original.model[-1].nc == 80, "Public source must have nc=80")
    parent, target = build(variant, baseline=True), build(variant)
    fresh, public = target.state_dict(), parent.state_dict()
    require(all(k in fresh and torch.equal(v, fresh[k]) for k, v in public.items()),
            "DPR constructor polluted public RNG/initial values")
    require(set(fresh) - set(public) == set(DPR_KEYS) and not set(public) - set(fresh),
            "Exactly four added state keys and no removed public keys required")
    require(set(DPR_KEYS) <= dict(target.named_parameters()).keys(), "Added states must all be trainable parameters")
    inherited = None
    if variant == MAIN:
        _, controlled_parent, inherited = controlled_cbr_lif(source)
        parent.load_state_dict(controlled_parent.state_dict(), strict=True)
    else:
        require(original.yaml["backbone"] == parent.yaml["backbone"] and original.yaml["head"] == parent.yaml["head"],
                "Public source semantics differ from C2 parent")
        require(set(original.state_dict()) == set(public), "Public source key inventory differs from C2")
        parent.load_state_dict(original.state_dict(), strict=True)
    target.load_state_dict({**fresh, **parent.state_dict()}, strict=True)
    public = parent.state_dict()
    target_state = target.state_dict()
    rows = [dict(source=key, target=key, shape=list(value.shape), equal=torch.equal(value, target_state[key]),
                 sha256=tensor_sha256(value)) for key, value in public.items()]
    require(all(row["equal"] for row in rows), "Public tensor values changed")
    structural = verify_model(target, variant, zero=True)
    report = dict(status="PASSED", variant=variant, base_commit=BASE_COMMIT, c2_commit=C2_COMMIT,
                  source=str(source.resolve()), source_sha256=SOURCE_SHA256, source_nc=80, target_nc=80, seed=42,
                  COMMON=rows, NEW_TRAINABLE=list(DPR_KEYS), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[],
                  SHAPE_MISMATCH=[], ALLOWED_CLASS_ADAPTATION=[], new_parameters=3072,
                  public_constructor_equal=True, source_storage="Original source tensors promoted to FP32 exactly",
                  structure=structural,
                  new_initial_values={key: dict(shape=list(target_state[key].shape), nonzero=0,
                      sha256=tensor_sha256(target_state[key])) for key in DPR_KEYS})
    if inherited is not None:
        report["original_cbr_lif_initialization"] = inherited
    return parent, target, report


def native_rebuild(cfg, weights, nc=1, channels=3):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = dict(nc=nc, channels=channels)
    return RTDETRTrainer.get_model(trainer, cfg=deepcopy(cfg), weights=weights, verbose=False)


def build_training_model(cfg, weights, data, variant=MAIN):
    """Native rebuild; preserve learned DPR values on nc=1 checkpoint restoration."""
    require(weights is not None, "A controlled initialization/checkpoint is required")
    channels = data.get("channels", 3)
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, data["nc"], channels)
    target = native_rebuild(cfg, weights, data["nc"], channels)
    verify_model(target, variant, zero=False)
    before, after = weights.state_dict(), target.state_dict()
    prefix = "model.26."
    allowed = {prefix + "denoising_class_embed.weight", prefix + "enc_score_head.weight", prefix + "enc_score_head.bias"}
    allowed |= {prefix + f"dec_score_head.{i}.{name}" for i in range(3) for name in ("weight", "bias")}
    require(set(before) == set(after), "Trainer rebuild changed state inventory")
    changed = {key for key in before if before[key].shape != after[key].shape}
    require(changed == (allowed if weights.model[-1].nc != data["nc"] else set()), "Unexpected Trainer class adaptation")
    require(all(torch.equal(value, after[key]) for key, value in before.items() if key not in changed),
            "Native Trainer reload changed an inherited/DPR tensor")
    require(all(torch.equal(value, after[key]) for key, value in parent.state_dict().items()),
            "Native nc1 parent/candidate public states differ")
    report = dict(status="PASSED", variant=variant, native_get_model=True, target_nc=data["nc"],
                  COMMON=[dict(name=key, shape=list(value.shape), equal=True, sha256=tensor_sha256(value))
                          for key, value in parent.state_dict().items()],
                  NEW_TRAINABLE=list(DPR_KEYS), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                  ALLOWED_CLASS_ADAPTATION=[dict(name=key, source_shape=list(before[key].shape),
                      target_shape=list(after[key].shape), exact_parent_nc1=True, sha256=tensor_sha256(after[key]))
                      for key in sorted(changed)],
                  loaded_exact=len(before) - len(changed), parent_nc1_public_exact=True,
                  parent_nc1_public_states=len(parent.state_dict()), learned_dpr_preserved=True,
                  parameters=sum(p.numel() for p in target.parameters()))
    return target, report


class _RebuildAuditStop(RuntimeError):
    pass


class RebuildAuditTrainer(RTDETRTrainer):
    """Exercise native Trainer construction with metadata, stopping before training."""
    def get_dataset(self):
        return dict(nc=1, channels=3, names={0: "crack"})

    def get_model(self, cfg=None, weights=None, verbose=True):
        variant = next((key for key, row in VARIANTS.items()
                        if Path(cfg.get("yaml_file", "")).name == row[0]), None)
        require(variant is not None, "Unknown initialization YAML at actual train entry")
        model, self.dpr_rebuild_audit = build_training_model(cfg, weights, self.data, variant)
        return model

    def train(self):
        require(isinstance(self.model, RTDETRDetectionModel), "Native train entry did not rebuild model")
        raise _RebuildAuditStop("Controlled audit stop before optimizer, dataloader, or any training batch")


def audit_train_entry(checkpoint_path, variant=MAIN):
    entry = RTDETR(str(checkpoint_path))
    random_state, numpy_state = random.getstate(), np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic, cudnn_benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    try:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), tempfile.TemporaryDirectory(prefix="dpr_rebuild_") as tmp:
            # No package update lookup is relevant to a local model reconstruction audit.
            with patch("ultralytics.engine.model.checks.check_pip_update_available", return_value=None), \
                    patch("ultralytics.engine.trainer.select_device", return_value=torch.device("cpu")), \
                    patch.dict(os.environ):
                try:
                    entry.train(trainer=RebuildAuditTrainer, data="metadata-only-rebuild-audit.yaml", device="cpu",
                                project=tmp, name="rebuild", epochs=200, seed=42, batch=16, imgsz=640,
                                workers=0, verbose=False, plots=False, save=False)
                except _RebuildAuditStop:
                    pass
                else:
                    raise RuntimeError("Rebuild audit failed to stop before training")
        report = deepcopy(entry.trainer.dpr_rebuild_audit)
        report.update(actual_train_entry=True, path="RTDETR.train -> RTDETRTrainer.__init__ -> get_model",
                      real_dataset_used=False, optimizer_steps=0, batches=0,
                      device_selection="Audit fixes CPU selection without changing CUDA_VISIBLE_DEVICES",
                      boundary="Stopped before optimizer/dataloader; this is reconstruction evidence only")
        verify_model(entry.model, variant, zero=True)
        return report
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_deterministic, cudnn_benchmark


def initialize(source, output, variant=MAIN, verify_existing=False):
    output = Path(output)
    require(not output.exists() or verify_existing, f"Existing initialization preserved: use --verify-existing: {output}")
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                      date=datetime.now(timezone.utc).isoformat(), dpr_provenance=deepcopy(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with output.open("xb") as stream:
            torch.save(checkpoint, stream)
    else:
        existing = torch_load(output, map_location="cpu")
        require(existing.get("epoch") == -1 and existing.get("optimizer") is None and existing.get("ema") is None,
                "Existing file is not an untrained initialization")
        require(existing.get("dpr_provenance", {}).get("source_sha256") == SOURCE_SHA256 and
                existing.get("dpr_provenance", {}).get("variant") == variant, "Existing initialization provenance differs")
    restored = RTDETR(str(output)).model
    require(set(target.state_dict()) == set(restored.state_dict()) and
            all(torch.equal(value, restored.state_dict()[key]) for key, value in target.state_dict().items()),
            "Serialized initialization differs")
    verify_model(restored, variant, zero=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        rebuilt, native_report = build_training_model(restored.yaml, restored, dict(nc=1, channels=3), variant)
    verify_model(rebuilt, variant, zero=True)
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True,
                  native_trainer_rebuild=native_report, train_entry_rebuild=audit_train_entry(output, variant),
                  runtime=runtime(), status="PASSED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default=MAIN)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-existing", action="store_true", help="Reaudit an existing exact controlled init without replacing it")
    args = parser.parse_args()
    output = args.output or ROOT / "weights" / f"{args.variant}_controlled_init.pt"
    report_path = args.report or ROOT / "outputs/dpr" / args.variant / "init.json"
    require(not report_path.exists(), f"Existing report preserved: {report_path}")
    torch.set_num_threads(4)
    try:
        report = initialize(args.source, output, args.variant, args.verify_existing)
    except Exception as error:
        write_json(report_path, dict(status="FAILED", variant=args.variant, error=repr(error),
                                    output=str(output), source=str(args.source)))
        raise
    write_json(report_path, report)
    print(f"PASSED: {report_path.resolve()}")


if __name__ == "__main__":
    main()
