"""Strict SDB-P3 initialization from the fixed, untrained C2 source.

All original state paths (including CBR/LIF) are retained. Native nc adaptation
is audited against the corresponding parent at the exact same CPU RNG state.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

import torch
from init_lif_down import ROOT, MODEL_DIR, SOURCE_SHA256, C2_COMMIT, require, sha256, write_json
import init_c19_lif_v1 as pair
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import Conv, LIFDown, RTDETRDecoder, RTDETRDecoderCBR, SDBRepC3, SDBP3
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE_COMMIT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-sdb-p3-v1"
VARIANTS = {
    "cbr_lif_sdb_p3_v1": ("rtdetr-resnet18-lite-cbr-lif-sdb-p3-v1.yaml",
                          "rtdetr-resnet18-lite-cbr-lif-down.yaml",
                          "cbr_lif_sdb_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
    "sdb_p3_v1": ("rtdetr-resnet18-lite-sdb-p3-v1.yaml", "rtdetr-resnet18-lite.yaml",
                  "sdb_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug"),
}
PARAMETERS = {"cbr_lif_sdb_p3_v1": (20177861, 19973061), "sdb_p3_v1": (20110868, 19905812)}
NEW_PARAMETERS = 28096


def is_added(key):
    return key.startswith("model.19.sdb.")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime():
    info = pair.runtime()
    import ultralytics
    from ultralytics.nn.modules import sdb_p3
    require(Path(sdb_p3.__file__).resolve() == ROOT / "ultralytics-main/ultralytics/nn/modules/sdb_p3.py",
            "SDB imported from another checkout")
    info.update(sdb_module=sdb_p3.__file__, ultralytics_version=ultralytics.__version__,
                cudnn=torch.backends.cudnn.version(), base_commit=BASE_COMMIT)
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    # Bind all tracked execution/configuration files with cross-platform LF hashing.
    files += ["tools/init_sdb_p3.py", "tools/check_sdb_p3.py", "tools/train_sdb_p3.py",
              "ultralytics-main/ultralytics/nn/modules/sdb_p3.py"]
    info["source_hashes"] = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in sorted(set(files)) if (ROOT / p).is_file() and Path(p).suffix in {".py", ".yaml", ".sh"}}
    info["tracked_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    return info


def build(variant="cbr_lif_sdb_p3_v1", nc=80, baseline=False):
    require(variant in VARIANTS, "Unknown SDB variant")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(baseline)]), nc=nc, verbose=False)


def verify_model(model, variant="cbr_lif_sdb_p3_v1", zero=False):
    pair.source_contract()
    require(variant in VARIANTS and type(model) is RTDETRDetectionModel, "Wrong model/variant")
    parent = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected = deepcopy(parent)
    expected["head"][11] = [[18, 4], 3, "SDBRepC3", [256, 0.5, 32]]
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")),
            "SDB target may change only model.19 from its parent")
    require(len(model.model) == 27 and type(model.model[19]) is SDBRepC3, "Wrong model.19 wrapper")
    block = model.model[19]
    require(block.f == [18, 4] and 4 in model.save and len(block.m) == 3, "P2 save/from or repeats changed")
    require(type(block.sdb) is SDBP3 and sum(isinstance(m, SDBP3) for m in model.modules()) == 1, "Exactly one SDB required")
    require(block.cv1.conv.in_channels == 512 and block.cv1.conv.out_channels == 128,
            "RepC3 original c1/e changed")
    require(sum(p.numel() for p in block.sdb.parameters()) == NEW_PARAMETERS, "SDB parameter count changed")
    require(not list(block.sdb.buffers()), "SDB v1 has no buffers")
    sdb = block.sdb
    require((sdb.p2_channels, sdb.p3_channels, sdb.detail_channels) == (64, 256, 32),
            "SDB v1 channel contract changed")
    for name, cin, cout, kernel, padding, groups, bias in (
        ("W_d", 256, 32, 1, 0, 1, False), ("DW3", 32, 32, 3, 1, 32, False),
        ("W_q", 256, 32, 1, 0, 1, False), ("W_g", 96, 32, 1, 0, 1, True),
        ("W_o", 32, 256, 1, 0, 1, False),
    ):
        layer = getattr(sdb, name)
        require(type(layer) is torch.nn.Conv2d and
                (layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride,
                 layer.padding, layer.dilation, layer.groups, layer.bias is not None, layer.padding_mode) ==
                (cin, cout, (kernel, kernel), (1, 1), (padding, padding), (1, 1), groups, bias, "zeros"),
                f"SDB {name} convolution contract changed")
    for name in ("GN_D", "GN_Q"):
        norm = getattr(sdb, name)
        require(type(norm) is torch.nn.GroupNorm and
                (norm.num_groups, norm.num_channels, norm.eps, norm.affine) == (4, 32, 1e-5, True),
                f"SDB {name} GroupNorm contract changed")
    head = model.model[26]
    require(head.f == [19, 22, 25] and model.model[20].f == -1, "SDB P3 must feed both downsample and decoder")
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2,
            "Decoder geometry changed")
    combo = variant == "cbr_lif_sdb_p3_v1"
    require(type(model.model[20]) is (LIFDown if combo else Conv), "Wrong downsample")
    require(type(head) is (RTDETRDecoderCBR if combo else RTDETRDecoder), "Wrong decoder")
    require(type(model.model[23]) is Conv, "P4 downsample changed")
    if combo:
        require(head.cbr.rho == head.cbr.normal_fraction == .10, "Original CBR settings changed")
        require(hasattr(model.model[20], "bn"), "Original LIF BN protection lost")
    if head.nc == 1:
        fused = not hasattr(block.cv1, "bn")
        require(sum(p.numel() for p in model.parameters()) == PARAMETERS[variant][int(fused)], "Wrong nc1 parameter count")
    if zero:
        require(torch.count_nonzero(block.sdb.W_o.weight).item() == 0, "SDB output must start at zero")
        require(torch.count_nonzero(sdb.W_g.bias).item() == 0, "SDB gate bias must start at zero")
        for name in ("W_d", "DW3", "W_q", "W_g"):
            weight = getattr(sdb, name).weight
            require(torch.isfinite(weight).all().item() and torch.count_nonzero(weight).item() > 0,
                    f"SDB {name} must start finite and nonzero")
        for name in ("GN_D", "GN_Q"):
            norm = getattr(sdb, name)
            require(torch.equal(norm.weight, torch.ones_like(norm.weight)) and
                    torch.equal(norm.bias, torch.zeros_like(norm.bias)),
                    f"SDB {name} affine initialization changed")
        if combo:
            require(torch.count_nonzero(model.model[20].O_proj.weight).item() == 0, "LIF init changed")
            require(all(torch.count_nonzero(p).item() == 0 for p in head.cbr.offset_out.parameters()), "CBR init changed")
    return {"p2": 4, "concat": 18, "p3": 19, "downsample": 20, "p4": 22, "p5": 25, "decoder": 26}


def audit_common(parent, target):
    before, after = parent.state_dict(), target.state_dict()
    new = set(after) - set(before)
    require(new == {k for k in after if is_added(k)}, "Unexpected added states")
    require(not (set(before) - set(after)), "Missing parent states")
    require(all(v.shape == after[k].shape and torch.equal(v, after[k]) for k, v in before.items()),
            "Original parameter/buffer values differ")
    params = dict(target.named_parameters())
    require(new <= params.keys(), "Unexpected SDB buffers")
    return dict(COMMON=[dict(name=k, shape=list(v.shape), sha256=tensor_hash(v), equal=True) for k, v in before.items()],
                NEW_TRAINABLE=sorted(new), NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[],
                ALLOWED_CLASS_ADAPTATION=[], new_parameters=NEW_PARAMETERS)


def controlled_models(source, variant="cbr_lif_sdb_p3_v1"):
    # Reuse the successful parent's strict untrained-source/CBR/LIF provenance checks.
    c2, combo, parent_report = pair.controlled_models(source)
    parent = combo if variant == "cbr_lif_sdb_p3_v1" else c2
    target = build(variant)
    fresh = target.state_dict()
    constructed_parent = build(variant, baseline=True)
    audit_common(constructed_parent, target)
    target.load_state_dict({**fresh, **parent.state_dict()}, strict=True)
    report = audit_common(parent, target)
    other = build("sdb_p3_v1" if variant == "cbr_lif_sdb_p3_v1" else "cbr_lif_sdb_p3_v1")
    require(all(torch.equal(v, other.state_dict()[k]) for k, v in target.state_dict().items() if is_added(k)),
            "Variants have different SDB initialization")
    verify_model(target, variant, zero=True)
    report.update(source=str(Path(source).resolve()), source_sha256=sha256(source), source_nc=80, target_nc=80,
                  variant=variant, base_commit=BASE_COMMIT, c2_commit=C2_COMMIT, seed=42,
                  parent_audit=parent_report, constructor_common_exact=True, variants_new_exact=True,
                  new_state_hashes={k: tensor_hash(v) for k, v in target.state_dict().items() if is_added(k)})
    return parent, target, report


def class_adaptation_keys():
    prefix = "model.26."
    return {prefix + "denoising_class_embed.weight", prefix + "enc_score_head.weight", prefix + "enc_score_head.bias"} | {
        prefix + f"dec_score_head.{i}.{part}" for i in range(3) for part in ("weight", "bias")}


def build_training_model(cfg, weights, data, variant="cbr_lif_sdb_p3_v1"):
    require(weights is not None, "Controlled weights required")
    require(data["nc"] == 1, "This experiment requires nc=1")
    channels = data.get("channels", 3)
    # Reference must not advance native target classification-adaptation randomness.
    with torch.random.fork_rng(devices=[]):
        parent = pair.native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, data["nc"], channels)
    target = pair.native_rebuild(cfg, weights, data["nc"], channels)
    verify_model(target, variant)
    before, after = weights.state_dict(), target.state_dict()
    require(set(before) == set(after), "Native rebuild changed state inventory")
    changed = {k for k in before if before[k].shape != after[k].shape}
    allowed = class_adaptation_keys() if weights.model[-1].nc != data["nc"] else set()
    require(changed == allowed, "Only the nine original nc classification tensors may adapt")
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in changed),
            "Trainer overwrote initialized/learned state")
    report = audit_common(parent, target)
    report.update(ALLOWED_CLASS_ADAPTATION=[dict(name=k, source_shape=list(before[k].shape), target_shape=list(after[k].shape),
                       parent_nc1_equal=torch.equal(parent.state_dict()[k], after[k])) for k in sorted(changed)],
                  native_get_model=True, parent_nc1_common_exact=True, learned_new_state_preserved=True,
                  loaded_exact=len(before) - len(changed), target_nc=1)
    return target, report


def initialize(source, output, variant="cbr_lif_sdb_p3_v1"):
    output = Path(output)
    require(not output.exists(), f"Existing initial checkpoint preserved: {output}")
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                      date=datetime.now(timezone.utc).isoformat(), sdb_p3_provenance=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    reloaded = RTDETR(str(output)).model
    require(set(target.state_dict()) == set(reloaded.state_dict()) and
            all(torch.equal(v, reloaded.state_dict()[k]) for k, v in target.state_dict().items()), "Serialized init changed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        _, native_audit = build_training_model(reloaded.yaml, reloaded, {"nc": 1, "channels": 3}, variant)
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True,
                  native_nc1=native_audit, runtime=runtime(), status="PASSED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="cbr_lif_sdb_p3_v1")
    for name in ("source", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.report, initialize(args.source, args.output, args.variant))


if __name__ == "__main__":
    main()
