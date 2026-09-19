"""BLC identity, strict state migration and auditable experiment inputs."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch
from init_c19_lif_v1 import (ROOT, MODEL_DIR, SOURCE_SHA256, require, sha256, runtime,
                             source_contract, controlled_models as pair_models, native_rebuild)
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import BLC, BLCBlocks, LIFDown, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-blc-v1"
MAIN = Path(os.environ.get("BLC_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
VARIANTS = {
    "cbr_lif_blc_v1": ("rtdetr-resnet18-lite-cbr-lif-blc-v1.yaml", "rtdetr-resnet18-lite-cbr-lif-down.yaml"),
    "blc_v1": ("rtdetr-resnet18-lite-blc-v1.yaml", "rtdetr-resnet18-lite.yaml"),
}
COUNTS = {"cbr_lif_blc_v1": (20158326, 19953526), "blc_v1": (20091333, 19886277)}
KEYS = {"model.5.blc." + s for s in ("Wd.weight", "Wg.weight", "Wg.bias", "Wo.weight")}


def write_json(path, value):
    """Exclusive reports: a failed run can never silently replace older evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def serializable(item):
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)  # Explicit "nan"/"inf" evidence, never an invalid JSON number.
        if isinstance(item, dict):
            return {k: serializable(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [serializable(v) for v in item]
        return item
    with path.open("x", encoding="utf-8") as stream:
        json.dump(serializable(value), stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        stream.write("\n")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def paths(variant):
    require(variant in VARIANTS, "Unknown BLC variant")
    name = variant + "_rtdetr_r18_lite_e200_b16_onlineaug"
    return dict(name=name, run=MAIN / "runs/c_series" / name,
                init=ROOT / "weights" / (variant + "_controlled_init.pt"),
                source=MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                evidence=ROOT / "outputs/blc_v1" / variant,
                data=Path(os.environ.get("BLC_DATA", str(MAIN / "configs/crack_autodl.yaml"))))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def code_identity():
    # Include every dependency affecting initialization, native train/val, and BLC
    # evidence. LF normalization permits a Windows -> Linux checkout.
    files = list((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files += list((ROOT / "ultralytics-main/ultralytics/cfg").rglob("*.yaml"))
    files += list((ROOT / "tools").glob("*.py")) + list((ROOT / "tools").glob("*blc*.sh"))
    files += [ROOT / "docs/blc_v1/parent_args.yaml", ROOT / "docs/blc_v1/parent_dataset_inventory.json"]
    rows = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(set(files))}
    return dict(sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), files=rows)


def build(variant, nc=1, parent=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(parent)]), nc=nc, verbose=False)


def verify_model(model, variant, zero=False, fused=False):
    source_contract()
    expected = YAML.load(MODEL_DIR / VARIANTS[variant][1])
    expected["backbone"][5] = [-1, 1, "BLCBlocks", [128, "BasicBlock", 2, 3, "relu", 32]]
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "BLC topology mismatch")
    stage = model.model[5]
    require(type(stage) is BLCBlocks and len(stage.blocks) == 2, "BLC must wrap original stage 5")
    require(sum(isinstance(m, BLC) for m in model.modules()) == 1, "Exactly one BLC required")
    require(model.model[6].f == -1 and model.model[17].f == 5, "Both P3 consumers must use enhanced P3")
    require(set(n for n, _ in model.named_parameters() if ".blc." in n) == KEYS, "Unexpected new parameters")
    require(not list(stage.blc.buffers()), "BLC has no tensor buffers")
    require(sum(p.numel() for p in stage.blc.parameters()) == 8561, "BLC parameter count")
    head = model.model[26]
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3
            and head.decoder.eval_idx == 2, "Decoder contract changed")
    if variant == "cbr_lif_blc_v1":
        require(type(model.model[20]) is LIFDown and type(head) is RTDETRDecoderCBR, "Original CBR/LIF missing")
        require(head.cbr.rho == head.cbr.normal_fraction == .10, "CBR rho changed")
    else:
        require(not any(isinstance(m, (LIFDown, RTDETRDecoderCBR)) for m in model.modules()), "Single BLC includes CBR/LIF")
    if head.nc == 1:
        require(sum(p.numel() for p in model.parameters()) == COUNTS[variant][int(fused)], "Parameter count mismatch")
    if zero:
        require(torch.count_nonzero(stage.blc.Wd.weight) > 0, "Wd must be nonzero")
        require(all(torch.count_nonzero(p) == 0 for p in
                    (stage.blc.Wg.weight, stage.blc.Wg.bias, stage.blc.Wo.weight)), "BLC zero initialization changed")


def audit_states(parent, candidate):
    before, after = parent.state_dict(), candidate.state_dict()
    new = set(after) - set(before)
    report = dict(COMMON=[dict(name=k, shape=list(v.shape), equal=k in after and torch.equal(v, after[k]),
                              sha256=tensor_hash(v)) for k, v in before.items()],
                  NEW_TRAINABLE=sorted(new & dict(candidate.named_parameters()).keys()),
                  NEW_BUFFER=sorted(new - dict(candidate.named_parameters()).keys()),
                  MISSING=sorted(set(before) - set(after)), UNEXPECTED=sorted(new - KEYS),
                  SHAPE_MISMATCH=[k for k in before if k in after and before[k].shape != after[k].shape])
    require(new == KEYS and not any(report[k] for k in ("NEW_BUFFER", "MISSING", "UNEXPECTED", "SHAPE_MISMATCH")),
            "State key/shape inventory differs")
    require(all(row["equal"] for row in report["COMMON"]), "Common state value changed")
    return report


def rebuild(cfg, weights, variant, nc=1):
    """Native get_model with an isolated corresponding parent construction."""
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, nc)
    target = native_rebuild(cfg, weights, nc)
    verify_model(target, variant)
    audit = audit_states(parent, target)
    before, after = weights.state_dict(), target.state_dict()
    allowed = {"model.26." + n for n in ("denoising_class_embed.weight", "enc_score_head.weight", "enc_score_head.bias")}
    allowed |= {f"model.26.dec_score_head.{i}.{s}" for i in range(3) for s in ("weight", "bias")}
    changed = {k for k in before if before[k].shape != after[k].shape}
    require(changed == (allowed if weights.model[-1].nc != nc else set()), "Unexpected class adaptation")
    require(all(torch.equal(v, after[k]) for k, v in before.items() if k not in changed), "Trainer overwrote initialized state")
    audit.update(ALLOWED_CLASS_ADAPTATION=sorted(changed), native_get_model=True, nc=nc,
                 classification_parent_candidate_exact=True, new_state_exact=True)
    return target, audit


def initialize(source, output, variant):
    output = Path(output)
    require(not output.exists(), f"Existing controlled initialization preserved: {output}")
    c2, pair, parent_audit = pair_models(source)
    parent = pair if variant == "cbr_lif_blc_v1" else c2
    candidate = build(variant, nc=80)
    constructor = audit_states(parent=build(variant, nc=80, parent=True), candidate=candidate)
    candidate.load_state_dict({**candidate.state_dict(), **parent.state_dict()}, strict=True)
    transfer = audit_states(parent, candidate)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        target, trainer_audit = rebuild(candidate.yaml, candidate, variant)
    verify_model(target, variant, zero=True)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(output), "task": "detect"}
    target.task, target.pt_path = "detect", str(output)
    report = dict(status="PASSED", variant=variant, source_sha256=sha256(source), parent=BASE,
                  parent_initialization=parent_audit, constructor=constructor, transfer=transfer,
                  nc1_rebuild=trainer_audit, code=code_identity(), runtime=runtime())
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, blc_provenance=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    loaded = RTDETR(str(output)).model
    require(all(torch.equal(v, loaded.state_dict()[k]) for k, v in target.state_dict().items()), "Init serialization changed values")
    report.update(output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True)
    return report


class BLCTrainer(RTDETRTrainer):
    """Only audits native reconstruction; training math and grouping stay native."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(weights is not None, "BLC start requires the controlled checkpoint")
        variant = "cbr_lif_blc_v1" if cfg["head"][-1][2] == "RTDETRDecoderCBR" else "blc_v1"
        model, audit = rebuild(cfg, weights, variant, self.data["nc"])
        self.blc_rebuild_audit = audit
        return model


def recipe(variant):
    p = paths(variant)
    parent = YAML.load(ROOT / "docs/blc_v1/parent_args.yaml")
    require(len(parent) == 109, "Expected exact 109-field parent recipe")
    result = {**parent, "model": str(p["init"]), "name": p["name"], "save_dir": str(p["run"]),
              "project": str(p["run"].parent), "data": str(p["data"])}
    return result, [dict(field=k, parent=parent[k], candidate=v) for k, v in result.items() if parent[k] != v]


def dataset_identity(data):
    from c19_lif_v1_data import dataset_inventory
    from ultralytics.data.utils import check_det_dataset
    spec = YAML.load(data)
    require(spec.get("train") == "images/train" and spec.get("val") == "images/val"
            and spec.get("test") == "images/test" and spec.get("names") == {0: "crack"}, "Data split/classes changed")
    resolved = check_det_dataset(str(data), autodownload=False)
    actual = dataset_inventory(Path(resolved["path"]))
    expected = json.loads((ROOT / "docs/blc_v1/parent_dataset_inventory.json").read_text(encoding="utf-8"))
    require(actual == expected, "Image lists/label hashes differ from the successful parent; counts alone are insufficient")
    return dict(inventory=actual, data_yaml=spec, resolved_root=str(resolved["path"]), data_sha256=sha256(data))


def evidence_context(variant, data=True):
    p = paths(variant)
    return dict(variant=variant, code=code_identity(), init_sha256=sha256(p["init"]),
                source_sha256=sha256(p["source"]), data=dataset_identity(p["data"]) if data else None,
                recipe=recipe(variant)[0], runtime=runtime())
