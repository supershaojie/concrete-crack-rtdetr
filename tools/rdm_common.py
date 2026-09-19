"""Small shared contracts for RDM; reuse the successful mother's initialization."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from init_c19_lif_v1 import (
    ROOT, MODEL_DIR, SOURCE_SHA256, require, sha256, runtime, source_contract,
    controlled_models as mother_models, native_rebuild,
)
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import RDM, RDMBlocks, LIFDown, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-rdm-v1"
MAIN = Path(os.environ.get("RDM_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
VARIANTS = {
    "cbr_lif_rdm_v1": ("rtdetr-resnet18-lite-cbr-lif-rdm-v1.yaml", "rtdetr-resnet18-lite-cbr-lif-down.yaml", 20162661, 19957861),
    "rdm_v1": ("rtdetr-resnet18-lite-rdm-v1.yaml", "rtdetr-resnet18-lite.yaml", 20095668, 19890612),
}
NEW = {f"model.5.rdm.{name}.weight" for name in ("Wd", "Dl", "Dc", "Pc", "Wg", "Wo")} | {"model.5.rdm.Wg.bias"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def lf_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def paths(variant):
    name = variant + "_rtdetr_r18_lite_e200_b16_onlineaug"
    return dict(name=name, run=MAIN / "runs/c_series" / name,
                output=ROOT / "outputs/rdm_v1" / variant,
                init=ROOT / "weights" / (variant + "_controlled_init.pt"),
                source=MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                data=Path(os.environ.get("RDM_DATA", str(MAIN / "configs/crack_autodl.yaml"))))


def code_identity():
    # LF-normalized contents, not commit IDs: a documentation commit needs no rerun.
    files = list((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files += list((ROOT / "tools").glob("*.py")) + list((ROOT / "tools").glob("*.sh"))
    files += list(MODEL_DIR.glob("*.yaml"))
    files += [ROOT / "docs/rdm_v1/parent_args.yaml", ROOT / "docs/rdm_v1/data_identity.json"]
    manifest = {p.relative_to(ROOT).as_posix(): lf_hash(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=manifest)


def build(variant, nc=80, parent=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[variant][int(parent)]), nc=nc, verbose=False)


def verify_model(model, variant, zero=False):
    source_contract()
    expected = deepcopy(YAML.load(MODEL_DIR / VARIANTS[variant][1]))
    expected["backbone"][5] = [-1, 1, "RDMBlocks", [128, "BasicBlock", 2, 3, "relu", 32]]
    require(type(model) is RTDETRDetectionModel, "Native RTDETR model required")
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")), "Only model.5 may differ from this parent")
    stage = model.model[5]
    require(type(stage) is RDMBlocks and len(stage.blocks) == 2, "RDM must follow two complete BasicBlocks")
    require(sum(isinstance(m, RDM) for m in model.modules()) == 1, "Exactly one RDM required")
    require(model.model[6].f == -1 and model.model[17].f == 5, "Both original P3 consumers required")
    require({k for k in model.state_dict() if ".rdm." in k} == NEW, "Exactly seven new states required")
    require(not list(stage.rdm.buffers()), "RDM must not have tensor buffers")
    require(sum(p.numel() for p in stage.rdm.parameters()) == 12896, "RDM parameter delta changed")
    head = model.model[26]
    require(head.hidden_dim == 256 and head.num_queries == 300 and len(head.decoder.layers) == 3 and head.decoder.eval_idx == 2, "Decoder contract changed")
    if variant == "cbr_lif_rdm_v1":
        require(type(model.model[20]) is LIFDown and type(head) is RTDETRDecoderCBR, "Original CBR/LIF required")
        require(head.cbr.rho == head.cbr.normal_fraction == .10, "Original CBR rho changed")
    if head.nc == 1:
        # Native is_fused() counts all norms and remains False for this R18
        # backbone after native fuse(). Inspect its actual neck Conv-BN state.
        expected_count = VARIANTS[variant][2 if hasattr(model.model[8], "bn") else 3]
        require(sum(p.numel() for p in model.parameters()) == expected_count, "Model parameter count differs")
    if zero:
        require(torch.count_nonzero(stage.rdm.Wo.weight) == 0, "Fresh Wo must be zero")
        require(torch.count_nonzero(stage.rdm.Wg.bias) == 0, "Fresh gate bias must be zero")
        require(all(torch.count_nonzero(getattr(stage.rdm, n).weight) > 0 for n in ("Wd", "Dl", "Dc", "Pc", "Wg")), "Only Wo weight may be zero")


def controlled_models(source, variant):
    # The successful mother's routine validates the nc80/epoch=-1 common source
    # and constructs original C2 and CBR+LIF without borrowing third modules.
    c2, pair, _ = mother_models(source)
    parent = pair if variant == "cbr_lif_rdm_v1" else c2
    target = build(variant)
    fresh_parent = build(variant, parent=True)
    public, fresh = parent.state_dict(), target.state_dict()
    require(all(torch.equal(v, fresh[k]) for k, v in fresh_parent.state_dict().items()), "RDM constructor changed public RNG")
    missing = sorted(set(public) - set(fresh))
    extra = set(fresh) - set(public)
    mismatch = sorted(k for k in public if k in fresh and public[k].shape != fresh[k].shape)
    require(not missing and not mismatch and extra == NEW, "Unexpected common/new state inventory")
    target.load_state_dict({**fresh, **public}, strict=True)
    require(all(torch.equal(v, target.state_dict()[k]) for k, v in public.items()), "Common values not transferred exactly")
    verify_model(target, variant, zero=True)
    return parent, target, dict(COMMON=len(public), NEW=sorted(NEW), MISSING=missing, UNEXPECTED=[], SHAPE_MISMATCH=mismatch,
                               common_parameters_and_buffers_exact=True, public_constructor_equal=True, new_buffers=[],
                               source_sha256=SOURCE_SHA256, source_nc=80, source_epoch=-1)


def training_rebuild(cfg, weights, data, variant, fresh=True):
    # Reference construction consumes no caller RNG. Candidate uses native flow.
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR / VARIANTS[variant][1]), weights, data["nc"], data["channels"])
    target = native_rebuild(cfg, weights, data["nc"], data["channels"])
    verify_model(target, variant, zero=fresh)
    before, after = weights.state_dict(), target.state_dict()
    adapted = {k for k in before if before[k].shape != after[k].shape}
    allowed = {"model.26." + n for n in ("denoising_class_embed.weight", "enc_score_head.weight", "enc_score_head.bias")}
    allowed |= {f"model.26.dec_score_head.{i}.{s}" for i in range(3) for s in ("weight", "bias")}
    require(adapted == (allowed if weights.model[26].nc != data["nc"] else set()), "Unexpected nc adaptation")
    require(set(before) == set(after) and all(torch.equal(v, after[k]) for k, v in before.items() if k not in adapted), "Trainer lost loaded state")
    require(all(torch.equal(v, after[k]) for k, v in parent.state_dict().items()), "Adapted parent/common state differs")
    return target, dict(native_get_model=True, COMMON=len(after) - 7, NEW=sorted(NEW), MISSING=[], UNEXPECTED=[],
                        SHAPE_MISMATCH=[], ALLOWED_CLASS_ADAPTATION=sorted(adapted), public_exact=True, new_exact=True)


def initialize(source, output, variant):
    _, target, audit = controlled_models(source, variant)
    output = Path(output)
    reused = output.exists()
    if reused:
        checkpoint = torch_load(output, map_location="cpu")
        require(checkpoint.get("epoch") == -1 and all(checkpoint.get(k) is None for k in
                ("ema", "optimizer", "scaler", "updates", "best_fitness", "train_metrics", "train_results")), "Existing init has trained state")
        require(checkpoint.get("rdm_provenance", {}).get("variant") == variant, "Existing init belongs to another variant")
        restored = checkpoint["model"].float()
    else:
        target.eval()
        target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
        target.task, target.pt_path = "detect", str(output.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            torch.save(dict(epoch=-1, model=deepcopy(target).float(), ema=None, optimizer=None, scaler=None,
                            updates=None, best_fitness=None, train_metrics=None, train_results=None, train_args=target.args,
                            date=datetime.now(timezone.utc).isoformat(), rdm_provenance=dict(variant=variant, **audit)), stream)
        restored = torch_load(output, map_location="cpu")["model"].float()
    verify_model(restored, variant, zero=True)
    require(set(restored.state_dict()) == set(target.state_dict()) and all(torch.equal(v, restored.state_dict()[k]) for k, v in target.state_dict().items()), "Init value mismatch; preserve file and investigate")
    return dict(status="PASSED", variant=variant, reused=reused, output=str(output), output_sha256=sha256(output), **audit)


def recipe(variant):
    original = YAML.load(ROOT / "docs/rdm_v1/parent_args.yaml")
    p = paths(variant)
    target = dict(original, model=str(p["init"]), data=str(p["data"]), project=str(MAIN / "runs/c_series"), name=p["name"], save_dir=str(p["run"]))
    return target, [dict(field=k, parent=v, target=target[k], changed=v != target[k]) for k, v in original.items()]


def data_identity(data):
    from c19_lif_v1_data import dataset_inventory
    actual = YAML.load(data)
    expected = YAML.load(ROOT / "docs/c19_lif_v1/c2_data.yaml")
    root = (MAIN / "datasets/crack_det").resolve()
    expected["path"] = str(root)
    normalized = dict(actual, path=str(Path(actual["path"]).resolve()))
    require(normalized == expected, "Data splits/classes/root differ from mother")
    inventory = dataset_inventory(root)
    frozen = json.loads((ROOT / "docs/rdm_v1/data_identity.json").read_text())
    require(inventory == frozen, "Data path/label inventory differs from verified mother data")
    # Validate box geometry as well as identity; no resplitting or image copying.
    for label in (root / "labels").glob("*/*.txt"):
        for row in label.read_text().splitlines():
            values = list(map(float, row.split()))
            if values:
                require(len(values) == 5 and values[0] == 0 and all(0 <= x <= 1 for x in values[1:]) and values[3] > 0 and values[4] > 0, "Invalid GT geometry: " + str(label))
    return canonical(dict(config=normalized, inventory=inventory))


def identity(variant, include_data=True):
    p = paths(variant)
    return dict(variant=variant, code=code_identity(), init_sha256=sha256(p["init"]) if p["init"].is_file() else None,
                source_sha256=sha256(p["source"]) if p["source"].is_file() else None,
                data=data_identity(p["data"]) if include_data else None, recipe=canonical(recipe(variant)[0]))
