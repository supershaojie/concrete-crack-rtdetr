"""Pinned TCR identities, strict parent initialization, fingerprints and JSON."""
from __future__ import annotations

import contextlib
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import numpy as np
import torch
import ultralytics
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules import ConvTCR, LIFDown, RTDETRDecoderCBR
from ultralytics.models.rtdetr.train import RTDETRTrainer
import init_c19_lif_v1 as parent

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-tcr-v1"
MAIN = Path(os.environ.get("TCR_V1_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
OUT = ROOT / "outputs/tcr_v1"
DOC = ROOT / "docs/tcr_v1"
MODEL = parent.MODEL_DIR / "rtdetr-resnet18-lite-cbr-lif-down-tcr-v1.yaml"
NAME = "tcr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
RUN = MAIN / "runs/c_series" / NAME
INIT = ROOT / "weights/tcr_v1_controlled_init.pt"
SOURCE = MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
SESSION = "tcr-v1-training"
SOURCE_SHA = parent.SOURCE_SHA256
NEW = {"model.17.tcr.P.weight", "model.17.tcr.O.weight"}
require, sha256 = parent.require, parent.sha256


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean_json(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def append_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json(data), ensure_ascii=False, allow_nan=False) + "\n")


def file_identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path)) if path.is_file() else dict(path=str(path), status="PENDING")


def source_identity():
    names = git("ls-files").splitlines()
    # Includes only execution/config inputs, never generated timestamp reports.
    names = sorted(set(n for n in names if n.startswith(("tools/", "ultralytics-main/ultralytics/"))))
    names += ["docs/tcr_v1/research.yaml", "docs/tcr_v1/mother_args.yaml", MODEL.relative_to(ROOT).as_posix()]
    rows = {n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in names if (ROOT / n).is_file()}
    # Before commit include new TCR files as well.
    for folder in (ROOT / "tools", ROOT / "ultralytics-main/ultralytics/nn/modules"):
        for p in folder.glob("*tcr*"):
            if p.is_file():
                rows[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    payload = json.dumps(rows, sort_keys=True).encode()
    return dict(sha256=hashlib.sha256(payload).hexdigest(), files=rows)


def runtime():
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "ultralytics-main"), "Wrong ultralytics import")
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, version=ultralytics.__version__, commit=git("rev-parse", "HEAD"),
                original_modules=parent.source_contract())


@contextlib.contextmanager
def isolated_rng():
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


def build(nc=1):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL), nc=nc, verbose=False)


def verify_model(model, zero=False):
    parent.source_contract()
    expected = deepcopy(YAML.load(parent.MODEL_DIR / parent.CONFIGS["pair"]))
    expected["head"][17 - len(expected["backbone"])][2] = "ConvTCR"
    require(type(model) is RTDETRDetectionModel, "Native RTDETR model required")
    require(all(model.yaml[k] == expected[k] for k in ("head", "backbone", "scales")), "TCR topology/config drift")
    require(type(model.model[17]) is ConvTCR and model.model[17].f == 5, "TCR must follow original node 17")
    require(sum(isinstance(m, ConvTCR) for m in model.modules()) == 1, "Exactly one TCR required")
    require(model.model[17].tcr.enabled, "Formal TCR must be enabled")
    require(type(model.model[20]) is LIFDown and hasattr(model.model[20], "bn"), "Original LIF fusion protection lost")
    head = model.model[26]
    require(type(head) is RTDETRDecoderCBR and head.f == [19, 22, 25], "CBR paths changed")
    require(head.num_queries == 300 and len(head.decoder.layers) == 3 and head.cbr.rho == .1, "Decoder contract changed")
    require(sum(p.numel() for p in model.model[17].tcr.parameters()) == 36864, "Wrong TCR count")
    if head.nc == 1 and hasattr(model.model[17], "bn"):
        require(sum(p.numel() for p in model.parameters()) == 20186629 and len(model.state_dict()) == 554, "Unexpected nc1 inventory")
    if zero:
        require(torch.count_nonzero(model.model[17].tcr.O.weight) == 0, "TCR O initialization must be zero")
    return dict(node=17, from_node=5, head_nodes=[19, 22, 25], added_parameters=36864,
                parameters=sum(p.numel() for p in model.parameters()), states=len(model.state_dict()))


def controlled(source):
    _, mother, inherited = parent.controlled_models(source)
    target = build(nc=80)
    public, fresh = mother.state_dict(), target.state_dict()
    require(set(fresh) - set(public) == NEW and not set(public) - set(fresh), "Unexpected state keys")
    constructor = parent.build()
    require(all(torch.equal(v, fresh[k]) for k, v in constructor.state_dict().items()), "TCR disturbed mother construction/RNG")
    target.load_state_dict({**fresh, **public}, strict=True)
    require(all(torch.equal(v, target.state_dict()[k]) for k, v in public.items()), "Public values differ")
    verify_model(target, zero=True)
    return mother, target, dict(parent_source_audit=inherited, new_keys=sorted(NEW), missing=[], unexpected=[],
                               constructor_equal=True, common=[dict(name=k, shape=list(v.shape), equal=True) for k, v in public.items()])


def native_rebuild(cfg, weights, data, resume=False):
    dummy = RTDETRTrainer.__new__(RTDETRTrainer)
    dummy.data = data
    # The isolated parent consumes precisely the same incoming RNG as native target.
    if not resume:
        with torch.random.fork_rng(devices=[]):
            mother = RTDETRTrainer.get_model(dummy, cfg=str(parent.MODEL_DIR / parent.CONFIGS["pair"]), weights=weights, verbose=False)
    model = RTDETRTrainer.get_model(dummy, cfg=deepcopy(cfg), weights=weights, verbose=False)
    verify_model(model, zero=not resume)
    old, new = weights.state_dict(), model.state_dict()
    allow = {"model.26." + n for n in ("denoising_class_embed.weight", "enc_score_head.weight", "enc_score_head.bias")}
    allow |= {f"model.26.dec_score_head.{i}.{s}" for i in range(3) for s in ("weight", "bias")}
    changed = {k for k in old if old[k].shape != new[k].shape}
    require(set(old) == set(new), "Unknown missing/unexpected Trainer state")
    require(changed == (allow if weights.model[-1].nc != data["nc"] else set()), "Unknown class adaptation")
    require(all(torch.equal(v, new[k]) for k, v in old.items() if k not in changed), "Native load changed initialized tensors")
    if not resume:
        require(all(torch.equal(v, new[k]) for k, v in mother.state_dict().items()), "Native nc1 mother common tensors differ")
    return model, dict(native_get_model=True, resume=resume, common_exact=True, allowed_class_adaptation=sorted(changed),
                       common_states=len(new) - 2, new_keys=sorted(NEW), missing=[], unexpected=[])


def initialize(source, output):
    output = Path(output)
    require(not output.exists(), f"Existing initialization preserved: {output}")
    _, target, report = controlled(source)
    from ultralytics.cfg import DEFAULT_CFG_DICT
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL), "task": "detect"}
    target.eval()
    checkpoint = dict(epoch=-1, best_fitness=None, model=target.float(), ema=None, updates=None,
                      optimizer=None, scaler=None, train_args=target.args, train_metrics=None, train_results=None,
                      tcr_v1=dict(base=BASE, source_sha256=SOURCE_SHA, research=YAML.load(DOC / "research.yaml")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        torch.save(checkpoint, f)
    restored = torch_load(output, map_location="cpu")["model"]
    require(all(torch.equal(v, restored.state_dict()[k]) for k, v in target.state_dict().items()), "Serialization differs")
    report.update(status="PASS", source=file_identity(source), output=file_identity(output), reload_exact=True)
    return report


def recipe():
    mother = YAML.load(DOC / "mother_args.yaml")
    require(len(mother) == 109, "Expected complete 109-field mother recipe")
    target = dict(mother)
    target.update(model=str(INIT), name=NAME, save_dir=str(RUN))
    require(MAIN.as_posix() == "/root/autodl-tmp/projects/Crack_RTDETR", "Formal root migration requires an explicit content audit")
    rows = [dict(field=k, mother=mother[k], tcr=target[k], changed=mother[k] != target[k]) for k in mother]
    require({r["field"] for r in rows if r["changed"]} == {"model", "name", "save_dir"}, "Recipe changed beyond identity")
    return target, rows


def optimizer_audit(model, optimizer):
    ids = [id(v) for group in optimizer.param_groups for v in group["params"]]
    params = list(model.named_parameters())
    require(all(p.requires_grad and ids.count(id(p)) == 1 for _, p in params), "Frozen, missing or duplicated optimizer parameter")
    require(len(ids) == len(params), "Unexpected optimizer tensor")
    names = {id(p): n for n, p in params}
    return [dict(index=i, lr=g["lr"], weight_decay=g["weight_decay"], param_group=g.get("param_group"),
                 parameters=[names[id(p)] for p in g["params"]]) for i, g in enumerate(optimizer.param_groups)]
