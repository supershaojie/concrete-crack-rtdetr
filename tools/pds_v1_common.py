"""Shared PDS identity, exact public-state audit and strict report writing."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
from init_c19_lif_v1 import (build, build_training_model, controlled_models, verify_model,
                             source_contract, SOURCE_SHA256, require, sha256, runtime)
from ultralytics.nn.modules.pds import CONFIG
from ultralytics.models.rtdetr.pds import PDSDetectionModel
from ultralytics.utils import YAML

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-pds-v1"
RUN_NAME = "pds_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
MAIN = Path(os.environ.get("PDS_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR")).resolve()
OUT = ROOT / "outputs/pds_v1"
INIT = ROOT / "weights/pds_v1_controlled_init.pt"
RUN = MAIN / "runs/c_series" / RUN_NAME
SOURCE = MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"


def utc():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    """Strict JSON never hides a nonfinite value or replaces the original exception."""
    bad = []
    def clean(x, loc="$"):
        if isinstance(x, torch.Tensor):
            return clean(x.detach().cpu().tolist(), loc)
        if isinstance(x, float) and not math.isfinite(x):
            bad.append(loc)
            return None
        if isinstance(x, dict):
            return {str(k): clean(v, loc + "." + str(k)) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v, f"{loc}[{i}]") for i, v in enumerate(x)]
        if isinstance(x, Path):
            return str(x)
        return x
    result = clean(value)
    if bad:
        result = dict(result, nonfinite=True, nonfinite_paths=bad)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    import ultralytics
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    require(Path(ultralytics.__file__).resolve() == ROOT / "ultralytics-main/ultralytics/__init__.py", "Wrong import root")
    available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    info = dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if available else None,
                ultralytics=ultralytics.__file__, commit=git("rev-parse", "HEAD"))
    info.update(criterion=inspect.getfile(RTDETRDetectionLoss), base=BASE, branch=git("branch", "--show-current"),
                module_hashes=source_contract(), clip_max_norm=10., cuda_available=available)
    if available:
        info.update(gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    cudnn=torch.backends.cudnn.version())
    return info


def state_audit(native, wrapped):
    a, b = native.state_dict(), wrapped.state_dict()
    extra = sorted(set(b) - set(a))
    require(len(a) == 552 and extra == sorted("pds_head." + k for k in wrapped.pds_head.state_dict()),
            "Public/PDS key inventory mismatch")
    require(len(extra) == 11 and set(a) <= set(b), "Unexpected public key removal")
    rows = []
    for k, v in a.items():
        require(v.shape == b[k].shape and torch.equal(v, b[k]), f"Public value differs: {k}")
        rows.append(dict(name=k, shape=list(v.shape), sha256=hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()))
    return dict(public=rows, public_states=len(a), added=extra, added_buffers=[],
                public_parameters=sum(p.numel() for p in native.parameters()),
                training_parameters=sum(p.numel() for p in wrapped.parameters()))


def native_view(model):
    """Independent native object; no mutation of a live training module."""
    from copy import deepcopy
    from ultralytics.nn.tasks import RTDETRDetectionModel
    native = deepcopy(model)
    if isinstance(native, PDSDetectionModel):
        del native.pds_head
        for attr in ("pds_epoch", "pds_stats", "pds_config"):
            if hasattr(native, attr):
                delattr(native, attr)
        native.__class__ = RTDETRDetectionModel
    verify_model(native)
    return native


def recipe():
    original = YAML.load(ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml")
    target = dict(original)
    target.update(model=str(INIT), data=str(MAIN / "configs/crack_autodl.yaml"),
                  project=str(RUN.parent), name=RUN_NAME, save_dir=str(RUN))
    require(len(target) == 109, "Expected complete 109-field parent args")
    return target


def check_clean():
    require(not git("status", "--porcelain", "--untracked-files=no"), "Tracked worktree modifications: refuse production")


def source_fingerprint():
    paths = git("ls-files", "tools", "ultralytics-main/ultralytics", "docs/pds_v1", "docs/c19_lif_v1",
                "configs", ".gitattributes").splitlines()
    return digest({p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths})


def archive_file(path):
    path = Path(path)
    if path.exists():
        path.rename(path.with_name(path.name + "." + utc() + ".archive"))
