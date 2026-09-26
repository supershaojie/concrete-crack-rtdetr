"""Pinned PEQ identities, reproducibility and strict finite JSON utilities."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-peq-v1"
REMOTE = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
MAIN = Path(os.environ.get("PEQ_V1_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR")).resolve()
OUT = ROOT / "outputs/peq_v1"
RUN_NAME = "peq_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
RUN = MAIN / "runs/c_series" / RUN_NAME
SESSION = "peq-v1-training"
PYTHON = "/root/miniconda3/envs/rtdetr/bin/python"
SOURCE = MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
SOURCE_SHA = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
BEST_SHA = "24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa"
INIT = ROOT / "weights/peq_v1_controlled_init.pt"
MODEL_YAML = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down-peq-v1.yaml"
MODULE_HASHES = dict(lif_down="26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7",
                     cbr="d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), status="PRESENT", bytes=path.stat().st_size, sha256=sha256(path)) if path.is_file() else dict(path=str(path.resolve()), status="PENDING")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def clean_json(value, nonfinite=None, location="$"):
    if nonfinite is None:
        nonfinite = []
    if isinstance(value, dict):
        return {str(k): clean_json(v, nonfinite, f"{location}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v, nonfinite, f"{location}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist(), nonfinite, location)
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist(), nonfinite, location)
    if isinstance(value, np.generic):
        return clean_json(value.item(), nonfinite, location)
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(location)
        return None
    return value


def json_bytes(value):
    nonfinite = []
    result = clean_json(value, nonfinite)
    if nonfinite:
        if not isinstance(result, dict):
            result = dict(value=result)
        result["nonfinite"] = True
        result["nonfinite_locations"] = nonfinite
    return (json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode("utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp_{os.getpid()}")
    temporary.write_bytes(json_bytes(value))
    os.replace(temporary, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = json.loads(json_bytes(value))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(safe, ensure_ascii=False, allow_nan=False)+"\n")


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def source_identity():
    # Hash only implementation/config, normalized to LF; neither reports nor timestamps nor HEAD.
    names = git("ls-files").splitlines()
    names += [p.relative_to(ROOT).as_posix() for folder in (ROOT/"tools", ROOT/"ultralytics-main/ultralytics", ROOT/"docs/peq_v1")
              for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    names = sorted({n for n in names if
        (n.startswith("ultralytics-main/ultralytics/") and Path(n).suffix in (".py", ".yaml")) or
        (n.startswith("tools/") and (Path(n).suffix in (".py", ".sh"))) or
        n in ("docs/peq_v1/research.yaml", "docs/peq_v1/mother_args.yaml")})
    files = {name: hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in names}
    return dict(sha256=digest_json(files), files=files)


def module_contract():
    result = {}
    for name, expected in MODULE_HASHES.items():
        path = ROOT/f"ultralytics-main/ultralytics/nn/modules/{name}.py"
        result[name] = dict(raw_sha256=sha256(path), lf_sha256=hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
        require(result[name]["lf_sha256"] == expected, f"Original {name} changed")
    return result


def environment():
    import ultralytics
    from ultralytics.models.rtdetr.peq_model import PEQDetectionModel
    from ultralytics.models.utils.peq_loss import PEQDetectionLoss
    require(Path(ultralytics.__file__).resolve() == ROOT/"ultralytics-main/ultralytics/__init__.py", "Installed pip package is shadowing worktree")
    return dict(python=platform.python_version(), executable=sys.executable, torch=torch.__version__,
                cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                total_gpu_memory=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                ultralytics_version=ultralytics.__version__, ultralytics_file=ultralytics.__file__,
                model_type=f"{PEQDetectionModel.__module__}.{PEQDetectionModel.__name__}",
                criterion_type=f"{PEQDetectionLoss.__module__}.{PEQDetectionLoss.__name__}",
                platform=platform.platform())


@contextmanager
def isolated_rng():
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


def dataset_identity(data_path, splits=("train", "val", "test"), strict_counts=True):
    """Read identity only. Probe passes splits=('val',) so it never reads test labels."""
    from ultralytics.utils import YAML
    data_path = Path(data_path)
    cfg = YAML.load(data_path)
    root = Path(cfg["path"]).resolve()
    require(cfg["names"] in ({0: "crack"}, ["crack"]), "Expected one crack class")
    expected = {"train": (6048,45573), "val": (1728,12840), "test": (864,6663)}
    output = {}
    for split in splits:
        require(cfg[split] == f"images/{split}", "Split layout changed")
        folder = root/cfg[split]
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"))
        require(files, f"Empty/missing {split}")
        paths, labels, images, boxes = [], [], [], 0
        for image in files:
            label = root/"labels"/split/image.relative_to(folder).with_suffix(".txt")
            require(label.is_file(), f"Missing label {label}")
            rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            values = np.asarray(rows, dtype=np.float64).reshape(-1,5)
            require(np.isfinite(values).all() and np.all(values[:,0]==0) and
                    np.all(values[:,3:]>0) and np.all(values[:,1:]>=0) and np.all(values[:,1:]<=1), f"Invalid labels {label}")
            rel = image.relative_to(root).as_posix()
            paths.append(rel)
            labels.append([label.relative_to(root).as_posix(),sha256(label)])
            images.append([rel,image.stat().st_size,sha256(image)])
            boxes += len(rows)
        require(boxes > 0, f"Missing GT in {split}")
        if strict_counts:
            require((len(files),boxes)==expected[split], f"Unexpected {split} counts: {(len(files),boxes)}")
        output[split] = dict(images=len(files), ground_truth=boxes, image_list=paths,
                             paths_sha256=digest_json(paths), labels_sha256=digest_json(labels),
                             images_sha256=digest_json(images))
    return dict(config=identity(data_path), root=str(root), splits=output,
                limitation="Known same-source augmented images cross splits; same-protocol comparison only")
