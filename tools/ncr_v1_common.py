"""NCR run identity and read-only evidence helpers (no model import at module import)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
PARENT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-ncr-v1"
ORIGIN = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
RUN_NAME = "ncr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SOURCE_SHA = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
NCR = dict(lambda_ncr=.25, min_side_px=4., huber_delta=1., ramp_start=4, ramp_epochs=15)
COUNTS = dict(train=(6048, 45573), val=(1728, 12840), test=(864, 6663))
POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300, augment=False,
            rect=False, seed=42, device="0", plots=True, save_json=False, save_txt=False, deterministic=True)
PARENT_MAP = dict(val=.52454272, test=.52200902)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def paths(main=MAIN):
    main = Path(main).resolve()
    return dict(main=main, output=ROOT/"outputs/ncr_v1", run=main/"runs/c_series"/RUN_NAME,
                source=main/"weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                init=ROOT/"weights/ncr_v1_controlled_init.pt", data=main/"configs/crack_autodl.yaml")


def recipe(main=MAIN):
    p = paths(main)
    parent = yaml.safe_load((ROOT/"docs/ncr_v1/parent_args.yaml").read_text(encoding="utf-8"))
    archived = yaml.safe_load((ROOT/"docs/c19_lif_v1/resolved_formal_config.yaml").read_text(encoding="utf-8"))
    require(set(parent) == set(archived), "Full parent archive fields differ")
    differences = {k for k in parent if parent[k] != archived[k]}
    require(differences <= {"model"}, "Actual parent training archive differs from committed recipe")
    args = dict(parent, model=str(p["init"]), data=str(p["data"]), name=RUN_NAME,
                project=str(p["run"].parent), save_dir=str(p["run"]))
    rows = [dict(field=k, parent=parent[k], ncr=args[k], reason="experiment identity / absolute root relocation")
            for k in parent if parent[k] != args[k]]
    require({r["field"] for r in rows} <= {"model", "data", "name", "project", "save_dir"}, "Recipe drift")
    return args, rows


def environment():
    import numpy as np
    import torch
    import ultralytics
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT/"ultralytics-main"), "Wrong package import")
    return dict(python=platform.python_version(), executable=sys.executable, torch=str(torch.__version__),
                numpy=np.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__)


def source_identity():
    require(git("merge-base", PARENT, "HEAD") == PARENT, "Experiment does not derive from the specified parent")
    unchanged = ["ultralytics-main/ultralytics/nn/modules/cbr.py", "ultralytics-main/ultralytics/nn/modules/lif_down.py",
                 "ultralytics-main/ultralytics/nn/modules/head.py", "ultralytics-main/ultralytics/nn/modules/transformer.py",
                 "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml",
                 "ultralytics-main/ultralytics/models/utils/ops.py", "ultralytics-main/ultralytics/utils/loss.py"]
    for name in unchanged:
        expected = subprocess.check_output(["git", "show", PARENT+":"+name], cwd=ROOT).replace(b"\r\n", b"\n")
        require((ROOT/name).read_bytes().replace(b"\r\n", b"\n") == expected, "Parent source identity changed: "+name)
    files = set(git("ls-files", "ultralytics-main/ultralytics", "tools").splitlines())
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"tools").glob("*ncr_v1*.*"))
    files.add("ultralytics-main/ultralytics/models/utils/ncr.py")
    files.update("docs/ncr_v1/"+name for name in ("parent_args.yaml", "resolved_formal_config.yaml", "parent_dataset_inventory.json", "recipe_diff.json"))
    hashes = {n: hashlib.sha256((ROOT/n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for n in sorted(files) if (ROOT/n).is_file() and Path(n).suffix in {".py", ".sh", ".yaml", ".json"}}
    return dict(head=git("rev-parse", "HEAD"), parent=PARENT, origin=git("remote", "get-url", "origin"),
                branch=git("branch", "--show-current"), files_sha256_lf=hashes, code_sha256=digest(hashes))


def data_identity(data, main):
    from c19_lif_v1_data import dataset_inventory
    actual = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    expected = dict(path=str(Path(main).resolve()/"datasets/crack_det"), train="images/train", val="images/val",
                    test="images/test", names={0: "crack"})
    actual["path"] = str(Path(actual["path"]).resolve())
    require(actual == expected, "Dataset path, splits or classes differ from parent")
    inventory = dataset_inventory(Path(actual["path"]))
    for split, count in COUNTS.items():
        require((inventory[split]["images"], inventory[split]["boxes"]) == count, "Dataset counts differ: "+split)
    require(inventory == read(ROOT/"docs/ncr_v1/parent_dataset_inventory.json"), "Dataset split paths or label hashes differ from the archived parent")
    return dict(yaml_sha256=sha(data), config=actual, splits=inventory)


def identity(main=MAIN, data=None):
    p = paths(main); args, changes = recipe(main)
    data = Path(data or p["data"])
    require(p["source"].is_file() and sha(p["source"]) == SOURCE_SHA, "Public ImageNet initialization missing/hash mismatch")
    result = dict(source=source_identity(), training_args=args, recipe_sha256=digest(args), recipe_changes=changes,
                  ncr=NCR, public_init_sha256=SOURCE_SHA, data=data_identity(data, main), environment=environment())
    # YAML class IDs use integer keys; JSON manifests use strings. Canonicalize
    # before both recording AND comparing so valid preflights can be reused.
    return json.loads(json.dumps(result, default=str))
