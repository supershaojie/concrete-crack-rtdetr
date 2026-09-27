"""ARG fixed identities, complete recipe and reproducibility evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

from init_c19_lif_v1 import source_contract, initialize, SOURCE_SHA256, sha256, require, runtime as mother_runtime
from ultralytics.models.rtdetr.arg_loss import FORMULA
from ultralytics.models.rtdetr.arg_model import write_json
from ultralytics.utils import YAML

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-arg-v1"
ORIGIN = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
RUN_NAME = "arg_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SESSION = "arg-v1-training"
MAIN = Path(os.environ.get("ARG_V1_MAIN", str(ROOT.parent / "Crack_RTDETR") if os.name == "nt" else "/root/autodl-tmp/projects/Crack_RTDETR"))
OUT = ROOT / "outputs/arg_v1"
RUN = MAIN / "runs/c_series" / RUN_NAME
INIT = OUT / "arg_v1_init.pt"
SOURCE = MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"
SERVER_PYTHON = "/root/miniconda3/envs/rtdetr/bin/python"
COUNTS = dict(train=(6048, 45573), val=(1728, 12840), test=(864, 6663))


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=30).strip()


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def runtime():
    import inspect
    import torch
    from ultralytics.models.rtdetr.arg_loss import ARGDetectionLoss
    return dict(mother_runtime(), criterion=inspect.getfile(ARGDetectionLoss), module_hashes=source_contract(),
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                formula=FORMULA)


def code_identity(clean=False):
    require(git("remote", "get-url", "origin") == ORIGIN, "Unexpected origin")
    require(git("branch", "--show-current") in (BRANCH, ""), "Not ARG branch/detached delivery")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True, timeout=30)
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if clean:
        require(not dirty, "ARG worktree is dirty; preserve changes and commit before server operations")
    files = git("ls-files", "tools", "ultralytics-main/ultralytics", "configs", "docs/c19_lif_v1/c2_args.yaml", "docs/c19_lif_v1/c2_data.yaml").splitlines()
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}
    return dict(commit=git("rev-parse", "HEAD"), source_lf_sha256=digest(hashes), dirty=dirty, files=hashes, base=BASE)


def data_config():
    expected = YAML.load(ROOT / "docs/c19_lif_v1/c2_data.yaml")
    expected["path"] = str((MAIN / "datasets/crack_det").resolve())
    path = MAIN / "configs/crack_autodl.yaml"
    if os.name == "nt":
        path = OUT / "local_data.yaml"
        if not path.exists():
            YAML.save(path, expected)
    actual = YAML.load(path)
    actual["path"] = str(Path(actual["path"]).resolve())
    require(actual == expected, "Data config differs from mother's path/splits/classes")
    return path


def inventory(data_path, save_to=None):
    """Content hashes for images AND label bytes; fixed split, no dataset rewriting."""
    import math
    from PIL import Image
    data = YAML.load(data_path)
    root = Path(data["path"])
    rows, splits = [], {}
    for split, expected in COUNTS.items():
        folder = root / data[split]
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
        require(bool(images), f"Missing {split}: {folder}")
        count, local = 0, []
        for index, image in enumerate(images):
            label = root / "labels" / split / image.relative_to(folder).with_suffix(".txt")
            values = [list(map(float, line.split())) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            require(all(len(v) == 5 and v[0] == 0 and all(math.isfinite(x) for x in v)
                        and all(0 <= x <= 1 for x in v[1:]) and v[3] > 0 and v[4] > 0 for v in values), f"Invalid label: {label}")
            with Image.open(image) as im:
                shape = [im.height, im.width]
            row = dict(split=split, image=image.relative_to(root).as_posix(), image_sha256=sha256(image),
                       label=label.relative_to(root).as_posix(), label_sha256=sha256(label), boxes=len(values), size_hw=shape)
            local.append(row)
            count += len(values)
            if (index + 1) % 1000 == 0 or index + 1 == len(images):
                print(f"ARG data identity {split}: {index + 1}/{len(images)} images hashed", flush=True)
        require((len(images), count) == expected, f"{split} counts {(len(images), count)} != historical {expected}")
        splits[split] = dict(images=len(images), boxes=count, content_sha256=digest(local))
        rows.extend(local)
    if save_to:
        import gzip
        with gzip.open(save_to, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    return dict(splits=splits, content_sha256=digest(rows), config_sha256=sha256(data_path), config=data)


def recipe(data_path):
    source = YAML.load(ROOT / "docs/c19_lif_v1/c2_args.yaml")
    actual = MAIN / "runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
    if actual.exists():
        actual_args = YAML.load(actual)
        require(set(actual_args) == set(source) and all(type(actual_args[k]) is type(v) and actual_args[k] == v for k, v in source.items()), "Actual C2 args differ from authoritative archive")
    target = dict(source, model=str(INIT), data=str(data_path), project=str(RUN.parent), name=RUN_NAME, save_dir=str(RUN))
    changes = [{"field": k, "mother": v, "arg": target[k], "reason": "experiment/path identity"} for k, v in source.items() if v != target[k]]
    require({r["field"] for r in changes} <= {"model", "data", "project", "name", "save_dir"}, "Illegal recipe change")
    return target, dict(fields=len(source), authoritative="docs/c19_lif_v1/c2_args.yaml", actual_archive_checked=actual.exists(), changes=changes, loss_parameters=FORMULA)


def binding(clean=True, verify_data=True):
    plan = read_json(OUT / "prepare.json")
    require(plan is not None and plan["status"] == "PASS", "Run prepare successfully first")
    code = code_identity(clean)
    require(sha256(SOURCE) == SOURCE_SHA256 and sha256(INIT) == plan["init_sha256"], "Source/init changed")
    path = data_config()
    data = inventory(path) if verify_data else plan["data"]
    require(data == plan["data"], "Data identity changed since prepare")
    args, diff = recipe(path)
    require(args == plan["args"], "Recipe changed since prepare")
    return dict(code=code, data=data, init_sha256=plan["init_sha256"], args_sha256=digest(args), formula=FORMULA)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    code = code_identity(clean=True)
    if os.name == "posix":
        sync = read_json(OUT / "sync.json", {})
        require(sync.get("commit") == code["commit"] and Path(sync.get("worktree", "/missing")).resolve() == ROOT,
                "Run the delivered sync_arg_v1.sh with the exact SHA first")
    source_contract()
    require(SOURCE.is_file() and sha256(SOURCE) == SOURCE_SHA256, "Missing/wrong public source checkpoint")
    data_path = data_config()
    data = inventory(data_path, OUT / "data_manifest.jsonl.gz")
    args, diff = recipe(data_path)
    if INIT.exists():
        prior = read_json(OUT / "initialization.json")
        require(prior and prior["output_sha256"] == sha256(INIT), "Existing init has no matching provenance; preserved")
    else:
        write_json(OUT / "initialization.json", initialize(SOURCE, INIT))
    record = dict(status="PASS", created=now(), code=code, runtime=runtime(), args=args, data=data,
                  source_sha256=SOURCE_SHA256, init_sha256=sha256(INIT), recipe_diff=diff)
    prior = read_json(OUT / "prepare.json")
    if prior:
        require(all(record[k] == prior[k] for k in ("args", "data", "source_sha256", "init_sha256")), "Existing prepare identity differs; preserved")
        return prior
    YAML.save(OUT / "train_args.yaml", args)
    write_json(OUT / "recipe_diff.json", diff)
    write_json(OUT / "prepare.json", record)
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(OUT / "source_snapshot.tar.gz"), "HEAD"], cwd=ROOT, check=True, timeout=60)
    return record
