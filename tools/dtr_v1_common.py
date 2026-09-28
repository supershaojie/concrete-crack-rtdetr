"""DTR identities and fixed recipe. Reuses the original C19 public initializer."""
from __future__ import annotations
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
from experiment_runtime import require, now, write_json, read_json, sha256, digest, file_info
from init_c19_lif_v1 import source_contract, initialize, SOURCE_SHA256, runtime as mother_runtime
from ultralytics.models.rtdetr.dtr_loss import FORMULA, DTRDetectionLoss
from ultralytics.models.rtdetr.dtr_model import DTRDetectionModel
from ultralytics.utils import YAML

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-dtr-v1"
ORIGIN = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
RUN_NAME = "dtr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SESSION = "dtr-v1-training"
MAIN = Path(os.environ.get("DTR_V1_MAIN", "D:/MyProjects/Crack_RTDETR" if os.name == "nt"
                           else "/root/autodl-tmp/projects/Crack_RTDETR")).resolve()
OUT = ROOT / "outputs/dtr_v1"
RUN = ROOT / "runs/c_series" / RUN_NAME
INIT = OUT / "dtr_v1_init.pt"
SOURCE = MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"
SERVER_PYTHON = "/root/miniconda3/envs/rtdetr/bin/python"
COUNTS = dict(train=(6048, 45573), val=(1728, 12840), test=(864, 6663))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=30).strip()


def runtime():
    import torch
    return dict(mother_runtime(), criterion=inspect.getfile(DTRDetectionLoss),
                model=inspect.getfile(DTRDetectionModel), formula=FORMULA,
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None)


def code_identity(clean=False):
    require(git("remote", "get-url", "origin") == ORIGIN, "Unexpected origin")
    require(git("branch", "--show-current") in (BRANCH, ""), "Wrong DTR branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True, timeout=30)
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if clean:
        require(not dirty, "DTR worktree is dirty; preserve/commit changes before server operations")
    names = git("ls-files", "tools", "ultralytics-main/ultralytics", "configs", "docs/c19_lif_v1").splitlines()
    files = {n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in names}
    return dict(commit=git("rev-parse", "HEAD"), base=BASE, source_lf_sha256=digest(files), files=files, dirty=dirty)


def data_config():
    data = YAML.load(ROOT / "docs/c19_lif_v1/c2_data.yaml")
    data["path"] = str((MAIN / "datasets/crack_det").resolve())
    path = MAIN / "configs/crack_autodl.yaml" if os.name == "posix" else OUT / "local_data.yaml"
    if os.name == "nt" and not path.exists():
        YAML.save(path, data)
    actual = YAML.load(path)
    actual["path"] = str(Path(actual["path"]).resolve())
    require(actual == data, "Data YAML differs from mother's paths/splits/classes")
    return path


def canonical_data(data):
    # ARG ef9cb7e: JSON persists class keys as strings.
    names = {str(k): v for k, v in data["names"].items()}
    require(len(names) == len(data["names"]), "Colliding class keys")
    return dict(data, names=names)


def directory_stamps(config):
    root = Path(config["path"])
    dirs = [root / branch / split for branch in ("images", "labels") for split in COUNTS]
    return {str(p): p.stat().st_mtime_ns for p in dirs}


def snapshot(data_path, refresh=False, source=None, manifest=None):
    """One confirmed snapshot; downstream operations never enumerate split images."""
    config = canonical_data(YAML.load(data_path))
    saved = read_json(OUT / "data_snapshot.json")
    if saved and not refresh:
        require(saved["config"] == config, "Snapshot paths/classes changed; explicit prepare --refresh-data required")
        require(saved["config_sha256"] == sha256(data_path), "Data YAML fingerprint changed; restore it or explicitly refresh")
        require(saved["directory_stamps"] == directory_stamps(config), "Data directory changed; explicit refresh required")
        require(saved["manifest_sha256"] == sha256(OUT / "data_manifest.jsonl.gz"), "Snapshot manifest changed")
        return saved
    candidates = []
    if source:
        require(manifest is not None, "--snapshot requires --manifest")
        candidates.append((Path(source), Path(manifest)))
    for folder in (MAIN / "outputs/arg_v1", MAIN.parent / "Crack_RTDETR-arg_v1/outputs/arg_v1"):
        candidates += [(folder / "prepare.json", folder / "data_manifest.jsonl.gz"),
                       (folder / "local_inventory.json", folder / "local_data_manifest.jsonl.gz")]
    reused, rows = None, []
    if not refresh:
        for identity_path, manifest_path in candidates:
            if not identity_path.is_file() or not manifest_path.is_file():
                continue
            identity = read_json(identity_path)
            identity = identity.get("data", identity)
            if identity.get("config") != config:
                continue
            with gzip.open(manifest_path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            require(digest(rows) == identity["content_sha256"], "Prior data manifest fingerprint mismatch")
            reused = dict(identity=str(identity_path), manifest=str(manifest_path), sha256=sha256(manifest_path))
            break
    if not reused:
        from PIL import Image
        import math
        root = Path(config["path"])
        for split in COUNTS:
            folder = root / config[split]
            images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
            for i, image in enumerate(images):
                label = root / "labels" / split / image.relative_to(folder).with_suffix(".txt")
                values = [list(map(float, line.split())) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                require(all(len(v) == 5 and v[0] == 0 and all(math.isfinite(x) for x in v)
                            and all(0 <= x <= 1 for x in v[1:]) and v[3] > 0 and v[4] > 0 for v in values), f"Invalid {label}")
                with Image.open(image) as im:
                    shape = [im.height, im.width]
                rows.append(dict(split=split, image=image.relative_to(root).as_posix(), image_sha256=sha256(image),
                    label=label.relative_to(root).as_posix(), label_sha256=sha256(label), boxes=len(values), size_hw=shape))
                if (i + 1) % 500 == 0 or i + 1 == len(images):
                    print(f"DTR snapshot {split}: {i+1}/{len(images)}", flush=True)
    splits = {}
    for split, counts in COUNTS.items():
        local = [r for r in rows if r["split"] == split]
        require((len(local), sum(r["boxes"] for r in local)) == counts, f"Unexpected {split} counts")
        splits[split] = dict(images=len(local), boxes=counts[1], content_sha256=digest(local))
    if saved:
        shutil.copy2(OUT / "data_snapshot.json", OUT / ("data_snapshot_" + saved["content_sha256"] + ".json"))
        shutil.copy2(OUT / "data_manifest.jsonl.gz", OUT / ("data_manifest_" + saved["content_sha256"] + ".jsonl.gz"))
    with gzip.open(OUT / "data_manifest.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    result = dict(config=config, config_sha256=sha256(data_path), splits=splits, content_sha256=digest(rows),
                  manifest_sha256=sha256(OUT / "data_manifest.jsonl.gz"), directory_stamps=directory_stamps(config),
                  created=now(), reused=reused,
                  verification="Confirmed immutable snapshot; no repeated content scan. In-place edits require explicit refresh.")
    write_json(OUT / "data_snapshot.json", result)
    return result


def recipe(data_path):
    source = YAML.load(ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml")
    archive = MAIN / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
    if archive.exists():
        actual = YAML.load(archive)
        require(actual == source and all(type(actual[k]) is type(v) for k, v in source.items()), "Mother actual args differ from archive")
    target = dict(source, model=str(INIT), data=str(data_path), project=str(RUN.parent), name=RUN_NAME, save_dir=str(RUN))
    changes = [dict(field=k, mother=v, dtr=target[k], reason="experiment/path identity") for k, v in source.items() if v != target[k]]
    require({r["field"] for r in changes} <= {"model", "data", "project", "name", "save_dir"}, "Illegal recipe change")
    return target, dict(fields=len(source), authoritative="docs/c19_lif_v1/resolved_formal_config.yaml",
                       actual_archive_checked=archive.exists(), changes=changes, loss_parameters=FORMULA)


def binding(clean=True):
    plan = read_json(OUT / "prepare.json")
    require(plan and plan["status"] == "PASS", "Run prepare first")
    require(SOURCE.is_file() and sha256(SOURCE) == SOURCE_SHA256, "Wrong public initialization source")
    require(INIT.is_file() and sha256(INIT) == plan["init_sha256"], "Prepared init changed")
    data_path = data_config()
    data = snapshot(data_path)
    args, _ = recipe(data_path)
    require(data == plan["data"] and args == plan["args"], "Data/recipe changed since prepare")
    return dict(code=code_identity(clean), data=data, init_sha256=plan["init_sha256"],
                source_sha256=SOURCE_SHA256, args_sha256=digest(args), formula=FORMULA)


def prepare(refresh=False, source=None, manifest=None):
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "training_identity.json").exists(), "Training identity exists; prepare cannot replace this run")
    code = code_identity(clean=True)
    source_contract()
    require(SOURCE.is_file() and sha256(SOURCE) == SOURCE_SHA256, "Missing/wrong public source")
    path = data_config()
    data = snapshot(path, refresh, source, manifest)
    args, diff = recipe(path)
    if INIT.exists():
        prior = read_json(OUT / "initialization.json")
        require(prior and prior["output_sha256"] == sha256(INIT), "Existing init lacks valid provenance")
    else:
        write_json(OUT / "initialization.json", initialize(SOURCE, INIT))
    result = dict(status="PASS", created=now(), code=code, runtime=runtime(), args=args, data=data,
                  source_sha256=SOURCE_SHA256, init_sha256=sha256(INIT), recipe_diff=diff)
    prior = read_json(OUT / "prepare.json")
    if prior:
        write_json(OUT / ("prepare_" + digest(prior) + ".json"), prior)
    YAML.save(OUT / "train_args.yaml", args)
    shutil.copy2(path, OUT / "data_config.yaml")
    write_json(OUT / "recipe_diff.json", diff)
    write_json(OUT / "prepare.json", result)
    return result


def strict_amp_resources():
    """Reuse mother's offline assets; never download a different AMP model."""
    from ultralytics.utils import ASSETS
    result = []
    for target, candidates in (
        (ASSETS / "bus.jpg", [MAIN / "ultralytics-main/ultralytics/assets/bus.jpg", MAIN / "bus.jpg"]),
        (ROOT / "yolo26n.pt", [MAIN / "yolo26n.pt", MAIN / "weights/yolo26n.pt"]),
    ):
        if not target.exists():
            src = next((p for p in candidates if p.is_file()), None)
            require(src, f"Missing mother's offline AMP asset: {target.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as a, target.open("xb") as b:
                shutil.copyfileobj(a, b)
        result.append(file_info(target))
    return result
