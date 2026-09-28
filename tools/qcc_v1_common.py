"""QCC fixed identities, complete recipe and reproducibility evidence."""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

from init_c19_lif_v1 import source_contract, initialize, SOURCE_SHA256, sha256, require, runtime as mother_runtime
from ultralytics.models.rtdetr.qcc_loss import FORMULA
from ultralytics.models.rtdetr.qcc_io import write_json
from ultralytics.utils import YAML

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-qcc-v1"
ORIGIN = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
RUN_NAME = "qcc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SESSION = "qcc-v1-training"
MAIN = Path(os.environ.get("QCC_V1_MAIN", "D:/MyProjects/Crack_RTDETR" if os.name == "nt" else "/root/autodl-tmp/projects/Crack_RTDETR"))
OUT = ROOT / "outputs/qcc_v1"
RUN = MAIN / "runs/c_series" / RUN_NAME
INIT = OUT / "qcc_v1_init.pt"
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
    from ultralytics.models.rtdetr.qcc_loss import QCCDetectionLoss
    return dict(mother_runtime(), criterion=inspect.getfile(QCCDetectionLoss), module_hashes=source_contract(),
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                formula=FORMULA)


def code_identity(clean=False):
    require(git("remote", "get-url", "origin") == ORIGIN, "Unexpected origin")
    require(git("branch", "--show-current") in (BRANCH, ""), "Not QCC branch/detached delivery")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True, timeout=30)
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if clean:
        require(not dirty, "QCC worktree is dirty; preserve changes and commit before server operations")
    files = git("ls-files", "tools", "ultralytics-main/ultralytics", "configs", "docs/c19_lif_v1/c2_args.yaml", "docs/c19_lif_v1/c2_data.yaml", "docs/c19_lif_v1/resolved_formal_config.yaml").splitlines()
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
    require(data_identity_config(actual) == data_identity_config(expected), "Data config differs from mother's path/splits/classes")
    return path


def data_identity_config(data):
    """Copy config with JSON-stable class IDs; never merge colliding IDs."""
    names = {}
    source = data["names"]
    for class_id, name in (enumerate(source) if isinstance(source, list) else source.items()):
        key = str(class_id)
        require(key.isdigit() and str(int(key)) == key, f"Invalid class ID {class_id!r}")
        require(key not in names, f"Class ID collision after string normalization: {key!r}")
        names[key] = name
    return dict(data, names=names)


def inventory(data_path, save_to=None):
    """Content hashes for images AND label bytes; fixed split, no dataset rewriting."""
    import math
    from PIL import Image
    data = data_identity_config(YAML.load(data_path))
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
                print(f"QCC data identity {split}: {index + 1}/{len(images)} images hashed", flush=True)
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
    mother = YAML.load(ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml")
    identity_fields = {"model", "data", "project", "name", "save_dir"}
    require(set(source) == set(mother) and all(type(mother[k]) is type(v) and mother[k] == v
        for k, v in source.items() if k not in identity_fields), "Mother paired full recipe differs from archived C2 recipe")
    actual = MAIN / "runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
    if actual.exists():
        actual_args = YAML.load(actual)
        require(set(actual_args) == set(source) and all(type(actual_args[k]) is type(v) and actual_args[k] == v for k, v in source.items()), "Actual C2 args differ from authoritative archive")
    target = dict(source, model=str(INIT), data=str(data_path), project=str(RUN.parent), name=RUN_NAME, save_dir=str(RUN))
    changes = [{"field": k, "mother": v, "qcc": target[k], "reason": "experiment/path identity"} for k, v in source.items() if v != target[k]]
    require({r["field"] for r in changes} <= {"model", "data", "project", "name", "save_dir"}, "Illegal recipe change")
    return target, dict(fields=len(source), authoritative="docs/c19_lif_v1/c2_args.yaml",
        paired_mother="docs/c19_lif_v1/resolved_formal_config.yaml", complete_nonidentity_recipe_equal=True,
        actual_archive_checked=actual.exists(), changes=changes, loss_parameters=FORMULA)


def cached_data():
    """Config + existing identity only: no inventory or image/label traversal."""
    saved = read_json(OUT / "data_identity.json")
    require(saved and saved["status"] == "PASS", "Missing prepared data snapshot; run prepare")
    current = data_identity_config(YAML.load(data_config()))
    require(current == saved["data"]["config"], "Data config changed: explicit prepare --recheck-data required")
    for path, stamp in saved.get("directory_stamps", {}).items():
        require(Path(path).is_dir() and Path(path).stat().st_mtime_ns == stamp,
                f"Data directory changed: {path}; explicitly recheck-data")
    return saved["data"]


def prepare_data(path, reuse=None, recheck=False):
    """Reuse fixed known snapshots, or hash raw data once, only in prepare."""
    import gzip
    import shutil
    prior = read_json(OUT / "data_identity.json")
    if prior and not recheck:
        cached_data()
        return prior["data"]
    config = data_identity_config(YAML.load(path))
    candidates = [Path(reuse)] if reuse else [MAIN.parent / "Crack_RTDETR-arg_v1/outputs/arg_v1/prepare.json"]
    used = None
    if not recheck:
        for candidate in candidates:
            if not candidate.is_file():
                if reuse:
                    raise FileNotFoundError(candidate)
                continue
            record = read_json(candidate)
            data = record.get("data")
            require(record.get("status") == "PASS" and data, f"Unqualified identity: {candidate}")
            old = data_identity_config(data["config"])
            old["path"] = str(Path(old["path"]).resolve())
            require(old == config, f"Snapshot source/path mapping/splits/classes differ: {candidate}")
            manifest = candidate.parent / "data_manifest.jsonl.gz"
            require(manifest.is_file(), f"Snapshot lacks original manifest: {candidate}")
            with gzip.open(manifest, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            require(digest(rows) == data["content_sha256"], "Reused manifest content digest mismatch")
            for split, expected in COUNTS.items():
                selected = [r for r in rows if r["split"] == split]
                require((len(selected), sum(r["boxes"] for r in selected)) == expected, "Snapshot count mismatch")
                require(len({r["image"] for r in selected}) == len(selected), "Snapshot contains duplicate images")
                require(digest(selected) == data["splits"][split]["content_sha256"], "Snapshot split digest mismatch")
                for r in selected:
                    require(not Path(r["image"]).is_absolute() and ".." not in Path(r["image"]).parts
                            and Path(r["image"]).as_posix().startswith(config[split].rstrip("/") + "/"), "Snapshot path mapping mismatch")
                    relative = Path(r["image"]).relative_to(config[split])
                    require(r["label"] == (Path("labels") / split / relative.with_suffix(".txt")).as_posix(), "Snapshot label path mapping mismatch")
            data = dict(data, config=config, config_sha256=sha256(path))
            shutil.copyfile(manifest, OUT / "data_manifest.jsonl.gz")
            used = dict(path=str(candidate.resolve()), sha256=sha256(candidate), manifest_sha256=sha256(manifest),
                        code=record.get("code"), note="Reused fixed snapshot; raw image/label bytes NOT rehashed this time")
            break
    if used is None:
        data = inventory(path, OUT / ("rechecked_manifest.jsonl.gz" if prior else "data_manifest.jsonl.gz"))
    if prior:
        require(data == prior["data"], "Explicit data recheck changed snapshot; existing experiment preserved")
    stamps = {str(Path(config["path"]) / config[s]): (Path(config["path"]) / config[s]).stat().st_mtime_ns for s in COUNTS}
    stamps.update({str(Path(config["path"]) / "labels" / s): (Path(config["path"]) / "labels" / s).stat().st_mtime_ns for s in COUNTS})
    write_json(OUT / "data_identity.json", dict(status="PASS", data=data, reused_from=used, directory_stamps=stamps,
        checked_at=now(), policy="Fixed snapshot; subsequent commands read identity/config and top-level directory mtimes only. In-place byte edits require explicit --recheck-data."))
    return data


def binding(clean=True, verify_data=False):
    plan = read_json(OUT / "prepare.json")
    require(plan is not None and plan["status"] == "PASS", "Run prepare successfully first")
    code = code_identity(clean)
    require(plan["code"] == code, "Prepared code identity changed; run prepare again before preflight/start")
    require(sha256(SOURCE) == SOURCE_SHA256 and sha256(INIT) == plan["init_sha256"], "Source/init changed")
    require((OUT / "source_snapshot.tar.gz").is_file()
        and sha256(OUT / "source_snapshot.tar.gz") == plan["source_snapshot_sha256"], "Prepared source snapshot missing/changed")
    path = data_config()
    require(not verify_data, "Full data verification is available only via prepare --recheck-data")
    data = cached_data()
    require(data == plan["data"], "Data identity changed since prepare")
    args, diff = recipe(path)
    require(args == plan["args"], "Recipe changed since prepare")
    return dict(code=code, data=data, init_sha256=plan["init_sha256"], args_sha256=digest(args), formula=FORMULA)


def prepare(reuse_data=None, recheck_data=False):
    OUT.mkdir(parents=True, exist_ok=True)
    code = code_identity(clean=True)
    if os.name == "posix":
        sync = read_json(OUT / "sync.json", {})
        require(sync.get("commit") == code["commit"] and Path(sync.get("worktree", "/missing")).resolve() == ROOT,
                "Run the delivered sync_qcc_v1.sh with the exact SHA first")
    source_contract()
    require(SOURCE.is_file() and sha256(SOURCE) == SOURCE_SHA256, "Missing/wrong public source checkpoint")
    data_path = data_config()
    data = prepare_data(data_path, reuse_data, recheck_data)
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
        if prior["code"] == code:
            require((OUT / "source_snapshot.tar.gz").is_file()
                and sha256(OUT / "source_snapshot.tar.gz") == prior["source_snapshot_sha256"], "Prepared source snapshot missing/changed")
            return prior
        require(not (OUT / "training_identity.json").exists(), "Prepared code changed after dispatch; existing experiment preserved")
        write_json(OUT / "history" / ("prepare_" + prior["code"]["commit"] + ".json"), prior)
    YAML.save(OUT / "train_args.yaml", args)
    write_json(OUT / "recipe_diff.json", diff)
    snapshot = OUT / "source_snapshot.tar.gz"
    temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(temporary), "HEAD"], cwd=ROOT, check=True, timeout=60)
    os.replace(temporary, snapshot)
    record["source_snapshot_sha256"] = sha256(snapshot)
    write_json(OUT / "prepare.json", record)  # PASS only after all required evidence exists
    return record
