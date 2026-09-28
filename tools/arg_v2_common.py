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
from ultralytics.models.rtdetr.arg_v2_loss import FORMULA
from ultralytics.models.rtdetr.arg_v2_model import write_json
from ultralytics.utils import YAML

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-arg-v2"
ORIGIN = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
RUN_NAME = "arg_v2_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SESSION = "arg-v2-training"
V1_START = "ef9cb7e05e5557f7dd06c95cf2361998a284adc9"
_common_git = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True, timeout=30).strip()
MAIN = Path(os.environ.get("ARG_V2_MAIN", str((ROOT / _common_git).resolve().parent)))
OUT = ROOT / "outputs/arg_v2"
RUN = MAIN / "runs/c_series" / RUN_NAME
INIT = OUT / "arg_v2_init.pt"
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
    from ultralytics.models.rtdetr.arg_v2_loss import ARGv2DetectionLoss
    return dict(mother_runtime(), criterion=inspect.getfile(ARGv2DetectionLoss), module_hashes=source_contract(),
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                formula=FORMULA)


def code_identity(clean=False):
    require(git("remote", "get-url", "origin") == ORIGIN, "Unexpected origin")
    require(git("branch", "--show-current") in (BRANCH, ""), "Not ARG branch/detached delivery")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True, timeout=30)
    subprocess.run(["git", "merge-base", "--is-ancestor", V1_START, "HEAD"], cwd=ROOT, check=True, timeout=30)
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


def data_identity_config(data):
    """Copy config with JSON-stable class IDs; never merge colliding IDs."""
    names = {}
    for class_id, name in data["names"].items():
        key = str(class_id)
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


def manifest_rows(path, data):
    """Verify the saved manifest itself, without touching any raw image/label."""
    import gzip
    from pathlib import PurePosixPath
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    require(digest(rows) == data["content_sha256"], "Saved manifest/data hash mismatch")
    seen = set()
    for row in rows:
        for key in ("image", "label"):
            p = PurePosixPath(row[key])
            require(not p.is_absolute() and ".." not in p.parts, "Unsafe snapshot path")
        split = row["split"]
        prefix = PurePosixPath(data["config"][split])
        relative = PurePosixPath(row["image"]).relative_to(prefix)
        require(row["label"] == (PurePosixPath("labels") / split / relative.with_suffix(".txt")).as_posix(),
                "Snapshot label/path mapping changed")
        require(row["image"] not in seen, "Duplicate snapshot image")
        seen.add(row["image"])
    for split, expected in COUNTS.items():
        local = [r for r in rows if r["split"] == split]
        count = (len(local), sum(r["boxes"] for r in local))
        require(count == expected, "Snapshot split counts differ: " + split)
        require(data["splits"][split] == dict(images=count[0], boxes=count[1], content_sha256=digest(local)),
                "Snapshot split hash differs: " + split)
    return rows


def snapshot_watch(config, rows=None, previous=None):
    """Cheap directory identity/mtime guard; never reads image or label bytes."""
    root = Path(config["path"]).resolve()
    if previous is None:
        directories = {config[k] for k in COUNTS}
        directories.update("labels/" + k for k in COUNTS)
        directories.update(str(Path(r[k]).parent).replace("\\", "/") for r in rows for k in ("image", "label"))
    else:
        directories = previous["directories"]
    stat = root.stat()
    return dict(root=str(root), device=stat.st_dev, inode=stat.st_ino,
                directories={name: (root / name).stat().st_mtime_ns for name in sorted(directories)})


def cached_data(data_path):
    """Use the declared immutable snapshot, not a claim of a fresh full rehash."""
    snapshot = read_json(OUT / "data_snapshot.json")
    require(snapshot and snapshot["status"] == "PASS", "Run prepare to establish a data snapshot")
    data = snapshot["data"]
    require(snapshot["snapshot_id"] == digest(data), "Corrupt data snapshot identity")
    require(data_identity_config(YAML.load(data_path)) == data["config"] and sha256(data_path) == data["config_sha256"],
            "Data config changed; use recheck-data and preserve evidence")
    require(snapshot_watch(data["config"], previous=snapshot["watch"]) == snapshot["watch"],
            "Data directory changed; run recheck-data before continuing")
    return data


def prepare_snapshot(data_path, reuse_from=None):
    if (OUT / "data_snapshot.json").exists():
        return cached_data(data_path)
    import shutil
    manifest = OUT / "data_manifest.jsonl.gz"
    candidate = Path(reuse_from) if reuse_from else MAIN.parent / "Crack_RTDETR-arg_v1/outputs/arg_v1/prepare.json"
    if reuse_from:
        require(candidate.is_file(), "Explicit source snapshot does not exist")
    reused, rejected = None, None
    if candidate.is_file():
        prior = read_json(candidate)
        try:
            require(prior.get("status") == "PASS", "Source prepare/snapshot is not PASS")
            data = prior["data"]
            require(data["config"] == data_identity_config(YAML.load(data_path)) and data["config_sha256"] == sha256(data_path),
                    "Source snapshot path mapping/config differs")
            source_manifest = candidate.parent / "data_manifest.jsonl.gz"
            rows = manifest_rows(source_manifest, data)
            watch = snapshot_watch(data["config"], rows=rows)
            if "watch" in prior:
                require(watch == prior["watch"], "Source snapshot directory metadata changed")
            else:
                created_ns = int(datetime.fromisoformat(prior["created"]).timestamp() * 1e9)
                require(all(t <= created_ns for t in watch["directories"].values()),
                        "Legacy snapshot directories changed after prepare")
            shutil.copyfile(source_manifest, manifest)
            reused = dict(record=str(candidate.resolve()), record_sha256=sha256(candidate),
                          manifest_sha256=sha256(source_manifest), raw_files_rehashed=False)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            if reuse_from:
                raise
            rejected = dict(candidate=str(candidate), reason=str(error))
    if not reused:
        data = inventory(data_path, manifest)
        rows = manifest_rows(manifest, data)
        watch = snapshot_watch(data["config"], rows=rows)
    write_json(OUT / "data_snapshot.json", dict(status="PASS", created=now(), snapshot_id=digest(data),
        data=data, watch=watch, manifest_sha256=sha256(manifest), reused_from=reused, rejected_candidate=rejected,
        verification="saved manifest validated; raw data reused as a frozen snapshot" if reused else "one full image/label content inventory",
        mutation_contract="Keep raw data immutable. Config/directory changes are detected cheaply; in-place file edits require explicit recheck-data."))
    return data


def recheck_data():
    """Explicit full verification; never silently replace the experiment's identity."""
    path = data_config()
    prior = read_json(OUT / "data_snapshot.json")
    require(prior, "Prepare a data snapshot first")
    folder = OUT / "data_rechecks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True)
    write_json(folder / "previous_snapshot.json", prior)
    report = dict(status="FAIL", created=now(), snapshot_id=prior["snapshot_id"])
    try:
        data = inventory(path, folder / "data_manifest.jsonl.gz")
        require(data == prior["data"], "Data content identity changed; original snapshot preserved")
        rows = manifest_rows(folder / "data_manifest.jsonl.gz", data)
        prior.update(status="PASS", watch=snapshot_watch(data["config"], rows=rows), last_full_recheck=str(folder))
        write_json(OUT / "data_snapshot.json", prior)
        report["status"] = "PASS"
    except BaseException as error:
        report["error"] = repr(error)
        prior.update(status="INVALID", invalidated_by=str(folder))
        write_json(OUT / "data_snapshot.json", prior)
        raise
    finally:
        write_json(folder / "recheck.json", report)
    return report


def binding(clean=True):
    plan = read_json(OUT / "prepare.json")
    require(plan is not None and plan["status"] == "PASS", "Run prepare successfully first")
    code = code_identity(clean)
    require(sha256(SOURCE) == SOURCE_SHA256 and sha256(INIT) == plan["init_sha256"], "Source/init changed")
    path = data_config()
    data = cached_data(path)
    require(data == plan["data"], "Data identity changed since prepare")
    args, diff = recipe(path)
    require(args == plan["args"], "Recipe changed since prepare")
    return dict(code=code, data=data, init_sha256=plan["init_sha256"], args_sha256=digest(args), formula=FORMULA)


def prepare(reuse_from=None):
    OUT.mkdir(parents=True, exist_ok=True)
    code = code_identity(clean=True)
    if os.name == "posix":
        sync = read_json(OUT / "sync.json", {})
        require(sync.get("commit") == code["commit"] and Path(sync.get("worktree", "/missing")).resolve() == ROOT,
                "Run the delivered sync_arg_v2.sh with the exact SHA first")
    source_contract()
    require(SOURCE.is_file() and sha256(SOURCE) == SOURCE_SHA256, "Missing/wrong public source checkpoint")
    data_path = data_config()
    data = prepare_snapshot(data_path, reuse_from)
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
        require(prior["code"]["commit"] == code["commit"], "Existing prepare belongs to another code SHA; preserve and inspect")
        if not (OUT / "source_snapshot.tar.gz").is_file():
            subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(OUT / "source_snapshot.tar.gz"),
                            prior["code"]["commit"]], cwd=ROOT, check=True, timeout=60)
        return prior
    YAML.save(OUT / "train_args.yaml", args)
    write_json(OUT / "recipe_diff.json", diff)
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(OUT / "source_snapshot.tar.gz"), "HEAD"], cwd=ROOT, check=True, timeout=60)
    write_json(OUT / "prepare.json", record)
    return record
