"""Identity, atomic files and read-only handoff ingestion (no ML imports)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile
from contextlib import contextmanager

IDS = ["fasterrcnn_r50_fpn", "yolov5m", "yolov8m", "yolo11m", "yolov13l",
       "yolo26m", "rtdetr_c2", "lcbr_detr"]
NAMES = ["Faster R-CNN", "YOLOv5m", "YOLOv8m", "YOLOv11m", "YOLOv13-L",
         "YOLOv26m", "RT-DETR-R18", "ours"]
ROOT = Path(__file__).resolve().parents[3]
PACKAGE_LIMIT = 100_000_000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def absolute_io_path(path):
    path=Path(path).resolve()
    # Archive source trees exceed legacy MAX_PATH when nested inside the outer ZIP.
    if os.name=="nt" and not str(path).startswith("\\\\?\\"):
        value=str(path)
        path=Path("\\\\?\\UNC\\"+value[2:] if value.startswith("\\\\") else "\\\\?\\"+value)
    return path


def path_value(model, key):
    value = model["paths"].get(key)
    return value.get("absolute_path") if isinstance(value, dict) else value


def safe_extract(archive, destination):
    destination = absolute_io_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            relative = PurePosixPath(entry.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or ":" in entry.filename:
                raise ValueError("Unsafe archive member: " + entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlink refused: " + entry.filename)
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escaped destination")
        source.extractall(destination)


def prepare(source, output):
    """Accept the implementation ZIP, final handoff ZIP, or an extracted directory."""
    source, output = absolute_io_path(source), absolute_io_path(output)
    if source.is_file():
        target = output / "inputs" / sha256(source)[:16]
        complete = target / ".extracted.json"
        if not complete.exists():
            safe_extract(source, target)
            write_json(complete, {"source": str(source), "sha256": sha256(source)})
        source = target
    registries = list(source.glob("model_registry.json")) + list(source.glob("*/model_registry.json"))
    if not registries:
        nested = list(source.glob("*/inputs/Chapter4_Visualization_Handoff_Final_*.zip"))
        if len(nested) != 1:
            raise ValueError("Expected exactly one final handoff or registry in " + str(source))
        return prepare(nested[0], output)
    if len(registries) != 1:
        raise ValueError("Ambiguous handoff registries")
    base = registries[0].parent
    inventory = read_json(base / "SHA256SUMS.json")
    for item in inventory:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe handoff manifest member")
        path = base / relative
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Handoff identity mismatch: " + item["path"])
    actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()
              and "__pycache__" not in p.parts and p.name != "SHA256SUMS.json"}
    if actual != {item["path"] for item in inventory}:
        raise ValueError("Handoff file set differs from inventory")
    registry = read_json(base / "model_registry.json")
    order = read_json(base / "model_order.json")
    if order["model_ids"] != IDS or order["display_names"] != NAMES:
        raise ValueError("Eight-model order/name contract differs")
    if [m["id"] for m in registry["models"]] != IDS:
        raise ValueError("Registry identity order differs")
    if [m["model"] for m in registry["models"]]!=NAMES:
        raise ValueError("Registry display names differ from the fixed order")
    gt = read_json(base / "data/test_gt.json")
    if (len(gt["images"]), len(gt["annotations"])) != (864, 6663):
        raise ValueError("Fixed test identity differs")
    if sha256(base / "data/test_gt.json") != registry["common_test_data"]["test_gt_sha256"]:
        raise ValueError("Shared GT SHA differs")
    return base, {"manifest_files_checked": len(inventory), "images": 864, "gt_boxes": 6663,
                  "handoff": str(base), "registry_sha256": sha256(base / "model_registry.json"),
                  "model_inference_or_cam_performed": False}


def model_environment(model, output):
    """Keep settings/caches outside the formal source/run directories."""
    output = Path(output).resolve()
    runtime = output / "runtime" / model["id"]
    runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               PYTHONUTF8="1", YOLO_AUTOINSTALL="false", YOLOv5_AUTOINSTALL="false",
               WANDB_MODE="disabled", CLEARML_OFFLINE_MODE="1",
               YOLO_CONFIG_DIR=str(runtime / "settings"), MPLCONFIGDIR=str(runtime / "matplotlib"),
               XDG_CONFIG_HOME=str(runtime / "config"), XDG_CACHE_HOME=str(runtime / "cache"),
               TORCH_HOME=str(runtime / "torch"))
    vendor = path_value(model, "vendor")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), vendor]))
    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        env.pop(key, None)
    for prefix in ("YOLOV8M","YOLO11M","YOLOV13L","YOLO26M"):
        env.pop(prefix+"_FROZEN_RUN",None)
    return env


def worker_command(model, request):
    python = path_value(model, "python")
    wrapper = path_value(model, "python_wrapper")
    # The recorded v13 wrapper accepts the Python arguments, not a second interpreter.
    return (["bash", wrapper] if wrapper else [python]) + [
        "-m", "benchmarks.visualization.chapter4.worker", "--request", str(request)]


def progress(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, file=sys.stdout)
    except ImportError:
        return iterable


@contextmanager
def run_lock(output):
    """Nonblocking lock on this visualization run only; never locks a training tree."""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/".chapter4.lock").open("a+b") as stream:
        if stream.tell()==0:stream.write(b"0");stream.flush()
        stream.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("This visualization output already has an active command; use status/logs or a different --out") from error
        try:yield
        finally:
            stream.seek(0)
            if os.name=="nt":msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
