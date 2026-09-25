"""Pinned identities, strict evidence IO and dataset/recipe audit for BMC v1."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-bmc-v1"
REMOTE = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
NAME = "bmc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SESSION = "bmc-v1-training"
MAIN = Path(os.environ.get("BMC_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
OUT = ROOT / "outputs/bmc_v1"
RUN = MAIN / "runs/c_series" / NAME
INIT = ROOT / "weights/bmc_v1_controlled_init.pt"
MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"
SOURCE = MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def sha(path, lf=False):
    digest = hashlib.sha256()
    if lf:
        digest.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    else:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def strict_value(value, nonfinite, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(dict(path=path, value=repr(value)))
        return None
    if isinstance(value, dict):
        return {str(k): strict_value(v, nonfinite, path + "." + str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(v, nonfinite, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nonfinite = []
    value = strict_value(value, nonfinite)
    if nonfinite:
        value = dict(value, nonfinite=nonfinite) if isinstance(value, dict) else dict(value=value, nonfinite=nonfinite)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def runtime():
    import torch
    import ultralytics
    from init_c19_lif_v1 import source_contract
    require(Path(ultralytics.__file__).resolve() == ROOT / "ultralytics-main/ultralytics/__init__.py", "Wrong ultralytics import")
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, version=ultralytics.__version__, commit=git("rev-parse", "HEAD"),
                modules=source_contract(), model_class="ultralytics.models.utils.bmc.BMCDetectionModel",
                criterion_class="ultralytics.models.utils.bmc.BMCLoss",
                bmc_source={p.relative_to(ROOT).as_posix(): sha(p, lf=True) for p in
                    [*sorted((ROOT / "tools").glob("*bmc*.py")),
                     ROOT / "ultralytics-main/ultralytics/models/utils/bmc.py",
                     ROOT / "ultralytics-main/ultralytics/models/utils/ops.py"]},
                dependencies={k: importlib.metadata.version(k) for k in
                ("numpy", "scipy", "torchvision", "PyYAML", "opencv-python")})


def research():
    from ultralytics.utils import YAML
    from ultralytics.models.utils.bmc import BMCConfig
    config = YAML.load(ROOT / "configs/bmc_v1.yaml")
    require(config == BMCConfig().__dict__, "Formal BMC v1 research configuration changed")
    return BMCConfig(**config)


def verify_delivery():
    record = read(OUT / "delivery.json")
    require(record["sha"] == git("rev-parse", "HEAD") and record["branch"] == BRANCH, "Delivery SHA/branch changed; run sync")
    require(record["worktree"] == str(ROOT.resolve()) and record["main"] == str(MAIN.resolve()), "Wrong delivery worktree/main")
    require(git("remote", "get-url", "origin") == REMOTE, "Wrong origin")
    require(not git("status", "--porcelain", "--untracked-files=normal"), "Dirty source worktree; preserve edits and resolve before dispatch")
    require(git("merge-base", BASE, "HEAD") == BASE, "Not descended from pinned mother")
    return record


def source_files():
    # Content identity excludes generated reports, timestamps and delivery documentation.
    files = list((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files += list((ROOT / "ultralytics-main/ultralytics").rglob("*.yaml"))
    files += list((ROOT / "tools").glob("*.py")) + list((ROOT / "tools").glob("*.sh"))
    files += [ROOT / "configs/bmc_v1.yaml", ROOT / "docs/bmc_v1/parent_args.yaml",
              ROOT / "docs/bmc_v1/parent_dataset_inventory.json", ROOT / "docs/bmc_v1/parent_assets.json",
              ROOT / "ultralytics-main/pyproject.toml"]
    return {p.relative_to(ROOT).as_posix(): sha(p, lf=True) for p in sorted(files)}


def data_identity(data_path):
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import YAML
    from c19_lif_v1_data import dataset_inventory
    data = check_det_dataset(str(data_path), autodownload=False)
    expected = YAML.load(ROOT / "docs/c19_lif_v1/c2_data.yaml")
    actual = YAML.load(data_path)
    require({k: v for k, v in actual.items() if k != "path"} ==
            {k: v for k, v in expected.items() if k != "path"}, "Data classes/split definitions changed")
    root = Path(data["path"])
    inventory = dataset_inventory(root)
    require(inventory == read(ROOT / "docs/bmc_v1/parent_dataset_inventory.json"), "Dataset lists/labels differ from trained mother")
    # The historical manifest identifies labels and paths. Add byte identity for all images.
    image_rows = []
    for split in ("train", "val", "test"):
        for p in sorted((root / "images" / split).rglob("*")):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_rows.append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p)))
    content = digest(dict(inventory=inventory, images=image_rows))
    require(content == read(ROOT / "docs/bmc_v1/parent_assets.json")["verified_local_dataset_content_sha256"],
            "Image bytes differ from the verified local crack_det identity; report the relocation conflict")
    return dict(config=str(Path(data_path).resolve()), root=str(root.resolve()), inventory=inventory,
                images=image_rows, content_sha256=content)


def fingerprint(refresh_data=True):
    prepared = read(OUT / "prepare.json")
    require(prepared["status"] == "PASS", "prepare is not complete")
    current_data = data_identity(prepared["data"]["config"]) if refresh_data else prepared["data"]
    info = runtime()
    info.pop("commit")  # only actual source/config content invalidates the capacity result
    identity = dict(source=source_files(), research=research().__dict__, runtime=info,
                    init_sha256=sha(INIT), public_source_sha256=sha(prepared["source"]),
                    args_sha256=sha(OUT / "train_args.yaml", lf=True), data=current_data["content_sha256"])
    resources = read(OUT / "amp_resources.json")
    identity["amp_resources"] = {r["name"]: sha(r["path"]) for r in resources}
    require(all(identity["amp_resources"][r["name"]] == r["sha256"] for r in resources), "AMP check resources changed")
    require(current_data["content_sha256"] == prepared["data"]["content_sha256"], "Data changed since prepare")
    require(identity["init_sha256"] == prepared["init_sha256"], "Controlled initialization changed")
    return dict(sha256=digest(identity), identity=identity)
