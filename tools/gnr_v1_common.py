"""GNR identities and preparation. Heavy imports are confined to explicit actions."""
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
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-gnr-v1"
REMOTE = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
SERVER_MAIN = "/root/autodl-tmp/projects/Crack_RTDETR"
SERVER_WT = "/root/autodl-tmp/projects/Crack_RTDETR-gnr_v1"
SERVER_PY = "/root/miniconda3/envs/rtdetr/bin/python"
RUN_NAME = "gnr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
OUT = ROOT / "outputs/gnr_v1"
RUN = ROOT / "runs/c_series" / RUN_NAME
INIT = ROOT / "weights/gnr_v1_controlled_init.pt"
MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"
SOURCE_SHA = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
COUNTS = {"train": (6048, 45573), "val": (1728, 12840), "test": (864, 6663)}
ALIASES = (
    "/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1/weights/c19_lif_v1_controlled_init.pt",
    "/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/weights/c19_lif_v1_controlled_init.pt",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", timeout=30).strip()


def source_identity(clean=True):
    require(git("remote", "get-url", "origin") == REMOTE, "Unexpected Git remote")
    require(git("branch", "--show-current") in (BRANCH, ""), "Unexpected GNR branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True, timeout=30)
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    require(not clean or not dirty, "Commit/review source changes before server operations; worktree preserved")
    files = set((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files.update((ROOT / "tools").glob("*.py"))
    files.update((ROOT / "tools").glob("*gnr_v1*.sh"))
    files.update([MODEL, ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml", ROOT / "docs/c19_lif_v1/c2_data.yaml"])
    files.update([ROOT / "docs/gnr_v1/gnr_config.json", ROOT / "docs/gnr_v1/resolved_formal_config.yaml"])
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(files)}
    return {"commit": git("rev-parse", "HEAD"), "base": BASE, "branch": git("branch", "--show-current"),
            "dirty": dirty, "functional_sha256": digest(hashes), "files": hashes}


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_yaml(path, value):
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def compare_recipe(actual, expected):
    require(set(actual) == set(expected) and len(expected) == 109, "Mother recipe must contain exactly the archived 109 fields")
    normalized = dict(actual)
    alias = None
    if actual["model"] != expected["model"] and actual["model"] in ALIASES and expected["model"] in ALIASES:
        alias = {"actual": actual["model"], "archived": expected["model"], "reason": "documented mother gatefix path drift; comparison copy only"}
        normalized["model"] = expected["model"]
    differences = {k: {"actual": normalized[k], "expected": v, "actual_type": type(normalized[k]).__name__, "expected_type": type(v).__name__}
                   for k, v in expected.items() if type(normalized[k]) is not type(v) or normalized[k] != v}
    require(not differences, f"Mother recipe/type mismatch: {differences}")
    return {"fields_checked": len(expected), "alias": alias, "other_differences": differences}


def recipe(data_path, mother_args=None):
    archived = load_yaml(ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml")
    documented = load_yaml(ROOT / "docs/gnr_v1/resolved_formal_config.yaml")
    expected_document = dict(archived, model=SERVER_WT + "/weights/gnr_v1_controlled_init.pt",
        data=SERVER_MAIN + "/configs/crack_autodl.yaml", project=SERVER_WT + "/runs/c_series",
        name=RUN_NAME, save_dir=SERVER_WT + "/runs/c_series/" + RUN_NAME)
    require(set(documented) == set(expected_document) and len(documented) == 109 and
            all(type(documented[k]) is type(v) and documented[k] == v for k, v in expected_document.items()),
            "Documented full formal recipe differs from the mother plus exact experiment paths")
    if mother_args and Path(mother_args).is_file():
        actual = load_yaml(mother_args)
        audit = compare_recipe(actual, archived)
        audit.update(actual_file=str(Path(mother_args).resolve()), actual_sha256=sha256(mother_args), actual_status="CHECKED")
    else:
        actual = archived
        audit = {"fields_checked": len(archived), "actual_status": "PENDING", "missing": str(mother_args), "alias": None}
    target = dict(archived, model=str(INIT), data=str(Path(data_path).resolve()), project=str(RUN.parent), name=RUN_NAME, save_dir=str(RUN))
    audit["changes"] = [{"field": k, "mother": actual[k], "experiment": target[k], "reason": "verified experiment/output identity or same-data relocation"}
                        for k in actual if actual[k] != target[k]]
    require({r["field"] for r in audit["changes"]} <= {"model", "data", "project", "name", "save_dir"}, "Illegal recipe change")
    return target, audit


def metadata(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def data_snapshot(data_path, destination, refresh=False):
    """One content scan; later reuse checks saved file/directory metadata only."""
    import math
    from PIL import Image
    path, destination = Path(data_path).resolve(), Path(destination)
    if destination.exists() and not refresh:
        saved = read_json(destination)
        require(Path(saved["data_yaml"]).resolve() == path, "Data YAML location changed; explicitly prepare --refresh-data")
        validate_snapshot(saved)
        return saved
    config = load_yaml(path)
    expected = load_yaml(ROOT / "docs/c19_lif_v1/c2_data.yaml")
    expected["path"] = str(Path(config["path"]).resolve())
    config["path"] = expected["path"]
    require(config == expected, f"Data split/class configuration differs: {config}")
    root = Path(config["path"])
    records, splits, directories = [], {}, {}
    for split, counts in COUNTS.items():
        folder = root / config[split]
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        require(images, f"Missing/empty split: {folder}")
        local, gt_count = [], 0
        for image in images:
            label = root / "labels" / split / image.relative_to(folder).with_suffix(".txt")
            values = [list(map(float, line.split())) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            require(all(len(v) == 5 and v[0] == 0 and all(math.isfinite(x) for x in v) and all(0 <= x <= 1 for x in v[1:]) and v[3] > 0 and v[4] > 0 for v in values), f"Invalid label: {label}")
            require(len({tuple(v) for v in values}) == len(values), f"Duplicate labels would be removed by mother loader: {label}")
            with Image.open(image) as im:
                shape = [im.height, im.width]
                im.verify()
            row = {"split": split, "image": image.relative_to(root).as_posix(), "label": label.relative_to(root).as_posix(),
                   "image_sha256": sha256(image), "label_sha256": sha256(label), "image_stat": metadata(image), "label_stat": metadata(label),
                   "size_hw": shape, "gt_cxcywh_normalized": values, "boxes": len(values)}
            local.append(row)
            gt_count += len(values)
        require((len(images), gt_count) == counts, f"{split} counts {(len(images), gt_count)} != fixed mother {counts}")
        records.extend(local)
        splits[split] = {"images": len(images), "boxes": gt_count, "content_sha256": digest(local),
                         "image_ids": [r["image"] for r in local]}
        for directory in (folder, root / "labels" / split):
            for item in [directory, *[p for p in directory.rglob("*") if p.is_dir()]]:
                directories[item.relative_to(root).as_posix()] = item.stat().st_mtime_ns
        print(f"GNR data {split}: {len(images)} images, {gt_count} GT", flush=True)
    saved = {"version": "gnr_content_stat_v1", "root": str(root), "data_yaml": str(path), "data_sha256": sha256(path),
             "config": config, "records": records, "directories": directories, "splits": splits}
    saved["sha256"] = digest(saved)
    if destination.exists():
        destination.rename(destination.with_name(f"data_snapshot_{now()}.json"))
    write_json(destination, saved)
    return saved


def validate_snapshot(saved):
    require(saved["version"] == "gnr_content_stat_v1" and digest({k: v for k, v in saved.items() if k != "sha256"}) == saved["sha256"], "Data snapshot corrupted/unsupported")
    require(sha256(saved["data_yaml"]) == saved["data_sha256"], "Data YAML changed; explicitly prepare --refresh-data")
    root = Path(saved["root"])
    for row in saved["records"]:
        for kind in ("image", "label"):
            require((root / row[kind]).is_file() and metadata(root / row[kind]) == row[kind + "_stat"], f"Data metadata changed: {row[kind]}; explicitly prepare --refresh-data")
    for relative, stamp in saved["directories"].items():
        require((root / relative).is_dir() and (root / relative).stat().st_mtime_ns == stamp, f"Directory changed: {relative}; explicitly prepare --refresh-data")


def prepare(args):
    from init_c19_lif_v1 import initialize, source_contract, build_training_model, controlled_models, runtime
    from ultralytics.utils.patches import torch_load
    from ultralytics.models.rtdetr.gnr_loss import FORMULA
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "training.json").exists(), "Prepare is immutable after formal training begins")
    code = source_identity(clean=not args.development)
    main = Path(args.main).resolve()
    if not args.development:
        require(ROOT == Path(SERVER_WT) and main == Path(SERVER_MAIN) and Path(sys.executable) == Path(SERVER_PY),
                "Formal prepare requires the documented server worktree, MAIN and Python paths")
    require(read_json(ROOT / "docs/gnr_v1/gnr_config.json") == FORMULA, "Documented GNR formula differs from the active implementation")
    source = main / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
    require(source.is_file() and sha256(source) == SOURCE_SHA, f"Missing/wrong public initialization: {source}")
    data = Path(args.data or main / "configs/crack_autodl.yaml").resolve()
    mother_args = Path(args.mother_args or main / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml")
    train_args, audit = recipe(data, mother_args)
    snapshot = data_snapshot(data, OUT / "data_snapshot.json", args.refresh_data)
    source_contract()
    if INIT.exists():
        init_record = read_json(OUT / "initialization.json")
        if init_record is None:
            # Recover only a provably identical completed file left by an interrupted
            # post-save audit. Unknown or trained files are never overwritten.
            saved = torch_load(INIT, map_location="cpu")
            require(saved.get("epoch") == -1 and saved.get("c19_lif_v1_provenance", {}).get("source_sha256") == SOURCE_SHA,
                    "Existing init has no recoverable mother provenance; preserved")
            _, expected, init_record = controlled_models(source)
            actual = saved["model"].state_dict()
            require(set(actual) == set(expected.state_dict()) and all(torch.equal(v, actual[k]) for k, v in expected.state_dict().items()),
                    "Existing init values differ from controlled mother mapping; preserved")
            init_record.update(output=str(INIT), output_sha256=sha256(INIT), reload_exact=True, status="passed", runtime=runtime(),
                               recovery="exact controlled-state verification after interrupted post-save audit")
            write_json(OUT / "initialization.json", init_record)
        require(init_record and init_record["output_sha256"] == sha256(INIT) and init_record["source_sha256"] == SOURCE_SHA, "Existing init provenance mismatch; preserved")
    else:
        # Native asset loader strips apostrophes in absolute Windows usernames.
        # The same file's relative path preserves its bytes and mother initializer.
        require(Path.cwd().resolve() == ROOT, "prepare must run from this worktree")
        init_record = initialize(source, INIT.relative_to(ROOT))
        write_json(OUT / "initialization.json", init_record)
    weights = torch_load(INIT, map_location="cpu")["model"].float()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model, mapping = build_training_model(str(MODEL), weights, {"nc": 1, "channels": 3})
    structure = {k: list(v.shape) for k, v in model.state_dict().items()}
    write_json(OUT / "nc1_loading.json", mapping)
    write_json(OUT / "structure.json", {"parameters_unfused": sum(p.numel() for p in model.parameters()), "state_shapes": structure})
    record = {"status": "PREPARED", "created": now(), "development": args.development, "main": str(main), "source": str(source),
              "source_sha256": SOURCE_SHA, "init_sha256": sha256(INIT), "structure_sha256": digest(structure), "code": code,
              "data_yaml": str(data), "data_sha256": snapshot["sha256"], "args": train_args, "recipe_audit": audit,
              "mother_args": str(mother_args), "formula": FORMULA, "run": str(RUN),
              "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda}}
    save_yaml(OUT / "train_args.yaml", train_args)
    save_yaml(OUT / "mother_args.yaml", load_yaml(mother_args) if mother_args.is_file() else load_yaml(ROOT / "docs/c19_lif_v1/resolved_formal_config.yaml"))
    save_yaml(OUT / "data_config.yaml", load_yaml(data))
    write_json(OUT / "recipe_audit.json", audit)
    if (OUT / "prepare.json").exists():
        write_json(OUT / f"prepare_{now()}.json", read_json(OUT / "prepare.json"))
    (OUT / "source_from_mother.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", BASE, "HEAD"], cwd=ROOT))
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(OUT / "source_snapshot.tar.gz"), "HEAD"], cwd=ROOT, check=True, timeout=60)
    record["artifacts"] = {name: sha256(OUT / name) for name in (
        "initialization.json", "nc1_loading.json", "structure.json", "data_snapshot.json", "recipe_audit.json",
        "train_args.yaml", "mother_args.yaml", "data_config.yaml", "source_snapshot.tar.gz", "source_from_mother.patch")}
    write_json(OUT / "prepare.json", record)
    return {"status": "PREPARED", "run": str(RUN), "out": str(OUT), "actual_mother_args": audit["actual_status"], "formal_training_started": False}


def binding(clean=True, check_data=True):
    import torch
    plan = read_json(OUT / "prepare.json")
    require(plan and plan["status"] == "PREPARED", "Run prepare first")
    require(plan["runtime"] == {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda},
            "Python/PyTorch/CUDA environment changed; prepare and bounded checks must be repeated explicitly")
    code = source_identity(clean)
    require(code["functional_sha256"] == plan["code"]["functional_sha256"], "Functional code changed; prepare/preflight evidence invalid")
    require(sha256(plan["source"]) == SOURCE_SHA and sha256(INIT) == plan["init_sha256"], "Initialization identity changed")
    require(load_yaml(OUT / "train_args.yaml") == plan["args"], "Training recipe changed")
    snapshot = read_json(OUT / "data_snapshot.json")
    require(snapshot["sha256"] == plan["data_sha256"], "Snapshot identity changed")
    if check_data:
        validate_snapshot(snapshot)
    recipe_now, audit = recipe(plan["data_yaml"], plan["mother_args"])
    require(recipe_now == plan["args"] and audit == plan["recipe_audit"], "Mother/current recipe or identity changed; prepare again")
    identity = {"functional_sha256": code["functional_sha256"], "recipe_sha256": digest(plan["args"]), "data_sha256": snapshot["sha256"],
                "source_sha256": SOURCE_SHA, "init_sha256": plan["init_sha256"], "structure_sha256": plan["structure_sha256"],
                "run": str(RUN), "formula": plan["formula"], "runtime": plan["runtime"]}
    return {"sha256": digest(identity), "identity": identity}
