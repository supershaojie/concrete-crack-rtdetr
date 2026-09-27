"""Pinned RMD identities, full recipe audit, read-only data inventory and strict evidence IO."""
from __future__ import annotations

from copy import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
import torch
import ultralytics
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.models.rtdetr.rmd_v1 import RMDDetectionModel, RMDLoss, FORMULA
from ultralytics.nn.tasks import RTDETRDetectionModel
from init_c19_lif_v1 import source_contract, initialize, verify_model, SOURCE_SHA256, MODEL_DIR, CONFIGS

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
BRANCH = "exp-rtdetr-r18-lite-rmd-v1"
RUN = "rmd_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SESSION = "rmd-v1-training"
SERVER_MAIN = Path("/root/autodl-tmp/projects/Crack_RTDETR")
SERVER_PYTHON = Path("/root/miniconda3/envs/rtdetr/bin/python")
OUT = ROOT / "outputs/rmd_v1"
MODEL = MODEL_DIR / CONFIGS["pair"]
PARENT_RUN = "c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug"
HISTORICAL_COUNTS = {"train": (6048, 45573), "val": (1728, 12840), "test": (864, 6663)}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def unique_name():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def json_safe(value, bad, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(path); return None
    if isinstance(value, dict):
        return {str(k): json_safe(v, bad, path + "." + str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v, bad, path + f"[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    return value


def strict_record(data):
    bad = []
    result = json_safe(data, bad)
    if bad:
        result = dict(result, nonfinite_paths=bad)
    return result


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(strict_record(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(strict_record(data), ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def git(*args, binary=False):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=not binary, timeout=45).strip()


def config():
    c = read_json(ROOT/"configs/rmd_v1.json")
    # These are a priori gates, never adapt them to a probe's outcome.
    require(c["formula_version"] == FORMULA and (c["eta_max"], c["tau"], c["eps_num"]) == (.5, .5, 1e-7), "Fixed RMD formula changed")
    require((c["applicability_min_valid_batches"], c["applicability_min_gt_coverage_k_ge_2"],
             c["applicability_min_mean_abs_weight_minus_one"]) == (8, .5, .01), "Fixed applicability gates changed")
    require(c["preflight_max_micro_batches"] == 16 and c["preflight_max_seconds"] == 900, "Fixed diagnostic limits changed")
    require((c["formal_batch"], c["formal_imgsz"], c["ramp_completed_epochs"]) == (16, 640, [5, 20]), "Fixed recipe/schedule changed")
    require((c["base_commit"], c["branch"], c["run_name"], c["tmux"]) == (BASE, BRANCH, RUN, SESSION), "Fixed experiment identity changed")
    return c


def runtime():
    require(Path(ultralytics.__file__).resolve() == ROOT/"ultralytics-main/ultralytics/__init__.py", "Wrong worktree Ultralytics import")
    return dict(executable=sys.executable, python=platform.python_version(), torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                ultralytics=ultralytics.__file__, ultralytics_version=ultralytics.__version__,
                criterion_source=inspect.getfile(RMDLoss), commit=git("rev-parse", "HEAD"),
                module_hashes=source_contract(), historical_server=dict(python="3.10.13", torch="2.1.2+cu121", gpu="RTX4090"))


def code_identity(clean=False):
    origin = git("remote", "get-url", "origin")
    require(origin.rstrip("/").removesuffix(".git") == "https://github.com/supershaojie/concrete-crack-rtdetr", "Unexpected repository origin")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, capture_output=True).returncode == 0,
            "Worktree is not descended from the pinned mother")
    if clean:
        require(not git("status", "--porcelain"), "RMD worktree must be clean, including untracked source")
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "--", "tools", "configs", "ultralytics-main/ultralytics").splitlines()
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(set(paths))}
    return dict(commit=git("rev-parse", "HEAD"), base=BASE, origin=origin, lf_files=hashes, digest=digest(hashes))


def settings(main=None, data=None, c2_args=None):
    main = Path(main or SERVER_MAIN).resolve()
    candidate = main/"runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
    return dict(main=main, data=Path(data).resolve() if data else main/"configs/crack_autodl.yaml",
                c2_args=Path(c2_args).resolve() if c2_args else candidate if candidate.is_file() else ROOT/"docs/c19_lif_v1/c2_args.yaml",
                source=main/"weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                init=OUT/"rmd_v1_controlled_init.pt", run=main/"runs/c_series"/RUN,
                parent=main/"runs/c_series"/PARENT_RUN/"weights/best.pt")


def recipe(p):
    original = YAML.load(p["c2_args"])
    expected = YAML.load(ROOT/"docs/c19_lif_v1/c2_args.yaml")
    require(set(original) == set(expected), "Authoritative full args field inventory changed")
    require(all(type(v) is type(original[k]) and original[k] == v for k, v in expected.items()), "Authoritative args type/value differs from mother")
    actual = dict(original, model=str(p["init"]), data=str(p["data"]), project=str(p["run"].parent),
                  name=RUN, save_dir=str(p["run"]))
    diff = [dict(field=k, original=original[k], actual=actual[k], changed=actual[k] != original[k],
                 reason="experiment/path identity" if actual[k] != original[k] else "unchanged") for k in sorted(original)]
    require({x["field"] for x in diff if x["changed"]} <= {"model", "name", "save_dir", "project", "data"}, "Forbidden recipe difference")
    return actual, diff


def inventory(p, output=None, splits=("train", "val", "test")):
    import numpy as np
    source = YAML.load(p["data"])
    expected = YAML.load(ROOT/"docs/c19_lif_v1/c2_data.yaml")
    root = (p["main"]/"datasets/crack_det").resolve()
    expected["path"] = str(root)
    source["path"] = str(Path(source["path"]).resolve())
    require(source == expected, "Actual data.yaml path/splits/classes changed")
    result = {}
    for split in splits:
        counts = HISTORICAL_COUNTS[split]
        images = sorted(f for f in (root/"images"/split).rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        rows, gt_count = [], 0
        print(f"RMD data identity: hashing {split}, {len(images)} images and labels (read-only)", flush=True)
        for index, f in enumerate(images):
            label = root/"labels"/split/f.relative_to(root/"images"/split).with_suffix(".txt")
            text = label.read_text(encoding="utf-8")
            labels = [line.split() for line in text.splitlines() if line.strip()]
            for line in labels:
                values = np.asarray(line, dtype=float)
                require(len(values) == 5 and values[0] == 0 and np.isfinite(values).all() and
                        (values[1:] >= 0).all() and (values[1:] <= 1).all() and (values[3:] > 0).all(), "Invalid label " + str(label))
            gt_count += len(labels)
            rows.append(dict(image=f.relative_to(root).as_posix(), image_sha256=sha256(f), image_bytes=f.stat().st_size,
                             label=label.relative_to(root).as_posix(), label_sha256=sha256(label), boxes=len(labels)))
            if (index+1) % 512 == 0:
                print(f"RMD data identity: {split} {index+1}/{len(images)}", flush=True)
        require((len(rows), gt_count) == counts, f"Data identity counts differ for {split}: {(len(rows), gt_count)} != {counts}")
        result[split] = dict(images=len(rows), boxes=gt_count, content_sha256=digest(rows))
        if output:
            write_json(Path(output)/f"inventory_{split}.json", dict(root=str(root), files=rows, **result[split]))
    return dict(root=str(root), yaml_sha256=sha256(p["data"]), splits=result)


def verify_pair(model):
    require(type(model) in (RTDETRDetectionModel, RMDDetectionModel), "Expected original mother/RMD training wrapper")
    view = copy(model)  # read-only type view for the mother's exact verification function
    view.__class__ = RTDETRDetectionModel
    verify_model(view)
    require(model.model[-1].nc == 1, "Single-class trained model required")
    return sum(v.numel() for v in model.parameters())


def prepare(p):
    require(not p["run"].exists(), "Existing formal run preserved; use status/resume/evaluation")
    require(p["source"].is_file(), "Public initialization missing: " + str(p["source"]))
    require(sha256(p["source"]) == SOURCE_SHA256, "Public initialization hash mismatch")
    config(); info = runtime(); args, diff = recipe(p)
    OUT.mkdir(parents=True, exist_ok=True)
    data = inventory(p, OUT/"data_identity")
    init_record = OUT/"initialization.json"
    if p["init"].exists():
        saved = read_json(init_record)
        require(saved and saved["source_sha256"] == SOURCE_SHA256 and saved["output_sha256"] == sha256(p["init"]), "Existing init lacks matching provenance; preserved")
    else:
        write_json(init_record, initialize(p["source"], p["init"]))
    record = dict(status="PASS", time=now(), runtime=info, code=code_identity(), paths=p, data=data,
                  source_sha256=SOURCE_SHA256, init_sha256=sha256(p["init"]), args=args, formula=config())
    record["binding"] = digest({k: record[k] for k in ("code", "data", "source_sha256", "init_sha256", "args", "formula")})
    write_json(OUT/"prepare.json", record)
    write_json(OUT/"history"/(unique_name()+"_prepare.json"), record)
    write_json(OUT/"recipe_diff.json", dict(fields=diff, loss_config=config(), actual_criterion_gains=dict(class_=1, bbox=5, giou=2)))
    YAML.save(OUT/"formal_args.yaml", args)
    return record


def verify_prepared(p, clean=False, diagnostic_splits=None):
    record = read_json(OUT/"prepare.json")
    require(record and record["status"] == "PASS", "Run prepare first")
    args, _ = recipe(p)
    require(args == record["args"] and config() == record["formula"], "Prepared recipe/formula changed")
    require(code_identity(clean) == record["code"], "Prepared code identity changed; rerun prepare/preflight before training")
    require(sha256(p["source"]) == record["source_sha256"] and sha256(p["init"]) == record["init_sha256"], "Initialization changed")
    actual = inventory(p, splits=diagnostic_splits or ("train", "val", "test"))
    require(actual["root"] == record["data"]["root"] and actual["yaml_sha256"] == record["data"]["yaml_sha256"], "Prepared data.yaml changed")
    require(all(actual["splits"][k] == record["data"]["splits"][k] for k in actual["splits"]), "Prepared dataset content changed")
    return record


def offline_amp_resources(p):
    from ultralytics.utils import ASSETS
    evidence = []
    for dest, candidates in ((ROOT/"yolo26n.pt", [p["main"]/"yolo26n.pt", p["main"]/"weights/yolo26n.pt"]),
                             (ASSETS/"bus.jpg", [p["main"]/"ultralytics-main/ultralytics/assets/bus.jpg"])):
        if not dest.is_file():
            found = next((f for f in candidates if f.is_file()), None)
            require(found is not None, "Offline native AMP resource missing: " + str(dest))
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(found, dest)
        evidence.append(dict(path=str(dest), sha256=sha256(dest)))
    return evidence


def optimizer_groups(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [dict(parameters=[names[id(p)] for p in g["params"]],
                 weight_decay=g["weight_decay"], lr=g["lr"], param_group=g.get("param_group"), betas=g.get("betas"))
            for g in optimizer.param_groups]


@contextmanager
def native_amp_guard():
    """The original check must actually pass, not return True after its offline skip."""
    from ultralytics.utils import LOGGER
    messages = []
    class Capture(logging.Handler):
        def emit(self, record):
            if "AMP:" in record.getMessage():
                messages.append(record.getMessage())
    handler = Capture(); LOGGER.addHandler(handler)
    try:
        yield messages
        require(any("checks passed" in m for m in messages) and not any("skipped" in m for m in messages),
                "Original AMP check was not verified: " + repr(messages))
    finally:
        LOGGER.removeHandler(handler)


def aggregate(rows):
    valid = [r for r in rows if r.get("residuals_computed") and r.get("positives", 0)]
    gt = sum(r.get("gt_occurrences", 0) for r in rows)
    positive = sum(r["positives"] for r in valid)
    coverage = sum(r.get("gt_occurrences", 0) for r in rows if r.get("K", 0) >= 2)
    distribution_summary = {}
    for key in ("weight", "similarity"):
        entries = [r[key] for r in valid if r[key]["count"]]
        n = sum(e["count"] for e in entries)
        total = sum(e["sum"] for e in entries)
        total_sq = sum(e["sum_sq"] for e in entries)
        distribution_summary[key] = dict(count=n, mean=total/n if n else None,
            std=math.sqrt(max(0., total_sq/n-(total/n)**2)) if n else None,
            min=min(e["min"] for e in entries) if entries else None,
            max=max(e["max"] for e in entries) if entries else None)
    k_counts = {}
    for row in rows:
        key = str(row.get("K", 0)); k_counts[key] = k_counts.get(key, 0) + row.get("gt_occurrences", 0)
    layers = {}
    ordinary = {}
    for row in rows:
        for key, value in row.get("ordinary", {}).items():
            ordinary[key] = ordinary.get(key, 0.) + value
        for layer in row.get("dn_layers", []):
            name = layer["layer"]
            entry = layers.setdefault(name, {"batches": 0, "original": {}, "weighted": {}})
            entry["batches"] += 1
            for mode in ("original", "weighted"):
                for key, value in layer[mode].items():
                    entry[mode][key] = entry[mode].get(key, 0.) + value
    return dict(batches=len(rows), valid_batches=len(valid), gt_occurrences=gt, k_gt_occurrences=k_counts,
                gt_coverage_k_ge_2=coverage/gt if gt else None,
                positive_dn=sum(r.get("positives", 0) for r in rows), weight_observations=positive,
                mean_abs_weight_minus_one=sum(r["abs_delta_sum"] for r in valid)/positive if positive else None,
                fraction_abs_delta_ge_001=sum(r["abs_delta_ge_001"] for r in valid)/positive if positive else None,
                distribution=distribution_summary, dn_layer_loss_sums=layers, ordinary_loss_sums=ordinary,
                unmatched_gt=sum(r.get("unmatched_gt", 0) for r in rows))


def source_snapshot(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(folder/"source_snapshot.tar.gz"), "HEAD"],
                   cwd=ROOT, check=True, timeout=60)
    (folder/"source_from_mother.patch").write_bytes(git("diff", "--binary", BASE, "HEAD", binary=True))
    code = code_identity()
    with tarfile.open(folder/"actual_source_files.tar.gz", "w:gz") as archive:
        for name in code["lf_files"]:
            archive.add(ROOT/name, arcname=name, recursive=False)
    with (folder/"pip_freeze.txt").open("wb") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True, timeout=45)
    write_json(folder/"source_identity.json", dict(code=code, runtime=runtime(),
        snapshot_sha256=sha256(folder/"source_snapshot.tar.gz"), actual_source_sha256=sha256(folder/"actual_source_files.tar.gz")))


def parent_identity(path):
    """Only original C19+LIF trained checkpoints with independently recorded source provenance."""
    path = Path(path).resolve()
    require(path.is_file(), "Trained mother checkpoint missing: " + str(path))
    ckpt = torch_load(path, map_location="cpu")
    model = (ckpt.get("ema") or ckpt.get("model")).float()
    require(type(model) is RTDETRDetectionModel, "Diagnostic checkpoint must be original mother, no third candidate")
    verify_pair(model)
    args = ckpt.get("train_args", {})
    require(args.get("name") == PARENT_RUN, "Diagnostic checkpoint has another experiment identity")
    require(ckpt.get("train_results") or ckpt.get("epoch", -1) >= 0, "Diagnostic checkpoint has no trained-state evidence")
    provenance = ckpt.get("git", {}).get("commit")
    init_root = Path(args.get("model", "/nonexistent/model.pt")).parent.parent
    for f in (path.parents[1]/"source_record.json", path.parents[1]/"plan.json",
              init_root/"outputs/c19_lif_v1/source_record.json", init_root/"outputs/c19_lif_v1/plan.json"):
        saved = read_json(f, {})
        provenance = provenance or saved.get("runtime", {}).get("commit")
    require(isinstance(provenance, str) and re.fullmatch(r"[0-9a-f]{40}", provenance),
            "Trained mother source SHA missing; provide its original source_record.json in the original run directory")
    require(not git("diff", provenance, BASE, "--", "ultralytics-main/ultralytics/nn", "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"), "Trained mother model source differs from pinned original")
    return model, dict(path=str(path), sha256=sha256(path), source_commit=provenance,
                       model_class=type(model).__module__+"."+type(model).__name__, train_args=args,
                       role="read-only applicability probe; NEVER formal initialization")
