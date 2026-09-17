"""Shared identities and immutable recipe for the two BFR-P4 experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import socket
from datetime import datetime, timezone
import uuid

from init_bfr_p4 import (ROOT, MODEL_DIR, VARIANTS, SOURCE_SHA256, sha256,
                         require, write_json, runtime, build_training_model,
                         verify_model, audit_checkpoint)
from ultralytics.utils import YAML
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from c19_lif_v1_data import dataset_inventory

MAIN = Path(os.environ.get("BFR_P4_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))


def paths(variant):
    name = VARIANTS[variant][2]
    return dict(name=name, metadata=ROOT / "outputs/bfr_p4" / variant,
                run=MAIN / "runs/c_series" / name,
                source=MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                initialized=ROOT / "weights" / (variant + "_controlled_init.pt"),
                data=MAIN / "configs/crack_autodl.yaml")


def recipe(variant, initialized, data):
    parent = YAML.load(ROOT / "docs/bfr_p4/parent_args.yaml")
    appendix = YAML.load(ROOT / "docs/bfr_p4/appendix_args.yaml")
    require(parent == appendix, "Successful parent recipe differs from Appendix A")
    result = dict(parent)
    p = paths(variant)
    result.update(model=str(Path(initialized).resolve()), data=str(Path(data).resolve()),
                  name=p["name"], project=str(p["run"].parent), save_dir=str(p["run"]))
    require(result["epochs"] == 200 and result["patience"] == 50 and
            result["batch"] == 16 and result["imgsz"] == 640 and result["amp"] is True and
            result["exist_ok"] is False and result["resume"] is False, "Recipe invariants changed")
    rows = [{"field": k, "parent": parent[k], "target": result[k], "changed": parent[k] != result[k]}
            for k in sorted(parent)]
    require({r["field"] for r in rows if r["changed"]} <=
            {"model", "data", "name", "project", "save_dir"}, "Recipe drift")
    return result, rows


def code_identity(clean=True):
    """Hash the entire tracked source surface, not merely experiment entrypoints."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    untracked_code = [name for name in untracked if name.startswith(("tools/", "ultralytics-main/ultralytics/", "docs/bfr_p4/"))]
    if clean:
        require(not dirty, "Commit code changes and rerun affected checks before the server gate")
        require(not untracked_code, "Untracked active source is outside the committed code identity: " + repr(untracked_code))
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    relevant = [n for n in listed if n and
                (n.startswith(("tools/", "ultralytics-main/ultralytics/", "docs/bfr_p4/")) or n == ".gitattributes")]
    files = {n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
             for n in relevant if (ROOT / n).is_file()}
    require(files, "No code identity files")
    return {"commit": head, "files_lf_sha256": files, "dirty": bool(dirty), "untracked_code": untracked_code}


def server_resource_check():
    """Do not overlap another GPU job; report it without changing anyone's process."""
    query = ["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    processes = subprocess.check_output(query, text=True).strip()
    others = []
    for line in processes.splitlines():
        first = line.split(",", 1)[0].strip()
        if first.isdigit() and int(first) != os.getpid():
            others.append(line)
    if others:
        raise FileNotFoundError("PENDING GPU 0 occupied; preserve existing processes: " + "; ".join(others))
    return {"compute_processes": processes, "memory": subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
        text=True).strip()}


def pid_confirmed_absent(pid):
    """Conservative Linux check. A live/reused PID or unsupported host never permits recovery."""
    require(type(pid) is int and pid > 0, "Invalid lock owner PID")
    if os.name != "posix" or not Path("/proc").is_dir():
        return False
    try:
        os.kill(pid, 0)  # POSIX existence check only; never used on Windows.
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


def reserve_lock(lock, variant, commit, resume=False, absent_check=None):
    """Only explicit resume can archive one demonstrably dead, same-host reservation."""
    lock = Path(lock)
    absent_check = absent_check or pid_confirmed_absent
    host = socket.gethostname()
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        require(resume, "Existing active reservation protected; recovery is only available in explicit resume")
        require(lock.is_dir() and not lock.is_symlink(), "Unexpected lock path type")
        owner_path = lock / "owner.json"
        require(owner_path.is_file() and not owner_path.is_symlink(), "Missing/unsafe reservation owner")
        raw = owner_path.read_bytes()
        owner = json.loads(raw)
        require(owner.get("variant") == variant and owner.get("commit") == commit and owner.get("hostname") == host,
                "Reservation owner identity/hostname differs; preserve and investigate")
        pid = owner.get("pid")
        require(type(pid) is int and pid > 0 and absent_check(pid),
                "Reservation PID exists, may be reused, or cannot be proven absent; preserved")
        require(owner_path.read_bytes() == raw and absent_check(pid), "Reservation changed during stale-owner inspection")
        archived = lock.with_name(lock.name + ".stale." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "." + uuid.uuid4().hex[:8])
        require(not archived.exists(), "Unexpected stale evidence collision")
        lock.rename(archived)
    lock.mkdir(exist_ok=False)
    write_json(lock / "owner.json", {"pid": os.getpid(), "hostname": host, "variant": variant, "commit": commit})
    return lock


def data_identity(config):
    resolved = check_det_dataset(str(config), autodownload=False)
    require(resolved["nc"] == 1, "Expected nc=1 crack dataset")
    root = Path(resolved["path"]).resolve()
    for split in ("train", "val", "test"):
        locations = resolved[split]
        require(isinstance(locations, str) and Path(locations).resolve() == root / "images" / split,
                "Actual split path differs from the successful split: " + split)
    inventory = dataset_inventory(root)
    expected = json.loads((ROOT / "docs/bfr_p4/parent_dataset_inventory.json").read_text(encoding="utf-8"))
    require(inventory == expected, "Actual split paths/labels differ from successful parent")
    return {"config_sha256": sha256(config), "inventory": inventory, "root": str(root)}


def identity(variant, source, initialized, data, clean=True):
    require(sha256(source) == SOURCE_SHA256, "Unified untrained source SHA mismatch")
    init_audit = audit_checkpoint(initialized, variant, require_untrained=True)
    args, _ = recipe(variant, initialized, data)
    return {"variant": variant, "code": code_identity(clean), "source_sha256": sha256(source),
            "initialization_sha256": sha256(initialized), "initialization_audit": init_audit,
            "data": data_identity(data), "recipe": args}


def check_identity(expected, actual):
    # Runtime paths are deliberately part of the identity: local diagnostics cannot open a server gate.
    require(expected == actual, "Preflight identity changed (code/config/source/init/data/variant); rerun preflight")


def optimizer_audit(model, optimizer):
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    named = {name: ids.count(id(p)) for name, p in model.named_parameters() if p.requires_grad}
    require(all(v == 1 for v in named.values()) and len(ids) == len(named),
            "Every trainable parameter must occur exactly once in the optimizer")
    return {"optimizer": type(optimizer).__name__, "parameter_tensors": len(ids),
            "new_parameter_occurrences": {n: v for n, v in named.items() if n.startswith("model.22.bfr.")},
            "gn_bias_exception": "Registered; DC-only affine bias is a mathematical zero-gradient direction"}


def trainer_class(variant, audit_path=None):
    class ControlledBFRTrainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model, audit = build_training_model(cfg, weights, self.data, variant)
            if audit_path:
                write_json(audit_path, audit)
            return model
    return ControlledBFRTrainer


def require_preflight(report, current):
    require(report.get("status") == "PASSED" and report.get("scope") == "server_B16_640_native_AMP",
            "PASSED server B16/640/native AMP preflight required")
    check_identity(report["identity"], current)
    cap = report.get("capacity", {})
    require(cap.get("batch") == 16 and cap.get("imgsz") == 640 and cap.get("amp") is True and
            cap.get("effective_optimizer_updates", 0) >= 2 and cap.get("gradient_startup") == "PASSED",
            "Missing real B16/640 AMP effective-update evidence")
    require(report.get("native_resume", {}).get("status") == "PASSED", "Native learned-state resume not verified")
    require(report.get("learned_lifecycle", {}).get("status") == "PASSED", "Learned nonzero BFR lifecycle not verified")
    require(report.get("engineering", {}).get("status") == "PASSED", "Server engineering checks missing")
    require(report.get("mathematics", {}).get("status") == "PASSED", "CPU/CUDA mathematical checks missing")
    engineering = report["engineering"]
    require(Path(engineering.get("report", "")).is_file() and
            sha256(engineering["report"]) == engineering.get("sha256"),
            "Referenced engineering evidence missing or changed")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
