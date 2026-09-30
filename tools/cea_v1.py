#!/usr/bin/env python3
"""CEA v1 server actions. Formal training starts only with explicit start/resume."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

# Make this exact checkout authoritative even when another PYTHONPATH is exported.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Apply the fixed single-GPU recipe before the first CUDA runtime query. This is
# process-local visibility, not a GPU lock/reservation or a change to other jobs.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from cea_v1_common import (OUT, RUN, MAIN, INIT, REMOTE, BRANCH, BASE, SOURCE_SHA256, require, sha256,
                           read_json, write_json, git, runtime, recipe, algorithm, audit_data, initialize,
                           functional_identity, current_identity, prepared_identity)
import torch
from ultralytics.models.rtdetr.cea_trainer import CEATrainer
from ultralytics.utils import ASSETS
from ultralytics.utils.patches import torch_load


def amp_resources(main):
    rows = []
    for destination, sources in (
        (ASSETS / "bus.jpg", [main / "ultralytics-main/ultralytics/assets/bus.jpg", main / "bus.jpg"]),
        (ROOT / "yolo26n.pt", [main / "yolo26n.pt", main / "weights/yolo26n.pt"]),
    ):
        origin = str(destination)
        if not destination.is_file():
            source = next((p for p in sources if p.is_file()), None)
            if source is None:
                rows.append(dict(path=str(destination), status="PENDING", reason="native AMP-check asset absent; reuse mother asset before preflight"))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as inp, destination.open("xb") as out: shutil.copyfileobj(inp, out)
            origin = str(source)
        rows.append(dict(path=str(destination), status="AVAILABLE", sha256=sha256(destination), origin=origin))
    write_json(OUT / "amp_assets.json", rows)
    return rows


def prepare(args):
    from cea_v1_process import run_lock
    with run_lock():
        require(git("remote", "get-url", "origin") == REMOTE, "Actual remote differs from verified source repository")
        require(git("branch", "--show-current") == BRANCH, "Expected isolated CEA branch")
        require(git("merge-base", BASE, "HEAD") == BASE, "CEA branch is not derived from pinned mother")
        require(not git("status", "--porcelain", "--untracked-files=no"), "Commit tracked source changes before prepare")
        info = runtime()
        cfg, differences = recipe(args.main, args.parent_args)
        data, reuse = audit_data(cfg["data"])
        source = args.main / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
        init = initialize(source)
        assets = amp_resources(args.main)
        row = dict(status="PREPARED", runtime=info, recipe=cfg, recipe_audit=differences, data=data,
                   data_audit_status=reuse, initialization=init, algorithm=algorithm(), amp_assets=assets,
                   functional_source=functional_identity(), mother=BASE)
        row["identity"] = current_identity(row)
        previous = OUT / "prepare.json"
        if previous.exists() and read_json(previous).get("identity") != row["identity"] and RUN.exists():
            raise RuntimeError("Existing formal run belongs to a different identity; preserve it")
        write_json(previous, row)
        print(json.dumps(dict(status=row["status"], init=str(INIT), init_sha256=init["output_sha256"],
                             data_audit=reuse, functional_sha256=row["identity"]["functional_sha256"],
                             amp_assets=assets, next="diagnose and preflight (bounded, no formal launch)"), indent=2))


def show_diagnosis(identity):
    path = OUT / "diagnose.json"
    if path.exists():
        row = read_json(path)
        if row.get("identity") != identity:
            print("Diagnosis: PENDING (stale/different identity)")
        else:
            print("Diagnosis: " + row.get("status", "PENDING") + " " + row.get("mechanism", row.get("reason", "")))
            for split, totals in row.get("splits", {}).items():
                print(split + ": " + json.dumps(totals, ensure_ascii=False))
            print("Gradient diagnostics: " + json.dumps(row.get("gradient_diagnostics", [])))
            if row.get("mechanism", "").startswith("NO_CANDIDATE_IMPROVEMENT"):
                print("不建议长训：完整诊断没有发现候选改善。你执行 start 的运行意图仍被尊重。")
    else:
        print("Diagnosis: PENDING; mechanism benefit is unverified")


def ready(action):
    prepared, identity = prepared_identity()
    preflight = read_json(OUT / "preflight.json")
    require(preflight.get("status") == "TECHNICAL_PASS" and preflight.get("identity") == identity,
            "start/resume requires successful real-data preflight of this exact functional identity")
    require(preflight.get("effective_updates", 0) > 0, "Preflight lacks an effective optimizer update")
    require(torch.cuda.is_available(), "CUDA required for B16/640/AMP formal run")
    assets = read_json(OUT / "amp_assets.json")
    require(all(row["status"] == "AVAILABLE" and sha256(row["path"]) == row["sha256"] for row in assets), "Native AMP resources absent/changed")
    if action == "start":
        require(not RUN.exists(), f"Formal run already exists, preserved: {RUN}. Use status/resume for incomplete run")
    else:
        require(not (RUN / "training_complete.json").exists(), "Training already completed; run explicit val/test instead of resuming")
        require(read_json(OUT / "formal_identity.json")["identity"] == identity, "Cannot resume another run")
        checkpoint = torch_load(RUN / "weights/last.pt", map_location="cpu")
        require(0 <= checkpoint.get("epoch", -1) < prepared["recipe"]["epochs"] - 1,
                "Checkpoint finished/stripped; cannot fully resume")
        require(all(checkpoint.get(k) is not None for k in ("optimizer", "scaler", "ema", "updates")),
                "Checkpoint lacks complete native optimizer/scaler/EMA state")
        require(getattr(checkpoint["ema"], "cea_identity", None) == identity, "Resume checkpoint identity mismatch")
    show_diagnosis(identity)
    return prepared, identity


def train_worker(action):
    prepared, identity = ready(action)
    if action == "start":
        write_json(OUT / "formal_identity.json", dict(identity=identity, preflight_sha256=sha256(OUT / "preflight.json"),
                   initialization="public controlled init, never preflight or mother best"))
    cfg = deepcopy(prepared["recipe"])
    if action == "resume":
        cfg.update(model=str(RUN / "weights/last.pt"), resume=str(RUN / "weights/last.pt"))
    trainer = CEATrainer(overrides=cfg, cea_config=algorithm(), cea_identity=identity)
    trainer.train()
    complete = read_json(RUN / "training_complete.json")
    require(complete["status"] == "TRAINING_COMPLETE", "Native training completion record missing")
    from cea_v1_eval import best_lock
    best_lock(identity)
    print("训练结束；最佳训练 val 与独立 FP32 test 不同。请单独执行 test；pack 不会自动执行。")


def run_worker(args):
    from cea_v1_process import run_lock, mark_worker
    require(args.stage_dir is not None and args.run_id, "Internal worker identity missing")
    folder = args.stage_dir.resolve()
    require(folder.is_relative_to((OUT / "workers").resolve()), "Worker output outside this experiment")
    with run_lock():
        mark_worker(args.action, args.run_id, folder)
        started = time.monotonic()
        try:
            if args.action in ("start", "resume"):
                train_worker(args.action)
            elif args.action in ("val", "test", "finish"):
                from cea_v1_eval import evaluate
                evaluate("test" if args.action == "finish" else args.action)
            else:
                from cea_v1_diagnostics import diagnose, preflight
                if args.action == "preflight":
                    assets = read_json(OUT / "amp_assets.json")
                    require(all(r["status"] == "AVAILABLE" for r in assets), "PENDING: native AMP resources missing; see amp_assets.json")
                    row = preflight(folder, args.max_batches, args.seconds)
                else:
                    row = diagnose(folder, args.main, args.mother_best, args.seconds)
                return 0 if row["status"] in ("TECHNICAL_PASS", "COMPLETED") else 2
            write_json(folder / "report.json", dict(status="COMPLETED", action=args.action, seconds=time.monotonic()-started))
            return 0
        except BaseException as error:
            status = "RESOURCE_ERROR" if isinstance(error, torch.cuda.OutOfMemoryError) else "PENDING" if "PENDING:" in str(error) else "FAIL"
            row = dict(status=status, action=args.action, reason=repr(error), seconds=time.monotonic()-started)
            write_json(folder / "report.json", row)
            if args.action in ("diagnose", "preflight"): write_json(OUT / f"{args.action}.json", row)
            traceback.print_exc(); print(json.dumps(row, indent=2))
            return 3 if status == "RESOURCE_ERROR" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "diagnose", "preflight", "status", "start", "resume", "val", "test", "finish", "pack", "attach"))
    parser.add_argument("--main", type=Path, default=MAIN)
    parser.add_argument("--parent-args", type=Path)
    parser.add_argument("--mother-best", type=Path)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--stage-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.worker: return run_worker(args)
    from cea_v1_process import status, attach, bounded, launch, run_lock
    if args.action == "status": status(); return 0
    if args.action == "attach": return attach()
    if args.action == "prepare": prepare(args); return 0
    if args.action == "pack":
        from cea_v1_eval import pack
        with run_lock(): pack()
        return 0
    if args.action in ("diagnose", "preflight"):
        prepared_identity()
        extra = ["--main", str(args.main), "--seconds", str(args.seconds), "--max-batches", str(args.max_batches)]
        if args.mother_best: extra += ["--mother-best", str(args.mother_best)]
        return bounded(args.action, args.seconds, extra)
    if args.action in ("start", "resume"): ready(args.action)
    else:
        from cea_v1_eval import best_lock
        _, identity = prepared_identity(); best_lock(identity)
    launch(args.action)
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as error:
        print(f"CEA action stopped: {error}", file=sys.stderr); sys.exit(2)
