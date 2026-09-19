"""RDM lifecycle: explicit actions; preflight/pack never launch formal training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import tarfile
import traceback

import numpy as np
import torch
from rdm_common import (ROOT, MAIN, VARIANTS, SOURCE_SHA256, require, paths, recipe, runtime,
                        initialize, identity, data_identity, code_identity, verify_model, write_json, sha256, torch_load)


@contextmanager
def action_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "action.lock"
    # A leftover lock is intentionally preserved: inspect its PID before removal.
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()) + "\n")
    try:
        yield
    finally:
        lock.unlink()


def preflight_failures(report, current):
    problems = []
    if report.get("status") != "PASSED":
        problems.append("server.status=" + str(report.get("status", "NOT_RUN")))
    for key in ("start", "save_reload", "resume", "half_ema_val", "backend_fp32", "backend_half"):
        if report.get(key, {}).get("status") != "PASSED":
            problems.append(key + "=" + report.get(key, {}).get("status", "NOT_RUN"))
    if report.get("identity") != current:
        problems.append("tested code/init/source/data/recipe identity differs")
    if not 3 <= report.get("training_batches", 0) <= 16:
        problems.append("training batch budget/evidence invalid")
    if report.get("start", {}).get("effective_updates", 0) < 2 or report.get("resume", {}).get("effective_updates", 0) < 1:
        problems.append("effective optimizer updates insufficient")
    if not report.get("temporary_files_cleaned"):
        problems.append("preflight temporary cleanup unconfirmed")
    if report.get("half_ema_val", {}).get("batches") != 1:
        problems.append("native half EMA validation must contain exactly one batch")
    if report.get("start", {}).get("batches", 9) > 8:
        problems.append("startup exceeded eight batches")
    return problems


def require_ready(variant):
    p = paths(variant)
    current = identity(variant)
    local_path, server_path = p["output"] / "local.json", p["output"] / "preflight.json"
    require(local_path.is_file() and server_path.is_file(), "Missing local.json/preflight.json; run init and preflight")
    local, report = json.loads(local_path.read_text()), json.loads(server_path.read_text())
    require(local.get("status") == "PASSED" and local.get("code") == current["code"], "Local checks missing/failed/stale")
    problems = preflight_failures(report, current)
    require(not problems, "Start refused: " + "; ".join(problems))
    require(current["source_sha256"] == SOURCE_SHA256, "Public init source differs")
    return current


def formal(variant, checkpoint=None):
    from preflight_rdm import RDMTrainer, optimizer_inventory, find_amp_resources
    p = paths(variant)
    current = require_ready(variant)
    find_amp_resources()
    if checkpoint is None:
        require(not p["run"].exists(), "Run already exists; no name2 or overwrite: " + str(p["run"]))
        initialize(p["source"], p["init"], variant)  # Read/verify/reuse; no preflight state.
    else:
        checkpoint = checkpoint.resolve()
        require(checkpoint.parent == (p["run"] / "weights").resolve(), "Resume must select this experiment's run/weights checkpoint")
        saved = torch_load(checkpoint, map_location="cpu")
        require(0 <= saved.get("epoch", -1) < 199 and saved.get("optimizer") is not None and saved.get("ema") is not None, "Resume needs unfinished native training state")
        require(saved["train_args"]["name"] == p["name"], "Checkpoint experiment identity differs")
        verify_model(saved["ema"].float(), variant)
        record = json.loads((p["output"] / "run_identity.json").read_text())
        require(record == current, "Run identity changed; do not restart or silently resume another implementation")
        del saved
    with action_lock(p["output"]):
        args, diff = recipe(variant)
        write_json(p["output"] / "resolved_recipe.json", args)
        write_json(p["output"] / "recipe_diff.json", diff)
        if checkpoint is None:
            write_json(p["output"] / "run_identity.json", current)
        args.pop("save_dir", None)
        if checkpoint is not None:
            args.update(model=str(checkpoint), resume=str(checkpoint), exist_ok=True)
        class Trainer(RDMTrainer):
            pass
        Trainer.variant = variant
        state = dict(status="RUNNING", variant=variant, environment=runtime(), checkpoint=str(checkpoint) if checkpoint else None,
                     formal_training="STARTED", final_test="NOT_RUN", last_epoch_completed=0)
        trainer = None
        try:
            trainer = Trainer(overrides=args)
            def setup(t):
                require(t.save_dir.resolve() == p["run"].resolve(), "Native output path changed")
                require(t.amp, "Native AMP was disabled")
                state["optimizer"] = optimizer_inventory(t)
                state["trainer_loading"] = t.rdm_loading
                write_json(p["output"] / "state.json", state)
            def epoch(t):
                state["last_epoch_completed"] = t.epoch + 1
                write_json(p["output"] / "state.json", state)
            trainer.add_callback("on_pretrain_routine_end", setup)
            trainer.add_callback("on_fit_epoch_end", epoch)
            trainer.add_callback("on_train_batch_start", lambda t: setattr(t, "_oom_retries", 3))
            write_json(p["output"] / "state.json", state)
            trainer.train()
            state["status"] = "COMPLETED_200" if trainer.epoch + 1 == 200 else "EARLY_STOPPED" if trainer.stopper.possible_stop else "INTERRUPTED"
        except KeyboardInterrupt:
            state["status"] = "INTERRUPTED"
            raise
        except BaseException as error:
            state.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(limit=8))
            raise
        finally:
            state["time_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(p["output"] / "state.json", state)


def evaluate(variant, checkpoint, split, val_report):
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from c19_lif_v1_results import postprocess, POLICY, EVAL
    p = paths(variant)
    require(checkpoint and checkpoint.is_file(), "Explicit --checkpoint required")
    data = data_identity(p["data"])
    current = code_identity()
    digest = sha256(checkpoint)
    if split == "test":
        require(val_report and val_report.is_file(), "Test needs --val-report for the selected frozen checkpoint")
        previous = json.loads(val_report.read_text())
        require(previous["status"] == "PASSED" and previous["split"] == "val" and previous["checkpoint_sha256"] == digest and previous["variant"] == variant, "Test/val selection mismatch")
        require(previous["data"] == data and previous["code"] == current and previous["policy"] == POLICY, "Frozen val/test identity changed")
    destination = p["output"] / (split + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    report = dict(status="RUNNING", split=split, variant=variant, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
                  data=data, code=current, environment=runtime(), policy=POLICY, settings=EVAL, phase="load")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "metrics.json", report)
    class CorrectedValidator(RTDETRValidator):
        def postprocess(self, preds):
            return postprocess(preds, self.args.imgsz, self.args.conf)[0]
    try:
        model = RTDETR(str(checkpoint))
        verify_model(model.model, variant)
        require(model.model.model[26].nc == 1, "Evaluation requires nc1")
        report["phase"] = "full_" + split
        write_json(destination / "metrics.json", report)
        metrics = model.val(validator=CorrectedValidator, **dict(EVAL, data=str(p["data"]), split=split, device="0", plots=True,
                            save_json=False, save_txt=False, project=str(destination), name="plots", exist_ok=False))
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid AP")
        require(sha256(checkpoint) == digest and data_identity(p["data"]) == data, "Evaluation inputs changed")
        report.update(status="PASSED", precision=float(metrics.box.mp), recall=float(metrics.box.mr), AP50=float(metrics.box.map50),
                      AP75=float(ap[:, 5].mean()), mAP50_95=float(metrics.box.map), AP_by_IoU=ap.tolist(),
                      IoU_thresholds=[round(.5+i*.05, 2) for i in range(10)], speed=metrics.speed)
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        write_json(destination / "metrics.json", report)
    return report


def pack(variant):
    p = paths(variant)
    files = [ROOT / "tools" / n for n in ("rdm.py", "rdm_common.py", "check_rdm.py", "check_rdm_ops.py", "preflight_rdm.py", "rdm_server.sh", "sync_rdm.sh")]
    files += list((ROOT / "docs/rdm_v1").glob("*"))
    files += [ROOT / "ultralytics-main/ultralytics/nn/modules/rdm.py", ROOT / "ultralytics-main/ultralytics/nn/tasks.py", ROOT / "ultralytics-main/ultralytics/nn/modules/__init__.py"]
    files += list((ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr").glob("*rdm-v1.yaml"))
    files += list(p["output"].rglob("*.json"))
    for directory in (p["run"], p["output"]):
        files += [f for f in directory.rglob("*") if f.suffix in {".csv", ".yaml", ".png"}]
    selected = {}
    for f in files:
        if not f.is_file() or f.stat().st_size > 2 * 1024**2 or "rdm-v1-preflight-" in str(f):
            continue
        if f.is_relative_to(ROOT):
            name = "worktree/" + f.relative_to(ROOT).as_posix()
        else:
            name = "run/" + f.relative_to(p["run"]).as_posix()
        selected[name] = f.read_bytes()
    for f in p["output"].glob("*.log"):
        with f.open("rb") as stream:
            stream.seek(max(0, f.stat().st_size - 32768))
            selected["log_tails/" + f.name] = stream.read()
    summary = dict(variant=variant, environment=runtime(), formal_training="NOT_RUN" if not p["run"].exists() else "SEE_STATE",
                   val="SEE_REPORT" if list(p["output"].glob("val_*/metrics.json")) else "NOT_RUN",
                   test="SEE_REPORT" if list(p["output"].glob("test_*/metrics.json")) else "NOT_RUN",
                   excluded="weights, datasets, reference ZIP, raw predictions, temporary checkpoints")
    selected["summary.json"] = json.dumps(summary, indent=2).encode()
    selected["SHA256SUMS"] = "".join(__import__("hashlib").sha256(v).hexdigest()+"  "+k+"\n" for k, v in sorted(selected.items())).encode()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, content in selected.items():
            member = tarfile.TarInfo(name); member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    require(payload.tell() <= 20 * 1024**2, "Light pack exceeds 20MiB; inspect existing results")
    dest = p["output"] / ("rdm_light_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".tar.gz")
    with dest.open("xb") as stream:
        stream.write(payload.getvalue())
    return dict(path=str(dest), bytes=dest.stat().st_size, sha256=sha256(dest))


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("environment", "init", "check", "preflight", "plan", "start", "resume", "val", "test", "pack"))
    parser.add_argument("--variant", choices=VARIANTS, default="cbr_lif_rdm_v1")
    parser.add_argument("--both", action="store_true", help="init only: additionally persist C2+RDM init")
    parser.add_argument("--checkpoint", type=Path, help="explicit native checkpoint for resume/val/test")
    parser.add_argument("--val-report", type=Path, help="test only: frozen selected checkpoint's completed val metrics.json")
    args = parser.parse_args()
    require(not args.both or args.action == "init", "--both is only meaningful for init")
    torch.set_num_threads(4)
    p = paths(args.variant)
    result = None
    if args.action == "environment":
        result = runtime()
        subprocess.run(["nvidia-smi"], check=False)
    elif args.action == "init":
        with action_lock(p["output"]):
            result = []
            for variant in VARIANTS if args.both else [args.variant]:
                q = paths(variant)
                row = initialize(q["source"], q["init"], variant)
                write_json(q["output"] / "initialization.json", row)
                result.append(row)
    elif args.action in {"check", "preflight"}:
        from check_rdm import local_checks
        with action_lock(p["output"]):
            local = p["output"] / "local.json"
            if args.action == "check" or not local.exists() or json.loads(local.read_text()).get("code") != code_identity() or json.loads(local.read_text()).get("status") != "PASSED":
                local_checks(p["source"], local)
            if args.action == "preflight":
                from preflight_rdm import server_preflight
                report = p["output"] / "preflight.json"
                require(not report.exists(), "Preserve prior preflight report; move it to a dated review file before an explicit rerun")
                result = server_preflight(args.variant, report)
                if result["status"] != "PASSED":
                    print(json.dumps(dict(status=result["status"], error=result.get("error"), report=str(report)), indent=2))
                    return 2
            else:
                result = dict(status=json.loads(local.read_text())["status"], report=str(local))
    elif args.action == "plan":
        result = dict(variant=args.variant, paths={k:str(v) for k,v in p.items()}, recipe=recipe(args.variant)[0],
                      formal_training="NOT_STARTED" if not p["run"].exists() else "SEE_STATE", final_test="NOT_RUN", pending=[])
        try:
            require_ready(args.variant)
        except Exception as error:
            result["pending"].append(str(error))
    elif args.action in {"start", "resume"}:
        require(args.action != "resume" or args.checkpoint is not None, "resume requires --checkpoint")
        formal(args.variant, args.checkpoint if args.action == "resume" else None)
    elif args.action in {"val", "test"}:
        result = evaluate(args.variant, args.checkpoint, args.action, args.val_report)
    elif args.action == "pack":
        p["output"].mkdir(parents=True, exist_ok=True)
        result = pack(args.variant)
    if result is not None:
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
