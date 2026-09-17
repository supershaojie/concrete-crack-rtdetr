"""Explicit plan/start/resume/status. Neither plan nor preflight launches training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import torch

from bfr_p4_lifecycle import (ROOT, VARIANTS, paths, recipe, identity, require_preflight,
                             read_json, write_json, require, sha256, trainer_class,
                             optimizer_audit, verify_model, server_resource_check, reserve_lock)


def make_plan(args):
    p = paths(args.variant)
    actual = identity(args.variant, args.source, args.initialized, args.data)
    plan = {"status": "PLANNED", "formal_training": "NOT_STARTED", "final_test": "NOT_RUN",
            "identity": actual, "preflight": str(args.preflight.resolve()),
            "preflight_sha256": sha256(args.preflight) if args.preflight.is_file() else None,
            "gate": "PENDING"}
    if args.preflight.is_file():
        report = read_json(args.preflight)
        require_preflight(report, actual)
        plan["gate"] = "PASSED"
    p["metadata"].mkdir(parents=True, exist_ok=True)
    write_json(p["metadata"] / "plan.json", plan)
    _, rows = recipe(args.variant, args.initialized, args.data)
    write_json(p["metadata"] / "recipe_diff.json", rows)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def launch(args, resume=False):
    p = paths(args.variant)
    current = identity(args.variant, args.source, args.initialized, args.data)
    preflight = read_json(args.preflight)
    require_preflight(preflight, current)
    require(torch.cuda.is_available(), "CUDA required; B16/640/native AMP cannot be downgraded")
    require(os.environ.get("CONDA_DEFAULT_ENV") == "rtdetr", "Activate the existing rtdetr environment")
    server_resource_check()
    from preflight_bfr_p4 import amp_resources
    require(amp_resources() == preflight["amp_resources"], "Native AMP resources changed since preflight")
    state_path = p["metadata"] / "training_state.json"
    previous = read_json(state_path) if state_path.is_file() else {}
    overrides = dict(current["recipe"])
    checkpoint = None
    if resume:
        require(args.checkpoint is not None, "Resume requires an explicit last.pt checkpoint")
        checkpoint = args.checkpoint.resolve()
        require(checkpoint == (p["run"] / "weights/last.pt").resolve(), "Resume only this run's last.pt")
        require(previous.get("status") in {"FAILED", "INTERRUPTED", "RUNNING"},
                "Completed/early-stopped runs must not be extended or restarted")
        require(previous.get("identity") == current, "Run identity changed; do not restore into a different experiment")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(0 <= saved.get("epoch", -1) < 199 and saved.get("optimizer") and saved.get("scaler"),
                "Resume needs an unfinished native optimizer/scaler checkpoint")
        require(saved.get("ema") is not None and torch.count_nonzero(saved["ema"].model[22].bfr.wo.weight) > 0,
                "Learned BFR state absent from resume checkpoint")
        verify_model(saved["ema"], args.variant, zero=False)
        expected_saved = dict(overrides)
        # Native resume records the checkpoint as model/resume after the first restart.
        differences = {k: [v, saved.get("train_args", {}).get(k)] for k, v in expected_saved.items()
                       if k not in {"model", "resume"} and saved.get("train_args", {}).get(k) != v}
        require(not differences, "Resume checkpoint recipe differs: " + repr(differences))
        overrides.update(model=str(checkpoint), resume=str(checkpoint))
    else:
        require(not p["run"].exists() and not previous,
                "Existing run/state protected; use explicit resume for an interrupted native run")
    lock = p["run"].with_name(p["run"].name + ".bfr-active.lock")
    reserve_lock(lock, args.variant, current["code"]["commit"], resume=resume)
    state = {"status": "RUNNING", "identity": current, "pid": os.getpid(),
             "started": datetime.now(timezone.utc).isoformat(), "completed_epochs": previous.get("completed_epochs", 0),
             "final_eval": "NOT_STARTED", "final_test": "NOT_RUN", "resume": resume,
             "resume_checkpoint_sha256": sha256(checkpoint) if checkpoint else None}
    p["metadata"].mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    Base = trainer_class(args.variant, p["metadata"] / "trainer_loading.json")

    class RecordingTrainer(Base):
        def validate(self):
            state["epoch_validation"] = {"status": "RUNNING", "epoch": int(self.epoch) + 1}
            write_json(state_path, state)
            try:
                result = super().validate()
                state["epoch_validation"]["status"] = "PASSED"
                return result
            except BaseException as error:
                state["epoch_validation"].update(status="FAILED", error=repr(error))
                raise
            finally:
                write_json(state_path, state)

        def final_eval(self):
            state["final_eval"] = "RUNNING"
            write_json(state_path, state)
            try:
                result = super().final_eval()
                state["final_eval"] = "PASSED"
                return result
            except BaseException as error:
                state["final_eval"] = "FAILED"
                state["final_eval_error"] = repr(error)
                raise
            finally:
                write_json(state_path, state)

    trainer = None
    try:
        trainer = RecordingTrainer(overrides=overrides)
        def started(t):
            require(bool(t.amp) and t.args.batch == 16 and t.args.imgsz == 640, "Native AMP/B16/640 was changed")
            require(type(t.optimizer) is torch.optim.AdamW, "Optimizer changed")
            verify_model(t.model, args.variant, zero=not resume)
            actual = vars(t.args)
            differences = {k: [v, actual.get(k)] for k, v in overrides.items()
                           if type(v) is not type(actual.get(k)) or v != actual.get(k)}
            require(not differences, "Effective training recipe drift: " + repr(differences))
            write_json(p["metadata"] / "training_setup.json", {
                "optimizer": optimizer_audit(t.model, t.optimizer), "actual_args": actual,
                "start_epoch": t.start_epoch, "scaler": t.scaler.state_dict(), "recipe_differences": differences})
        def completed_epoch(t):
            # Record completed optimization before epoch validation/final_eval can fail.
            state["completed_epochs"] = max(state["completed_epochs"], int(t.epoch) + 1)
            if state["completed_epochs"] >= 200:
                state["status"] = "COMPLETED_200"
            write_json(state_path, state)
        trainer.add_callback("on_train_start", started)
        trainer.add_callback("on_train_batch_start", lambda t: setattr(t, "_oom_retries", 3))
        trainer.add_callback("on_train_epoch_end", completed_epoch)
        trainer.train()
        state["status"] = "COMPLETED_200" if state["completed_epochs"] >= 200 else "EARLY_STOPPED"
        state["exit_code"] = 0
    except KeyboardInterrupt:
        if state["completed_epochs"] < 200:
            state["status"] = "INTERRUPTED"
        state["exit_code"] = 130
        raise
    except BaseException as error:
        if state["completed_epochs"] < 200:
            state["status"] = "FAILED"
        state.update(error=repr(error), exit_code=1)
        raise
    finally:
        state["finished"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
        # Remove only our known owner file and now-empty lock, never results.
        (lock / "owner.json").unlink()
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "start", "resume", "status"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_bfr_p4_v1")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--initialized", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    p = paths(args.variant)
    for key in ("source", "initialized", "data"):
        if getattr(args, key) is None:
            setattr(args, key, p[key])
    if args.preflight is None:
        args.preflight = p["metadata"] / "server_preflight/preflight.json"
    if args.mode == "status":
        path = p["metadata"] / "training_state.json"
        print(json.dumps(read_json(path) if path.is_file() else
                         {"status": "NOT_STARTED", "final_test": "NOT_RUN"}, ensure_ascii=False, indent=2))
    elif args.mode == "plan":
        make_plan(args)
    else:
        launch(args, resume=args.mode == "resume")


if __name__ == "__main__":
    main()
