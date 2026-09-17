"""Explicit CSR-P3 plan/start/resume lifecycle. Never starts training during plan."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

from init_csr_p3 import ROOT, VARIANTS, SOURCE_SHA256, build_training_model, require, runtime, sha256, verify_model, write_json
import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML

MAIN = Path(os.environ.get("CSR_P3_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
PARENT_ARGS = ROOT / "configs/csr_p3_parent_args.yaml"
IDENTITY_FIELDS = {"model", "name", "project", "data", "save_dir"}


def now():
    return datetime.now(timezone.utc).isoformat()


def paths(variant):
    name = VARIANTS[variant][2]
    return dict(run=MAIN / "runs/c_series" / name,
                launch=ROOT / "outputs/csr_p3" / variant,
                init=ROOT / "weights" / (variant + "_controlled_init.pt"),
                source=MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                data=MAIN / "configs/crack_autodl.yaml", name=name)


def recipe(variant, initialized, data=None):
    parent = YAML.load(PARENT_ARGS)
    target = dict(parent)
    target.update(model=str(Path(initialized).resolve()), data=str(Path(data or paths(variant)["data"]).resolve()),
                  project=str(MAIN / "runs/c_series"), name=VARIANTS[variant][2],
                  save_dir=str(paths(variant)["run"]))
    rows = [dict(field=k, parent=parent[k], target=target[k], changed=parent[k] != target[k],
                 reason="experiment identity; data equivalence checked by preflight" if parent[k] != target[k] else "inherited")
            for k in parent]
    require({r["field"] for r in rows if r["changed"]} <= IDENTITY_FIELDS, "Non-identity recipe changed")
    require(target["exist_ok"] is False and target["amp"] is True and target["batch"] == 16 and target["imgsz"] == 640,
            "Formal B16/640/native AMP recipe changed")
    return target, rows


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def process_token(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def live_process(state):
    token = process_token(state.get("pid", 0))
    return token is not None and token == state.get("process_token")


def run_state(p):
    state_path = p["launch"] / "training_state.json"
    if not state_path.is_file():
        return dict(status="NOT_STARTED", completed_epochs=0, final_eval="NOT_RUN")
    state = load_json(state_path)
    if state.get("status") in {"RUNNING", "DISPATCHED"} and not live_process(state):
        session = state.get("session")
        tmux_active = bool(session and shutil.which("tmux") and subprocess.run(
            ["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0)
        if not tmux_active:
            state = dict(state, status="INTERRUPTED", note="No live worker/tmux; prior epoch/final_eval evidence retained")
    return state


def require_clean():
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(),
            "Commit execution changes before preflight/start")
    require(Path(__file__).resolve().is_relative_to(ROOT.resolve()), "Unexpected tool import path")
    import ultralytics
    require(Path(ultralytics.__file__).resolve().is_relative_to((ROOT / "ultralytics-main").resolve()),
            "ultralytics must be imported from this worktree")


def plan(variant, initialized=None, data=None, source=None):
    p = paths(variant)
    require(not (p["launch"] / "plan.json").exists(), "Existing plan protected; inspect it, do not silently overwrite")
    target, rows = recipe(variant, initialized or p["init"], data)
    p["launch"].mkdir(parents=True, exist_ok=True)
    YAML.save(p["launch"] / "train_args.yaml", target)
    shutil.copyfile(PARENT_ARGS, p["launch"] / "parent_args.yaml")
    write_json(p["launch"] / "parameter_diff.json", rows)
    record = dict(variant=variant, args=target, source=str(Path(source or p["source"]).resolve()),
                  recipe_path=str(p["launch"] / "train_args.yaml"), recipe_sha256=sha256(p["launch"] / "train_args.yaml"),
                  parent_recipe_sha256=sha256(PARENT_ARGS), runtime=runtime(), created=now(),
                  formal_training="NOT_STARTED", test="NOT_RUN")
    write_json(p["launch"] / "plan.json", record)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def check_preflight(record, report_path):
    from preflight_csr_p3 import identity
    report = load_json(report_path)
    require(report.get("status") == "PASSED" and report.get("mode") == "server", "A PASSED server preflight is required")
    capacity = report.get("server_capacity", {})
    require(capacity.get("status") == "PASSED" and capacity.get("batch") == 16 and capacity.get("imgsz") == 640
            and capacity.get("amp") is True and capacity.get("effective_updates", 0) >= 2,
            "Missing B16/640 native AMP capacity with >=2 effective optimizer updates")
    expected = identity(record["variant"], Path(record["source"]), Path(record["args"]["model"]),
                        Path(record["args"]["data"]), Path(record["recipe_path"]))
    require(expected.get("source_sha256") == SOURCE_SHA256 and expected.get("init_sha256")
            and expected.get("data_inventory"), "Preflight source/init/data identity incomplete")
    require(report.get("identity") == expected, "Preflight commit/source/init/data/recipe/code identity changed; rerun preflight")
    require(sha256(record["recipe_path"]) == record["recipe_sha256"], "Planned recipe changed")
    require(sha256(PARENT_ARGS) == record["parent_recipe_sha256"], "Parent recipe archive changed")
    return report


def native_amp_context(report):
    """Use exactly the preflight local resources and reject native AMP silent skips."""
    from preflight_csr_p3 import enforced_amp_check
    from ultralytics.utils import ASSETS
    resources = report["server_capacity"].get("amp_resources", {})
    require(set(resources) == {"weights", "bus"}, "Native AMP preflight resource evidence missing")
    for name, item in resources.items():
        path = Path(item["path"])
        require(path.is_file() and sha256(path) == item["sha256"], "Native AMP resource changed: " + name)
    require(Path(resources["bus"]["path"]).resolve() == (ASSETS / "bus.jpg").resolve(), "Native AMP bus asset belongs to another worktree")
    return patch("ultralytics.engine.trainer.check_amp",
                 side_effect=lambda model: enforced_amp_check(model, Path(resources["weights"]["path"])))


def launch(variant, report_path, resume=False, foreground=False):
    require_clean()
    require(torch.cuda.is_available(), "Formal training requires CUDA; no CPU recipe substitution")
    require(os.environ.get("CONDA_DEFAULT_ENV") == "rtdetr", "Activate existing rtdetr Conda environment")
    p = paths(variant)
    record = load_json(p["launch"] / "plan.json")
    native_amp_context(check_preflight(record, report_path))
    state = run_state(p)
    require(state["status"] not in {"RUNNING", "DISPATCHED"}, "Existing active worker protected")
    require(record["runtime"]["commit"] == runtime()["commit"], "Plan belongs to another commit")
    if resume:
        require(p["run"].is_dir(), "Resume requires the existing unique run")
        require(state.get("completed_epochs", 0) < 200 and state.get("status") not in {"COMPLETED_200", "EARLY_STOPPED"},
                "Completed training cannot resume; final_eval failures retain completed weights")
        checkpoint = p["run"] / "weights/last.pt"
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(0 <= ckpt.get("epoch", -1) < 199 and ckpt.get("optimizer") is not None, "Need unstripped resumable last.pt")
        require(ckpt.get("ema") is not None and ckpt.get("scaler") is not None, "Resume checkpoint lacks EMA/scaler")
        verify_model(ckpt.get("ema") or ckpt["model"], variant, zero=False)
        del ckpt
    else:
        require(not p["run"].exists(), "Unique run directory exists; use explicit resume, never name2")
        checkpoint = Path(record["args"]["model"])
        require(state["status"] in {"NOT_STARTED", "FAILED", "INTERRUPTED"}, "Previous active start protected")
    lock = p["run"].with_name(p["run"].name + ".csr_p3.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        owner = load_json(lock / "owner.json")
        require((resume or not p["run"].exists()) and owner.get("variant") == variant and owner.get("worktree") == str(ROOT)
                and not live_process(owner), "Existing launch reservation protected")
    else:
        lock.mkdir(exist_ok=False)
    write_json(lock / "owner.json", dict(pid=os.getpid(), process_token=process_token(os.getpid()), variant=variant, worktree=str(ROOT)))
    attempt = p["launch"] / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    attempt.mkdir(exist_ok=False)
    shutil.copyfile(report_path, attempt / "preflight.json")
    shutil.copyfile(record["args"]["data"], attempt / "data_config.yaml")
    dispatch = dict(mode="resume" if resume else "start", variant=variant, checkpoint=str(checkpoint),
                    checkpoint_sha256=sha256(checkpoint), preflight_sha256=sha256(attempt / "preflight.json"),
                    commit=runtime()["commit"], created=now())
    write_json(attempt / "dispatch.json", dispatch)
    session = "csr-p3-" + variant
    state.update(status="DISPATCHED", pid=os.getpid(), process_token=process_token(os.getpid()),
                 session=session, attempt=str(attempt), final_eval="NOT_RUN")
    write_json(p["launch"] / "training_state.json", state)
    if foreground:
        return worker(variant, attempt)
    require(shutil.which("tmux"), "tmux is required, or explicitly use --foreground")
    require(subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode != 0,
            "tmux session exists; preserving it")
    command = [sys.executable, "-u", str(ROOT / "tools/train_csr_p3.py"), "worker", "--variant", variant, "--attempt", str(attempt)]
    exit_path, log = attempt / "process_exit_code.txt", attempt / "console.log"
    shell = "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
    shell += "trap 'rc=$?; printf \"%s\\n\" \"$rc\" > " + shlex.quote(str(exit_path)) + "; exit \"$rc\"' EXIT\n"
    shell += "source /root/miniconda3/etc/profile.d/conda.sh\nconda activate rtdetr\n"
    shell += "export PYTHONPATH=" + shlex.quote(str(ROOT / "ultralytics-main")) + "\nexport YOLO_AUTOINSTALL=false\n"
    shell += "export CSR_P3_MAIN=" + shlex.quote(str(MAIN)) + "\ncd " + shlex.quote(str(ROOT)) + "\n"
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"):
        value = os.environ.get(key)
        shell += ("unset " + key if value is None else "export " + key + "=" + shlex.quote(value)) + "\n"
    shell += " ".join(map(shlex.quote, command)) + " 2>&1 | tee " + shlex.quote(str(log)) + "\n"
    with (attempt / "worker.sh").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(shell)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash " + shlex.quote(str(attempt / "worker.sh"))], check=True)
    print(f"DISPATCHED: tmux {session}; log {log}; inspect status, not tmux [exited]")


def epoch_rows(csv_path):
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def worker(variant, attempt):
    p, attempt = paths(variant), Path(attempt)
    record, dispatch = load_json(p["launch"] / "plan.json"), load_json(attempt / "dispatch.json")
    state = run_state(p)
    state.update(status="RUNNING", pid=os.getpid(), process_token=process_token(os.getpid()), started=now(), final_eval="NOT_RUN")
    code = 1
    trainer_ref = [None]
    write_json(p["launch"] / "training_state.json", state)
    def persist(**items):
        state.update(items)
        write_json(p["launch"] / "training_state.json", state)
        write_json(attempt / "training_state.json", state)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        require_clean()
        preflight = check_preflight(record, attempt / "preflight.json")
        require(sha256(attempt / "preflight.json") == dispatch["preflight_sha256"], "Preflight record modified")
        require(sha256(dispatch["checkpoint"]) == dispatch["checkpoint_sha256"], "Checkpoint changed after dispatch")
        resumed = dispatch["mode"] == "resume"
        require(resumed or not p["run"].exists(), "Run appeared before startup; refusing overwrite/name2")

        class RecordingTrainer(RTDETRTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model, audit = build_training_model(cfg, weights, self.data, variant, zero=not resumed)
                write_json(attempt / "nc1_loading.json", audit)
                return model

            def final_eval(self):
                persist(final_eval="RUNNING", training_finished=True,
                        completed_epochs=max(state.get("completed_epochs", 0), getattr(self, "epoch", -1) + 1))
                try:
                    super().final_eval()
                except BaseException as error:
                    persist(final_eval="FAILED", final_eval_error=repr(error))
                    raise
                persist(final_eval="PASSED")

        model = RTDETR(dispatch["checkpoint"])
        def setup(trainer):
            trainer_ref[0] = trainer
            verify_model(trainer.model, variant, zero=not resumed)
            require(Path(trainer.save_dir).resolve() == p["run"].resolve(), "Run path changed/name2 rejected")
            require(bool(trainer.amp), "Native AMP check disabled AMP; refusing changed recipe")
            expected = dict(record["args"])
            if resumed:
                expected.update(model=dispatch["checkpoint"], resume=dispatch["checkpoint"])
            differences = {k: [v, vars(trainer.args).get(k)] for k, v in expected.items()
                           if type(vars(trainer.args).get(k)) is not type(v) or vars(trainer.args).get(k) != v}
            require(not differences, f"Actual training recipe changed: {differences}")
            counts = [id(v) for group in trainer.optimizer.param_groups for v in group["params"]]
            require(type(trainer.optimizer) is torch.optim.AdamW, "Expected AdamW")
            coverage = {n: counts.count(id(v)) for n, v in trainer.model.named_parameters()}
            require(all(v == 1 for v in coverage.values()), "Optimizer must cover every parameter exactly once")
            YAML.save(attempt / "actual_train_args.yaml", vars(trainer.args))
            write_json(attempt / "training_setup.json", dict(start_epoch=trainer.start_epoch, amp=bool(trainer.amp), optimizer="AdamW",
                       csr_optimizer_occurrences={n: v for n, v in coverage.items() if ".csr." in n}, resume=resumed))
            persist(completed_epochs=trainer.start_epoch, training_finished=False)
        def epoch_end(trainer):
            completed = getattr(trainer, "epoch", trainer.start_epoch - 1) + 1
            persist(completed_epochs=max(state.get("completed_epochs", 0), completed))
            if completed in (40, 80) and state.get("final_eval") == "NOT_RUN":
                parent_path = MAIN / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/results.csv"
                parent = epoch_rows(parent_path)
                current = epoch_rows(p["run"] / "results.csv")
                observation = dict(epoch=completed, current=current[-1] if current else None,
                                   parent=parent[completed-1] if len(parent) >= completed else None,
                                   note="Same-epoch val observation only; no LR/patience/stop-rule changes")
                write_json(attempt / f"observation_epoch_{completed}.json", observation)
                print(json.dumps(observation, ensure_ascii=False), flush=True)
        def disable_oom_retry(trainer):
            trainer._oom_retries = 3
        model.add_callback("on_train_start", setup)
        model.add_callback("on_fit_epoch_end", epoch_end)
        model.add_callback("on_train_batch_start", disable_oom_retry)
        arguments = dict(record["args"])
        if resumed:
            arguments.update(resume=True, model=dispatch["checkpoint"])
        with native_amp_context(preflight):
            model.train(trainer=RecordingTrainer, **arguments)
        code = 0
        require(state.get("final_eval") == "PASSED", "Native final_eval did not complete")
        persist(status="COMPLETED_200" if state.get("completed_epochs", 0) >= 200 else "EARLY_STOPPED")
    except KeyboardInterrupt as error:
        code = 130
        persist(status="INTERRUPTED", error=repr(error))
        raise
    except BaseException as error:
        code = 1
        persist(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        persist(exit_code=code, finished=now())
        write_json(attempt / "exit_code.json", dict(exit_code=code, completed_epochs=state.get("completed_epochs", 0), final_eval=state.get("final_eval")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "start", "resume", "status", "worker"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_csr_p3_v1")
    parser.add_argument("--init", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--attempt", type=Path, help="Internal worker attempt directory")
    parser.add_argument("--foreground", action="store_true", help="Explicit synchronous start/resume instead of tmux")
    args = parser.parse_args()
    if args.mode == "plan":
        plan(args.variant, args.init, args.data, args.source)
    elif args.mode in ("start", "resume"):
        require(args.preflight is not None, "--preflight is required")
        launch(args.variant, args.preflight, resume=args.mode == "resume", foreground=args.foreground)
    elif args.mode == "worker":
        require(args.attempt is not None, "--attempt is required")
        worker(args.variant, args.attempt)
    else:
        print(json.dumps(run_state(paths(args.variant)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
