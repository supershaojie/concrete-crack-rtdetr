"""Pinned SDB-P3 plan/start/resume/status; no automatic training from preflight."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import traceback

import torch
from init_sdb_p3 import (ROOT, VARIANTS, SOURCE_SHA256, BRANCH, require, sha256,
                         build_training_model, verify_model, is_added)
from ultralytics import RTDETR
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from c19_lif_v1_data import dataset_inventory

MAIN = Path(os.environ.get("SDB_P3_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
DEFAULT_VARIANT = "cbr_lif_sdb_p3_v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def code_identity():
    """LF hashes bind the actual executable sources, recipes and identity reference."""
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
        "tools", "ultralytics-main/ultralytics", "configs", "docs/sdb_p3"], cwd=ROOT).decode().split("\0")
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in sorted(set(names)) if name and (ROOT / name).is_file() and
            (Path(name).suffix in {".py", ".yaml", ".sh"} or
             name in {"docs/sdb_p3/parent_dataset_inventory.json", "docs/sdb_p3/parent_results.csv"})}


def paths(variant=DEFAULT_VARIANT):
    name = VARIANTS[variant][2]
    return dict(run=MAIN / "runs/c_series" / name, launch=ROOT / "outputs/sdb_p3" / variant,
                init=ROOT / "weights" / f"{variant}_controlled_init.pt",
                source=MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                data=MAIN / "configs/crack_autodl.yaml", preflight=ROOT / "outputs/sdb_p3_preflight" / variant / "checks.json")


def recipe(init, variant=DEFAULT_VARIANT, data=None):
    """Every field originates in the actual successful parent args snapshot."""
    original = YAML.load(ROOT / "docs/sdb_p3/parent_args.yaml")
    target = dict(original)
    p = paths(variant)
    target.update(model=str(Path(init).resolve()), data=str(Path(data or p["data"]).resolve()),
                  project=str(p["run"].parent), name=p["run"].name, save_dir=str(p["run"]))
    changed = {k for k in original if original[k] != target[k]}
    require(changed <= {"model", "data", "project", "name", "save_dir"}, "Nonidentity recipe change")
    require(target["epochs"] == 200 and target["patience"] == 50 and target["batch"] == 16 and
            target["imgsz"] == 640 and target["amp"] is True and target["exist_ok"] is False,
            "Archived successful recipe contract changed")
    rows = [dict(field=k, parent=original[k], target=target[k], changed=k in changed,
                 reason="experiment identity / verified equivalent data relocation" if k in changed else "inherited")
            for k in original]
    return target, rows


def verify_data(data):
    data = Path(data)
    raw, expected = YAML.load(data), YAML.load(ROOT / "docs/sdb_p3/parent_data.yaml")
    require({k: v for k, v in raw.items() if k != "path"} ==
            {k: v for k, v in expected.items() if k != "path"}, "Parent split/classes configuration changed")
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "SDB-P3 requires nc=1")
    root = Path(resolved["path"]).resolve()
    for split in ("train", "val", "test"):
        require(Path(resolved[split]).resolve() == root / "images" / split, "Unexpected split path: " + split)
    inventory = dataset_inventory(root)
    parent = json.loads((ROOT / "docs/sdb_p3/parent_dataset_inventory.json").read_text(encoding="utf-8"))
    require(inventory == parent, "Dataset split paths/label contents differ from successful CBR+LIF parent")
    return inventory


def require_preflight(report_path, initialized, data, variant=DEFAULT_VARIANT):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    require(report.get("scope") == "server" and report.get("status") == "PASSED", "Matching PASSED server preflight required")
    identity, capacity = report.get("identity", {}), report.get("capacity", {})
    require(identity.get("commit") == head() and identity.get("variant") == variant, "Preflight SHA/variant mismatch")
    require(identity.get("files") == code_identity(), "Executable sources/configs changed after preflight")
    require(identity.get("source_sha256") == SOURCE_SHA256, "Wrong common source")
    require(identity.get("init_sha256") == sha256(initialized), "Controlled initialization changed")
    require(identity.get("data_sha256") == sha256(data), "Data configuration changed")
    require(identity.get("dataset_inventory") == verify_data(data), "Data identity changed after preflight")
    require(capacity.get("status") == "PASSED" and capacity.get("batch") == 16 and capacity.get("imgsz") == 640 and
            capacity.get("amp") is True and capacity.get("effective_updates", 0) >= 2,
            "B16/640/native AMP >=2 actual optimizer updates required")
    return report


def clean_code():
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True).strip(), "Commit source changes, then rerun preflight")
    require(os.environ.get("SDB_P3_SHA") == head(), "Source this experiment environment with the delivered full SHA")


def make_plan(args):
    p = paths(args.variant)
    initialized, data = Path(args.initialized or p["init"]), Path(args.data or p["data"])
    report_path = Path(args.preflight or p["preflight"])
    clean_code()
    require(initialized.is_file() and data.is_file() and report_path.is_file(), "Initialization/data/preflight missing")
    checked = require_preflight(report_path, initialized, data, args.variant)
    resolved, rows = recipe(initialized, args.variant, data)
    plan = dict(variant=args.variant, commit=head(), files=code_identity(), args=resolved,
                initialized=str(initialized.resolve()), init_sha256=sha256(initialized), data=str(data.resolve()),
                data_sha256=sha256(data), dataset_inventory=checked["identity"]["dataset_inventory"],
                preflight=str(report_path.resolve()), preflight_sha256=sha256(report_path),
                run=str(p["run"]), parent_args_sha256=sha256(ROOT / "docs/sdb_p3/parent_args.yaml"))
    p["launch"].mkdir(parents=True, exist_ok=True)
    dest = p["launch"] / "plan.json"
    if dest.exists():
        require(json.loads(dest.read_text(encoding="utf-8")) == plan, "Existing plan differs; preserved")
    else:
        write_json(dest, plan)
    write_json(p["launch"] / "parameter_diff.json", rows)
    YAML.save(p["launch"] / "train_args.yaml", resolved)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def validate_plan(plan):
    clean_code()
    require(plan["commit"] == head() and plan["files"] == code_identity(), "Plan code identity changed")
    require(plan["init_sha256"] == sha256(plan["initialized"]), "Planned initial state changed")
    require(plan["preflight_sha256"] == sha256(plan["preflight"]), "Preflight report changed")
    require(plan["parent_args_sha256"] == sha256(ROOT / "docs/sdb_p3/parent_args.yaml"), "Parent recipe changed")
    require(plan["args"] == recipe(plan["initialized"], plan["variant"], plan["data"])[0], "Planned recipe changed")
    require_preflight(plan["preflight"], plan["initialized"], plan["data"], plan["variant"])


def process_token(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def latest_attempt(p):
    attempts = sorted((p["launch"] / "attempts").glob("[0-9][0-9][0-9][0-9]"))
    return attempts[-1] if attempts else None


def run_state(p):
    attempt = latest_attempt(p)
    if not attempt:
        return dict(status="NOT_STARTED", completed_epochs=0, final_validation="NOT_RUN")
    file = attempt / "state.json"
    result = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else dict(status="DISPATCHED", completed_epochs=0)
    process = attempt / "process.json"
    if process.is_file():
        info = json.loads(process.read_text(encoding="utf-8"))
        token = process_token(info["pid"])
        if token and token == info["token"]:
            return result
    exit_file = attempt / "process_exit_code.txt"
    if exit_file.is_file():
        code = int(exit_file.read_text().strip())
        result["shell_exit_code"] = code
        if code and result["status"] not in {"FAILED", "INTERRUPTED"}:
            result["status"] = "INTERRUPTED" if code in {130, 137, 143} else "FAILED"
    if result["status"] in {"RUNNING", "FINAL_VALIDATING", "DISPATCHED"}:
        session = "sdb-" + p["run"].name
        active = shutil.which("tmux") and subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0
        if not active:
            result["status"] = "INTERRUPTED"
    return result


def dispatch(args, resume=False):
    p = paths(args.variant)
    plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    require(os.environ.get("CONDA_DEFAULT_ENV") == "rtdetr" and torch.cuda.is_available(), "Existing rtdetr CUDA environment required")
    require(shutil.which("tmux"), "Install/provide tmux before explicit start")
    require(ROOT.resolve() != MAIN.resolve(), "Independent worktree required")
    state = run_state(p)
    require(state["status"] not in {"RUNNING", "DISPATCHED", "FINAL_VALIDATING"}, "An existing worker is active")
    session = "sdb-" + p["run"].name
    require(subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode != 0, "tmux session already exists")
    checkpoint = None
    if resume:
        require(state["status"] in {"FAILED", "INTERRUPTED"}, "Resume only an interrupted/failed unfinished run")
        require(state.get("completed_epochs", 0) < 200 and state.get("final_validation") != "FAILED", "Training rounds completed; final validation failure needs evaluation, not retraining")
        checkpoint = p["run"] / "weights/last.pt"
        require(checkpoint.is_file(), "No actual last.pt to resume; existing failed output preserved")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(0 <= ckpt.get("epoch", -1) < 199 and ckpt.get("optimizer") is not None,
                "Checkpoint has no unfinished optimizer/epoch state; no reset/retrain allowed")
        del ckpt
    else:
        require(state["status"] == "NOT_STARTED" and not p["run"].exists(), "Existing unique run preserved; use explicit resume when eligible")
        lock = p["run"].with_name(p["run"].name + ".sdb-p3.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.mkdir(exist_ok=False)
        write_json(lock / "owner.json", dict(worktree=str(ROOT), commit=head(), variant=args.variant, created=now()))
    env_file = ROOT / "outputs/sdb_p3.env"
    require(env_file.is_file(), "Run sync_sdb_p3.sh to create this worktree environment")
    attempts = p["launch"] / "attempts"
    attempts.mkdir(exist_ok=True)
    number = len(list(attempts.glob("[0-9][0-9][0-9][0-9]"))) + 1
    attempt = attempts / f"{number:04d}"
    attempt.mkdir(exist_ok=False)
    write_json(attempt / "dispatch.json", dict(mode="resume" if resume else "start", session=session,
                checkpoint=str(checkpoint) if checkpoint else None, checkpoint_sha256=sha256(checkpoint) if checkpoint else None))
    write_json(attempt / "state.json", dict(status="DISPATCHED", completed_epochs=state.get("completed_epochs", 0), final_validation="NOT_RUN"))
    command = [sys.executable, "-u", str(ROOT / "tools/train_sdb_p3.py"), "worker", "--variant", args.variant, "--attempt", str(attempt)]
    code_file, console = attempt / "process_exit_code.txt", attempt / "console.log"
    script = "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
    script += "trap 'rc=$?; printf \"%s\\n\" \"$rc\" > " + shlex.quote(str(code_file)) + "' EXIT\n"
    script += "source " + shlex.quote(str(env_file)) + "\n"
    script += "cd " + shlex.quote(str(ROOT)) + "\n" + " ".join(map(shlex.quote, command)) + " >> " + shlex.quote(str(console)) + " 2>&1\n"
    (attempt / "worker.sh").write_text(script, encoding="utf-8")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash " + shlex.quote(str(attempt / "worker.sh"))], check=True)
    print(f"DISPATCHED {session}\nLog: {console}")


def disable_oom_retry(trainer):
    trainer._oom_retries = 3


def read_epoch_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]


def worker(args):
    p, attempt = paths(args.variant), Path(args.attempt).resolve()
    require(attempt.parent == (p["launch"] / "attempts").resolve(), "Unexpected worker output")
    plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
    launch = json.loads((attempt / "dispatch.json").read_text(encoding="utf-8"))
    state = dict(status="RUNNING", completed_epochs=0, final_validation="NOT_RUN", started=now())
    code = 1
    def save_state():
        write_json(attempt / "state.json", state)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        validate_plan(plan)
        write_json(attempt / "process.json", dict(pid=os.getpid(), token=process_token(os.getpid())))
        checkpoint = launch["checkpoint"] or plan["initialized"]
        if launch["checkpoint"]:
            require(sha256(checkpoint) == launch["checkpoint_sha256"], "Resume checkpoint changed since dispatch")
        else:
            require(not p["run"].exists(), "Unique run directory appeared after reservation; preserve it")

        class RecordingTrainer(RTDETRTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model, audit = build_training_model(cfg, weights, self.data, args.variant)
                write_json(attempt / "nc1_loading.json", audit)
                return model

            def final_eval(self):
                state.update(status="FINAL_VALIDATING", final_validation="RUNNING")
                save_state()
                try:
                    super().final_eval()
                except BaseException:
                    state["final_validation"] = "FAILED"
                    save_state()
                    raise
                state["final_validation"] = "PASSED"
                save_state()

        model = RTDETR(str(checkpoint))
        model.add_callback("on_train_batch_start", disable_oom_retry)
        def setup(trainer):
            # on_train_start runs before native trainer.epoch is assigned.
            state["completed_epochs"] = int(trainer.start_epoch)
            state["start_epoch"] = int(trainer.start_epoch)
            verify_model(trainer.model, args.variant, zero=launch["mode"] == "start")
            require(bool(trainer.amp), "Native AMP was disabled; refusing changed recipe")
            require(Path(trainer.save_dir).resolve() == p["run"].resolve(), "Native auto-suffix output refused")
            actual = vars(trainer.args)
            expected = dict(plan["args"])
            if launch["mode"] == "resume":
                expected.update(model=str(checkpoint), resume=str(checkpoint))
            differences = {k: [v, actual.get(k)] for k, v in expected.items()
                           if type(actual.get(k)) is not type(v) or actual.get(k) != v}
            require(not differences, f"Actual recipe differs: {differences}")
            ids = [id(v) for group in trainer.optimizer.param_groups for v in group["params"]]
            added = {n: ids.count(id(v)) for n, v in trainer.model.named_parameters() if is_added(n)}
            require(added and all(v == 1 for v in added.values()), "Every new parameter must occur exactly once in optimizer")
            require(type(trainer.optimizer) is torch.optim.AdamW, "Optimizer changed")
            YAML.save(attempt / "actual_train_args.yaml", actual)
            write_json(attempt / "training_setup.json", dict(amp=True, start_epoch=trainer.start_epoch,
                       new_optimizer_occurrences=added, parameters=sum(v.numel() for v in trainer.model.parameters()),
                       recipe_differences=differences))
            save_state()
        model.add_callback("on_train_start", setup)
        def epoch_record(trainer):
            epoch = int(trainer.epoch) + 1
            state["completed_epochs"] = max(state["completed_epochs"], epoch)
            save_state()
            if epoch in {40, 80}:
                reference = next((r for r in read_epoch_rows(ROOT / "docs/sdb_p3/parent_results.csv") if int(r["epoch"]) == epoch), None)
                actual = next((r for r in read_epoch_rows(p["run"] / "results.csv") if int(r["epoch"]) == epoch), None)
                observation = dict(epoch=epoch, parent=reference, current=actual,
                                   note="Same-epoch val observation only; no automatic stop or schedule change; patience=50")
                write_json(attempt / f"observation_epoch_{epoch}.json", observation)
                print(json.dumps(observation, ensure_ascii=False), flush=True)
        model.add_callback("on_fit_epoch_end", epoch_record)
        train_args = dict(plan["args"])
        if launch["mode"] == "resume":
            train_args.update(model=str(checkpoint), resume=True)
        model.train(trainer=RecordingTrainer, **train_args)
        state["status"] = "COMPLETED_200" if state["completed_epochs"] == 200 else "EARLY_STOPPED"
        code = 0
    except KeyboardInterrupt as error:
        state.update(status="INTERRUPTED", error=repr(error), traceback=traceback.format_exc())
        code = 130
        raise
    except BaseException as error:
        state.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        state.update(exit_code=code, finished=now())
        save_state()
        write_json(attempt / "exit_code.json", dict(exit_code=code))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "start", "resume", "status", "worker"))
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--initialized", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "plan":
        make_plan(args)
    elif args.mode in {"start", "resume"}:
        dispatch(args, resume=args.mode == "resume")
    elif args.mode == "worker":
        worker(args)
    else:
        p = paths(args.variant)
        print(json.dumps(dict(**run_state(p), run=str(p["run"]), latest_attempt=str(latest_attempt(p))), indent=2))


if __name__ == "__main__":
    main()
