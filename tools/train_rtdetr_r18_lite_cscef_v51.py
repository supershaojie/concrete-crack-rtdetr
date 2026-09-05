"""Prepare a field-locked C2 -> C17 launch plan. Training requires --execute or --tmux explicitly."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import traceback

from init_rtdetr_r18_lite_cscef_v51_controlled import (
    DEFAULT_OUTPUT, ROOT, SOURCE_SHA256, V51_CFG, require, runtime_info, sha256, write_json,
)
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML

SERVER_ROOT = Path("/root/autodl-tmp/projects/Crack_RTDETR")
DEFAULT_C2_ARGS = SERVER_ROOT / "runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
DEFAULT_NAME = "c17_rtdetr_r18_lite_cscef_v51_e200_b16_onlineaug"
ALLOWED_CHANGES = {"model", "name", "save_dir"}
# Sanity checks supplement, and NEVER replace, reading the full authoritative file.
C2_KEY_FIELDS = {
    "epochs": 200, "patience": 50, "batch": 16, "imgsz": 640, "device": 0, "workers": 8,
    "pretrained": True, "resume": False, "optimizer": "AdamW", "seed": 42, "deterministic": True,
    "rect": False, "cos_lr": True, "close_mosaic": 10, "amp": True,
    "lr0": 0.0005, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0001,
    "warmup_epochs": 5, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "box": 7.5, "cls": 0.5, "dfl": 1.5, "nbs": 64,
    "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.35, "degrees": 5, "translate": 0.1,
    "scale": 0.4, "shear": 1.5, "perspective": 0.0002, "flipud": 0.2, "fliplr": 0.5,
    "mosaic": 0.8, "mixup": 0.05, "cutmix": 0, "copy_paste": 0, "auto_augment": None, "erasing": 0,
}
OFFICIAL_TEST = {
    "split": "test", "imgsz": 640, "batch": 16, "workers": 8, "device": 0, "seed": 42,
    "conf": 0.001, "iou": 0.7, "max_det": 300, "half": False, "augment": False,
}


def build_locked_args(baseline: dict, initialized: Path, project: Path | None, name: str) -> tuple[dict, list[dict]]:
    """Preserve every input field, reject incomplete recipes and permit only model/output changes."""
    require(isinstance(baseline, dict), "C2 args must contain a YAML mapping.")
    missing = sorted(set(DEFAULT_CFG_DICT) - set(baseline))
    require(not missing, f"C2 args is incomplete for this source version; missing fields: {missing}. No defaults substituted.")
    require(not (set(baseline) - set(DEFAULT_CFG_DICT) - {"save_dir", "augmentations"}),
            "Unknown C2 fields require source-version review.")
    for key, expected in C2_KEY_FIELDS.items():
        actual = baseline.get(key)
        # Numeric device 0 and serialized '0', and YAML null/'None', are equivalent sanity-check representations.
        check = int(actual) if key == "device" and actual == "0" else actual
        check = None if key == "auto_augment" and check == "None" else check
        require(check == expected, f"C2 sanity check failed: {key}={actual!r}, expected {expected!r}.")
    require(baseline.get("task") == "detect" and baseline.get("mode") == "train", "Expected C2 detection training args.")
    require(baseline.get("cfg") is None, "C2 cfg override must be reviewed before generating a locked launch.")
    require(baseline.get("exist_ok") is False, "C2 exist_ok must be False to protect previous experiments.")
    require(isinstance(baseline.get("project"), str) and bool(baseline["project"]), "C2 project is required.")
    require(project is None or str(project) == baseline["project"], "project must equal the original C2 value exactly.")
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None
            and name not in {".", ".."}, "Experiment name must be a simple directory identifier.")
    target = deepcopy(baseline)
    target.update(model=str(initialized), name=name)
    if "save_dir" in target:
        path_type = PurePosixPath if baseline["project"].startswith("/") else Path
        target["save_dir"] = str(path_type(baseline["project"]) / name)
    rows = [{"field": key, "C2": baseline[key], "V51": target[key], "equal": same_value(baseline[key], target[key]),
             "change_allowed": key in ALLOWED_CHANGES} for key in sorted(baseline)]
    require(all(row["equal"] or row["change_allowed"] for row in rows), "Forbidden recipe difference.")
    require(set(target) == set(baseline), "Recipe fields were added or removed.")
    # get_cfg validates the exact complete dictionary; do not pass its merged defaults to train().
    validated = vars(get_cfg(cfg=deepcopy(target), overrides=deepcopy(target)))
    require(same_recipe(validated, target), "Configuration validation silently changed fields or values.")
    return target, rows


def same_value(a, b) -> bool:
    """Do not treat a changed YAML type (e.g. bool/int) as an equal recipe value."""
    return type(a) is type(b) and a == b


def same_recipe(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(same_value(a[key], b[key]) for key in a)


def prepare(args: argparse.Namespace) -> tuple[dict, Path]:
    if not args.c2_args.is_file():
        raise FileNotFoundError(f"Required authoritative C2 args.yaml: {args.c2_args}. No fallback to C13 or defaults.")
    baseline = YAML.load(args.c2_args)
    target, rows = build_locked_args(baseline, args.initialized.resolve(), args.project, args.name)
    output_dir = Path(target["project"]) / args.name
    require(not output_dir.exists(), f"Experiment already exists; select a fresh --name: {output_dir}")
    require(args.initialized.is_file(), f"Missing clean V51 initialization: {args.initialized}")
    require(args.audit_report.is_file(), f"Missing initialization/compatibility audit: {args.audit_report}")
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    require(audit.get("status") == "passed", "A passing CPU and CUDA audit is required before preparing a server launch.")
    require(audit["initialization"]["source_sha256"] == SOURCE_SHA256, "Audit used a different C2 initialization.")
    checkpoint_hash = sha256(args.initialized)
    require(audit["initialization"]["initialized_sha256"] == checkpoint_hash, "Audit does not match this checkpoint.")
    runtime = runtime_info()
    require(audit["runtime"].get("code_sha256") == runtime["code_sha256"], "Code changed since the audit; rerun it.")
    data_path = Path(target["data"])
    require(data_path.is_file(), f"C2 data configuration is unavailable: {data_path}. The data field will not be rewritten.")
    plan = {
        "status": "prepared_no_training", "runtime": runtime, "c2_args": str(args.c2_args.resolve()),
        "c2_args_sha256": sha256(args.c2_args), "initialized": str(args.initialized.resolve()),
        "initialized_sha256": checkpoint_hash, "model_yaml": str(V51_CFG),
        "audit_report": str(args.audit_report.resolve()), "audit_sha256": sha256(args.audit_report),
        "data_config_sha256": sha256(data_path), "target_args": target, "field_comparison": rows,
        "allowed_changes": sorted(ALLOWED_CHANGES), "official_test_reference_only": OFFICIAL_TEST,
        "note": "No dataset loaded, no training or val/test evaluation executed by preparation.",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    require(not any(args.report_dir.iterdir()), "Report directory is not empty; use --plan for an existing preparation.")
    plan_path = args.report_dir / "launch_plan.json"
    write_json(plan_path, plan)
    YAML.save(args.report_dir / "train_args.yaml", target)
    print(f"Prepared {len(rows)}-field C2 comparison: {plan_path.resolve()}")
    print(f"Changed fields: {[row['field'] for row in rows if not row['equal']]}")
    return plan, plan_path


def validate_plan(plan: dict, report_dir: Path) -> None:
    """Recheck the prepared recipe and its inputs in the actual worker process, without rewriting it."""
    runtime = runtime_info()
    require(not runtime["git_status"], "Commit all worktree changes before starting the formal experiment.")
    require(runtime["git_commit"] == plan["runtime"]["git_commit"], "Git commit changed since preparation.")
    require(runtime["code_sha256"] == plan["runtime"]["code_sha256"], "Source changed since preparation.")
    for key in ("torch_version", "ultralytics_version", "python", "ultralytics_file"):
        require(runtime[key] == plan["runtime"][key], f"Runtime {key} changed since preparation.")
    for field in ("c2_args", "initialized", "audit_report"):
        hash_key = "audit_sha256" if field == "audit_report" else field + "_sha256"
        require(sha256(Path(plan[field])) == plan[hash_key], f"{field} changed before launch.")
    baseline = YAML.load(plan["c2_args"])
    expected, rows = build_locked_args(baseline, Path(plan["initialized"]), None, plan["target_args"]["name"])
    require(same_recipe(expected, plan["target_args"]) and rows == plan["field_comparison"], "Prepared recipe was modified.")
    require(plan["allowed_changes"] == sorted(ALLOWED_CHANGES), "Invalid field change policy.")
    require(same_recipe(YAML.load(report_dir / "train_args.yaml"), expected), "train_args.yaml differs from launch plan.")
    require(sha256(Path(expected["data"])) == plan["data_config_sha256"], "C2 data configuration changed.")
    audit = json.loads(Path(plan["audit_report"]).read_text(encoding="utf-8"))
    require(audit.get("status") == "passed" and audit["runtime"]["code_sha256"] == runtime["code_sha256"],
            "Current source has no passing CPU/CUDA audit.")
    require(audit["initialization"]["source_sha256"] == SOURCE_SHA256
            and audit["initialization"]["initialized_sha256"] == plan["initialized_sha256"], "Audit provenance mismatch.")
    output_dir = Path(expected.get("save_dir", str(Path(expected["project"]) / expected["name"])))
    require(not output_dir.exists(), f"Experiment already exists: {output_dir}")
    require(not any((report_dir / name).exists() for name in ("console.log", "exit_code.json", "launch_claim.json")),
            "This preparation has already been launched; logs will not be overwritten.")


def locked_trainer(target: dict):
    """Guard the real Trainer boundary as well as the saved plan; no head/RNG intervention."""
    class C2LockedTrainer(RTDETRTrainer):
        def __init__(self, overrides=None, _callbacks=None):
            actual = deepcopy(overrides)
            actual.pop("session", None)  # API runtime object, never a training recipe field.
            require(same_recipe(actual, target), "RTDETR API changed the complete C2 recipe before Trainer.")
            super().__init__(cfg=deepcopy(target), overrides=deepcopy(overrides), _callbacks=_callbacks)
            require(same_recipe(vars(self.args), target), "Trainer changed the locked C2 recipe during setup.")

    return C2LockedTrainer


def execute(plan: dict, report_dir: Path) -> None:
    """Capture native/Python stdout/stderr and an exit code when explicitly started in a separate process."""
    validate_plan(plan, report_dir)
    # Exclusive claim prevents concurrent launchers from sharing a log/exit status.
    with (report_dir / "launch_claim.json").open("x", encoding="utf-8") as file:
        json.dump({"pid": os.getpid(), "git_commit": plan["runtime"]["git_commit"]}, file)
    log_path = report_dir / "console.log"
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    exit_code = 1
    started = datetime.now(timezone.utc).isoformat()
    try:
        with log_path.open("x", encoding="utf-8") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                print(json.dumps({"git_commit": plan["runtime"]["git_commit"], "ultralytics_file": plan["runtime"]["ultralytics_file"],
                                  "c2_args_sha256": plan["c2_args_sha256"], "initialized_sha256": plan["initialized_sha256"]}))
                # The complete original C2 dictionary with only whitelisted replacements.
                RTDETR(plan["initialized"]).train(trainer=locked_trainer(plan["target_args"]),
                                                  **deepcopy(plan["target_args"]))
                exit_code = 0
            except BaseException as error:
                if isinstance(error, KeyboardInterrupt):
                    exit_code = 130
                elif isinstance(error, SystemExit):
                    exit_code = error.code if isinstance(error.code, int) else 1
                traceback.print_exc()
                raise
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        write_json(report_dir / "exit_code.json", {"exit_code": exit_code, "started_utc": started,
                                                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                                                   "console_log": str(log_path.resolve())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-args", type=Path, default=DEFAULT_C2_ARGS, help="Original complete C2 args; required, no fallback.")
    parser.add_argument("--initialized", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-report", type=Path, default=ROOT / "outputs/cscef_v51/audit.json")
    parser.add_argument("--project", type=Path, default=None, help="Optional assertion only; must equal C2 project exactly.")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/cscef_v51/launch_c17")
    parser.add_argument("--plan", type=Path, help="Reuse a prepared launch_plan.json without overwriting preparation files.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Explicitly start formal training and capture console/exit code.")
    mode.add_argument("--tmux", action="store_true", help="Explicitly start training in an independent detached tmux session.")
    args = parser.parse_args()
    if args.plan:
        require(args.execute or args.tmux, "--plan requires explicit --execute or --tmux.")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        args.report_dir = args.plan.resolve().parent
        plan_path = args.plan.resolve()
        validate_plan(plan, args.report_dir)
    else:
        plan, plan_path = prepare(args)
    if args.tmux:
        require(os.name == "posix", "tmux launch is intended for the Linux server.")
        validate_plan(plan, args.report_dir)
        session = f"cscef_v51_{plan['target_args']['name']}"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--execute",
                   "--plan", str(plan_path.resolve())]
        # A tmux server may predate conda activation. Pass the current interpreter and environment explicitly.
        exports = {"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT / "ultralytics-main"), "PYTHONUNBUFFERED": "1"}
        exports.update({key: os.environ[key] for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "LD_LIBRARY_PATH")
                        if key in os.environ})
        shell_command = shlex.join(["env", *(f"{k}={v}" for k, v in exports.items()), *command])
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), shell_command], check=True)
        print(f"Started tmux session {session}; console and exit code: {args.report_dir.resolve()}")
    elif args.execute:
        execute(plan, args.report_dir)
    else:
        print("Preparation only. No training or dataset evaluation was started.")


if __name__ == "__main__":
    main()
