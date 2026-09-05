"""Prepare a field-locked C2 -> C15 launch plan. Training requires --execute or --tmux explicitly."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import traceback

from init_rtdetr_r18_lite_cscef_v5_controlled import (
    DEFAULT_OUTPUT, ROOT, SOURCE_SHA256, V5_CFG, require, runtime_info, sha256, write_json,
)
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.utils import YAML

SERVER_ROOT = Path("/root/autodl-tmp/projects/Crack_RTDETR")
DEFAULT_C2_ARGS = SERVER_ROOT / "runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
DEFAULT_NAME = "c15_rtdetr_r18_lite_cscef_v5_e200_b16_onlineaug"
ALLOWED_CHANGES = {"model", "project", "name", "save_dir"}
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


def build_locked_args(baseline: dict, initialized: Path, project: Path, name: str) -> tuple[dict, list[dict]]:
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
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None
            and name not in {".", ".."}, "Experiment name must be a simple directory identifier.")
    target = deepcopy(baseline)
    target.update(model=str(initialized.resolve()), project=str(project.resolve()), name=name)
    if "save_dir" in target:
        target["save_dir"] = str(project.resolve() / name)
    rows = [{"field": key, "C2": baseline[key], "V5": target[key], "equal": baseline[key] == target[key],
             "change_allowed": key in ALLOWED_CHANGES} for key in sorted(baseline)]
    require(all(row["equal"] or row["change_allowed"] for row in rows), "Forbidden recipe difference.")
    require(set(target) == set(baseline), "Recipe fields were added or removed.")
    # get_cfg validates the exact complete dictionary; do not pass its merged defaults to train().
    get_cfg(overrides=deepcopy(target))
    return target, rows


def prepare(args: argparse.Namespace) -> tuple[dict, Path]:
    if not args.c2_args.is_file():
        raise FileNotFoundError(f"Required authoritative C2 args.yaml: {args.c2_args}. No fallback to C13 or defaults.")
    baseline = YAML.load(args.c2_args)
    target, rows = build_locked_args(baseline, args.initialized, args.project, args.name)
    output_dir = Path(target["project"]) / args.name
    require(not output_dir.exists(), f"Experiment already exists; select a fresh --name: {output_dir}")
    require(args.initialized.is_file(), f"Missing clean V5 initialization: {args.initialized}")
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
        "initialized_sha256": checkpoint_hash, "model_yaml": str(V5_CFG),
        "audit_report": str(args.audit_report.resolve()), "audit_sha256": sha256(args.audit_report),
        "data_config_sha256": sha256(data_path), "target_args": target, "field_comparison": rows,
        "allowed_changes": sorted(ALLOWED_CHANGES), "official_test_reference_only": OFFICIAL_TEST,
        "note": "No dataset loaded, no training or val/test evaluation executed by preparation.",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    require(not (args.report_dir / "console.log").exists(), "Console log already exists; use a fresh --report-dir.")
    plan_path = args.report_dir / "launch_plan.json"
    write_json(plan_path, plan)
    YAML.save(args.report_dir / "train_args.yaml", target)
    print(f"Prepared {len(rows)}-field C2 comparison: {plan_path.resolve()}")
    print(f"Changed fields: {[row['field'] for row in rows if not row['equal']]}")
    return plan, plan_path


def execute(plan: dict, report_dir: Path) -> None:
    """Capture native/Python stdout/stderr and an exit code when explicitly started in a separate process."""
    require(not plan["runtime"]["git_status"], "Commit all worktree changes before starting the formal experiment.")
    require(sha256(Path(plan["c2_args"])) == plan["c2_args_sha256"], "C2 args changed before launch.")
    require(sha256(Path(plan["initialized"])) == plan["initialized_sha256"], "Initialization changed before launch.")
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
                RTDETR(plan["initialized"]).train(**deepcopy(plan["target_args"]))
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
    parser.add_argument("--audit-report", type=Path, default=ROOT / "outputs/cscef_v5/audit.json")
    parser.add_argument("--project", type=Path, default=SERVER_ROOT / "runs/c_series")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/cscef_v5/launch_c15")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Explicitly start formal training and capture console/exit code.")
    mode.add_argument("--tmux", action="store_true", help="Explicitly start training in an independent detached tmux session.")
    args = parser.parse_args()
    plan, _ = prepare(args)
    if args.tmux:
        require(os.name == "posix", "tmux launch is intended for the Linux server.")
        require(not plan["runtime"]["git_status"], "Commit all worktree changes before launching tmux.")
        session = f"cscef_v5_{args.name}"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--execute",
                   "--c2-args", str(args.c2_args.resolve()), "--initialized", str(args.initialized.resolve()),
                   "--audit-report", str(args.audit_report.resolve()), "--project", str(args.project.resolve()),
                   "--name", args.name, "--report-dir", str(args.report_dir.resolve())]
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(command)], cwd=ROOT, check=True)
        print(f"Started tmux session {session}; console and exit code: {args.report_dir.resolve()}")
    elif args.execute:
        execute(plan, args.report_dir)
    else:
        print("Preparation only. No training or dataset evaluation was started.")


if __name__ == "__main__":
    main()
