"""Plan, explicitly start, or resume the pinned TRC v1 experiment (no implicit training)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import uuid

import torch
from init_trc_v1 import (ROOT, MODEL_DIR, VARIANTS, DEFAULT_VARIANT, SOURCE_SHA256, require, runtime,
                         sha256, verify_model, build_training_model, is_added)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.data.utils import check_det_dataset
from c19_lif_v1_data import dataset_inventory
from train_c19_lif_v1 import ensure_amp_resources, disable_oom_retry, optimizer_groups, process_token

MAIN = Path(os.environ.get("TRC_V1_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
ARCHIVE = ROOT / "docs/trc_v1/cbr_lif_args.yaml"
IDENTITY_FIELDS = {"model", "name", "project", "save_dir", "data"}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    """Atomic replacement for this invocation's own state; immutable plans use x mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def recipe(c2_path, variant, initialized, *, data=None, project=None):
    """Inherit every archived field; only explicit identity/environment paths differ."""
    source, expected = YAML.load(c2_path), YAML.load(ARCHIVE)
    require(set(source) == set(expected), "Authoritative recipe fields changed or are missing")
    require(all(type(source[k]) is type(expected[k]) and source[k] == expected[k] for k in source),
            "Recipe differs from the complete successful CBR + LIF archive")
    require(variant in VARIANTS, "Unknown TRC variant")
    project = Path(project or MAIN / "runs/c_series").resolve()
    target = dict(source)
    target.update(model=str(Path(initialized).resolve()), name=VARIANTS[variant][2],
                  project=str(project), save_dir=str(project / VARIANTS[variant][2]),
                  data=str(Path(data or MAIN / "configs/crack_autodl.yaml").resolve()))
    rows = [dict(field=k, original=source[k], proposed=target[k], changed=source[k] != target[k],
                 reason="model/output identity or verified environment path" if source[k] != target[k] else "inherited")
            for k in sorted(source)]
    require({r["field"] for r in rows if r["changed"]} <= IDENTITY_FIELDS, "Recipe mutation outside identity fields")
    return target, rows


def verify_data_config(path):
    actual = dict(YAML.load(path))
    expected = dict(YAML.load(ROOT / "docs/c19_lif_v1/c2_data.yaml"))
    require(isinstance(actual.get("path"), str) and Path(actual["path"]).is_absolute(), "Use an explicit absolute dataset root")
    actual.pop("path")
    expected.pop("path")
    require(actual == expected, "Original train/val/test splits or crack class changed")


def read_plan(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.get("schema") == "trc_v1_plan_v1", "Invalid TRC plan")
    return value


def initialization_audit(initialized, source, variant):
    """Read the audit embedded in the exact checkpoint, independent of --report location."""
    value = torch_load(initialized, map_location="cpu")
    provenance = value.get("trc_v1_provenance", {})
    require(provenance.get("variant") == variant, "Initialization variant/provenance missing")
    require(provenance.get("source_sha256") == sha256(source) == SOURCE_SHA256, "Initialization source must be the fixed ImageNet checkpoint")
    require(value.get("epoch") == -1 and value.get("optimizer") is None, "Initialization has training updates")
    return dict(origin="initialized checkpoint trc_v1_provenance", checkpoint_sha256=sha256(initialized),
                source_sha256=SOURCE_SHA256, provenance=provenance)


def create_plan(args):
    from trc_v1_common import fingerprint
    output = args.plan.resolve()
    require(not output.exists(), "Existing plan protected; select a new plan path")
    for path in (args.source, args.initialized, args.data, args.c2_args):
        require(path.is_file(), "Missing plan input: " + str(path))
    target, rows = recipe(args.c2_args, args.variant, args.initialized, data=args.data, project=args.project)
    verify_data_config(target["data"])
    require(not Path(target["save_dir"]).exists(), "Existing formal run protected")
    require(not output.is_relative_to(Path(target["save_dir"])), "Plan must remain outside the formal run")
    report = dict(schema="trc_v1_plan_v1", created=now(), variant=args.variant, args=target,
                  parameter_diff=rows, source=str(args.source.resolve()), initialized=str(args.initialized.resolve()),
                  authoritative_args=str(args.c2_args.resolve()), authoritative_args_sha256=sha256(args.c2_args),
                  preflight=str(args.preflight.resolve()), runtime=runtime(),
                  fingerprint=fingerprint(args.variant, args.source, args.initialized, args.data),
                  initialization_audit=initialization_audit(args.initialized, args.source, args.variant),
                  formal_training="NOT_STARTED", test="NOT_RUN")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(plan=str(output), run=target["save_dir"], fields=len(target),
                         formal_training="NOT_STARTED", run_directory_created=False), indent=2))
    return report


def verify_plan(plan):
    from trc_v1_common import fingerprint, verify_preflight
    require(runtime()["commit"] == plan["runtime"]["commit"], "Source HEAD changed after plan")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(),
            "Commit/review tracked source changes before formal execution")
    expected, _ = recipe(plan["authoritative_args"], plan["variant"], plan["initialized"],
                         data=plan["args"]["data"], project=plan["args"]["project"])
    require(expected == plan["args"], "Plan recipe was changed")
    require(sha256(plan["authoritative_args"]) == plan["authoritative_args_sha256"], "Archived recipe changed")
    require(plan["initialization_audit"] == initialization_audit(plan["initialized"], plan["source"], plan["variant"]),
            "Initialization audit differs from the selected checkpoint")
    require(fingerprint(plan["variant"], plan["source"], plan["initialized"], plan["args"]["data"]) == plan["fingerprint"],
            "Source, config, fixed weights or initialization changed after planning")
    verify_preflight(plan["preflight"], plan["variant"], plan["source"], plan["initialized"], plan["args"]["data"])
    verify_data_config(plan["args"]["data"])
    return plan


def arguments_differ(actual, expected, *, resume=None):
    expected = dict(expected)
    if resume:
        expected.update(model=str(resume), resume=str(resume))
    return {k: [v, actual.get(k)] for k, v in expected.items()
            if type(actual.get(k)) is not type(v) or actual.get(k) != v}


def validate_resume_checkpoint(checkpoint, plan):
    """Validate before native resume can create directories or adjust epochs."""
    checkpoint = Path(checkpoint).resolve()
    require(checkpoint == Path(plan["args"]["save_dir"]).resolve() / "weights/last.pt", "Resume only this run's last.pt")
    value = torch_load(checkpoint, map_location="cpu")
    epoch = value.get("epoch", -1)
    require(isinstance(epoch, int) and 0 <= epoch < plan["args"]["epochs"] - 1,
            "Completed/stripped checkpoint cannot resume or extend the 200-epoch recipe")
    require(value.get("optimizer") is not None and value.get("scaler") is not None,
            "Resume checkpoint must preserve optimizer and native AMP scaler")
    differences = arguments_differ(value.get("train_args", {}), plan["args"])
    # A resumed checkpoint records its own model/resume identity, never a new recipe.
    for k in ("model", "resume"):
        differences.pop(k, None)
    require(not differences, "Resume training arguments differ: " + repr(differences))
    model = value.get("ema") if value.get("ema") is not None else value.get("model")
    require(model is not None, "Resume model/EMA missing")
    verify_model(model, plan["variant"], zero=False)
    return dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint), epoch=epoch,
                learned_trc_preserved=True, optimizer_restored_by="native RTDETRTrainer.resume_training")


def completion_state(trainer):
    epochs = trainer.epoch + 1
    if epochs >= trainer.args.epochs:
        return "COMPLETED_200_EPOCHS"
    if trainer.stop and epochs - trainer.stopper.best_epoch >= trainer.stopper.patience:
        return "EARLY_STOPPED_PATIENCE"
    return "INTERRUPTED"


def execute(plan_path, resume=False, checkpoint=None):
    plan_path = Path(plan_path).resolve()
    plan = verify_plan(read_plan(plan_path))
    require(torch.cuda.is_available(), "Formal training requires the existing server CUDA environment")
    require(os.environ.get("CONDA_DEFAULT_ENV") == "rtdetr", "Activate the existing rtdetr environment")
    run, metadata = Path(plan["args"]["save_dir"]), plan_path.parent
    require(run.exists() if resume else not run.exists(), "Run existence incompatible with start/resume; no overwrite or name2")
    checkpoint = (Path(checkpoint) if checkpoint else run / "weights/last.pt").resolve() if resume else None
    resume_audit = validate_resume_checkpoint(checkpoint, plan) if resume else None
    resolved = check_det_dataset(plan["args"]["data"], autodownload=False)
    inventory = dataset_inventory(Path(resolved["path"]))
    require(resolved["nc"] == 1, "Expected the original crack nc=1 dataset")
    inventory_path = metadata / "dataset_inventory.json"
    if resume:
        require(json.loads(inventory_path.read_text(encoding="utf-8")) == inventory, "Dataset paths/labels changed since start")
    lock = run.with_name(run.name + ".trc_v1.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive shared lock prevents competing workers across different worktrees.
    with lock.open("x", encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(), process_token=process_token(os.getpid()), plan=str(plan_path)), f)
    attempt = metadata / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    original_signal = signal.getsignal(signal.SIGTERM)
    state = dict(status="STARTING", mode="resume" if resume else "start", pid=os.getpid(),
                 process_token=process_token(os.getpid()), started=now(), attempt=str(attempt),
                 plan_sha256=sha256(plan_path), optimizer_updates_before_start=0 if not resume else "restored")

    def save_state():
        atomic_json(attempt / "state.json", state)
        atomic_json(metadata / "training_state.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Signal " + str(signum))

    try:
        attempt.mkdir(parents=True, exist_ok=False)
        save_state()
        signal.signal(signal.SIGTERM, interrupted)
        ensure_amp_resources(MAIN, attempt / "amp_resources.json")
        shutil.copyfile(plan["preflight"], attempt / "preflight.json")
        shutil.copyfile(plan["authoritative_args"], attempt / "authoritative_args.yaml")
        shutil.copyfile(plan["args"]["data"], attempt / "data_config.yaml")
        atomic_json(attempt / "source_record.json", dict(runtime=runtime(), fingerprint=plan["fingerprint"], resume=resume_audit))
        atomic_json(attempt / "parameter_diff.json", plan["parameter_diff"])
        if not resume:
            require(not inventory_path.exists(), "Existing training audit protected; use a new plan directory")
            atomic_json(inventory_path, inventory)
        (attempt / "pip_freeze.txt").write_bytes(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"]))

        class RecordingTrainer(RTDETRTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model, audit = build_training_model(cfg, weights, self.data, plan["variant"])
                atomic_json(attempt / "nc1_loading.json", audit)
                return model

        model = RTDETR(str(checkpoint if resume else plan["initialized"]))
        model.add_callback("on_train_batch_start", disable_oom_retry)

        def setup(trainer):
            verify_model(trainer.model, plan["variant"], zero=not resume)
            require(bool(trainer.amp), "Native AMP check failed; stopping without changing the recipe")
            differences = arguments_differ(vars(trainer.args), plan["args"], resume=checkpoint)
            require(not differences, "Effective recipe changed: " + repr(differences))
            require(trainer.save_dir.resolve() == run.resolve(), "Native Trainer redirected the run")
            ids = [id(v) for g in trainer.optimizer.param_groups for v in g["params"]]
            require(type(trainer.optimizer) is torch.optim.AdamW, "Expected original AdamW")
            require(all(ids.count(id(v)) == 1 for v in trainer.model.parameters()), "Optimizer parameter absent or duplicated")
            added = {n: ids.count(id(v)) for n, v in trainer.model.named_parameters() if is_added(n)}
            require(len(added) == 3 and all(v == 1 for v in added.values()), "TRC must enter AdamW exactly once")
            YAML.save(attempt / "actual_train_args.yaml", vars(trainer.args))
            atomic_json(attempt / "training_setup.json", dict(amp=bool(trainer.amp), optimizer="AdamW",
                        added_optimizer_occurrences=added, optimizer_groups=optimizer_groups(trainer.model, trainer.optimizer),
                        recipe_differences=differences, resume=resume_audit))
            state.update(status="RUNNING", start_epoch=trainer.start_epoch)
            save_state()

        def epoch_end(trainer):
            state.update(completed_epochs=trainer.epoch + 1, best_epoch=trainer.stopper.best_epoch)
            save_state()

        model.add_callback("on_train_start", setup)
        model.add_callback("on_fit_epoch_end", epoch_end)
        kwargs = dict(plan["args"])
        if resume:
            kwargs.update(model=str(checkpoint), resume=str(checkpoint))
        else:
            # Atomic reservation after all gates; explicit save_dir prevents native name incrementing.
            run.mkdir(parents=True, exist_ok=False)
        model.train(trainer=RecordingTrainer, **kwargs)
        state.update(status=completion_state(model.trainer), completed_epochs=model.trainer.epoch + 1, exit_code=0)
    except KeyboardInterrupt as error:
        state.update(status="INTERRUPTED", exit_code=130, error=repr(error))
        raise
    except BaseException as error:
        state.update(status="FAILED", exit_code=1, error=repr(error))
        raise
    finally:
        state["finished"] = now()
        save_state()
        signal.signal(signal.SIGTERM, original_signal)
        lock.unlink()
    return state


def status(plan_path):
    path = Path(plan_path).resolve().parent / "training_state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else dict(status="NOT_STARTED")
    if state["status"] in {"RUNNING", "STARTING"} and process_token(state["pid"]) != state.get("process_token"):
        state = dict(state, status="INTERRUPTED_UNCLEAN", note="Process no longer matches; inspect checkpoint and stale lock before resuming")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--initialized", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--data", type=Path, default=MAIN / "configs/crack_autodl.yaml")
    p.add_argument("--c2-args", type=Path, default=ARCHIVE)
    p.add_argument("--project", type=Path, default=MAIN / "runs/c_series")
    p.add_argument("--plan", type=Path)
    for name in ("start", "resume", "status"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        if name == "resume":
            p.add_argument("--checkpoint", type=Path)
    p = sub.add_parser("prepare-amp")
    p.add_argument("--main", type=Path, default=MAIN)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        args.plan = args.plan or ROOT / "outputs/trc_v1" / args.variant / "plan.json"
        create_plan(args)
    elif args.mode == "prepare-amp":
        require(not args.output.exists(), "Existing AMP resource audit protected")
        ensure_amp_resources(args.main, args.output)
    elif args.mode == "status":
        status(args.plan)
    else:
        execute(args.plan, resume=args.mode == "resume", checkpoint=getattr(args, "checkpoint", None))


if __name__ == "__main__":
    main()
