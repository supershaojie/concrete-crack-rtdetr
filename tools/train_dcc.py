"""DCC's explicit plan/start/resume lifecycle; never launches as a side effect of preflight."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from dcc_common import (ROOT, MODEL_DIR, VARIANTS, BASE_COMMIT, SOURCE_SHA256, require,
                        sha256, write_json, runtime, verify_model, build_training_model, is_added)
from ultralytics import RTDETR
from ultralytics.data.utils import check_det_dataset, img2label_paths
from dcc_checkpoint import DCCCheckpointTrainer, require_checkpoint_policy, optimizer_param_names
from ultralytics.utils import YAML

DEFAULT_MAIN = Path(os.environ.get("DCC_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
IDENTITY_FIELDS = {"model", "name", "project", "save_dir", "data"}


def code_identity():
    """Content identity survives the final commit without rewriting old test evidence."""
    files = list((ROOT / "tools").glob("*dcc*.py")) + list((ROOT / "docs/dcc").glob("*.yaml"))
    files += list((ROOT / "tools").glob("*dcc*.sh"))
    files += [ROOT / "docs/dcc/parent_dataset_identity.json", ROOT / "docs/dcc/environment.sh"]
    files += list((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files += list(MODEL_DIR.glob("*.yaml"))
    files += [ROOT / "tools" / name for name in ("init_c19_lif_v1.py", "init_lif_down.py", "lif_down_topology.py",
              "c19_lif_v1_data.py", "c19_lif_v1_probe.py", "c19_lif_v1_diagnostic.py", "c19_lif_v1_cutoff.py")]
    files += [MODEL_DIR / item[0] for item in VARIANTS.values()]
    files += [ROOT / "ultralytics-main/ultralytics" / name for name in (
        "nn/modules/dcc.py", "nn/modules/__init__.py", "nn/tasks.py", "nn/autobackend.py",
        "engine/trainer.py", "models/rtdetr/train.py", "models/rtdetr/val.py",
        "nn/modules/cbr.py", "nn/modules/lif_down.py")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(set(files)) if p.is_file()}


def recipe(variant, init, main=DEFAULT_MAIN, data=None, parent_args=None):
    """Use all fields of the inspected successful parent, replacing identity only."""
    archived = ROOT / "docs/dcc/parent_actual_args.yaml"
    require(archived.is_file(), "Missing inspected successful parent recipe archive")
    parent = YAML.load(archived)
    if parent_args:
        actual = YAML.load(parent_args)
        require(actual == parent, "Actual successful parent args differ from the archived authority")
    main, init = Path(main).resolve(), Path(init).resolve()
    name = VARIANTS[variant][2]
    target = dict(parent)
    target.update(model=str(init), data=str(Path(data or main / "configs/crack_autodl.yaml").resolve()),
                  project=str(main / "runs/c_series"), name=name,
                  save_dir=str(main / "runs/c_series" / name))
    require(parent["epochs"] == 200 and parent["batch"] == 16 and parent["imgsz"] == 640 and
            parent["nbs"] == 64 and parent["amp"] is True and parent["freeze"] is None and
            parent["optimizer"] == "AdamW" and parent["lr0"] == .0005 and parent["exist_ok"] is False,
            "Archive is not the required successful 200-epoch online-augmentation recipe")
    differences = [{"field": k, "parent": parent[k], "target": target[k], "changed": parent[k] != target[k]}
                   for k in sorted(parent)]
    require({r["field"] for r in differences if r["changed"]} <= IDENTITY_FIELDS, "Recipe mutation")
    return target, differences


def dataset_identity(data):
    """Fingerprint actual split image paths and label bytes; never re-split or run inference."""
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "DCC requires one crack class")
    root = Path(resolved["path"]).resolve()
    result = {"config_sha256": sha256(data), "root": str(root),
              "names": {str(k): v for k, v in resolved["names"].items()}, "splits": {}}
    for split, expected in (("train", 6048), ("val", 1728), ("test", 864)):
        locations = resolved.get(split)
        require(locations, "Missing split " + split)
        files = []
        for location in locations if isinstance(locations, list) else [locations]:
            location = Path(location)
            if location.is_dir():
                files += [p.resolve() for p in location.rglob("*") if p.suffix.lower() in
                          {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
            elif location.is_file():
                for line in location.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        p = Path(line.strip())
                        files.append((location.parent / p if not p.is_absolute() else p).resolve())
        files = sorted(files)
        require(len(files) == len(set(files)) == expected, "Unexpected/duplicate image inventory for " + split)
        identities, labels, boxes = [], [], 0
        for image, label in zip(files, img2label_paths([str(p) for p in files])):
            require(image.is_file(), "Missing image " + str(image))
            label = Path(label)
            require(label.is_file(), "Missing label " + str(label))
            rows = [r.split() for r in label.read_text(encoding="utf-8").splitlines() if r.strip()]
            for row in rows:
                require(len(row) == 5 and float(row[0]) == 0 and
                        all(__import__("math").isfinite(float(v)) for v in row) and
                        all(0 <= float(v) <= 1 for v in row[1:]), "Invalid crack label " + str(label))
            boxes += len(rows)
            identities.append(image.relative_to(root).as_posix())
            labels.append(label.relative_to(root).as_posix() + ":" + sha256(label))
        result["splits"][split] = dict(images=len(files), boxes=boxes,
            paths_sha256=hashlib.sha256("\n".join(identities).encode()).hexdigest(),
            labels_sha256=hashlib.sha256("\n".join(labels).encode()).hexdigest())
    historical = ROOT / "docs/dcc/parent_dataset_identity.json"
    require(historical.is_file(), "Missing successful-parent actual dataset identity archive")
    authority = json.loads(historical.read_text(encoding="utf-8"))
    for split, identity in result["splits"].items():
        expected = authority[split]
        for target_key, parent_key in (("images", "images"), ("boxes", "boxes"),
                                       ("paths_sha256", "split_paths_sha256"),
                                       ("labels_sha256", "label_inventory_sha256")):
            require(identity[target_key] == expected[parent_key], "Dataset differs from historical parent: " + split + "." + target_key)
    return result


def checkpoint(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def verify_initialization(path, variant):
    ckpt = checkpoint(path)
    require(ckpt.get("epoch") == -1 and all(ckpt.get(k) is None for k in ("optimizer", "scaler", "ema")),
            "Start requires a fresh controlled initialization without learned training state")
    metadata = ckpt.get("dcc", {})
    require(metadata.get("variant") == variant and metadata.get("source_sha256") == SOURCE_SHA256 and
            metadata.get("base_commit") == BASE_COMMIT, "Controlled initialization identity missing/mismatched")
    verify_model(ckpt["model"], variant, zero=True)
    return metadata


def optimizer_coverage(model, optimizer):
    ids = Counter(id(p) for group in optimizer.param_groups for p in group["params"])
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    require(set(ids) == expected and all(v == 1 for v in ids.values()), "Optimizer drops/duplicates trainable parameters")
    require(all(p.requires_grad for p in model.parameters()), "Unexpected frozen common/new parameter")
    return {"trainable_tensors": len(expected), "every_parameter_exactly_once": True,
            "new": {n: ids[id(p)] for n, p in model.named_parameters() if is_added(n)}}


def disable_oom_retry(trainer):
    # Preserve B16 even in the native first-epoch auto-recovery branch.
    trainer._oom_retries = 3


def strict_gate(preflight_path, checks_path, variant, init, data):
    require(preflight_path and checks_path, "start/resume require explicit preflight and engineering check reports")
    capacity = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    checks = json.loads(Path(checks_path).read_text(encoding="utf-8"))
    require(capacity.get("status") == "PASSED", "Native B16/640 preflight is not PASSED")
    require(capacity.get("variant") == variant and capacity.get("init_sha256") == sha256(init), "Preflight init/variant changed")
    require(capacity.get("code_identity") == code_identity(), "Code/config differs from capacity-tested tree")
    current_runtime = runtime()
    for key in ("python", "torch", "cuda", "gpu", "ultralytics", "worktree"):
        require(capacity.get("runtime", {}).get(key) == current_runtime.get(key),
                "Capacity check must run in the actual training environment: " + key)
    require(capacity.get("dataset_identity") == dataset_identity(data), "Actual data path/label identity changed")
    c = capacity.get("capacity", {})
    require(c.get("batch") == 16 and c.get("imgsz") == 640 and c.get("AMP") is True and
            c.get("effective_updates", 0) >= 2 and c.get("observed_batches", 99) <= 16,
            "Missing bounded B16/640 native AMP update evidence")
    require(checks.get("status") in {"PASSED", "PRECISION_NOTE"}, "Engineering lifecycle checks are not complete")
    require(checks.get("variant") == variant and checks.get("initialization_sha256") == sha256(init),
            "Engineering check initialization/variant changed")
    require(checks.get("code_identity") == code_identity(), "Code/config differs from engineering-tested tree")
    for key in ("python", "torch", "cuda", "gpu", "ultralytics", "worktree"):
        require(checks.get("runtime", {}).get(key) == current_runtime.get(key),
                "Engineering checks must run in the actual training environment: " + key)
    require(not checks.get("pending") and not checks.get("failed"), "Required engineering items remain PENDING/FAILED")
    for device in ("cpu_fp32", "cuda_fp32", "cuda_native_amp"):
        observed = checks.get("devices", {}).get(device, {})
        lifecycle = observed.get("lifecycle", {})
        accepted_lifecycle = lifecycle.get("status") == "PASSED"
        # Independent CUDA backward equality is a separate requirement. Neither
        # restore correctness, same-gradient replay nor parent variation waives it.
        require(lifecycle.get("checkpoint_policy") == "optimizer_fp32_v1",
                "Engineering check used the historical FP16 optimizer serializer")
        require(lifecycle.get("restoration", {}).get("status") == "PASSED" and
                lifecycle.get("same_gradient_replay", {}).get("status") == "PASSED",
                "Strict complete-state restore/replay evidence missing: " + device)
        if device == "cuda_native_amp":
            require(lifecycle.get("same_gradient_replay", {}).get("amp_scaler_path") is True,
                    "Same-gradient replay did not exercise native AMP/scaler")
        require(lifecycle.get("raw_next_update_allclose") is True,
                "Native resume next-update mismatch remains unresolved: " + device)
        require(observed.get("status") == "PASSED" and accepted_lifecycle and
                observed.get("fusion", {}).get("status") == "PASSED", "Missing engineering/lifecycle/fusion gate: " + device)
        if device != "cpu_fp32":
            require(observed.get("fusion", {}).get("cuda_half", {}).get("status") == "PASSED", "CUDA half lifecycle gate missing")
    return {"capacity_report_sha256": sha256(preflight_path), "checks_report_sha256": sha256(checks_path)}


def plan(args):
    recipe_args, differences = recipe(args.variant, args.init, args.main, args.data, args.parent_args)
    return {"variant": args.variant, "runtime": runtime(), "code_identity": code_identity(),
            "args": recipe_args, "recipe_differences": differences,
            "formal_training": "NOT_STARTED", "final_test": "NOT_RUN",
            "checkpoint_policy": "optimizer_fp32_v1", "checkpoint_ema_dtype": "float16"}


def run(args):
    require(torch.cuda.is_available(), "CUDA unavailable; formal training not launched")
    require(ROOT.resolve() != args.main.resolve() and (ROOT / ".git").is_file(), "Use the independent DCC worktree")
    require(not subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(),
                                        "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                        text=True).strip(), "Commit tested source before formal start/resume")
    info = plan(args)
    train_args = info["args"]
    require(Path(train_args["data"]).is_file(), "Missing actual dataset config")
    info["initialization"] = verify_initialization(args.init, args.variant)
    info["init_sha256"] = sha256(args.init)
    info["gates"] = strict_gate(args.preflight, args.checks, args.variant, args.init, train_args["data"])
    out = ROOT / "outputs/dcc" / args.variant / "training"
    run_dir = Path(train_args["save_dir"])
    if args.mode == "start":
        require(not run_dir.exists(), "Existing formal run protected: " + str(run_dir))
        require(not out.exists(), "Existing launch metadata protected: " + str(out))
        reservation = run_dir.with_name(run_dir.name + ".dcc.lock")
        reservation.parent.mkdir(parents=True, exist_ok=True)
        reservation.mkdir(exist_ok=False)
        write_json(reservation / "owner.json", dict(variant=args.variant, worktree=str(ROOT),
                   pid=os.getpid(), init_sha256=info["init_sha256"], code_identity=info["code_identity"]))
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", info)
        selected, saved = args.init, None
    else:
        require(args.checkpoint and args.checkpoint.is_file(), "resume requires --checkpoint from unfinished training")
        require(out.is_dir(), "Missing original DCC launch identity")
        original = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        require(original["variant"] == args.variant and original["code_identity"] == code_identity() and
                original["init_sha256"] == info["init_sha256"], "Resume launch source/initialization changed")
        saved = checkpoint(args.checkpoint)
        # Validate source precision before native loading can upcast old moments.
        info["resume_checkpoint_policy"] = require_checkpoint_policy(saved)
        require(0 <= saved.get("epoch", -1) < 199 and saved.get("optimizer") is not None and
                saved.get("ema") is not None and saved.get("scaler") is not None,
                "Checkpoint is completed/stripped or lacks native optimizer/scaler/epoch/EMA")
        require(args.checkpoint.resolve().parent == (run_dir / "weights").resolve(), "Checkpoint is from another run")
        for k, value in train_args.items():
            if k not in {"model", "resume"}:
                require(saved["train_args"].get(k) == value, "Resume recipe differs: " + k)
        verify_model(saved["ema"], args.variant, zero=False)
        selected = args.checkpoint
        train_args["resume"] = str(selected.resolve())
    state_path = out / "state.json"
    state = {"status": "RUNNING", "pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat(),
             "completed_epochs": saved["epoch"] + 1 if saved else 0, "final_eval": "NOT_STARTED", "mode": args.mode}
    write_json(state_path, state)
    training_variant = args.variant

    class RecordingTrainer(DCCCheckpointTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model, audit = build_training_model(cfg, weights, self.data, training_variant)
            write_json(out / ("resume_rebuild.json" if saved else "start_rebuild.json"), audit)
            return model

        def final_eval(self):
            state["final_eval"] = "RUNNING"
            write_json(state_path, state)
            try:
                result = super().final_eval()
                state["final_eval"] = "COMPLETED"
                return result
            except BaseException:
                state["final_eval"] = "FAILED"
                raise
            finally:
                write_json(state_path, state)

    def setup(trainer):
        verify_model(trainer.model, args.variant, zero=saved is None)
        require(trainer.amp and trainer.args.batch == 16 and trainer.args.imgsz == 640, "Native AMP/B16/640 changed")
        require(type(trainer.optimizer) is torch.optim.AdamW, "Optimizer changed")
        actual = vars(trainer.args)
        ignored = {"model", "resume"} if saved else set()
        changes = {k: [v, actual.get(k)] for k, v in train_args.items() if k not in ignored and actual.get(k) != v}
        require(not changes, "Actual native recipe changed: " + repr(changes))
        if saved:
            require(optimizer_param_names(trainer.model, trainer.optimizer) ==
                    saved["dcc_checkpoint"]["optimizer_param_names"],
                    "Native optimizer parameter-name/group-order mapping differs")
            require(trainer.start_epoch == saved["epoch"] + 1 and trainer.scaler.state_dict() == saved["scaler"] and
                    trainer.ema.updates == saved["updates"], "Native resume did not restore epoch/scaler/EMA updates")
            require(len(trainer.optimizer.state) == len(saved["optimizer"]["state"]), "Native optimizer restore mismatch")
            restored = trainer.optimizer.state_dict()
            require(restored["param_groups"] == saved["optimizer"]["param_groups"],
                    "Native optimizer group hyperparameters/order differ")
            require(set(restored["state"]) == set(saved["optimizer"]["state"]),
                    "Native optimizer state IDs differ")
            for key, parameters in saved["optimizer"]["state"].items():
                require(set(restored["state"][key]) == set(parameters), "Native optimizer state fields differ")
                for field, value in parameters.items():
                    observed = restored["state"][key][field]
                    equal = (observed.dtype == value.dtype and torch.equal(observed.detach().cpu(), value.detach().cpu())
                             if torch.is_tensor(value) else observed == value)
                    require(equal, "Native optimizer state restore mismatch: %s.%s" % (key, field))
            saved_ema = {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                         for k, v in saved["ema"].state_dict().items()}
            for name, model in (("model", trainer.model), ("EMA", trainer.ema.ema)):
                actual_state = model.state_dict()
                require(set(actual_state) == set(saved_ema) and
                        all(torch.equal(v.detach().cpu(), saved_ema[k]) for k, v in actual_state.items()),
                        "Native " + name + " weight/buffer restore mismatch")
        write_json(out / "setup.json", {"coverage": optimizer_coverage(trainer.model, trainer.optimizer),
                   "native_resume": bool(saved), "start_epoch": trainer.start_epoch, "runtime": runtime()})
        YAML.save(out / "actual_train_args.yaml", actual)

    def epoch_complete(trainer):
        # The native final_eval also emits on_fit_epoch_end; epoch itself is not advanced there.
        state["completed_epochs"] = int(trainer.epoch) + 1
        write_json(state_path, state)

    try:
        model = RTDETR(str(selected))
        model.add_callback("on_train_batch_start", disable_oom_retry)
        model.add_callback("on_train_start", setup)
        model.add_callback("on_fit_epoch_end", epoch_complete)
        model.train(trainer=RecordingTrainer, **train_args)
        state["status"] = "COMPLETED_200" if state["completed_epochs"] >= 200 else "EARLY_STOPPED"
    except KeyboardInterrupt:
        state["status"] = "INTERRUPTED"
        raise
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["finished"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "start", "resume", "status"))
    p.add_argument("--variant", choices=VARIANTS, default="cbr_lif_dcc_v1")
    p.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    p.add_argument("--data", type=Path)
    p.add_argument("--init", type=Path)
    p.add_argument("--parent-args", type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--checks", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, help="Optional plan JSON; exclusive creation")
    return p


def main():
    args = parser().parse_args()
    args.init = args.init or ROOT / "weights" / (args.variant + "_controlled_init.pt")
    if args.mode == "status":
        path = ROOT / "outputs/dcc" / args.variant / "training/state.json"
        print(path.read_text(encoding="utf-8") if path.is_file() else json.dumps({"status": "NOT_STARTED", "final_test": "NOT_RUN"}))
    elif args.mode == "plan":
        result = plan(args)
        if args.output:
            require(not args.output.exists(), "Existing plan protected")
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
