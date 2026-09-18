"""GRA plan/start/resume/status. Formal training is an explicit separate command."""
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
from init_gra import (ROOT, MODEL_DIR, VARIANTS, BASE_COMMIT, SOURCE_SHA256, require,
                      sha256, runtime, write_json, verify_model, build_training_model)
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML, ASSETS
from ultralytics.utils.patches import torch_load
from ultralytics.data.utils import check_det_dataset
from c19_lif_v1_data import dataset_inventory

MAIN = Path(os.environ.get("GRA_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
BRANCH = "exp-rtdetr-r18-lite-gra-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_manifest():
    """Content identity, independent of commit metadata and local newline convention."""
    files = set((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files.update((ROOT / "tools").glob("*.py"))
    files.update(ROOT / "tools" / n for n in ("init_c19_lif_v1.py", "init_lif_down.py", "lif_down_topology.py", "c19_lif_v1_data.py",
                                             "c19_lif_v1_probe.py", "c19_lif_v1_diagnostic.py"))
    files.update(MODEL_DIR / v[i] for v in VARIANTS.values() for i in (0, 1))
    files.update(MODEL_DIR / n for n in ("rtdetr-resnet18-lite-lif-down.yaml", "rtdetr-resnet18-lite-cbr.yaml"))
    files.update((ROOT / "tools").glob("*gra*.sh"))
    files.add(ROOT / "docs/gra/parent_args.yaml")
    if (ROOT / "docs/gra/parent_data_identity.json").is_file():
        files.add(ROOT / "docs/gra/parent_data_identity.json")
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(files)}


def recipe(variant, initialized, data=None):
    original = YAML.load(ROOT / "docs/gra/parent_args.yaml")
    target = dict(original)
    target.update(model=str(Path(initialized).resolve()), data=str(Path(data or MAIN / "configs/crack_autodl.yaml").resolve()),
                  project=str(MAIN / "runs/c_series"), name=VARIANTS[variant][2],
                  save_dir=str(MAIN / "runs/c_series" / VARIANTS[variant][2]))
    rows = [{"field": k, "parent": original[k], "candidate": target[k], "changed": original[k] != target[k],
             "reason": "model/output identity or verified equivalent dataset location" if original[k] != target[k] else "unchanged"}
            for k in sorted(original)]
    require({r["field"] for r in rows if r["changed"]} <= {"model", "name", "save_dir", "data", "project"}, "Recipe changed")
    return target, rows


def inventory(data):
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "Expected nc=1")
    # The successful dataset uses canonical images/{train,val,test} and matching labels.
    root = Path(resolved["path"]).resolve()
    for split in ("train", "val", "test"):
        require(Path(resolved[split]).resolve() == root / "images" / split,
                "Split must match original canonical path: " + split)
    result = dataset_inventory(root)
    require([result[s]["images"] for s in ("train", "val", "test")] == [6048, 1728, 864], "Original split counts changed")
    require(result == read_json(ROOT / "docs/gra/parent_data_identity.json"),
            "Dataset paths/label identities differ from original successful parent")
    return result


def optimizer_coverage(model, optimizer):
    occurrences = Counter(id(p) for group in optimizer.param_groups for p in group["params"])
    rows = {n: occurrences[id(p)] for n, p in model.named_parameters() if p.requires_grad}
    require(all(v == 1 for v in rows.values()), "Trainable parameters missing/duplicated in optimizer")
    require(set(occurrences) == {id(p) for p in model.parameters() if p.requires_grad}, "Unexpected/frozen optimizer parameter")
    require(type(optimizer) is torch.optim.AdamW, "Optimizer must be AdamW")
    return {"status": "PASSED", "trainable_tensors": len(rows), "occurrences": rows,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def ensure_amp_resources():
    """Reuse existing assets; no downloads, installation, or GPU-idle policy."""
    import shutil
    rows = []
    for target, candidates in ((ASSETS / "bus.jpg", [MAIN / "ultralytics-main/ultralytics/assets/bus.jpg", MAIN / "bus.jpg"]),
                               (ROOT / "yolo26n.pt", [MAIN / "yolo26n.pt", MAIN / "weights/yolo26n.pt"])):
        if not target.is_file():
            source = next((p for p in candidates if p.is_file()), None)
            require(source is not None, "PENDING: existing AMP resource missing: " + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        rows.append({"path": str(target), "sha256": sha256(target)})
    return rows


def verify_initialization(variant, initialized, audit, source):
    require(Path(source).is_file() and sha256(source) == SOURCE_SHA256, "Wrong/missing public source")
    evidence = read_json(audit)
    require(evidence.get("status") == "PASSED" and evidence.get("variant") == variant, "Initialization audit missing/wrong variant")
    require(evidence.get("source_sha256") == SOURCE_SHA256 and evidence.get("output_sha256") == sha256(initialized), "Initialization identity changed")
    ckpt = torch_load(initialized, map_location="cpu")
    require(ckpt.get("epoch") == -1 and not ckpt.get("optimizer") and not ckpt.get("ema") and not ckpt.get("scaler"), "Start requires untrained controlled init")
    verify_model(ckpt["model"], variant, zero=True)
    return evidence


def verify_delivery(expected_sha):
    require(len(expected_sha) == 40 and all(c in "0123456789abcdef" for c in expected_sha), "Full lowercase delivery SHA required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head == expected_sha, "Worktree HEAD differs from delivery SHA")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, head], cwd=ROOT).returncode == 0, "Wrong parent ancestry")
    require((ROOT / ".git").is_file(), "Independent linked worktree required")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    require(not dirty, "Tracked runtime source has uncommitted changes")
    return head


def gate(args):
    verify_delivery(args.expected_sha)
    verify_initialization(args.variant, args.initialized, args.audit, args.source)
    capacity, checks = read_json(args.preflight), read_json(args.checks)
    manifest = source_manifest()
    require(capacity.get("status") == "PASSED" and capacity.get("variant") == args.variant, "Server preflight missing/failed")
    require(capacity.get("batch") == 16 and capacity.get("imgsz") == 640 and capacity.get("amp") is True,
            "Required B16/640 native AMP preflight missing")
    require(capacity.get("effective_updates", 0) >= 2 and 1 <= len(capacity.get("batches", [])) <= 16, "Insufficient bounded effective updates")
    require(capacity.get("initialized_sha256") == sha256(args.initialized) and capacity.get("source_sha256") == SOURCE_SHA256, "Preflight init identity changed")
    require(capacity.get("source_manifest") == manifest, "Runtime source differs from server preflight")
    require(capacity.get("data_sha256") == sha256(args.data) and capacity.get("dataset_inventory") == inventory(args.data), "Dataset changed since preflight")
    require(checks.get("status") in {"PASSED", "LOCAL_CHECKS_PASSED", "PASSED_LOCAL_CHECKS_SERVER_PENDING"}, "Mathematics/wiring/learned-state checks missing/failed")
    require(checks.get("source_manifest") == manifest, "Runtime source differs from structural checks; rerun check_gra")
    require(checks.get("source_sha256") == SOURCE_SHA256, "Structural checks public source identity differs")
    for boundary in ("cpu_fp32", "cuda_fp32", "cuda_amp"):
        require(checks.get("variants", {}).get(args.variant, {}).get(boundary, {}).get("status") == "PASSED",
                "Required real-data learned-state boundary missing: " + boundary)
    require(checks.get("module_cuda", {}).get("status") == "PASSED", "CUDA module checks missing/failed")
    require(checks.get("data") == capacity["dataset_inventory"], "Structural checks used a different dataset")
    current = runtime()
    for evidence in (capacity, checks):
        require(all(evidence.get("runtime", {}).get(k) == current.get(k) for k in ("torch", "cuda", "gpu", "executable", "ultralytics")),
                "Checks/preflight must run in the actual server worktree/environment")
    require(torch.cuda.is_available(), "CUDA required; no environment upgrades performed")
    return capacity, checks


class RecordingTrainer(RTDETRTrainer):
    """Real native Trainer reconstruction; learned state is never reset."""
    gra_variant = "cbr_lif_gra_v1"
    gra_metadata = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model, audit = build_training_model(cfg, weights, self.data, self.gra_variant)
        if self.gra_metadata:
            write_json(self.gra_metadata / "trainer_rebuild.json", audit)
        return model

    def final_eval(self):
        if self.gra_metadata:
            write_json(self.gra_metadata / "final_eval.json", {"status": "RUNNING", "split": "val"})
        try:
            result = super().final_eval()
        except BaseException as error:
            if self.gra_metadata:
                write_json(self.gra_metadata / "final_eval.json", {"status": "FAILED", "error": repr(error), "split": "val"})
            raise
        if self.gra_metadata:
            write_json(self.gra_metadata / "final_eval.json", {"status": "COMPLETED", "split": "val"})
        return result


def run_training(args, resume=False):
    capacity, _ = gate(args)
    train_args, differences = recipe(args.variant, args.initialized, args.data)
    run = Path(train_args["save_dir"])
    metadata = ROOT / "outputs/gra" / args.variant / "training"
    metadata.mkdir(parents=True, exist_ok=True)
    if resume:
        checkpoint = Path(args.checkpoint).resolve()
        require(checkpoint == (run / "weights/last.pt").resolve(), "Native resume only from this experiment's last.pt")
        ckpt = torch_load(checkpoint, map_location="cpu")
        require(0 <= ckpt.get("epoch", -1) < 199 and ckpt.get("optimizer") and ckpt.get("ema") and ckpt.get("scaler"),
                "Resume requires real unfinished checkpoint with optimizer/scaler/EMA; completed/stripped checkpoint cannot resume")
        require((metadata / "launch_identity.json").is_file(), "Missing original launch identity")
        previous = read_json(metadata / "launch_identity.json")
        require(previous["variant"] == args.variant and previous["source_manifest"] == source_manifest() and
                previous["initialized_sha256"] == sha256(args.initialized) and previous["dataset_inventory"] == capacity["dataset_inventory"],
                "Resume source/init/dataset changed")
        for key, value in train_args.items():
            if key not in {"model", "resume"}:
                require(ckpt["train_args"].get(key) == value, "Resume recipe changed: " + key)
        train_args.update(model=str(checkpoint), resume=str(checkpoint))
    else:
        require(not run.exists(), "Existing run protected; use explicit resume for unfinished training")
        require(not (metadata / "launch_identity.json").exists(), "Existing launch metadata protected")
    # An atomic lock only covers this run. Other experiments/GPU processes remain allowed.
    lock = run.with_name(run.name + ".gra-active.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir(exist_ok=False)
    write_json(lock / "owner.json", {"pid": os.getpid(), "worktree": str(ROOT), "variant": args.variant})
    state = {"status": "RUNNING", "completed_epochs": 0, "final_eval": "NOT_RUN", "mode": "resume" if resume else "start",
             "pid": os.getpid(), "runtime": runtime(), "run": str(run), "started": datetime.now(timezone.utc).isoformat()}
    trainer = None
    try:
        ensure_amp_resources()
        if not resume:
            write_json(metadata / "launch_identity.json", {"variant": args.variant, "source_manifest": source_manifest(),
                       "initialized_sha256": sha256(args.initialized), "source_sha256": SOURCE_SHA256,
                       "dataset_inventory": capacity["dataset_inventory"], "data_sha256": sha256(args.data), "runtime": runtime(),
                       "preflight_sha256": sha256(args.preflight), "checks_sha256": sha256(args.checks)})
        YAML.save(metadata / "train_args.yaml", train_args)
        write_json(metadata / "recipe_diff.json", differences)
        write_json(metadata / "training_state.json", state)
        class Trainer(RecordingTrainer):
            gra_variant = args.variant
            gra_metadata = metadata
        trainer = Trainer(overrides=train_args)

        def started(t):
            verify_model(t.model, args.variant, zero=not resume)
            require(t.amp and t.batch_size == 16 and t.args.imgsz == 640, "Formal AMP/B16/640 changed")
            actual = vars(t.args)
            mismatch = {k: [v, actual.get(k)] for k, v in train_args.items() if actual.get(k) != v}
            require(not mismatch, "Effective recipe differs: " + repr(mismatch))
            write_json(metadata / "optimizer_coverage.json", optimizer_coverage(t.model, t.optimizer))
            YAML.save(metadata / "actual_args.yaml", actual)
            if resume:
                write_json(metadata / "resume_state.json", {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
                           "start_epoch": t.start_epoch, "optimizer_state_entries": len(t.optimizer.state),
                           "scaler": t.scaler.state_dict(), "ema_updates": t.ema.updates})

        def epoch_done(t):
            state["completed_epochs"] = t.epoch + 1
            write_json(metadata / "training_state.json", state)

        trainer.add_callback("on_train_start", started)
        trainer.add_callback("on_train_batch_start", lambda t: setattr(t, "_oom_retries", 3))
        trainer.add_callback("on_fit_epoch_end", epoch_done)
        trainer.train()
        state["status"] = "COMPLETED_200" if state["completed_epochs"] == 200 else "EARLY_STOPPED"
        state["exit_code"] = 0
    except KeyboardInterrupt:
        state.update(status="INTERRUPTED", exit_code=130)
        raise
    except BaseException as error:
        state.update(status="FAILED", error=repr(error), exit_code=1)
        raise
    finally:
        if (metadata / "final_eval.json").is_file():
            state["final_eval"] = read_json(metadata / "final_eval.json")["status"]
        state["finished"] = datetime.now(timezone.utc).isoformat()
        write_json(metadata / "training_state.json", state)
        # Only remove our small active lock after a handled exit; stale lock stays reviewable after SIGKILL.
        (lock / "owner.json").unlink()
        lock.rmdir()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("plan", "start", "resume", "status"):
        q = sub.add_parser(command)
        q.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_gra_v1")
        if command == "status":
            continue
        q.add_argument("--initialized", type=Path, required=True)
        q.add_argument("--data", type=Path, default=MAIN / "configs/crack_autodl.yaml")
        if command == "plan":
            q.add_argument("--output", type=Path)
        else:
            for flag in ("audit", "preflight", "checks", "source"):
                q.add_argument("--" + flag, type=Path, required=True)
            q.add_argument("--expected-sha", required=True)
        if command == "resume":
            q.add_argument("--checkpoint", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.command == "plan":
        values, differences = recipe(args.variant, args.initialized, args.data)
        report = {"status": "PLAN_ONLY", "formal_training": "NOT_STARTED", "args": values, "differences": differences}
        if args.output:
            require(not args.output.exists(), "Existing plan protected")
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "status":
        state = ROOT / "outputs/gra" / args.variant / "training/training_state.json"
        print(state.read_text(encoding="utf-8") if state.is_file() else '{"status":"NOT_STARTED","final_eval":"NOT_RUN"}')
    else:
        run_training(args, resume=args.command == "resume")


if __name__ == "__main__":
    main()
