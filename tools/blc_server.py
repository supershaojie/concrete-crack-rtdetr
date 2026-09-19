"""BLC environment/init/preflight/plan/start/resume/val/test/pack entry points."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
import traceback

import torch
from blc_common import *
from blc_probe_io import temporary_probe, retained_size, compact_audit


def environment(strict=False):
    info = runtime()
    info.update(base=BASE, branch=BRANCH, root=str(ROOT), origin=subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True).strip())
    require(info["origin"].removesuffix(".git").rstrip("/") in
            ("https://github.com/supershaojie/concrete-crack-rtdetr", "git@github.com:supershaojie/concrete-crack-rtdetr",
             "ssh://git@github.com/supershaojie/concrete-crack-rtdetr"), "Wrong repository origin")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    require((ROOT/".git").is_file(), "BLC must use an independent linked worktree")
    if strict:
        require(os.environ.get("CONDA_DEFAULT_ENV") == "rtdetr", "Activate existing rtdetr Conda")
        require(platform.python_version_tuple()[:2] == ("3", "10") and info["torch"] == "2.1.2+cu121", "Server environment mismatch; do not upgrade")
        require(torch.cuda.is_available() and "4090" in info["gpu"], "RTX4090 server CUDA required for formal capacity")
    return info


def init_one(variant):
    p = paths(variant)
    require(p["source"].is_file() and sha256(p["source"]) == SOURCE_SHA256, "Specified public source missing or wrong SHA")
    if p["init"].exists():
        ckpt = torch_load(p["init"], map_location="cpu")
        require(ckpt.get("epoch") == -1 and ckpt.get("optimizer") is None and ckpt.get("ema") is None,
                "Existing file is not untrained controlled_init")
        require(ckpt.get("blc_provenance", {}).get("source_sha256") == SOURCE_SHA256, "Existing init source identity mismatch")
        verify_model(ckpt["model"], variant, zero=True)
        print("Existing controlled_init preserved; init-preflight will re-audit every value:", p["init"], flush=True)
        return
    report = initialize(p["source"], p["init"], variant)
    write_json(p["evidence"]/("init-"+stamp()+".json"), report)
    print(p["init"], report["output_sha256"], flush=True)


def init_preflight(variant):
    from blc_preflight import api_reconstruction
    p = paths(variant)
    folder = p["evidence"]/("init-preflight-"+stamp())
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", phase="init-preflight", variant=variant)
    try:
        report["context"] = evidence_context(variant)
        with temporary_probe(folder, "controlled-reference-audit", report) as tmp:
            temp = tmp/"reference.pt"
            report["current_source_audit"] = compact_audit(initialize(p["source"], temp, variant))
            expected = torch_load(temp, map_location="cpu")["model"].state_dict()
            saved = torch_load(p["init"], map_location="cpu")
            require(saved.get("epoch") == -1 and saved.get("optimizer") is None and saved.get("ema") is None,
                    "Existing file is not untrained controlled_init")
            actual = saved["model"].state_dict()
            require(set(actual) == set(expected) and all(torch.equal(v, actual[k]) for k, v in expected.items()),
                    "Existing controlled_init differs from current source/initialization logic")
            verify_model(saved["model"], variant, zero=True)
            report["structure"] = dict(status="PASSED", exact_saved_state=True, tensors=len(actual),
                                        topology_parameters_and_zero_initialization=True)
            report["actual_train_api"] = compact_audit(api_reconstruction(variant, tmp))
        report["math"] = dict(status="NOT_RUN", reason="Light source/state/structure/Trainer audit; standalone model math checks are separate")
        require(sha256(p["init"]) == report["context"]["init_sha256"], "Init audit changed controlled_init")
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["retained_bytes_before_report"] = retained_size(folder)
        write_json(folder/"report.json", report)
        print(folder/"report.json", report["status"], flush=True)
    return report


def require_evidence(variant, phase):
    current = evidence_context(variant)
    files = sorted(paths(variant)["evidence"].glob(phase+"-*/report.json"))
    require(files, f"PENDING: run {phase} for {variant}")
    report = json.loads(files[-1].read_text(encoding="utf-8"))
    require(report.get("status") == "PASSED", f"{files[-1]} is {report.get('status')}: inspect retained evidence")
    recorded = report.get("context", {})
    for key in ("variant", "code", "init_sha256", "source_sha256", "data", "recipe"):
        require(recorded.get(key) == current[key], f"Stale {phase} evidence: {key}")
    for key in ("python", "torch", "cuda", "gpu", "ultralytics"):
        require(recorded.get("runtime", {}).get(key) == current["runtime"][key], f"{phase} environment changed: {key}")
    if phase == "preflight":
        require(report["capacity"]["status"] == "PASSED" and report["capacity"]["batch"] == 16
                and report["capacity"]["imgsz"] == 640 and report["capacity"]["AMP"] is True
                and report["capacity"]["effective_updates"] >= 2, "Missing capacity evidence")
        require(report["lifecycle"]["status"] == report["native_half_ema_epoch_val"]["status"] == report["native_resume"]["status"] == "PASSED",
                "Missing lifecycle/half EMA/native resume evidence")
    return str(files[-1])


def preflight(variant):
    from blc_preflight import capacity
    environment(strict=True)
    # Both configurations must have current source/structure/Trainer evidence.
    for other in VARIANTS:
        require_evidence(other, "init-preflight")
    p = paths(variant)
    folder = p["evidence"]/("preflight-"+stamp())
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", phase="preflight", variant=variant, stage="native_amp_resources",
                  capacity=dict(status="NOT_RUN"), lifecycle=dict(status="NOT_RUN"),
                  native_half_ema_epoch_val=dict(status="NOT_RUN"), native_resume=dict(status="NOT_RUN"))
    created_resources = []
    retained_before = retained_size(p["evidence"])
    try:
        report["context"] = evidence_context(variant)
        from ultralytics.utils import ASSETS
        import shutil
        resources = []
        report["native_amp_resources"] = resources
        for dest, candidates in (
            (ROOT/"yolo26n.pt", [MAIN/"yolo26n.pt", MAIN/"weights/yolo26n.pt"]),
            (ASSETS/"bus.jpg", [MAIN/"ultralytics-main/ultralytics/assets/bus.jpg", MAIN/"bus.jpg"]),
        ):
            if not dest.is_file():
                source = next((q for q in candidates if q.is_file()), None)
                require(source is not None, "Native AMP check resource missing (no automatic download): "+str(dest))
                dest.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as src, dest.open("xb") as dst:
                    created_resources.append(dest)
                    shutil.copyfileobj(src, dst)
            resources.append(dict(path=str(dest), sha256=sha256(dest), bytes=dest.stat().st_size,
                                  temporary=dest in created_resources, purpose="unchanged native AMP check"))
        capacity(variant, folder, report)
        report["status"] = "PASSED" if all(report[key]["status"] == "PASSED" for key in
            ("capacity", "lifecycle", "native_half_ema_epoch_val", "native_resume")) else "PENDING"
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_errors = []
        for dest in created_resources:
            try:
                if not any(row["path"] == str(dest) for row in report.get("native_amp_resources", [])) and dest.is_file():
                    report.setdefault("native_amp_resources", []).append(dict(path=str(dest), sha256=sha256(dest),
                        bytes=dest.stat().st_size, temporary=True, purpose="partial native AMP resource copy"))
                dest.unlink()
            except OSError as error:
                cleanup_errors.append(dict(path=str(dest), error=repr(error)))
        if cleanup_errors:
            report.update(status="FAILED", resource_cleanup_errors=cleanup_errors)
        for resource in report.get("native_amp_resources", []):
            resource["cleaned"] = resource["temporary"] and not Path(resource["path"]).exists()
        report["new_retained_bytes_before_report"] = retained_size(p["evidence"]) - retained_before
        report["retention_target_bytes"] = 10 * 1024 * 1024
        write_json(folder/"report.json", report)
        print(folder/"report.json", report["status"], flush=True)
        print("New retained bytes (including report; log may still grow):", retained_size(p["evidence"]) - retained_before, flush=True)
    if report["status"] != "PASSED":
        raise SystemExit("PENDING: capacity finished but lifecycle precision evidence requires review; start remains blocked")


@contextmanager
def experiment_lock(variant):
    import fcntl
    p = paths(variant)
    lock = p["run"].with_suffix(".blc.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def start_or_resume(variant, resume=False, ablation_after_review=False):
    from blc_preflight import avoid_oom_resize
    environment(strict=True)
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(),
            "Tracked implementation changed; commit/revalidate before training")
    require_evidence(variant, "init-preflight")
    # The single-module ablation is explicitly held until main benefit is reviewed.
    if variant == "blc_v1":
        require(ablation_after_review, "Single BLC ablation requires explicit --ablation-after-review after reviewing main-val benefit")
        vals = sorted(paths("cbr_lif_blc_v1")["evidence"].glob("val-*/metrics.json"))
        require(vals and json.loads(vals[-1].read_text(encoding="utf-8"))["status"] == "PASSED", "Completed main-val evidence missing")
    require_evidence(variant, "preflight")
    p = paths(variant)
    with experiment_lock(variant):
        if resume:
            checkpoint_path = p["run"]/"weights/last.pt"
            require(checkpoint_path.is_file(), "No formal BLC last.pt to resume")
            checkpoint = torch_load(checkpoint_path, map_location="cpu")
            require(0 <= checkpoint.get("epoch", -1) < 199 and checkpoint.get("optimizer") is not None
                    and checkpoint.get("ema") is not None and checkpoint.get("scaler") is not None,
                    "Checkpoint finished/deploy-only/untrained; do not restart completed training")
            train_args = checkpoint["train_args"]
            expected, _ = recipe(variant)
            for key, value in expected.items():
                if key not in {"model", "resume"}:
                    require(train_args.get(key) == value, "Resume recipe mismatch: "+key)
            verify_model(checkpoint["ema"].float(), variant)
            wrapper = RTDETR(str(checkpoint_path))
            kwargs = dict(resume=True)
        else:
            require(not p["run"].exists(), f"Existing formal run preserved: {p['run']}; inspect state and use resume for an unfinished checkpoint")
            wrapper = RTDETR(str(p["init"]))
            kwargs, _ = recipe(variant)
        wrapper.add_callback("on_train_batch_start", avoid_oom_resize)
        # Native epoch validation and best selection are unchanged. No final test.
        launch = dict(status="RUNNING", action="resume" if resume else "start", context=evidence_context(variant))
        record = p["evidence"]/("training-"+stamp()+".json")
        try:
            wrapper.train(trainer=BLCTrainer, **kwargs)
            launch["status"] = "COMPLETED"
        except BaseException as error:
            launch.update(status="FAILED_OR_INTERRUPTED", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            if (p["run"]/"weights/best.pt").is_file():
                launch["best_sha256"] = sha256(p["run"]/"weights/best.pt")
            write_json(record, launch)


def pack(variant):
    p = paths(variant)
    output = ROOT/"outputs/blc_v1"/(variant+"-light-"+stamp()+".tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = set()
    files.update(ROOT/name for name in code_identity()["files"])
    for folder in (ROOT/"docs/blc_v1", p["evidence"]):
        if folder.exists():
            files.update(q for q in folder.rglob("*") if q.is_file())
    for pattern in ("*blc*.py", "*blc*.sh"):
        files.update((ROOT/"tools").glob(pattern))
    files.update([ROOT/"ultralytics-main/ultralytics/nn/modules/blc.py", ROOT/"ultralytics-main/ultralytics/nn/modules/__init__.py",
                  ROOT/"ultralytics-main/ultralytics/nn/tasks.py"])
    files.update(MODEL_DIR/name for pair in VARIANTS.values() for name in pair)
    for name in ("args.yaml", "results.csv", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
        if (p["run"]/name).is_file():
            files.add(p["run"]/name)
    rows, omitted, payload = [], [], {}
    for path in sorted(files):
        if path.suffix.lower() not in {".py", ".sh", ".yaml", ".json", ".md", ".csv", ".png", ".jpg", ".log", ".txt"}:
            omitted.append(str(path)); continue
        if path.stat().st_size > 4*1024*1024:
            omitted.append(str(path)); continue
        if path.is_relative_to(ROOT):
            name = path.relative_to(ROOT).as_posix()
        else:
            name = "training/"+path.name
        raw = path.read_bytes()
        if path.suffix == ".log":
            raw = b"\n".join(raw.splitlines()[-200:])
        payload[name] = raw
        rows.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    metrics = {split: sorted(p["evidence"].glob(split+"-*/metrics.json")) for split in ("val", "test")}
    meta = dict(variant=variant, git=environment(), manifest=rows, omitted=omitted,
                missing=[name for name in ("args.yaml", "results.csv") if not (p["run"]/name).exists()],
                evaluation={split: "NOT_RUN" if not paths else str(paths[-1]) for split, paths in metrics.items()},
                excludes="weights, datasets, reference ZIP, large tensors; no training/evaluation is run by pack")
    payload["MANIFEST.json"] = json.dumps(meta, ensure_ascii=False, indent=2).encode()
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, raw in payload.items():
            item = tarfile.TarInfo(name); item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    with tarfile.open(output, "r:gz") as archive:
        require(set(archive.getnames()) == set(payload), "Archive file inventory differs")
        for name, raw in payload.items():
            require(archive.extractfile(name).read() == raw, "Archive readback differs")
    require(output.stat().st_size < 20*1024*1024, "Light package exceeds 20MiB; preserved for inspection")
    result = dict(archive=str(output.resolve()), bytes=output.stat().st_size, sha256=sha256(output), missing=meta["missing"], omitted=omitted)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["environment", "init", "init-preflight", "preflight", "plan", "start", "resume", "val", "test", "pack"])
    parser.add_argument("--variant", choices=VARIANTS, default=os.environ.get("BLC_VARIANT", "cbr_lif_blc_v1"))
    parser.add_argument("--both", action="store_true", help="init/init-preflight only; never starts ablation")
    parser.add_argument("--ablation-after-review", action="store_true", help="Explicit later single-module start, after main-val benefit review")
    args = parser.parse_args()
    torch.set_num_threads(4)
    require(not args.both or args.action in ("init", "init-preflight"), "--both applies only to initialization")
    for variant in VARIANTS if args.both else [args.variant]:
        if args.action == "environment":
            print(json.dumps(environment(), indent=2))
        elif args.action == "init":
            init_one(variant)
        elif args.action == "init-preflight":
            init_preflight(variant)
        elif args.action == "preflight":
            preflight(variant)
        elif args.action == "plan":
            config, diff = recipe(variant)
            print(json.dumps(dict(variant=variant, recipe=config, parent_diff=diff,
                                  formal_training="NOT_STARTED" if not paths(variant)["run"].exists() else "EXISTING_RUN_INSPECT",
                                  test="NOT_RUN", required=["init --both", "init-preflight --both", "preflight"]), indent=2))
        elif args.action in ("start", "resume"):
            start_or_resume(variant, args.action == "resume", args.ablation_after_review)
        elif args.action in ("val", "test"):
            from eval_blc import evaluate
            evaluate(variant, args.action)
        elif args.action == "pack":
            pack(variant)


if __name__ == "__main__":
    main()
