"""Idempotent evaluation, offline analysis and one complete archive. No training."""
from __future__ import annotations
import gzip
import io
import json
from pathlib import Path
import tarfile
import time
import traceback
import uuid
import subprocess

from dtr_v1_common import (ROOT, OUT, RUN, FORMULA, COUNTS, code_identity, data_config, snapshot,
                          read_json, write_json, require, sha256, digest, file_info, now, git)
from ultralytics.models.rtdetr.dtr_val import DTRValidator, EVAL, POLICY


def eval_identity(training):
    current = code_identity(clean=True)
    return dict(split_protocol=POLICY, settings=EVAL, checkpoint_sha256=sha256(RUN / "weights/best.pt"),
                data=training["binding"]["data"], eval_source_lf_sha256=current["source_lf_sha256"],
                formula=FORMULA, run_id=training["id"])


def verified_result(report, identity, split):
    if not report or report.get("status") != "PASS" or report.get("identity") != identity or report.get("split") != split:
        return False
    require(report["images"] == COUNTS[split][0] and report["gt_count"] == COUNTS[split][1], "Saved evaluation coverage mismatch")
    require(report["query_count"] == 300 * report["images"], "Incomplete ordinary-query export")
    for item in report["artifacts"].values():
        require(file_info(item["path"]) == item, f"Saved evaluation artifact changed/missing: {item['path']}")
    source = report.get("eval_source_snapshot")
    if source:
        require(file_info(source["path"]) == source, "Evaluation source snapshot changed/missing")
    return True


def evaluate(split):
    import numpy as np
    from ultralytics.utils.torch_utils import init_seeds
    from ultralytics.utils.patches import torch_load
    from dtr_v1 import active_workers, has_tmux
    require(split in ("val", "test"), "Only formal val/test")
    require(not active_workers() and not has_tmux(), "Wait for DTR worker/session to exit")
    require(read_json(OUT / "training_completed.json", {}).get("status") == "TRAINING_COMPLETED", "No native training completion evidence")
    training = read_json(OUT / "training_identity.json")
    require(training, "Missing run identity")
    data_path = data_config()
    require(snapshot(data_path) == training["binding"]["data"], "Training/evaluation data identity differs")
    best = RUN / "weights/best.pt"
    ckpt = torch_load(best, map_location="cpu")
    model = ckpt.get("ema") or ckpt.get("model")
    require(getattr(model, "dtr_identity", None) == training["id"], "Best belongs to a different experiment")
    best_epoch = ckpt["epoch"]
    del ckpt, model
    identity = eval_identity(training)
    if split == "test":
        require(verified_result(read_json(OUT / "val_lock.json"), identity, "val"), "Test requires complete same-best formal val")
    lock_path = OUT / f"{split}_lock.json"
    previous = read_json(lock_path)
    if verified_result(previous, identity, split):
        return dict(previous, reused=True)
    require(previous is None, "Existing successful evaluation identity differs; preserve it and resolve identity explicitly")
    # A completed report with missing lock is recoverable without another forward.
    for path in sorted((OUT / "evaluations").glob(f"*_{split}/metrics.json")):
        report = read_json(path)
        if verified_result(report, identity, split):
            write_json(lock_path, report)
            return dict(report, reused=True, repaired_lock=True)
    folder = OUT / "evaluations" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_" + split)
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAIL", identity=identity, split=split, started=now(), eval_commit=git("rev-parse", "HEAD"),
                  best_epoch_zero_based=best_epoch, best_rule="native val mAP50-95 fitness; native tie save rule",
                  report=str(folder / "metrics.json"), scope="full_split")
    source = OUT / "source_snapshot.tar.gz"
    if identity["eval_source_lf_sha256"] != training["binding"]["code"]["source_lf_sha256"]:
        source = OUT / "evaluation_sources" / (report["eval_commit"] + ".tar.gz")
        if not source.is_file():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(source), report["eval_commit"]],
                           cwd=ROOT, check=True, timeout=60)
    require(source.is_file(), "Missing actual evaluation source snapshot")
    report["eval_source_snapshot"] = file_info(source)
    settings = dict(EVAL, model=str(best), data=str(data_path), split=split, device="0", plots=True,
                    save_json=False, save_txt=False, project=str(folder), name="plots", exist_ok=False)
    validator = DTRValidator(args=settings, save_dir=folder / "plots")
    validator.export_path = folder / f"{split}_predictions_gt.jsonl.gz"
    validator.export_identity = dict(identity_sha256=digest(identity), split=split,
                                    checkpoint_sha256=identity["checkpoint_sha256"], run_id=training["id"])
    try:
        init_seeds(42, deterministic=True)
        metrics = validator(model=str(best))
        box = validator.metrics.box
        ap = np.asarray(box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Expected one-class finite ten-IoU AP")
        require(sha256(best) == identity["checkpoint_sha256"], "Best changed during evaluation")
        require(all(type(validator.actual_settings[k]) is type(v) and validator.actual_settings[k] == v for k, v in EVAL.items()),
                "Effective formal FP32 protocol differs")
        from ultralytics.utils.metrics import smooth
        threshold_index = int(smooth(box.f1_curve.mean(0), 0.1).argmax())
        curves = dict(PR=dict(recall=box.px.tolist(), precision=box.prec_values.tolist()),
                      confidence=box.px.tolist(), P=box.p_curve.tolist(), R=box.r_curve.tolist(), F1=box.f1_curve.tolist())
        write_json(folder / "curves.json", curves)
        report.update(status="PASS", metrics=metrics, Precision=float(box.p[0]), Recall=float(box.r[0]),
            F1=float(box.f1[0]), AP50=float(ap[0, 0]), AP75=float(ap[0, 5]), mAP50_95=float(ap.mean()),
            AP_by_IoU=ap[0].tolist(), IoU_thresholds=[round(.5 + .05 * i, 2) for i in range(10)],
            split_best_F1_confidence=float(box.px[threshold_index]),
            operating_point="native smoothed maximum F1 at IoU=0.50; one class (macro=micro); interpolated P/R/F1",
            precision_definition="TP/(TP+FP), not TN-based Accuracy",
            conf_filter="Native metrics use scores > 0.001. Export retains ALL 300 ordinary queries, including lower scores.",
            images=len(validator.seen_ids), query_count=validator.query_count, gt_count=validator.gt_count,
            actual_settings=validator.actual_settings,
            artifacts={k: file_info(p) for k, p in dict(predictions=validator.export_path,
                       curves=folder / "curves.json", statistics=folder / "native_statistics.npz").items()})
        require(verified_result(report, identity, split), "Evaluation incomplete")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["ended"] = now()
        write_json(folder / "metrics.json", report)
    write_json(lock_path, report)  # only after all exports/reports are closed and verified
    return report


def counts_at_threshold(export, threshold):
    """Offline, no image reads: native IoU matching on the requested score subset."""
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    matcher = BaseValidator.__new__(BaseValidator)
    matcher.iouv = torch.tensor([0.5])
    tp = fp = fn = images = 0
    with gzip.open(export, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            scores = torch.tensor(row["scores"])
            keep = scores > threshold
            boxes = torch.tensor(row["boxes"]).reshape(-1, 4)[keep]
            classes = torch.tensor(row["classes"])[keep]
            order = scores[keep].argsort(descending=True)
            boxes, classes = boxes[order], classes[order]
            gt = torch.tensor(row["gt_boxes"]).reshape(-1, 4)
            gt_classes = torch.tensor(row["gt_classes"])
            hit = int(matcher.match_predictions(classes, gt_classes, box_iou(gt, boxes)).sum()) if len(gt) and len(boxes) else 0
            tp += hit
            fp += len(boxes) - hit
            fn += len(gt) - hit
            images += 1
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(images=images, threshold=threshold, comparison="score > threshold", IoU=.5, TP=tp, FP=fp, FN=fn,
                Precision=precision, Recall=recall, F1=2 * tp / max(2 * tp + fp + fn, 1),
                averaging="nc=1; exact counts with native matching after threshold; no TN/Accuracy")


def offline_summary():
    val, test = read_json(OUT / "val_lock.json"), read_json(OUT / "test_lock.json")
    require(val and test and val["identity"] == test["identity"], "Both complete same-best locks required")
    for split, lock in (("val", val), ("test", test)):
        require(verified_result(lock, val["identity"], split), "Invalid evaluation lock")
    threshold = val["split_best_F1_confidence"]  # test never chooses the shared operating point
    summary = dict(threshold_source="formal val maximum F1 only; test never tunes threshold",
                   identity=val["identity"], shared_confidence=threshold, splits={})
    for split, lock in (("val", val), ("test", test)):
        summary["splits"][split] = dict(
            native_split_best={k: lock[k] for k in ("Precision", "Recall", "F1", "split_best_F1_confidence", "AP50", "AP75", "mAP50_95", "AP_by_IoU")},
            shared_val_threshold=counts_at_threshold(lock["artifacts"]["predictions"]["path"], threshold),
            native_conf_filter_counts=counts_at_threshold(lock["artifacts"]["predictions"]["path"], .001))
    write_json(OUT / "analysis.json", summary)
    return summary


def package():
    """Only package existing evidence. Never evaluate, scan data, or select weights."""
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    required = ("prepare.json", "initialization.json", "data_snapshot.json", "data_manifest.jsonl.gz",
                "recipe_diff.json", "train_args.yaml", "data_config.yaml", "training_identity.json", "source_snapshot.tar.gz",
                "actual_setup.json", "preflight.json", "training_completed.json", "mechanism.jsonl",
                "val_lock.json", "test_lock.json", "analysis.json")
    for name in required:
        if not (OUT / name).is_file():
            missing.append(name)
    pre = read_json(OUT / "preflight.json", {})
    if pre.get("status") != "PASS":
        missing.append("successful formal preflight")
    training = read_json(OUT / "training_identity.json", {})
    if training and pre.get("binding") != training.get("binding"):
        missing.append("preflight/training identity agreement")
    if read_json(OUT / "training_completed.json", {}).get("status") != "TRAINING_COMPLETED":
        missing.append("native training completion")
    dispatches = sorted((OUT / "dispatches").glob("*/dispatch.json"))
    if not dispatches:
        missing.append("dispatch/PID/log/real Python exit evidence")
    else:
        latest = dispatches[-1].parent
        for name in ("worker.json", "exit.json", "console.log", "python_exit_code.txt", "tee_exit_code.txt"):
            if not (latest / name).is_file():
                missing.append("latest dispatch/" + name)
    source = training.get("source_snapshot")
    if source and file_info(source["path"]) != source:
        missing.append("training source snapshot integrity")
    locks = {s: read_json(OUT / f"{s}_lock.json") for s in ("val", "test")}
    for s, lock in locks.items():
        if lock:
            try:
                require(verified_result(lock, lock["identity"], s), "incomplete")
            except Exception as error:
                missing.append(f"{s} artifact integrity: {error}")
    if locks["val"] and locks["test"] and locks["val"]["identity"] != locks["test"]["identity"]:
        missing.append("same-best val/test identity")
    for name in ("results.csv", "weights/best.pt", "weights/last.pt"):
        if not (RUN / name).is_file():
            missing.append(name)
    checkpoints = [file_info(RUN / "weights" / name) for name in ("best.pt", "last.pt")]
    best_lock = locks["val"]
    if best_lock and checkpoints[0]["sha256"] != best_lock.get("identity", {}).get("checkpoint_sha256"):
        missing.append("best checkpoint no longer matches evaluation")
    state = "INCOMPLETE" if missing else "COMPLETE"
    folder = OUT / "packages"
    folder.mkdir(exist_ok=True)
    path = folder / f"DTR_v1_{state}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.tar.gz"
    manifest = []
    with tarfile.open(path, "w:gz") as archive:
        def add_bytes(name, data):
            import hashlib
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), int(time.time())
            archive.addfile(info, io.BytesIO(data))
            manifest.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        def add_file(name, p):
            archive.add(p, arcname=name, recursive=False)
            manifest.append(dict(path=name, bytes=p.stat().st_size, sha256=sha256(p)))
        summary = dict(completeness=state, missing=missing, created=now(), formula=FORMULA,
                       weights_included=False, checkpoints=checkpoints,
                       best_epoch_zero_based=best_lock.get("best_epoch_zero_based") if best_lock else None,
                       best_rule="native val mAP50-95; same best for formal val/test",
                       current_commit=git("rev-parse", "HEAD"))
        add_bytes("README.md", (f"# DTR v1 {state}\n\nSee summary.json for all missing evidence. "
            "No raw dataset or checkpoint weight bodies included; hashes/paths/sizes are recorded. "
            "All available query+GT exports, curves, failures and recovery history are included. "
            "PASS applies only to the recorded scope; PENDING is not PASS. "
            "manifest.json intentionally does not contain its own hash.\n").encode())
        add_bytes("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False).encode())
        for prefix, base in (("evidence", OUT), ("training", RUN)):
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if not p.is_file() or folder in p.parents or p.suffix in {".pt", ".pth", ".tmp"} or p.name == "operation.lock":
                    continue
                add_file(prefix + "/" + p.relative_to(base).as_posix(), p)
        for p in sorted((ROOT / "docs/dtr_v1").glob("*")):
            if p.is_file():
                add_file("delivery/" + p.name, p)
        # Manifest is added last and explicitly excluded from its own hash list.
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        info = tarfile.TarInfo("manifest.json"); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with tarfile.open(path) as archive:
        import hashlib
        for row in manifest:
            h = hashlib.sha256()
            f = archive.extractfile(row["path"])
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
            require(h.hexdigest() == row["sha256"], "Archive integrity failure")
    return dict(file_info(path), completeness=state, missing=missing)


def finish():
    evaluate("val")
    evaluate("test")
    key = {s: sha256(OUT / f"{s}_lock.json") for s in ("val", "test")}
    prior = read_json(OUT / "finish.json", {})
    if prior.get("locks") == key and prior.get("analysis_sha256") == (
            sha256(OUT / "analysis.json") if (OUT / "analysis.json").is_file() else None):
        artifact = prior.get("package", {})
        if artifact.get("path") and file_info(artifact["path"])["sha256"] == artifact.get("sha256"):
            return dict(artifact, reused=True)
    offline_summary()
    result = package()
    require(result["completeness"] == "COMPLETE", f"Finish has missing evidence: {result['missing']}")
    write_json(OUT / "finish.json", dict(status="PASS", locks=key, package=result,
                                       analysis_sha256=sha256(OUT / "analysis.json"), completed=now()))
    return result
