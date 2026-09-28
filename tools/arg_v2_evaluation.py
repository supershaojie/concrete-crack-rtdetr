"""One complete FP32 evaluation per split, reusable locks and explicit recovery."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import traceback

from arg_v2_common import (ROOT, OUT, RUN, COUNTS, read_json, write_json, now, require,
                           sha256, digest, data_config, cached_data)
from ultralytics.models.rtdetr.arg_v2_val import ARGv2Validator, EVAL, POLICY


def metric_summary(metrics):
    """Same native operating point, not classification accuracy or a new threshold."""
    import numpy as np
    from ultralytics.utils.metrics import smooth
    box = metrics.box
    ap = np.asarray(box.all_ap)
    require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Expected finite one-class ten-IoU AP")
    precision, recall = float(box.mp), float(box.mr)
    curve = np.asarray(getattr(box, "f1_curve", []))
    x = np.asarray(getattr(box, "px", []))
    confidence = float(x[smooth(curve.mean(0), .1).argmax()]) if curve.size and x.size else None
    return dict(Precision=precision, Recall=recall,
        F1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        AP50=float(ap[0, 0]), AP75=float(ap[0, 5]), mAP50_95=float(ap.mean()),
        ap_by_class=ap.tolist(), ap_iou_thresholds=[round(.5 + .05 * i, 2) for i in range(10)],
        units="fraction; multiply by 100 for percentages",
        operating_point=dict(rule="native smooth(mean class F1-confidence curve, 0.1).argmax at IoU=0.50",
                             confidence=confidence, confidence_available=confidence is not None,
                             evaluator_source="ultralytics/utils/metrics.py::ap_per_class"))


def evaluation_context(during_final_eval=False):
    from arg_v2 import active_workers, has_tmux, evaluation_source
    if during_final_eval:
        owners = [read_json(p, {}) for p in (OUT / "dispatches").glob("*/worker.json")]
        require(any(row.get("pid") == os.getpid() for row in owners), "Final eval is only for the recorded training worker")
    else:
        require(not active_workers() and not has_tmux(), "Wait for ARG v2 worker/tmux to finish")
    completion = read_json(OUT / "training_completed.json")
    require(completion and completion["status"] == "TRAINING_COMPLETED", "No native training-completion evidence")
    training = read_json(OUT / "training_identity.json")
    require(training, "Missing training identity")
    identity = evaluation_source(training)
    data_path = data_config()
    data = cached_data(data_path)
    require(data == training["binding"]["data"], "Training/evaluation data identity differs")
    best = RUN / "weights/best.pt"
    require(best.is_file(), "No best checkpoint")
    return dict(training=training, completion=completion, eval_identity=identity, data=data,
                data_path=str(data_path), best=str(best), checkpoint_sha256=sha256(best))


def validate_record(record, split, context):
    """Reuse requires the complete export, protocol, identity and exact artifacts."""
    from arg_v2 import file_info
    require(record and record.get("status") == "PASS" and record.get("scope") == "full_split",
            "Evaluation is not a successful full split")
    require(record["split"] == split and record["checkpoint_sha256"] == context["checkpoint_sha256"]
            and record["data"] == context["data"] and record["eval_identity"] == context["eval_identity"]
            and record["settings"] == EVAL and record["policy"] == POLICY, "Evaluation identity/protocol differs")
    require(record.get("images") == COUNTS[split][0] and record.get("gt_boxes") == COUNTS[split][1]
            and record.get("queries_per_image") == 300, "Incomplete split/query/GT coverage")
    for name in ("predictions", "curves"):
        info = record.get(name)
        require(info and Path(info["path"]).is_file(), "Successful evaluation lacks " + name + "; use explicit --recover-export")
        require(file_info(info["path"]) == info, "Evaluation artifact changed: " + name + "; use explicit --recover-export")
    for key in ("Precision", "Recall", "F1", "AP50", "AP75", "mAP50_95", "ap_by_class", "operating_point"):
        require(key in record, "Incomplete formal metrics: " + key)
    return record


def recover_completed_record(split, context, recover_export=False):
    """Recover a crash after metrics publication but before lock publication, without inference."""
    for path in sorted((OUT / "evaluations").glob("*/metrics.json"), reverse=True):
        record = read_json(path)
        if record.get("status") != "PASS" or record.get("split") != split:
            continue
        require(record["checkpoint_sha256"] == context["checkpoint_sha256"] and record["data"] == context["data"],
                "Completed evaluation report pins another best/data snapshot")
        try:
            validate_record(record, split, context)
        except (RuntimeError, KeyError, OSError):
            require(recover_export, "Completed evaluation report lacks complete material; use explicit --recover-export")
            return None
        write_json(OUT / (split + "_lock.json"), record)
        return record
    return None


def evaluate(split, recover_export=False, during_final_eval=False):
    """Default export is mandatory. Existing successful results never silently rerun."""
    from arg_v2 import checkpoint_identity, file_info
    from ultralytics.utils.torch_utils import init_seeds
    require(split in ("val", "test"), "Unsupported evaluation split")
    context = evaluation_context(during_final_eval)
    if split == "test":
        validate_record(read_json(OUT / "val_lock.json"), "val", context)
    lock_path = OUT / (split + "_lock.json")
    locked = read_json(lock_path)
    if locked:
        # A recovery may fill missing material, never select a different checkpoint.
        require(locked["checkpoint_sha256"] == context["checkpoint_sha256"] and locked["data"] == context["data"],
                "Existing evaluation pins another best/data snapshot; preserved")
        try:
            return validate_record(locked, split, context)
        except (RuntimeError, KeyError, OSError):
            require(recover_export, "Existing evaluation material/identity incomplete; inspect and use " + split + " --recover-export explicitly")
    else:
        recovered = recover_completed_record(split, context, recover_export)
        if recovered:
            return recovered
    if split == "test":
        validate_record(read_json(OUT / "val_lock.json"), "val", context)
    checkpoint_identity(Path(context["best"]), context["training"])
    folder = OUT / "evaluations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + split)
    folder.mkdir(parents=True, exist_ok=False)
    if locked:
        write_json(folder / "previous_lock.json", locked)
    settings = dict(EVAL, model=context["best"], data=context["data_path"], split=split, device="0", plots=True,
                    save_json=False, save_txt=False, project=str(folder), name="plots", exist_ok=False)
    export_identity = dict(formula="ARG-v2-q2", checkpoint_sha256=context["checkpoint_sha256"], split=split,
        training_commit=context["eval_identity"]["training_commit"], eval_commit=context["eval_identity"]["eval_commit"],
        data_snapshot_id=digest(context["data"]), config_sha256=digest(settings), policy=POLICY)
    report = dict(status="IN_PROGRESS", split=split, policy=POLICY, scope="full_split", started=now(),
        checkpoint_sha256=context["checkpoint_sha256"], data=context["data"], eval_identity=context["eval_identity"],
        settings=EVAL, actual_settings=settings, report=str(folder / "metrics.json"), export_identity=export_identity)
    write_json(folder / "metrics.json", report)
    validator = ARGv2Validator(args=settings, save_dir=folder / "plots")
    validator.export_path = folder / (split + "_predictions_gt.jsonl.gz")
    validator.export_identity = export_identity
    try:
        init_seeds(42, deterministic=True)
        metrics = validator(model=context["best"])
        require(sha256(context["best"]) == context["checkpoint_sha256"], "Best changed during evaluation")
        require(all(type(validator.actual_settings[k]) is type(v) and validator.actual_settings[k] == v for k, v in EVAL.items()),
                "Effective independent FP32 evaluation protocol differs")
        require(len(validator.arg_seen) == COUNTS[split][0] and validator.arg_gt_count == COUNTS[split][1],
                "Evaluation coverage differs from snapshot")
        curves = folder / "curves.json"
        box = validator.metrics.box
        write_json(curves, {k: getattr(box, k).tolist() for k in
                           ("px", "p_curve", "r_curve", "f1_curve", "prec_values")})
        report.update(status="PASS", metrics=metrics, **metric_summary(validator.metrics),
            images=len(validator.arg_seen), gt_boxes=validator.arg_gt_count, queries_per_image=300,
            actual_settings=validator.actual_settings, predictions=file_info(validator.export_path), curves=file_info(curves),
            ended=now())
        # Durable complete report first; a subsequent process can publish a missing lock.
        write_json(folder / "metrics.json", report)
        validate_record(report, split, context)
        write_json(lock_path, report)
        failure = read_json(OUT / "final_eval.json", {})
        if locked or (split == "val" and failure.get("status") == "FINAL_EVAL_FAILED"):
            write_json(folder / "evaluation_recovery.json", dict(status="EVALUATION_RECOVERED",
                original_final_eval=failure, previous_lock=locked, report=report["report"], created=now()))
        return report
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), ended=now())
        write_json(folder / "metrics.json", report)
        raise
