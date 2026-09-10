"""Triad native final-box evaluation and streaming complete archive; never starts training/evaluation implicitly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import torch
from triad_compat import ROOT, require, runtime, sha256, topology, write_json, SOURCE_SHA256, VARIANTS, git
from ultralytics.utils import ops
from train_triad_compat import MAIN, paths, run_state
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML
from ultralytics.data.utils import check_det_dataset

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, seed=42, half=False, conf=.001, iou=.7, max_det=300, augment=False, rect=False)


def native_predictions(preds, imgsz, conf):
    """Established C24/C25/C26 independent-evaluation policy, with all-query export."""
    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
    selected, full, affected = [], [], 0
    for pred in raw:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        score, cls = pred[:, 4:].max(-1)
        order = score.argsort(descending=True)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[order]
        mask = rows[:, 4] > conf
        affected += int(not torch.equal(mask, score > conf))
        full.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5], used_for_metrics=mask))
        rows = rows[mask]
        selected.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return selected, full, affected


def image_record(pred, gt, dataset_root, conf):
    """One image, all 300 final FP32 predictions and all GT; no decimal rounding/clipping."""
    h, w = map(int, gt["ori_shape"])
    ih, iw = map(int, gt["imgsz"])
    scale = torch.tensor([w / iw, h / ih, w / iw, h / ih], dtype=torch.float32)
    boxes = pred["bboxes"].detach().float().cpu() * scale
    gt_boxes = gt["bboxes"].detach().float().cpu() * scale
    scores, classes = pred["conf"].detach().float().cpu(), pred["cls"].detach().cpu()
    require(all(torch.isfinite(t).all() for t in (boxes, gt_boxes, scores)), "Nonfinite prediction/GT export")
    rel = Path(gt["im_file"]).resolve().relative_to(Path(dataset_root).resolve()).as_posix()
    return dict(image=rel, original_size_hw=[h, w], box_format="xyxy", coordinate_space="original_image_pixels",
                coordinates="continuous, no rounding or clipping; RT-DETR stretch inversion; class_id is zero-based",
                predictions=[dict(class_id=int(c), score=float(s), bbox=b, used_for_metrics=bool(used))
                             for b, s, c, used in zip(boxes.tolist(), scores, classes, pred["used_for_metrics"].cpu())],
                ground_truth=[dict(class_id=int(c), bbox=b) for b, c in zip(gt_boxes.tolist(), gt["cls"].detach().cpu())])


def evaluate(variant, weights, data, split, output, device="0", val_report=None):
    weights, data, output = Path(weights).resolve(), Path(data).resolve(), Path(output).resolve()
    require(split in ("val", "test"), "Expected val/test")
    require(weights.is_file() and data.is_file(), "Missing checkpoint/data config")
    require(not output.exists(), f"Preserve previous evaluation: {output}")
    settings = dict(EVAL, data=str(data), split=split, device=device, plots=True, save_json=False, save_txt=False,
                    project=str(output), name="plots", exist_ok=False)
    digest, data_digest, info = sha256(weights), sha256(data), runtime()
    if split == "test":
        require(val_report and Path(val_report).is_file(), "Test requires completed val of the selected checkpoint")
        prior = json.loads(Path(val_report).read_text(encoding="utf-8"))
        require(prior["status"] == "completed" and prior["split"] == "val" and prior["checkpoint_sha256"] == digest,
                "Test checkpoint differs from validated selection")
        require(prior["data_sha256"] == data_digest and prior["policy"] == POLICY, "Val/test data/policy changed")
        require(prior["runtime"]["commit"] == info["commit"], "Source changed after validation; validate fixed source/checkpoint again")
        require(all(type(prior["settings"][k]) is type(v) and prior["settings"][k] == v for k, v in EVAL.items()),
                "Val/test settings changed")
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "Expected crack nc=1")
    output.mkdir(parents=True, exist_ok=False)
    seen, counts = set(), dict(images=0, predictions=0, metric_predictions=0, ground_truth=0, historical_mask_affected_images=0)
    report = dict(status="failed", runtime=info, split=split, settings=settings, checkpoint=str(weights),
                  checkpoint_sha256=digest, data_sha256=data_digest, data_config=YAML.load(data), policy=POLICY,
                  variant=variant, boxes="final decoder boxes; SR-CBR once when variant enables CBR", selection="Training val selects best; freeze checkpoint/config before independent test",
                  preflight_status="not_assessed_by_evaluation_function")
    stream_path = output / "predictions_gt.jsonl.gz"
    try:
        model = RTDETR(str(weights))
        topology(model.model, variant)
        require(model.model.model[-1].nc == 1, "Evaluation requires nc=1")
        report["parameters_unfused"] = sum(p.numel() for p in model.model.parameters())
        with gzip.open(stream_path, "xt", encoding="utf-8") as stream:
            class StreamingValidator(RTDETRValidator):
                def init_metrics(self, model):
                    super().init_metrics(model)
                    require(not self.training and not self.args.half, "Independent FP32 evaluation only")
                    require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()),
                            "Effective evaluation settings changed")
                    report["actual_settings"] = vars(self.args).copy()
                    YAML.save(output / "args.yaml", report["actual_settings"])
                    write_json(output / "effective_eval_config.json", report["actual_settings"])
                    files = [Path(p).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                             for p in self.dataloader.dataset.im_files]
                    require(len(files) == len(set(files)), "Duplicate split image paths")
                    self.expected_images = set(files)
                    report["expected_images"] = len(files)
                    report["split_paths_sha256"] = hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()

                def postprocess(self, preds):
                    selected, full, affected = native_predictions(preds, self.args.imgsz, self.args.conf)
                    counts["historical_mask_affected_images"] += affected
                    # Only the current batch travels to update_metrics, never a cross-batch activation cache.
                    for selected_image, all_queries in zip(selected, full):
                        require(len(all_queries["conf"]) == 300, "Incomplete regular-query export")
                        selected_image["_all_queries"] = all_queries
                    return selected

                def update_metrics(self, preds, batch):
                    for i, pred in enumerate(preds):
                        all_queries = pred.pop("_all_queries")
                        row = image_record(all_queries, self._prepare_batch(i, batch), self.data["path"], self.args.conf)
                        require(row["image"] not in seen, "Image exported more than once")
                        seen.add(row["image"])
                        counts["images"] += 1
                        counts["predictions"] += len(row["predictions"])
                        counts["metric_predictions"] += len(pred["conf"])
                        counts["ground_truth"] += len(row["ground_truth"])
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    return super().update_metrics(preds, batch)

                def finalize_metrics(self):
                    require(seen == self.expected_images, "Prediction/GT export does not cover the complete split")
                    return super().finalize_metrics()

            metrics = model.val(validator=StreamingValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Nonfinite/unexpected AP array")
        require(sha256(weights) == digest and sha256(data) == data_digest, "Checkpoint/data changed during evaluation")
        report.update(status="completed", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP75=float(ap[:, 5].mean()), mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      ap_iou_thresholds=[round(.5 + i * .05, 2) for i in range(10)], ap_by_class=ap.tolist(),
                      ap_class_index=np.asarray(metrics.box.ap_class_index).tolist(), speed_ms_per_image=metrics.speed,
                      predictions_gt_sha256=sha256(stream_path), export_complete=True, **counts)
    except BaseException as error:
        report.update(error=repr(error), **counts)
        raise
    finally:
        write_json(output / "metrics.json", report)
    (output / "results.csv").write_text("precision,recall,mAP50,AP75,mAP50_95\n" +
        ",".join(str(report[k]) for k in ("precision", "recall", "mAP50", "AP75", "mAP50_95")) + "\n", encoding="utf-8")
    return report


def verify_archive(destination):
    """Read back every byte and reject missing, duplicate, unsafe or altered members."""
    with tarfile.open(destination, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), "Duplicate archive member")
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        require(set(names) == {r["path"] for r in manifest} | {"MANIFEST.json"}, "Archive inventory mismatch")
        for row in manifest:
            path = PurePosixPath(row["path"])
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe archive path")
            member = archive.getmember(row["path"])
            require(member.isfile() and member.size == row["bytes"], "Archive size/type mismatch")
            digest = hashlib.sha256()
            with archive.extractfile(member) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            require(digest.hexdigest() == row["sha256"], f"Archive hash mismatch: {row['path']}")
    return manifest


def validate_complete(p):
    """Strict consistency checks, only for claiming complete evidence."""
    launch, run = p["launch"], p["run"]
    require(run_state(p) == "SUCCESS", "Worker/process/artifact completion required")
    require((launch / "exit_code.json").is_file() and (launch / "process_exit_code.txt").is_file(), "Training exit evidence missing")
    require(json.loads((launch / "exit_code.json").read_text(encoding="utf-8"))["exit_code"] == 0 and
            (launch / "process_exit_code.txt").read_text().strip() == "0", "Training failed/incomplete; complete outputs unavailable")
    required = ("initialization.json", "nc1_loading.json", "training_setup.json", "actual_train_args.yaml", "train_args.yaml",
                "parameter_diff.json", "authoritative_c2_args.yaml", "data_config.yaml", "pip_freeze.txt", "source_snapshot.tar.gz",
                "source_from_base.patch", "source_record.json", "plan.json", "process.json", "console.log", "worker.sh", "launch_state.json", "model.yaml", "preflight.json", "worker_complete.json")
    require(all((launch / n).is_file() for n in required), "Triad direct-start metadata incomplete")
    require(all((run / n).is_file() for n in ("weights/best.pt", "weights/last.pt", "args.yaml", "results.csv", "results.png")),
            "Complete training outputs missing")
    def require_plots(folder, training=False):
        names = {f.name for f in folder.rglob("*") if f.is_file()}
        for suffix in ("PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"):
            require(suffix in names or "Box" + suffix in names, f"Missing curve: {folder}/{suffix}")
        require({"confusion_matrix.png", "confusion_matrix_normalized.png"} <= names, "Missing confusion matrices")
        require(any(n.startswith("val_batch") and "pred" in n for n in names), "Missing prediction samples")
        if training:
            require(any(n.startswith("labels") for n in names) and any(n.startswith("train_batch") for n in names),
                    "Missing training labels/batch visualizations")
    require_plots(run, training=True)
    plan = json.loads((launch / "plan.json").read_text(encoding="utf-8"))
    require(plan["preflight"] == "passed", "Unexpected direct-start preflight record")
    require(runtime()["commit"] == plan["runtime"]["commit"], "Package source differs from training source")
    initialization = json.loads((launch / "initialization.json").read_text(encoding="utf-8"))
    require(initialization["source_sha256"] == SOURCE_SHA256, "Wrong initialization provenance")
    require(initialization["status"] == "passed" and initialization["output_sha256"] == plan["init_sha256"], "Initialization/plan mismatch")
    frozen = json.loads((launch / "source_record.json").read_text(encoding="utf-8"))
    require(sha256(launch / "source_snapshot.tar.gz") == frozen["snapshot_sha256"], "Source snapshot changed")
    require(sha256(launch / "authoritative_c2_args.yaml") == frozen["c2_args_sha256"] == plan["c2_args_sha256"], "C2 recipe evidence changed")
    require(sha256(launch / "data_config.yaml") == frozen["data_sha256"], "Frozen data config changed")
    require(YAML.load(launch / "train_args.yaml") == YAML.load(launch / "actual_train_args.yaml") == plan["args"], "Training args evidence mismatch")
    best_hash = sha256(run / "weights/best.pt")
    evaluations = {}
    for split in ("val", "test"):
        folder = launch / ("evaluation_" + split)
        require_plots(folder)
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        require(report["status"] == "completed" and report["split"] == split and report["policy"] == POLICY,
                f"{split} evaluation incomplete/wrong policy")
        require(report["checkpoint_sha256"] == best_hash and report["runtime"]["commit"] == plan["runtime"]["commit"],
                f"{split} is not selected best/source")
        require(report["data_sha256"] == sha256(launch / "data_config.yaml") and report["export_complete"], "Data/export mismatch")
        require(all(type(report["settings"][k]) is type(v) and report["settings"][k] == v for k, v in EVAL.items()), "Wrong evaluation settings")
        require(np.asarray(report["ap_by_class"]).shape == (1, 10), "Full AP array missing")
        stream = folder / "predictions_gt.jsonl.gz"
        require(sha256(stream) == report["predictions_gt_sha256"], "Prediction/GT stream changed")
        count, predictions, truths, seen = 0, 0, 0, set()
        with gzip.open(stream, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                require(row["image"] not in seen and len(row["predictions"]) == 300, "Incomplete/duplicate prediction image")
                seen.add(row["image"])
                count += 1
                predictions += len(row["predictions"])
                truths += len(row["ground_truth"])
        require(count == report["images"] == report["expected_images"] and predictions == report["predictions"] and truths == report["ground_truth"],
                "Prediction/GT coverage/count mismatch")
        require(hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest() == report["split_paths_sha256"], "Split image inventory mismatch")
        evaluations[split] = dict(checkpoint_sha256=best_hash, images=count, predictions=predictions, ground_truth=truths)
    return evaluations, best_hash


def missing_evidence(p):
    missing = []
    required_launch = ("initialization.json", "nc1_loading.json", "training_setup.json", "actual_train_args.yaml",
                       "train_args.yaml", "parameter_diff.json", "authoritative_c2_args.yaml", "data_config.yaml",
                       "pip_freeze.txt", "source_snapshot.tar.gz", "source_from_base.patch", "source_record.json",
                       "plan.json", "process.json", "console.log", "worker.sh", "launch_state.json", "exit_code.json",
                       "process_exit_code.txt", "amp_resources.json", "model.yaml", "preflight.json", "worker_complete.json")
    for name in required_launch:
        if not (p["launch"] / name).is_file(): missing.append("metadata/launch/" + name)
    for name in ("weights/best.pt", "weights/last.pt", "args.yaml", "results.csv", "results.png"):
        if not (p["run"] / name).is_file(): missing.append("training/" + name)
    for prefix, folder in [("training", p["run"])] + [("evaluation/" + split, p["launch"] / ("evaluation_" + split)) for split in ("val", "test")]:
        names = {f.name for f in folder.rglob("*") if f.is_file()}
        for curve in ("PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"):
            if curve not in names and "Box" + curve not in names: missing.append(prefix + "/" + curve)
        for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            if name not in names: missing.append(prefix + "/" + name)
        if not any(n.startswith("val_batch") and "pred" in n for n in names): missing.append(prefix + "/prediction_samples")
        if prefix == "training":
            for stem in ("labels", "train_batch"):
                if not any(n.startswith(stem) for n in names): missing.append(prefix + "/" + stem + "_visualizations")
        else:
            for name in ("metrics.json", "predictions_gt.jsonl.gz"):
                if name not in names: missing.append(prefix + "/" + name)
    return missing


def package(variant, destination=None, allow_incomplete=False):
    """Require complete evidence for the public pack-complete command; test fixtures may explicitly allow incomplete archives."""
    p = paths(variant)
    require(run_state(p) not in {"RUNNING", "FINISHING", "DISPATCHED", "INITIALIZING"}, "Do not package a changing active run")
    destination = Path(destination).resolve() if destination else MAIN / "downloads/triad_compat" / variant / f"{variant}_complete_{datetime.now():%Y%m%d_%H%M%S_%f}.tar.gz"
    require(all(not Path(str(destination) + suffix).exists() for suffix in ("", ".partial", ".sha256", ".inventory.json", ".verification.json")), "Preserve existing package and sidecars")
    launch, run = p["launch"], p["run"]
    require(not destination.is_relative_to(launch.resolve()) and not destination.is_relative_to(run.resolve()), "Archive destination cannot be inside its inputs")
    missing = missing_evidence(p)
    evaluations, best_hash = {}, sha256(run / "weights/best.pt") if (run / "weights/best.pt").is_file() else None
    if not missing:
        try:
            evaluations, best_hash = validate_complete(p)
        except (RuntimeError, KeyError, ValueError, OSError) as error:
            missing.append("invalid_evidence: " + str(error))
    require(allow_incomplete or not missing, "pack-complete requires complete successful training/val/test evidence: " + str(missing))
    files = {}

    def tree(folder, prefix, excluded=()):
        for f in sorted(folder.rglob("*")):
            require(not f.is_symlink(), f"Symlink in outputs: {f}")
            if f.is_file() and not any(x in f.relative_to(folder).parts for x in excluded):
                files[prefix + "/" + f.relative_to(folder).as_posix()] = f

    tree(run, "training")
    tree(launch, "console", ("evaluation_val", "evaluation_test"))
    for split in ("val", "test"):
        tree(launch / ("evaluation_" + split), split)
    logs = ROOT / "outputs/triad_compat_command_logs" / variant
    if logs.exists():
        # Current pack log is still being written; include only completed command logs.
        for exit_file in logs.glob("*.log.exit_code.txt"):
            for f in (exit_file, Path(str(exit_file).removesuffix(".exit_code.txt"))):
                require(f.is_file(), "Missing completed command log")
                files["metadata/command_logs/" + f.name] = f
    tracked = git("ls-files", "-z").split("\0")
    for name in tracked:
        if name and (name.startswith(("ultralytics-main/ultralytics/", "ultralytics-main/tests/", "docs/triad_compat/")) or
                     name in ("ultralytics-main/pyproject.toml", ".gitattributes", "docs/triad_compat/c2_args.yaml") or
                     name.startswith("tools/")):
            files["metadata/source/" + name] = ROOT / name
    metadata = dict(created=datetime.now(timezone.utc).isoformat(), runtime=runtime(), evaluations=evaluations,
                    training_best_sha256=best_hash, training_last_sha256=sha256(run / "weights/last.pt") if (run / "weights/last.pt").is_file() else None,
                    evidence_complete=not missing, missing_evidence=missing,
                    unified_init_sha256=SOURCE_SHA256, preflight_status="passed" if not missing else "incomplete_fixture",
                    scope="Complete existing training + same-pass FP32 val/test predictions/GT + metadata; no dataset or environment copy")
    extra = {"metadata/package.json": (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()}
    manifest = [dict(path=n, bytes=f.stat().st_size, sha256=sha256(f)) for n, f in sorted(files.items())]
    manifest += [dict(path=n, bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in extra.items()]
    extra["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with partial.open("xb") as stream, tarfile.open(fileobj=stream, mode="w|gz", dereference=True) as archive:
        for n, f in sorted(files.items()):
            archive.add(f, arcname=n, recursive=False)
        for n, value in extra.items():
            item = tarfile.TarInfo(n)
            item.size = len(value)
            archive.addfile(item, io.BytesIO(value))
    verify_archive(partial)
    # Hard link is atomic and refuses to overwrite a destination that appeared during packaging.
    os.link(partial, destination)
    partial.unlink()
    digest = sha256(destination)
    with Path(str(destination) + ".sha256").open("x", encoding="utf-8") as f:
        f.write(digest + "  " + destination.name + "\n")
    write_json(Path(str(destination) + ".inventory.json"), manifest)
    write_json(Path(str(destination) + ".verification.json"), dict(archive_integrity="passed", evidence_complete=not missing,
               missing_evidence=missing, sha256=digest, bytes=destination.stat().st_size, members=len(manifest) + 1))
    print(f"Archive verified (evidence_complete={not missing}): {destination}\nBytes: {destination.stat().st_size}\nSHA256: {digest}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("val", "test", "pack-complete"))
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--output", type=Path, help="Archive destination for pack-complete; formal eval paths stay fixed")
    args = parser.parse_args()
    torch.set_num_threads(4)
    require(not git("status", "--porcelain"),
            "Protect reproducibility: source has uncommitted changes")
    if args.mode == "pack-complete":
        package(args.variant, args.output)
    else:
        require(args.output is None, "Formal evaluations use independent fixed Triad directories")
        p = paths(args.variant)
        require(run_state(p) == "SUCCESS", "Training has not completed successfully")
        require(json.loads((p["launch"] / "exit_code.json").read_text(encoding="utf-8"))["exit_code"] == 0 and
                (p["launch"] / "process_exit_code.txt").read_text().strip() == "0", "Formal training not completed successfully")
        plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
        source_record = json.loads((p["launch"] / "source_record.json").read_text(encoding="utf-8"))
        require(runtime()["commit"] == plan["runtime"]["commit"], "Source changed since training")
        require(sha256(p["launch"] / "data_config.yaml") == source_record["data_sha256"], "Frozen data config changed")
        evaluate(args.variant, p["run"] / "weights/best.pt", p["launch"] / "data_config.yaml", args.mode,
                 p["launch"] / ("evaluation_" + args.mode),
                 val_report=p["launch"] / "evaluation_val/metrics.json" if args.mode == "test" else None)
