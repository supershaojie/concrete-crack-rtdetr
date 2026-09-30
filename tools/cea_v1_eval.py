"""Single-pass FP32 evaluation exports, best lock, reuse and offline-only packaging."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import torch

from cea_v1_common import (ROOT, OUT, RUN, BASE, require, sha256, digest, read_json, write_json,
                           prepared_identity, runtime, git, functional_files)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7,
            max_det=300, augment=False, rect=False, seed=42)
REFERENCE = dict(val_map50_95=52.454272, test_map50_95=52.200902,
                 test_P=86.0239, test_R=83.5359, test_F1=84.7617, test_AP50=89.1997, test_AP75=54.0024)


def best_lock(identity):
    complete = read_json(RUN / "training_complete.json")
    require(complete["status"] == "TRAINING_COMPLETE" and complete["cea_identity"] == identity, "Formal run not complete/same identity")
    selected = read_json(RUN / "best_selection.json")
    require(selected["cea_identity"] == identity, "Best selection identity mismatch")
    best = RUN / "weights/best.pt"
    require(best.is_file(), "Val-selected best missing; never substitute last")
    lock = dict(path=str(best), sha256=sha256(best), epoch_zero_based=selected["epoch_zero_based"],
                training_val_map50_95=selected["best_fitness"], identity=identity)
    path = OUT / "best_lock.json"
    if path.exists():
        require(read_json(path) == lock, "Locked best changed; preserve prior evaluations and investigate")
    else:
        write_json(path, lock)
    print("VAL-SELECTED BEST\n" + json.dumps(lock, ensure_ascii=False, indent=2))
    return lock


def postprocess_with_queries(predictions, imgsz, conf):
    """Mother corrected sorting/mask semantics, retaining original ordinary query index."""
    predictions = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    result = []
    for pred in predictions:
        require(torch.isfinite(pred).all(), "Nonfinite actual inference output")
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        score, cls = pred[:, 4:].max(-1)
        order = score.argsort(descending=True)
        all_queries = dict(bboxes=boxes[order], conf=score[order], cls=cls[order], query_index=order)
        mask = all_queries["conf"] > conf  # mask is computed AFTER sorting
        selected = {k: v[mask] for k, v in all_queries.items() if k != "query_index"}
        selected["_all_queries"] = all_queries
        result.append(selected)
    return result


def evaluation_key(lock, split):
    return dict(best_sha256=lock["sha256"], split=split, identity=lock["identity"], protocol=POLICY, settings=EVAL)


def reusable(report, key):
    return (report.get("status") == "COMPLETED" and report.get("key") == key and
            bool(report.get("artifacts")) and all(Path(p).is_file() and sha256(p) == h for p, h in report["artifacts"].items()))


def print_metrics(report, reused=False):
    m, split = report["metrics"], report["split"]
    print(("REUSED " if reused else "COMPLETED ") + split.upper() + " FP32")
    print(f"P={100*m['P']:.4f}% R={100*m['R']:.4f}% F1={100*m['F1']:.4f}% "
          f"AP50={100*m['AP50']:.4f}% AP75={100*m['AP75']:.4f}% mAP50-95={100*m['mAP50_95']:.6f}%")
    print(f"mAP50-95 vs mother: {100*m['mAP50_95']-REFERENCE[split+'_map50_95']:+.6f} percentage points")
    if split == "test":
        print(f"AP50/AP75 vs mother: {100*m['AP50']-REFERENCE['test_AP50']:+.4f}/"
              f"{100*m['AP75']-REFERENCE['test_AP75']:+.4f} percentage points")
    print("P/R/F1: each model's own maximum-F1 reporting point, verified against mother's archived protocol; "
          "test statistics do not select checkpoints, hyperparameters or a deployment threshold.")
    if split == "test":
        print(f"P/R/F1 vs mother: {100*m['P']-REFERENCE['test_P']:+.4f}/"
              f"{100*m['R']-REFERENCE['test_R']:+.4f}/{100*m['F1']-REFERENCE['test_F1']:+.4f} percentage points")
    print("best SHA256: " + report["key"]["best_sha256"])
    print("Report: " + report["report_path"] + "\nEvaluation finished. pack is a separate offline command.")


def evaluate(split):
    prepared, identity = prepared_identity()
    lock = best_lock(identity)
    key = evaluation_key(lock, split)
    pointer = OUT / f"evaluation_{split}.json"
    if pointer.exists():
        previous = read_json(pointer)
        if reusable(previous, key):
            print_metrics(previous, reused=True)
            return previous
        print("Existing evaluation incomplete/identity changed; preserving it and creating a new attempt.")
    folder = OUT / f"evaluation_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}"
    folder.mkdir(parents=True, exist_ok=False)
    report_path = folder / "metrics.json"
    stream_path = folder / "predictions_gt.jsonl.gz"
    stats_path = folder / "ap_pr_stats.npz"
    curves_path = folder / "curves.npz"
    report = dict(status="FAIL", split=split, key=key, runtime=runtime(), report_path=str(report_path),
                  historical_reference=REFERENCE, precision_recall_policy="native per-split maximum-F1 reporting only",
                  historical_precision_recall_threshold="same each-model maximum-F1 reporting policy; not identical numeric thresholds",
                  source_protocol="mother c19_lif_v1_results.postprocess; docs/cea_v1/mother_evidence.json")
    seen = set()
    counts = dict(images=0, ground_truth=0, all_queries=0, metric_queries=0)
    model = RTDETR(lock["path"])
    model.model.float().eval()
    report["parameters_before_backend"] = sum(p.numel() for p in model.model.parameters())
    require(report["parameters_before_backend"] in (20149765, 19944965), "Unexpected same-topology parameter count")
    with gzip.open(stream_path, "xt", encoding="utf-8") as stream:
        class ExportValidator(RTDETRValidator):
            def init_metrics(self, backend):
                super().init_metrics(backend)
                require(not self.training and not self.args.half, "Independent FP32 evaluation required")
                require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()), "Effective evaluation protocol changed")
                self.expected = {Path(p).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                                 for p in self.dataloader.dataset.im_files}
                require(len(self.expected) == len(self.dataloader.dataset.im_files), "Duplicate split image")
                report["actual_settings"] = vars(self.args).copy()

            def postprocess(self, preds):
                return postprocess_with_queries(preds, self.args.imgsz, self.args.conf)

            def update_metrics(self, preds, batch):
                for index, pred in enumerate(preds):
                    allq = pred.pop("_all_queries")
                    gt = self._prepare_batch(index, batch)
                    require(len(allq["conf"]) == 300, "Expected all 300 ordinary queries")
                    h, w = map(int, gt["ori_shape"]); ih, iw = map(int, gt["imgsz"])
                    scale = torch.tensor([w / iw, h / ih, w / iw, h / ih])
                    boxes = allq["bboxes"].detach().float().cpu() * scale
                    truths = gt["bboxes"].detach().float().cpu() * scale
                    path = Path(gt["im_file"]).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                    require(path not in seen, "Duplicate evaluation image")
                    seen.add(path)
                    row = dict(image=path, original_size_hw=[h, w], box_format="xyxy",
                               coordinate_space="original_image_pixels", conversion="RT-DETR stretch inversion, no rounding/clamp",
                               query_index_space="original ordinary final decoder query 0..299, DN excluded",
                               predictions=[dict(query_index=int(q), bbox=b, score=float(s), class_id=int(c), used_for_metrics=bool(s > EVAL["conf"]))
                                            for q, b, s, c in zip(allq["query_index"].cpu().tolist(), boxes.tolist(), allq["conf"].cpu().tolist(), allq["cls"].cpu().tolist())],
                               ground_truth=[dict(bbox=b, class_id=int(c)) for b, c in zip(truths.tolist(), gt["cls"].cpu().tolist())])
                    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    counts["images"] += 1; counts["ground_truth"] += len(truths)
                    counts["all_queries"] += len(boxes); counts["metric_queries"] += len(pred["conf"])
                stream.flush()
                super().update_metrics(preds, batch)

            def get_stats(self):
                np.savez_compressed(stats_path, **{k: np.concatenate(v, 0) for k, v in self.metrics.stats.items()},
                                    iou_thresholds=np.arange(.5, 1., .05), conf_threshold=np.array([EVAL["conf"]]))
                return super().get_stats()

            def finalize_metrics(self):
                require(seen == self.expected, "Evaluation did not cover full split")
                return super().finalize_metrics()

        try:
            metrics = model.val(validator=ExportValidator, **EVAL, data=prepared["recipe"]["data"], split=split,
                                device="0", project=str(folder), name="plots", exist_ok=False, plots=True,
                                save_json=False, save_txt=False)
            require(sha256(lock["path"]) == lock["sha256"], "Best changed during evaluation")
            ap = np.asarray(metrics.box.all_ap)
            report["parameters_after_backend"] = sum(p.numel() for p in model.model.parameters())
            require(report["parameters_after_backend"] == 19944965, "Fused parameter count differs from mother")
            require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid AP shape/value")
            inventory = prepared["data"]["split_inventory"][split]
            require(counts["images"] == inventory["images"] and counts["ground_truth"] == inventory["boxes"], "Split count mismatch")
            require(hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest() == inventory["split_paths_sha256"], "Split identity mismatch")
            p, r = float(metrics.box.mp), float(metrics.box.mr)
            np.savez_compressed(curves_path, all_ap=ap, **{k: np.asarray(getattr(metrics.box, k)) for k in
                                ("p_curve", "r_curve", "f1_curve", "px", "prec_values")})
            report.update(status="COMPLETED", metrics=dict(P=p, R=r, F1=2*p*r/(p+r) if p+r else 0.,
                          AP50=float(ap[:, 0].mean()), AP75=float(ap[:, 5].mean()), mAP50_95=float(ap.mean())),
                          ap_by_class=ap.tolist(), counts=counts, export_complete=True)
        except BaseException as error:
            report.update(status="RESOURCE_ERROR" if isinstance(error, torch.cuda.OutOfMemoryError) else "FAIL", error=repr(error), counts=counts)
            write_json(report_path, report); write_json(pointer, report)
            raise
    report["artifacts"] = {str(p): sha256(p) for p in (stream_path, stats_path, curves_path)}
    write_json(report_path, report); write_json(pointer, report)
    print_metrics(report)
    return report


def pack():
    """Purely offline: collect existing evidence, never load weights into a model."""
    from cea_v1_process import active_workers
    require(not active_workers(), "This experiment has an active writer; package after it exits")
    missing = []
    required = [OUT / (n + ".json") for n in ("prepare", "initialization", "preflight", "diagnose", "best_lock", "evaluation_val", "evaluation_test")]
    required += [RUN / n for n in ("results.csv", "args.yaml", "training_complete.json", "best_selection.json")]
    for path in required:
        if not path.is_file(): missing.append(str(path))
    for split in ("val", "test"):
        path = OUT / f"evaluation_{split}.json"
        if path.exists():
            row = read_json(path)
            if not reusable(row, row.get("key")):
                missing.append(f"{split}: successful evaluation/artifact integrity")
    if (OUT / "preflight.json").exists() and read_json(OUT / "preflight.json").get("status") != "TECHNICAL_PASS":
        missing.append("TECHNICAL_PASS preflight")
    if (OUT / "diagnose.json").exists() and read_json(OUT / "diagnose.json").get("status") != "COMPLETED":
        missing.append("completed diagnosis")
    prepared_path = OUT / "prepare.json"
    if prepared_path.exists():
        identity = read_json(prepared_path).get("identity")
        for name in ("preflight", "diagnose"):
            path = OUT / (name + ".json")
            if path.exists() and read_json(path).get("identity") != identity:
                missing.append(name + ": stale/different identity")
        for split in ("val", "test"):
            path = OUT / f"evaluation_{split}.json"
            if path.exists() and read_json(path).get("key", {}).get("identity") != identity:
                missing.append(split + ": different evaluation identity")
        if (RUN / "training_complete.json").exists() and read_json(RUN / "training_complete.json").get("cea_identity") != identity:
            missing.append("different training identity")
    for weight in (RUN / "weights/best.pt", RUN / "weights/last.pt"):
        if not weight.is_file(): missing.append(str(weight))
    lock_path = OUT / "best_lock.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        if not Path(lock["path"]).is_file() or sha256(lock["path"]) != lock["sha256"]:
            missing.append("changed/missing locked best")
        for split in ("val", "test"):
            path = OUT / f"evaluation_{split}.json"
            if path.exists() and read_json(path).get("key", {}).get("best_sha256") != lock["sha256"]:
                missing.append(split + ": wrong best hash")
    # Completion also needs the Python AND tee exit codes of the latest train/resume
    # attempt. A dead pane alone is never successful completion evidence.
    training_workers = []
    for p in (OUT / "workers").glob("*/launch.json"):
        launch = read_json(p)
        if launch.get("action") in ("start", "resume"):
            training_workers.append((launch.get("time", 0), p.parent))
    latest_worker = max(training_workers, default=(0, None))[1]
    if latest_worker is None or not (latest_worker / "exit_codes.json").exists():
        missing.append("training worker exit evidence")
    elif read_json(latest_worker / "exit_codes.json") != {"python": 0, "tee": 0}:
        missing.append("unsuccessful training Python/tee exit")
    for split in ("val", "test"):
        attempts = []
        for p in (OUT / "workers").glob("*/launch.json"):
            launch = read_json(p)
            if launch.get("action") == split or (split == "test" and launch.get("action") == "finish"):
                attempts.append((launch.get("time", 0), p.parent))
        latest = max(attempts, default=(0, None))[1]
        if latest is None or not (latest / "exit_codes.json").exists() or read_json(latest / "exit_codes.json") != {"python": 0, "tee": 0}:
            missing.append(split + ": successful Python/tee exit evidence")
    status = "INCOMPLETE" if missing else "COMPLETE"
    files = {"source/" + p.relative_to(ROOT).as_posix(): p for p in functional_files()}
    for p in (ROOT / "docs/cea_v1").glob("*"):
        if p.is_file(): files["source/" + p.relative_to(ROOT).as_posix()] = p
    # These are dedicated CEA directories. Exclude images, weights and package recursion.
    for folder, prefix in ((OUT, "outputs"), (RUN, "training")):
        for p in folder.rglob("*"):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in (".json", ".jsonl", ".yaml", ".csv", ".log", ".npz", ".gz", ".txt", ".sh") and "packages" not in p.relative_to(folder).parts:
                files[prefix + "/" + p.relative_to(folder).as_posix()] = p
    metadata = dict(status=status, missing=missing, runtime=runtime(), weight_bodies_included=False,
                    weights={str(p): sha256(p) for p in (RUN / "weights/best.pt", RUN / "weights/last.pt") if p.is_file()})
    extra = {"package.json": json.dumps(metadata, indent=2).encode(), "source_from_mother.patch": git("diff", BASE, "--").encode()}
    manifest = [dict(path=n, bytes=p.stat().st_size, sha256=sha256(p)) for n, p in sorted(files.items())]
    manifest += [dict(path=n, bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in extra.items()]
    package_id = digest(manifest)[:16]
    destination = OUT / "packages" / f"cea_v1_{status}_{package_id}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_archive(destination)
        print("REUSED " + str(destination)); return str(destination)
    extra["MANIFEST.json"] = json.dumps(manifest, indent=2).encode()
    with destination.open("xb") as output, tarfile.open(fileobj=output, mode="w|gz") as archive:
        for name, p in sorted(files.items()): archive.add(p, arcname=name, recursive=False)
        for name, content in extra.items():
            info = tarfile.TarInfo(name); info.size = len(content); archive.addfile(info, io.BytesIO(content))
    verify_archive(destination)
    digest_ = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest_ + "  " + destination.name + "\n", encoding="utf-8")
    print(json.dumps(dict(status=status, missing=missing, archive=str(destination), sha256=digest_), indent=2))
    return str(destination)


def verify_archive(path):
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        require(len(names) == len(set(names)), "Duplicate archive members")
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        require(set(names) == {r["path"] for r in manifest} | {"MANIFEST.json"}, "Archive inventory mismatch")
        for row in manifest:
            p = PurePosixPath(row["path"])
            require(not p.is_absolute() and ".." not in p.parts, "Unsafe archive member")
            stream = archive.extractfile(row["path"])
            h = hashlib.sha256(); count = 0
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(data); count += len(data)
            require(count == row["bytes"] and h.hexdigest() == row["sha256"], "Archive SHA256/size mismatch")
