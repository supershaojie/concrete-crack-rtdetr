"""Freeze val samples, evaluate C19/C2 under identical settings, and export a bounded evidence package."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from init_rtdetr_r18_lite_cbr_controlled import ROOT, require, sha256, write_json, runtime_info, verify_module
from ultralytics import RTDETR
from ultralytics.data.utils import img2label_paths
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy


def image_files(data, split="val"):
    config = YAML.load(data)
    root = Path(config.get("path", Path(data).resolve().parent))
    if not root.is_absolute(): root = Path(data).resolve().parent / root
    entries = config[split]
    images = []
    for entry in entries if isinstance(entries, list) else [entries]:
        path = Path(entry)
        if not path.is_absolute(): path = root / path
        if path.is_dir():
            images.extend(p.resolve() for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        elif path.suffix == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = Path(line.strip())
                    images.append(item.resolve() if item.is_absolute() else (path.parent / item).resolve())
        else:
            raise FileNotFoundError(path)
    images = sorted(set(images), key=lambda p: p.as_posix())
    require(images and all(p.is_file() for p in images), "Missing evaluation images.")
    return images


def freeze(data, output, limit):
    require(not output.exists(), f"Do not overwrite frozen sample manifest: {output}")
    require(1 <= limit <= 32, "Fixed sample limit must be 1..32.")
    files = image_files(data)
    require(len(files) >= limit, "Insufficient val images.")
    rows = []
    for image in files[:limit]:
        label = Path(img2label_paths([str(image)])[0])
        rows.append({"image": str(image), "image_sha256": sha256(image), "label": str(label),
                     "label_sha256": sha256(label) if label.is_file() else None})
    write_json(output, {"split": "val", "data_sha256": sha256(data), "created_utc": datetime.now(timezone.utc).isoformat(),
                        "selection": "first N lexicographically sorted absolute val paths, before model evaluation",
                        "samples": rows, "git_commit": runtime_info()["git_commit"]})
    print(f"Frozen {len(rows)} val samples: {output}")


def matches(before, gt):
    """One fixed Hungarian assignment maximizing total IoU; include low-IoU pairs explicitly."""
    if not len(before) or not len(gt): return [], []
    iou = box_iou(gt.float(), before.float()).cpu().numpy()
    gi, pi = linear_sum_assignment(-iou)
    return gi.tolist(), pi.tolist()


def grouped_diagnostics(rows, confidence=.25):
    """Localization/recall diagnostics from official tensors, distinct from official AP."""
    groups = defaultdict(list)
    errors = defaultdict(int)
    for row in rows:
        gt = torch.tensor(row["gt_xyxy"], dtype=torch.float32).reshape(-1, 4)
        pred = torch.tensor(row["prediction_xyxy_conf_cls"], dtype=torch.float32).reshape(-1, 6)
        pred = pred[pred[:, 4] >= confidence]
        gi, pi = matches(pred[:, :4], gt)
        assigned = {g: p for g, p in zip(gi, pi)}
        all_iou = box_iou(gt, pred[:, :4]) if len(gt) and len(pred) else torch.zeros(len(gt), len(pred))
        native_wh = (gt[:, 2:] - gt[:, :2]) * torch.tensor([row["original_shape"][1], row["original_shape"][0]]) / row["imgsz"]
        for g in range(len(gt)):
            iou = float(all_iou[g, assigned[g]]) if g in assigned else 0.
            w, h = native_wh[g].tolist()
            ratio, short = w / max(h, 1e-12), min(w, h)
            aspect = "w/h<0.5" if ratio < .5 else "0.5<=w/h<2" if ratio < 2 else "2<=w/h<5" if ratio < 5 else "w/h>=5"
            size = "short<8px" if short < 8 else "8<=short<16px" if short < 16 else "16<=short<32px" if short < 32 else "short>=32px"
            for key in ("all", aspect, size): groups[key].append(iou)
        for p in range(len(pred)):
            maximum = float(all_iou[:, p].max()) if len(gt) else 0.
            if maximum < .1: errors["background_max_iou_lt_0.1"] += 1
            elif maximum < .5: errors["localization_max_iou_0.1_to_0.5"] += 1
            elif p not in pi: errors["unassigned_overlap_ge_0.5"] += 1
            else: errors["assigned_overlap_ge_0.5"] += 1
    return {"confidence": confidence, "matching": "Hungarian max total IoU, one-to-one, no class ambiguity (nc=1)",
            "coordinate_note": "Official tensors use stretched imgsz xyxy; grouping widths/heights restored to native image pixels.",
            "not_official_AP_or_TIDE": True, "prediction_categories": dict(errors),
            "gt_groups": {k: {"count": len(v), "mean_assigned_iou": float(np.mean(v)),
                               "recall_iou50": float(np.mean(np.array(v) >= .5)),
                               "recall_iou75": float(np.mean(np.array(v) >= .75))} for k, v in groups.items()}}


def fixed_diagnostics(weights, manifest, data, device):
    require(manifest["split"] == "val" and manifest["data_sha256"] == sha256(data), "Frozen samples/data mismatch.")
    model = RTDETR(str(weights)).model.float().to(device).eval()
    verify_module(model, require_zero=False)
    rows = []
    with torch.inference_mode():
        for item in manifest["samples"]:
            image, label = Path(item["image"]), Path(item["label"])
            require(sha256(image) == item["image_sha256"], "Fixed image changed.")
            require((sha256(label) if label.is_file() else None) == item["label_sha256"], "Fixed label changed.")
            array = cv2.imread(str(image)); require(array is not None, f"Cannot decode {image}")
            tensor = torch.from_numpy(np.ascontiguousarray(cv2.resize(array, (640, 640))[:, :, ::-1].transpose(2, 0, 1)))
            result, d = model.predict(tensor[None].to(device).float() / 255, cbr_diagnostics=True)
            labels = np.loadtxt(label, ndmin=2) if label.is_file() and label.stat().st_size else np.empty((0, 5))
            require(labels.shape[1] == 5 and (not len(labels) or (labels[:, 0] == 0).all()), "Expected detection xywh labels nc=1.")
            gt = xywh2xyxy(torch.tensor(labels[:, 1:5], dtype=torch.float32))
            before, after = d["before"][0].cpu(), d["after"][0].cpu()
            scores = result[0][0, :, 4].cpu()
            indices = torch.where(scores >= .25)[0]
            gi, local = matches(xywh2xyxy(before[indices]), gt)
            pairs = []
            for g, p in zip(gi, local):
                q = int(indices[p])
                b, a = xywh2xyxy(before[q:q+1]), xywh2xyxy(after[q:q+1])
                ib, ia = float(box_iou(gt[g:g+1], b)[0, 0]), float(box_iou(gt[g:g+1], a)[0, 0])
                pairs.append({"gt_index": g, "query_index": q, "iou_before": ib, "iou_after": ia, "delta_iou": ia-ib,
                              "side_abs_error_reduction_ltrb": ((b-gt[g]).abs()-(a-gt[g]).abs())[0].tolist()})
            unit = d["tanh_offsets"][0].cpu()
            rows.append({**item, "gt_xywh": labels[:, 1:5].tolist(), "query_before_xywh": before.tolist(),
                         "query_after_xywh": after.tolist(), "query_scores": scores.tolist(), "fixed_pairs": pairs,
                         "mean_abs_displacement_normalized": d["displacement"].abs().mean().item(),
                         "mean_abs_tanh": unit.abs().mean().item(), "saturation_fraction_abs_tanh_ge_0.95": (unit.abs() >= .95).float().mean().item()})
    changes = [p["delta_iou"] for row in rows for p in row["fixed_pairs"]]
    return {"split": "val", "dtype": "FP32", "imgsz": 640, "confidence": .25,
            "matching": "Assign once from pre-CBR boxes (Hungarian max IoU); retain the same GT/query indices after CBR.",
            "interpretation": "Behavior diagnostic of a trained model; not a retrained removal ablation.",
            "mean_pair_delta_iou": float(np.mean(changes)) if changes else None, "matched_pairs": len(changes),
            "mean_abs_displacement_normalized": float(np.mean([r["mean_abs_displacement_normalized"] for r in rows])),
            "mean_saturation_fraction": float(np.mean([r["saturation_fraction_abs_tanh_ge_0.95"] for r in rows])), "samples": rows}


def evaluate(args):
    require(not args.output.exists(), f"Preserve previous evaluation: {args.output}")
    if args.split == "test":
        require(args.val_decision and args.val_decision.is_file(), "Fixed test requires --val-decision with prior val-based model selection.")
        decision = json.loads(args.val_decision.read_text(encoding="utf-8"))
        require(decision.get("selected_c19_sha256") == sha256(args.c19) and decision.get("reason"), "Test selection record does not identify this checkpoint/reason.")
    args.output.mkdir(parents=True)
    report = {"status": "failed", "split": args.split, "runtime": runtime_info(), "data_sha256": sha256(args.data),
              "metric_origin": "independent native RTDETRValidator re-evaluation, not training CSV",
              "prediction_storage": "gzip JSONL from unrounded official tensors; no additional decimal rounding", "models": {}}
    try:
        for label, path in (("C2", args.c2), ("C19", args.c19)):
            checkpoint_hash = sha256(path)
            model = RTDETR(str(path))
            require(model.model.model[-1].nc == 1, "Expected trained nc=1 checkpoint.")
            if label == "C19": verify_module(model.model, require_zero=False)
            parameters = sum(p.numel() for p in model.model.parameters())
            rows, captured = [], {}

            class EvidenceValidator(RTDETRValidator):
                def update_metrics(self, preds, batch):
                    for i, pred in enumerate(preds):
                        prepared = self._prepare_batch(i, batch)
                        values = torch.cat((pred["bboxes"], pred["conf"][:, None], pred["cls"][:, None]), -1)
                        rows.append({"image": prepared["im_file"], "original_shape": list(prepared["ori_shape"]),
                                     "imgsz": self.args.imgsz, "gt_xyxy": prepared["bboxes"].cpu().tolist(),
                                     "gt_classes": prepared["cls"].cpu().tolist(), "prediction_xyxy_conf_cls": values.cpu().tolist()})
                    super().update_metrics(preds, batch)

                def finalize_metrics(self):
                    super().finalize_metrics()
                    captured.update(effective_args=vars(self.args), resolved_data=self.data)

            settings = dict(data=str(args.data.resolve()), split=args.split, imgsz=640, batch=args.batch, workers=0,
                            device=args.device, half=False, conf=.001, iou=.7, max_det=300, augment=False,
                            rect=False, save_json=False, save_txt=False, plots=False, project=str(args.output.resolve()), name=label,
                            exist_ok=False, verbose=False)
            metrics = model.val(validator=EvidenceValidator, **settings)
            require(sha256(path) == checkpoint_hash, "Checkpoint changed during evaluation; freeze best.pt first.")
            ap = metrics.box.all_ap
            with gzip.open(args.output / f"{label}_predictions_gt.jsonl.gz", "wt", encoding="utf-8") as file:
                for row in rows: file.write(json.dumps(row, separators=(",", ":")) + "\n")
            report["models"][label] = {"checkpoint": str(path.resolve()), "checkpoint_sha256": checkpoint_hash,
                "parameters_unfused": parameters, "images": len(rows), "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50), "AP75": float(np.mean(ap[:, 5])),
                "mAP50_95": float(metrics.box.map), "AP_by_class_and_IoU": ap.tolist(), "IoU_thresholds": np.linspace(.5,.95,10).tolist(),
                "speed_ms_per_image": metrics.speed, "requested_args": settings, **captured,
                "localization_diagnostics": grouped_diagnostics(rows)}
        manifest = json.loads(args.samples.read_text(encoding="utf-8"))
        # Always use the frozen val subset, including after a fixed test evaluation.
        write_json(args.output / "fixed_val_diagnostics.json", fixed_diagnostics(args.c19, manifest, args.data,
                   "cpu" if args.device == "cpu" else "cuda:" + args.device))
        write_json(args.output / "fixed_val_samples.json", manifest)
        if args.val_decision:
            write_json(args.output / "val_selection.json", json.loads(args.val_decision.read_text(encoding="utf-8")))
        report["status"] = "completed"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.output / "metrics_summary.json", report)


def package(run, launch, evaluation, output, max_mb=20):
    """Explicit allowlist. Print sizes, then refuse >20 MB instead of silently dropping evidence."""
    require(not output.exists(), f"Refusing package overwrite: {output}")
    require(0 < max_mb <= 20, "Package budget must be within 20 MB.")
    metrics = json.loads((evaluation / "metrics_summary.json").read_text(encoding="utf-8"))
    require(metrics["status"] == "completed", "Evaluation is incomplete.")
    require(sha256(run / "weights/best.pt") == metrics["models"]["C19"]["checkpoint_sha256"], "Run best.pt differs from evaluated C19.")
    files = {}
    def add(path, name, required=True):
        if not path.is_file():
            require(not required, f"Required package evidence missing: {path}")
            return
        require(path.suffix.lower() not in {".pt", ".png", ".jpg"}, "Weights/images excluded.")
        files[name] = path.read_bytes()
    for name in ("args.yaml", "results.csv"): add(run / name, "training/" + name)
    for name in ("initialization.json", "audit.json", "launch_plan.json", "train_args.yaml", "actual_train_args.yaml",
                 "preflight.json", "tmux.json", "exit_code.json", "process_exit_code.json"):
        add(launch / name, "launch/" + name)
    for name in ("console.log", "bootstrap.log"):
        path = launch / name
        if path.is_file():
            with path.open("rb") as file:
                file.seek(max(0, path.stat().st_size - 64000))
                files["launch/" + name + ".tail.txt"] = file.read()
    for name in ("metrics_summary.json", "C2_predictions_gt.jsonl.gz", "C19_predictions_gt.jsonl.gz",
                 "fixed_val_diagnostics.json", "fixed_val_samples.json"):
        add(evaluation / name, "evaluation/" + name)
    add(evaluation / "val_selection.json", "evaluation/val_selection.json", required=False)
    modified = subprocess.check_output(["git", "diff", "--name-only", "67c3078", "HEAD"], cwd=ROOT, text=True).splitlines()
    # Include the full relevant C2 decoder/head/loss along with changed code for traceability.
    required_sources = ["ultralytics-main/ultralytics/nn/modules/head.py", "ultralytics-main/ultralytics/models/utils/loss.py"]
    for name in sorted(set(modified + required_sources)):
        if Path(name).suffix in {".py", ".yaml", ".md", ".json"}: add(ROOT / name, "source/" + name)
    files["source/runtime.json"] = (json.dumps(runtime_info(), indent=2) + "\n").encode()
    inventory = [{"name": k, "bytes": len(v)} for k, v in sorted(files.items(), key=lambda item: -len(item[1]))]
    print(json.dumps(inventory, indent=2), flush=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    payload = buffer.getvalue()
    print(f"Uncompressed bytes: {sum(len(v) for v in files.values())}; archive bytes: {len(payload)}", flush=True)
    require(len(payload) <= max_mb * 1024 * 1024, "Package exceeds 20MB; inspect printed large files before reducing evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file: file.write(payload)
    write_json(output.with_name(output.name + ".inventory.json"), {"files": inventory, "archive_bytes": len(payload), "sha256": sha256(output)})
    print(f"Package: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=16)
    p = sub.add_parser("evaluate")
    for name in ("c19", "c2", "data", "samples", "output"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--val-decision", type=Path)
    p.add_argument("--device", default="0"); p.add_argument("--batch", type=int, default=16)
    p = sub.add_parser("pack")
    for name in ("run", "launch", "evaluation", "output"): p.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "freeze": freeze(args.data, args.output, args.limit)
    elif args.command == "evaluate": evaluate(args)
    else: package(args.run, args.launch, args.evaluation, args.output)


if __name__ == "__main__":
    main()
