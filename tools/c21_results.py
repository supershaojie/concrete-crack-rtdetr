"""Independent C21 val/test with the C19 evaluation settings; allowlisted <=20 MiB export."""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile
import subprocess

import numpy as np
import torch

from init_rtdetr_r18_lite_sala_controlled import ROOT, require, sha256, write_json, runtime_info, verify_module, verify_protected
from ultralytics import RTDETR
from ultralytics.data.utils import img2label_paths
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML
from ultralytics.utils.metrics import box_iou


def shape_diagnostics(prepared, pred, groups):
    """Aggregate best-overlap coverage by axis-aligned GT shape, separate from native AP/recall.

    Each GT independently takes its best same-class prediction at confidence >= .25.
    This many-to-one diagnostic is not detection recall and does not alter native matching.
    Short side is measured in stretched 640-image pixels, not physical crack thickness.
    """
    boxes = prepared['bboxes']
    if not len(boxes):
        return
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    short = wh.amin(-1)
    ratio = wh.amax(-1) / short
    selected = pred['conf'] >= .25
    candidates = pred['bboxes'][selected]
    if len(candidates):
        overlaps = box_iou(boxes, candidates)
        overlaps *= (prepared['cls'][:, None] == pred['cls'][selected][None]).to(overlaps.dtype)
        best = overlaps.amax(-1)
    else:
        best = short.new_zeros(len(short))
    for side, aspect, overlap in zip(short.tolist(), ratio.tolist(), best.tolist()):
        bins = ['short_lt8' if side < 8 else 'short_8_to32' if side < 32 else 'short_ge32',
                'aspect_lt3' if aspect < 3 else 'aspect_3_to10' if aspect < 10 else 'aspect_ge10']
        for key in bins:
            row = groups.setdefault(key, {'gt_count':0, 'best_iou_sum':0., 'coverage_iou50_count':0, 'coverage_iou75_count':0})
            row['gt_count'] += 1
            row['best_iou_sum'] += overlap
            row['coverage_iou50_count'] += int(overlap >= .5)
            row['coverage_iou75_count'] += int(overlap >= .75)


def evaluate(args):
    require(not args.output.exists(), f"Preserve prior evaluation: {args.output}")
    verify_protected()
    checkpoint_hash = sha256(args.weights)
    settings = dict(data=str(args.data.resolve()), split=args.split, imgsz=640, batch=args.batch, workers=0,
                    device=args.device, half=False, conf=.001, iou=.7, max_det=300, augment=False,
                    rect=False, save_json=False, save_txt=False, plots=False, project=str(args.output.resolve()),
                    name="C21", exist_ok=False, verbose=False)
    if args.split == "test":
        require(args.val_report and args.val_report.is_file(), "Test requires the completed C21 val metrics_summary.json.")
        prior = json.loads(args.val_report.read_text(encoding="utf-8"))
        require(prior["status"] == "completed" and prior["split"] == "val", "Prior independent val incomplete.")
        require(prior["checkpoint_sha256"] == checkpoint_hash and prior["data_sha256"] == sha256(args.data),
                "Test checkpoint/data differs from independently validated C21 best.pt.")
        for key in ("imgsz", "batch", "half", "conf", "iou", "max_det", "augment", "rect"):
            require(prior["requested_args"][key] == settings[key], f"Val/test setting changed: {key}")
    args.output.mkdir(parents=True)
    report = {"status": "failed", "split": args.split, "runtime": runtime_info(), "data_sha256": sha256(args.data),
              "data_config": YAML.load(args.data), "checkpoint": str(args.weights.resolve()),
              "checkpoint_sha256": checkpoint_hash, "requested_args": settings,
              "metric_origin": "independent native RTDETRValidator; never training-end val or training CSV",
              "evidence_selection": "first 32 images in native deterministic validation order; full-precision predictions and GT",
              "comparison": "C19 settings retained; C2/C17/C19 are historical results and are not rerun"}
    count, evidence, captured, groups = 0, [], {}, {}

    class EvidenceValidator(RTDETRValidator):
        def update_metrics(self, preds, batch):
            nonlocal count
            for i, pred in enumerate(preds):
                count += 1
                prepared = self._prepare_batch(i, batch)
                shape_diagnostics(prepared, pred, groups)
                if len(evidence) >= 32:
                    continue
                values = torch.cat((pred["bboxes"], pred["conf"][:, None], pred["cls"][:, None]), -1)
                image = Path(prepared["im_file"])
                label = Path(img2label_paths([str(image)])[0])
                evidence.append({"image": str(image), "image_sha256": sha256(image), "label": str(label),
                    "label_sha256": sha256(label) if label.is_file() else None,
                    "original_shape": list(prepared["ori_shape"]), "imgsz": self.args.imgsz,
                    "coordinates": "stretched imgsz xyxy, native RTDETR metric tensors",
                    "gt_xyxy": prepared["bboxes"].cpu().tolist(), "gt_classes": prepared["cls"].cpu().tolist(),
                    "prediction_xyxy_conf_cls": values.cpu().tolist()})
            super().update_metrics(preds, batch)

        def finalize_metrics(self):
            super().finalize_metrics()
            captured.update(effective_args=vars(self.args), resolved_data=self.data)

    try:
        model = RTDETR(str(args.weights))
        verify_module(model.model, require_zero=False)
        require(model.model.model[-1].nc == 1, "Expected a trained one-class checkpoint.")
        parameters = sum(p.numel() for p in model.model.parameters())
        metrics = model.val(validator=EvidenceValidator, **settings)
        require(sha256(args.weights) == checkpoint_hash, "Checkpoint changed during evaluation.")
        ap = metrics.box.all_ap
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Expected finite one-class AP at ten IoUs.")
        with gzip.open(args.output / "key_predictions_gt.jsonl.gz", "wt", encoding="utf-8") as file:
            for row in evidence:
                file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        report.update(status="completed", parameters_unfused=parameters, images=count, evidence_images=len(evidence),
            precision=float(metrics.box.mp), recall=float(metrics.box.mr), mAP50=float(metrics.box.map50),
            AP75=float(np.mean(ap[:, 5])), mAP50_95=float(metrics.box.map), AP_by_class_and_IoU=ap.tolist(),
            IoU_thresholds=np.linspace(.5, .95, 10).tolist(), speed_ms_per_image=metrics.speed, **captured)
        report['shape_diagnostics'] = {'definition': shape_diagnostics.__doc__, 'confidence': .25,
            'groups': {k:{**v, 'mean_best_iou':v['best_iou_sum']/v['gt_count'],
                         'coverage_iou50':v['coverage_iou50_count']/v['gt_count'],
                         'coverage_iou75':v['coverage_iou75_count']/v['gt_count']} for k,v in groups.items()}}
        if args.val_report:
            report["prior_val_report_sha256"] = sha256(args.val_report)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.output / "metrics_summary.json", report)
    print(json.dumps({k: report[k] for k in ("split", "images", "precision", "recall", "mAP50", "AP75", "mAP50_95")}, indent=2))


def package(run, launch, val, test, output):
    require(not output.exists(), f"Preserve prior package: {output}")
    files = {}
    def add(path, name):
        require(path.is_file(), f"Required package evidence missing: {path}")
        require(path.suffix.lower() not in {".pt", ".pth", ".jpg", ".png"}, "Weights/images excluded.")
        files[name] = path.read_bytes()
    checkpoint_hash = sha256(run / "weights/best.pt")
    for split, folder in (("val", val), ("test", test)):
        metrics = json.loads((folder / "metrics_summary.json").read_text(encoding="utf-8"))
        require(metrics["status"] == "completed" and metrics["split"] == split, f"Incomplete {split} evaluation.")
        require(metrics["checkpoint_sha256"] == checkpoint_hash, f"{split} evaluated a different checkpoint.")
        for name in ("metrics_summary.json", "key_predictions_gt.jsonl.gz"):
            add(folder / name, f"evaluation/{split}/{name}")
    for name in ("args.yaml", "results.csv"):
        add(run / name, "training/" + name)
    for name in ("initialization.json", "audit.json", "launch_plan.json", "train_args.yaml", "actual_train_args.yaml",
                 "preflight.json", "tmux.json", "exit_code.json", "process_exit_code.json"):
        add(launch / name, "launch/" + name)
    for name in ('console.log', 'bootstrap.log', 'allocation.jsonl'):
        add(launch / name, 'launch/' + name)
    sources = ["tools/init_rtdetr_r18_lite_sala_controlled.py", "tools/audit_rtdetr_r18_lite_c21.py",
        "tools/train_rtdetr_r18_lite_c21.py", "tools/smoke_rtdetr_r18_lite_c21.py", "tools/c21_results.py",
        "tools/c21_tmux_worker.py", "tools/autodl_c21.sh", "docs/C21_SALA.md", "docs/C21_AUTODL.md",
        "docs/c21_source_trace.json", "docs/sala_protected.json", "ultralytics-main/tests/test_sala.py", "ultralytics-main/tests/test_c21_tools.py",
        "ultralytics-main/ultralytics/nn/tasks.py", "ultralytics-main/ultralytics/nn/modules/__init__.py",
        "ultralytics-main/ultralytics/nn/modules/sala.py", "ultralytics-main/ultralytics/nn/modules/transformer.py",
        "ultralytics-main/ultralytics/nn/modules/head.py", "ultralytics-main/ultralytics/models/utils/loss.py",
        "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-sala.yaml",
        "ultralytics-main/tests/fixtures/c2_original_args.yaml"]
    for name in sources:
        add(ROOT / name, "source/" + name)
    files["source/runtime.json"] = (json.dumps(runtime_info(), indent=2) + "\n").encode()
    files['source/changes_from_c2.patch'] = subprocess.check_output(
        ['git', 'diff', '67c3078e54a657fd96d65fee657a75fbb1dae0d6', 'HEAD', '--', 'tools', 'ultralytics-main/ultralytics'], cwd=ROOT)
    inventory = [{"name": k, "bytes": len(v)} for k, v in sorted(files.items(), key=lambda item: -len(item[1]))]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    payload = buffer.getvalue()
    print(json.dumps({"archive_bytes": len(payload), "largest_members": inventory[:15]}, indent=2), flush=True)
    require(len(payload) <= 20 * 1024 * 1024, "Package exceeds 20 MiB; largest members printed above. No evidence silently omitted.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        file.write(payload)
    write_json(output.with_name(output.name + ".inventory.json"), {"files": inventory, "archive_bytes": len(payload), "sha256": sha256(output)})
    output.with_name(output.name + ".sha256").write_text(sha256(output) + "  " + output.name + "\n", encoding="utf-8")
    print(f"Package: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate")
    for name in ("weights", "data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--val-report", type=Path)
    p.add_argument("--device", default="0"); p.add_argument("--batch", type=int, default=16)
    p = sub.add_parser("pack")
    for name in ("run", "launch", "val", "test", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "evaluate": evaluate(args)
    else: package(args.run, args.launch, args.val, args.test, args.output)


if __name__ == "__main__":
    main()
