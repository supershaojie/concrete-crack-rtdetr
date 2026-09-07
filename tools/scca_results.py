"""Same-policy val/test, bounded optional diagnostics and <=20 MiB result export."""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import io
import json
import random
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import torch
from init_scca import ROOT, VARIANTS, require, runtime, sha256, write_json
from train_scca import MAIN, paths
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.modules import SCCAAIFI
from ultralytics.utils import YAML, ops


def postprocess(preds, imgsz, conf):
    preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    result, affected = [], 0
    for pred in preds:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        score, cls = pred[:, 4:].max(-1)
        order = score.argsort(descending=True)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[order]
        mask = rows[:, 4] > conf
        affected += int(not torch.equal(mask, score > conf))
        rows = rows[mask]
        result.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return result, affected


def evaluate(weights, data, split, output, device="0", batch=16, val_report=None):
    weights, data, output = weights.resolve(), data.resolve(), output.resolve()
    require(weights.is_file(), f"Missing best/checkpoint: {weights}")
    require(not output.exists(), f"Preserve previous evaluation: {output}")
    settings = dict(data=str(data.resolve()), split=split, imgsz=640, batch=batch, workers=0, device=device,
                    half=False, conf=.001, iou=.7, max_det=300, augment=False, rect=False,
                    plots=True, save_json=False, save_txt=False, project=str(output), name="plots", exist_ok=False)
    digest = sha256(weights)
    if split == "test":
        require(val_report and val_report.is_file(), "Test requires completed val of selected best")
        prior = json.loads(val_report.read_text(encoding="utf-8"))
        require(prior["status"] == "completed" and prior["split"] == "val" and prior["checkpoint_sha256"] == digest,
                "Test checkpoint not the validated selection")
        require(prior["data_sha256"] == sha256(data), "Val/test data config differs")
        require(prior["policy"] == "corrected_sorted_conf_mask_v1", "Val/test postprocess policy differs")
        for k in ("imgsz", "batch", "half", "conf", "iou", "max_det", "augment", "rect"):
            require(prior["settings"][k] == settings[k], f"Val/test setting differs: {k}")
    output.mkdir(parents=True)
    counts = dict(images=0, historical_mask_affected_images=0)

    class ComparableValidator(RTDETRValidator):
        def postprocess(self, preds):
            result, changed = postprocess(preds, self.args.imgsz, self.args.conf)
            counts["historical_mask_affected_images"] += changed
            return result

        def update_metrics(self, preds, batch):
            counts["images"] += len(preds)
            return super().update_metrics(preds, batch)

    report = dict(status="failed", runtime=runtime(), split=split, settings=settings, checkpoint=str(weights),
                  checkpoint_sha256=digest, data_sha256=sha256(data), data_config=YAML.load(data),
                  policy="corrected_sorted_conf_mask_v1", comparison="Re-evaluate C2/C17 with this identical entry. Training best selection retains original C2 validator.")
    try:
        model = RTDETR(str(weights))
        require(model.model.model[-1].nc == 1, "Evaluation requires nc=1")
        report["parameters_unfused"] = sum(p.numel() for p in model.model.parameters())
        metrics = model.val(validator=ComparableValidator, **settings)
        ap = metrics.box.all_ap
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Nonfinite/unexpected AP")
        require(sha256(weights) == digest, "Checkpoint changed during evaluation")
        report.update(status="completed", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP75=float(ap[:, 5].mean()), mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      speed_ms_per_image=metrics.speed, **counts)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(output / "metrics.json", report)
    return report


def diagnose(weights, data, output, device="0", limit=4):
    """Read at most four deterministic val images, without changing normal forward or RNG."""
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.base import BaseDataset
    require(1 <= limit <= 4 and not output.exists(), "Diagnostic limit 1..4; preserve prior output")
    resolved = check_det_dataset(str(data), autodownload=False)
    dataset = object.__new__(BaseDataset)
    dataset.prefix, dataset.fraction = "", 1.0
    random_state = random.getstate()
    try:
        images = sorted(dataset.get_img_files(resolved["val"]))[:limit]
    finally:
        random.setstate(random_state)
    model = RTDETR(str(weights))
    modules = [m for m in model.model.modules() if isinstance(m, SCCAAIFI)]
    require(len(modules) == 1, "Expected one SCCA module")
    m, summaries = modules[0], []

    def hook(_ma, args, kwargs, result):
        if len(summaries) >= limit:
            return
        with torch.no_grad():
            x, s = kwargs["value"], result[0]
            d, attention = m.scca_channel(x, s)
            # Scalar reductions only; no full activation transfer, quantiles, or retained graphs.
            rms = lambda t: float(t.detach().float().square().mean().sqrt())
            dr, xr, sr = rms(d), rms(x), rms(s)
            a = attention.detach().float().cpu()
            raw = m.scca_temperature_raw.detach().float().cpu()
            summaries.append(dict(r_X=dr/(xr+1e-8), r_S=dr/(sr+1e-8), rms_D=dr, rms_X=xr, rms_S=sr,
                                  small_X_denominator=xr<1e-6, small_S_denominator=sr<1e-6,
                                  temperatures=(np.log(4)*raw.tanh()).exp().flatten().tolist(),
                                  entropy_by_head=(-(a*a.clamp_min(1e-30).log()).sum(-1)).mean((0, 2)).tolist(),
                                  S_spatial_variation_rms=rms(s.float()-s.float().mean(1, keepdim=True))))

    handles = []
    try:
        # Warm the backend first: do not count its zero-image warmup as a val diagnostic.
        list(model.predict(images[0], imgsz=640, batch=1, device=device, save=False, verbose=False, stream=True))
        handles.append(m.ma.register_forward_hook(hook, with_kwargs=True))
        for image in images:
            list(model.predict(image, imgsz=640, batch=1, device=device, save=False, verbose=False, stream=True))
    finally:
        for handle in handles: handle.remove()
    require(len(summaries) == len(images), "Diagnostics did not capture selected samples")
    write_json(output, dict(checkpoint_sha256=sha256(weights), images=[dict(path=p, sha256=sha256(p)) for p in images],
                           statistics=summaries, note="Exact scalar RMS/variation reductions; only four small attention matrices reach CPU. Small denominators marked separately. Not full metrics."))


def package(variant, destination=None):
    p = paths(variant)
    destination = destination or MAIN / "downloads/scca" / f"{variant}_scca_{datetime.now():%Y%m%d_%H%M%S_%f}.tar.gz"
    require(not destination.exists(), "Preserve old archive")
    content = {}
    allowed_plots = {"results.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
                     "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png"}

    def add(path, name):
        require(path.suffix.lower() in {".py", ".yaml", ".json", ".md", ".mmd", ".sh", ".log", ".txt", ".csv", ".png"}, "Non-allowlisted file")
        require(path.suffix.lower() != ".png" or path.name in allowed_plots, "Dataset/prediction images excluded")
        if path.suffix == ".log":
            # Full logs, losslessly compressed in bounded chunks; no tail truncation.
            compressed = io.BytesIO()
            with path.open("rb") as src, gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(chunk)
                    require(compressed.tell() <= 20 * 1024 * 1024, "Compressed log exceeds package limit")
            content[name + ".gz"] = compressed.getvalue()
        else:
            require(path.stat().st_size <= 20 * 1024 * 1024, f"Oversized evidence: {path}; preserve it separately")
            content[name] = path.read_bytes()

    for group, folder in (("launch", p["launch"]), ("training", p["run"])):
        if folder.exists():
            for f in sorted(folder.iterdir()):
                if f.is_file() and (f.suffix in {".json", ".yaml", ".csv", ".log", ".txt", ".sh"} or f.name == "results.png"):
                    add(f, group + "/" + f.name)
        else:
            content[group + "/NOT_RUN.txt"] = b"No records available. Not marked passed.\n"
    for split in ("val", "test"):
        folder = p["launch"] / ("evaluation_" + split)
        if folder.exists():
            for f in sorted(folder.rglob("*")):
                if f.is_file() and (f.suffix == ".json" or f.name in allowed_plots):
                    add(f, "evaluation/" + split + "/" + f.relative_to(folder).as_posix())
        else:
            content[f"evaluation/{split}/NOT_RUN.txt"] = b"Not evaluated; no metrics claimed.\n"
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        f = ROOT / name
        if f.is_file() and ((name.startswith("ultralytics-main/") and f.suffix in {".py", ".yaml"}) or
                           (name.startswith("tools/") and "scca" in name) or
                           name == "tools/init_rtdetr_r18_lite_cscef_v3_from_baseline.py" or
                           name.startswith("docs/scca/") or name == "experiment_records/scca_aifi.md"):
            add(f, "source/" + name)
    content["commit.json"] = (json.dumps(runtime(), indent=2) + "\n").encode()
    content["source.diff"] = subprocess.check_output(["git", "diff", "ed366ebdb514a83a5504243213a1580bf74a1a48", "HEAD", "--", "tools", "ultralytics-main", "docs/scca", "experiment_records/scca_aifi.md"], cwd=ROOT)
    manifest = [{"path": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in sorted(content.items())]
    content["MANIFEST.json"] = json.dumps(manifest, indent=2).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, value in sorted(content.items()):
            item = tarfile.TarInfo(name)
            item.size = len(value)
            archive.addfile(item, io.BytesIO(value))
    require(buffer.tell() <= 20 * 1024 * 1024, "Archive exceeds 20 MiB; no archive written, evidence preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as f: f.write(buffer.getvalue())
    destination.with_suffix(destination.suffix + ".sha256").write_text(sha256(destination) + "  " + destination.name + "\n")
    write_json(destination.with_suffix(destination.suffix + ".inventory.json"), manifest)
    print(destination, destination.stat().st_size, sha256(destination))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("val", "test", "pack", "diagnose"))
    parser.add_argument("variant", type=str.lower, choices=VARIANTS)
    parser.add_argument("--weights", type=Path, help="Also supports historical C2/C17 best for same-policy val")
    parser.add_argument("--data", type=Path, default=MAIN / "configs/crack_autodl.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-report", type=Path, help="Prior same-policy val report when using explicit weights/output")
    args = parser.parse_args()
    p = paths(args.variant)
    weights = args.weights or p["run"] / "weights/best.pt"
    if args.mode == "pack":
        package(args.variant, args.output)
    elif args.mode == "diagnose":
        diagnose(weights, args.data, args.output or p["launch"] / "diagnostics.json", args.device)
    else:
        if args.weights is None:
            state = json.loads((p["launch"] / "exit_code.json").read_text(encoding="utf-8"))
            require(state["exit_code"] == 0, "Formal training not completed successfully")
        evaluate(weights, args.data, args.mode, args.output or p["launch"] / ("evaluation_" + args.mode), args.device,
                 args.batch, (args.val_report or p["launch"] / "evaluation_val/metrics.json") if args.mode == "test" else None)
