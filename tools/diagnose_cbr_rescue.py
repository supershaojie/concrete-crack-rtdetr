"""C20 rescue step 1: inference-only observation and paired official validation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import gzip
import json
import logging
import os
from pathlib import Path
import sys
import traceback

from cbr_rescue_common import (ROOT, BASE, C20_SHA, INIT_SHA, require,
                              sha256, write_json, git, source_state, package)

RUNS = {
    "C2": "c2_rtdetr_r18_lite_e200_b16_onlineaug",
    "C17": "c17_rtdetr_r18_lite_cscef_v51_e200_b16_onlineaug",
    "C19": "c19_rtdetr_r18_lite_cbr_e200_b16_onlineaug",
    "C20": "c20_rtdetr_r18_lite_cscef_cbr_e200_b16_onlineaug",
}
SETTINGS = dict(imgsz=640, batch=16, conf=.001, iou=.7, max_det=300, half=False,
                augment=False, rect=False, workers=0, device="0", seed=42,
                plots=False, save_json=False, save_txt=False, cache=False, compile=False,
                single_cls=False, verbose=False, mode="val", task="detect")


def import_runtime(output):
    """Delay framework import so pack/help work without torch and no settings leak."""
    (output / "runtime_config").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "runtime_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(ROOT / "ultralytics-main"))
    import torch
    import ultralytics
    from ultralytics.utils.torch_utils import init_seeds
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "ultralytics-main"), "Wrong worktree import.")
    torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics": ultralytics.__version__, "ultralytics_file": ultralytics.__file__,
            "mode": "unfused FP32 eval + torch.inference_mode; no autocast; TF32 disabled",
            "seed": 42, "deterministic_warn_only": True, "diagnosis_timing_is_not_inference_performance": True}


def build_dataset(data, split, output, settings):
    """Native RT-DETR loader/transforms; native label verifier without disk cache or JPEG repair."""
    from PIL import Image
    from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
    from ultralytics.data.utils import img2label_paths, verify_image_label

    class ReadOnlyDataset(RTDETRDataset):
        def get_labels(self):
            self.label_files = img2label_paths(self.im_files)
            labels, self.diagnostic_warnings = [], []
            for image, label in zip(self.im_files, self.label_files):
                require(not Path(image).with_suffix(".npy").exists(),
                        f"Existing image .npy cache may be stale; diagnose from a clean image path: {image}")
                with Image.open(image) as im:
                    if im.format.lower() in {"jpg", "jpeg"}:
                        with open(image, "rb") as f:
                            f.seek(-2, 2)
                            require(f.read() == b"\xff\xd9", f"JPEG needs repair; read-only diagnosis stopped: {image}")
                result = verify_image_label((image, label, "diagnosis: ", False, len(self.data["names"]), 0, 0, False))
                name, lb, shape, segments, keypoints, nm, nf, ne, corrupt, msg = result
                require(name is not None and not corrupt, f"Invalid image/label (not silently skipped): {msg}")
                require(not segments, f"Expected horizontal detection labels: {label}")
                if msg:
                    self.diagnostic_warnings.append(msg)
                labels.append(dict(im_file=name, shape=shape, cls=lb[:, :1], bboxes=lb[:, 1:],
                                   segments=[], keypoints=keypoints, normalized=True, bbox_format="xywh"))
            return labels

    validator = RTDETRValidator(save_dir=output / "loader" / split, args={**settings, "split": split})
    dataset = ReadOnlyDataset(img_path=data[split], imgsz=settings["imgsz"], batch_size=settings["batch"],
                              augment=False, hyp=validator.args, rect=False, cache=False, data=data)
    require(len(dataset) > 0, f"Empty {split} split.")
    return dataset


def manifest(dataset):
    from ultralytics.data.utils import img2label_paths
    result = []
    for image, label in zip(dataset.im_files, img2label_paths(dataset.im_files)):
        image, label = Path(image).resolve(), Path(label).resolve()
        result.append({"image": str(image), "image_sha256": sha256(image), "label": str(label),
                       "label_sha256": sha256(label) if label.is_file() else None})
    require(len({r["image"] for r in result}) == len(result), "Duplicate images in split.")
    return result


def select_fixed(val_manifest, evidence, count=32):
    if evidence is None or not evidence.exists():
        rows = sorted(val_manifest, key=lambda r: r["image"].replace("\\", "/"))[:count]
        return {"origin": "new stable lexicographic val list; NOT claimed identical to old C20 evidence",
                "requested_evidence": str(evidence) if evidence else None, "rows": rows}
    require(evidence.is_file(), f"Evidence is not a file: {evidence}")
    with gzip.open(evidence, "rt", encoding="utf-8") as f:
        old = [json.loads(line) for line in f if line.strip()]
    require(len(old) == count, f"Expected {count} old evidence images, got {len(old)}; do not silently substitute.")
    selected = []
    for item in old:
        # Absolute path or an unambiguous filename+content identity permits a moved dataset.
        matches = [r for r in val_manifest if r["image"] == item["image"]]
        if not matches:
            name = item["image"].replace("\\", "/").rsplit("/", 1)[-1]
            matches = [r for r in val_manifest if Path(r["image"]).name == name
                       and r["image_sha256"] == item["image_sha256"]]
        require(len(matches) == 1, f"Old evidence cannot be resolved uniquely in val: {item['image']}")
        row = matches[0]
        require(row["image_sha256"] == item["image_sha256"] and row["label_sha256"] == item["label_sha256"],
                f"Old evidence image/label changed: {item['image']}")
        selected.append(row)
    require(len({r["image"] for r in selected}) == count, "Old evidence has duplicate images.")
    return {"origin": "restored old C20 evidence; every image and label hash verified", "evidence": str(evidence),
            "evidence_sha256": sha256(evidence), "rows": selected}


def metric_result(validator):
    import numpy as np
    validator.get_stats()
    m = validator.metrics.box
    ap = np.asarray(m.all_ap)
    require(ap.shape == (1, 10) and bool(np.isfinite(ap).all()), f"Expected finite one-class 10-IoU AP: {ap.shape}")
    return {"images": validator.seen, "precision": float(m.mp), "recall": float(m.mr),
            "AP50": float(m.map50), "AP75": float(ap[:, 5].mean()), "mAP50_95": float(m.map),
            "AP_by_class_and_IoU": ap.tolist(), "IoU_thresholds": np.linspace(.5, .95, 10).tolist(),
            "precision_recall_definition": "official DetMetrics operating point at smoothed best mean F1",
            "metric_implementation": "RTDETRValidator.update_metrics/get_stats -> native DetMetrics",
            "effective_args": vars(validator.args)}


def evaluate_model(name, weights, split, dataset, data, fixed, output, settings):
    import torch
    from ultralytics.data import build_dataloader
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.nn.modules.cbr import RTDETRDecoderCBR
    from ultralytics.nn.modules.cscef_v51 import CSCEFv51
    from ultralytics.nn.tasks import load_checkpoint
    from cbr_rescue_analysis import (AlignedValidator, MaskObserver, raw_tensor, cscef_observer,
                                     cbr_rows, write_csv, numeric_summary)

    folder = output / "evaluations" / f"{name}_{split}"
    folder.mkdir(parents=True, exist_ok=False)
    device = torch.device("cpu" if settings["device"] == "cpu" else "cuda:" + settings["device"])
    model, checkpoint = load_checkpoint(str(weights), device=device, fuse=False)
    del checkpoint
    model = model.float().eval()
    require(not model.training and all(not m.training for m in model.modules()), "Model is not entirely eval.")
    require(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()), "Expected unfused checkpoint with BatchNorm.")
    require(all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point()), "Expected FP32 parameters.")
    cbr = [(n, m) for n, m in model.named_modules() if isinstance(m, RTDETRDecoderCBR)]
    cs = [(n, m) for n, m in model.named_modules() if isinstance(m, CSCEFv51)]
    require(len(cbr) == int(name in {"C19", "C20"}), f"Unexpected CBR topology for {name}")
    require(len(cs) == int(name in {"C17", "C20"}), f"Unexpected CSCEF topology for {name}")
    require(len(model.names) == 1, "Expected one crack class.")
    paired = bool(cbr) and split == "val"
    routes = ("after", "before") if paired else ("after",)
    validators = {}
    for route in routes:
        for policy, cls in (("legacy", RTDETRValidator), ("aligned", AlignedValidator)):
            v = cls(save_dir=folder / f"{route}_{policy}", args={**settings, "split": split, "model": str(weights)})
            v.device, v.data, v.training, v.stride = device, data, False, model.stride
            v.iouv = v.iouv.to(device)
            v.init_metrics(model)
            validators[f"{route}_{policy}"] = v
    native = validators["after_legacy"]
    loader = build_dataloader(dataset, settings["batch"], 0, shuffle=False, rank=-1, drop_last=False, pin_memory=False)
    observer, cs_rows, query_rows, pairs, matching_images, parity = MaskObserver(settings["conf"]), [], [], [], [], []
    fixed_paths = {r["image"] for r in fixed["rows"]} if split == "val" else set()
    visited, sampled = [], set()
    try:
        with torch.inference_mode():
            for batch_i, batch in enumerate(loader):
                batch = native.preprocess(batch)
                images = [str(Path(p).resolve()) for p in batch["im_file"]]
                selected = [i for i, p in enumerate(images) if p in fixed_paths]
                sampled.update(images[i] for i in selected)
                visited.extend(images)
                # Check ordinary output before observing/fetching paired outputs. No fusion or AMP.
                check = batch_i == 0 or bool(selected)
                ordinary = raw_tensor(model.predict(batch["img"])).clone() if check else None
                hook_counts = {id(m): (len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()}
                context = cscef_observer(model, selected, images, cs_rows) if cs and selected else nullcontext()
                with context:
                    if paired:
                        pred, details = model.predict(batch["img"], cbr_diagnostics=True)
                    else:
                        pred, details = model.predict(batch["img"]), None
                require(all(hook_counts[id(m)] == (len(m._forward_hooks), len(m._forward_pre_hooks))
                            for m in model.modules()), "Observation hook leaked.")
                after = raw_tensor(pred).clone()
                require(after.shape[1:] == (300, 5), f"Unexpected raw shape: {after.shape}")
                if check:
                    error = float((after - ordinary).abs().max())
                    torch.testing.assert_close(after, ordinary, atol=1e-6, rtol=1e-5)
                    parity.append({"batch": batch_i, "images": images, "max_abs_error": error,
                                   "atol": 1e-6, "rtol": 1e-5, "hooks_removed": True})
                observer.observe(after, images)  # BEFORE native in-place scaling; scores are already sigmoid
                raw_routes = {"after": after}
                if paired:
                    torch.testing.assert_close(details["after"], after[..., :4], atol=0, rtol=0)
                    before = after.clone()
                    before[..., :4] = details["before"]
                    require(torch.equal(before[..., 4:], after[..., 4:]), "Classification scores changed across routes.")
                    raw_routes["before"] = before
                    for bi in selected:
                        pbatch = native._prepare_batch(bi, batch)
                        q, p, counts = cbr_rows({k: v[bi] for k, v in details.items()}, after[bi],
                                               pbatch["bboxes"] / settings["imgsz"], pbatch["cls"], images[bi])
                        query_rows.extend(q)
                        pairs.extend(p)
                        matching_images.extend(counts)
                for route, raw in raw_routes.items():
                    for policy in ("legacy", "aligned"):
                        validator = validators[f"{route}_{policy}"]
                        processed = validator.postprocess(raw.clone())  # native may mutate bbox view
                        validator.update_metrics(processed, batch)
                if batch_i % 10 == 0 or batch_i + 1 == len(loader):
                    print(f"{name} {split}: {len(visited)}/{len(dataset)} images; mask mismatches={observer.different_queries}", flush=True)
        require(visited == [str(Path(p).resolve()) for p in dataset.im_files], "Full split order/count mismatch.")
        require(sampled == fixed_paths, "Not all fixed val images were visited.")
        report = {"model": name, "split": split, "images": len(visited), "weights": str(weights),
                  "checkpoint_sha256": sha256(weights), "mode": "unfused_fp32", "paired_same_forward": paired,
                  "module_objects": {"cbr": [n for n, _ in cbr], "cscef": [n for n, _ in cs]},
                  "mask": observer.report(), "output_parity": parity,
                  "metrics": {key: metric_result(v) for key, v in validators.items()},
                  "native_preprocessing": "RTDETRDataset.load_image(rect_mode=False), build_transforms, native validator.preprocess",
                  "iou_and_max_det": "Recorded native settings; RTDETR postprocess has no NMS or extra max_det truncation; 300 queries"}
        if cs_rows:
            write_csv(folder / "cscef_fixed.csv", cs_rows)
            report["cscef"] = numeric_summary(cs_rows)
            report["cscef"]["confidence_meaning"] = "Structure confidence, NOT target classification probability."
        if paired:
            write_csv(folder / "cbr_queries_fixed.csv", query_rows)
            write_csv(folder / "cbr_pairs_fixed.csv", pairs)
            write_csv(folder / "cbr_matching_images.csv", matching_images)
            report["cbr"] = {"coordinates": "normalized image coordinates; horizontal bbox short side is not crack pixel width",
                "matching": "class-aware greedy descending BEFORE IoU, threshold=0, ties GT/query ascending; pairs held for AFTER",
                "low_iou_threshold": .5, "iou_change_tolerance": 1e-6, "edge_error_units": "normalized x or y; delta=after-before",
                "empty_handling": "no pairs for empty GT/no candidates; explicit counts in matching_images CSV",
                "center_units": "dx/dy and distance normalized to image; also dx/w and dy/h",
                "query_summaries": {scope: numeric_summary([r for r in query_rows if scope == "all" or r["score_ge_025"]],
                                                           excluded=("query", "class")) for scope in ("all", "score_ge_025")},
                "pair_summaries": {scope: numeric_summary([r for r in pairs if r["scope"] == scope],
                                                          excluded=("query", "gt")) for scope in ("all", "score_ge_025")}}
            from cbr_rescue_analysis import SIDES, distribution
            report["cbr"]["pooled_sides"] = {}
            for scope in ("all", "score_ge_025"):
                selected_rows = [r for r in query_rows if scope == "all" or r["score_ge_025"]]
                report["cbr"]["pooled_sides"][scope] = {
                    field: distribution([r[f"{side}_{field}"] for r in selected_rows for side in SIDES])
                    for field in ("abs_tanh", "saturated", "relative", "abs_relative", "positive", "negative", "zero")}
        write_json(folder / "report.json", report)
        return report
    finally:
        del model, loader, validators
        if device.type == "cuda":
            torch.cuda.empty_cache()


def conclusion(reports, policy):
    # Only val facts enter this decision. Test is restricted to evaluating the public mask defect.
    metric = lambda name, route: reports[f"{name}_val"]["metrics"][f"{route}_{policy}"]["mAP50_95"]
    c17 = metric("C17", "after")
    facts = {"policy": policy, "C17_val_mAP50_95": c17}
    for name in ("C19", "C20"):
        b, a = metric(name, "before"), metric(name, "after")
        facts[name] = {"before": b, "after": a, "after_minus_before": a - b,
                       "before_minus_C17": b - c17, "after_minus_C17": a - c17}
    c20 = facts["C20"]
    if c20["after_minus_before"] < -1e-6:
        advice = ("优先核查固定配对 CSV 中 C20 的误修正幅度和定位误差，判断是否足以支持唯一一次 CBR 约束调整。"
                  "当前仅凭 AP 下降或饱和率不能决定 rho 或宣称过度修正，尚不指定 CBR-v2 改法。")
    elif c20["after_minus_before"] > 1e-6 and c20["before_minus_C17"] < -1e-6:
        advice = ("CBR 末端仍有收益，但其输入预测弱于同口径 C17；优先只审查 CBR 训练连接，必要时另做有限梯度检查。"
                  "尚未证明梯度冲突，本轮不加入 detach 或训练补丁。")
    else:
        advice = "当前完整 val 未提供明确的末端补救线索；若固定样本也无可靠异常，建议停止这条组合路线并保留 C17。"
    return {"facts": facts, "single_priority": advice,
            "limitations": ["Fixed 32 images locate behavior; no causal proof from means, directions or saturation alone.",
                            "Before output is from the jointly trained checkpoint; it is not C17 or a retrained CBR ablation.",
                            "Test never enters recommendation. No hyperparameter scan or formal training.",
                            "FP32 unfused evaluation can differ from saved fused historical metrics.",
                            "No gradient paths tested; P3/query/main boxes/wh scaling/matching-quality targets remain distinct."]}


def compare_modules(reports):
    """Cross-checkpoint descriptive differences only; no causal labels or test inputs."""
    comparison = {"meaning": "Descriptive C20-minus-reference mean/median changes on the same fixed val list, not causal proof."}
    def differences(a, b):
        result = {}
        for key in a.keys() & b.keys():
            result[key] = {stat: None if a[key][stat] is None or b[key][stat] is None else b[key][stat] - a[key][stat]
                           for stat in ("mean", "median")}
        return result
    comparison["CSCEF_C20_minus_C17"] = differences(reports["C17_val"]["cscef"]["statistics"],
                                                    reports["C20_val"]["cscef"]["statistics"])
    comparison["CBR_C20_minus_C19"] = {
        scope: differences(reports["C19_val"]["cbr"]["query_summaries"][scope]["statistics"],
                           reports["C20_val"]["cbr"]["query_summaries"][scope]["statistics"])
        for scope in ("all", "score_ge_025")}
    return comparison


def run(args):
    output = args.output.resolve()
    require(not output.exists(), f"Preserve previous output: {output}")
    output.mkdir(parents=True)
    summary = {"status": "failed", "started_utc": datetime.now(timezone.utc).isoformat(), "reports": {}}
    input_hashes, dataset_manifests, state = {}, {}, None
    try:
        summary["runtime"] = import_runtime(output)
        from unittest.mock import patch
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import LOGGER
        handler = logging.FileHandler(output / "framework.log", encoding="utf-8")
        LOGGER.addHandler(handler)
        settings = {**SETTINGS, "device": args.device}
        summary["settings"] = settings
        state = source_state()
        require(not state["status"].strip(), "Diagnosis worktree must be clean and committed.")
        summary["source_before"] = state
        write_json(output / "source_before.json", state)
        (output / "code_diff.patch").write_bytes(git("diff", "--binary", BASE, "HEAD"))
        weights = {name: (getattr(args, name.lower()) or args.main / "runs/c_series" / run_name / "weights/best.pt").resolve()
                   for name, run_name in RUNS.items()}
        data_path = (args.data or args.main / "configs/crack_autodl.yaml").resolve()
        settings["data"] = str(data_path)
        init = (args.initialization or args.main / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt").resolve()
        inputs = {**weights, "data": data_path, "initialization": init}
        for name, path in inputs.items():
            require(path.is_file(), f"Required {name} file missing: {path}; locate it and use explicit --{name.lower()} argument.")
            input_hashes[str(path)] = sha256(path)
        require(input_hashes[str(weights["C20"])].lower() == C20_SHA, "C20 best SHA256 differs from verified checkpoint.")
        require(input_hashes[str(init)].lower() == INIT_SHA, "Unified initialization SHA256 mismatch (read only; never used for training).")
        # Font downloads are irrelevant to metrics and disabled; native path resolution is preserved.
        with patch("ultralytics.data.utils.check_font", lambda *a, **kw: None):
            data = check_det_dataset(str(data_path), autodownload=False)
        require(data["nc"] == 1 and data.get("test"), "Expected single-class data with explicit val/test.")
        summary["inputs"] = {k: {"path": str(p), "sha256": input_hashes[str(p)]} for k, p in inputs.items()}
        summary["resolved_data"] = data
        datasets = {split: build_dataset(data, split, output, settings) for split in ("val", "test")}
        for split, dataset in datasets.items():
            dataset_manifests[split] = manifest(dataset)
            write_json(output / f"{split}_manifest.json", {"split": split, "images": len(dataset),
                       "rows": dataset_manifests[split], "warnings": dataset.diagnostic_warnings})
        fixed = select_fixed(dataset_manifests["val"], args.evidence)
        require(len(fixed["rows"]) == 32, "Need 32 fixed val images; dataset too small.")
        write_json(output / "fixed_manifest.json", fixed)
        print(f"Fixed val sample: {fixed['origin']}; images={len(fixed['rows'])}", flush=True)
        if args.evidence and args.evidence.is_file():
            input_hashes[str(args.evidence.resolve())] = sha256(args.evidence)
        summary["fixed_selection"] = fixed["origin"]
        summary["test_count"] = {"actual": len(datasets["test"]), "historical": 864,
                                 "matches_historical": len(datasets["test"]) == 864}
        require(set(r["image"] for r in dataset_manifests["val"]).isdisjoint(r["image"] for r in dataset_manifests["test"]),
                "Val/test share an image path.")
        reports = summary["reports"]
        def evaluate(name, split):
            reports[f"{name}_{split}"] = evaluate_model(name, weights[name], split, datasets[split], data, fixed, output, settings)
            require(sha256(weights[name]) == input_hashes[str(weights[name])], f"Checkpoint changed: {name}")
        # Mandatory public-defect check: C20 full val and full test first.
        for name, split in (("C20", "val"), ("C20", "test"), ("C17", "val"), ("C19", "val")):
            evaluate(name, split)
        affected = any(r["mask"]["different_queries"] for r in reports.values())
        if affected:
            for name, split in (("C2", "val"), ("C2", "test"), ("C17", "test"), ("C19", "test")):
                evaluate(name, split)
        policy = "aligned" if affected else "legacy"
        summary["comparison_policy"] = policy
        summary["public_mask_affected"] = affected
        summary["history_policy"] = ("Keep old/new results separate; all C2/C17/C19/C20 val/test recomputed with aligned mask."
                                     if affected else "No mismatch observed in this unified mode; keep historical results and compare this run consistently.")
        summary["full_val_metrics"] = {key: value["metrics"] for key, value in reports.items() if value["split"] == "val"}
        summary["module_comparison"] = compare_modules(reports)
        summary["conclusion"] = conclusion(reports, policy)
        write_json(output / "full_val_metrics.json", summary["full_val_metrics"])
        write_json(output / "module_comparison.json", summary["module_comparison"])
        write_json(output / "mask_statistics.json", {k: r["mask"] for k, r in reports.items()})
        (output / "conclusion.md").write_text(json.dumps(summary["conclusion"]["facts"], indent=2) + "\n\n" +
            summary["conclusion"]["single_priority"] + "\n\n" + "\n".join(summary["conclusion"]["limitations"]) + "\n", encoding="utf-8")
        summary["status"] = "completed"
    except BaseException as error:
        summary["error"] = repr(error)
        summary["traceback"] = traceback.format_exc()
        raise
    finally:
        try:
            after = {p: sha256(p) if Path(p).is_file() else None for p in input_hashes}
            summary["input_hashes_before"], summary["input_hashes_after"] = input_hashes, after
            require(after == input_hashes, "A checkpoint/data/evidence input changed during diagnosis.")
            for split, rows in dataset_manifests.items():
                for row in rows:
                    require(sha256(row["image"]) == row["image_sha256"], f"Image changed: {row['image']}")
                    current = sha256(row["label"]) if Path(row["label"]).is_file() else None
                    require(current == row["label_sha256"], f"Label changed: {row['label']}")
            summary["dataset_content_unchanged"] = bool(dataset_manifests)
            if state:
                summary["source_after"] = source_state()
                require(summary["source_after"] == state, "Code changed during diagnosis.")
        except BaseException as error:
            summary["status"] = "failed"
            summary["integrity_error"] = repr(error)
            raise
        finally:
            summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
            summary["artifacts"] = sorted({"summary.json", "console.log", *(
                p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
                and "runtime_config" not in p.relative_to(output).parts and p.suffix in {".json", ".csv", ".md", ".txt", ".log", ".patch"})})
            write_json(output / "summary.json", summary)
            if "handler" in locals():
                LOGGER.removeHandler(handler)
                handler.close()


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        self.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--main", type=Path, default=Path("/root/autodl-tmp/projects/Crack_RTDETR"))
    p.add_argument("--output", type=Path, required=True)
    for key in ("data", "initialization", "c2", "c17", "c19", "c20", "evidence"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--device", default="0", choices=("0", "cpu"))
    p = sub.add_parser("pack")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        package(args.input, args.output)
        return
    # A separate parent log becomes the output's console.log after run has made its new directory.
    require(not args.output.exists(), f"Preserve prior output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_name(args.output.name + ".console.log")
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                run(args)
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:
            log.flush()
            if args.output.is_dir():
                (args.output / "console.log").write_bytes(log_path.read_bytes())


if __name__ == "__main__":
    main()
