"""First-forward FP32 exports and offline NCR analysis; no implicit inference in pack/status."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ncr_v1_common import COUNTS, EVAL, PARENT_MAP, POLICY, environment, now, read, require, sha, source_identity, write
from c19_lif_v1_results import postprocess, image_record  # exact parent corrected protocol
from init_c19_lif_v1 import verify_model
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.utils.ncr import distribution, ncr_terms
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.metrics import box_iou, smooth
from ultralytics.utils.ops import xyxy2xywh


class NCRExportValidator(RTDETRValidator):
    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = Path(output)
        self.export_counts = dict(images=0, predictions=0, ground_truth=0)
        self.export_seen = set()

    def init_metrics(self, model):
        super().init_metrics(model)
        require(not self.training and not self.args.half, "Formal evaluation must be independent FP32")
        require(all(getattr(self.args, k) == v for k, v in EVAL.items()), "Evaluation recipe drift")
        write(self.output/"actual_eval_args.json", vars(self.args))
        self.expected_images = {Path(f).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                                for f in self.dataloader.dataset.im_files}

    def postprocess(self, preds):
        selected, _ = postprocess(preds, self.args.imgsz, self.args.conf)
        full, _ = postprocess(preds, self.args.imgsz, -float("inf"))
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        # PyTorch RT-DETR returns (probabilities, (decoder boxes, logits, encoder boxes, logits, DN meta)).
        require(isinstance(preds, (list, tuple)) and len(preds) == 2 and len(preds[1]) == 5,
                "All-query export requires native PyTorch RT-DETR outputs")
        logits = preds[1][1][-1]
        for i, (keep, all_queries) in enumerate(zip(selected, full)):
            order = raw[i, :, 4:].max(-1).values.argsort(descending=True)
            require(len(order) == 300, "Expected all 300 ordinary queries")
            keep["_all_queries"] = (all_queries, order, raw[i, order, :4], logits[i, order])
        return selected

    def _process_batch(self, preds, batch):
        cached = preds.pop("_metric_tp", None)
        return cached if cached is not None else super()._process_batch(preds, batch)

    def update_metrics(self, preds, batch):
        with gzip.open(self.output/"queries_gt.jsonl.gz", "at", encoding="utf-8") as stream:
            for i, pred in enumerate(preds):
                all_queries, order, normalized, logits = pred.pop("_all_queries")
                gt = self._prepare_batch(i, batch)
                row = image_record(all_queries, gt, self.data["path"], self.args.conf)
                row.update(input_size_hw=list(map(int, gt["imgsz"])),
                           transform=dict(kind="RTDETR stretch", pad_xy=[0, 0],
                                          gain_xy=[gt["imgsz"][1]/gt["ori_shape"][1], gt["imgsz"][0]/gt["ori_shape"][0]]))
                for record, index, box, logit, input_box in zip(row["predictions"], order.tolist(), normalized.tolist(), logits.tolist(), all_queries["bboxes"].tolist()):
                    record.update(query_index=index, normalized_cxcywh=box, class_logits=logit, input_xyxy=input_box)
                normalized_gt = batch["bboxes"][batch["batch_idx"] == i].tolist()
                for record, box, input_box in zip(row["ground_truth"], normalized_gt, gt["bboxes"].tolist()):
                    record.update(normalized_cxcywh=box, input_xyxy=input_box)
                native = super()._process_batch(pred, gt)
                pred["_metric_tp"] = native  # the native update reuses these very same TP flags
                row["metric_query_indices"] = order[all_queries["conf"] > self.args.conf].tolist()
                row["metric_tp_iou50_to95"] = native["tp"].tolist()
                require(row["image"] not in self.export_seen, "Duplicate image in export")
                self.export_seen.add(row["image"])
                for key, count in (("images", 1), ("predictions", len(order)), ("ground_truth", len(row["ground_truth"]))):
                    self.export_counts[key] += count
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n")
        return super().update_metrics(preds, batch)

    def finalize_metrics(self):
        require(self.export_seen == self.expected_images, "Incomplete split export")
        return super().finalize_metrics()


def eval_identity(weights, data, split, folder, manifest):
    require(sha(data) == manifest["data"]["yaml_sha256"], "Frozen data YAML differs from training manifest")
    settings = dict(DEFAULT_CFG_DICT, **{})
    settings.update(EVAL, task="detect", mode="val", model=str(weights), data=str(data), split=split,
                    project=str(folder), name="plots", exist_ok=False, save_dir=str(folder/"plots"))
    return dict(best_sha256=sha(weights), split=split, settings=settings, policy=POLICY,
                evaluator_sha256=source_identity()["code_sha256"], data=manifest["data"])


def verify_eval(report, expected, folder):
    require(report.get("status") == "PASSED" and report.get("identity") == expected, "Evaluation identity/status differs")
    stream = Path(folder)/"queries_gt.jsonl.gz"
    require(stream.is_file(), "Successful evaluation lacks required query export; no automatic full re-inference")
    require(sha(stream) == report["export_sha256"], "Export checksum differs")
    require(sha(Path(folder)/"actual_eval_args.json") == report["actual_args_sha256"], "Actual eval args changed")
    require((report["images"], report["ground_truth"]) == COUNTS[expected["split"]], "Wrong formal split counts")
    require(report["predictions"] == report["images"]*300, "Missing ordinary queries")
    require(report["split_paths_sha256"] == expected["data"]["splits"][expected["split"]]["split_paths_sha256"], "Evaluated image IDs differ from training manifest")
    return report


def evaluate(weights, data, split, output, manifest):
    """At most one new full-split forward; reuse only identical successful evidence."""
    output = Path(output)
    # A stable location makes complete settings comparable across finish invocations.
    expected = eval_identity(weights, data, split, output, manifest)
    existing = read(output/"metrics.json", {})
    if existing.get("status") == "PASSED":
        return verify_eval(existing, expected, output)
    if output.exists():
        # Preserve failed/incomplete evidence, then permit one explicit finish retry.
        backup = output.with_name(output.name+"_failed_"+now().replace(":", "").replace(".", "_"))
        output.rename(backup)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", identity=expected, started=now(), environment=environment(), exit_code=1)
    write(output/"metrics.json", report)
    try:
        model = RTDETR(str(weights))
        verify_model(model.model)
        # Exactly Model.val's native construction/call, retaining the validator's export counters.
        validator = NCRExportValidator(args=expected["settings"], _callbacks=model.callbacks, output=output)
        validator(model=model.model)
        metrics = validator.metrics
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid AP evidence")
        p, r = float(metrics.box.mp), float(metrics.box.mr)
        raw = dict(Precision=p, Recall=r, F1=2*p*r/(p+r) if p+r else 0.,
                   AP50=float(metrics.box.map50), AP75=float(ap[0, 5]), mAP50_95=float(metrics.box.map))
        # Same library maximum-F1 PR point; descriptive reporting, no deployment threshold selection.
        best_i = int(smooth(metrics.box.f1_curve.mean(0), .1).argmax())
        report.update(status="PASSED", exit_code=0, finished=now(), raw=raw,
                      percent={k: v*100 for k, v in raw.items()}, classes=["all", "crack"],
                      ap_iou_thresholds=[round(.5+i*.05, 2) for i in range(10)], ap_raw=ap[0].tolist(),
                      ap_percent=(ap[0]*100).tolist(), native_PR_conf=float(metrics.box.px[best_i]),
                      parent_map_delta_pp=(raw["mAP50_95"]-PARENT_MAP[split])*100,
                      best=str(weights), export_sha256=sha(output/"queries_gt.jsonl.gz"),
                      split_paths_sha256=hashlib.sha256("\n".join(sorted(validator.export_seen)).encode()).hexdigest(),
                      actual_args_sha256=sha(output/"actual_eval_args.json"), **validator.export_counts)
        require(sha(weights) == expected["best_sha256"], "Best changed during evaluation")
        require(sha(data) == manifest["data"]["yaml_sha256"], "Data YAML changed during evaluation")
        verify_eval(report, expected, output)
    except BaseException as error:
        report.update(status="FAILED", error=repr(error), exit_code=1)
        raise
    finally:
        write(output/"metrics.json", report)
    return report


def native_pairs(iou, threshold):
    """Exact non-scipy native matching order, exposing GT IDs only for offline summaries."""
    pairs = np.array(np.nonzero(iou >= threshold)).T
    if len(pairs) > 1:
        pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs


def analyze(folder):
    """Offline only: native detection TP recall/errors + separate training-Hungarian mechanism diagnostics."""
    folder = Path(folder); report = read(folder/"metrics.json")
    require(report and report["status"] == "PASSED", "Offline analysis requires successful evaluation")
    stream = folder/"queries_gt.jsonl.gz"
    require(sha(stream) == report["export_sha256"], "Offline export changed")
    cached = read(folder/"analysis.json", {})
    if cached.get("export_sha256") == report["export_sha256"] and cached.get("code") == source_identity()["code_sha256"]:
        return cached
    matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
    groups = {name: dict(gt=0, recalled_iou50=0, recalled_iou75=0, relative_center=[], relative_size=[])
              for name in ("lt4px", "4to16px", "16to32px", "ge32px")}
    gates, centers, mechanism_sizes = [], [], []
    errors = dict(predictions_at_native_PR_conf=0, tp50=0, fp50=0, fn50=0, fp_best_iou_lt_half=0, duplicate_or_competing_fp=0)
    raw_sum, matches = 0., 0
    with gzip.open(stream, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            pred = torch.tensor([p["normalized_cxcywh"] for p in row["predictions"]])
            logits = torch.tensor([p["class_logits"] for p in row["predictions"]])
            h, w = row["original_size_hw"]
            gt = torch.tensor([g["normalized_cxcywh"] for g in row["ground_truth"]]).reshape(-1, 4)
            # This matching is ONLY for NCR geometry, never TP/AP.
            pi, gi = matcher(pred.unsqueeze(0), logits.unsqueeze(0), gt, torch.zeros(len(gt), dtype=torch.long), [len(gt)])[0]
            if len(gi):
                raw, gate, u, scale = ncr_terms(pred[pi], gt[gi], row["input_size_hw"])
                raw_sum += float(raw)*len(gi); matches += len(gi)
                gates.append(gate); centers.append(u.abs())
                mechanism_sizes.append((gt[gi, 2:] * torch.tensor(row["input_size_hw"])[[1,0]]).amin(-1))
            metric = [p for p in row["predictions"] if p["used_for_metrics"]]
            pb = torch.tensor([p["input_xyxy"] for p in metric]).reshape(-1, 4)
            gb = torch.tensor([g["input_xyxy"] for g in row["ground_truth"]]).reshape(-1, 4)
            ious = box_iou(gb, pb).numpy()
            saved = np.array(row["metric_tp_iou50_to95"], dtype=bool).reshape(len(metric), 10)
            # Validate AP evidence against the exact native rule without another network forward.
            for index, threshold in enumerate(torch.linspace(.5, .95, 10).tolist()):
                reproduced = np.zeros(len(metric), dtype=bool)
                pairs = native_pairs(ious, threshold)
                if len(pairs): reproduced[pairs[:,1]] = True
                require(np.array_equal(reproduced, saved[:, index]), "Offline native TP replay differs")
            high = np.array([p["score"] >= report["native_PR_conf"] for p in metric], dtype=bool)
            pairs50, pairs75 = native_pairs(ious, .5), native_pairs(ious, .75)
            pairs50 = pairs50[high[pairs50[:,1]]] if len(pairs50) else pairs50
            pairs75 = pairs75[high[pairs75[:,1]]] if len(pairs75) else pairs75
            true_queries = set(pairs50[:,1].tolist())
            errors["predictions_at_native_PR_conf"] += int(high.sum())
            errors["tp50"] += len(pairs50); errors["fp50"] += int(high.sum())-len(pairs50); errors["fn50"] += len(gt)-len(pairs50)
            for index in np.flatnonzero(high):
                if index not in true_queries:
                    key = "duplicate_or_competing_fp" if len(gt) and ious[:,index].max() >= .5 else "fp_best_iou_lt_half"
                    errors[key] += 1
            for index, g in enumerate(gt):
                short = float((g[2:]*torch.tensor(row["input_size_hw"])[[1,0]]).min())
                name = "lt4px" if short < 4 else "4to16px" if short < 16 else "16to32px" if short < 32 else "ge32px"
                group = groups[name]; group["gt"] += 1
                group["recalled_iou50"] += int(index in pairs50[:,0]); group["recalled_iou75"] += int(index in pairs75[:,0])
                matched = pairs50[pairs50[:,0] == index]
                if len(matched):
                    ih, iw = row["input_size_hw"]
                    p = xyxy2xywh(pb[int(matched[0,1])].reshape(1,4))[0]/torch.tensor([iw,ih,iw,ih])
                    floor = torch.tensor([4/row["input_size_hw"][1], 4/row["input_size_hw"][0]])
                    group["relative_center"].extend(((p[:2]-g[:2]).abs()/torch.maximum(g[2:],floor)).tolist())
                    group["relative_size"].extend(((p[2:]-g[2:]).abs()/torch.maximum(g[2:],floor)).tolist())
    for group in groups.values():
        for field in ("relative_center", "relative_size"):
            group[field] = distribution(torch.tensor(group[field]))
        for level in (50, 75):
            group[f"recall{level}"] = group[f"recalled_iou{level}"]/group["gt"] if group["gt"] else None
    gate = torch.cat(gates) if gates else torch.empty(0,2)
    center = torch.cat(centers) if centers else torch.empty(0,2)
    sizes = torch.cat(mechanism_sizes) if mechanism_sizes else torch.empty(0)
    mechanism_groups = {name: distribution(center[mask]) for name, mask in
                        (("lt4px", sizes<4), ("4to16px", (sizes>=4)&(sizes<16)),
                         ("16to32px", (sizes>=16)&(sizes<32)), ("ge32px", sizes>=32))}
    result = dict(status="PASSED", export_sha256=report["export_sha256"], code=source_identity()["code_sha256"],
                  method="Native detection TP replay; short-side recall/errors at native PR working point. No size-stratified AP claim.",
                  groups=groups, errors=errors, mechanism=dict(matching="training-style Hungarian, not detection TP", matches=matches,
                  active_fraction_xy=(gate>0).float().mean(0).tolist() if matches else [0.,0.],
                  h_xy=[distribution(gate[:,a]) for a in (0,1)], relative_center_by_short_side=mechanism_groups,
                  raw_per_match=raw_sum/max(matches,1)), limitation="Geometry/center alignment does not establish improved detection.")
    write(folder/"analysis.json", result)
    return result


def display(report):
    split = report["identity"]["split"]
    print(f"{split} all/crack: images={report['images']} GT={report['ground_truth']} exit={report['exit_code']}", flush=True)
    print(" | ".join(f"{k}={v:.6f}%" for k,v in report["percent"].items()), flush=True)
    print("AP50:0.05:0.95:", " ".join(f"{v:.6f}%" for v in report["ap_percent"]), flush=True)
    print(f"Delta parent mAP: {report['parent_map_delta_pp']:+.6f} pp; {POLICY}", flush=True)
    print("best:", report["best"], "SHA256:", report["identity"]["best_sha256"], flush=True)


def metrics_page(output):
    rows = []
    for split in ("val", "test"):
        result = read(Path(output)/("evaluation_"+split)/"metrics.json", {})
        if result.get("status") == "PASSED":
            display(result)
            rows.append("| "+split+" all/crack | "+" | ".join(f"{result['percent'][k]:.6f}" for k in
                        ("Precision", "Recall", "F1", "AP50", "AP75", "mAP50_95"))+f" | {result['parent_map_delta_pp']:+.6f} |")
    text = "# NCR v1 independent FP32 metrics\n\nPercent units; Precision is not Accuracy. F1 uses the same P/R point.\n\n"
    text += "| split | Precision | Recall | F1 | AP50 | AP75 | mAP50–95 | delta pp |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    text += "\n".join(rows)+"\n\nFull precision, AP grid, checkpoint/protocol identity and offline diagnostics are in each evaluation directory.\n"
    (Path(output)/"metrics.md").write_text(text, encoding="utf-8")
