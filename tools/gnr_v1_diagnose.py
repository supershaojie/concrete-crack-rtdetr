"""Bounded mother phenomenon probe; no optimizer, no test split, no AP claims."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import time

from gnr_v1_common import OUT, read_json, write_json, require, sha256, now

MOTHER_BEST_SHA = "24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa"


def diagnose(bind, folder, best, budget=900, val_images=64, train_batches=8):
    import torch
    from ultralytics import RTDETR
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.rtdetr.val import RTDETRDataset
    from ultralytics.models.rtdetr.gnr_model import forward_with_features, targets_from_batch, criterion_inputs
    from ultralytics.models.rtdetr.gnr_loss import GNRDetectionLoss, summarize
    from ultralytics.utils.torch_utils import init_seeds
    from ultralytics.utils.patches import torch_load
    from init_c19_lif_v1 import verify_model

    folder, best = Path(folder), Path(best)
    started = time.monotonic()
    require(0 < val_images <= 64 and 0 < train_batches <= 8 and 0 < budget <= 900, "Diagnosis bounds exceeded")
    report = {"status": "PENDING", "binding": bind, "best": str(best), "expected_best_sha256": MOTHER_BEST_SHA,
              "budget_seconds": budget, "limits": {"val_images": val_images, "train_batches": train_batches},
              "seed": 42, "splits": {}, "time": now(), "interpretation": "local last-Linear negative strength only; no mAP or whole-network gradient conclusion"}

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(folder / "diagnose.json", report)
        lines = [f"# GNR 母版诊断：{report['status']}", "", report["interpretation"], ""]
        for split, value in report["splits"].items():
            total = value.get("summary", {})
            lines.append(f"- {split}: images={len(value['images'])}, batches={len(value['batches'])}, changed={total.get('changed')}, removed_negative_strength_fraction={total.get('removed_negative_strength_fraction')}")
        if "reason" in report:
            lines += ["", report["reason"]]
        (folder / "diagnose.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    folder.mkdir(parents=True, exist_ok=True)
    if not best.is_file():
        report["reason"] = "Missing verified mother best checkpoint; prepare/code delivery remain available"
        save()
        return report
    require(sha256(best) == MOTHER_BEST_SHA, "Mother best hash differs from the successful archived mother; no fallback to last/init/other experiment")
    plan = read_json(OUT / "prepare.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)
    checkpoint = torch_load(best, map_location="cpu")
    selected_epoch = checkpoint.get("epoch")
    results = checkpoint.get("train_results") or {}
    # Stripped checkpoints use epoch=-1; recover the val-selected epoch from their archived training rows.
    if selected_epoch == -1 and results:
        key = next((k for k in results if k.strip() == "metrics/mAP50-95(B)"), None)
        epochs = next((v for k, v in results.items() if k.strip() == "epoch"), None)
        if key and epochs:
            import numpy as np
            selected_epoch = int(epochs[int(np.asarray(results[key]).argmax())]) - 1
    report["checkpoint"] = {"sha256": MOTHER_BEST_SHA, "stored_epoch": checkpoint.get("epoch"), "selected_epoch_zero_based": selected_epoch,
                            "epoch_source": "checkpoint epoch or archived training val maximum", "train_args": checkpoint.get("train_args")}
    require(selected_epoch is not None and selected_epoch >= 0, "Cannot verify selected mother best epoch")
    resolved = check_det_dataset(plan["data_yaml"], autodownload=False)
    require(resolved["nc"] == 1, "nc mismatch")
    try:
        for split in ("val", "train"):
            init_seeds(42, deterministic=True)
            # Separate instances prevent training-mode BN/DN/RNG from affecting val or formal training.
            model = RTDETR(str(best)).model.float().to(device)
            verify_model(model)
            model.train(split == "train")
            cfg = get_cfg(overrides=deepcopy(plan["args"]))
            dataset = RTDETRDataset(img_path=resolved[split], imgsz=640, batch_size=16, augment=split == "train",
                                   hyp=cfg, rect=False, cache=None, single_cls=False, data=resolved, fraction=1.)
            if split == "val":
                indices = sorted(random.Random(42).sample(range(len(dataset)), min(val_images, len(dataset))))
                view = torch.utils.data.Subset(dataset, indices)
                view.collate_fn = dataset.collate_fn
                loader = build_dataloader(view, batch=16, workers=0, shuffle=False, rank=-1)
                planned = [str(dataset.im_files[i]) for i in indices]
                limit = (len(indices) + 15) // 16
            else:
                loader = build_dataloader(dataset, batch=16, workers=cfg.workers, shuffle=True, rank=-1)
                planned, limit = None, train_batches
            value = {"images": [], "planned_val_images": planned, "batches": [], "complete": False, "summary": {}}
            report["splits"][split] = value
            distributions = {k: [] for k in ("a", "R", "w", "positive_cosine", "underfit")}
            criterion = GNRDetectionLoss(nc=1, use_vfl=True, aux_loss=False)
            for index, batch in enumerate(loader):
                if index >= limit or time.monotonic() - started >= budget:
                    break
                batch["img"] = batch["img"].to(device).float() / 255
                targets = targets_from_batch(batch)
                with torch.no_grad():
                    raw, h = forward_with_features(model, batch["img"], targets)
                    inputs, h, _, _, _ = criterion_inputs(raw, h)
                    _, details = criterion(inputs, targets, features=h, epoch=20, return_details=True)
                stats = summarize(details)
                positive_count = int((~details["negative"]).sum())
                positive_underfit = int(((inputs[1][-1, ..., 0].sigmoid() < details["quality"]) & ~details["negative"]).sum())
                stats.update(matched_positives=positive_count, underfit_positives=positive_underfit)
                value["images"].extend(map(str, batch["im_file"]))
                value["batches"].append(stats)
                grouped = details["ownership"] >= 0
                for key in distributions:
                    distributions[key].extend(details[key][grouped].cpu().tolist())
                totals = {k: sum(row[k] for row in value["batches"]) for k in
                          ("negatives", "grouped", "changed", "all_negative_strength", "grouped_negative_strength", "removed_strength", "matched_positives", "underfit_positives", "unmatched_gt_negative_queries")}
                totals["removed_negative_strength_fraction"] = totals["removed_strength"] / totals["all_negative_strength"] if totals["all_negative_strength"] else None
                totals["removed_grouped_strength_fraction"] = totals["removed_strength"] / totals["grouped_negative_strength"] if totals["grouped_negative_strength"] else None
                totals["p_pos_below_q_fraction"] = totals["underfit_positives"] / totals["matched_positives"] if totals["matched_positives"] else None
                totals["strongest_retained"] = all(row["strongest_retained"] for row in value["batches"])
                totals["quantiles"] = {k: torch.tensor(v).quantile(torch.tensor([0., .25, .5, .75, 1.])).tolist() if v else None for k, v in distributions.items()}
                totals["zero_denominator_note"] = "无可计量负梯度强度" if not totals["all_negative_strength"] else None
                value["summary"] = totals
                save()
            value["complete"] = len(value["batches"]) == limit
            del model, loader, dataset
            save()
        full = val_images == 64 and train_batches == 8 and all(v["complete"] for v in report["splits"].values()) and len(report["splits"]) == 2
        changed = sum(v["summary"].get("changed", 0) for v in report["splits"].values())
        report["coverage_complete"] = full
        report["status"] = "ZERO_INTERVENTION" if full and changed == 0 else "REVIEW" if changed else "PENDING"
        report["reason"] = "Complete fixed sample has zero intervention; long training is not recommended" if report["status"] == "ZERO_INTERVENTION" else "Review raw activation/strength and coverage; no automatic benefit threshold or accuracy claim"
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="RESOURCE_ERROR", reason=repr(error), coverage_complete=False)
        raise
    except BaseException as error:
        report.update(status="IMPLEMENTATION_ERROR", reason=repr(error), coverage_complete=False)
        raise
    finally:
        save()
    return report
