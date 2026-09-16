"""Explicit best/EMA val or test with the original CBR + LIF evaluation settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from init_trc_v1 import require, runtime, sha256, verify_model, is_added
from train_trc_v1 import read_plan, verify_plan, atomic_json, status, dataset_inventory
from c19_lif_v1_results import EVAL, POLICY, postprocess
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.data.utils import check_det_dataset


def evaluate(plan_path, split="test", output=None):
    plan_path = Path(plan_path).resolve()
    plan = verify_plan(read_plan(plan_path))
    require(split in {"val", "test"}, "Expected val/test")
    require(status(plan_path)["status"] in {"COMPLETED_200_EPOCHS", "EARLY_STOPPED_PATIENCE"},
            "Explicit evaluation requires successfully completed training")
    weights = Path(plan["args"]["save_dir"]) / "weights/best.pt"
    data = Path(plan["args"]["data"])
    output = Path(output).resolve() if output else plan_path.parent / ("evaluation_" + split)
    require(weights.is_file(), "Selected training best.pt missing")
    require(not output.exists(), "Existing evaluation protected; no name2/name3")
    resolved = check_det_dataset(str(data), autodownload=False)
    frozen = json.loads((plan_path.parent / "dataset_inventory.json").read_text(encoding="utf-8"))
    require(dataset_inventory(Path(resolved["path"])) == frozen, "Dataset split paths/labels changed since training")
    settings = dict(EVAL, data=str(data), split=split, device=plan["args"]["device"], plots=True,
                    save_json=False, save_txt=False, project=str(output), name="plots", exist_ok=False)
    digest, data_digest = sha256(weights), sha256(data)
    report = dict(status="FAILED", variant=plan["variant"], split=split, runtime=runtime(),
                  checkpoint=str(weights), checkpoint_sha256=digest, data_sha256=data_digest,
                  settings=settings, policy=POLICY, plan_sha256=sha256(plan_path),
                  selection="Original training-val best.pt; native loader uses EMA when present, otherwise stored model",
                  precision_recall_policy="each model's maximum-F1 working point", per_image_predictions="not exported")
    output.mkdir(parents=True, exist_ok=False)
    try:
        model = RTDETR(str(weights))
        verify_model(model.model, plan["variant"], zero=False)
        require(model.model.model[-1].nc == 1, "Evaluation requires task nc=1")
        learned = {n: p.detach().float().cpu().clone() for n, p in model.model.named_parameters() if is_added(n)}
        report["native_checkpoint_selection"] = "ema" if model.ckpt.get("ema") is not None else "model"

        class PinnedValidator(RTDETRValidator):
            def init_metrics(self, backend):
                super().init_metrics(backend)
                require(not self.training and not self.args.half, "Original independent FP32 evaluation required")
                require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()),
                        "Effective evaluation settings changed")
                require(self.save_dir.resolve() == (output / "plots").resolve(), "Evaluation directory redirected")
                report["actual_settings"] = vars(self.args).copy()

            def postprocess(self, preds):
                selected, affected = postprocess(preds, self.args.imgsz, self.args.conf)
                report["historical_mask_affected_images"] = report.get("historical_mask_affected_images", 0) + affected
                return selected

        metrics = model.val(validator=PinnedValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Unexpected or nonfinite AP array")
        parameters = dict(model.model.named_parameters())
        require(all(torch.equal(v, parameters[n].detach().float().cpu()) for n, v in learned.items()),
                "Evaluation/fusion changed learned TRC parameters")
        require(sha256(weights) == digest and sha256(data) == data_digest, "Checkpoint/data changed during evaluation")
        summary = dict(precision=float(metrics.box.mp), recall=float(metrics.box.mr), AP50=float(metrics.box.map50),
                       AP75=float(ap[:, 5].mean()), AP50_95=float(metrics.box.map))
        require(all(np.isfinite(v) for v in summary.values()), "Nonfinite evaluation metrics")
        report.update(status="PASSED", **summary, ap_by_class=ap.tolist(),
                      ap_iou_thresholds=[round(.5 + i * .05, 2) for i in range(10)],
                      speed_ms_per_image=metrics.speed, images=frozen[split]["images"],
                      ground_truth=frozen[split]["boxes"], split_inventory=frozen[split],
                      learned_trc_unchanged=True, speed_scope="native validator timing; not a separate latency benchmark")
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        atomic_json(output / "metrics.json", report)
        pointer = plan_path.parent / ("evaluation_" + split + "_pointer.json")
        atomic_json(pointer, dict(output=str(output), metrics_sha256=sha256(output / "metrics.json"),
                                 checkpoint_sha256=digest, status=report["status"], split=split))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.plan, args.split, args.output)
