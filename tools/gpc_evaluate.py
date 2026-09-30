"""Independent corrected FP32 evaluation with original query IDs and offline AP data."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.rtdetr.gpc import finite
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import init_seeds

PROTOCOL = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640,batch=16,workers=0,half=False,conf=0.001,iou=0.7,max_det=300,
            augment=False,rect=False,seed=42,device="0",plots=True,save_json=False,save_txt=False)
REFERENCE = dict(val=dict(mAP50_95=0.52454272),
                 test=dict(precision=.860239,recall=.835359,F1=.847617,AP50=.891997,AP75=.540024,mAP50_95=.52200902))


def corrected_predictions(predictions, imgsz, conf):
    # Same corrected sorted-mask formula as parent c19_lif_v1_results.postprocess.
    # Added query IDs travel with their row and never affect the formal confidence mask.
    predictions = predictions[0] if isinstance(predictions,(list,tuple)) else predictions
    finite("evaluation outputs",predictions)
    output=[]
    for pred in predictions:
        score,cls=pred[:,4:].max(-1)
        order=score.argsort(descending=True)
        boxes=ops.xywh2xyxy(pred[:,:4] * imgsz)
        full=dict(bboxes=boxes[order],conf=score[order],cls=cls[order],query_index=order)
        mask=full["conf"] > conf
        output.append({**{k:v[mask] for k,v in full.items() if k!="query_index"},"_all_queries":full})
    return output


class GPCEvidenceValidator(RTDETRValidator):
    """Constructor is explicit so no transient closure class enters a checkpoint."""
    def __init__(self,*args,evidence_dir,**kwargs):
        self.evidence_dir=Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True,exist_ok=True)
        self.export_seen=set()
        self.export_counts=dict(images=0,queries=0,ground_truth=0,metric_predictions=0)
        super().__init__(*args,**kwargs)

    def init_metrics(self,model):
        super().init_metrics(model)
        if self.training or self.args.half:
            raise RuntimeError("Independent FP32 evaluation only")
        for key,value in EVAL.items():
            if type(getattr(self.args,key)) is not type(value) or getattr(self.args,key)!=value:
                raise RuntimeError(f"Evaluation recipe differs: {key}")
        self.export_expected={str(Path(p).resolve()) for p in self.dataloader.dataset.im_files}

    def postprocess(self,preds):
        return corrected_predictions(preds,self.args.imgsz,self.args.conf)

    def update_metrics(self,preds,batch):
        with gzip.open(self.evidence_dir/"predictions_gt.jsonl.gz","at",encoding="utf-8") as stream:
            for i,pred in enumerate(preds):
                all_queries=pred.pop("_all_queries")
                gt=self._prepare_batch(i,batch)
                path=str(Path(gt["im_file"]).resolve())
                if path in self.export_seen or len(all_queries["conf"])!=300:
                    raise RuntimeError("Duplicate image or incomplete query export")
                self.export_seen.add(path)
                h,w=map(int,gt["ori_shape"]);ih,iw=map(int,gt["imgsz"])
                scale=torch.tensor([w/iw,h/ih,w/iw,h/ih])
                boxes=all_queries["bboxes"].detach().float().cpu()*scale
                gt_boxes=gt["bboxes"].detach().float().cpu()*scale
                finite("evaluation GT",gt_boxes)
                row=dict(image=str(Path(path).relative_to(Path(self.data["path"]).resolve()).as_posix()),
                         original_size_hw=[h,w],box_format="xyxy",coordinate_space="original_image_pixels",
                         inverse_map="RT-DETR stretch; no rounding/clamping",gt_class=gt["cls"].cpu().tolist(),
                         gt_boxes=gt_boxes.tolist(),pred_boxes=boxes.tolist(),
                         scores=all_queries["conf"].cpu().tolist(),classes=all_queries["cls"].cpu().tolist(),
                         original_query_index=all_queries["query_index"].cpu().tolist(),
                         used_for_metrics=(all_queries["conf"]>self.args.conf).cpu().tolist())
                stream.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
                self.export_counts["images"]+=1
                self.export_counts["queries"]+=len(boxes)
                self.export_counts["ground_truth"]+=len(gt_boxes)
                self.export_counts["metric_predictions"]+=len(pred["conf"])
        super().update_metrics(preds,batch)

    def get_stats(self):
        # Save before native metrics.process/clear_stats. These are the ten-IoU AP/PR inputs.
        arrays={key:np.concatenate(values,axis=0) for key,values in self.metrics.stats.items()}
        np.savez_compressed(self.evidence_dir/"ap_pr_inputs.npz",**arrays,iou_thresholds=self.iouv.cpu().numpy())
        return super().get_stats()

    def finalize_metrics(self):
        if self.export_seen!=self.export_expected:
            raise RuntimeError("Incomplete evaluation image coverage")
        return super().finalize_metrics()


def evaluate(best,data,split,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    init_seeds(42,deterministic=True)
    model=RTDETR(str(best))
    validator=GPCEvidenceValidator(args=dict(EVAL,data=str(data),split=split,model=str(best),
                                           project=str(output),name="plots",exist_ok=False),evidence_dir=output)
    validator(model=model.model.float())
    metrics=validator.metrics
    ap=np.asarray(metrics.box.all_ap)
    if ap.shape!=(1,10) or not np.isfinite(ap).all():
        raise RuntimeError("Nonfinite/unexpected AP array")
    p,r=float(metrics.box.mp),float(metrics.box.mr)
    from ultralytics.utils.metrics import smooth
    reported_confidence=float(metrics.box.px[smooth(metrics.box.f1_curve.mean(0),0.1).argmax()])
    report=dict(precision=p,recall=r,F1=2*p*r/(p+r) if p+r else 0.,AP50=float(metrics.box.map50),
                AP75=float(ap[:,5].mean()),mAP50_95=float(metrics.box.map),ap_by_class=ap.tolist(),
                protocol=PROTOCOL,settings=EVAL,counts=validator.export_counts,
                precision_recall_policy="native per-model maximum smoothed-F1 point; same parent reporting convention",
                reported_confidence=reported_confidence,
                threshold_note="Historical P/R/F1 comparison uses this reporting convention, never checkpoint/tuning selection from test",
                speed_ms_per_image=metrics.speed)
    report["delta_percentage_points"]={k:100*(report[k]-v) for k,v in REFERENCE[split].items()}
    np.savez_compressed(output/"curves.npz",ap=ap,confidence=metrics.box.px,
                        precision=metrics.box.p_curve,recall=metrics.box.r_curve,f1=metrics.box.f1_curve)
    return report


def print_metrics(report):
    print("\n"+report.get("split","").upper()+" FP32 / val-selected best",flush=True)
    print("Best:",report.get("best"),"epoch:",report.get("epoch_zero_based"),"SHA256:",report.get("best_sha256"),flush=True)
    for key in ("precision","recall","F1","AP50","AP75","mAP50_95"):
        if key in report:
            delta=report.get("delta_percentage_points",{}).get(key)
            print(f"{key}: {report[key]*100:.6f}%"+(f"  delta={delta:+.6f} percentage points" if delta is not None else ""),flush=True)
    print(report.get("precision_recall_policy",""),flush=True)
    if "reported_confidence" in report:print("Native reporting threshold:",report["reported_confidence"],flush=True)
    print("Evaluation complete. Packaging is a separate manual pack action.",flush=True)
