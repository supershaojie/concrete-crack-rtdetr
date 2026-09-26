"""FP32 independent evaluation, immutable val/test identity and geometric diagnostics."""
from __future__ import annotations

import gzip
from tcr_v1_core import *
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import init_seeds
from c19_lif_v1_results import postprocess as mother_postprocess, image_record, EVAL, POLICY


class CorrectedValidator(RTDETRValidator):
    def postprocess(self, preds):
        result, self.affected = mother_postprocess(preds, self.args.imgsz, self.args.conf)
        return result


def all_queries(preds, imgsz):
    tensor = preds[0] if isinstance(preds, (list,tuple)) else preds
    result = []
    for pred in tensor:
        score, cls = pred[:,4:].max(-1)
        order = score.argsort(descending=True)
        result.append(dict(bboxes=ops.xywh2xyxy(pred[:,:4]*imgsz)[order], conf=score[order], cls=cls[order], query_id=order))
    return result


def geometry_init():
    definitions = YAML.load(DOC / "research.yaml")["geometry_buckets"]
    return dict(definitions=definitions, warning=definitions["warning"], buckets={k:[dict(gt=0, matched_iou50=0, matched_iou75=0, best_iou_sum=0.) for _ in range(len(definitions[k+"_edges"])-1)]
                for k in ("short_side", "area", "aspect_ratio")})


def geometry_update(report, gt, predictions):
    boxes = gt["bboxes"].detach().float().cpu()
    pred = predictions["bboxes"].detach().float().cpu()[predictions["conf"].detach().cpu() >= .25]
    if not len(boxes): return
    ious = box_iou(boxes,pred) if len(pred) else torch.empty(len(boxes),0)
    matched = {}
    for threshold in (.5,.75):
        pairs = [(float(ious[g,p]), int(g), int(p)) for g,p in (ious >= threshold).nonzero().tolist()]
        used_gt, used_pred = set(), set()
        for _,g,p in sorted(pairs,reverse=True):
            if g not in used_gt and p not in used_pred:
                used_gt.add(g); used_pred.add(p)
        matched[threshold] = used_gt
    wh = (boxes[:,2:] - boxes[:,:2]).clamp_min(1e-12)
    ih, iw = gt["imgsz"]
    wh *= torch.tensor([640/iw, 640/ih])
    values = dict(short_side=wh.min(1).values, area=wh.prod(1), aspect_ratio=wh.max(1).values/wh.min(1).values)
    best = ious.max(1).values if len(pred) else torch.zeros(len(boxes))
    for name, vals in values.items():
        edges = report["definitions"][name+"_edges"]
        for g, value in enumerate(vals):
            index = next(i for i in range(len(edges)-1) if edges[i] <= float(value) < edges[i+1])
            row = report["buckets"][name][index]
            row["gt"] += 1; row["matched_iou50"] += int(g in matched[.5]); row["matched_iou75"] += int(g in matched[.75])
            row["best_iou_sum"] += float(best[g])


def evaluate(split):
    from tcr_v1_ops import verify_prepared, status
    prepared = verify_prepared()
    require(status()["phase"] == "COMPLETED", "Independent evaluation requires successful completed formal training")
    weights = RUN / "weights/best.pt"
    require(weights.is_file(), "This experiment has no formal best")
    identity = dict(weight=file_identity(weights), source=source_identity()["sha256"], commit=git("rev-parse","HEAD"),
                    data=prepared["data"], config=prepared["config"], policy=POLICY, settings=EVAL, enabled=True)
    folder = OUT / ("evaluation_"+split)
    prior = read_json(folder / "metrics.json")
    if prior:
        require(prior.get("status") == "completed" and prior.get("identity") == identity, "Existing different/incomplete evaluation preserved")
        require(sha256(folder/"predictions_gt.jsonl.gz") == prior["predictions_gt_sha256"], "Saved evaluation evidence changed")
        print("Reusing completed",split,"evaluation:",folder)
        return prior
    if split == "test":
        lock = read_json(OUT / "val_lock.json")
        require(lock and lock["identity"] == identity, "Explicit test requires exact independent val-locked best/config/data/source")
    require(not folder.exists(), "Incomplete evaluation directory preserved; inspect before retry")
    data = OUT / "data.yaml"
    resolved = check_det_dataset(str(data),autodownload=False)
    folder.mkdir(parents=True,exist_ok=False)
    report = dict(status="failed", split=split, identity=identity, runtime=runtime(), policy=POLICY,
                  evidence_scope="full_split", metric_source="independent FP32", boxes="original final CBR boxes",
                  precision_recall_policy="each model's own maximum of smoothed mean F1 curve; same native metrics",
                  geometry=geometry_init(), settings=EVAL)
    counts = dict(images=0,predictions=0,metric_predictions=0,ground_truth=0,historical_mask_affected_images=0)
    seen = set()
    stream_path = folder/"predictions_gt.jsonl.gz"
    try:
        api = RTDETR(str(weights)); verify_model(api.model)
        require(api.model.model[-1].nc == 1, "Formal nc1 required")
        with isolated_rng(), gzip.open(stream_path,"xt",encoding="utf-8") as stream:
            init_seeds(42,deterministic=True)
            class StreamingValidator(CorrectedValidator):
                def init_metrics(self,model):
                    super().init_metrics(model)
                    require(not self.training and not self.args.half, "Independent evaluation must be FP32")
                    require(all(getattr(self.args,k)==v for k,v in EVAL.items()), "Evaluation settings drift")
                    self.expected = {str(Path(p).resolve()) for p in self.dataloader.dataset.im_files}
                    require(self.expected, "Empty split")
                    report["actual_settings"] = vars(self.args)

                def postprocess(self,preds):
                    selected = super().postprocess(preds)
                    full = all_queries(preds,self.args.imgsz)
                    counts["historical_mask_affected_images"] += self.affected
                    for chosen, raw in zip(selected,full): chosen["_all"] = raw
                    return selected

                def update_metrics(self,preds,batch):
                    for i,pred in enumerate(preds):
                        full = pred.pop("_all")
                        gt = self._prepare_batch(i,batch)
                        path = str(Path(gt["im_file"]).resolve())
                        require(path not in seen, "Duplicate evaluation image")
                        seen.add(path)
                        row = image_record(full,gt,self.data["path"],self.args.conf)
                        row["image_id"] = row["image"]
                        for p,q in zip(row["predictions"],full["query_id"].tolist()): p["query_id"] = q
                        require(len(row["predictions"])==300,"Incomplete regular queries")
                        geometry_update(report["geometry"],gt,full)
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                        counts["images"]+=1; counts["ground_truth"]+=len(row["ground_truth"])
                        counts["predictions"]+=300; counts["metric_predictions"]+=len(pred["conf"])
                    stream.flush()
                    return super().update_metrics(preds,batch)

                def finalize_metrics(self):
                    require(seen == self.expected and counts["ground_truth"]>0,"Incomplete/empty GT split")
                    return super().finalize_metrics()

            metrics = api.val(validator=StreamingValidator, **EVAL, data=str(data), split=split, device="0",
                              plots=True, save_json=False, project=str(folder),name="plots",exist_ok=False)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape==(1,10) and np.isfinite(ap).all(),"Invalid AP array")
        require(file_identity(weights)==identity["weight"],"Best changed during evaluation")
        expected = prepared["data"]["inventory"][split]
        require(counts["images"]==expected["images"] and counts["ground_truth"]==expected["boxes"],"Full split counts changed")
        from ultralytics.utils.metrics import smooth
        f1 = np.asarray(metrics.box.f1_curve)
        index = int(smooth(f1.mean(0),.1).argmax())
        report.update(status="completed",mAP50_95=float(metrics.box.map),AP50=float(metrics.box.map50),AP75=float(ap[:,5].mean()),
                      precision=float(metrics.box.mp),recall=float(metrics.box.mr),ap_by_class=ap.tolist(),
                      ap_iou_thresholds=[round(.5+.05*i,2) for i in range(10)],
                      max_f1_confidence=float(metrics.box.px[index]),speed_ms_per_image=metrics.speed,
                      predictions_gt_sha256=sha256(stream_path), **counts)
        if split=="val": write_json(OUT/"val_lock.json",dict(identity=identity,metrics=str(folder/"metrics.json"),locked=utc()))
    except BaseException as error:
        report.update(error=repr(error),**counts)
        raise
    finally:
        write_json(folder/"metrics.json",report)
    return report
