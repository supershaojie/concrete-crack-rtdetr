"""Read-only mother opportunity probe and locked, same-forward PEQ/raw evaluation."""
from __future__ import annotations
import gzip
import time
import cv2
import numpy as np
import torch
from peq_v1_common import *
from ultralytics import RTDETR
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.cbr import RTDETRDecoderCBR
from ultralytics.nn.modules.peq import finite
from ultralytics.models.utils.peq_loss import aligned_iou
from ultralytics.models.rtdetr.peq_model import PEQDetectionModel
from ultralytics.models.rtdetr.peq_train import PEQValidator
from ultralytics.utils import ops, YAML
from ultralytics.utils.metrics import smooth

POLICY="corrected_sorted_conf_mask_v1"
EVAL=dict(imgsz=640,batch=16,workers=0,half=False,conf=.001,iou=.7,max_det=300,augment=False,rect=False,seed=42)


def batches(inventory, split, batch_size, imgsz=640, limit=None):
    """Read-only RTDETR eval resize(rect_mode=False), RGB and normalized cxcywh GT."""
    root=Path(inventory["root"])
    files=inventory["splits"][split]["image_list"][:limit]
    for start in range(0,len(files),batch_size):
        images, boxes, indices, names, shapes=[],[],[],[],[]
        for i,rel in enumerate(files[start:start+batch_size]):
            image=cv2.imread(str(root/rel))
            require(image is not None,f"Cannot decode {rel}")
            shapes.append(tuple(image.shape[:2]))
            resized=cv2.resize(image,(imgsz,imgsz),interpolation=cv2.INTER_LINEAR)
            images.append(torch.from_numpy(resized[:,:,::-1].copy()).permute(2,0,1))
            label=root/"labels"/split/Path(rel).relative_to(Path("images")/split).with_suffix(".txt")
            rows=[list(map(float,line.split())) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            boxes.extend(row[1:] for row in rows)
            indices.extend([i]*len(rows))
            names.append(str(root/rel))
        yield dict(img=torch.stack(images),bboxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                   cls=torch.zeros((len(boxes),1)),batch_idx=torch.tensor(indices,dtype=torch.long),
                   im_file=names,ori_shape=shapes,ratio_pad=[None]*len(names))


def device_batch(batch, device):
    return {k:(v.to(device).float()/255 if k=="img" else v.to(device)) if torch.is_tensor(v) else v
            for k,v in batch.items()}


def mother_diagnostic_forward(model, images):
    saved=[]
    value=images
    for module in model.model[:-1]:
        if module.f != -1:
            value=saved[module.f] if isinstance(module.f,int) else [value if j==-1 else saved[j] for j in module.f]
        value=module(value)
        saved.append(value if module.i in model.save else None)
    head=model.model[-1]
    require(type(head) is RTDETRDecoderCBR,"Probe requires the ORIGINAL mother head")
    result,details=head.forward_with_diagnostics([saved[j] for j in head.f])
    details["feature_shapes"]=[list(saved[j].shape) for j in head.f]
    return result,details


def stream_row(batch, index, before, after, raw, final, root, extras=None):
    h,w=map(int,batch["ori_shape"][index])
    scale=after.new_tensor([w,h,w,h])
    gt=batch["bboxes"][batch["batch_idx"]==index]
    finite(before,after,raw,final,gt)
    row=dict(image=Path(batch["im_file"][index]).resolve().relative_to(Path(root)).as_posix(),
             original_size_hw=[h,w],box_format="xyxy",coordinate_space="original_image_pixels",
             coordinates="continuous, no clipping/rounding; RT-DETR square stretch inversion",
             query_ids=list(range(len(after))),
             before_boxes=(ops.xywh2xyxy(before)*scale).cpu().tolist(),
             final_boxes=(ops.xywh2xyxy(after)*scale).cpu().tolist(),
             s_raw=raw.reshape(-1).cpu().tolist(),s_final=final.reshape(-1).cpu().tolist(),
             ground_truth=(ops.xywh2xyxy(gt)*scale).cpu().tolist(),class_id=0)
    if extras is not None:
        row.update(extras)
    return row


def cache_valid(folder, key, expected_images):
    report=read_json(folder/"metrics.json")
    if report is None:
        return None
    require(report.get("key")==key,"Existing evaluation identity differs")
    require(report.get("status") in ("COMPLETED","PARTIAL"),"Incomplete attempt preserved; use --retry")
    path=folder/"predictions_gt.jsonl.gz"
    require(report["stream"]["sha256"]==sha256(path),"Cached prediction stream changed")
    seen=[]
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            require(row["query_ids"]==list(range(300)),"Incomplete query cache")
            seen.append(row["image"])
    require(len(seen)==len(set(seen))==len(expected_images) and sorted(seen)==sorted(expected_images),"Incomplete image cache")
    return report


def evaluation_cache(folder, key, expected_images, split, retry):
    """A completed test is immutable, including an explicit --retry request."""
    if not folder.exists():
        return None
    previous=read_json(folder/"metrics.json",{})
    if split=="test" and previous.get("status")=="COMPLETED":
        return cache_valid(folder,key,expected_images)
    return cache_valid(folder,key,expected_images) if not retry else None


def probe(weights, data, device="0", batch_size=2, limit=None, retry=False):
    weights,data=Path(weights),Path(data)
    missing=[]
    if not weights.is_file():
        missing.append(str(weights))
    if not data.is_file():
        missing.append(str(data))
    else:
        config=YAML.load(data)
        for folder in (Path(config["path"])/config["val"],Path(config["path"])/"labels/val"):
            if not folder.is_dir():
                missing.append(str(folder))
    if missing:
        result=dict(status="PENDING",reason="Missing trained mother best or data; no random-model substitute",
                    missing=missing,weights=identity(weights),data=identity(data))
        write_json(OUT/"probe_pending.json",result)
        return result
    module_contract()
    require(sha256(weights)==BEST_SHA,"Probe best must match historical mother SHA256")
    require(batch_size>0 and (limit is None or limit>0),"Invalid probe batch/limit")
    inventory=dataset_identity(data,splits=("val",))
    image_list=inventory["splits"]["val"]["image_list"][:limit]
    settings=dict(precision="FP32",imgsz=640,augment=False,all_queries=300,batch=batch_size,limit=limit,split="val")
    # Cache only files actually used by the read-only probe. Unrelated dispatch,
    # pack, and test-harness edits cannot change the mother's inference.
    source_files=source_identity()["files"]
    probe_source={k:v for k,v in source_files.items() if k.startswith("ultralytics-main/ultralytics/") or
                  k in ("tools/peq_v1_eval.py","tools/peq_v1_common.py")}
    contract=dict(weights=identity(weights),data=inventory,source=digest_json(probe_source),
                  source_files=probe_source,settings=settings)
    key=digest_json(contract)
    folder=OUT/"probe"/key[:20]
    cached=cache_valid(folder,key,image_list) if not retry else None
    if cached:
        return cached
    if folder.exists():
        require(retry,"Partial probe preserved; use --retry for separate attempt")
        folder=folder.with_name(folder.name+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    folder.mkdir(parents=True)
    report=dict(status="FAILED",key=key,contract=contract,scope="PARTIAL" if limit else "FULL_VAL",
                note="Candidate coverage/max-IoU proxy are not Recall/AP or an AP upper bound",
                image_list=image_list,environment=environment())
    scores_list,proxy_list=[],[]
    gt_count=images=queries=0
    covered=np.zeros(3,dtype=np.int64)
    rank_outside_gt=np.zeros(3,dtype=np.int64)
    pairs=inverted=ties=0
    bin_counts=np.zeros(10,dtype=np.int64);bin_sums=np.zeros(10);bin_high=np.zeros((10,3),dtype=np.int64)
    iou_change_sum=displacement_sum=0.
    stream_path=folder/"predictions_gt.jsonl.gz"
    start=time.monotonic()
    try:
        with isolated_rng(),torch.inference_mode():
            model=RTDETR(str(weights)).model.float().eval()
            require(model.model[-1].nc==1,"Mother probe requires nc=1")
            device=torch.device("cpu" if str(device)=="cpu" else f"cuda:{device}")
            model.to(device)
            bn_before={k:v.clone() for k,v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
            with gzip.open(stream_path,"xt",encoding="utf-8") as stream:
                for batch in batches(inventory,"val",batch_size,limit=limit):
                    batch=device_batch(batch,device)
                    (y,raw),details=mother_diagnostic_forward(model,batch["img"])
                    report.setdefault("feature_shapes_first_batch",details["feature_shapes"])
                    require(details["feature_shapes"][0][1:]==[256,80,80],"Mother P3 at 640 changed")
                    for i in range(len(y)):
                        b0,b1=details["before"][i],details["after"][i]
                        score=y[i,:,4].float()
                        gt=batch["bboxes"][batch["batch_idx"]==i]
                        if len(gt):
                            matrix=aligned_iou(b1[:,None,:].float(),gt[None,:,:].float())
                            proxy=matrix.max(1).values
                            proxy0=aligned_iou(b0[:,None,:].float(),gt[None,:,:].float()).max(1).values
                            best=matrix.max(0).values
                        else:
                            matrix=score.new_empty((300,0));proxy=proxy0=score.new_zeros(300);best=score.new_empty(0)
                        high,low=score[proxy>=.75],score[proxy<.5]
                        comparison=low[:,None]-high[None,:]
                        pairs+=comparison.numel();inverted+=int((comparison>0).sum());ties+=int((comparison==0).sum())
                        rank=torch.empty_like(score,dtype=torch.long)
                        rank[score.argsort(descending=True)]=torch.arange(1,301,device=device)
                        per_gt=[]
                        for j in range(len(gt)):
                            item=dict(gt_index=j,max_final_iou=float(best[j]),candidate_quality={})
                            for k,tau in enumerate((.5,.75,.9)):
                                qualified=matrix[:,j]>=tau;has=bool(qualified.any());covered[k]+=has
                                best_rank=int(rank[qualified].min()) if has else None
                                rank_outside_gt[k]+=bool(has and best_rank>len(gt))
                                item["candidate_quality"][str(tau)]=dict(exists=has,best_score=float(score[qualified].max()) if has else None,
                                                                       best_score_rank=best_rank)
                            per_gt.append(item)
                        scores=score.cpu().numpy();quality=proxy.cpu().numpy()
                        bins=np.minimum((scores*10).astype(int),9)
                        for k in range(10):
                            selected=quality[bins==k];bin_counts[k]+=len(selected);bin_sums[k]+=float(selected.sum())
                            bin_high[k]+=np.array([(selected>=t).sum() for t in (.5,.75,.9)])
                        scores_list.append(scores);proxy_list.append(quality)
                        iou_change_sum+=float((proxy-proxy0).sum())
                        displacement_sum+=float((b1-b0).norm(dim=-1).sum())
                        images+=1;queries+=300;gt_count+=len(gt)
                        row=stream_row(batch,i,b0,b1,score,score,inventory["root"],
                                       dict(gt_candidate_coverage=per_gt,max_iou_proxy=quality.tolist()))
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                    if images%100<batch_size:
                        print(f"probe {images}/{len(image_list)} images, {time.monotonic()-start:.1f}s",flush=True)
            require(all(torch.equal(v,model.state_dict()[k]) for k,v in bn_before.items()),"Probe changed BN buffers")
        require(images==len(image_list) and images>0,"Incomplete/empty probe")
        scores=np.concatenate(scores_list);quality=np.concatenate(proxy_list)
        corr=float(np.corrcoef(scores,quality)[0,1]) if np.std(scores)>0 and np.std(quality)>0 else None
        inversions=inverted/pairs if pairs else None;coverage=(covered/gt_count).tolist() if gt_count else [None]*3
        supported=coverage[1] is not None and coverage[1]>=.5 and inversions is not None and inversions>=.01
        report.update(status="PARTIAL" if limit else "COMPLETED",images=images,queries=queries,ground_truth=gt_count,
                      coverage_thresholds=[.5,.75,.9],gt_with_candidate=covered.tolist(),candidate_coverage=coverage,
                      accurate_candidate_best_rank_outside_top_gt_count=rank_outside_gt.tolist(),
                      compared_high_low_pairs=pairs,inverted_pairs=inverted,tied_pairs=ties,inversion_rate=inversions,
                      raw_score_proxy_pearson=corr,mean_max_iou_change=iou_change_sum/queries,
                      mean_cxcywh_displacement_norm=displacement_sum/queries,
                      score_bins=[dict(low=k/10,high=(k+1)/10,queries=int(bin_counts[k]),
                                       mean_max_iou=bin_sums[k]/bin_counts[k] if bin_counts[k] else None,
                                       quality_positive_counts=bin_high[k].tolist()) for k in range(10)],
                      opportunity="PARTIAL_NO_FULL_VAL_CONCLUSION" if limit else
                      ("Ranking opportunity detected; training still required" if supported else "PROBE_DOES_NOT_SUPPORT_PRIORITY_LONG_TRAINING"),
                      opportunity_rule="Descriptive only: coverage@.75 >= .50 AND inversion rate >= .01; no hyperparameter search",
                      bn_unchanged=True,weights_unchanged=sha256(weights)==BEST_SHA,
                      seconds=time.monotonic()-start,stream=identity(stream_path))
        require(report["weights_unchanged"],"Probe weights changed")
    except BaseException as error:
        report["error"]=repr(error)
        raise
    finally:
        write_json(folder/"metrics.json",report)
    return report

def metric_summary(metrics):
    box=metrics.box
    ap=np.asarray(box.all_ap)
    require(ap.shape==(1,10) and np.isfinite(ap).all(),"Invalid AP array")
    index=int(smooth(np.asarray(box.f1_curve).mean(0),.1).argmax())
    return dict(mAP50_95=float(box.map),AP50=float(box.map50),AP75=float(ap[:,5].mean()),
                AP_by_threshold=ap[0].tolist(),iou_thresholds=[round(.5+i*.05,2) for i in range(10)],
                precision=float(box.mp),recall=float(box.mr),f1=float(np.asarray(box.f1).mean()),
                confidence_at_max_f1=float(np.asarray(box.px)[index]),
                precision_recall_rule="Each score mode: own max smoothed mean F1 at IoU .50; smoothing .1, 1000 confidence grid",
                fitness=float(metrics.fitness))


def evaluate(split, data, retry=False):
    require(split in ("val","test"),"Only independent val/test")
    weights=RUN/"weights/best.pt"
    require(weights.is_file(),"This PEQ run has no best.pt; no substitution allowed")
    prepared=read_json(OUT/"prepared.json")
    require(prepared and prepared["status"]=="PASS","prepare identity missing")
    state=read_json(OUT/"state.json",{})
    require(state.get("phase")=="COMPLETED","Training must finish before independent val/test")
    require(source_identity()["sha256"]==prepared["identity"]["source_sha256"],"Source changed since preparation")
    inventory=dataset_identity(data)
    require(inventory==prepared["identity"]["data"],"Data differs from preparation")
    contract=dict(checkpoint=identity(weights),source=source_identity()["sha256"],
                  train_args=identity(OUT/"train_args.yaml"),research=identity(ROOT/"docs/peq_v1/research.yaml"),
                  data=inventory,settings=EVAL,score="mean_sigmoid(z_detached+tanh(u_k))")
    key=digest_json(contract)
    lock=read_json(OUT/"val_lock.json")
    if split=="test":
        require(lock and lock["key"]==key,"Test requires checkpoint/source/config/data locked by independent val")
    elif lock and lock["key"]!=key:
        require(not (OUT/"evaluation_test").exists(),"An attempted test identity cannot be replaced")
    folder=OUT/f"evaluation_{split}"
    cached=evaluation_cache(folder,key,inventory["splits"][split]["image_list"],split,retry)
    if cached:
        if split=="val" and not lock:
            write_json(OUT/"val_lock.json",dict(key=key,contract=contract,metrics=identity(folder/"metrics.json"),locked=utc()))
        return cached
    if folder.exists():
        require(retry,"Previous evaluation preserved; use --retry for explicit archival")
        folder.rename(folder.with_name(folder.name+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
    folder.mkdir(parents=True)
    report=dict(status="FAILED",key=key,split=split,contract=contract,policy=POLICY,environment=environment(),
                source="independent_FP32",score_mode="PEQ",raw_score_mode="same-final-box diagnostic only")
    count=truths=metric_predictions=raw_predictions=0
    stream_path=folder/"predictions_gt.jsonl.gz"
    try:
        with isolated_rng(),torch.inference_mode():
            loaded=RTDETR(str(weights))
            require(type(loaded.model) is PEQDetectionModel and loaded.model.model[-1].peq.config["enabled"],"Not an enabled PEQ checkpoint")
            model=AutoBackend(model=loaded.model,device=torch.device("cuda:0"),fp16=False,fuse=True,verbose=False)
            model.eval()
            require(model.pt and not model.fp16,"Independent eval must stay PyTorch FP32")
            validators=[]
            for mode in ("peq","raw"):
                validator=PEQValidator(args=dict(EVAL,data=str(data),split=split,device="0",plots=True,
                                                 save_json=False,save_txt=False,project=str(folder),name=mode),
                                       save_dir=folder/mode)
                validator.data={"path":inventory["root"],"names":{0:"crack"},"nc":1,"channels":3,
                                split:str(Path(inventory["root"])/"images"/split)}
                validator.training=False
                validator.device=torch.device("cuda:0")
                validator.iouv=validator.iouv.to(validator.device)
                validator.init_metrics(model)
                validators.append(validator)
            with gzip.open(stream_path,"xt",encoding="utf-8") as stream:
                for batch in batches(inventory,split,16):
                    batch=device_batch(batch,torch.device("cuda:0"))
                    y,raw=model(batch["img"])
                    payload=raw[5]
                    require(y.shape[1:]==(300,5) and torch.equal(y[...,4:],payload["s_final"]),"PEQ score route changed")
                    raw_y=torch.cat((y[...,:4],payload["s_raw"]),-1)
                    selections=[validators[0].postprocess(y),validators[1].postprocess(raw_y)]
                    for validator,selected in zip(validators,selections):
                        validator.update_metrics(selected,batch)
                    for i in range(len(y)):
                        row=stream_row(batch,i,payload["b0"][i],payload["b1"][i],payload["s_raw"][i],payload["s_final"][i],inventory["root"])
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                        count+=1;truths+=len(row["ground_truth"])
                        metric_predictions+=len(selections[0][i]["conf"]);raw_predictions+=len(selections[1][i]["conf"])
                    if count%100<16:
                        print(f"{split}: {count}/{inventory['splits'][split]['images']}",flush=True)
            for validator in validators:
                validator.get_stats()
                validator.finalize_metrics()
            require(count==inventory["splits"][split]["images"] and truths==inventory["splits"][split]["ground_truth"] and truths>0,
                    "Incomplete/empty evaluation")
            report.update(status="COMPLETED",images=count,ground_truth=truths,predictions=count*300,
                          metric_predictions=metric_predictions,raw_metric_predictions=raw_predictions,
                          peq=metric_summary(validators[0].metrics),raw_diagnostic=metric_summary(validators[1].metrics),
                          stream=identity(stream_path),actual_settings=vars(validators[0].args),
                          same_forward=True,model_type=type(model.model).__name__,fuse=True)
            require(sha256(weights)==contract["checkpoint"]["sha256"],"Checkpoint changed during evaluation")
    except BaseException as error:
        report["error"]=repr(error)
        raise
    finally:
        write_json(folder/"metrics.json",report)
    if split=="val":
        write_json(OUT/"val_lock.json",dict(key=key,contract=contract,metrics=identity(folder/"metrics.json"),locked=utc()))
    return report
