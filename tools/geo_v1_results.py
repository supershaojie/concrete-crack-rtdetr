"""One-pass formal evaluation, result locks, offline curves/counts and complete evidence pack."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

from geo_v1_common import (ROOT,OUT,RUN,FORMULA,BASE,RELIABILITY,COUNTS,code_identity,data_config,
    verify_snapshot,read_json,write_json,sha256,require,digest,now,git,runtime)
from ultralytics.models.rtdetr.geo_val import GEOValidator,EVAL,POLICY


def file_info(path,hash_file=True):
    p=Path(path)
    return dict(path=str(p),exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else None,
        sha256=sha256(p) if hash_file and p.is_file() else None)


def checkpoint_identity(path,identity,resume=False):
    from ultralytics.utils.patches import torch_load
    ckpt=torch_load(path,map_location="cpu")
    model=ckpt.get("ema") or ckpt.get("model")
    require(getattr(model,"geo_identity",None)==identity,"Checkpoint is not bound to this GEO run")
    require(type(model).__module__=="ultralytics.models.rtdetr.geo_model","Checkpoint is not an importable GEO model")
    if resume:
        require(0<=ckpt.get("epoch",-1)<199 and ckpt.get("optimizer") is not None and ckpt.get("ema") is not None
            and ckpt.get("scaler") and ckpt.get("updates") is not None,
            "last is completed/stripped or lacks optimizer/scaler/EMA; use finish for evaluation recovery")
    return ckpt


def eval_identity(split):
    completion=read_json(OUT/"training_completed.json",{})
    require(completion.get("status")=="TRAINING_COMPLETED","No native training-completion evidence")
    training=read_json(OUT/"training_identity.json")
    require(training,"Missing run identity")
    current=code_identity(); original=training["binding"]["code"]
    changed={p for p in set(current["files"])|set(original["files"]) if current["files"].get(p)!=original["files"].get(p)}
    allowed={"ultralytics-main/ultralytics/nn/autobackend.py","ultralytics-main/ultralytics/models/rtdetr/geo_val.py","tools/geo_v1_results.py"}
    require(changed<=allowed,f"Evaluation-only repair changed training source: {sorted(changed-allowed)}")
    path=data_config(); verify_snapshot(path,training["binding"]["data"])
    checkpoint=RUN/"weights/best.pt"
    require(checkpoint.is_file(),"Native selected best.pt is missing; never select another checkpoint")
    return dict(split=split,checkpoint_sha256=sha256(checkpoint),data=training["binding"]["data"],settings=EVAL,policy=POLICY,
        training_commit=original["commit"],evaluation_source_sha256=current["source_lf_sha256"],
        evaluation_commit=current["commit"],allowed_eval_repairs=sorted(changed),formula=FORMULA)


def valid_record(report,identity):
    if not report or report.get("status")!="PASS" or report.get("identity")!=identity: return False
    if identity.get("split") not in COUNTS: return False
    fields=("Precision","Recall","F1","AP50","AP75","mAP50_95","native_best_F1_confidence")
    if any(not isinstance(report.get(k),(int,float)) or not math.isfinite(report[k]) for k in fields): return False
    if len(report.get("AP_by_IoU",[]))!=10 or not all(math.isfinite(v) for v in report["AP_by_IoU"]): return False
    required={"predictions","curves"}
    if not required<=report.get("artifacts",{}).keys(): return False
    if report.get("images")!=COUNTS[identity["split"]][0]: return False
    for info in report["artifacts"].values():
        p=Path(info["path"])
        if not p.is_file() or p.stat().st_size!=info["bytes"] or sha256(p)!=info["sha256"]: return False
    return True


def reuse_result(split,identity):
    path=OUT/f"{split}_lock.json"
    lock=read_json(path)
    if lock:
        require(valid_record(lock,identity),f"Existing {split} lock identity/files differ; preserve and investigate")
        return lock
    # A completed metrics record can survive a crash immediately before lock write.
    for p in sorted((OUT/"evaluations").glob(f"*_{split}/metrics.json")):
        report=read_json(p)
        if valid_record(report,identity):
            restored=dict(report,recovered_lock_at=now(),report=str(p))
            write_json(path,restored)
            return restored
    return None


def evaluate(split):
    import numpy as np
    from ultralytics.utils.metrics import smooth
    from ultralytics.utils.torch_utils import init_seeds
    identity=eval_identity(split)
    old=reuse_result(split,identity)
    if old: return old
    if split=="test":
        val_identity=dict(identity,split="val")
        require(reuse_result("val",val_identity),"Formal test requires same-best complete val first")
    training=read_json(OUT/"training_identity.json")
    ckpt=checkpoint_identity(RUN/"weights/best.pt",training)
    best_epoch=ckpt.get("epoch")
    del ckpt
    folder=OUT/"evaluations"/(time.strftime("%Y%m%d_%H%M%S")+f"_{split}")
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(status="FAIL",identity=identity,started=now(),scope="full split FP32",best_epoch_zero=best_epoch,runtime=runtime(),
        best_rule="native training val mAP50-95, same best for val/test; no reselection",artifacts={})
    settings=dict(EVAL,model=str(RUN/"weights/best.pt"),data=str(data_config()),split=split,device="0",plots=True,
        save_json=False,save_txt=False,project=str(folder),name="plots",exist_ok=False)
    validator=GEOValidator(args=settings,save_dir=folder/"plots")
    validator.export_path=folder/f"{split}_predictions_gt.jsonl.gz"
    validator.export_identity=identity
    try:
        print(f"GEO formal {split}: FP32 B16/640, all 300 queries + GT exported in this pass",flush=True)
        init_seeds(42,deterministic=True)
        metrics=validator(model=str(RUN/"weights/best.pt"))
        ap=np.asarray(validator.metrics.box.all_ap)
        require(ap.shape==(1,10) and np.isfinite(ap).all(),"Expected nc1 finite ten-IoU AP")
        require(all(type(validator.actual_settings[k]) is type(v) and validator.actual_settings[k]==v for k,v in EVAL.items()),"Effective formal protocol drift")
        require(sha256(RUN/"weights/best.pt")==identity["checkpoint_sha256"],"Best checkpoint changed")
        with gzip.open(OUT/"data_manifest.jsonl.gz","rt",encoding="utf-8") as f:
            expected={r["image"] for r in map(json.loads,f) if r["split"]==split}
        require(validator.geo_seen==expected,"Formal export does not cover snapshot image identities")
        box=validator.metrics.box
        curves={"iou_for_PR_and_confidence":.5,"class_average":"nc1 crack", "official_conf_filter":"score > 0.001",
            "export_conf_filter":None,"curves":[dict(x=np.asarray(x).tolist(),y=np.asarray(y).tolist(),xlabel=xl,ylabel=yl)
                for x,y,xl,yl in box.curves_results]}
        index=int(smooth(np.asarray(box.f1_curve).mean(0),.1).argmax())
        confidence=float(np.asarray(box.px)[index])
        write_json(folder/"curves.json",curves)
        report.update(status="PASS",metrics=metrics,Precision=float(box.mp),Recall=float(box.mr),F1=float(np.asarray(box.f1).mean()),
            AP50=float(ap[0,0]),AP75=float(ap[0,5]),mAP50_95=float(ap.mean()),AP_by_IoU=ap[0].tolist(),
            IoU_thresholds=[round(.5+.05*i,2) for i in range(10)],
            metric_definition="P is detection precision, not Accuracy; P/R/F1 use each split's native smoothed best-F1 point at IoU=.5, nc1",
            native_best_F1_confidence=confidence,images=len(validator.geo_seen),actual_settings=validator.actual_settings,
            artifacts=dict(predictions=file_info(validator.export_path),curves=file_info(folder/"curves.json")))
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report["ended"]=now(); write_json(folder/"metrics.json",report)
    lock=dict(report,report=str(folder/"metrics.json"))
    require(valid_record(lock,identity),"Result did not meet completion contract; no success lock written")
    write_json(OUT/f"{split}_lock.json",lock)
    return lock


def counts_at_threshold(path,confidence):
    """Offline native IoU matching after the chosen confidence filter; no image reads."""
    import numpy as np
    import torch
    from ultralytics.models.rtdetr.val import RTDETRValidator
    evaluator=RTDETRValidator.__new__(RTDETRValidator)
    evaluator.iouv=torch.tensor([.5]); evaluator.niou=1
    tp=fp=fn=images=0
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        for row in map(json.loads,stream):
            scores=np.asarray(row["scores"])
            idx=np.argsort(-scores)[scores[np.argsort(-scores)]>confidence]
            boxes=torch.tensor(row["boxes"],dtype=torch.float32).reshape(-1,4)[idx]
            cls=torch.tensor(row["classes"],dtype=torch.float32)[idx]
            gt=torch.tensor(row["gt_boxes"],dtype=torch.float32).reshape(-1,4)
            gc=torch.tensor(row["gt_classes"],dtype=torch.float32).reshape(-1)
            correct=evaluator._process_batch(dict(bboxes=boxes,cls=cls),dict(bboxes=gt,cls=gc))["tp"]
            matched=int(correct.sum()); tp+=matched; fp+=len(boxes)-matched; fn+=len(gt)-matched; images+=1
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return dict(TP=tp,FP=fp,FN=fn,Precision=p,Recall=r,F1=2*p*r/max(p+r,1e-12),images=images,
        confidence=confidence,confidence_rule="score > threshold",IoU=.5,class_average="nc1 crack",matching="native RTDETR validator IoU matching")


def offline_summary():
    locks={s:read_json(OUT/f"{s}_lock.json") for s in ("val","test")}
    require(all(locks.values()),"Both complete split locks are required")
    for split,lock in locks.items(): require(valid_record(lock,eval_identity(split)),f"Invalid {split} lock")
    # Confidence is selected from val only. Do not optimize it on test.
    threshold=max(.001,locks["val"]["native_best_F1_confidence"])
    summary=dict(created=now(),threshold_source="val native best-F1 confidence, lower bounded by official .001 boundary",
        val_selected_confidence=threshold,checkpoint_sha256=locks["val"]["identity"]["checkpoint_sha256"],
        fixed_val_threshold={s:counts_at_threshold(v["artifacts"]["predictions"]["path"],threshold) for s,v in locks.items()},
        native_split_best_F1={s:dict(Precision=v["Precision"],Recall=v["Recall"],F1=v["F1"],confidence=v["native_best_F1_confidence"],
            counts_at_that_threshold=counts_at_threshold(v["artifacts"]["predictions"]["path"],v["native_best_F1_confidence"])) for s,v in locks.items()},
        note="Offline counts rematch at their threshold; native interpolated P/R need not equal integer-count ratios. Test never selects deployment threshold.")
    write_json(OUT/"offline_summary.json",summary)
    return summary


def completeness():
    """Only inspect existing evidence. No model load, data scan, inference or selection."""
    missing=[]
    for name in ("prepare.json","initialization.json","training_identity.json","actual_setup.json","training_completed.json",
                 "preflight.json","mechanism.jsonl","offline_summary.json","training_source_snapshot.tar.gz"):
        if not (OUT/name).is_file(): missing.append(name)
    training=read_json(OUT/"training_identity.json")
    locks=[]
    for split in ("val","test"):
        lock=read_json(OUT/f"{split}_lock.json")
        if not lock or not valid_record(lock,lock.get("identity",{})): missing.append(f"{split}: complete evaluation/export/curves/lock")
        else: locks.append(lock)
    if len(locks)==2:
        a,b=(dict(x["identity"]) for x in locks); a.pop("split"); b.pop("split")
        if a!=b: missing.append("val/test identity mismatch")
        if training and (a["data"]!=training["binding"]["data"] or a["training_commit"]!=training["binding"]["code"]["commit"]):
            missing.append("evaluation/training identity mismatch")
        best=RUN/"weights/best.pt"
        if best.is_file() and sha256(best)!=a["checkpoint_sha256"]: missing.append("best checkpoint differs from evaluation locks")
    if read_json(OUT/"preflight.json",{}).get("status")!="PASS": missing.append("preflight PASS")
    if not (RUN/"results.csv").is_file(): missing.append("training results.csv")
    for name in ("best.pt","last.pt"):
        if not (RUN/"weights"/name).is_file(): missing.append(name)
    exits=[read_json(p) for p in (OUT/"dispatches").glob("*/exit.json")]
    if not any(x.get("exit_code")==0 and x.get("status")=="TRAINING_COMPLETED" for x in exits): missing.append("successful worker Python exit")
    return dict(status="COMPLETE" if not missing else "INCOMPLETE",missing=missing)


def pack(status_snapshot):
    OUT.mkdir(parents=True,exist_ok=True)
    state=completeness(); stamp=time.strftime("%Y%m%d_%H%M%S")
    filename=OUT/f"GEO_v1_{state['status']}_{stamp}.tar.gz"
    require(not filename.exists(),"Package already exists; preserved")
    manifest=[]
    with tarfile.open(filename,"w:gz") as archive:
        def add_bytes(name,content):
            entry=tarfile.TarInfo(name); entry.size=len(content); entry.mtime=int(time.time())
            archive.addfile(entry,io.BytesIO(content))
            manifest.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        def add_file(name,path):
            entry=archive.gettarinfo(str(path),arcname=name)
            with path.open("rb") as f: archive.addfile(entry,f)
            manifest.append(dict(path=name,bytes=path.stat().st_size,sha256=sha256(path)))
        add_bytes("README.md",("# GEO v1 evidence\nStatus: "+state["status"]+"\nMissing: "+json.dumps(state["missing"])+
            "\nRaw datasets and weight bodies are NOT included. Complete ordinary-query predictions and GT are required for COMPLETE.\n"
            "P means detection Precision, not Accuracy. Native split best-F1 reports and fixed-val-threshold reports are distinct.\n"
            "No formal training result or AP benefit is implied by diagnostic PASS. Manifest excludes its own hash.\n").encode())
        add_bytes("formula.json",json.dumps(FORMULA,indent=2).encode())
        for name in git("ls-files").splitlines():
            p=ROOT/name
            if p.is_file(): add_file("source/"+name,p)
        for prefix,folder in (("evidence",OUT),("training",RUN)):
            if not folder.exists(): continue
            for p in sorted(folder.rglob("*")):
                if not p.is_file() or p==filename or p.suffix in {".pt",".pth",".tmp"} or p.name.startswith("GEO_v1_"): continue
                add_file(prefix+"/"+p.relative_to(folder).as_posix(),p)
        summary=dict(completeness=state,status=status_snapshot,weights=[file_info(RUN/"weights"/n) for n in ("best.pt","last.pt")],
            includes_weights=False,base=BASE,reliability_reference=RELIABILITY,code=code_identity(clean=False))
        add_bytes("summary.json",json.dumps(summary,ensure_ascii=False,allow_nan=False,default=str,indent=2).encode())
        # Deliberately do not append manifest.json to its own inventory.
        content=json.dumps(manifest,indent=2).encode(); entry=tarfile.TarInfo("manifest.json"); entry.size=len(content)
        archive.addfile(entry,io.BytesIO(content))
    with tarfile.open(filename) as archive:
        for row in manifest:
            h=hashlib.sha256(); size=0
            with archive.extractfile(row["path"]) as f:
                for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk); size+=len(chunk)
            require(size==row["bytes"] and h.hexdigest()==row["sha256"],"Archive integrity failure")
    return dict(file_info(filename),**state)
