"""Prepare, preflight, smoke, cache, select, explain, render and pack Chapter 4."""
from __future__ import annotations
import argparse
import codecs
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .common import (IDS, ROOT, fingerprint, git_head, model_environment, path_value,
                     prepare, progress, read_json, run_lock, sha256, worker_command, write_json)
from .cache import ARCHIVED, ground_truth, read_cache
from .analysis import propose, statistics
from .cam import METHOD_VERSION
from .pack import pack
from .render import candidate_previews, matrix


class Pipeline:
    def __init__(self,args):
        self.args=args;self.out=args.out.resolve();self.out.mkdir(parents=True,exist_ok=True)
        previous=read_json(self.out/"run.json") if (self.out/"run.json").exists() else {}
        source=args.handoff or previous.get("handoff")
        if not source:raise ValueError("First invocation requires --handoff IMPLEMENTATION_OR_FINAL.zip")
        self.base,self.integrity=prepare(source,self.out)
        self.registry=read_json(self.base/"model_registry.json")
        self.config=read_json(args.config or (self.out/"config.resolved.json" if (self.out/"config.resolved.json").exists() else ROOT/"configs/chapter4_visualization.json"))
        if self.config["model_ids"]!=IDS or self.config["sample_count"] not in (4,5,6):
            raise ValueError("Default model identities/sample count contract differs")
        if args.threshold is not None:self.config["display_threshold"]=args.threshold
        if args.sample_count is not None:self.config["sample_count"]=args.sample_count
        if not 0<self.config["display_threshold"]<1:raise ValueError("Invalid shared display threshold")
        fixed_cam=dict(method=METHOD_VERSION,batch=1,weighting_epsilon=1e-6,gradient_floor=1e-12,
                       positive_response_floor=1e-12,colormap="JET-256-linear-segments-v1",
                       layer_policy="fixed equal-weight stride8/16/32 classification-connected detection features v1",
                       target="arithmetic mean of the same fixed GT-associated native crack detection probabilities",
                       normalization="per-scale positive min-max; equal-weight mean including unused=zero; full-image min-max")
        if any(self.config["cam"].get(k)!=v for k,v in fixed_cam.items()):
            raise ValueError("Frozen CAM implementation/protocol differs; new methods require a distinct implementation version")
        if not 300<=self.config["layout"]["dpi"]<=600:raise ValueError("Paper raster DPI must be 300–600")
        if not 0<self.config["cam"]["minimum_correspondence_iou"]<=1:raise ValueError("Invalid correspondence IoU")
        if args.overrides:
            overrides=read_json(args.overrides)
            for mid,values in overrides.items():
                model=next(m for m in self.registry["models"] if m["id"]==mid)
                for key,value in values.items():
                    if key not in model["paths"]:raise ValueError("Unknown path override: "+key)
                    model["paths"][key]={"absolute_path":value,"evidence_level":"EXPLICIT_VISUALIZATION_PATH_OVERRIDE",
                                           "original_record":model["paths"][key]}
        elif previous and (self.out/"registry.resolved.json").exists():
            self.registry=read_json(self.out/"registry.resolved.json")
        extensions=self.out/"registry.extensions.json"
        if extensions.exists():
            for model in read_json(extensions):
                if model["id"] not in [m["id"] for m in self.registry["models"]]:
                    self.registry["models"].insert(-1,model)
        self.models=self.registry["models"]
        self.mids=[m["id"] for m in self.models]
        if self.mids[0]!=IDS[0] or self.mids[-1]!=IDS[-1]:raise ValueError("Faster first / ours last invariant lost")
        self.registry["fixed_model_ids"]=self.mids
        self.registry["fixed_model_order"]=[m["model"] for m in self.models]
        self.images=ground_truth(self.base)
        self.image_root=Path(args.image_root or previous.get("image_root") or self.registry["common_test_data"]["test_images"]["absolute_path"]).resolve()
        self.predictions={};self.provenance={}
        # Changes to presentation never trigger full-test inference. CAM keys use relevant semantics only.
        self.cam_protocol=dict(config=self.config["cam"],implementation={name:sha256(Path(__file__).with_name(name))
                             for name in ("worker.py","cam.py","geometry.py","analysis.py","common.py")})
        self.protocol=fingerprint(self.cam_protocol)
        config_fp=fingerprint(self.config)
        if (self.out/"config.resolved.json").exists():
            old=read_json(self.out/"config.resolved.json")
            if fingerprint(old)!=config_fp:write_json(self.out/"cases"/("config.previous."+fingerprint(old)[:12]+".json"),old)
        self.run=dict(handoff=str(self.base),original_registry_sha256=sha256(self.base/"model_registry.json"),
                      image_root=str(self.image_root),output=str(self.out),visualization_head=git_head(ROOT),
                      formal_source_commits={m["id"]:m.get("code_commit") for m in self.models},
                      config_fingerprint=config_fp,cam_protocol_fingerprint=self.protocol,
                      original_metrics_unchanged=True,deferred_models=["D-FINE-M"],
                      inference_or_gpu_validation_claimed_by_prepare=False)
        write_json(self.out/"run.json",self.run)
        write_json(self.out/"config.resolved.json",self.config)
        write_json(self.out/"registry.resolved.json",self.registry)
        write_json(self.out/"reports/input_integrity.json",self.integrity)

    def update(self,stage,completed=0,total=1,**details):
        value=dict(stage=stage,completed=completed,total=total,percent=round(100*completed/max(1,total),2),
                   utc=datetime.now(timezone.utc).isoformat(),**details)
        write_json(self.out/"progress.json",value)
        print(f"[{stage}] {completed}/{total} ({value['percent']}%) "+str(details),flush=True)

    def selected_models(self):
        selected=[m for m in self.models if not self.args.model_id or m["id"] in self.args.model_id]
        if self.args.model_id and set(self.args.model_id)-set(self.mids):raise ValueError("Unknown --model-id")
        return selected

    def dispatch(self,model,action,images=None,**extra):
        directory=self.out/"requests"/model["id"];directory.mkdir(parents=True,exist_ok=True)
        request_file=directory/(action+".json")
        response=self.out/"responses"/model["id"]/(action+".json")
        request=dict(model=model,action=action,handoff=str(self.base),output=str(self.out),response=str(response),
                     images=list(images or self.images.values()),image_root=str(self.image_root),
                     config=self.config,device="cpu" if action=="preflight" else self.args.device,
                     protocol_fingerprint=self.protocol)
        request.update(extra)
        write_json(request_file,request)
        env=model_environment(model,self.out)
        if action=="preflight":env["CUDA_VISIBLE_DEVICES"]=""
        log=self.out/"logs"/(model["id"]+"."+action+".log");log.parent.mkdir(parents=True,exist_ok=True)
        command=worker_command(model,request_file)
        with log.open("a",encoding="utf-8",newline="") as stream:
            stream.write("\nARGV "+json.dumps(command,ensure_ascii=False)+"\n");stream.flush()
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     bufsize=-1)
            decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
            # Binary read1 preserves TQDM's carriage returns and streams short updates
            # immediately; universal-newline line iteration would turn them into spam.
            while True:
                block=process.stdout.read1(4096)
                if not block:break
                chunk=decoder.decode(block)
                print(chunk,end="",flush=True);stream.write(chunk);stream.flush()
            tail=decoder.decode(b"",final=True)
            if tail:print(tail,end="",flush=True);stream.write(tail)
            code=process.wait()
        result=read_json(response) if response.is_file() else dict(status="failed",error="worker did not write response")
        if code or (result.get("status")!="passed" and action!="cam"):
            raise RuntimeError(f"{model['id']} {action} failed; see {log}: {result.get('error','per-image mismatch')}")
        return result

    def preflight(self):
        self.update("preflight")
        data=read_json(self.base/"data/test_file_records.json")["records"]
        label_root=self.image_root.parent.parent/"labels/test"
        records=[]
        for row in data:
            image=self.image_root/Path(row["image"]).name
            label=label_root/Path(row["label"]).name
            records.append(dict(image=row["image"],image_bytes_match=image.is_file() and image.stat().st_size==row["image_bytes"],
                                label_sha_match=label.is_file() and sha256(label)==row["label_sha256"]))
        all_data=all(r["image_bytes_match"] and r["label_sha_match"] for r in records)
        report=dict(input_integrity=self.integrity,scope="CPU preflight; no CAM/forward validation claim",
                    fixed_test=dict(records=records,all_image_sizes_and_label_sha_match=all_data),models={})
        if not self.args.offline and not self.args.skip_server_collector:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONUTF8="1")
            log=self.out/"logs/readonly_collector.log";log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("w",encoding="utf-8") as stream:
                subprocess.run([sys.executable,"-X","utf8",str(self.base/"scripts/collect_server_evidence.py"),
                                "--registry",str(self.base/"model_registry.json"),"--out-parent",str(self.out/"server_evidence")],
                               env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            report["readonly_collector_log"]=str(log)
        failures=[]
        # Ours identity is checked first; publication/table order remains unchanged.
        check_order=sorted(self.selected_models(),key=lambda m:m["id"]!="lcbr_detr")
        for i,model in enumerate(check_order):
            try:
                if self.args.offline:
                    value=dict(status="pending_server",weight=model["checkpoint"]["sha256"],model_restored=False)
                else:value=self.dispatch(model,"preflight",images=[])
            except Exception as error:
                value=dict(status="failed",error=str(error));failures.append(model["id"])
            report["models"][model["id"]]=value
            self.update("preflight",i+1,len(check_order),model=model["id"],status=value["status"])
            write_json(self.out/"reports/preflight.json",report)
        report["status"]="offline_only" if self.args.offline else "failed" if failures or not all_data else "passed"
        write_json(self.out/"reports/preflight.json",report)
        if not self.args.offline and (failures or not all_data):raise ValueError("Server preflight failed; see reports/preflight.json")
        return report

    def source_cache(self,model):
        mid=model["id"]
        if mid in ARCHIVED:return self.base/ARCHIVED[mid],model["prediction_sha256"],"formal_archive"
        path=path_value(model,"test_predictions")
        if path and Path(path).is_file():
            expected=model.get("prediction_sha256")
            seal=path_value(model,"prediction_seal") or path_value(model,"test_metrics")
            if seal and Path(seal).is_file():
                receipt=read_json(seal)
                if mid=="lcbr_detr":
                    if receipt.get("status")!="completed" or receipt.get("checkpoint_sha256")!=model["checkpoint"]["sha256"] or receipt.get("images")!=864 or receipt.get("export_complete") is not True:
                        raise ValueError("Ours formal cache completion/weight receipt differs")
                    expected=receipt.get("predictions_gt_sha256")
                elif receipt.get("status")!="completed" or receipt.get("images")!=864 or receipt.get("sha256")!=expected:
                    raise ValueError("Formal prediction completion receipt differs: "+mid)
            elif mid=="lcbr_detr":
                write_json(self.out/"reports/ours_unverified_existing_cache.json",dict(path=str(path),
                    status="preserved_unverified",reason="No original completion/identity receipt; use original-entry re-export in new run"))
                return None,None,"unverified_original_cache_preserved"
            else:raise ValueError("Formal server cache lacks verified completion receipt: "+mid)
            if not expected:raise ValueError("No verified expected cache SHA: "+mid)
            return Path(path),expected,"formal_server_cache"
        return None,None,"missing"

    def prediction_stage(self):
        failures=[]
        for i,model in enumerate(self.models):
            mid=model["id"]
            try:
                source,expected,kind=self.source_cache(model)
                signature=fingerprint(dict(id=mid,weight=model["checkpoint"]["sha256"],code=model.get("code_commit"),
                                           preprocessing=model.get("preprocessing"),source_sha=expected,adapter=sha256(Path(__file__).with_name("cache.py")),
                                           export_implementation=None if source else {name:sha256(Path(__file__).with_name(name)) for name in ("worker.py","geometry.py","common.py")}))
                folder=self.out/"predictions"/mid/signature[:20]
                normalized=folder/"normalized.json.gz";receipt_path=folder/"provenance.json"
                if normalized.is_file() and receipt_path.is_file():
                    receipt=read_json(receipt_path)
                    if sha256(normalized)!=receipt["normalized_sha256"]:raise ValueError("Normalized cache was altered")
                    if source and sha256(source)!=receipt["source_file_sha256"]:raise ValueError("Formal source cache was altered")
                    with gzip.open(normalized,"rt",encoding="utf-8") as stream:records=json.load(stream)
                else:
                    if source:
                        records,provenance=read_cache(source,model,self.images,expected)
                    elif self.args.offline:
                        raise FileNotFoundError("Formal cache requires server: "+mid)
                    else:
                        if mid=="lcbr_detr":
                            result=self.dispatch(model,"formal-export",destination=str(folder/"native_export"))
                            source=Path(result["prediction_file"])
                            records,provenance=read_cache(source,model,self.images,result["sha256"])
                        else:
                            result=self.dispatch(model,"predictions")
                            records={key:[{**p,"prediction_index":j,"raw":p} for j,p in enumerate(rows)] for key,rows in result["images"].items()}
                            if set(records)!=set(self.images):raise ValueError("Re-export test set incomplete")
                            provenance=dict(native_identity=result["identity"],response_path=str(self.out/"responses"/mid/"predictions.json"))
                        kind="new_visualization_reexport"
                    folder.mkdir(parents=True,exist_ok=True)
                    temporary=normalized.with_name(normalized.name+".partial")
                    with gzip.open(temporary,"wt",encoding="utf-8") as stream:json.dump(records,stream,separators=(",",":"),allow_nan=False)
                    temporary.replace(normalized)
                    receipt=dict(provenance=provenance,kind=kind,source_file_sha256=sha256(source) if source else None,
                                 normalized_sha256=sha256(normalized),signature=signature,weight_sha256=model["checkpoint"]["sha256"])
                    write_json(receipt_path,receipt)
                self.predictions[mid]=records;self.provenance[mid]={**receipt,"normalized_path":str(normalized)}
            except Exception as error:
                failures.append(dict(model_id=mid,error=repr(error)))
            self.update("predictions",i+1,len(self.models),model=mid,available=mid in self.predictions)
        write_json(self.out/"reports/prediction_sources.json",dict(models=self.provenance,failures=failures))
        if failures and not self.args.offline:raise ValueError("Incomplete eight-model predictions; see reports/prediction_sources.json")
        return self.predictions

    def smoke(self):
        if not self.predictions:self.prediction_stage()
        if self.args.offline:raise ValueError("Smoke requires actual native model inference; offline archive validation is separate")
        images=sorted(self.images.values(),key=lambda im:im["id"])[:self.config["smoke_sample_count"]]
        if not any(im["width"]!=im["height"] for im in images):raise ValueError("Smoke must include a non-square image")
        old=read_json(self.out/"reports/smoke.json") if (self.out/"reports/smoke.json").exists() else {}
        previous_models=old.get("models",{}) if old.get("protocol_fingerprint")==self.protocol else {}
        report=dict(images=[im["file_name"] for im in images],tolerances=self.config["smoke_tolerance"],models=previous_models)
        failures=[]
        for i,model in enumerate(self.selected_models()):
            try:
                if model["id"] not in self.predictions:raise ValueError("Missing native reference cache")
                result=self.dispatch(model,"smoke",images=images,reference={im["file_name"]:self.predictions[model["id"]][im["file_name"]] for im in images})
                result["reference_kind"]=self.provenance[model["id"]]["kind"]
                result["reference_provenance"]=self.provenance[model["id"]]
                report["models"][model["id"]]=result
            except Exception as error:
                failures.append(model["id"]);report["models"][model["id"]]=dict(status="failed",error=str(error))
            self.update("smoke",i+1,len(self.selected_models()),model=model["id"])
            write_json(self.out/"reports/smoke.json",report)
        report["status"]="failed" if failures else "passed";report["protocol_fingerprint"]=self.protocol
        write_json(self.out/"reports/smoke.json",report)
        if failures:raise ValueError("Cache/direct-forward smoke failed for "+str(failures)+"; do not relax tolerances automatically")
        return report

    def validation(self):
        """Freeze/check the same structural three-scale rule on validation images first."""
        from PIL import Image
        import numpy as np
        v8=next(m for m in self.models if m["id"]=="yolov8m")
        # val.txt is produced beside registered test.txt by the pinned data exporter.
        listing=Path(self.args.validation_list or Path(path_value(v8,"test_image_list")).with_name("val.txt"))
        if not listing.is_file():raise ValueError("Validation list pending; supply --validation-list (no test layer search)")
        paths=[Path(s.strip()) for s in listing.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not paths or not all(p.is_absolute() for p in paths):raise ValueError("Expected formal absolute validation image list")
        if len({p.parent for p in paths})!=1:raise ValueError("Validation images must share the registered split directory")
        val_root=paths[0].parent;labels=val_root.parent.parent/"labels/val"
        images=[];cases=[]
        for p in sorted(paths):
            with Image.open(p) as im:w,h=im.size
            boxes=[]
            for j,line in enumerate((labels/(p.stem+".txt")).read_text(encoding="utf-8").splitlines()):
                values=list(map(float,line.split()))
                if len(values)!=5 or values[0]!=0:raise ValueError("Invalid validation crack labels")
                _,x,y,bw,bh=values
                boxes.append(dict(id=f"val:{p.stem}:{j}",category_id=1,bbox=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
            if not boxes:continue
            key="images/val/"+p.name
            images.append(dict(file_name=key,width=w,height=h,annotations=boxes))
            target=max(boxes,key=lambda r:(r["bbox"][2]-r["bbox"][0])*(r["bbox"][3]-r["bbox"][1]))
            cases.append(dict(image=key,target_gt_ids=[target["id"]],display_threshold=self.config["display_threshold"]))
            if len(images)>=self.config["validation_sample_count"] and any(im["width"]!=im["height"] for im in images):break
        if len(images)!=self.config["validation_sample_count"] or not any(im["width"]!=im["height"] for im in images):
            raise ValueError("Three deterministic validation images including non-square are required")
        old=read_json(self.out/"layer_protocol.json") if (self.out/"layer_protocol.json").exists() else {}
        previous_models=old.get("models",{}) if old.get("protocol_fingerprint")==self.protocol else {}
        report=dict(status="passed",protocol_fingerprint=self.protocol,validation_list=str(listing),
                    validation_list_sha256=sha256(listing),case_selection="first sorted nonempty validation images; largest GT instance; no layer/appearance search",models=previous_models)
        for model in self.selected_models():
            result=self.dispatch(model,"cam",images=images,cases=cases,image_root=str(val_root),
                                 destination=str(self.out/"validation_cam"/model["id"]/self.protocol[:20]))
            report["models"][model["id"]]=result
            if result["status"]!="passed":report["status"]="failed"
            write_json(self.out/"layer_protocol.json",report)
        if report["status"]!="passed":raise ValueError("Fixed layer rule failed validation; inspect validation CAM metadata, no per-test layer tuning")
        return report

    def candidates(self):
        if not self.predictions:self.prediction_stage()
        if set(self.mids)!=set(self.predictions):raise ValueError("Selection requires complete eight-model pool")
        details=statistics(self.images,self.predictions,self.mids,self.config["display_threshold"],self.out/"statistics")
        proposed,document=propose(self.images,self.predictions,details,self.mids,self.image_root,self.config["display_threshold"],
                                  self.out/"candidates",count=self.config["sample_count"],candidates=self.config["candidate_count"])
        candidate_previews(document,self.images,self.predictions,self.models,self.image_root,self.out/"previews",self.config["layout"])
        write_json(self.out/"cases/cases.proposed.json",proposed)
        write_json(self.out/"cases/review.template.json",dict(reviewer="FILL_AFTER_REAL_IMAGE_REVIEW",cases=[
            dict(image=c["image"],gt_confirmed=False,scene_tags=[],notes="",preview_sha256=sha256(self.out/"previews"/(Path(c["image"]).stem+".png"))) for c in proposed["cases"]]))
        return proposed

    def cases(self):
        path=self.args.cases or self.out/"cases/cases.reviewed.json"
        if not Path(path).is_file():path=self.out/"cases/cases.proposed.json"
        if not Path(path).is_file():return self.candidates()
        doc=read_json(path)
        if len(doc["cases"]) not in (4,5,6) or len({c["image"] for c in doc["cases"]})!=len(doc["cases"]):raise ValueError("Cases must have 4–6 unique samples")
        for c in doc["cases"]:
            if c["image"] not in self.images:raise ValueError("Unknown canonical case")
            if c["display_threshold"]!=self.config["display_threshold"]:raise ValueError("Cases and common display threshold differ")
            image=self.images[c["image"]];region=c["view_region"]
            if len(region)!=4 or not (0<=region[0]<region[2]<=image["width"] and 0<=region[1]<region[3]<=image["height"]):raise ValueError("Invalid common case region")
            if not c["target_gt_ids"] or not set(c["target_gt_ids"]).issubset({g["id"] for g in image["annotations"]}):raise ValueError("Invalid fixed target GT identity")
            expected=c.get("image_identity",{}).get("sha256")
            if expected and expected!=sha256(self.image_root/Path(c["image"]).name):raise ValueError("Case image changed since proposal/visual review")
        if doc["status"]=="reviewed" and not doc.get("visual_review_evidence"):raise ValueError("Reviewed status requires explicit real image/GT review evidence")
        write_json(self.out/"cases"/("cases.active."+fingerprint(doc)[:16]+".json"),doc)
        return doc

    def cam(self):
        if self.args.offline:raise ValueError("CAM requires real pinned native environments")
        layer=self.out/"layer_protocol.json"
        if not layer.exists() or not self.has_verified_models(read_json(layer)):
            self.validation()
        smoke=self.out/"reports/smoke.json"
        if not smoke.exists() or not self.has_verified_models(read_json(smoke)):self.smoke()
        doc=self.cases();index=read_json(self.out/"cam_index.json") if (self.out/"cam_index.json").exists() else {}
        total=len(self.selected_models())*len(doc["cases"]);completed=0
        for model in self.selected_models():
            mid=model["id"];index.setdefault(mid,{})
            tasks=[]
            for case in doc["cases"]:
                key=case["image"]
                cachekey=self.cam_key(model,case)
                destination=self.out/"cam"/mid/cachekey[:24]
                meta=destination/(Path(key).stem+".json");npz=meta.with_suffix(".npz")
                if meta.is_file() and npz.is_file() and read_json(meta).get("status")=="passed" and read_json(meta).get("cache_key")==cachekey and read_json(meta).get("npz_sha256")==sha256(npz):
                    entry=dict(status="passed",npz=str(npz),metadata=str(meta),cache_key=cachekey)
                    index[mid][key]=entry;completed+=1
                else:
                    tasks.append(dict(case=case,key=key,cachekey=cachekey,destination=destination,meta=meta,npz=npz))
            write_json(self.out/"cam_index.json",index)
            if tasks:
                result=self.dispatch(model,"cam",images=[self.images[t["key"]] for t in tasks],cases=[t["case"] for t in tasks],
                    destination=str(self.out/"cam"/mid),destinations={t["key"]:str(t["destination"]) for t in tasks},
                    progress_context=dict(completed=completed,total=total))
                for task in tasks:
                    key,cachekey,meta,npz,destination=(task[k] for k in ("key","cachekey","meta","npz","destination"))
                    entry=result["images"][key]
                    if entry["status"]=="passed":
                        data=read_json(meta);data.update(cache_key=cachekey,npz_sha256=sha256(npz));write_json(meta,data)
                        entry["cache_key"]=cachekey
                    else:
                        entry["failure_file"]=str(destination/(Path(key).stem+".failed.json"))
                        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        write_json(self.out/"reports/cam_failures"/mid/(stamp+"_"+Path(key).stem+".json"),
                                   dict(model_id=mid,weight_sha256=model["checkpoint"]["sha256"],source_commit=model.get("code_commit"),
                                        cache_key=cachekey,result=entry))
                    index[mid][key]=entry
                completed+=len(tasks)
            self.update("cam",completed,total,model=mid,status="completed_model",computed=len(tasks),reused=len(doc["cases"])-len(tasks))
            write_json(self.out/"cam_index.json",index)
        return index

    def has_verified_models(self,report):
        if report.get("status")!="passed" or report.get("protocol_fingerprint")!=self.protocol:return False
        for model in self.selected_models():
            result=report.get("models",{}).get(model["id"],{})
            identity=result.get("identity",{})
            if result.get("status")!="passed" or identity.get("weight_sha256")!=model["checkpoint"]["sha256"] or identity.get("registry_record_fingerprint")!=fingerprint(model):return False
        return True

    def cam_key(self,model,case):
        return fingerprint(dict(model_id=model["id"],weight=model["checkpoint"]["sha256"],source=model.get("code_commit"),
            registry_record_fingerprint=fingerprint(model),preprocessing=model.get("preprocessing"),protocol=self.protocol,
            image_sha256=sha256(self.image_root/Path(case["image"]).name),target_gt_ids=case["target_gt_ids"],
            correspondence_display_threshold=case["display_threshold"]))

    def render(self):
        if not self.predictions:self.prediction_stage()
        statistics_path=self.out/"statistics/per_image.json"
        old=read_json(statistics_path) if statistics_path.exists() else {}
        if old.get("model_order")!=self.mids or old.get("display_threshold")!=self.config["display_threshold"]:
            statistics(self.images,self.predictions,self.mids,self.config["display_threshold"],self.out/"statistics")
        doc=self.cases()
        index=read_json(self.out/"cam_index.json") if (self.out/"cam_index.json").exists() else {}
        for model in self.models:
            for case in doc["cases"]:
                entry=index.get(model["id"],{}).get(case["image"])
                if entry and entry.get("status")=="passed":
                    if entry.get("cache_key")!=self.cam_key(model,case) or not Path(entry["metadata"]).is_file() or not Path(entry["npz"]).is_file() or read_json(entry["metadata"]).get("npz_sha256")!=sha256(entry["npz"]):
                        index[model["id"]][case["image"]]={"status":"stale_or_altered_cam"}
        write_json(self.out/"cam_index.json",index)
        records={}
        for heatmap,stem in ((False,"detection_comparison"),(True,"heatmap_comparison")):
            records[stem]=matrix(doc,self.images,self.predictions,self.models,self.image_root,self.out/"figures"/stem,
                                 cam_index=index,heatmap=heatmap,config={**self.config["layout"],"full_panel_dir":str(self.out/"selected")},draft=True)
        for pending in records["heatmap_comparison"]["pending"]:
            mid,key=pending["model_id"],pending["image"]
            if pending["status"]=="render_failed":
                index.setdefault(mid,{}).setdefault(key,{}).update(status="render_failed",render_error=pending["error"])
                write_json(self.out/"reports/cam_render_failures"/mid/(Path(key).stem+".json"),pending)
        write_json(self.out/"cam_index.json",index)
        for case in doc["cases"]:
            key=case["image"];name=Path(key).name
            original=self.out/"selected/originals"/name;original.parent.mkdir(parents=True,exist_ok=True)
            if not original.exists():shutil.copyfile(self.image_root/name,original)
            write_json(self.out/"selected/annotations"/(Path(key).stem+".json"),dict(case=case,
                       image_sha256=sha256(original),gt=self.images[key]["annotations"],
                       model_predictions={mid:self.predictions[mid][key] for mid in self.mids}))
        write_json(self.out/"reports/render.json",records)
        return records

    def review(self):
        if not self.args.review_record:raise ValueError("--review-record with real preview/GT observations is required")
        doc=self.cases();review=read_json(self.args.review_record)
        if not review.get("reviewer") or "FILL" in review["reviewer"]:raise ValueError("Record actual reviewer")
        evidence={r["image"]:r for r in review["cases"]}
        for case in doc["cases"]:
            row=evidence[case["image"]]
            preview=self.out/"previews"/(Path(case["image"]).stem+".png")
            if not row.get("gt_confirmed") or not row.get("scene_tags") or not row.get("notes") or row["preview_sha256"]!=sha256(preview):
                raise ValueError("Incomplete real image/GT/scene review evidence: "+case["image"])
        doc.update(status="reviewed",visual_review_evidence=review)
        write_json(self.out/"cases/cases.reviewed.json",doc)
        return doc

    def add_model(self):
        if not self.args.model_record:raise ValueError("--model-record must contain a future formal registry record and adapter_entry")
        record=read_json(self.args.model_record)
        for key in ("id","model","run_id","checkpoint","paths","code_commit","adapter_entry"):
            if key not in record:raise ValueError("Missing extension field "+key)
        if record["id"] in IDS or "ssd" in record["id"].lower():raise ValueError("Cannot replace fixed identities or add SSD")
        if len(record["checkpoint"]["sha256"])!=64:raise ValueError("Formal extension weight SHA required")
        path=self.out/"registry.extensions.json"
        entries=read_json(path) if path.exists() else []
        entries=[r for r in entries if r["id"]!=record["id"]]+[record]
        write_json(path,entries)
        print("Registered future model before ours. Existing cases/prediction/CAM caches retained. Run preflight/predictions, smoke --model-id ID, cam --model-id ID, render, pack.",flush=True)

    def full(self):
        self.preflight();self.prediction_stage()
        if self.args.cases:
            if not (self.out/"statistics/per_image.json").exists():self.candidates()
            self.cases()
        else:self.candidates()
        self.render()  # First reviewable pair always exists before the expensive explanation pass.
        try:
            self.smoke();self.validation();self.cam()
        except Exception as error:
            write_json(self.out/"reports/full_failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                       original_results_unchanged=True,cam_complete=False))
            self.render();pack(self.out)
            raise
        self.render()
        index=read_json(self.out/"cam_index.json")
        failed=[dict(model_id=m["id"],image=c["image"]) for m in self.models for c in self.cases()["cases"]
                if index.get(m["id"],{}).get(c["image"],{}).get("status")!="passed"]
        write_json(self.out/"reports/full.json",dict(status="failed_cam_cells" if failed else "computed_pending_visual_review" if self.cases()["status"]!="reviewed" else "reviewed",
                    model_order=self.mids,selected_images=[c["image"] for c in self.cases()["cases"]],failed_cells=failed,
                    utc=datetime.now(timezone.utc).isoformat(),cam_protocol_fingerprint=self.protocol))
        receipt=pack(self.out)
        self.update("full",1,1,status="failed_cam_cells" if failed else "computed_pending_visual_review" if self.cases()["status"]!="reviewed" else "reviewed",failed=failed,package=receipt)
        if failed:raise ValueError("CAM failures retained in draft and package; inspect cam_index.json")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","preflight","smoke","predictions","candidates","validate-layers","cam","render","full","review","add-model","pack","status"])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--handoff",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--overrides",type=Path)
    parser.add_argument("--image-root",type=Path)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--cases",type=Path)
    parser.add_argument("--model-id",action="append")
    parser.add_argument("--model-record",type=Path)
    parser.add_argument("--review-record",type=Path)
    parser.add_argument("--validation-list",type=Path)
    parser.add_argument("--sample-count",type=int,choices=[4,5,6])
    parser.add_argument("--threshold",type=float)
    parser.add_argument("--offline",action="store_true",help="Archive/local-data checks only; no native model or GPU success claims")
    parser.add_argument("--skip-server-collector",action="store_true")
    args=parser.parse_args()
    if args.action=="status":
        print(json.dumps(read_json(args.out/"progress.json"),ensure_ascii=False,indent=2));return
    with run_lock(args.out):
        pipeline=Pipeline(args)
        try:
            if args.action=="prepare":print(json.dumps(pipeline.integrity,ensure_ascii=False,indent=2))
            elif args.action=="predictions":pipeline.prediction_stage()
            elif args.action=="validate-layers":pipeline.validation()
            elif args.action=="add-model":pipeline.add_model()
            elif args.action=="pack":pack(pipeline.out)
            else:getattr(pipeline,args.action)()
        except Exception as error:
            pipeline.update(args.action,status="failed",error=repr(error))
            write_json(pipeline.out/"reports"/(args.action+".failure.json"),dict(error=repr(error),traceback=traceback.format_exc()))
            raise


if __name__=="__main__":main()
