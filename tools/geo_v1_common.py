"""GEO identities and frozen recipe. Reuses the mother's audited initialization.

Data JSON class normalization, LF source hashing and bounded process cleanup
derive from ARG reliability revision ef9cb7e05e5557f7dd06c95cf2361998a284adc9.
No ARG model, loss, checkpoint or training state is imported.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"ultralytics-main"))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

from init_c19_lif_v1 import source_contract, initialize, SOURCE_SHA256, sha256, require, runtime as mother_runtime
from ultralytics.models.rtdetr.geo_loss import FORMULA
from ultralytics.models.rtdetr.geo_model import write_json
from ultralytics.utils import YAML

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
RELIABILITY = "ef9cb7e05e5557f7dd06c95cf2361998a284adc9"
BRANCH = "exp-rtdetr-r18-lite-geo-v1"
ORIGIN = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
RUN_NAME = "geo_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
SESSION = "geo-v1-training"
MAIN = Path(os.environ.get("GEO_V1_MAIN", "D:/MyProjects/Crack_RTDETR" if os.name=="nt" else "/root/autodl-tmp/projects/Crack_RTDETR"))
OUT = ROOT/"outputs/geo_v1"
RUN = MAIN/"runs/c_series"/RUN_NAME
INIT = OUT/"geo_v1_public_init.pt"
SOURCE = MAIN/"weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
SERVER_PYTHON = "/root/miniconda3/envs/rtdetr/bin/python"
COUNTS = dict(train=(6048,45573),val=(1728,12840),test=(864,6663))


def now(): return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True,timeout=40).strip()


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def runtime():
    import inspect
    import importlib.metadata
    import torch
    from ultralytics.models.rtdetr.geo_loss import GEODetectionLoss
    from ultralytics.models.rtdetr.geo_model import GEODetectionModel,GEOTrainer
    from ultralytics.models.rtdetr.geo_val import GEOValidator
    versions={}
    for package in ("numpy","torchvision","opencv-python","scipy","albumentations"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    return dict(mother_runtime(),packages=versions,criterion=inspect.getfile(GEODetectionLoss),model=inspect.getfile(GEODetectionModel),
        trainer=inspect.getfile(GEOTrainer),validator=inspect.getfile(GEOValidator),module_hashes=source_contract(),
        gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        formula=FORMULA,ddp="UNVERIFIED")


def code_identity(clean=True):
    require(git("remote","get-url","origin")==ORIGIN,"Unexpected origin; repository preserved")
    require(git("branch","--show-current") in (BRANCH,""),"Not GEO branch or pinned detached checkout")
    subprocess.run(["git","merge-base","--is-ancestor",BASE,"HEAD"],cwd=ROOT,check=True,timeout=30)
    dirty=git("status","--porcelain","--untracked-files=normal")
    if clean: require(not dirty,"GEO worktree has uncommitted changes; preserved; commit before server operations")
    names=git("ls-files","tools","ultralytics-main/ultralytics","configs","docs/c19_lif_v1/c2_args.yaml","docs/c19_lif_v1/c2_data.yaml").splitlines()
    hashes={p:hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in names}
    return dict(commit=git("rev-parse","HEAD"),source_lf_sha256=digest(hashes),files=hashes,dirty=dirty,base=BASE)


def canonical_config(value):
    result=dict(value)
    result["path"]=str(Path(value["path"]).resolve())
    result["names"]={str(k):v for k,v in value["names"].items()}
    require(len(result["names"])==len(value["names"]),"Colliding normalized class IDs")
    return result


def data_config():
    expected=YAML.load(ROOT/"docs/c19_lif_v1/c2_data.yaml")
    expected["path"]=str((MAIN/"datasets/crack_det").resolve())
    path=MAIN/"configs/crack_autodl.yaml" if os.name=="posix" else OUT/"local_data.yaml"
    if os.name=="nt" and not path.exists(): YAML.save(path,expected)
    require(path.is_file(),f"Missing original data YAML: {path}")
    require(canonical_config(YAML.load(path))==canonical_config(expected),"Dataset YAML differs from original crack_det split")
    return path


def directory_markers(config):
    root=Path(config["path"])
    folders=[root/config[s] for s in COUNTS]+[root/"labels"/s for s in COUNTS]
    return {str(p):p.stat().st_mtime_ns for p in folders}


def scan_data(path,manifest):
    """One explicit prepare scan only; never called by status/evaluate/pack."""
    import math
    from PIL import Image
    data=canonical_config(YAML.load(path)); root=Path(data["path"])
    rows=[]; splits={}
    for split,expected in COUNTS.items():
        folder=root/data[split]
        images=sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"})
        local=[]; count=0
        require(bool(images),f"Missing {split} images: {folder}")
        for index,image in enumerate(images):
            label=root/"labels"/split/image.relative_to(folder).with_suffix(".txt")
            values=[list(map(float,line.split())) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            require(all(len(v)==5 and v[0]==0 and all(math.isfinite(x) for x in v) and all(0<=x<=1 for x in v[1:])
                and v[3]>0 and v[4]>0 for v in values),f"Invalid original label: {label}")
            with Image.open(image) as im: shape=[im.height,im.width]
            local.append(dict(split=split,image=image.relative_to(root).as_posix(),image_sha256=sha256(image),
                label=label.relative_to(root).as_posix(),label_sha256=sha256(label),boxes=len(values),size_hw=shape))
            count+=len(values)
            if (index+1)%500==0 or index+1==len(images): print(f"GEO snapshot {split} {index+1}/{len(images)}",flush=True)
        require((len(images),count)==expected,f"{split} counts differ: {(len(images),count)} != {expected}")
        splits[split]=dict(images=len(images),boxes=count,content_sha256=digest(local)); rows.extend(local)
    with gzip.open(manifest,"wt",encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row,allow_nan=False)+"\n")
    return dict(config=data,config_sha256=sha256(path),content_sha256=digest(rows),splits=splits)


def validate_manifest(data,manifest):
    with gzip.open(manifest,"rt",encoding="utf-8") as f: rows=[json.loads(line) for line in f]
    require(digest(rows)==data["content_sha256"],"Saved data manifest hash mismatch")
    require(len({r["image"] for r in rows})==len(rows),"Duplicate image ID in manifest")
    for split,expected in COUNTS.items():
        local=[r for r in rows if r["split"]==split]
        require((len(local),sum(r["boxes"] for r in local))==expected,"Manifest split counts differ")
        require(digest(local)==data["splits"][split]["content_sha256"],"Manifest split digest differs")


def snapshot(path,refresh=False):
    target=OUT/"data_snapshot.json"; manifest=OUT/"data_manifest.jsonl.gz"
    prior=read_json(target)
    if prior and not refresh:
        verify_snapshot(path,prior)
        return prior
    require(not (OUT/"training_identity.json").exists(),"Dataset identity frozen by training dispatch")
    data=None; origin=None
    if not refresh:
        for folder in (MAIN.parent/"Crack_RTDETR-arg_v1/outputs/arg_v1",MAIN/"outputs/arg_v1"):
            for record_name,manifest_name in (("prepare.json","data_manifest.jsonl.gz"),("local_inventory.json","local_data_manifest.jsonl.gz")):
                record=folder/record_name; listing=folder/manifest_name
                old=read_json(record)
                if not old or not listing.is_file(): continue
                if record_name=="prepare.json" and old.get("status")!="PASS": continue
                candidate=old.get("data",old)
                if canonical_config(candidate["config"])!=canonical_config(YAML.load(path)): continue
                validate_manifest(candidate,listing)
                data=dict(candidate,config=canonical_config(YAML.load(path)),config_sha256=sha256(path))
                shutil.copyfile(listing,manifest)
                origin=dict(record=str(record),record_sha256=sha256(record),manifest=str(listing),method="reuse confirmed snapshot; no raw images/labels reread")
                break
            if data is not None: break
    if data is None:
        data=scan_data(path,manifest)
        origin=dict(method="explicit full prepare scan",created=now())
    result=dict(data=data,manifest_sha256=sha256(manifest),directory_markers=directory_markers(data["config"]),origin=origin,
        assumption="Dataset immutable after confirmed snapshot; in-place edits require prepare --refresh-data before dispatch")
    if prior: write_json(OUT/"history"/(time.strftime("%Y%m%d_%H%M%S")+"_previous_snapshot.json"),prior)
    write_json(target,result)
    return result


def verify_snapshot(path,record):
    require(canonical_config(YAML.load(path))==record["data"]["config"] and sha256(path)==record["data"]["config_sha256"],"Data YAML changed")
    require(sha256(OUT/"data_manifest.jsonl.gz")==record["manifest_sha256"],"Saved manifest changed")
    require(directory_markers(record["data"]["config"])==record["directory_markers"],"Dataset directories changed; investigate and explicitly refresh snapshot")


def recipe(data_path):
    source=YAML.load(ROOT/"docs/c19_lif_v1/c2_args.yaml")
    actual=MAIN/"runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
    if actual.exists():
        v=YAML.load(actual)
        require(v.keys()==source.keys() and all(type(v[k]) is type(x) and v[k]==x for k,x in source.items()),"Authoritative C2 archive drift")
    args=dict(source,model=str(INIT),data=str(data_path),project=str(RUN.parent),name=RUN_NAME,save_dir=str(RUN))
    rows=[dict(field=k,mother=v,geo=args[k],reason="experiment/path identity") for k,v in source.items() if v!=args[k]]
    require({r["field"] for r in rows}<={"model","data","project","name","save_dir"},"Illegal recipe change")
    return args,dict(fields=len(source),authoritative="docs/c19_lif_v1/c2_args.yaml",actual_archive_checked=actual.exists(),
        changes=rows,unchanged=[k for k in source if args[k]==source[k]],formula=FORMULA)


def prepare(refresh=False):
    OUT.mkdir(parents=True,exist_ok=True)
    code=code_identity(); source_contract()
    require(SOURCE.is_file() and sha256(SOURCE)==SOURCE_SHA256,"Missing/wrong public initialization")
    path=data_config(); data=snapshot(path,refresh)
    args,diff=recipe(path)
    if INIT.exists():
        provenance=read_json(OUT/"initialization.json",{})
        require(provenance.get("output_sha256")==sha256(INIT) and provenance.get("source_sha256")==SOURCE_SHA256,"Existing init provenance differs; preserved")
    else: write_json(OUT/"initialization.json",initialize(SOURCE,INIT))
    report=dict(status="PASS",created=now(),code=code,runtime=runtime(),args=args,data=data,
        source_sha256=SOURCE_SHA256,init_sha256=sha256(INIT),recipe_diff=diff)
    prior=read_json(OUT/"prepare.json")
    if prior:
        require(all(prior[k]==report[k] for k in ("args","init_sha256","source_sha256")),"Preparation identity drift")
        if prior["data"]!=data: require(refresh,"Unexpected data snapshot change")
    write_json(OUT/"prepare.json",report); write_json(OUT/"recipe_diff.json",diff)
    YAML.save(OUT/"train_args.yaml",args)
    return report


def binding():
    plan=read_json(OUT/"prepare.json",{})
    require(plan.get("status")=="PASS","Run prepare first")
    code=code_identity(); path=data_config(); verify_snapshot(path,plan["data"])
    args,_=recipe(path)
    require(args==plan["args"] and sha256(INIT)==plan["init_sha256"] and sha256(SOURCE)==SOURCE_SHA256,"Recipe/initialization changed")
    return dict(code=code,data=plan["data"],args_sha256=digest(args),init_sha256=plan["init_sha256"],formula=FORMULA)


def process_identity(pid):
    import psutil
    try:
        p=psutil.Process(pid)
        return dict(pid=pid,created=p.create_time(),command=p.cmdline())
    except (psutil.NoSuchProcess,psutil.AccessDenied): return None


def run_bounded(command,seconds,label):
    require(seconds>0,f"No time remaining for {label}")
    started=time.monotonic()
    proc=subprocess.Popen(command,cwd=ROOT,start_new_session=os.name!="nt")
    try:
        while True:
            remaining=seconds-(time.monotonic()-started)
            if remaining<=0: raise TimeoutError(f"{label}: exceeded {seconds:.1f}s boundary")
            try: return proc.wait(timeout=min(20,remaining))
            except subprocess.TimeoutExpired: print(f"GEO stage={label} elapsed={time.monotonic()-started:.1f}s remaining={remaining:.1f}s",flush=True)
    except BaseException:
        if proc.poll() is None:
            import psutil
            descendants=psutil.Process(proc.pid).children(recursive=True)
            for child in reversed(descendants):
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            if os.name!="nt": os.killpg(proc.pid,signal.SIGTERM)
            else: proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name!="nt": os.killpg(proc.pid,signal.SIGKILL)
                else: proc.kill()
                proc.wait(timeout=5)
            _,alive=psutil.wait_procs(descendants,timeout=1)
            for child in alive:
                try: child.kill()
                except psutil.NoSuchProcess: pass
        raise


def amp_resources():
    from ultralytics.utils import ASSETS
    result=[]
    for name,target,candidates in (("bus.jpg",ASSETS/"bus.jpg",[MAIN/"ultralytics-main/ultralytics/assets/bus.jpg"]),
        ("yolo26n.pt",ROOT/"yolo26n.pt",[MAIN/"yolo26n.pt",MAIN/"weights/yolo26n.pt"])):
        if not target.is_file():
            source=next((p for p in candidates if p.is_file()),None)
            require(source,f"Missing mother's offline AMP resource {name}; no downloads or skipped AMP checks")
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open("rb") as a,target.open("xb") as b: shutil.copyfileobj(a,b)
        result.append(dict(path=str(target),sha256=sha256(target)))
    return result
