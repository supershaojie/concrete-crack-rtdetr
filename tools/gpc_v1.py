"""GPC lifecycle: prepare/diagnose/preflight/status/start/resume/val/test/finish/pack.

No GPU-idle checks. No implicit formal training, packaging or downloads. All
reservations are local to this experiment's output directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ultralytics-main"))
os.environ["YOLO_AUTOINSTALL"]="false"
import torch
import ultralytics
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.gpc import GPCConfig
from init_c19_lif_v1 import initialize, runtime as parent_runtime, SOURCE_SHA256, build_training_model
from gpc_trainer import GPCTrainer, PreflightStop

BASE="a0459d6a652cb702699087c88fa39a3e4c4087ec"
REMOTE="https://github.com/supershaojie/concrete-crack-rtdetr.git"
BRANCH="exp-rtdetr-r18-lite-gpc-v1"
MAIN=Path(os.environ.get("GPC_MAIN","/root/autodl-tmp/projects/Crack_RTDETR"))
OUT=ROOT/"outputs/gpc_v1"
NAME="gpc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug"
RUN=ROOT/"runs/c_series"/NAME
INIT=ROOT/"weights/gpc_v1_controlled_init.pt"
SOURCE=MAIN/"weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
PARENT_RUN=MAIN/"runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug"
ALGORITHM=ROOT/"docs/gpc_v1/algorithm_config.yaml"
RECIPE=ROOT/"docs/gpc_v1/resolved_formal_config.yaml"
ALIASES=("/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1/weights/c19_lif_v1_controlled_init.pt",
         "/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/weights/c19_lif_v1_controlled_init.pt")


class Pending(RuntimeError):
    pass


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def read(path,default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def write(path,value):
    GPCTrainer._write_json(path,value)


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True,timeout=30).strip()


def runtime():
    info=parent_runtime()
    require(Path(ultralytics.__file__).resolve()==ROOT/"ultralytics-main/ultralytics/__init__.py","Wrong ultralytics import")
    info.update(model_class="ultralytics.models.rtdetr.gpc.GPCDetectionModel",
                criterion_class="ultralytics.models.utils.loss.RTDETRDetectionLoss",
                trainer_class="gpc_trainer.GPCTrainer",branch=git("branch","--show-current"))
    return info


def functional_digest(training_only=False):
    # Text-normalized hashes are invariant to Windows checkout CRLF; documentation prose is excluded.
    paths=sorted((ROOT/"ultralytics-main/ultralytics").rglob("*.py"))
    if training_only:
        # AutoBackend is only used by independent/final evaluation, not the training main/aux forward.
        paths=[p for p in paths if p.relative_to(ROOT).as_posix()!="ultralytics-main/ultralytics/nn/autobackend.py"]
    paths+=sorted((ROOT/"ultralytics-main/ultralytics/cfg").rglob("*.yaml"))
    paths+=([ROOT/"tools/gpc_trainer.py"] if training_only else sorted((ROOT/"tools").glob("*gpc*.py")))
    paths += [ROOT/"tools"/name for name in ("init_c19_lif_v1.py","init_lif_down.py","lif_down_topology.py","c19_lif_v1_data.py")]
    entries={str(path.relative_to(ROOT).as_posix()):hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for path in paths}
    entries["algorithm"]=digest(YAML.load(ALGORITHM))
    entries["recipe"]=digest(YAML.load(RECIPE))
    return digest(entries)


def verify_repository():
    require(git("remote","get-url","origin")==REMOTE,"Unexpected repository origin")
    require(git("branch","--show-current")==BRANCH,"Use the GPC experiment branch")
    subprocess.run(["git","merge-base","--is-ancestor",BASE,"HEAD"],cwd=ROOT,check=True,timeout=30)
    require(ROOT.resolve()!=MAIN.resolve(),"Use an isolated experiment worktree")


def recipe():
    archived=YAML.load(ROOT/"docs/c19_lif_v1/resolved_formal_config.yaml")
    if not (PARENT_RUN/"args.yaml").is_file():
        raise Pending("Missing authoritative parent recipe: "+str(PARENT_RUN/"args.yaml"))
    actual=YAML.load(PARENT_RUN/"args.yaml")
    normalized=deepcopy(actual)
    alias=None
    if normalized.get("model") in ALIASES:
        alias=normalized["model"]
        normalized["model"]=ALIASES[0]
    differences={k:[archived.get(k),normalized.get(k)] for k in set(archived)|set(normalized)
                 if type(archived.get(k)) is not type(normalized.get(k)) or archived.get(k)!=normalized.get(k)}
    require(not differences,"Authoritative parent recipe differs: "+json.dumps(differences))
    target=deepcopy(archived)
    target.update(model=str(INIT),data=str(MAIN/"configs/crack_autodl.yaml"),project=str(RUN.parent),name=NAME,save_dir=str(RUN))
    pinned=YAML.load(RECIPE)
    # Exact fixed server locations are the delivered contract; no arbitrary recipe/path overrides.
    require(target==pinned and all(type(target[k]) is type(pinned[k]) for k in target),"Resolved GPC recipe/path differs from pinned file")
    diff=[dict(field=k,parent=archived[k],gpc=target[k],reason="GPC experiment identity/output path") for k in target if target[k]!=archived[k]]
    require({x["field"] for x in diff}<={"model","project","name","save_dir","data"},"Non-identity recipe change")
    return target,dict(fields_checked=len(archived),exact_model_alias=alias,changes=diff)


def data_identity(data_path):
    resolved=check_det_dataset(str(data_path),autodownload=False)
    require(resolved["nc"]==1 and resolved["names"]=={0:"crack"},"Dataset must be nc=1 crack")
    dataset=Path(resolved["path"]).resolve()
    for split in ("train","val","test"):
        require(Path(resolved[split]).resolve()==dataset/"images"/split,"Unexpected split path/identity: "+split)
    # Cheap metadata inventory permits reuse of an already verified content audit.
    files=sorted(p for folder in (dataset/"images",dataset/"labels") for p in folder.rglob("*")
                 if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".txt"})
    signature=digest([(p.relative_to(dataset).as_posix(),p.stat().st_size,p.stat().st_mtime_ns) for p in files])
    cached=read(OUT/"data_identity.json",{})
    if cached.get("metadata_signature")==signature and cached.get("dataset")==str(dataset) and cached.get("yaml_sha256")==sha(data_path):
        print("Dataset audit REUSED",flush=True)
        return cached
    from c19_lif_v1_data import dataset_inventory
    inventory=dataset_inventory(dataset)
    for split,expected in dict(train=(6048,45573),val=(1728,12840),test=(864,6663)).items():
        require((inventory[split]["images"],inventory[split]["boxes"])==expected,"Dataset count differs; inspect without changing labels: "+split)
    # Hash bytes once, without decoding images or evaluating the network.
    content=digest([(p.relative_to(dataset).as_posix(),sha(p)) for p in files])
    result=dict(dataset=str(dataset),yaml_sha256=sha(data_path),inventory=inventory,content_sha256=content,metadata_signature=signature)
    write(OUT/"data_identity.json",result)
    return result


def current_identity(args,data):
    config=YAML.load(ALGORITHM)
    require(set(config)==set(GPCConfig.__dataclass_fields__) and GPCConfig.from_dict(config)==GPCConfig(),"Formal GPC v1 must use the complete fixed enabled configuration")
    return dict(functional_sha256=functional_digest(),algorithm=YAML.load(ALGORITHM),recipe=args,
                training_source_sha256=functional_digest(training_only=True),
                source_sha256=SOURCE_SHA256,init_sha256=sha(INIT),data_sha256=data["content_sha256"],
                data_yaml_sha256=data["yaml_sha256"],run=str(RUN),output=str(OUT),
                runtime_versions=dict(python=sys.version.split()[0],torch=str(torch.__version__),cuda=torch.version.cuda,ultralytics=ultralytics.__version__))


def verified_prepared(for_evaluation=False):
    prepared=read(OUT/"prepared.json")
    if not prepared or prepared.get("status")!="READY":
        raise Pending("Run prepare successfully first")
    verify_repository()
    args,_=recipe()
    data=data_identity(args["data"])
    identity=current_identity(args,data)
    if identity!=prepared["identity"]:
        # Completed weights keep their original training identity. An evaluation/tool-only repair
        # may supplement failed evaluation, while the new evaluation digest invalidates its cache.
        old={k:v for k,v in prepared["identity"].items() if k!="functional_sha256"}
        current={k:v for k,v in identity.items() if k!="functional_sha256"}
        require(for_evaluation and (RUN/"training_finished.json").is_file() and old==current,
                "Source/config/init/data identity changed; prepare and preflight again")
        prepared=dict(prepared,evaluation_functional_sha256=identity["functional_sha256"],
                      note="Completed training identity preserved; supplement evaluation with current tool/backend code")
    return prepared


@contextmanager
def experiment_lock(wait_seconds=0):
    # OS advisory lock automatically releases on process exit, even after OOM.
    if os.name!="posix":
        raise Pending("Server lifecycle actions require Linux; local checks run with check_gpc_v1.py")
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"experiment.lock").open("a+") as stream:
        deadline=time.monotonic()+wait_seconds
        while True:
            try:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic()>=deadline:
                    raise RuntimeError("This GPC output/run already has an active worker") from error
                time.sleep(.1)  # Only a bounded same-experiment dispatch handshake, never a GPU wait.
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def prepare():
    verify_repository()
    info=runtime()
    missing=[str(p) for p in (SOURCE,PARENT_RUN/"args.yaml",MAIN/"configs/crack_autodl.yaml") if not p.is_file()]
    if missing:
        write(OUT/"prepare_status.json",dict(status="PENDING",missing=missing,runtime=info))
        raise Pending("Missing server inputs: "+", ".join(missing))
    require(sha(SOURCE)==SOURCE_SHA256,"Public untrained source hash mismatch")
    args,comparison=recipe()
    data=data_identity(args["data"])
    old=read(OUT/"initialization.json",{})
    if INIT.exists():
        require(old.get("output_sha256")==sha(INIT) and old.get("source_sha256")==SOURCE_SHA256,"Unknown initialization exists; preserved")
        print("Controlled initialization REUSED",flush=True)
    else:
        write(OUT/"initialization.json",initialize(SOURCE,INIT))
    # Exercise the parent's audited nc80 -> nc1 loader now, before formal dispatch.
    weights=torch_load(INIT,map_location="cpu")["model"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        _,audit=build_training_model(weights.yaml,weights,dict(nc=1,channels=3))
    write(OUT/"nc1_prepare_audit.json",audit)
    write(OUT/"recipe_comparison.json",comparison)
    identity=current_identity(args,data)
    frozen=read(OUT/"prepared.json")
    if frozen and RUN.exists() and frozen["identity"]!=identity:
        checked=verified_prepared(for_evaluation=True)
        write(OUT/"evaluation_environment.json",checked)
        print("Completed training identity preserved; val/test may supplement evaluation.",flush=True)
        return
    write(OUT/"prepared.json",dict(status="READY",identity=identity,runtime=info,base=BASE))
    print("READY:",OUT/"prepared.json",flush=True)


def process_info(pid):
    try:
        proc=Path("/proc")/str(pid)
        return dict(pid=int(pid),start_token=(proc/"stat").read_text().rsplit(")",1)[1].split()[19],
                    cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode().strip(),
                    cwd=str((proc/"cwd").resolve()),run=str(RUN))
    except (OSError,ValueError,IndexError):
        return None


def live_worker(record):
    saved=record.get("process") or {}
    now=process_info(saved.get("pid",0))
    return bool(now and now==saved and str(ROOT/"tools/gpc_v1.py") in now["cmd"] and "_worker" in now["cmd"])


def status():
    print("GPC run:",RUN,"\nOutput:",OUT,flush=True)
    for name in ("prepared","diagnose","preflight","dispatch","worker"):
        record=read(OUT/(name+".json"),{})
        if name=="worker" and record:
            record=dict(record,actual_worker_alive=live_worker(record))
            if record.get("status")=="RUNNING" and not record["actual_worker_alive"]:
                record["observed_status"]="EXITED_WITHOUT_FINAL_RECORD"
        if name=="dispatch" and record.get("exit_file"):
            record["shell_exit_codes"]=read(record["exit_file"],{})
        fields=("status","observed_status","actual_worker_alive","process","action","reason","error","conclusion",
                "seconds","microbatches","effective_updates","skipped_updates","session","window","log","exit_code","shell_exit_codes")
        compact={k:record[k] for k in fields if k in record}
        if record.get("identity"):compact["functional_sha256"]=record["identity"]["functional_sha256"]
        print(name+":",json.dumps(compact or {"status":"PENDING / not recorded"},ensure_ascii=False,indent=2),flush=True)
    for name in ("training_finished.json","best_selection.json"):
        print(name+":",json.dumps(read(RUN/name,{}),ensure_ascii=False),flush=True)
    csv=RUN/"results.csv"
    if csv.is_file():
        with csv.open("rb") as stream:
            stream.seek(max(0,csv.stat().st_size-4000))
            print(stream.read().decode(errors="replace"),flush=True)
    print("A retained tmux pane is not proof of a live training process.",flush=True)


def worker_train(action,prepared):
    args=deepcopy(prepared["identity"]["recipe"])
    if action=="start":
        require(not RUN.exists(),"Existing formal run preserved; inspect status/use valid resume")
    else:
        checkpoint=RUN/"weights/last.pt"
        require(not (RUN/"training_finished.json").exists(),"Training completed; supplement evaluation only")
        require(checkpoint.is_file(),"Missing same-run last checkpoint")
        ckpt=torch_load(checkpoint,map_location="cpu")
        require(ckpt.get("optimizer") is not None and ckpt.get("scaler") is not None and 0<=ckpt.get("epoch",-1)<199,"Checkpoint is incomplete/stripped/completed")
        saved=ckpt.get("ema") or ckpt.get("model")
        require(getattr(saved,"gpc_identity",None)==prepared["identity"],"Resume run identity differs")
        args.update(model=str(checkpoint),resume=str(checkpoint))
    trainer=GPCTrainer(overrides=args,algorithm=prepared["identity"]["algorithm"],identity=prepared["identity"])
    trainer.train()
    print("Training finished:",json.dumps(read(RUN/"training_finished.json",{})),flush=True)
    print("Best training val:",json.dumps(read(RUN/"best_selection.json",{})),flush=True)
    print("Best:",RUN/"weights/best.pt","\nIndependent FP32 test is a separate action.",flush=True)


def lock_best(prepared):
    require((RUN/"training_finished.json").is_file(),"Training has not recorded completion")
    best=RUN/"weights/best.pt"
    selection=read(RUN/"best_selection.json")
    require(best.is_file() and selection,"Missing val-selected best/selection record; last is never substituted")
    require(selection["identity"]==prepared["identity"],"Best selection run identity differs")
    result=dict(best=str(best),best_sha256=sha(best),epoch_zero_based=selection["epoch_zero_based"],
                selector=selection["selector"],identity=prepared["identity"])
    locked=read(OUT/"best_lock.json")
    require(not locked or locked==result,"Locked best changed")
    if not locked:
        write(OUT/"best_lock.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def evaluate_split(split,prepared):
    from gpc_evaluate import evaluate,print_metrics,PROTOCOL,EVAL
    locked=lock_best(prepared)
    evaluation_code=functional_digest()
    key=digest(dict(locked,split=split,protocol=PROTOCOL,settings=EVAL,evaluation_functional_sha256=evaluation_code))
    folder=OUT/"evaluation"/split
    prior=read(folder/"latest.json",{})
    if prior.get("status")=="SUCCESS" and prior.get("identity_key")==key:
        if all(Path(path).is_file() and sha(path)==checksum for path,checksum in prior.get("artifacts",{}).items()) and prior.get("artifacts"):
            print("REUSED",flush=True)
            print_metrics(prior)
            return prior
        print("Prior metrics exist but artifacts are incomplete; a new bounded split attempt is necessary.",flush=True)
    attempt=folder/("attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
    report=dict(locked,split=split,identity_key=key,status="RUNNING",attempt=str(attempt),
                evaluation_functional_sha256=evaluation_code,runtime=runtime())
    write(folder/"latest.json",report)
    try:
        result=evaluate(locked["best"],prepared["identity"]["recipe"]["data"],split,attempt)
        require(sha(locked["best"])==locked["best_sha256"],"Best changed during evaluation")
        report.update(result,status="SUCCESS",exit_code=0)
        report["artifacts"]={str(path):sha(path) for path in sorted(attempt.rglob("*")) if path.is_file()}
        write(attempt/"metrics.json",report)
        # Bind the report itself as a mandatory artifact in the latest pointer.
        report["artifacts"][str(attempt/"metrics.json")]=sha(attempt/"metrics.json")
        print_metrics(report)
    except BaseException as error:
        report.update(status="RESOURCE_ERROR" if isinstance(error,torch.cuda.OutOfMemoryError) else "FAIL",error=repr(error),exit_code=1)
        raise
    finally:
        write(folder/"latest.json",report)
    return report


def preflight_worker(prepared):
    start=time.monotonic()
    attempt=OUT/"preflight_runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    report=dict(status="PENDING",identity=prepared["identity"],budget=dict(microbatches=16,seconds=900),runtime=runtime())
    trainer=None
    try:
        if not torch.cuda.is_available():
            raise Pending("CUDA unavailable")
        # Algorithm and real-model integration checks within this same total watchdog budget.
        from check_gpc_v1 import run as checks
        cpu_integration=checks("cpu",attempt/"integration_cpu.json")
        require(cpu_integration["status"]=="PASS","Exact CPU native-path integration checks failed")
        integration=checks("cuda",attempt/"integration.json")
        require(integration["status"]=="PASS","Algorithm/native-path integration checks failed")
        args=deepcopy(prepared["identity"]["recipe"])
        args.update(project=str(attempt),name="native_run",save_dir=str(attempt/"native_run"))
        torch.cuda.reset_peak_memory_stats()
        trainer=GPCTrainer(overrides=args,algorithm=prepared["identity"]["algorithm"],identity=prepared["identity"],preflight=True)
        trainer.gpc_deadline=start+900
        try:
            trainer.train()
        except PreflightStop as error:
            report["stop_reason"]=str(error)
        updates=trainer.gpc_updates
        rows=trainer.gpc_rows
        effective=sum(x["effective"] for x in updates)
        active=any(x.get("auxiliary_images")==8 for x in rows)
        changed=any(x["effective"] and x["gradients_finite"] and x["auxiliary_images"]==8 and x["final_regression_bias_max_change"]>0 for x in updates)
        report.update(status="TECHNICAL_PASS" if effective and changed and active and rows else "PENDING",
                      microbatches=len(rows),effective_updates=effective,skipped_updates=len(updates)-effective,
                      updates=updates,rows=rows,integration=integration,cpu_integration=cpu_integration,auxiliary_B8_observed=active,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if report["status"]=="PENDING":
            report["reason"]="Budget ended without all required effective-update/B8/parameter-change evidence"
    except Pending as error:
        report.update(status="PENDING",reason=str(error))
    except BaseException as error:
        report.update(status="RESOURCE_ERROR" if isinstance(error,torch.cuda.OutOfMemoryError) else "FAIL",error=repr(error))
        raise
    finally:
        report.update(seconds=time.monotonic()-start,attempt=str(attempt),formal_run_touched=False)
        write(OUT/"preflight.json",report)
        print("Preflight:",report["status"],OUT/"preflight.json",flush=True)
    return report["status"]=="TECHNICAL_PASS"


def bounded(action):
    # One owned subprocess and a hard wall-clock watchdog; no GPU/process polling or retries.
    prepared=verified_prepared()
    log=OUT/(action+".log")
    command=[sys.executable,"-u",str(ROOT/"tools/gpc_v1.py"),"_worker",action]
    with experiment_lock():
        previous=read(OUT/(action+".json"))
        if previous:
            write(OUT/"history"/(action+"_"+str(time.time_ns())+".json"),previous)
        write(OUT/(action+".json"),dict(status="RUNNING",identity=prepared["identity"],
              started=datetime.now(timezone.utc).isoformat(),log=str(log)))
    with log.open("a",encoding="utf-8") as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=child_environment(),start_new_session=True)
        print(f"{action}: PID {child.pid}; <=900 seconds; log {log}",flush=True)
        try:
            code=child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            # Kill only this invocation's private process group (including its data-loader children).
            import signal
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=5)
            old=read(OUT/(action+".json"),{})
            old.update(status="PENDING",reason="900 second watchdog budget exhausted",identity=prepared["identity"],log=str(log))
            write(OUT/(action+".json"),old)
            raise Pending("Bounded budget ended; see "+str(log))
    report=read(OUT/(action+".json"),{})
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if code:
        if report.get("status")=="RESOURCE_ERROR":
            raise torch.cuda.OutOfMemoryError(f"{action} worker reported a real resource error; see {log}")
        if report.get("status")=="FAIL":
            raise RuntimeError(f"{action} worker failed; see {log}")
        raise Pending(f"{action} did not pass (worker exit {code}); see {log}")


def child_environment():
    result=os.environ.copy()
    result.update(PYTHONPATH=str(ROOT/"ultralytics-main")+os.pathsep+str(ROOT/"tools"),
                  YOLO_AUTOINSTALL="false",PYTHONUNBUFFERED="1",GPC_MAIN=str(MAIN))
    return result


def dispatch(action):
    prepared=verified_prepared(for_evaluation=action in ("val","test","finish"))
    if action in ("start","resume"):
        preflight=read(OUT/"preflight.json",{})
        require(preflight.get("status")=="TECHNICAL_PASS" and preflight.get("identity")==prepared["identity"],"Valid same-identity TECHNICAL_PASS preflight required")
        diagnosis=read(OUT/"diagnose.json",dict(status="PENDING",reason="Mother best diagnostic not available"))
        if diagnosis.get("identity")!=prepared["identity"]:
            diagnosis=dict(status="PENDING",reason="No diagnosis for this functional identity",previous_conclusion=diagnosis.get("conclusion"))
        print("Mechanism diagnosis:",json.dumps(diagnosis,ensure_ascii=False),flush=True)
        if action=="start":
            require(not RUN.exists(),"Formal run already exists; preserved")
        else:
            require(not (RUN/"training_finished.json").exists(),"Training complete; use val/test, never restart it")
    require(shutil.which("tmux"),"tmux is required to retain the worker page")
    require(not live_worker(read(OUT/"worker.json",{})),"GPC worker already active")
    pending=read(OUT/"dispatch.json",{})
    if pending.get("status")=="DISPATCHED":
        # A worker can have exited without updating dispatch; its attempt exit evidence resolves that case.
        exit_path=Path(pending.get("exit_file","/nonexistent"))
        require(exit_path.is_file(),"Previous dispatch has no exit evidence; inspect its own pane/log before retry")
    session="gpc-v1-training" if action in ("start","resume") else "gpc-v1-"+action
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    folder=OUT/"workers"/(action+"_"+stamp)
    folder.mkdir(parents=True,exist_ok=False)
    script=folder/"worker.sh";log=folder/"console.log";exit_file=folder/"exit.json"
    command=[sys.executable,"-u",str(ROOT/"tools/gpc_v1.py"),"_worker",action]
    env=child_environment()
    keys=("PYTHONPATH","YOLO_AUTOINSTALL","PYTHONUNBUFFERED","GPC_MAIN","PATH","LD_LIBRARY_PATH","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","OMP_NUM_THREADS","MKL_NUM_THREADS","CUBLAS_WORKSPACE_CONFIG")
    exports="".join(("export "+k+"="+shlex.quote(env[k]) if k in env else "unset "+k)+"\n" for k in keys)
    script.write_text("#!/usr/bin/env bash\nset -uo pipefail\ncd "+shlex.quote(str(ROOT))+"\n"+exports+
        "set +e\n"+shlex.join(command)+" 2>&1 | tee "+shlex.quote(str(log))+"\n"+
        'codes=("${PIPESTATUS[@]}")\nprintf \'{"python":%s,"tee":%s}\\n\' "${codes[0]}" "${codes[1]}" > '+shlex.quote(str(exit_file))+"\n"+
        'printf "\\nGPC worker finished. Python=%s tee=%s\\n" "${codes[0]}" "${codes[1]}"\n'+
        'if (( codes[0] != 0 )); then exit "${codes[0]}"; fi\nexit "${codes[1]}"\n',encoding="utf-8")
    if subprocess.run(["tmux","has-session","-t","="+session],capture_output=True).returncode:
        subprocess.run(["tmux","new-session","-d","-s",session,"-c",str(ROOT),"bash --noprofile --norc"],check=True)
        target=session+":0"
    else:
        target=subprocess.check_output(["tmux","new-window","-d","-P","-F","#{window_id}","-t","="+session,"-n",action+"-"+stamp,"-c",str(ROOT),"bash --noprofile --norc"],text=True).strip()
    subprocess.run(["tmux","set-window-option","-t",target,"remain-on-exit","on"],check=True)
    write(OUT/"dispatch.json",dict(status="DISPATCHED",action=action,session=session,window=target,exit_file=str(exit_file),log=str(log),identity=prepared["identity"]))
    subprocess.run(["tmux","send-keys","-t",target,"exec bash "+shlex.quote(str(script)),"C-m"],check=True)
    print("Dispatched. Attach:","tmux attach -t "+session,flush=True)


def worker(action):
    with experiment_lock(wait_seconds=10):
        prepared=verified_prepared(for_evaluation=action in ("val","test","finish"))
        record=dict(status="RUNNING",action=action,process=process_info(os.getpid()),identity=prepared["identity"],started=datetime.now(timezone.utc).isoformat())
        write(OUT/"worker.json",record)
        code=1
        try:
            if action in ("start","resume"):
                preflight=read(OUT/"preflight.json",{})
                require(preflight.get("status")=="TECHNICAL_PASS" and preflight.get("identity")==prepared["identity"],"Worker requires same-identity technical preflight")
                worker_train(action,prepared)
            elif action=="preflight":
                if not preflight_worker(prepared):
                    raise Pending("Technical preflight PENDING")
            elif action=="diagnose":
                from gpc_diagnose import diagnose
                report=diagnose(prepared)
                write(OUT/"diagnose.json",report)
                if report["status"]!="COMPLETE":
                    raise Pending(report.get("reason","Diagnosis incomplete"))
            elif action in ("val","test"):
                evaluate_split(action,prepared)
            elif action=="finish":
                evaluate_split("val",prepared)
                evaluate_split("test",prepared)
            code=0
            record["status"]="SUCCESS"
        except Pending as error:
            code=2;record.update(status="PENDING",reason=str(error))
            raise
        except BaseException as error:
            record.update(status="RESOURCE_ERROR" if isinstance(error,torch.cuda.OutOfMemoryError) else "FAIL",error=repr(error))
            raise
        finally:
            record.update(exit_code=code,finished=datetime.now(timezone.utc).isoformat())
            write(OUT/"worker.json",record)
            print("Worker:",record["status"],"exit",code,flush=True)
            if action in ("start","resume"):
                print("Training completion:",json.dumps(read(RUN/"training_finished.json",{})),flush=True)
                print("Best training val / zero-based epoch:",json.dumps(read(RUN/"best_selection.json",{})),flush=True)
                print("Best checkpoint:",RUN/"weights/best.pt","; independent FP32 test is separate.",flush=True)


def pack():
    # Deliberately independent of prepared/torch inference/data availability.
    missing=[]
    required=[OUT/"prepared.json",OUT/"diagnose.json",OUT/"preflight.json",RUN/"results.csv",
              RUN/"training_finished.json",OUT/"best_lock.json",OUT/"evaluation/val/latest.json",OUT/"evaluation/test/latest.json"]
    for path in required:
        if not path.is_file():
            missing.append(str(path))
    for split in ("val","test"):
        report=read(OUT/f"evaluation/{split}/latest.json",{})
        if report.get("status")!="SUCCESS":
            missing.append(split+" successful evaluation")
        if report.get("identity")!=read(OUT/"prepared.json",{}).get("identity") or report.get("best_sha256")!=read(OUT/"best_lock.json",{}).get("best_sha256"):
            missing.append(split+" matching run/best identity")
        for path,checksum in report.get("artifacts",{}).items():
            if not Path(path).is_file() or sha(path)!=checksum:
                missing.append("missing/changed artifact: "+path)
    if (read(OUT/"preflight.json",{}).get("status")!="TECHNICAL_PASS" or
        read(OUT/"preflight.json",{}).get("identity")!=read(OUT/"prepared.json",{}).get("identity")):
        missing.append("valid technical preflight")
    if (read(OUT/"diagnose.json",{}).get("status")!="COMPLETE" or
        read(OUT/"diagnose.json",{}).get("identity")!=read(OUT/"prepared.json",{}).get("identity")):
        missing.append("complete bounded mechanism diagnosis")
    state="INCOMPLETE" if missing else "COMPLETE"
    stage=OUT/"pack_metadata"
    stage.mkdir(parents=True,exist_ok=True)
    (stage/"source.patch").write_text(git("diff","--binary",BASE,"HEAD"),encoding="utf-8")
    write(stage/"version.json",dict(base=BASE,head=git("rev-parse","HEAD"),branch=git("branch","--show-current"),remote=git("remote","get-url","origin"),functional_sha256=functional_digest()))
    write(stage/"weights.json",{str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in (RUN/"weights/best.pt",RUN/"weights/last.pt") if path.is_file()})
    files={}
    for folder in (OUT,RUN,ROOT/"docs/gpc_v1"):
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix not in {".pt",".pth",".lock"} and not path.name.endswith(".tmp") and "packages" not in path.parts and "preflight_runs" not in path.parts:
                    files[path.relative_to(ROOT).as_posix()]=path
    # Keep bounded preflight/diagnostic evidence, excluding any weights or dataset pictures.
    for path in (OUT/"preflight_runs").rglob("*") if (OUT/"preflight_runs").exists() else []:
        if path.is_file() and path.suffix in {".json",".jsonl",".csv",".yaml",".log"}:
            files[path.relative_to(ROOT).as_posix()]=path
    manifest={name:dict(sha256=sha(path),bytes=path.stat().st_size) for name,path in sorted(files.items())}
    report=dict(status=state,missing=missing,files=manifest,weights_included=False,raw_data_included=False)
    package_dir=OUT/"packages";package_dir.mkdir(exist_ok=True)
    key=digest(report)[:16];archive=package_dir/f"gpc_v1_{state}_{key}.tar.gz"
    manifest_path=package_dir/f"gpc_v1_{state}_{key}.manifest.json"
    write(manifest_path,report)
    if not archive.exists():
        with tarfile.open(archive,"w:gz") as tar:
            for name,path in sorted(files.items()):
                tar.add(path,arcname=name,recursive=False)
            tar.add(manifest_path,arcname="MANIFEST.json")
    with tarfile.open(archive,"r:gz") as tar:
        for name,item in manifest.items():
            stream=tar.extractfile(name)
            require(stream is not None and hashlib.sha256(stream.read()).hexdigest()==item["sha256"],"Archive verification failed: "+name)
    archive.with_suffix(archive.suffix+".sha256").write_text(sha(archive)+"  "+archive.name+"\n",encoding="utf-8")
    print(state,archive,"\nMissing:",missing,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","diagnose","preflight","status","start","resume","val","test","finish","pack","_worker"))
    parser.add_argument("worker_action",nargs="?")
    args=parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.action=="status":
            status()
        elif args.action=="_worker":
            require(args.worker_action in ("diagnose","preflight","start","resume","val","test","finish"),"Invalid worker action")
            worker(args.worker_action)
        elif args.action in ("diagnose","preflight"):
            bounded(args.action)
        else:
            with experiment_lock():
                if args.action=="prepare":prepare()
                elif args.action=="pack":pack()
                else:dispatch(args.action)
    except Pending as error:
        print("PENDING:",error,flush=True)
        return 2
    except BaseException as error:
        print("RESOURCE_ERROR" if isinstance(error,torch.cuda.OutOfMemoryError) else "FAIL",repr(error),flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
