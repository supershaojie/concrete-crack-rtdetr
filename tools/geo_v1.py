"""GEO v1 explicit lifecycle: prepare/preflight/status/start/resume/val/test/finish/pack."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from geo_v1_common import (ROOT,MAIN,OUT,RUN,INIT,SOURCE,SESSION,SERVER_PYTHON,FORMULA,
    read_json,write_json,require,now,runtime,sha256,binding,prepare,code_identity,
    process_identity,run_bounded,amp_resources)
from geo_v1_results import file_info,checkpoint_identity,evaluate,offline_summary,pack,completeness


def has_tmux():
    return bool(shutil.which("tmux") and subprocess.run(["tmux","has-session","-t","="+SESSION],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10).returncode==0)


def active_workers():
    import psutil
    found=[]
    for p in psutil.process_iter(["pid","cmdline","create_time"]):
        cmd=p.info["cmdline"] or []
        if "_worker" in cmd and any(str(ROOT/"tools/geo_v1.py").replace("\\","/")==x.replace("\\","/") for x in cmd): found.append(p.info)
    return found


@contextmanager
def operation_lock():
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/"operation.lock"
    if path.exists():
        old=read_json(path); live=process_identity(old["pid"])
        require(not live or live["created"]!=old["created"],f"GEO operation active: {old}")
        path.unlink()  # this experiment's verified stale lock only
    with path.open("x",encoding="utf-8") as f: json.dump(process_identity(os.getpid()),f)
    try: yield
    finally: path.unlink(missing_ok=True)


def status():
    dispatches=sorted((OUT/"dispatches").glob("*/dispatch.json"))
    latest=dispatches[-1].parent if dispatches else None
    return dict(run=str(RUN),session=SESSION,tmux=has_tmux(),active_workers=active_workers(),
        dispatch=read_json(latest/"dispatch.json") if latest else None,
        worker=read_json(latest/"worker.json") if latest else None,
        exit=read_json(latest/"exit.json") if latest else None,
        python_exit=(latest/"python_exit_code.txt").read_text().strip() if latest and (latest/"python_exit_code.txt").exists() else None,
        progress=read_json(OUT/"progress.json"),training=read_json(OUT/"training_completed.json"),
        preflight=read_json(OUT/"preflight.json",{}).get("status","NOT_RUN"),
        evaluations={s:read_json(OUT/f"{s}_lock.json",{}).get("status","NOT_RUN") for s in ("val","test")},
        weights=[file_info(RUN/"weights"/s,False) for s in ("best.pt","last.pt")],
        log=str(latest/"console.log") if latest else None)


def server_environment():
    import torch
    import numpy
    require(os.name=="posix" and Path(sys.executable).resolve()==Path(SERVER_PYTHON).resolve(),"Use the documented Linux server Python")
    require(torch.cuda.is_available(),"CUDA device 0 unavailable")
    require(sys.version_info[:2]==(3,10) and str(torch.__version__).startswith("2.1.2") and torch.version.cuda=="12.1" and numpy.__version__=="1.26.4",
        f"Mother environment mismatch: torch={torch.__version__}, CUDA={torch.version.cuda}, numpy={numpy.__version__}; no automatic dependency changes")
    return runtime()


def gpu_idle():
    require(shutil.which("nvidia-smi"),"nvidia-smi unavailable; cannot verify GPU occupancy")
    result=subprocess.check_output(["nvidia-smi","--id=0","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"],text=True,timeout=15)
    occupied=[line for line in result.splitlines() if line.strip() and line.split(",",1)[0].strip()!=str(os.getpid())]
    require(not occupied,"GPU 0 occupied; preserve all other processes and retry when free: "+"; ".join(occupied))


def preflight(seconds,micro_batches):
    require(1<=seconds<=900 and 4<=micro_batches<=16,"Boundary must be <=900s and 4..16 micro-batches")
    started=time.monotonic(); folder=OUT/"preflights"/time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(status="PENDING",start_eligible=False,started=now(),folder=str(folder),checks={},
        boundary=dict(seconds=seconds,micro_batches=micro_batches))
    write_json(OUT/"preflight.json",report)
    try:
        print("GEO preflight: identity and CPU formula/routing",flush=True)
        report["binding"]=binding()
        code=run_bounded([sys.executable,str(ROOT/"tools/check_geo_v1.py"),"--math-only","--output",str(folder/"cpu")],
            seconds-(time.monotonic()-started),"CPU math and routing")
        require(code==0,"CPU math/routing failed")
        report["checks"]["cpu"]=dict(status="PASS",report=str(folder/"cpu/checks.json"))
        if os.name!="posix":
            report["reason"]="Windows local checks do not qualify formal server B16/640 AMP; execute delivered server preflight"
            return report
        report["environment"]=server_environment(); gpu_idle(); report["amp_resources"]=amp_resources()
        write_json(folder/"request.json",dict(seconds=seconds-(time.monotonic()-started),micro_batches=micro_batches,binding=report["binding"]))
        code=run_bounded([sys.executable,str(ROOT/"tools/geo_v1_preflight.py"),"--folder",str(folder)],
            seconds-(time.monotonic()-started),"native server lifecycle")
        report["gpu"]=read_json(folder/"gpu.json",{})
        report["checks"].update(report["gpu"].get("checks",{}))
        require(code in (0,2),f"Server lifecycle failed with exit {code}")
        required=("cpu","cuda_b16_amp","native_scale","mechanism","resume","new_process_val")
        report["start_eligible"]=all(report["checks"].get(k,{}).get("status")=="PASS" for k in required)
        report["status"]="PASS" if report["start_eligible"] else "PENDING"
        return report
    except TimeoutError as error:
        report.update(status="PENDING",reason=str(error),start_eligible=False)
        return report
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report["ended"]=now(); report["elapsed_seconds"]=time.monotonic()-started
        write_json(folder/"preflight.json",report); write_json(OUT/"preflight.json",report)


def guarded_preflight(seconds,micro_batches):
    require(1<=seconds<=900 and 4<=micro_batches<=16,"Preflight maximum is 16 micro-batches/900 seconds")
    command=[sys.executable,str(ROOT/"tools/geo_v1.py"),"_preflight","--seconds",str(seconds),"--micro-batches",str(micro_batches)]
    try:
        code=run_bounded(command,seconds,"entire preflight including setup")
        report=read_json(OUT/"preflight.json",{})
        if code not in (0,2):
            report.update(status="FAIL",start_eligible=False,child_exit=code); write_json(OUT/"preflight.json",report)
        return report
    except TimeoutError as error:
        report=read_json(OUT/"preflight.json",{})
        report.update(status="PENDING",start_eligible=False,reason=str(error),ended=now())
        write_json(OUT/"preflight.json",report)
        return report


def verify_preflight(pre,current):
    require(pre.get("status")=="PASS" and pre.get("start_eligible"),"Preflight is not PASS/start eligible; see status and saved evidence")
    require(pre.get("binding")==current,"Preflight source/recipe/data/init identity differs")
    for name in ("cpu","cuda_b16_amp","native_scale","mechanism","resume","new_process_val"):
        require(pre.get("checks",{}).get(name,{}).get("status")=="PASS",f"Required preflight check is not PASS: {name}")
    gpu=pre.get("gpu",{}); cap=pre["checks"]["cuda_b16_amp"]
    require(0<gpu.get("micro_batches",0)<=16 and cap.get("effective_updates",0)>0 and cap.get("batch")==16
        and cap.get("imgsz")==640 and cap.get("amp") is True,"No bounded native B16/640 AMP optimizer update evidence")


def dispatch(resume=False):
    server_environment()
    require(not active_workers() and not has_tmux(),"GEO run active; use status; existing run preserved")
    require(shutil.which("tmux"),"tmux is required")
    current=binding(); verify_preflight(read_json(OUT/"preflight.json",{}),current)
    gpu_idle(); amp_resources()
    require(not (OUT/"training_completed.json").exists(),"Training completed; run finish, never restart training for eval recovery")
    identity=read_json(OUT/"training_identity.json")
    if resume:
        require(identity and identity["binding"]==current,"Resume requires this run's original code/recipe/data/init")
        checkpoint_identity(RUN/"weights/last.pt",identity,True)
    else:
        require(not RUN.exists(),"Existing run preserved; use status/resume")
        if identity:
            require(identity["binding"]==current and not list((OUT/"dispatches").glob("*/worker.claim")),
                "Existing worker history preserved; use status/resume")
        else:
            identity=dict(binding=current,run=str(RUN),session=SESSION,created=now())
            write_json(OUT/"training_identity.json",identity)
        if not (OUT/"training_source_snapshot.tar.gz").exists():
            subprocess.run(["git","archive","--format=tar.gz","--output="+str(OUT/"training_source_snapshot.tar.gz"),"HEAD"],cwd=ROOT,check=True,timeout=60)
    identifier=time.strftime("%Y%m%d_%H%M%S")+("_resume" if resume else "_start")
    folder=OUT/"dispatches"/identifier; folder.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-u",str(ROOT/"tools/geo_v1.py"),"_worker","--dispatch",identifier]
    record=dict(created=now(),resume=resume,command=command,cwd=str(ROOT),identity=identity,status="DISPATCHED")
    write_json(folder/"dispatch.json",record)
    environment=dict(PYTHONPATH=str(ROOT/"ultralytics-main")+os.pathsep+str(ROOT/"tools"),GEO_V1_MAIN=str(MAIN),PYTHONUNBUFFERED="1",YOLO_AUTOINSTALL="false")
    for k in ("PATH","LD_LIBRARY_PATH","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","CUBLAS_WORKSPACE_CONFIG"): environment[k]=os.environ.get(k)
    exports="".join(("unset "+k if v is None else "export "+k+"="+shlex.quote(v))+"\n" for k,v in environment.items())
    shell=("#!/usr/bin/env bash\nset -uo pipefail\ncd "+shlex.quote(str(ROOT))+"\n"+exports+
        shlex.join(command)+" 2>&1 | tee -a "+shlex.quote(str(folder/"console.log"))+"\n"+
        "codes=(\"${PIPESTATUS[@]}\")\nprintf '%s\\n' \"${codes[0]}\" > "+shlex.quote(str(folder/"python_exit_code.txt"))+"\n"+
        "printf '%s\\n' \"${codes[1]}\" > "+shlex.quote(str(folder/"tee_exit_code.txt"))+"\nexit \"${codes[0]}\"\n")
    (folder/"worker.sh").write_text(shell,encoding="utf-8",newline="\n")
    subprocess.run(["tmux","new-session","-d","-s",SESSION,"bash "+shlex.quote(str(folder/"worker.sh"))],check=True,timeout=15)
    return dict(status="DISPATCHED",session=SESSION,log=str(folder/"console.log"),worker_command=command)


def worker(identifier):
    from ultralytics.models.rtdetr.geo_model import GEOTrainer
    require(Path(identifier).name==identifier,"Invalid dispatch ID")
    folder=OUT/"dispatches"/identifier; record=read_json(folder/"dispatch.json")
    require(record and os.environ.get("TMUX"),"Worker requires recorded tmux dispatch")
    session=subprocess.check_output(["tmux","display-message","-p","-t",os.environ["TMUX_PANE"],"#S"],text=True,timeout=10).strip()
    require(session==SESSION,"Wrong tmux session")
    with (folder/"worker.claim").open("x") as stream: stream.write(str(os.getpid()))
    write_json(folder/"worker.json",dict(process_identity(os.getpid()),started=now(),command=record["command"]))
    result=dict(status="FAILED",exit_code=1,started=now())
    try:
        require(binding()==record["identity"]["binding"],"Identity changed after dispatch")
        amp_resources()
        args=deepcopy(read_json(OUT/"prepare.json")["args"])
        if record["resume"]:
            checkpoint_identity(RUN/"weights/last.pt",record["identity"],True)
            args["model"]=args["resume"]=str(RUN/"weights/last.pt")
        else: require(not RUN.exists(),"Run appeared after dispatch; preserved")
        trainer=GEOTrainer(overrides=args); trainer.geo_output=OUT
        def attach(t):
            t.model.geo_identity=deepcopy(record["identity"]); t.ema.ema.geo_identity=deepcopy(record["identity"])
            actual=vars(t.args)
            require(all(type(actual[k]) is type(v) and actual[k]==v for k,v in args.items()),"Effective complete training recipe drift")
        trainer.add_callback("on_pretrain_routine_end",attach)
        trainer.train()
        require((OUT/"training_completed.json").is_file(),"No native training completion event")
        result.update(status="TRAINING_COMPLETED",exit_code=0)
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc(),geometry=getattr(error,"details",None))
        raise
    finally:
        result.update(ended=now(),best=file_info(RUN/"weights/best.pt"),last=file_info(RUN/"weights/last.pt"))
        write_json(folder/"exit.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare").add_argument("--refresh-data",action="store_true",help="Explicitly rescan changed data before any dispatch")
    for name in ("status","start","resume","val","test","finish","pack"): sub.add_parser(name)
    for name in ("preflight","_preflight"):
        p=sub.add_parser(name); p.add_argument("--seconds",type=int,default=900); p.add_argument("--micro-batches",type=int,default=16)
    sub.add_parser("_worker").add_argument("--dispatch",required=True)
    args=parser.parse_args()
    if args.command=="status": result=status()
    elif args.command=="_worker": worker(args.dispatch); return
    elif args.command=="_preflight": result=preflight(args.seconds,args.micro_batches)
    else:
        with operation_lock():
            if args.command=="prepare": result=prepare(args.refresh_data)
            elif args.command=="preflight":
                require(not active_workers() and not has_tmux(),"GEO training active; preflight must remain isolated")
                result=guarded_preflight(args.seconds,args.micro_batches)
            elif args.command=="start": result=dispatch(False)
            elif args.command=="resume": result=dispatch(True)
            elif args.command=="pack": result=pack(status())
            elif args.command in ("val","test","finish"):
                require(not active_workers() and not has_tmux(),"Wait for this GEO worker/session to exit")
                server_environment()
                if args.command=="finish":
                    evaluate("val"); evaluate("test"); offline_summary(); result=pack(status())
                else: result=evaluate(args.command)
    def compact(x):
        if isinstance(x,dict): return {k:(f"{len(v)} entries; see saved JSON" if k in ("files","optimizer_groups") else compact(v)) for k,v in x.items()}
        if isinstance(x,list): return [compact(v) for v in x]
        return x
    print(json.dumps(compact(result),ensure_ascii=False,allow_nan=False,indent=2,default=str),flush=True)
    if args.command in ("preflight","_preflight") and not result.get("start_eligible"): raise SystemExit(2)


if __name__=="__main__": main()
