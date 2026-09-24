"""PDS F lifecycle CLI: no action implicitly starts training or final test."""
from __future__ import annotations
import argparse
import csv
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import traceback
import uuid
from pds_v1_common import *
from pds_v1_prepare import prepare, record_delivery, current_identity, require_gate, verify_delivery

SESSION="pds-v1-training"
SERVER_PYTHON="/root/miniconda3/envs/rtdetr/bin/python"


def live_processes():
    import psutil
    rows=[]
    for p in psutil.process_iter(["pid","cmdline","create_time"]):
        try:
            cmd=p.info["cmdline"] or []
            target=str(ROOT/"tools/pds_v1.py").replace("\\","/")
            if target in [str(v).replace("\\","/") for v in cmd] and "_worker" in cmd:
                children=[dict(pid=c.pid,command=c.cmdline()) for c in p.children(recursive=True)]
                rows.append(dict(pid=p.pid,command=cmd,created=p.create_time(),children=children))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return rows


def session_alive():
    return shutil.which("tmux") is not None and subprocess.run(
        ["tmux","has-session","-t","="+SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def no_active():
    require(not live_processes(),"This PDS worktree already has an active verified worker")
    require(not session_alive(),"Dedicated pds-v1-training tmux session still present")


@contextmanager
def operation_lock():
    import psutil
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/"operation.lock"
    if lock.exists():
        row=read_json(lock)
        if psutil.pid_exists(row["pid"]):
            require(abs(psutil.Process(row["pid"]).create_time()-row["created"])>1,"Another PDS operation is active")
        archive_file(lock)
    with lock.open("x",encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
    try:
        yield
    finally:
        lock.unlink()


def preflight():
    identity=current_identity()
    previous=OUT/"preflight.json"
    if previous.exists():
        old=read_json(previous)
        if old.get("status")=="PASS" and old.get("identity")==identity:
            return dict(status="PASS_REUSED",report=str(previous),note="No repeated heavy preflight")
    no_active()
    folder=OUT/("preflight_"+utc())
    folder.mkdir(parents=True)
    report=dict(status="FAILED",identity=identity,time=utc(),folder=str(folder))
    try:
        cpu_env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        commands=[
            ("mechanism_and_model", "check_pds_v1.py", ["--source",str(SOURCE),"--device","cpu"], "cpu", "checks.json"),
            ("lifecycle", "check_pds_v1_lifecycle.py", ["--source",str(SOURCE),"--dataset",str(MAIN/"datasets/crack_det")], "lifecycle", "lifecycle.json"),
            ("coverage", "pds_v1_probe.py", ["coverage","--source",str(SOURCE),"--dataset",str(MAIN/"datasets/crack_det"),"--device","cpu"], "coverage", "coverage.json")]
        for key,script,extra,subdir,filename in commands:
            with (folder/(subdir+"_console.log")).open("w",encoding="utf-8") as log:
                subprocess.run([sys.executable,"-u",str(ROOT/"tools"/script),*extra,"--output",str(folder/subdir)],
                               cwd=ROOT,env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            report[key]=read_json(folder/subdir/filename)
        cmd=[sys.executable,"-u",str(ROOT/"tools/pds_v1_probe.py"),"capacity","--output",str(folder)]
        with (folder/"capacity_console.log").open("w",encoding="utf-8") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)  # directly created PDS diagnostic group only
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                raise RuntimeError("PDS isolated capacity exceeded 900 s")
        report["capacity"]=read_json(folder/"capacity.json") if (folder/"capacity.json").exists() else {}
        require(code==0 and report["capacity"].get("status")=="PASS","Capacity failed; see original traceback/log")
        require(current_identity()==identity,"Inputs changed during preflight")
        report["status"]="PASS"
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        if (folder/"capacity.json").exists():
            report["capacity"]=read_json(folder/"capacity.json")
        raise
    finally:
        archive_file(previous)
        write_json(previous,report)
    return report


def completed_rows(run=None):
    run=RUN if run is None else run
    path=run/"results.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def archive_failed_start():
    no_active()
    require(RUN.exists(),"No failed formal run directory to archive")
    require(not list(RUN.rglob("*.pt")) and not completed_rows(),"Checkpoints/completed epochs exist; use true resume")
    destination=RUN.with_name(RUN.name+".failed_setup_"+utc())
    require(RUN.resolve().parent==(MAIN/"runs/c_series").resolve(),"Unexpected output root")
    RUN.rename(destination)
    result=dict(status="ARCHIVED_FAILED_SETUP",from_path=str(RUN),to_path=str(destination),time=utc())
    write_json(OUT/("archive_"+utc()+".json"),result)
    return result


def worker_script(command,console,dispatch_id):
    exports=dict(PYTHONPATH=str(ROOT/"ultralytics-main")+":"+str(ROOT/"tools"),
                 PYTHONUNBUFFERED="1",YOLO_AUTOINSTALL="false",PDS_MAIN=str(MAIN),PDS_DISPATCH=dispatch_id)
    for key in ("PATH","LD_LIBRARY_PATH","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","CUBLAS_WORKSPACE_CONFIG"):
        if key in os.environ:
            exports[key]=os.environ[key]
    shell="#!/usr/bin/env bash\nset -uo pipefail\ncd "+shlex.quote(str(ROOT))+"\n"
    shell+="".join("export "+k+"="+shlex.quote(v)+"\n" for k,v in exports.items())
    shell+="set +e\n"+" ".join(map(shlex.quote,command))+" 2>&1 | tee "+shlex.quote(str(console))+"\n"
    shell+='codes=("${PIPESTATUS[@]}")\nrc="${codes[0]}"\ntee_rc="${codes[1]}"\n'
    shell+="printf '{\"dispatch\":\"%s\",\"python_exit\":%s,\"tee_exit\":%s}\\n' "+shlex.quote(dispatch_id)+' "$rc" "$tee_rc" > '+shlex.quote(str(OUT/f"shell_exit_{dispatch_id}.json"))+"\n"
    shell+='if [ "$rc" -ne 0 ]; then exit "$rc"; fi\nexit "$tee_rc"\n'
    return shell


def dispatch(resume=False):
    require(os.name=="posix" and Path(sys.executable).resolve()==Path(SERVER_PYTHON).resolve(),"Use fixed server Python/Linux")
    no_active()
    require_gate()
    if resume:
        from pds_v1_trainer import validate_resume
        from ultralytics.utils.patches import torch_load
        last=RUN/"weights/last.pt"
        require(last.is_file(),"No resumable last.pt; resume cannot repair setup failure")
        validate_resume(torch_load(last,map_location="cpu"))
    else:
        require(not RUN.exists(),"Output exists: use resume or archive-failed-start for a setup-only failure")
    require(shutil.which("tmux"),"tmux unavailable")
    dispatch_id=uuid.uuid4().hex
    console=OUT/f"console_{utc()}.log"
    script=OUT/f"worker_{dispatch_id}.sh"
    command=[SERVER_PYTHON,"-u",str(ROOT/"tools/pds_v1.py"),"_worker","--dispatch",dispatch_id]
    if resume:
        command.append("--resume")
    script.write_text(worker_script(command,console,dispatch_id),encoding="utf-8")
    for name in ("dispatch.json","state.json","exit.json"):
        archive_file(OUT/name)
    row=dict(dispatch=dispatch_id,phase="DISPATCHED",time=utc(),resume=resume,log=str(console),
             command=command,sha=git("rev-parse","HEAD"),session=SESSION,preflight_sha256=sha256(OUT/"preflight.json"))
    write_json(OUT/"dispatch.json",row)
    write_json(OUT/"state.json",row)
    try:
        subprocess.run(["tmux","new-session","-d","-s",SESSION,"bash "+shlex.quote(str(script))],check=True)
    except BaseException as error:
        write_json(OUT/"state.json",dict(row,phase="FAILED",error=repr(error)))
        raise
    return row


def worker(dispatch_id,resume):
    from pds_v1_trainer import PDSTrainer
    require(os.environ.get("PDS_DISPATCH")==dispatch_id,"Worker dispatch environment differs")
    row=read_json(OUT/"dispatch.json")
    require(row["dispatch"]==dispatch_id and row["sha"]==git("rev-parse","HEAD"),"Stale worker dispatch")
    exit_code=1
    result=dict(dispatch=dispatch_id,pid=os.getpid(),time=utc(),phase="SETTING_UP")
    trainer=None
    try:
        require_gate()
        args=YAML.load(OUT/"train_args.yaml")
        if resume:
            args.update(model=str(RUN/"weights/last.pt"),resume=str(RUN/"weights/last.pt"),exist_ok=True)
        else:
            require(not RUN.exists(),"Run appeared after dispatch; refuse overwrite")
        write_json(OUT/"state.json",result)
        trainer=PDSTrainer(overrides=args)
        require(trainer.save_dir.resolve()==RUN.resolve(),"Unexpected native save_dir")
        trainer.train()
        result.update(phase="COMPLETED",completed_epochs=trainer.epoch+1,
                      early_stop=trainer.epoch+1<trainer.epochs,configured_epochs=trainer.epochs,
                      effective_steps=trainer.pds_effective_steps)
        exit_code=0
    except BaseException as error:
        result.update(phase="FAILED",error=repr(error),traceback=traceback.format_exc())
        if trainer is not None:
            result["last_epoch"]=getattr(trainer,"epoch",None)
        raise
    finally:
        result.update(exit_code=exit_code,finished=utc(),log=row["log"])
        write_json(OUT/"state.json",result)
        write_json(OUT/"exit.json",result)


def status():
    processes=live_processes()
    row=read_json(OUT/"dispatch.json") if (OUT/"dispatch.json").exists() else {}
    state=read_json(OUT/"state.json") if (OUT/"state.json").exists() else dict(phase="NOT_STARTED")
    if row and state.get("dispatch")!=row["dispatch"]:
        state=dict(row)
    result=dict(state=state,tmux=session_alive(),processes=processes,completed_epochs=len(completed_rows()))
    if row:
        shell=OUT/f"shell_exit_{row['dispatch']}.json"
        if shell.exists():
            result["shell_exit"]=read_json(shell)
            if result["shell_exit"]["python_exit"]!=0 or result["shell_exit"]["tee_exit"]!=0:
                result["state"]["phase"]="FAILED"
        elif not processes and not result["tmux"] and state.get("phase") not in ("COMPLETED","FAILED"):
            result["state"]=dict(state,phase="FAILED",reason="No verified live worker/session or completion record")
        path=Path(row["log"])
        if path.exists():
            with path.open("rb") as f:
                f.seek(max(path.stat().st_size-6000,0))
                result["log_tail"]=f.read().decode("utf-8",errors="replace")
    return result


def evaluate(split):
    from pds_v1_deploy import deploy
    from c19_lif_v1_results import evaluate as parent_evaluate
    no_active()
    verify_delivery()
    current_identity()
    output=OUT/"evaluation"
    deploy_path=output/"deploy.pt"
    lock=output/"selection.json"
    if split=="test":
        require(lock.is_file(),"Test needs completed independent val and locked weight")
    if lock.exists():
        selected=read_json(lock)
        require(sha256(deploy_path)==selected["deploy_sha256"],"Locked deploy changed")
        require(sha256(RUN/"weights/best.pt")==selected["source_sha256"],"Best checkpoint changed after val")
        require(selected["identity"]==current_identity(),"Evaluation identity changed")
    elif split=="val":
        output.mkdir(parents=True,exist_ok=True)
        if not deploy_path.exists():
            selected=deploy(RUN/"weights/best.pt",deploy_path,output/"deployment.json")
        else:
            selected=read_json(output/"deployment.json")
            require(sha256(deploy_path)==selected["deploy_sha256"],"Deploy report mismatch")
            require(sha256(RUN/"weights/best.pt")==selected["source_sha256"],"Best changed since deploy")
        selected["identity"]=current_identity()
    metrics=output/split/"metrics.json"
    if metrics.exists():
        previous=read_json(metrics)
        require(previous["status"]=="completed","Prior evaluation failed; preserve evidence and inspect before rerun")
        require(previous["checkpoint_sha256"]==sha256(deploy_path),"Previous evaluation weight mismatch")
        if split=="val" and not lock.exists():
            write_json(lock,selected)
        return dict(status="EXISTING_RESULT",metrics=previous)
    marker=output/(split+"_started.json")
    require(not marker.exists(),f"{split} already attempted; no automatic repeat test")
    write_json(marker,dict(time=utc(),checkpoint_sha256=sha256(deploy_path)))
    result=parent_evaluate(deploy_path,Path(recipe()["data"]),split,output/split,
                           val_report=output/"val/metrics.json" if split=="test" else None,
                           runtime_info=environment())
    if split=="val":
        write_json(lock,selected)
    return result


def pack():
    verify_delivery()
    folder=OUT/("light_"+utc())
    folder.mkdir(parents=True)
    (folder/"source.patch").write_bytes(subprocess.check_output(["git","diff","--binary",BASE,"HEAD"],cwd=ROOT))
    write_json(folder/"identity.json",dict(base=BASE,sha=git("rev-parse","HEAD"),branch=BRANCH,status=status()))
    archive=OUT/("PDS_v1_LIGHT_"+utc()+".tar.gz")
    manifest=[]
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.is_relative_to(folder) or p.suffix not in (".json",".yaml",".md",".csv"):
            continue
        if p.stat().st_size>2_000_000 or any(x.startswith("light_") for x in p.relative_to(OUT).parts):
            continue
        manifest.append(p)
    with tarfile.open(archive,"w:gz") as tar:
        tar.add(folder/"source.patch",arcname="source.patch")
        tar.add(folder/"identity.json",arcname="identity.json")
        for p in manifest:
            tar.add(p,arcname="reports/"+p.relative_to(OUT).as_posix())
        for name in ("args.yaml", "results.csv", "pds_setup.json", "pds_optimizer_latest.json"):
            p=RUN/name
            if p.is_file() and p.stat().st_size <= 2_000_000:
                tar.add(p,arcname="training/"+name)
        for p in sorted((ROOT/"docs/pds_v1").glob("*")):
            if p.is_file() and p.stat().st_size<2_000_000:
                tar.add(p,arcname="docs/"+p.name)
    return dict(status="LIGHT",path=str(archive),sha256=sha256(archive),bytes=archive.stat().st_size)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    for name in ("prepare","preflight","start","status","resume","val","test","pack","archive-failed-start"):
        sub.add_parser(name)
    sub.add_parser("sync").add_argument("sha")
    q=sub.add_parser("_worker")
    q.add_argument("--dispatch",required=True)
    q.add_argument("--resume",action="store_true")
    return p


def main():
    a=parser().parse_args()
    torch.set_num_threads(4)
    if a.action=="_worker":
        return worker(a.dispatch,a.resume)
    if a.action=="status":
        result=status()
    else:
        with operation_lock():
            actions=dict(prepare=prepare,preflight=preflight,start=lambda:dispatch(False),
                         resume=lambda:dispatch(True),val=lambda:evaluate("val"),test=lambda:evaluate("test"),
                         pack=pack,**{"archive-failed-start":archive_failed_start,"sync":lambda:record_delivery(a.sha)})
            result=actions[a.action]()
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str,allow_nan=False))


if __name__=="__main__":
    main()
