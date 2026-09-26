"""Safe single-experiment dispatch, current-process status and explicit preparation."""
from __future__ import annotations
from copy import deepcopy
import csv
import shlex
import shutil
import signal
import subprocess
import traceback
import uuid
from peq_v1_common import *
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def process_info(pid):
    """Linux PID identity; never signal or inspect unrelated processes as this experiment."""
    if not pid or not sys.platform.startswith("linux"):
        return None
    folder=Path("/proc")/str(pid)
    try:
        command=(folder/"cmdline").read_bytes().split(b"\0")
        command=[s.decode(errors="replace") for s in command if s]
        text=(folder/"stat").read_text()
        fields=text[text.rfind(")")+2:].split()
        return dict(pid=int(pid),command=command,start_token=fields[19],
                    state=fields[0],cwd=str((folder/"cwd").resolve()))
    except (OSError,ValueError,IndexError):
        return None


def owns_process(info, dispatch):
    return bool(info and info["state"]!="Z" and str(ROOT/"tools/peq_v1.py") in info["command"] and
                "worker" in info["command"] and dispatch in info["command"] and info["cwd"]==str(ROOT))


def resolve_status(state, exit_record, process, tmux_alive=False):
    """Pure state resolution for NOT_STARTED/RUNNING/FAILED/completed fixtures."""
    result=dict(state or {})
    result.setdefault("phase","NOT_STARTED")
    dispatch=result.get("dispatch")
    if exit_record and exit_record.get("dispatch")==dispatch:
        result.update(exit_record)
        result["phase"]="COMPLETED" if exit_record.get("python_exit")==0 and exit_record.get("tee_exit")==0 else "FAILED"
        return result
    active=owns_process(process,dispatch) and (not result.get("start_token") or result["start_token"]==process["start_token"])
    result["active"]=bool(active)
    if active:
        result["process"]=process
    elif result["phase"] in ("SETTING_UP","RUNNING") and not tmux_alive:
        result["phase"]="FAILED";result["reason"]="Recorded worker exited without a current-dispatch exit record"
    elif result["phase"]=="DISPATCHED" and not tmux_alive:
        result["phase"]="FAILED";result["reason"]="Dispatch has no active tmux session"
    return result


def tmux_exists():
    if not shutil.which("tmux"):
        return False
    return subprocess.run(["tmux","has-session","-t","="+SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def status():
    state=read_json(OUT/"state.json",{})
    current=resolve_status(state,read_json(OUT/"exit.json"),process_info(state.get("pid")),tmux_exists())
    log=Path(state["console_log"]) if state.get("console_log") else None
    current["log_tail"]=log.read_text(encoding="utf-8",errors="replace").splitlines()[-15:] if log and log.is_file() else []
    children=[]
    if current.get("active"):
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    data=(entry/"stat").read_text()
                    fields=data[data.rfind(")")+2:].split()
                    if int(fields[1])==current["pid"]:
                        child=process_info(int(entry.name))
                        if child:
                            children.append(child)
                except (OSError,ValueError,IndexError):
                    continue
    current["children"]=children
    current["run"]=str(RUN)
    return current


def update_state(dispatch, **changes):
    state=read_json(OUT/"state.json",{})
    require(state.get("dispatch")==dispatch,"Stale dispatch cannot overwrite current state")
    state.update(changes)
    state["updated"]=utc()
    write_json(OUT/"state.json",state)
    append_json(OUT/"events.jsonl",dict(dispatch=dispatch,at=utc(),**changes))
    return state


def verify_delivery():
    record=read_json(OUT/"delivery.json")
    require(record and record["commit"]==git("rev-parse","HEAD") and record["branch"]==BRANCH,"Run sync with the exact delivered FULL_SHA first")
    require(record["worktree"]==str(ROOT) and record["main"]==str(MAIN),"Delivery paths differ")
    require(git("branch","--show-current")==BRANCH,"Wrong experiment branch")
    require(not git("status","--porcelain","--untracked-files=no"),"Tracked source has local changes")
    require(record["source_sha256"]==source_identity()["sha256"],"Source differs from sync identity")
    return record


def record_delivery(sha):
    require(isinstance(sha,str) and len(sha)==40 and all(c in "0123456789abcdef" for c in sha),"Need full lowercase SHA")
    require(git("rev-parse","HEAD")==sha and git("branch","--show-current")==BRANCH,"Wrong checkout")
    require(git("remote","get-url","origin")==REMOTE,"Wrong remote")
    require((ROOT/".git").is_file(),"Expected linked worktree")
    require(not git("status","--porcelain","--untracked-files=no"),"Dirty worktree")
    result=dict(commit=sha,base=BASE,branch=BRANCH,remote=REMOTE,worktree=str(ROOT),main=str(MAIN),
                source_sha256=source_identity()["sha256"],synchronized=utc())
    write_json(OUT/"delivery.json",result)
    from peq_v1_delivery import write_delivery
    write_delivery(sha)
    return result


def recipe():
    original=YAML.load(ROOT/"docs/peq_v1/mother_args.yaml")
    require(len(original)==109,"Expected 109 authoritative mother arguments")
    actual={**original,"model":str(INIT),"name":RUN_NAME,"save_dir":str(RUN)}
    rows=[dict(field=k,mother=original[k],peq=actual[k],changed=original[k]!=actual[k],
               reason="experiment identity" if k in ("model","name","save_dir") else None) for k in original]
    require({r["field"] for r in rows if r["changed"]}=={"model","name","save_dir"},"Recipe changed beyond identity")
    return actual,rows


def prepared_identity(data):
    return dict(source_sha256=source_identity()["sha256"],source_weight=identity(SOURCE),initialization=identity(INIT),
                research=identity(ROOT/"docs/peq_v1/research.yaml"),args=identity(OUT/"train_args.yaml"),
                data=dataset_identity(data),environment=environment(),module_hashes=module_contract())


def prepare():
    verify_delivery()
    require(sys.platform.startswith("linux") and str(MAIN)=="/root/autodl-tmp/projects/Crack_RTDETR","Formal prepare requires specified Linux server layout")
    from init_peq_v1 import initialize
    from train_c19_lif_v1 import ensure_amp_resources
    args,rows=recipe()
    data=Path(args["data"])
    pending=[str(p) for p in (SOURCE,data) if not p.is_file()]
    if data.is_file():
        data_cfg=YAML.load(data)
        data_root=Path(data_cfg["path"])
        pending.extend(str(data_root/k/split) for k in ("images","labels") for split in ("train","val","test")
                       if not (data_root/k/split).is_dir())
    if pending:
        result=dict(status="PENDING",missing=pending)
        write_json(OUT/"prepared.json",result)
        return result
    require(sha256(SOURCE)==SOURCE_SHA,"Wrong public source")
    YAML.save(OUT/"train_args.yaml",args)
    write_json(OUT/"args_diff_109.json",rows)
    initialize()
    ensure_amp_resources(MAIN,OUT/"amp_resources.json")
    import cv2
    from ultralytics.utils import ASSETS
    require(cv2.imread(str(ASSETS/"bus.jpg")) is not None,"AMP bus asset is corrupt")
    amp_weight=torch_load(ROOT/"yolo26n.pt",map_location="cpu")
    require(amp_weight.get("model") is not None or amp_weight.get("ema") is not None,"AMP check weight invalid")
    current=prepared_identity(data)
    audited=read_json(ROOT/"docs/peq_v1/dataset_identity.json")
    require(audited is not None,"Committed data content audit missing")
    for split in ("train","val","test"):
        for key in ("images","ground_truth","paths_sha256","labels_sha256","images_sha256"):
            require(current["data"]["splits"][split][key]==audited["splits"][split][key],
                    f"Server data differs from audited mother content: {split}/{key}")
    current_cfg=YAML.load(data)
    require({k:v for k,v in current_cfg.items() if k!="path"}==
            {k:v for k,v in audited["config_semantics"].items() if k!="path"},"Data path migration changed semantics")
    write_json(OUT/"data_path_migration.json",dict(status="PASS",local_config=audited["config"],server_config=current["data"]["config"],
               local_root=audited["root"],server_root=current["data"]["root"],all_split_contents_equal=True))
    result=dict(status="PASS",identity=current,identity_key=digest_json(current),created=utc(),
                args_diff_109=identity(OUT/"args_diff_109.json"),amp_resources=read_json(OUT/"amp_resources.json"))
    write_json(OUT/"prepared.json",result)
    write_json(OUT/"source_identity.json",source_identity())
    (OUT/"source_from_mother.patch").write_bytes(subprocess.check_output(["git","diff","--binary",BASE,"HEAD"],cwd=ROOT))
    (OUT/"pip_freeze.txt").write_bytes(subprocess.check_output([sys.executable,"-m","pip","freeze"]))
    return result


def require_prepared():
    verify_delivery()
    record=read_json(OUT/"prepared.json")
    require(record and record["status"]=="PASS","prepare has not passed")
    args=YAML.load(OUT/"train_args.yaml")
    current=prepared_identity(args["data"])
    require(record["identity"]==current,"Preparation identity is stale; rerun prepare")
    return record,args


def valid_last():
    path=RUN/"weights/last.pt"
    require(path.is_file(),"No valid last.pt to resume")
    ckpt=torch_load(path,map_location="cpu")
    require(ckpt.get("epoch",-1)>=0 and ckpt.get("optimizer") is not None and ckpt.get("scaler") is not None and
            ckpt.get("ema") is not None,"last.pt lacks resumable optimizer/scaler/EMA/epoch state")
    require(ckpt["epoch"]<199 and ckpt["ema"].model[-1].peq.config["enabled"],"Last checkpoint is completed or PEQ disabled")
    return path


def dispatch(resume=False):
    prepared,args=require_prepared()
    preflight=read_json(OUT/"preflight.json")
    require(preflight and preflight["status"]=="PASS" and preflight["identity_key"]==prepared["identity_key"],
            "Current B16/640 preflight has not passed")
    require(shutil.which("tmux"),"tmux is unavailable")
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/"dispatch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not status().get("active") and not tmux_exists(),"This PEQ experiment is already active")
        if resume:
            valid_last()
        else:
            require(not RUN.exists(),"Existing run preserved; use resume or explicit archive-failed when eligible")
        token=uuid.uuid4().hex
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log=OUT/f"console_{stamp}_{token}.log"
        worker_script=OUT/f"worker_{token}.sh"
        command=[PYTHON,str(ROOT/"tools/peq_v1.py"),"worker","--dispatch",token]
        if resume:
            command.append("--resume-worker")
        shell="\n".join((
            "#!/usr/bin/env bash","set -uo pipefail",f"cd {shlex.quote(str(ROOT))}",
            f"export PYTHONPATH={shlex.quote(str(ROOT/'ultralytics-main')+':'+str(ROOT/'tools'))}",
            "export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false",
            shlex.join(command)+" 2>&1 | tee "+shlex.quote(str(log)),
            'codes=("${PIPESTATUS[@]}")',
            shlex.join([PYTHON,str(ROOT/"tools/peq_v1.py"),"finish","--dispatch",token])+
            ' --python-exit "${codes[0]}" --tee-exit "${codes[1]}"',
            'exit "${codes[0]}"',""))
        worker_script.write_text(shell,encoding="utf-8")
        state=dict(phase="DISPATCHED",dispatch=token,created=utc(),resume=resume,session=SESSION,
                   console_log=str(log),run=str(RUN),delivery=git("rev-parse","HEAD"),
                   preflight_sha256=sha256(OUT/"preflight.json"))
        write_json(OUT/"state.json",state)
        try:
            subprocess.run(["tmux","new-session","-d","-s",SESSION,"bash",str(worker_script)],check=True)
        except BaseException as error:
            update_state(token,phase="FAILED",error=repr(error))
            raise
        return state


def worker(token,resume=False):
    code=1
    try:
        prepared,args=require_prepared()
        state=read_json(OUT/"state.json")
        require(state["dispatch"]==token and sha256(OUT/"preflight.json")==state["preflight_sha256"],"Dispatch identity changed")
        info=process_info(os.getpid())
        update_state(token,phase="SETTING_UP",pid=os.getpid(),start_token=info["start_token"],command=info["command"])
        from ultralytics import RTDETR
        from ultralytics.models.rtdetr.peq_train import PEQTrainer
        from ultralytics.models.rtdetr.peq_stats import PEQStatistics
        if not resume:
            require(not RUN.exists(),"Run directory appeared after dispatch; preserved")
        model=RTDETR(str(valid_last() if resume else INIT))
        if resume:
            args=dict(args,model=str(RUN/"weights/last.pt"),resume=str(RUN/"weights/last.pt"))

        def setup(trainer):
            require(trainer.amp,"Native AMP check disabled AMP; formal settings cannot change")
            require(type(trainer.optimizer) is torch.optim.AdamW and trainer.world_size==1,"Formal single-GPU AdamW required")
            expected=dict(args)
            actual=vars(trainer.args)
            differences={k:[v,actual.get(k)] for k,v in expected.items() if v!=actual.get(k) or type(v) is not type(actual.get(k))}
            require(not differences,f"Effective recipe changed: {differences}")
            require(trainer.save_dir.resolve()==RUN.resolve(),"Native Trainer selected another run directory")
            YAML.save(OUT/"actual_train_args.yaml",actual)
            write_json(OUT/"training_setup.json",dict(dispatch=token,loading=trainer.peq_loading_audit,
                       parameters=sum(p.numel() for p in trainer.model.parameters()),amp=trainer.amp,
                       native_amp_check_messages=trainer.peq_amp_check_messages,
                       batch=trainer.batch_size,accumulate=trainer.accumulate,scaler=trainer.scaler.state_dict(),
                       native_loss_batch_multiplier="No additional B multiplier in pinned RTDETR model/Trainer",
                       optimizer="AdamW",start_epoch=trainer.start_epoch,environment=environment()))
            trainer.model.peq_stats=PEQStatistics()
            trainer.ema.ema.peq_stats=PEQStatistics()

        def batch_started(trainer):
            trainer._oom_retries=3

        def batch_finished(trainer):
            if trainer.model.peq_stats.total_batches==1:
                update_state(token,phase="RUNNING",epoch=trainer.epoch+1,real_batch_completed=True)

        def epoch_finished(trainer):
            row=dict(dispatch=token,epoch=trainer.epoch+1,at=utc(),
                     train=trainer.model.peq_stats.report(),val=trainer.ema.ema.peq_stats.report())
            append_json(OUT/"mechanism_epochs.jsonl",row)
            for domain in ("train","val"):
                for event in row[domain]["events"]:
                    append_json(OUT/"mechanism_events.jsonl",dict(dispatch=token,epoch=trainer.epoch+1,domain=domain,**event))
            trainer.model.peq_stats.reset();trainer.ema.ema.peq_stats.reset()
            update_state(token,phase="RUNNING",epoch=trainer.epoch+1)

        model.add_callback("on_train_start",setup)
        model.add_callback("on_train_batch_start",batch_started)
        model.add_callback("on_train_batch_end",batch_finished)
        model.add_callback("on_fit_epoch_end",epoch_finished)
        model.train(trainer=PEQTrainer,**args)
        epoch=model.trainer.epoch+1
        update_state(token,phase="COMPLETED",actual_epochs=epoch,
                     stop_reason="epoch_limit" if epoch==200 else "native_early_stop",python_finished=True)
        code=0
    except BaseException as error:
        update_state(token,phase="FAILED",exception=repr(error))
        traceback.print_exc()
        raise
    finally:
        write_json(OUT/f"python_exit_{token}.json",dict(dispatch=token,python_exit=code,at=utc()))


def finish(token, python_exit, tee_exit):
    result=dict(dispatch=token,python_exit=python_exit,tee_exit=tee_exit,finished=utc())
    state=read_json(OUT/"state.json",{})
    require(state.get("dispatch")==token,"Old exit cannot overwrite current dispatch")
    require(python_exit!=0 or read_json(OUT/f"python_exit_{token}.json",{}).get("python_exit")==0,
            "Python process exited without successful worker completion")
    write_json(OUT/"exit.json",result)
    return update_state(token,phase="COMPLETED" if python_exit==tee_exit==0 else "FAILED",**{k:v for k,v in result.items() if k!="dispatch"})


def archive_failed():
    require(not status().get("active") and not tmux_exists(),"This experiment is active")
    require(RUN.is_dir(),"No failed startup directory")
    files=[p for p in RUN.rglob("*") if p.is_file()]
    require(files and all(p.name=="args.yaml" or p.name=="results.csv" for p in files),
            "Safe restart only permits startup args.yaml and an empty CSV; other evidence preserved")
    csv_path=RUN/"results.csv"
    if csv_path.is_file():
        with csv_path.open(newline="",encoding="utf-8") as stream:
            require(not list(csv.DictReader(stream)),"CSV has real training rows; use valid last to resume")
    destination=RUN.with_name(RUN.name+"_startup_failed_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    require(RUN.resolve().parent==MAIN/"runs/c_series" and not destination.exists(),"Unsafe archive destination")
    RUN.rename(destination)
    write_json(OUT/"state.json",dict(phase="NOT_STARTED",archived_failed_start=str(destination),at=utc()))
    return dict(status="ARCHIVED",path=str(destination))
