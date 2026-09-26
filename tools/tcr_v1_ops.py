"""Experiment-specific prepare/gates/dispatch/status/resume and read-only packaging."""
from __future__ import annotations

import csv
import io
import shlex
import shutil
import signal
import tarfile
import time
import uuid

from tcr_v1_core import *
from c19_lif_v1_data import dataset_inventory


def verify_delivery():
    info=read_json(OUT/"delivery.json")
    require(info and info["commit"]==git("rev-parse","HEAD") and info["branch"]==BRANCH,"Run sync_tcr_v1.sh FULL_SHA first")
    require(Path(info["worktree"]).resolve()==ROOT and Path(info["main"]).resolve()==MAIN,"Delivery path changed")
    require(git("branch","--show-current")==BRANCH,"Wrong experiment branch")
    require(not git("status","--porcelain"),"Source modified or untracked files appeared after delivery")
    parent.source_contract()
    return info


def dataset_identity(config):
    config=Path(config)
    d=YAML.load(config)
    expected=YAML.load(ROOT/"docs/c19_lif_v1/c2_data.yaml")
    require(d==expected,"Data config differs from authoritative mother")
    dataset=Path(d["path"])
    inv=dataset_inventory(dataset)
    require({s:(v["images"],v["boxes"]) for s,v in inv.items()}==dict(train=(6048,45573),val=(1728,12840),test=(864,6663)),"Dataset counts differ from protocol")
    # File size + mtime inventory catches replaced images between cached checks;
    # labels are fully content-hashed by the mother inventory on every check.
    images=[]
    for split in ("train","val","test"):
        for p in sorted((dataset/"images"/split).rglob("*")):
            if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
                st=p.stat(); images.append([p.relative_to(dataset).as_posix(),st.st_size,st.st_mtime_ns])
    return dict(config_sha256=sha256(config),inventory=inv,image_stat_sha256=hashlib.sha256(json.dumps(images).encode()).hexdigest(),
                image_identity_rule="relative paths, byte size and mtime_ns; label files use full SHA256; source dataset path unchanged",
                dataset=str(dataset))


def fingerprint(payload):
    return hashlib.sha256(json.dumps(clean_json(payload),sort_keys=True).encode()).hexdigest()


def current_identity():
    info=runtime()
    return dict(source=source_identity()["sha256"],source_weight=file_identity(SOURCE),init=file_identity(INIT),
                data=dataset_identity(OUT/"data.yaml"),config=file_identity(OUT/"train_args.yaml"),
                research=file_identity(DOC/"research.yaml"),
                environment={k:info[k] for k in ("python","torch","cuda","gpu","gpu_bytes","ultralytics","version")})


def prepare():
    delivery=verify_delivery(); OUT.mkdir(parents=True,exist_ok=True)
    active=status()
    require(not active["processes"] and not active.get("session_alive"),"Active experiment protected from prepare changes")
    info=runtime(); require(SOURCE.is_file() and sha256(SOURCE)==SOURCE_SHA,"Missing or wrong public untrained source")
    args,diff=recipe()
    data=Path(args["data"]); require(data.is_file(),"Server data config missing")
    inventory=dataset_identity(data)
    previous=read_json(OUT/"initialization.json")
    if INIT.exists():
        require(previous and previous["output"]==file_identity(INIT) and previous["source"]==file_identity(SOURCE),"Existing init has no matching audit; preserved")
    else:
        write_json(OUT/"initialization.json",initialize(SOURCE,INIT))
    shutil.copyfile(data,OUT/"data.yaml")
    YAML.save(OUT/"train_args.yaml",args)
    write_json(OUT/"args_diff_109.json",diff)
    # Verify the authority actually used for this delivery, not current framework defaults.
    write_json(OUT/"recipe_authority.json",dict(archive=file_identity(DOC/"mother_args.yaml"),fields=109,
               source="User's extracted successful C19+LIF training/args.yaml, matched to contract appendix",
               base_archive_difference="model path contains -gatefix; remaining 108 fields equal base resolved_formal_config"))
    from train_c19_lif_v1 import ensure_amp_resources
    ensure_amp_resources(MAIN,OUT/"amp_resources.json")
    from PIL import Image
    from ultralytics.utils import ASSETS
    with Image.open(ASSETS/"bus.jpg") as image: image.verify()
    check=torch_load(ROOT/"yolo26n.pt",map_location="cpu")
    require(isinstance(check,dict) and (check.get("model") is not None or check.get("ema") is not None),"check_amp weight resource unreadable")
    identity=current_identity()
    result=dict(status="PREPARED",fingerprint=fingerprint(identity),**identity,prepared=utc(),runtime=info,delivery=delivery,
                data_limitation="Known related augmented images cross splits; same-protocol comparison, not independent-original generalization")
    write_json(OUT/"prepared.json",result)
    (OUT/"source_from_mother.patch").write_bytes(subprocess.check_output(["git","diff","--binary",BASE,"HEAD"],cwd=ROOT))
    write_json(OUT/"source_inventory.json",source_identity())
    print("Prepared:",OUT,"\nFingerprint:",result["fingerprint"])
    return result


def verify_prepared():
    verify_delivery()
    saved=read_json(OUT/"prepared.json")
    require(saved is not None,"Run prepare first")
    current=current_identity()
    require(fingerprint(current)==saved["fingerprint"],"Source/config/init/data/environment changed; run prepare and preflight")
    return saved


def ensure_gate(prepared):
    gate=read_json(OUT/"preflight_gate.json")
    require(gate and gate["fingerprint"]==prepared["fingerprint"],"No current B16/640 preflight")
    report=read_json(gate["report"])
    require(preflight_accepted(report),"Preflight capacity/strict FP32 fusion did not pass")
    require(file_identity(gate["report"])["sha256"]==gate["sha256"],"Preflight report altered")
    return gate


def preflight_accepted(report):
    """Only completed capacity + strict FP32 evidence can admit training.

PASS_STRICT_FP32_ONLY explicitly does not certify fusion at runtime precision.
No fallback can admit an early-node or mother-protocol strict FP32 failure.
"""
    if not report or report.get("status") not in {"PASS", "PASS_STRICT_FP32_ONLY"}:
        return False
    fusion=report.get("fusion",{})
    return (report.get("capacity_status")=="PASS" and report.get("effective_updates",0)>=2
            and report.get("post_O_P_gradient") is True and report.get("original_gradients") is True
            and report.get("fusion_trainer_unchanged") is True and fusion.get("strict_accepted") is True
            and (report["status"]=="PASS") == (fusion.get("runtime_accepted") is True))


def preflight():
    prepared=verify_prepared()
    old=read_json(OUT/"preflight_gate.json")
    if old and old.get("fingerprint")==prepared["fingerprint"]:
        ensure_gate(prepared); print("Reusing current preflight:",old["report"]); return old
    folder=OUT/"preflights"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8])
    log=folder.with_suffix(".log"); log.parent.mkdir(parents=True,exist_ok=True)
    require(os.name=="posix","B16/640 server preflight runs on the server; local checks use check_tcr_v1.py")
    command=[sys.executable,"-u",str(ROOT/"tools/tcr_v1.py"),"_preflight","--folder",str(folder)]
    with log.open("x",encoding="utf-8") as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            report=read_json(folder/"preflight.json",{})
            report.update(status="FAILED",error="900-second wall-clock bound reached",fingerprint=prepared["fingerprint"])
            write_json(folder/"preflight.json",report)
            raise RuntimeError(f"Preflight timed out; evidence preserved: {folder}")
    report=read_json(folder/"preflight.json",{})
    require(code==0 and preflight_accepted(report),f"Preflight failed; inspect {log} and {folder}")
    gate=dict(fingerprint=prepared["fingerprint"],report=str(folder/"preflight.json"),sha256=sha256(folder/"preflight.json"),log=str(log),
              status=report["status"],fusion=report["fusion"])
    write_json(OUT/"preflight_gate.json",gate); print("B16/640 preflight",report["status"],":",folder)
    if report["status"]=="PASS_STRICT_FP32_ONLY":
        print("Runtime-precision fusion FAILED at the original tolerances; see retained evidence. This gate certifies strict FP32 fusion only.")
    return gate


def process_rows():
    rows=[]
    if os.name!="posix": return rows
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            argv=(p/"cmdline").read_bytes().decode().strip("\0").split("\0")
            stat=(p/"stat").read_text(); tail=stat[stat.rfind(")")+2:].split()
            rows.append(dict(pid=int(p.name),ppid=int(tail[1]),start_token=tail[19],argv=argv))
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    return rows


def owned_processes(rows=None):
    rows=process_rows() if rows is None else rows
    scripts={str(ROOT/"tools/tcr_v1.py"),str(ROOT/"tools/tcr_v1.sh")}
    own=[r for r in rows if any(a in scripts for a in r["argv"]) and ("_worker" in r["argv"] or "_dispatch" in r["argv"])]
    ids={r["pid"] for r in own}
    while True:
        children={r["pid"] for r in rows if r["ppid"] in ids}
        if children<=ids: break
        ids|=children
    return [r for r in rows if r["pid"] in ids]


def update_state(dispatch, **changes):
    state=read_json(OUT/"state.json",{})
    require(state.get("dispatch")==dispatch,"Stale worker cannot write current dispatch state")
    state.update(changes); state["dispatch"]=dispatch; state["updated"]=utc()
    write_json(OUT/"state.json",state)
    write_json(OUT/"dispatches"/dispatch/"state.json",state)


def status_from(state, exit_info, processes, session_alive=False):
    state=dict(state or {})
    if not state: return dict(phase="NOT_STARTED",processes=processes)
    current_exit=exit_info if exit_info and exit_info.get("dispatch")==state.get("dispatch") else None
    active=bool(processes or session_alive)
    if current_exit:
        state.update(python_exit_code=current_exit["python_exit_code"],tee_exit_code=current_exit["tee_exit_code"])
        if current_exit["python_exit_code"]!=0 or current_exit["tee_exit_code"]!=0: state["phase"]="FAILED"
        elif state.get("phase")!="COMPLETED": state["phase"]="FAILED"; state["reason"]="Process exited without completion evidence"
    elif not active and state.get("phase") in {"DISPATCHED","SETTING_UP","RUNNING"}:
        state["phase"]="FAILED"; state["reason"]="No matching live process/session and no matching exit record"
    state.update(processes=processes,session_alive=session_alive)
    return state


def status():
    state=read_json(OUT/"state.json")
    exists=False
    if shutil.which("tmux"):
        exists=subprocess.run(["tmux","has-session","-t",SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    exit_info=read_json(OUT/"dispatches"/state["dispatch"]/"exit.json") if state else None
    result=status_from(state,exit_info,owned_processes(),exists)
    if state and state.get("console") and Path(state["console"]).is_file():
        with Path(state["console"]).open("rb") as f:
            f.seek(max(0,Path(state["console"]).stat().st_size-6000)); result["log_tail"]=f.read().decode(errors="replace").splitlines()[-15:]
    result["run"]=str(RUN); result["output"]=str(OUT)
    return result


def valid_resume():
    last=RUN/"weights/last.pt"
    require(last.is_file(),"Resume requires a real last.pt")
    ckpt=torch_load(last,map_location="cpu")
    require(0<=ckpt.get("epoch",-1)<199 and ckpt.get("optimizer") and ckpt.get("scaler") is not None and ckpt.get("ema") is not None,"last has no resumable optimizer/scaler/EMA/epoch")
    verify_model(ckpt["ema"].float())
    actual=ckpt["train_args"]; planned=YAML.load(OUT/"train_args.yaml")
    require(all(actual.get(k)==v for k,v in planned.items() if k not in {"model","resume"}),"Resume recipe differs from original run")
    return file_identity(last)


def launch(resume=False):
    require(os.name=="posix" and shutil.which("tmux"),"Server Linux and tmux required")
    prepared=verify_prepared(); ensure_gate(prepared)
    current=status(); require(not current["processes"] and not current.get("session_alive"),"This experiment is already active")
    formal=read_json(OUT/"formal_identity.json")
    if resume:
        require(formal and formal["fingerprint"]==prepared["fingerprint"],"Resume identity differs from original formal run")
        checkpoint=valid_resume()
    else:
        require(not RUN.exists(),"Existing run preserved. Use resume for valid last or archive-failed-start explicitly")
        checkpoint=file_identity(INIT)
    reservation=OUT/"launch.lock"
    with reservation.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
    try:
        require(not owned_processes(),"Another dispatch appeared")
        dispatch=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        folder=OUT/"dispatches"/dispatch; folder.mkdir(parents=True,exist_ok=False)
        log=OUT/f"console_{dispatch}.log"
        state=dict(dispatch=dispatch,phase="DISPATCHED",resume=resume,console=str(log),session=SESSION,
                   checkpoint=checkpoint,created=utc(),fingerprint=prepared["fingerprint"])
        write_json(OUT/"state.json",state); write_json(folder/"state.json",state)
        if not resume: write_json(OUT/"formal_identity.json",dict(fingerprint=prepared["fingerprint"],dispatch=dispatch,created=utc()))
        command=["bash",str(ROOT/"tools/tcr_v1.sh"),"_dispatch",dispatch,"resume" if resume else "start"]
        result=subprocess.run(["tmux","new-session","-d","-s",SESSION,shlex.join(command)],capture_output=True,text=True)
        if result.returncode:
            update_state(dispatch,phase="FAILED",error=result.stderr)
            raise RuntimeError(result.stderr)
        print("Dispatched:",dispatch,"\nConsole:",log,"\ntmux attach -t",SESSION)
        return state
    finally:
        reservation.unlink()


def record_exit(dispatch, python_code, tee_code):
    folder=OUT/"dispatches"/dispatch
    write_json(folder/"exit.json",dict(dispatch=dispatch,python_exit_code=python_code,tee_exit_code=tee_code,finished=utc()))
    state=read_json(OUT/"state.json",{})
    if state.get("dispatch")==dispatch and (python_code or tee_code):
        update_state(dispatch,phase="FAILED",python_exit_code=python_code,tee_exit_code=tee_code)


def archive_failed_start():
    result=status()
    require(not result["processes"] and not result.get("session_alive"),"Active experiment protected")
    require(RUN.is_dir() and (RUN/"args.yaml").is_file(),"No failed setup directory")
    require(not any(RUN.rglob("*.pt")),"Checkpoints exist; preserve and use valid resume")
    csv_path=RUN/"results.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as f: require(not list(csv.DictReader(f)),"Training CSV has data rows")
    # Only a setup failure with args and empty weights dir is eligible.
    allowed={"args.yaml","results.csv"}
    require(all(p.name in allowed for p in RUN.rglob("*") if p.is_file()),"Run contains additional evidence; manual inspection required")
    destination=RUN.with_name(RUN.name+"_failed_setup_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    require(not destination.exists(),"Archive path exists")
    RUN.rename(destination)
    append_json(OUT/"archived_starts.jsonl",dict(original=str(RUN),archived=str(destination),time=utc(),state=result))
    print("Preserved failed setup at:",destination)
    return dict(archived=str(destination),next="Run start explicitly")


def package(include_predictions=False):
    require(not status()["processes"] and not status().get("session_alive"),"Package stable completed/stopped evidence")
    destination=MAIN/"downloads/tcr_v1"/("TCR_v1_LIGHT_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]+".tar.gz")
    files={}; omitted=[]; prediction_files={}
    for folder,prefix in ((OUT,"evidence"),(RUN,"training"),(DOC,"docs")):
        if not folder.exists(): continue
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.is_symlink(): continue
            rel=p.relative_to(folder).as_posix(); name=prefix+"/"+rel
            if p.suffix in {".pt",".torchscript"} or "fixture" in rel or p.name.endswith(".log") or p.name=="predictions_gt.jsonl.gz":
                omitted.append(file_identity(p))
                if p.name=="predictions_gt.jsonl.gz" and ("evaluation_val/" in rel or "evaluation_train/" in rel): prediction_files[name]=p
            else: files[name]=p
    for p in (SOURCE,INIT,RUN/"weights/best.pt",RUN/"weights/last.pt"):
        if p.is_file() and not any(r.get("path")==str(p.resolve()) for r in omitted): omitted.append(file_identity(p))
    for name in git("ls-files").splitlines():
        if name.startswith(("tools/","ultralytics-main/ultralytics/")) or name==".gitattributes": files["source/"+name]=ROOT/name
    logs={}
    for p in sorted(OUT.rglob("*.log")):
        with p.open("rb") as f:
            f.seek(max(0,p.stat().st_size-16000)); logs[str(p)]=f.read().decode(errors="replace")
    evidence=dict(status=status(),source=source_identity(),commit=git("rev-parse","HEAD"),omitted_files=omitted,
                  weight_inventory={k:file_identity(p) for k,p in dict(public_source=SOURCE,controlled_init=INIT,best=RUN/"weights/best.pt",last=RUN/"weights/last.pt").items()},
                  stages={k:read_json(OUT/v,dict(status="NOT_RUN")) for k,v in dict(prepare="prepared.json",preflight="preflight_gate.json",probe="probe/probe.json",val="evaluation_val/metrics.json",test="evaluation_test/metrics.json").items()},
                  limitation="No dataset, large weights, full raw logs or per-image predictions in LIGHT. No evaluation triggered.")
    extra={"EVIDENCE.json":(json.dumps(clean_json(evidence),ensure_ascii=False,indent=2)+"\n").encode(),
           "LOG_TAILS.json":(json.dumps(logs,ensure_ascii=False,indent=2)+"\n").encode(),
           "README.txt":b"TCR v1 evidence only. Missing stages are NOT_RUN/PENDING. Predictions and weights remain on server; see EVIDENCE.json.\n"}
    write_archive(destination,files,extra)
    outputs=[str(destination)]
    if include_predictions:
        error_path=destination.with_name(destination.name.replace("LIGHT","ERROR_ANALYSIS"))
        metadata={"README.json":json.dumps(clean_json(dict(scope="Existing train/val only; no test; no rerun",geometry=YAML.load(DOC/"research.yaml")["geometry_buckets"],val=read_json(OUT/"evaluation_val/metrics.json",dict(status="NOT_RUN")))),allow_nan=False).encode()}
        write_archive(error_path,prediction_files,metadata); outputs.append(str(error_path))
    print("FileZilla absolute archive paths:\n"+"\n".join(outputs))
    return dict(archives=outputs)


def write_archive(destination,files,extra):
    from c19_lif_v1_results import verify_archive
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    manifest=[dict(path=n,bytes=p.stat().st_size,sha256=sha256(p)) for n,p in sorted(files.items())]
    manifest += [dict(path=n,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in extra.items()]
    extra=dict(extra); extra["MANIFEST.json"]=json.dumps(manifest,indent=2).encode()
    with destination.open("xb") as stream,tarfile.open(fileobj=stream,mode="w:gz") as archive:
        for n,p in sorted(files.items()): archive.add(p,arcname=n,recursive=False)
        for n,value in extra.items():
            info=tarfile.TarInfo(n); info.size=len(value); archive.addfile(info,io.BytesIO(value))
    verify_archive(destination)
    digest=sha256(destination)
    Path(str(destination)+".sha256").write_text(digest+"  "+destination.name+"\n")
    write_json(Path(str(destination)+".verification.json"),dict(status="PASS",sha256=digest,bytes=destination.stat().st_size,members=len(manifest)+1))
