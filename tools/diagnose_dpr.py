"""Finite DPR failure attribution. Diagnostic reports can never authorize start."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback

import torch
from init_dpr import ROOT, MAIN, VARIANTS, SOURCE_SHA256, require, sha256, controlled_models, build_training_model, native_rebuild
from train_dpr import atomic_json, code_identity, environment, paths, recipe, timestamp
from preflight_dpr import (seed42, make_trainer, one_step, bounded_updates, move, clone_live, snapshot,
                           compare, storages, live_pair, classify_b, cpu_copy)
from c19_lif_v1_data import real_batch
from dpr_data import dataset_identity
from dpr_diagnostics import GradientObserver, gradient_mapping_check, half_decomposition, TARGET, SCHEMA
from check_dpr import audit_learned_model, audit_cuda_half
from ultralytics import RTDETR
from ultralytics.utils import YAML


def observed_pair(trainer, batch, amp, device, output):
    move(trainer, "cpu")
    first, second = clone_live(trainer), clone_live(trainer)
    require(not (storages(first)&storages(second)), "Diagnostic copies share storage")
    initial = compare(snapshot(first), snapshot(second), exact=True)
    results, gradients, forwards, mapping, kernels, added = [], [], [], [], [], []
    for index, current in enumerate((first, second)):
        move(current, device)
        seed42()
        with GradientObserver(current.model) as observation:
            row, grad, forward = one_step(current, batch, amp)
        require(row["effective_update"] and row["finite_gradients"], "Independent diagnostic update was not effective/finite")
        initial_module_state = cpu_copy(observation.initial.state_dict())
        evidence, kernel, groups = observation.evidence(grad, row["scale_before"], amp, device)
        artifact = output/f'gradient_replay_{index}.pt'
        torch.save(dict(module_initial=initial_module_state, captured=observation.values,
                        actual_unscaled_groups=groups, actual_unscaled_W=grad[TARGET+".conv.weight"],
                        scale=row["scale_before"], amp=amp, device=device, training=observation.initial.training,
                        purpose="local replay only"), artifact)
        evidence["replay_artifact"] = dict(path=str(artifact), sha256=sha256(artifact), bytes=artifact.stat().st_size)
        results.append(row); gradients.append(grad); forwards.append(forward)
        mapping.append(evidence); kernels.append(kernel); added.append(groups)
        move(current, "cpu")
        del observation
        if device == "cuda": torch.cuda.empty_cache()
    state, gradient, forward = compare(snapshot(first),snapshot(second)), compare(gradients[0],gradients[1]), compare(forwards[0],forwards[1],exact=True)
    fields = ("buffers","ema_buffers","groups","scaler","updates","epoch","start_epoch")
    a, b = snapshot(first), snapshot(second)
    exact = compare({k:a[k] for k in fields},{k:b[k] for k in fields},exact=True)
    control = dict(initial_exact=initial["exact"], storage_independent=True, forward_exact=forward["exact"],
                   loss_exact=results[0]["loss"]==results[1]["loss"], exact_metadata_buffers=exact["exact"],
                   raw_allclose=state["raw_allclose"] and gradient["raw_allclose"], state=state, gradients=gradient,
                   forward=forward, steps=results, atol=2e-5, rtol=2e-4,
                   quantization="NONE; original independent live FP32 state and gradients")
    delta = gradient_mapping_check(kernels[0]-kernels[1], {k:added[0][k]-added[1][k] for k in added[0]})
    return control, dict(schema=SCHEMA, admission_eligible=False, runs=mapping, difference_propagation=delta,
                         long_term_trajectory_equivalence="NOT_CLAIMED",
                         note="Adjoint/local replay evidence does not turn original independent gradients into allclose")


def worker(args):
    torch.set_num_threads(args.threads)
    report = dict(report_kind="dpr_failure_diagnostic_v1", observation_schema=SCHEMA, admission_eligible=False,
                  status="RUNNING", collection_status="RUNNING", formal_training="NOT_STARTED", final_test="NOT_RUN",
                  capacity="NOT_RUN_DIAGNOSTIC", checkpoint_A="NOT_RERUN_DIAGNOSTIC", variant=args.variant, modes={})
    start = time.monotonic()
    try:
        report["code_identity"] = code_identity()
        report["environment"] = environment()
        from ultralytics.nn.modules.dpr import thop_dpr_semantics
        report["profiler"] = thop_dpr_semantics()
        require(sha256(args.source)==SOURCE_SHA256, "Wrong public source")
        report["public_source_sha256"] = sha256(args.source)
        report["initialization_sha256"] = sha256(args.initialized)
        report["dataset_identity"] = dataset_identity(args.data)
        report["recipe"], report["recipe_diff"] = recipe(args.variant,args.initialized,args.data)
        parent80, _, _ = controlled_models(args.source,args.variant)
        initialized = RTDETR(str(args.initialized)).model
        seed42()
        candidate, rebuild = build_training_model(initialized.yaml,initialized,dict(nc=1,channels=3),args.variant)
        report["native_trainer_rebuild"] = rebuild
        seed42()
        parent = native_rebuild(parent80.yaml,parent80,nc=1,channels=3)
        del parent80, initialized
        batch_cpu, samples = real_batch(Path(YAML.load(args.data)["path"]),size=160,count=2)
        report["batch"] = dict(batch=2,imgsz=160,real_GT=int(batch_cpu["bboxes"].shape[0]),samples=samples,
                                scope="same bounded real-GT/DN diagnostic as original lifecycle, not B16 capacity")
        torch.save(batch_cpu,args.output/'batch.pt')
        modes = (("cpu",False,"cpu_fp32"),("cuda",False,"cuda_fp32"),("cuda",True,"cuda_native_amp"))
        for device, amp, label in modes:
            entry = report["modes"][label] = dict(status="PENDING")
            if args.device not in ("all",device) or (device=="cuda" and not torch.cuda.is_available()):
                entry["reason"]="Device excluded/unavailable"
                continue
            if amp and args.scope=="inference":
                entry["reason"]="AMP backward outside requested inference scope"
                continue
            destination=args.output/label
            destination.mkdir()
            print("BEGIN diagnostic "+label,flush=True)
            p = c = None
            try:
                batch={k:v.to(device) for k,v in batch_cpu.items()}
                seed42(); p=make_trainer(deepcopy(parent),device,amp,args.variant)
                parent_steps=[]
                for _ in range(16):
                    row,_,_=one_step(p,batch,amp);parent_steps.append(row)
                    if sum(x["effective_update"] for x in parent_steps)>=2:break
                require(sum(x["effective_update"] for x in parent_steps)>=2,"Insufficient effective parent updates")
                entry["parent_updates"]=parent_steps
                parent_b=live_pair(p,batch,amp,device) if args.scope!="inference" else None
                move(p,"cpu")
                seed42();c=make_trainer(deepcopy(candidate),device,amp,args.variant)
                entry["candidate_updates"]=bounded_updates(c,batch,amp)
                if args.scope!="inference":
                    candidate_b,adjoint=observed_pair(c,batch,amp,device,destination)
                    entry["B"]=classify_b(parent_b,candidate_b,"NOT_RERUN_DIAGNOSTIC",device)
                    entry["gradient_attribution"]=adjoint
                    atomic_json(args.output/'diagnostic.json',report)
                if not amp and args.scope!="gradients":
                    move(c,device)
                    learned=destination/'learned_diagnostic.pt'
                    torch.save(dict(model=deepcopy(c.model).cpu().float().eval(),ema=None,epoch=0,
                                    train_args={"task":"detect"},purpose="bounded diagnostic; cannot resume formal training",
                                    diagnostic_identity=dict(variant=args.variant,code=report["code_identity"],
                                        source_sha256=report["public_source_sha256"],initialization_sha256=report["initialization_sha256"],
                                        batch_sha256=sha256(args.output/'batch.pt'))),learned)
                    entry["learned_artifact"]=dict(path=str(learned),sha256=sha256(learned),bytes=learned.stat().st_size)
                    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
                    try:
                        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                        entry["inference"]=audit_learned_model(c.model.eval(),batch["img"])
                        atomic_json(args.output/'diagnostic.json',report)
                        if device=="cuda":
                            entry["original_half_checks"]=audit_cuda_half(c.model,batch["img"])
                            atomic_json(args.output/'diagnostic.json',report)
                            entry["half_decomposition"]=half_decomposition(c.model,batch["img"],p.model)
                    finally:
                        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
                entry["status"]="COLLECTED"
            except Exception:
                entry.update(status="FAILED",traceback=traceback.format_exc())
                print(entry["traceback"],flush=True)
            finally:
                if p is not None:move(p,"cpu")
                if c is not None:move(c,"cpu")
                del p,c
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                atomic_json(args.output/'diagnostic.json',report)
        report["collection_status"]="FAILED" if any(x["status"]=="FAILED" for x in report["modes"].values()) else "COMPLETED"
        report["raw_blockers"] = [mode+"/"+key for mode, item in report["modes"].items()
                                  for key in ("B", "inference", "original_half_checks")
                                  if item.get(key, {}).get("status") == "FAILED"]
        report["status"]="FAILED" if report["collection_status"]=="FAILED" else "BLOCKED" if report["raw_blockers"] else "DIAGNOSTIC_COMPLETE"
        report["reason"]="Diagnostic evidence only. Original raw B/half/backend leaves and required full preflight govern admission."
    except Exception:
        report.update(status="FAILED",collection_status="FAILED",traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        report["elapsed_seconds"]=time.monotonic()-start
        report["code_identity_at_end"]=code_identity()
        report["source_stable"]=report.get("code_identity")==report["code_identity_at_end"]
        if not report["source_stable"]:
            report.update(status="FAILED", collection_status="FAILED", reason="Source changed during collection; retained evidence cannot be reused")
        atomic_json(args.output/'diagnostic.json',report)
    print(report["collection_status"],str(args.output/'diagnostic.json'),flush=True)
    return 0 if report["collection_status"]=="COMPLETED" and report["status"]=="DIAGNOSTIC_COMPLETE" else 3


def supervise(args):
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),"--worker","--variant",args.variant,
             "--source",str(args.source),"--initialized",str(args.initialized),"--data",str(args.data),
             "--output",str(args.output),"--device",args.device,"--scope",args.scope,"--threads",str(args.threads)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",
                             start_new_session=os.name!="nt",env={**os.environ,"PYTHONUNBUFFERED":"1"})
    def stream():
        with (args.output/'worker.log').open('x',encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
    reader=threading.Thread(target=stream,daemon=True);reader.start()
    timed_out=False
    def stop_owned():
        if process.poll() is None:
            if os.name=="nt":subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
            else:os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name=="nt":process.kill()
                else:os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=10)
    try:
        exit_code=process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;stop_owned();exit_code=124
    except KeyboardInterrupt:
        stop_owned();exit_code=130
    reader.join(timeout=10)
    summary=dict(report_kind="dpr_failure_diagnostic_supervisor_v1",admission_eligible=False,
                 worker_exit=exit_code,timeout_seconds=args.timeout,timed_out=timed_out,
                 formal_training="NOT_STARTED",final_test="NOT_RUN",capacity="NOT_RUN",output=str(args.output))
    partial = json.loads((args.output/'diagnostic.json').read_text(encoding='utf-8')) if (args.output/'diagnostic.json').is_file() else {}
    summary["status"] = "TIMEOUT" if timed_out else partial.get("status", "FAILED" if exit_code else "PENDING")
    summary["collection_status"] = partial.get("collection_status", "PENDING")
    summary["missing"] = [name for name in ("diagnostic.json", "worker.log") if not (args.output/name).is_file()]
    atomic_json(args.output/'exit_status.json',summary)
    # A small independent evidence package; weights/replay tensors stay on this host.
    archive=args.output/'diagnostic_light.tar.gz'
    members=[p for p in args.output.iterdir() if p.is_file() and p.suffix in ('.json','.log')]
    sources=[ROOT/'tools'/name for name in ('diagnose_dpr.py','dpr_diagnostics.py','check_dpr.py','preflight_dpr.py','train_dpr.py','dpr_server.sh')]
    manifest={}
    with tarfile.open(archive,'x:gz') as bundle:
        for path in members+sources:
            name=('source/'+path.name) if path in sources else path.name
            bundle.add(path,arcname=name,recursive=False)
            manifest[name]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    with tarfile.open(archive) as bundle:
        for name,row in manifest.items():
            raw=bundle.extractfile(name).read();require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Archive readback failed')
    require(archive.stat().st_size<20*1024*1024,'Diagnostic light package exceeds 20 MiB')
    atomic_json(args.output/'package_manifest.json',dict(files=manifest,archive_sha256=sha256(archive),
                omitted='All .pt weights, raw replay tensors, source datasets; retained paths/hashes in diagnostic.json'))
    print(json.dumps(dict(**summary,package=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive)),indent=2),flush=True)
    return exit_code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=tuple(VARIANTS),default=MAIN)
    for name in ('source','initialized','data','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--device',choices=('cpu','cuda','all'),default='all')
    parser.add_argument('--scope',choices=('all','inference','gradients'),default='all')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();p=paths(args.variant)
    args.source=args.source or p['source'];args.initialized=args.initialized or p['init'];args.data=args.data or p['data']
    args.output=(args.output or p['meta']/('diagnose_'+timestamp())).resolve()
    require(10<=args.timeout<=1800 and 1<=args.threads<=16,'Finite timeout/thread bounds required')
    return worker(args) if args.worker else supervise(args)


if __name__=='__main__':raise SystemExit(main())
