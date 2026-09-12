"""Lifecycle guard tests with explicit fixtures/mocks; never launch tmux or train."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from init_c19_lif_v1 import ROOT, require, runtime, sha256, write_json
import train_c19_lif_v1 as train
import c19_lif_v1_results as results
import c19_lif_v1_data as checks


def rejected(action):
    try:
        action()
    except (RuntimeError, FileExistsError):
        return True
    raise AssertionError("Expected protective rejection")


def run(output):
    info = runtime()
    report = dict(runtime=info, scope="MOCKED lifecycle/preflight validator and real archive IO; evidence predicate tested separately; no real tmux dispatch/training", full_server_preflight="bounded_required")
    with tempfile.TemporaryDirectory(prefix="c19_lif_v1_ops_", dir=ROOT / "outputs") as d:
        temporary = Path(d)
        main, root = temporary / "main", temporary / "worktree"
        root.mkdir(); (main / "configs").mkdir(parents=True)
        (root / "docs/c19_lif_v1").mkdir(parents=True)
        (root / "docs/c19_lif_v1/c2_args.yaml").write_bytes((ROOT / "docs/c19_lif_v1/c2_args.yaml").read_bytes())
        data = main / "configs/crack_autodl.yaml"; data.write_text("names: [crack]\n")
        source = temporary / "source.pt"; source.write_bytes(b"TEST FIXTURE, NOT A CHECKPOINT")
        p = dict(name="fixture", run=main / "runs/c_series" / train.VARIANTS["c19_lif_v1"][2], launch=root / "outputs/c19_lif_v1", init=root / "init.pt", source=source,
                 c2_args=ROOT / "docs/c19_lif_v1/c2_args.yaml")
        calls = []
        def dispatch(command, **kwargs):
            calls.append(command)
            if any(str(c).endswith('check_c19_lif_v1.py') for c in command):
                write_json(root/'outputs/c19_lif_v1_preflight/checks.json', dict(status='PASSED',capacity=dict(status='PASSED',batch=16,imgsz=640,AMP=True), cpu=dict(fuse={'FP32':dict(status='PASSED',acceptance='PASSED')}), cuda=dict(fuse={k:dict(status='PASSED',acceptance='PASSED') for k in ('FP32','half_fuse','amp_fuse')})))
            return subprocess.CompletedProcess(command, 1 if "has-session" in command and not any("new-session" in c for c in calls) else 0)
        def initialization(src, dest, variant):
            dest.write_bytes(b"MOCK INITIALIZATION; NEVER TRAIN")
            return {"source_sha256": sha256(src), "status": "MOCKED"}
        def snapshot(folder, c2, config):
            write_json(folder / "source_record.json", dict(data_sha256=sha256(config)))
        with ExitStack() as stack:
            for obj, name, value in ((train, "ROOT", root), (train, "MAIN", main), (train, "paths", lambda _: p),
                                     (train, "runtime", lambda: info), (train, "initialize", initialization),
                                     (train, "record_source", snapshot), (train, "ensure_amp_resources", lambda *a: None),
                                     (train, "verify_server_environment", lambda *a: None),
                                     (train, "require_preflight", lambda *a: None),
                                     (train, "verify_data_config", lambda *a: None),
                                     (train, "verify_delivery", lambda: {"commit": info["commit"]}),
                                     (train, "duplicate_processes", lambda: []),
                                     (train, "check_det_dataset", lambda *a, **k: dict(nc=1, path=str(main), train=str(main), val=str(main), test=str(main)))):
                stack.enter_context(patch.object(obj, name, value))
            stack.enter_context(patch.object(checks, 'dataset_inventory', return_value={'val':dict(images=1728,boxes=12840),'test':dict(images=864,boxes=6663)}))
            stack.enter_context(patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "rtdetr"}))
            stack.enter_context(patch.object(train.torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(train.shutil, "which", return_value="mock-tmux"))
            stack.enter_context(patch.object(train.subprocess, "check_output", return_value=""))
            stack.enter_context(patch.object(train.subprocess, "run", side_effect=dispatch))
            train.start_direct("c19_lif_v1")
            plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
            require(plan["args"]["batch"] == 16 and plan["args"]["epochs"] == 200 and plan["full_server_preflight"] == "bounded_required", "Direct recipe changed")
            require(len([c for c in calls if "new-session" in c]) == 1, "Dispatch count")
            require(len([c for c in calls if any(str(v).endswith("check_c19_lif_v1.py") for v in c)])==1,"start-direct must run exactly one preflight")
            require(not (p["launch"] / "audit.json").exists(), "Unexpected audit gate")
            report["bounded_preflight_then_dispatch"] = "passed (mocked finite preflight and dispatch)"
            report["duplicate_reservation_rejected"] = rejected(lambda: train.start_direct("c19_lif_v1"))
            require(train.run_state(p) == "DISPATCHED", "Dispatch state")
            class FailingModel:
                def __init__(self, *a): pass
                def add_callback(self, *a): pass
                def train(self, **kwargs): raise train.torch.cuda.OutOfMemoryError("MOCK OOM")
            stack.enter_context(patch.object(train, "RTDETR", FailingModel))
            stack.enter_context(patch.object(train, "process_token", return_value="fixture-start-token"))
            try: train.worker("c19_lif_v1")
            except train.torch.cuda.OutOfMemoryError: pass
            else: raise AssertionError("OOM swallowed")
            require(train.run_state(p) == "FAILED", "OOM not failed")
            require(plan["args"]["batch"] == 16, "OOM changed batch")
            report["worker_oom_exit"] = "passed: mocked OOM propagates, exit_code=1, batch stays 16"
            # A failed subprocess must preserve its partial report, lock and init;
            # no worker may be dispatched. This is separate from the success fixture.
            failed_root,failed_main=temporary/'failed_worktree',temporary/'failed_main'
            (failed_root/'docs/c19_lif_v1').mkdir(parents=True)
            (failed_root/'docs/c19_lif_v1/c2_args.yaml').write_bytes((ROOT/'docs/c19_lif_v1/c2_args.yaml').read_bytes())
            (failed_main/'configs').mkdir(parents=True)
            (failed_main/'configs/crack_autodl.yaml').write_bytes(data.read_bytes())
            failed_paths=dict(p,run=failed_main/'runs/c_series'/train.VARIANTS['c19_lif_v1'][2],
                              launch=failed_root/'outputs/c19_lif_v1',init=failed_root/'init.pt')
            failure_calls=[]
            def failed_dispatch(command,**kwargs):
                failure_calls.append(command)
                if any(str(c).endswith('check_c19_lif_v1.py') for c in command):
                    write_json(failed_root/'outputs/c19_lif_v1_preflight/checks.json',dict(status='FAILED',capacity=dict(status='NOT_RUN'),
                        cpu=dict(fuse=dict(FP32=dict(status='FAILED_REAL_NUMERICAL_MISMATCH',failure=dict(key='lif_residual'))))))
                    raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,1 if 'has-session' in command else 0)
            with patch.object(train,'ROOT',failed_root),patch.object(train,'MAIN',failed_main),patch.object(train,'paths',return_value=failed_paths),patch.object(train.subprocess,'run',side_effect=failed_dispatch):
                try:train.start_direct('c19_lif_v1')
                except subprocess.CalledProcessError:pass
                else:raise AssertionError('Failed preflight swallowed')
                state=json.loads((failed_paths['launch']/'launch_state.json').read_text(encoding='utf-8'))
                require(state['status']=='failed' and state['partial_preflight_sha256']==sha256(failed_paths['launch']/'preflight.json'),'Partial preflight lost')
                require(failed_paths['init'].is_file() and failed_paths['run'].with_name(failed_paths['run'].name+'.c19_lif_v1.lock').is_dir(),'Failed evidence/lock lost')
                require(not any('new-session' in c for c in failure_calls),'Failed preflight dispatched a worker')
                report['failed_preflight_preserved']='passed: partial report SHA, FAILED state, init and reservation retained; no dispatch'
        # Explicit state fixtures, not actual training evidence.
        q = dict(launch=temporary / "state", run=temporary / "state_run")
        q["launch"].mkdir()
        states = [train.run_state(q)]
        write_json(q['launch']/'launch_state.json',dict(status='checking',pid=123,process_token='checking-token'))
        with patch.object(train,'process_token',return_value='checking-token'):checking_live=train.run_state(q)
        with patch.object(train,'process_token',return_value='reused-token'):checking_reused=train.run_state(q)
        require(checking_live=='CHECKING' and checking_reused=='FAILED','Checking owner token classification')
        report['checking_states']=dict(live=checking_live,reused=checking_reused)
        write_json(q['launch']/'launch_state.json',dict(status='dispatched'))
        write_json(q["launch"] / "process.json", dict(pid=123, process_token="abc"))
        write_json(q["launch"] / "training_state.json", dict(status="training"))
        with patch.object(train, "process_token", return_value="abc"): states.append(train.run_state(q))
        with patch.object(train, "process_token", return_value="different"): states.append(train.run_state(q))
        write_json(q["launch"] / "exit_code.json", dict(exit_code=0))
        (q["launch"] / "process_exit_code.txt").write_text("0\n")
        states.append(train.run_state(q))
        for name in ("weights/best.pt", "weights/last.pt", "results.csv"):
            path = q["run"] / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture")
        states.append(train.run_state(q))
        require(states == ["NOT_STARTED", "RUNNING", "FAILED", "FAILED", "SUCCESS"], "State classification")
        report["states"] = states
        # Absent val/test must stop packaging without inference or an output archive.
        with patch.object(results, "paths", return_value=p):
            report['incomplete_pack_rejected'] = rejected(lambda: results.package(temporary/'incomplete.tar.gz'))
            require(not (temporary/'incomplete.tar.gz').exists(), 'Incomplete package created')
        # Exercise the actual streaming package writer with explicit metadata-validation mocks.
        p['run'].mkdir(parents=True)
        (p['run']/'results.csv').write_bytes(os.urandom(21*1024*1024))
        with patch.object(results, 'paths', return_value=p), patch.object(results, 'missing_evidence', return_value=[]), patch.object(results, 'validate_complete', return_value=({},None)):
            archive=results.package(temporary/'fixture.tar.gz')
            results.verify_archive(archive)
            report['archive_io_fixture']=dict(bytes=archive.stat().st_size, integrity='passed', metadata_validation='MOCKED', overwrite_rejected=rejected(lambda: results.package(archive)))
        a = type("Trainer", (), {})(); a._oom_retries = 0
        train.disable_oom_retry(a); require(a._oom_retries == 3, "Native OOM guard not set")
    report["status"] = "passed"
    write_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
