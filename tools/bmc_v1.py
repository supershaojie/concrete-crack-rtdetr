"""BMC v1 lifecycle. Formal training/test are only dispatched by explicit user commands."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

from bmc_v1_common import (ROOT, OUT, MAIN, RUN, INIT, SOURCE, MODEL, BASE, BRANCH, REMOTE, NAME, SESSION,
                           require, read, write, sha, git, runtime, research, data_identity, fingerprint, verify_delivery, utc)


def prepare(options):
    import torch
    torch.set_num_threads(4)
    from ultralytics.utils import YAML
    from init_c19_lif_v1 import initialize, SOURCE_SHA256
    from bmc_v1_training import BMCTrainer
    from train_c19_lif_v1 import ensure_amp_resources
    require(not processes() and not session_active(), "This experiment is active; preserve its prepared identity")
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(status="PENDING", runtime=runtime(), source=str(options.source or SOURCE), pending=[])
    archive = ROOT / "docs/bmc_v1/parent_args.yaml"
    authority = Path(options.parent_args) if options.parent_args else MAIN / "runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
    if not authority.is_file():
        authority = archive
        report["authority_note"] = "Exact archived successful mother args; server live copy unavailable"
    expected, actual = YAML.load(archive), YAML.load(authority)
    differences = {k: [expected.get(k), actual.get(k)] for k in expected.keys() | actual.keys()
                   if type(expected.get(k)) is not type(actual.get(k)) or expected.get(k) != actual.get(k)}
    write(OUT / "parent_args_diff.json", differences)
    require(not differences, "Authoritative mother args differ; inspect parent_args_diff.json")
    args = dict(actual, model=str(INIT), name=NAME, project=str(MAIN / "runs/c_series"), save_dir=str(RUN))
    data_path = Path(options.data) if options.data else MAIN / "configs/crack_autodl.yaml"
    args["data"] = str(data_path)
    changes = {k: dict(before=actual[k], after=v, reason="experiment identity" if k in {"model", "name", "save_dir"}
                      else "explicit path relocation, matched mother data manifest") for k, v in args.items() if actual[k] != v}
    require(set(changes) <= {"model", "name", "save_dir", "project", "data"}, "Unexpected recipe modification")
    YAML.save(OUT / "train_args.yaml", args)
    shutil.copyfile(authority, OUT / "authoritative_parent_args.yaml")
    shutil.copyfile(ROOT / "configs/bmc_v1.yaml", OUT / "research.yaml")
    write(OUT / "recipe_diff.json", dict(fields=len(args), rows=changes, parent_file=str(authority), parent_sha256=sha(authority)))
    try:
        if not data_path.is_file():
            report["pending"].append(f"Missing data config: {data_path}")
        else:
            try:
                report["data"] = data_identity(data_path)
                write(OUT / "data_identity.json", report["data"])
            except FileNotFoundError as error:
                report["pending"].append("Missing dataset asset: " + str(error))
        source = Path(report["source"])
        if not source.is_file():
            report["pending"].append(f"Missing public source: {source}")
        else:
            require(sha(source) == SOURCE_SHA256, "Wrong public untrained source hash")
            if INIT.exists():
                prior = read(OUT / "initialization.json")
                require(sha(INIT) == prior["output_sha256"] and prior["source_sha256"] == SOURCE_SHA256, "Existing init identity changed")
            else:
                write(OUT / "initialization.json", initialize(source, INIT))
            # Real Trainer.get_model reconstruction, with native RNG/class adaptation.
            from ultralytics import RTDETR
            source_model = RTDETR(str(INIT)).model
            trainer = BMCTrainer.__new__(BMCTrainer)
            trainer.data, trainer.audit_folder, trainer.bmc_config = dict(nc=1, channels=3), OUT, research()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                rebuilt = trainer.get_model(source_model.yaml, source_model, verbose=False)
            require(sum(p.numel() for p in rebuilt.parameters()) == 20149765, "Unexpected parameter count")
            report.update(init_sha256=sha(INIT), source_sha256=SOURCE_SHA256, trainer_initialization=trainer.initialization_audit)
        try:
            ensure_amp_resources(MAIN, OUT / "amp_resources.json")
        except (OSError, ValueError) as error:
            report["pending"].append("AMP resources: " + repr(error))
        report["status"] = "PASS" if not report["pending"] else "PENDING"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        write(OUT / "prepare.json", report)
    return report


def preflight():
    """A process-group watchdog bounds the whole server gate, including setup/checks."""
    command = [sys.executable, str(ROOT / "tools/bmc_v1.py"), "_preflight"]
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=os.name == "posix")
    try:
        code = process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        report = dict(status="FAIL", error="Whole preflight watchdog reached 900 seconds", finished=utc())
        write(OUT / "preflight.json", report)
        return report
    require(code == 0, "Preflight failed; inspect outputs/bmc_v1/preflight.json and console")
    return read(OUT / "preflight.json")


def preflight_body():
    import torch
    from check_bmc_v1 import run
    verify_delivery()
    identity = fingerprint()
    path = OUT / "preflight.json"
    if path.exists():
        prior = read(path)
        if prior.get("status") == "PASS" and prior.get("fingerprint") == identity["sha256"]:
            return dict(prior, reused=True)
    report = dict(status="PENDING", fingerprint=identity["sha256"], identity=identity["identity"],
                  started=utc(), capacity="PENDING", runtime=runtime())
    try:
        report["cpu_checks"] = run("cpu")
        from check_bmc_v1_lifecycle import lifecycle_checks
        report["lifecycle"] = lifecycle_checks("0" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3:
            report["reason"] = "Missing CUDA or local small-GPU interface only; server B16/640 capacity PENDING"
            return report
        capacity_path = OUT / f"capacity_{utc()}.json"
        from bmc_v1_diagnostics import capacity
        report["capacity"] = capacity(capacity_path)
        require(report["capacity"]["status"] == "PASS", "Bounded capacity failed")
        require(identity["sha256"] == fingerprint()["sha256"], "Inputs changed during preflight")
        report["status"] = "PASS"
        return report
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["finished"] = utc()
        write(path, report)


def processes():
    found = []
    if os.name != "posix":
        return found
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv = folder.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if str(ROOT / "tools/bmc_v1.py") in argv and "_worker" in argv:
                found.append(dict(pid=int(folder.name), argv=argv,
                                  token=folder.joinpath("stat").read_text().split(")", 1)[1].split()[19]))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def session_active():
    return bool(shutil.which("tmux") and subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True).returncode == 0)


def status():
    dispatch_file = OUT / "dispatch.json"
    if not dispatch_file.exists():
        return dict(state="NOT_STARTED", processes=processes())
    dispatch = read(dispatch_file)
    identity = dispatch["id"]
    state_file, exit_file = OUT / f"state_{identity}.json", OUT / f"exit_{identity}.json"
    state = read(state_file) if state_file.exists() else dict(state="DISPATCHED")
    active = processes()
    matching = [p for p in active if identity in p["argv"]]
    if exit_file.exists():
        ending = read(exit_file)
        state.update(state="COMPLETED" if ending["python_exit_code"] == 0 and ending["tee_exit_code"] == 0 else "FAILED", exit=ending)
    elif not matching and not session_active():
        state.update(state="FAILED", reason="Dispatched worker/session absent; exit not recorded")
    log = OUT / f"console_{identity}.log"
    tail = ""
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size-8192))
            tail = stream.read().decode(errors="replace")
    children = []
    for process in matching:
        child = subprocess.run(["ps", "-o", "pid,ppid,args", "--ppid", str(process["pid"])], capture_output=True, text=True)
        children.append(child.stdout)
    return dict(**state, dispatch=identity, processes=active, child_processes=children, log=str(log), log_tail=tail)


@contextmanager
def dispatch_lock():
    require(os.name == "posix", "tmux dispatch runs on the Linux server")
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def archive_failed_run():
    require(not processes() and not session_active(), "Experiment active; preserving run")
    require(RUN.resolve().parent == (MAIN / "runs/c_series").resolve() and RUN.name == NAME, "Archive path outside experiment")
    require(RUN.is_dir() and not list(RUN.rglob("*.pt")), "Run has checkpoints; use genuine resume")
    results = RUN / "results.csv"
    if results.exists():
        with results.open(newline="", encoding="utf-8") as stream:
            require(not list(csv.DictReader(stream)), "Run has training result rows; preserve it")
    archived = RUN.with_name(RUN.name + "_failed_" + utc())
    RUN.rename(archived)
    write(OUT / f"archived_{utc()}.json", dict(source=str(RUN), destination=str(archived)))
    return archived


def start(resume=False, archive_failed=False):
    with dispatch_lock():
        verify_delivery()
        require(not processes() and not session_active(), "This experiment already has an active worker/tmux")
        check = read(OUT / "preflight.json")
        identity = fingerprint()
        require(check["status"] == "PASS" and check["fingerprint"] == identity["sha256"], "No current BMC engineering preflight PASS")
        if resume:
            from ultralytics.utils.patches import torch_load
            checkpoint = RUN / "weights/last.pt"
            require(checkpoint.is_file(), "No valid last.pt; resume cannot repair initialization failures")
            state = torch_load(checkpoint, map_location="cpu")
            require(state.get("epoch", -1) >= 0 and all(state.get(k) is not None for k in ("optimizer", "scaler", "ema")), "Checkpoint is stripped/incomplete")
            require(state["train_args"]["name"] == NAME and Path(state["train_args"]["save_dir"]).resolve() == RUN.resolve(), "Wrong experiment checkpoint")
            require(state["epoch"] < 199, "200 epochs already reached")
        elif RUN.exists():
            require(archive_failed, "Run directory exists. Use resume for a valid last, or start --archive-failed for an empty failed setup")
            archive_failed_run()
        require(shutil.which("tmux"), "tmux is missing")
        identity_id = utc() + "_" + uuid.uuid4().hex[:8]
        dispatch = dict(id=identity_id, sha=git("rev-parse", "HEAD"), resume=resume,
                        fingerprint=identity["sha256"], created=utc(), run=str(RUN))
        write(OUT / "dispatch.json", dispatch)
        write(OUT / f"dispatch_{identity_id}.json", dispatch)
        write(OUT / f"state_{identity_id}.json", dict(state="DISPATCHED", dispatch=identity_id))
        command = "bash " + shlex.quote(str(ROOT / "tools/bmc_v1.sh")) + " _session " + shlex.quote(identity_id)
        completed = subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "-c", str(ROOT), command], capture_output=True, text=True)
        if completed.returncode:
            write(OUT / f"exit_{identity_id}.json", dict(python_exit_code=completed.returncode, tee_exit_code=0,
                                                       error=completed.stderr, dispatch=identity_id))
            raise RuntimeError(completed.stderr)
        return dict(state="DISPATCHED", dispatch=identity_id, session=SESSION, log=str(OUT / f"console_{identity_id}.log"))


def worker(identity_id):
    import torch
    from ultralytics.utils import YAML
    from bmc_v1_training import BMCTrainer, EpochCollector
    dispatch = read(OUT / f"dispatch_{identity_id}.json")
    require(read(OUT / "dispatch.json")["id"] == identity_id, "Stale dispatch")
    verify_delivery()
    require(dispatch["fingerprint"] == fingerprint()["sha256"], "Dispatch inputs changed")
    state_file = OUT / f"state_{identity_id}.json"
    write(state_file, dict(state="SETTING_UP", dispatch=identity_id, pid=os.getpid()))
    args = YAML.load(OUT / "train_args.yaml")
    if dispatch["resume"]:
        args["resume"] = str(RUN / "weights/last.pt")
    else:
        require(not RUN.exists(), "Run appeared since dispatch; preserved")
    trainer = BMCTrainer(overrides=args)
    trainer.audit_folder, trainer.dispatch = OUT, identity_id
    mechanism = OUT / "mechanism"
    mechanism.mkdir(exist_ok=True)
    trainer.mechanism_collector = EpochCollector(mechanism, identity_id)

    def setup(t):
        require(bool(t.amp), "Native check_amp disabled AMP; refusing recipe change")
        expected = YAML.load(OUT / "train_args.yaml")
        allowed = {"model", "resume"} if dispatch["resume"] else set()
        differences = {k: [v, getattr(t.args, k, None)] for k, v in expected.items() if k not in allowed and
                       (type(getattr(t.args, k, None)) is not type(v) or getattr(t.args, k, None) != v)}
        require(not differences, f"Effective recipe differs: {differences}")
        require(Path(t.save_dir).resolve() == RUN.resolve(), "Native Trainer incremented the run name")
        require(all(p.requires_grad for p in t.model.parameters()), "Unexpected frozen parameter")
        ids = [id(p) for g in t.optimizer.param_groups for p in g["params"]]
        require(all(ids.count(id(p)) == 1 for p in t.model.parameters()), "Missing/duplicated optimizer parameter")
        YAML.save(OUT / "actual_train_args.yaml", vars(t.args))
        write(OUT / f"setup_{identity_id}.json", dict(amp=t.amp, scaler=t.scaler.state_dict(), start_epoch=t.start_epoch,
                criterion_epoch=t.model.criterion.epoch, optimizer=type(t.optimizer).__name__, accumulate=t.accumulate,
                parameter_groups=[dict(kind=g.get("param_group"), count=len(g["params"]), lr=g["lr"], weight_decay=g["weight_decay"]) for g in t.optimizer.param_groups]))

    def processed(t):
        write(state_file, dict(state="RUNNING", phase="TRAINING_RUNNING", dispatch=identity_id, pid=os.getpid(),
                              epoch_zero_based=t.epoch, processed_batches=t.mechanism_collector.step, updated=utc()))

    trainer.add_callback("on_train_start", setup)
    trainer.add_callback("on_train_batch_end", processed)
    try:
        trainer.train()
    except BaseException as error:
        if trainer.mechanism_collector.batches:
            partial = trainer.mechanism_collector.summary()
            partial.update(completion="PARTIAL_ON_EXCEPTION", error=repr(error))
            write(mechanism / f"partial_{identity_id}_{utc()}.json", partial)
        raise
    write(state_file, dict(state="COMPLETED", dispatch=identity_id, pid=os.getpid(), actual_last_epoch_zero_based=trainer.epoch,
                          actual_completed_epochs=trainer.epoch+1, early_stopped=trainer.epoch+1 < trainer.epochs, finished=utc()))
    return dict(state="COMPLETED", epochs=trainer.epoch+1)


def delivery(full_sha):
    require(len(full_sha) == 40 and full_sha == git("rev-parse", "HEAD"), "Full SHA does not equal checkout")
    require(git("branch", "--show-current") == BRANCH and git("merge-base", BASE, "HEAD") == BASE, "Wrong experiment branch ancestry")
    require(git("remote", "get-url", "origin") == REMOTE, "Wrong remote")
    write(OUT / "delivery.json", dict(sha=full_sha, base=BASE, branch=BRANCH, worktree=str(ROOT.resolve()), main=str(MAIN.resolve()), created=utc()))
    from bmc_v1_results import delivery_documents
    delivery_documents(full_sha)
    return read(OUT / "delivery.json")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest="action", required=True)
    p = actions.add_parser("prepare")
    p.add_argument("--source", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--parent-args", type=Path)
    actions.add_parser("preflight")
    actions.add_parser("_preflight")
    p = actions.add_parser("probe")
    p.add_argument("--parent-best", type=Path)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--device", default="0")
    p = actions.add_parser("start")
    p.add_argument("--archive-failed", action="store_true")
    for action in ("status", "resume", "val", "test", "pack"):
        actions.add_parser(action)
    p = actions.add_parser("_worker")
    p.add_argument("--dispatch", required=True)
    p = actions.add_parser("_exit")
    p.add_argument("--dispatch", required=True)
    p.add_argument("--code", type=int, required=True)
    p.add_argument("--tee-code", type=int, required=True)
    p = actions.add_parser("_delivery")
    p.add_argument("--sha", required=True)
    p = actions.add_parser("_capacity")
    p.add_argument("--report", type=Path, required=True)
    return result


def main():
    options = parser().parse_args()
    os.environ["YOLO_AUTOINSTALL"] = "false"
    OUT.mkdir(parents=True, exist_ok=True)
    action = options.action
    if action == "prepare":
        result = prepare(options)
    elif action == "preflight":
        result = preflight()
    elif action == "_preflight":
        result = preflight_body()
    elif action == "probe":
        from bmc_v1_diagnostics import probe
        result = probe(options.parent_best, options.count, options.device)
    elif action == "start":
        result = start(archive_failed=options.archive_failed)
    elif action == "resume":
        result = start(resume=True)
    elif action == "status":
        result = status()
    elif action == "_worker":
        result = worker(options.dispatch)
    elif action == "_exit":
        result = dict(dispatch=options.dispatch, python_exit_code=options.code, tee_exit_code=options.tee_code, finished=utc())
        write(OUT / f"exit_{options.dispatch}.json", result)
    elif action == "_delivery":
        result = delivery(options.sha)
    elif action == "_capacity":
        from bmc_v1_diagnostics import capacity
        result = capacity(options.report)
    else:
        from bmc_v1_results import evaluate, package
        result = package() if action == "pack" else evaluate(action)
    print(json.dumps({k: v for k, v in result.items() if k not in
          {"data", "trainer_initialization", "identity", "cpu_checks", "images", "model"}}, ensure_ascii=False, indent=2))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if not isinstance(error, SystemExit):
            write(OUT / f"error_{utc()}.json", dict(error=repr(error), traceback=traceback.format_exc(), argv=sys.argv))
        raise
