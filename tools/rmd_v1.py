"""RMD v1 operations: prepare/preflight/probe/status/start/resume/val/test/pack. No command implicitly starts long training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback

from rmd_v1_common import *
from ultralytics.models.rtdetr.rmd_v1 import RMDTrainer, RMDDetectionModel, sync_epoch
from ultralytics.utils.torch_utils import unwrap_model


def process_token(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def workers():
    result = []
    for file in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            parts = file.read_bytes().decode().split("\0")
            if str(ROOT/"tools/rmd_v1.py") in parts and "_worker" in parts and int(file.parent.name) != os.getpid():
                result.append(dict(pid=int(file.parent.name), token=process_token(file.parent.name), command=parts))
        except (OSError, UnicodeDecodeError):
            continue
    return result


def session_exists():
    return bool(shutil.which("tmux") and subprocess.run(["tmux", "has-session", "-t", "="+SESSION], capture_output=True).returncode == 0)


def checkpoint_identity(p):
    records = {}
    for name in ("best", "last"):
        path = p["run"]/f"weights/{name}.pt"
        records[name] = dict(path=str(path), exists=path.is_file(), sha256=sha256(path) if path.is_file() else None,
                             bytes=path.stat().st_size if path.is_file() else None)
    return records


def status(p):
    current = read_json(OUT/"current_attempt.json", {})
    attempt = Path(current["attempt"]) if current else None
    def shell_code(name):
        path = attempt/name if attempt else None
        return path.read_text().strip() if path and path.is_file() else None
    return dict(runtime=runtime(), workers=workers(), tmux=dict(name=SESSION, exists=session_exists()),
                dispatch=read_json(attempt/"dispatch.json") if attempt else None,
                process=read_json(attempt/"process.json") if attempt else None,
                python_exit_code=shell_code("python_exit_code.txt"), tee_exit_code=shell_code("tee_exit_code.txt"),
                exit=read_json(OUT/"latest_exit.json"), training=read_json(OUT/"training_completed.json"),
                last_epoch=read_json(OUT/"epoch.json"), final_eval=read_json(OUT/"final_eval.json"),
                preflight=read_json(OUT/"preflight.json"), probe=read_json(OUT/"probe.json"),
                checkpoints={n: dict(path=str(p["run"]/f"weights/{n}.pt"), exists=(p["run"]/f"weights/{n}.pt").is_file()) for n in ("best", "last")},
                console_log=str(OUT/"console.log"), attempt_log=str(attempt/"console.log") if attempt else None,
                recovery_records=[str(f) for f in (OUT/"recovery").glob("*.json")])


def gates(prepared):
    pointer = read_json(OUT/"preflight.json")
    require(pointer and not pointer.get("timed_out") and pointer.get("process_exit_code") == 0,
            "A completed, bounded preflight is required")
    require(sha256(pointer["report"]) == pointer["report_sha256"], "Preflight report changed")
    report = read_json(pointer["report"])
    require(report.get("status") != "FAIL" and pointer["status"] == report["status"], "Failed/inconsistent preflight report")
    require(report.get("binding") == prepared["binding"], "Preflight code/data/init/recipe binding changed")
    require(report.get("micro_batches", 17) <= 16 and report.get("elapsed_seconds", 901) <= 900, "Preflight bounds exceeded or missing")
    for key in ("correctness", "capacity", "resume", "lifecycle"):
        require(report.get(key, {}).get("status") == "PASS", f"Mandatory preflight gate {key} is not PASS")
    require(report["capacity"]["effective_updates"] >= 1 and report["capacity"]["batch"] == 16 and report["capacity"]["imgsz"] == 640,
            "Capacity proof lacks a genuine B16/640 optimizer update")
    app = report["applicability"]
    later = read_json(OUT/"probe.json")
    if later:
        require(not later.get("timed_out") and later.get("process_exit_code") == 0, "Latest explicit probe failed or timed out")
        require(sha256(later["report"]) == later["report_sha256"], "Probe report changed")
        probe = read_json(later["report"])
        require(probe.get("binding") == prepared["binding"], "Probe identity changed")
        app = probe["applicability"]
    require(app.get("status") == "APPLICABILITY_PASS", "Formal start refused: " + app.get("status", "APPLICABILITY_PENDING") + ": " + app.get("reason", ""))
    negative = read_json(OUT/"not_applicable.json", {})
    require(not (negative.get("binding") == prepared["binding"] and
                 negative.get("parent_sha256") == app["trained_mother"]["sha256"]),
            "NOT_APPLICABLE is retained for this code/data/mother identity; do not rerun probes to fish for PASS")
    require(app["valid_batches"] >= 8 and app["gt_coverage_k_ge_2"] >= .5 and app["mean_abs_weight_minus_one"] >= .01,
            "Fixed applicability numbers do not satisfy the saved verdict")
    parent = app["trained_mother"]
    require(Path(parent["path"]).is_file() and sha256(parent["path"]) == parent["sha256"], "Probe mother checkpoint identity changed")
    return dict(preflight=pointer, explicit_probe=later, applicability=app)


def valid_resume(p, identity):
    require(not (OUT/"training_completed.json").exists(), "Training already completed; use val --recover-final-eval, not resume")
    last = p["run"]/"weights/last.pt"
    require(last.is_file(), "Resume requires this experiment's last.pt")
    previous = read_json(OUT/"checkpoint_identity.json")
    require(previous and previous["last"]["sha256"] == sha256(last), "Last checkpoint differs from native save record")
    ckpt = torch_load(last, map_location="cpu")
    model = ckpt.get("ema") or ckpt.get("model")
    require(isinstance(model, RMDDetectionModel), "Resume checkpoint is not RMD")
    verify_pair(model)
    require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values() if v.is_floating_point()), "Nonfinite last checkpoint")
    require(getattr(model, "rmd_identity", None) == identity["model_identity"], "Resume checkpoint belongs to a different run/code/initialization")
    require(ckpt.get("epoch", -1) >= 0 and all(ckpt.get(k) is not None for k in ("optimizer", "scaler", "ema", "updates")),
            "Last checkpoint lacks full native optimizer/EMA/scaler state")
    for k, value in identity["prepared"]["args"].items():
        if k not in {"model", "resume"}:
            require(type(ckpt["train_args"].get(k)) is type(value) and ckpt["train_args"].get(k) == value, "Resume recipe changed: " + k)
    return last, ckpt["epoch"]+1


def dispatch(p, resume=False):
    require(os.name == "posix" and Path(sys.executable).resolve() == SERVER_PYTHON.resolve(), "Formal worker must use the specified server rtdetr Python")
    require(torch.cuda.is_available() and shutil.which("tmux"), "Formal CUDA and tmux are required")
    require(not workers() and not session_exists(), "This experiment already has an active worker/tmux session")
    require(not (OUT/"check.lock").exists(), "Wait for the running preflight/probe to finish")
    prepared = verify_prepared(p, clean=True)
    gate_record = gates(prepared)
    offline_amp_resources(p)
    identity = read_json(OUT/"training_identity.json")
    if resume:
        require(identity and identity["prepared"] == prepared, "Original training identity changed")
        last, next_epoch = valid_resume(p, identity)
    else:
        require(not p["run"].exists(), "Formal run exists; never overwrite it with exist_ok=True")
        require(identity is None and not (OUT/"current_attempt.json").exists(), "Experiment already dispatched; inspect status before recovery")
        next_epoch = 0
    lock = OUT/"dispatch.lock"
    lock.mkdir(exist_ok=False)
    try:
        require(not workers() and not session_exists(), "Concurrent dispatch detected")
        attempt = OUT/"attempts"/unique_name(); attempt.mkdir(parents=True, exist_ok=False)
        if not resume:
            source_snapshot(OUT/"training_source")
            identity = dict(created=now(), prepared=prepared, gates=gate_record,
                            model_identity=dict(scope="formal", formula=FORMULA, run=str(p["run"]), training_sha=prepared["code"]["commit"],
                                                training_binding=prepared["binding"], source_sha256=prepared["source_sha256"]))
            write_json(OUT/"training_identity.json", identity)
        command = [str(SERVER_PYTHON), "-u", str(ROOT/"tools/rmd_v1.py"), "_worker", "--main", str(p["main"]),
                   "--data", str(p["data"]), "--c2-args", str(p["c2_args"]), "--attempt", str(attempt)]
        if resume: command.append("--resume-worker")
        quote = shlex.quote
        shell = "\n".join(["#!/usr/bin/env bash", "set -u -o pipefail", "cd " + quote(str(ROOT)),
            "export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false", "export PYTHONPATH=" + quote(str(ROOT/"ultralytics-main")+":"+str(ROOT/"tools")),
            "set +e", " ".join(map(quote, command))+" 2>&1 | tee -a "+quote(str(attempt/"console.log"))+" "+quote(str(OUT/"console.log")),
            'codes=("${PIPESTATUS[@]}")', 'python_code="${codes[0]}"', 'tee_code="${codes[1]}"',
            'printf "%s\\n" "$python_code" > '+quote(str(attempt/"python_exit_code.txt")),
            'printf "%s\\n" "$tee_code" > '+quote(str(attempt/"tee_exit_code.txt")),
            'if [ "$python_code" -ne 0 ]; then exit "$python_code"; fi', 'exit "$tee_code"', ""])
        script = attempt/"worker.sh"
        with script.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(shell)
        dispatch_record = dict(status="DISPATCHED", time=now(), command=command, worker_script=str(script), worker_sha256=sha256(script),
                               tmux=SESSION, next_epoch=next_epoch, resume=resume, binding=prepared["binding"])
        write_json(attempt/"dispatch.json", dispatch_record)
        write_json(OUT/"current_attempt.json", dict(attempt=str(attempt)))
        try:
            subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "bash "+quote(str(script))], check=True, timeout=15)
        except BaseException as error:
            write_json(attempt/"dispatch_failure.json", dict(error=repr(error), time=now()))
            raise
        return dispatch_record
    finally:
        lock.rmdir()  # remove only our own empty reservation, never another experiment's state


class RunTrainer(RMDTrainer):
    def __init__(self, p, identity, attempt, **kwargs):
        self.rmd_paths, self.rmd_identity, self.rmd_attempt = p, identity, attempt
        self.rmd_rows = []
        super().__init__(**kwargs)
        self.add_callback("on_train_epoch_start", self.epoch_start)
        self.add_callback("on_train_batch_start", self.no_batch_reduction)
        self.add_callback("on_train_batch_end", self.batch_end)
        self.add_callback("on_train_epoch_end", self.epoch_end)
        self.add_callback("on_model_save", self.saved)

    def _setup_train(self):
        with native_amp_guard() as messages:
            super()._setup_train()
        require(self.amp and isinstance(self.model, RMDDetectionModel), "Actual native AMP/RMD training model missing")
        verify_pair(self.model)
        for k, v in self.rmd_identity["prepared"]["args"].items():
            if self.resume and k in {"model", "resume"}: continue
            require(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v, "Actual recipe changed: " + k)
        from rmd_v1_preflight import audit_optimizer
        audit = audit_optimizer(self) if not self.resume else dict(status="PASS", groups=optimizer_groups(self.model, self.optimizer),
                                                                  resume="original checkpoint optimizer restored by native trainer")
        self.model.rmd_identity = deepcopy(self.rmd_identity["model_identity"])
        self.ema.ema.rmd_identity = deepcopy(self.rmd_identity["model_identity"])
        YAML.save(self.rmd_attempt/"actual_train_args.yaml", vars(self.args))
        write_json(self.rmd_attempt/"setup.json", dict(amp_messages=messages, optimizer_audit=audit,
                   rebuild=self.rmd_rebuild_audit, start_epoch=self.start_epoch, runtime=runtime()))

    @staticmethod
    def no_batch_reduction(trainer):
        trainer._oom_retries = 3

    @staticmethod
    def epoch_start(trainer):
        trainer.rmd_rows = []

    @staticmethod
    def batch_end(trainer):
        row = deepcopy(unwrap_model(trainer.model).criterion.statistics)
        require(row is not None and row["epoch"] == trainer.epoch, "Epoch/criterion diagnostics out of sync")
        trainer.rmd_rows.append(row)
        append_json(OUT/"mechanism_batches.jsonl", dict(row, attempt=str(trainer.rmd_attempt)))

    @staticmethod
    def epoch_end(trainer):
        row = dict(aggregate(trainer.rmd_rows), epoch=trainer.epoch, r=max(0., min(1., (trainer.epoch-5)/15)),
                   attempt=str(trainer.rmd_attempt), formula=FORMULA, finished=now())
        append_json(OUT/"mechanism_epochs.jsonl", row)

    @staticmethod
    def saved(trainer):
        write_json(OUT/"checkpoint_identity.json", checkpoint_identity(trainer.rmd_paths))
        write_json(OUT/"epoch.json", dict(completed_epochs=trainer.epoch+1, fitness=trainer.fitness,
                   best_fitness=trainer.best_fitness, stop=bool(trainer.stop), saved=now(),
                   training_binding=trainer.rmd_identity["prepared"]["binding"]))

    def final_eval(self):
        # Called only after the native loop reaches its original stopping condition.
        require(self.stop and (self.epoch+1 >= self.epochs or self.epoch+1-self.stopper.best_epoch >= self.args.patience),
                "No native epoch/patience training completion evidence")
        write_json(OUT/"training_completed.json", dict(status="TRAINING_COMPLETED", time=now(), completed_epochs=self.epoch+1,
                   reason="epochs" if self.epoch+1 >= self.epochs else "patience", training_binding=self.rmd_identity["prepared"]["binding"],
                   native_checkpoint_before_final_eval=checkpoint_identity(self.rmd_paths)))
        write_json(OUT/"final_eval.json", dict(status="FINAL_EVAL_RUNNING", training_sha=self.rmd_identity["prepared"]["code"]["commit"]))
        try:
            super().final_eval()  # preserve native AMP/validation/strip/checkpoint strategy
            write_json(OUT/"final_eval.json", dict(status="FINAL_EVAL_COMPLETED", time=now(), metrics=self.metrics))
        except BaseException as error:
            write_json(OUT/"final_eval.json", dict(status="FINAL_EVAL_FAILED", time=now(), error=repr(error), traceback=traceback.format_exc()))
            raise
        finally:
            write_json(OUT/"checkpoint_identity.json", checkpoint_identity(self.rmd_paths))


def worker(p, attempt, resume=False):
    attempt = Path(attempt).resolve(); code = 1
    try:
        require(os.environ.get("TMUX"), "Formal worker requires its independent tmux session")
        name = subprocess.check_output(["tmux", "display-message", "-p", "#S"], text=True).strip()
        require(name == SESSION and not workers(), "Wrong tmux session or duplicate RMD worker")
        require(read_json(OUT/"current_attempt.json")["attempt"] == str(attempt), "Worker dispatch identity mismatch")
        prepared = verify_prepared(p, clean=True); gates(prepared)
        identity = read_json(OUT/"training_identity.json")
        require(identity["prepared"] == prepared, "Original training identity changed")
        args = dict(prepared["args"])
        if resume:
            last, _ = valid_resume(p, identity)
            args.update(model=str(last), resume=str(last))
        else:
            require(not p["run"].exists(), "Run appeared after dispatch; preserved")
        write_json(attempt/"process.json", dict(pid=os.getpid(), token=process_token(os.getpid()), started=now(), command=sys.argv))
        trainer = RunTrainer(p, identity, attempt, overrides=args)
        trainer.train()
        require(read_json(OUT/"training_completed.json", {}).get("status") == "TRAINING_COMPLETED", "No training completion record")
        code = 0
    except KeyboardInterrupt:
        code = 130; raise
    except BaseException:
        write_json(attempt/"failure.json", dict(time=now(), traceback=traceback.format_exc()))
        raise
    finally:
        record = dict(exit_code=code, finished=now(), attempt=str(attempt), training=read_json(OUT/"training_completed.json"), final_eval=read_json(OUT/"final_eval.json"))
        write_json(attempt/"exit.json", record); write_json(OUT/"latest_exit.json", record)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "probe", "status", "start", "resume", "val", "test", "pack",
                    "_worker", "_check-worker", "_lifecycle", "_resume-check"))
    p.add_argument("--main", type=Path, default=SERVER_MAIN)
    p.add_argument("--data", type=Path)
    p.add_argument("--c2-args", type=Path)
    p.add_argument("--probe-weights", type=Path, help="Original trained C19+LIF best.pt, used only in the isolated probe")
    p.add_argument("--local", action="store_true", help="CPU correctness only; server capacity/applicability remain PENDING")
    p.add_argument("--include-predictions", action="store_true", help="Include separate val/test all-query prediction streams in pack")
    p.add_argument("--full", action="store_true", help="Optional FULL pack also includes final best/last weights")
    p.add_argument("--recover-final-eval", action="store_true", help="Evaluate completed training after native final_eval failure; preserve original failure/exit")
    p.add_argument("--allow-eval-revision", action="store_true", help="Allow only documented evaluator/warmup file changes; retain original training SHA")
    p.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--resume-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--probe-only", action="store_true", help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    p = settings(args.main, args.data, args.c2_args)
    command = args.command
    if command == "prepare": result = prepare(p)
    elif command == "status": result = status(p)
    elif command in ("start", "resume"): result = dispatch(p, resume=command == "resume")
    elif command == "_worker": result = worker(p, args.attempt, args.resume_worker)
    elif command in ("preflight", "probe", "_check-worker"):
        from rmd_v1_preflight import run_bounded, child
        require(not workers() and not session_exists(), "Diagnostics are isolated from active formal training")
        if command == "_check-worker": result = child(p, args.output, args.probe_only, args.local, args.probe_weights)
        else: result = run_bounded(p, command == "probe", args.local, args.probe_weights)
    elif command == "_resume-check":
        from rmd_v1_preflight import resume_check
        result = resume_check(args.checkpoint, args.output)
    elif command == "_lifecycle":
        from rmd_v1_eval import evaluate_checkpoint
        result = evaluate_checkpoint(args.checkpoint, args.data, args.output, one_batch=True)
    elif command in ("val", "test"):
        from rmd_v1_eval import evaluate_formal
        result = evaluate_formal(p, command, workers(), args.allow_eval_revision, args.recover_final_eval)
    elif command == "pack":
        from rmd_v1_eval import pack
        require(not workers(), "Pack after worker exits so evidence and checkpoint hashes are stable")
        result = pack(p, args.include_predictions, args.full)
    if result is not None:
        display = result
        if command == "prepare":
            display = dict(status=result["status"], binding=result["binding"], report=str(OUT/"prepare.json"),
                           source_sha256=result["source_sha256"], init_sha256=result["init_sha256"],
                           data=result["data"]["splits"], next="preflight (never starts formal training)")
        elif command in ("preflight", "probe", "_check-worker"):
            display = {k: result[k] for k in ("status", "binding", "elapsed_seconds", "micro_batches", "error", "reason") if k in result}
            display["gates"] = {k: result.get(k, {}).get("status", "PENDING") for k in ("correctness", "capacity", "resume", "lifecycle", "applicability")}
            display["report_pointer"] = str(OUT/("probe.json" if command == "probe" or args.probe_only else "preflight.json"))
        elif command in ("val", "test"):
            display = {k: result[k] for k in ("status", "report", "report_sha256")}
        elif command == "pack":
            display = {k: result[k] for k in ("path", "sha256", "files", "format", "weights", "includes_predictions", "prediction_files")}
        print(json.dumps(strict_record(display), ensure_ascii=False, indent=2, allow_nan=False))
    if command in ("preflight", "probe"):
        return 0 if result["status"] in ("PASS", "APPLICABILITY_PASS") else 2
    if command == "_check-worker":
        return 1 if result["status"] == "FAIL" else 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print("RMD REFUSED/FAILED:", str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
