"""Fresh R1 targeted evidence only: CUDA B1--B8 and true EMA H0--H4.

No CPU/capacity matrix, no formal admission, no training run, no final test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
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
from train_dpr import atomic_json, code_identity, environment, paths, recipe, timestamp, revoke_permit
from preflight_dpr import make_trainer, seed42, one_step, bounded_updates, move, checkpoint_a, classify_b
from dpr_r1_evidence import live_pair_r1, half_paths, batch_hash, file_record, native_probe_arguments
from dpr_acceptance import CONTRACT, SUPPLEMENT_SCHEMA, policy, scope, validate_b, validate_half, validate_supplement
from c19_lif_v1_data import real_batch
from dpr_data import dataset_identity
from ultralytics import RTDETR
from ultralytics.utils import YAML


def collect_mode(args, parent, candidate, batch_cpu, mode, output, identity, session, entry):
    device, amp = ("cpu" if mode == "cpu_fp32" else "cuda"), mode == "cuda_native_amp"
    batch = {k:v.to(device) for k,v in batch_cpu.items()}
    binding = dict(code_identity=identity, session=session, mode=mode, variant=args.variant, batch_sha256=batch_hash(batch_cpu))
    entry.update(mode=mode, policy=policy(), binding=binding, device=device, amp=amp, batch=2, imgsz=160, status="BLOCKED")
    output.mkdir(parents=True, exist_ok=False)
    seed42(); p = make_trainer(deepcopy(parent), device, amp, args.variant)
    native_probe_arguments(p)
    parent_steps = []
    for _ in range(16):
        row, _, _ = one_step(p, batch, amp); parent_steps.append(row)
        if sum(r["effective_update"] for r in parent_steps) >= 2: break
    require(sum(r["effective_update"] for r in parent_steps) >= 2, "Insufficient parent updates")
    entry["parent_updates"] = parent_steps
    (output / "parent").mkdir()
    parent_b, parent_extra = live_pair_r1(p, batch, amp, device, output / "parent")
    del p; gc.collect()
    seed42(); c = make_trainer(deepcopy(candidate), device, amp, args.variant)
    native_probe_arguments(c)
    entry["updates"] = bounded_updates(c, batch, amp)
    entry["gradients"] = entry["updates"]["gradients"]
    # A shares the actual serializer with formal training, on this fresh mode.
    entry["A"] = dict(status="FAILED", stage="not_started")
    entry["A"] = checkpoint_a(c, batch, amp, args.variant, device, output / "A", entry["A"])
    entry["A_binding"] = deepcopy(binding)
    entry["A_artifact"] = file_record(c.last)
    (output / "candidate").mkdir()
    candidate_b, candidate_extra = live_pair_r1(c, batch, amp, device, output / "candidate", candidate=True)
    entry["B"] = classify_b(parent_b, candidate_b, entry["A"]["status"], device)
    entry["B_evidence"] = dict(parent=parent_extra, candidate=candidate_extra,
        mapping=candidate_extra.pop("mapping"), updated_function=candidate_extra.pop("updated_function"))
    entry["B_evidence"]["raw_observations"] = dict(candidate_failed_names_subset_of_parent=
        set(candidate_b["state"]["failed_tensors"]) <= set(parent_b["state"]["failed_tensors"]),
        candidate_only_failed_names=sorted(set(candidate_b["state"]["failed_tensors"])-set(parent_b["state"]["failed_tensors"])))
    try:
        entry["assessment"] = validate_b(entry, mode)
        entry["status"] = "PASSED"
    except Exception as error:
        entry["assessment"] = dict(status="REJECTED", reason=str(error))
    return c


def worker(args):
    torch.set_num_threads(args.threads)
    report = dict(contract=CONTRACT, report_schema=SUPPLEMENT_SCHEMA, admission_eligible=False, policy=policy(),
                  capability_scope=scope(), session=str(args.output), variant=args.variant, status="BLOCKED",
                  formal_training="NOT_STARTED", final_test="NOT_RUN", capacity="NOT_RUN_TARGETED", modes={})
    started = time.monotonic()
    try:
        report["code_identity"] = code_identity(); report["environment"] = environment()
        require(torch.cuda.is_available(), "CUDA unavailable: targeted supplement requires GPU")
        require(sha256(args.source) == SOURCE_SHA256, "Wrong public initialization source")
        report["source_sha256"] = sha256(args.source); report["initialization_sha256"] = sha256(args.initialized)
        report["dataset_identity"] = dataset_identity(args.data)
        report["recipe"], report["recipe_diff"] = recipe(args.variant, args.initialized, args.data)
        parent80, _, _ = controlled_models(args.source, args.variant)
        initialized = RTDETR(str(args.initialized)).model
        seed42(); candidate, report["rebuild"] = build_training_model(initialized.yaml, initialized, dict(nc=1, channels=3), args.variant)
        seed42(); parent = native_rebuild(parent80.yaml, parent80, nc=1, channels=3)
        del parent80, initialized
        batch, samples = real_batch(Path(YAML.load(args.data)["path"]), size=160, count=2)
        require(len(batch["bboxes"]) > 0, "No real GT")
        torch.save(batch, args.output / "batch.pt")
        report["batch"] = dict(samples=samples, real_GT=len(batch["bboxes"]), fingerprint=batch_hash(batch), artifact=file_record(args.output / "batch.pt"))
        for mode in args.modes:
            print("BEGIN R1 " + mode, flush=True)
            entry = report["modes"][mode] = {}
            current = collect_mode(args, parent, candidate, batch, mode, args.output / mode, report["code_identity"], report["session"], entry)
            atomic_json(args.output / "supplement.json", report)
            if mode == "cuda_native_amp":
                destination = args.output / "true_ema_half"; destination.mkdir()
                report["half"] = half_paths(current, batch, destination)
                try: report["half_assessment"] = validate_half(report["half"])
                except Exception as error: report["half_assessment"] = dict(status="REJECTED", reason=str(error))
            move(current, "cpu"); del current; gc.collect(); torch.cuda.empty_cache()
        report["collection_status"] = "COMPLETED"
    except Exception:
        report.update(collection_status="FAILED", traceback=traceback.format_exc()); print(report["traceback"], flush=True)
    finally:
        half_progress = args.output / "true_ema_half/half_progress.json"
        if "half" not in report and half_progress.is_file():
            report["half"] = json.loads(half_progress.read_text(encoding="utf-8"))
        report["code_identity_at_end"] = code_identity()
        report["source_stable"] = report.get("code_identity") == report["code_identity_at_end"]
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            report["assessment"] = validate_supplement(report, identity=code_identity())
            report["status"] = "SUPPLEMENT_PASSED"
        except Exception as error:
            report["assessment"] = dict(status="BLOCKED", reason=str(error))
        atomic_json(args.output / "supplement.json", report)
    print(json.dumps(dict(status=report["status"], assessment=report["assessment"], report=str(args.output / "supplement.json"))), flush=True)
    return 0 if report["status"] == "SUPPLEMENT_PASSED" else 3


def supervise(args):
    revoke_permit(args.variant)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--variant", args.variant,
               "--source", str(args.source), "--initialized", str(args.initialized), "--data", str(args.data),
               "--output", str(args.output), "--threads", str(args.threads), "--modes", *args.modes]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               start_new_session=os.name != "nt", env={**os.environ, "PYTHONUNBUFFERED":"1"})
    def stream():
        with (args.output / "worker.log").open("x", encoding="utf-8") as log:
            for line in process.stdout: log.write(line); log.flush(); print(line, end="", flush=True)
    reader = threading.Thread(target=stream, daemon=True); reader.start()
    def stop():
        if process.poll() is None:
            if os.name == "nt": subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            else: os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt": process.kill()
                else: os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    try: code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired: stop(); code = 124
    except KeyboardInterrupt: stop(); code = 130
    finally: stop()
    reader.join(timeout=10)
    atomic_json(args.output / "exit_status.json", dict(worker_exit=code, timeout_seconds=args.timeout,
        status="BLOCKED" if code else "SUPPLEMENT_PASSED", admission_eligible=False, formal_training="NOT_STARTED", final_test="NOT_RUN"))
    # JSON/log/source only; raw tensor artifacts stay on the evidence host.
    archive = args.output / "r1_light.tar.gz"
    sources = sorted((ROOT / "tools").glob("*dpr*.py")) + [ROOT / "tools/dpr_server.sh"]
    members = list(args.output.glob("*.json")) + list(args.output.glob("*.log"))
    members += list((args.output / "true_ema_half").glob("*.json"))
    manifest = {}
    with tarfile.open(archive, "x:gz") as bundle:
        for path in members + sources:
            name = "source/" + path.name if path in sources else path.relative_to(args.output).as_posix()
            bundle.add(path, arcname=name, recursive=False); manifest[name] = file_record(path)
    with tarfile.open(archive) as bundle:
        import hashlib
        for name, row in manifest.items():
            value = bundle.extractfile(name).read()
            require(len(value) == row["bytes"] and hashlib.sha256(value).hexdigest() == row["sha256"], "Package readback failed")
    require(archive.stat().st_size < 20*1024*1024, "R1 light package exceeds 20 MiB")
    atomic_json(args.output / "package_manifest.json", dict(files=manifest, archive=file_record(archive), tensors="retained on evidence host, omitted from package"))
    return code


def verify(path, args, server=False):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    require(report.get("variant") == args.variant and report.get("source_sha256") == sha256(args.source)
            and report.get("initialization_sha256") == sha256(args.initialized)
            and report.get("dataset_identity") == dataset_identity(args.data)
            and report.get("recipe") == recipe(args.variant, args.initialized, args.data)[0], "Supplement input identity differs")
    require(report.get("environment") == environment(), "Supplement runtime differs")
    if server: require(environment()["server_environment"], "Server supplement required before full admission")
    return validate_supplement(report, code_identity())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=MAIN)
    for name in ("source", "initialized", "data", "output", "verify"): parser.add_argument("--"+name, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=("cuda_fp32", "cuda_native_amp"), default=["cuda_fp32", "cuda_native_amp"],
                        help="Bounded development subsets remain BLOCKED; both modes are required for a passing supplement")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(); p = paths(args.variant)
    require(len(args.modes) == len(set(args.modes)), "No repeated modes/retries in one supplement")
    args.source = args.source or p["source"]; args.initialized = args.initialized or p["init"]; args.data = args.data or p["data"]
    args.output = (args.output or p["meta"] / ("r1_"+timestamp())).resolve()
    require(10 <= args.timeout <= 1800 and 1 <= args.threads <= 16, "Finite timeout/thread bounds required")
    if args.verify:
        print(json.dumps(verify(args.verify, args), indent=2)); return 0
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__": raise SystemExit(main())
