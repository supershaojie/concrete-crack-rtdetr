"""Direct AutoDL lifecycle; no full-preflight/pass marker dependency."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
import urllib.request

import torch
from init_c26 import ROOT, VARIANTS, initialize, require, runtime, sha256, verify_model, write_json, is_added, C24_COMMIT
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML, ASSETS
from ultralytics.data.utils import check_det_dataset

MAIN = Path(os.environ.get("C26_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))


def verify_server_environment(info):
    require(sys.version_info[:2] == (3, 10), "C26 formal server requires existing Python 3.10")
    require(info["torch"] == "2.1.2+cu121" and info["cuda"] == "12.1", "Expected existing PyTorch 2.1.2+cu121; do not upgrade")
    require("4090" in (info["gpu"] or ""), "Expected RTX 4090 formal server")


def record_source(folder, c2_args, data):
    """Freeze reproducibility evidence before dispatch, without model inference."""
    shutil.copyfile(c2_args, folder / "authoritative_c2_args.yaml")
    shutil.copyfile(data, folder / "data_config.yaml")
    (folder / "pip_freeze.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(folder / "source_snapshot.tar.gz"), "HEAD",
                    "tools", "ultralytics-main/ultralytics", "ultralytics-main/tests", "ultralytics-main/pyproject.toml",
                    "configs", "docs/c26", "docs/scca/c2_args.yaml", ".gitattributes"], cwd=ROOT, check=True)
    (folder / "source_from_c24.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", C24_COMMIT, "HEAD"], cwd=ROOT))
    write_json(folder / "source_record.json", dict(runtime=runtime(), snapshot_sha256=sha256(folder / "source_snapshot.tar.gz"),
               data_sha256=sha256(data), c2_args_sha256=sha256(c2_args), full_server_preflight="NOT_RUN", cbr_box_diagnostics="NOT_RUN"))


def optimizer_groups(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [dict(index=i, weight_decay=g.get("weight_decay"), lr=g.get("lr"),
                 parameters=[names[id(p)] for p in g["params"]]) for i, g in enumerate(optimizer.param_groups)]


def paths(variant):
    name = VARIANTS[variant][2]
    return dict(name=name, run=MAIN / "runs/c_series" / name, launch=ROOT / "outputs/c26",
                init=ROOT / "weights" / "c26_cbr_scca_controlled_init.pt",
                source=MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                c2_args=MAIN / "runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml")


def recipe(c2_path, variant, init):
    source, expected = YAML.load(c2_path), YAML.load(ROOT / "docs/scca/c2_args.yaml")
    require(set(source) == set(expected), "C2 recipe fields changed/missing; inspect actual args")
    require(all(type(source[k]) is type(expected[k]) and source[k] == expected[k] for k in source),
            "C2 actual recipe differs from verified 109-field archive")
    target = dict(source)
    target.update(model=str(Path(init).resolve()), name=VARIANTS[variant][2],
                  save_dir=str(Path(source["project"]) / VARIANTS[variant][2]))
    rows = [{"field": k, "C2": source[k], variant: target[k], "changed": source[k] != target[k]} for k in sorted(source)]
    require({r["field"] for r in rows if r["changed"]} == {"model", "name", "save_dir"}, "Unexpected recipe difference")
    return target, rows


def rebuild_audit(source, target, variant):
    verify_model(target, variant, zero=True)
    before, after = source.state_dict(), target.state_dict()
    incompatible = {k for k in before if k in after and before[k].shape != after[k].shape}
    head = len(target.model) - 1
    expected = {f"model.{head}.denoising_class_embed.weight", f"model.{head}.enc_score_head.weight",
                f"model.{head}.enc_score_head.bias"}
    expected |= {f"model.{head}.dec_score_head.{i}.{suffix}" for i in range(3) for suffix in ("weight", "bias")}
    require(not set(before) ^ set(after), "Train API changed state keys")
    require(incompatible == (expected if source.model[-1].nc != target.model[-1].nc else set()), "Unexpected nc rebuild skips")
    equal = [k for k in before if k not in incompatible and torch.equal(before[k].cpu(), after[k].cpu())]
    require(len(equal) + len(incompatible) == len(before), "Train API failed to load public/new state values")
    return dict(nc_source=source.model[-1].nc, nc_target=target.model[-1].nc, loaded_exact=len(equal),
                missing=[], unexpected=[], classification_shape_skips=sorted(incompatible))


def disable_oom_retry(trainer):
    # Native trainer checks this counter before reducing batch; set it before every batch.
    trainer._oom_retries = 3


def ensure_amp_resources(main, report):
    """Use trusted local copies first; only official Ultralytics URLs as fallback."""
    records = []
    for name, dest, url, candidates in (
        ("bus.jpg", ASSETS / "bus.jpg", "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
         [main / "ultralytics-main/ultralytics/assets/bus.jpg", main / "bus.jpg"]),
        ("yolo26n.pt", ROOT / "yolo26n.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
         [main / "yolo26n.pt", main / "weights/yolo26n.pt"]),
    ):
        origin = "existing worktree copy"
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            found = next((p for p in candidates if p.is_file()), None)
            if found:
                shutil.copyfile(found, dest)
                origin = str(found)
            else:
                # Exclusive write: never replace another process's completed resource.
                with urllib.request.urlopen(url, timeout=60) as response, dest.open("xb") as f:
                    shutil.copyfileobj(response, f)
                origin = url
        records.append(dict(name=name, path=str(dest), source=origin, sha256=sha256(dest)))
    write_json(report, records)


def start_direct(variant):
    p = paths(variant)
    require(shutil.which("tmux"), "tmux required for SSH-independent training")
    require(torch.cuda.is_available(), "AutoDL CUDA unavailable")
    require(os.environ.get("CONDA_DEFAULT_ENV") == "rtdetr", "Activate existing rtdetr environment")
    info = runtime()
    verify_server_environment(info)
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Protect uncommitted source: commit/review changes before launch")
    require(not p["run"].exists(), f"Existing results protected: {p['run']}")
    session = "c26-training"
    require(subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0,
            f"Session already exists: {session}")
    # Atomic, persistent reservation shared by all worktrees; preserve failure evidence too.
    lock = p["run"].with_name(p["run"].name + ".c26.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir(exist_ok=False)
    write_json(lock / "owner.json", dict(pid=os.getpid(), worktree=str(ROOT), variant=variant, runtime=info))
    p["launch"].mkdir(parents=True, exist_ok=False)
    write_json(p["launch"] / "launch_state.json", dict(status="initializing", full_server_preflight="NOT_RUN"))
    try:
        require(p["c2_args"].is_file() and p["source"].is_file(), "C2 args/source missing")
        args, rows = recipe(p["c2_args"], variant, p["init"])
        require(Path(args["save_dir"]) == p["run"] and Path(args["data"]).is_file(), "Formal paths differ/missing")
        data = check_det_dataset(args["data"], autodownload=False)
        require(data["nc"] == 1, "Expected nc=1")
        for split in ("train", "val", "test"):
            locations = data.get(split)
            require(locations, f"Missing data split {split}")
            for location in locations if isinstance(locations, list) else [locations]:
                require(Path(location).exists(), f"Missing {split} path: {location}")
        record_source(p["launch"], p["c2_args"], Path(args["data"]))
        ensure_amp_resources(MAIN, p["launch"] / "amp_resources.json")
        write_json(p["launch"] / "initialization.json", initialize(p["source"], p["init"], variant))
        YAML.save(p["launch"] / "train_args.yaml", args)
        write_json(p["launch"] / "parameter_diff.json", rows)
        plan = dict(variant=variant, runtime=info, args=args, c2_args_sha256=sha256(p["c2_args"]),
                    init_sha256=sha256(p["init"]), run=str(p["run"]), session=session,
                    cbr_box_diagnostics="NOT_RUN", full_server_preflight="NOT_RUN", full_server_preflight_note="完整服务器预检未执行",
                    mode="start-direct", created=datetime.now(timezone.utc).isoformat())
        write_json(p["launch"] / "plan.json", plan)
        worker = p["launch"] / "worker.sh"
        console = p["launch"] / "console.log"
        command = [sys.executable, "-u", str(ROOT / "tools/train_c26.py"), "worker", variant]
        # An existing tmux server may have another experiment's environment.
        env = {"PYTHONPATH": str(ROOT / "ultralytics-main"), "YOLO_AUTOINSTALL": "false", "C26_MAIN": str(MAIN)}
        for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
            env[key] = os.environ.get(key)
        environment = "".join(("unset " + key if value is None else "export " + key + "=" + shlex.quote(value)) + "\n"
                              for key, value in env.items())
        # Worker shell records even a Python startup failure; no pipe hides the exit code.
        worker.write_text("#!/usr/bin/env bash\nset -uo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n" +
            environment + " ".join(map(shlex.quote, command)) + " >> " + shlex.quote(str(console)) + " 2>&1\nrc=$?\n" +
            "printf '%s\\n' \"$rc\" > " + shlex.quote(str(p["launch"] / "process_exit_code.txt")) + "\nexit \"$rc\"\n", encoding="utf-8")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash " + shlex.quote(str(worker))], check=True)
        write_json(p["launch"] / "launch_state.json", dict(status="dispatched", session=session, log=str(console), full_server_preflight="NOT_RUN"))
        print(f"Started {variant}: tmux {session}; log {console}; 完整服务器预检未执行")
    except BaseException as error:
        write_json(p["launch"] / "launch_state.json", dict(status="failed", error=repr(error), full_server_preflight="NOT_RUN"))
        raise


def worker(variant):
    p = paths(variant)
    code = 1
    try:
        plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
        require(runtime()["commit"] == plan["runtime"]["commit"], "Worktree commit changed since dispatch")
        require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Worktree changed since dispatch")
        require(sha256(p["init"]) == plan["init_sha256"], "Initialization changed")
        require(not p["run"].exists(), "Results appeared after reservation; refusing overwrite")
        write_json(p["launch"] / "process.json", dict(pid=os.getpid(), session=plan["session"], started=datetime.now(timezone.utc).isoformat()))

        class RecordingTrainer(RTDETRTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model = super().get_model(cfg, weights, verbose)
                write_json(p["launch"] / "nc1_loading.json", rebuild_audit(weights, model, variant))
                return model

        model = RTDETR(str(p["init"]))
        model.add_callback("on_train_batch_start", disable_oom_retry)

        def setup(trainer):
            verify_model(trainer.model, variant, zero=True)
            require(bool(trainer.amp), "Native AMP check disabled AMP; stopping rather than changing formal recipe")
            actual = vars(trainer.args)
            differences = {k: [v, actual.get(k)] for k, v in plan["args"].items() if type(actual.get(k)) is not type(v) or actual.get(k) != v}
            require(not differences, f"Actual training recipe changed: {differences}")
            YAML.save(p["launch"] / "actual_train_args.yaml", actual)
            ids = [id(v) for g in trainer.optimizer.param_groups for v in g["params"]]
            added = {n: ids.count(id(v)) for n, v in trainer.model.named_parameters() if is_added(n)}
            require(all(count == 1 for count in added.values()), "New parameters absent/duplicated in optimizer")
            write_json(p["launch"] / "training_setup.json", dict(amp=bool(trainer.amp), parameters=sum(v.numel() for v in trainer.model.parameters()),
                       optimizer=type(trainer.optimizer).__name__, added_optimizer_occurrences=added, optimizer_groups=optimizer_groups(trainer.model, trainer.optimizer), recipe_differences=differences,
                       batch=trainer.args.batch, full_server_preflight="NOT_RUN"))

        model.add_callback("on_train_start", setup)
        model.train(trainer=RecordingTrainer, **plan["args"])
        code = 0
    except KeyboardInterrupt:
        code = 130
        raise
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        write_json(p["launch"] / "exit_code.json", dict(exit_code=code, finished=datetime.now(timezone.utc).isoformat()))


def status(variant):
    p = paths(variant)
    print(f"Experiment: {variant}\nRun: {p['run']}\nLog: {p['launch'] / 'console.log'}")
    if shutil.which("tmux"):
        active = subprocess.run(["tmux", "has-session", "-t", "c26-training"], capture_output=True).returncode == 0
        print("tmux active:", active)
    for name in ("launch_state.json", "process.json", "exit_code.json", "process_exit_code.txt"):
        path = p["launch"] / name
        print(name + ": " + (path.read_text(encoding="utf-8") if path.exists() else "not recorded"))
    log = p["launch"] / "console.log"
    if log.exists():
        # Bounded tail, even for a long formal run.
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 8000))
            print(f.read().decode("utf-8", errors="replace"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start-direct", "worker", "status"))
    parser.add_argument("variant", type=str.lower, choices=VARIANTS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"start-direct": start_direct, "worker": worker, "status": status}[args.mode](args.variant)
