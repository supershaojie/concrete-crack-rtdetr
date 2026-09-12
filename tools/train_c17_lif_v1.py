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
from init_c17_lif_v1 import ROOT, MODEL_DIR, VARIANTS, initialize, require, runtime, sha256, verify_model, write_json, is_added, BASE_COMMIT, build_training_model
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML, ASSETS
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import RTDETRDetectionModel

MAIN = Path(os.environ.get("C17_LIF_V1_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))


def verify_server_environment(info):
    require(torch.cuda.is_available(), "CUDA required for formal AutoDL training")
    require(sys.version.split()[0]=='3.10.13' and info['torch']=='2.1.2+cu121' and info['cuda']=='12.1',
            'Environment differs from successful AutoDL Python3.10.13/torch2.1.2+cu121; report and review, no installation')
    require('4090' in info['gpu'],'Expected the existing single RTX4090; report changed hardware')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Use the existing physical GPU0; do not select another GPU automatically')


def bounded_preflight(p, data):
    from c17_lif_v1_data import manifest, verify_manifest
    actual=manifest(data['path'])
    expected=json.loads((ROOT/'docs/c17_lif_v1/dataset_manifest.json').read_text(encoding='utf8'))
    verify_manifest(actual,expected);write_json(p['launch']/'dataset_manifest.json',actual)
    command=[sys.executable,'-u',str(ROOT/'tools/check_c17_lif_v1.py'),'--source',str(p['source']),
             '--real-dataset',str(data['path']),'--output',str(p['launch']/'preflight'),
             '--capacity','--data-config',str(p['launch']/'data_config.yaml')]
    with (p['launch']/'preflight.log').open('x',encoding='utf8') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((p['launch']/'preflight/checks.json').read_text(encoding='utf8'))
    require(report['status']=='PASSED' and isinstance(report['server_B16_640'],dict),'Finite B16/640 AMP preflight incomplete')
    require(report['server_B16_640']['input']==[16,3,640,640],'Wrong capacity batch/shape')
    write_json(p['launch']/'preflight.json',report)
    source_record=json.loads((p['launch']/'source_record.json').read_text(encoding='utf8'))
    source_record['full_server_preflight']='PASSED';write_json(p['launch']/'source_record.json',source_record)


def record_source(folder, c2_args, data):
    """Freeze reproducibility evidence before dispatch, without model inference."""
    shutil.copyfile(c2_args, folder / "authoritative_c2_args.yaml")
    shutil.copyfile(data, folder / "data_config.yaml")
    (folder / "pip_freeze.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(folder / "source_snapshot.tar.gz"), "HEAD",
                    "tools", "ultralytics-main/ultralytics", "ultralytics-main/tests", "ultralytics-main/pyproject.toml",
                    "configs", "docs/c17_lif_v1", "docs/c17_lif_v1/c2_args.yaml", ".gitattributes"], cwd=ROOT, check=True)
    (folder / "source_from_base.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", BASE_COMMIT, "HEAD"], cwd=ROOT))
    write_json(folder / "source_record.json", dict(runtime=runtime(), snapshot_sha256=sha256(folder / "source_snapshot.tar.gz"),
               data_sha256=sha256(data), c2_args_sha256=sha256(c2_args), full_server_preflight="PENDING"))


def optimizer_groups(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [dict(index=i, weight_decay=g.get("weight_decay"), lr=g.get("lr"),
                 parameters=[names[id(p)] for p in g["params"]]) for i, g in enumerate(optimizer.param_groups)]


def paths(variant="c17_lif_v1"):
    name = VARIANTS[variant][2]
    return dict(name=name, run=MAIN / "runs/c_series" / name, launch=ROOT / "outputs/c17_lif_v1",
                init=ROOT / "weights" / "c17_lif_v1_controlled_init.pt",
                source=MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                c2_args=MAIN / "runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml")


def recipe(c2_path, variant, init):
    source, expected = YAML.load(c2_path), YAML.load(ROOT / "docs/c17_lif_v1/c2_args.yaml")
    require(set(source) == set(expected), "C2 recipe fields changed/missing; inspect actual args")
    require(all(type(source[k]) is type(expected[k]) and source[k] == expected[k] for k in source),
            "C2 actual recipe differs from verified 109-field archive")
    target = dict(source)
    target.update(model=str(Path(init).resolve()), name=VARIANTS[variant][2],
                  project=str(MAIN / "runs/c_series"), data=str(MAIN / "configs/crack_autodl.yaml"),
                  save_dir=str(MAIN / "runs/c_series" / VARIANTS[variant][2]))
    rows = [{"field": k, "C2": source[k], variant: target[k], "changed": source[k] != target[k]} for k in sorted(source)]
    changed = {r["field"] for r in rows if r["changed"]}
    require({"model", "name", "save_dir"} <= changed <= {"model", "name", "save_dir", "project", "data"}, "Unexpected recipe difference")
    for row in rows:
        if row["changed"]:
            row["reason"] = "experiment model/output identity" if row["field"] in {"model", "name", "save_dir"} else "explicit C17_LIF_V1_MAIN root relocation; same data/recipe required"
    return target, rows


def verify_data_config(path):
    source, expected = YAML.load(path), YAML.load(ROOT / "docs/c17_lif_v1/c2_data.yaml")
    expected["path"] = str((MAIN / "datasets/crack_det").resolve())
    actual = dict(source)
    actual["path"] = str(Path(actual["path"]).resolve())
    require(actual == expected, "C2 data path/splits/classes changed")


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


def verify_delivery():
    """Trust the sync record plus commit/worktree identity, including detached HEAD."""
    record = ROOT / "outputs/c17_lif_v1_delivery.json"
    require(record.is_file(), "Run sync_c17_lif_v1.sh with the delivered full SHA first")
    info = json.loads(record.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(info["commit"] == head and info["branch"] == "codex/rtdetr-c17-lif-v1", "Pinned delivery changed")
    require(Path(info["worktree"]).resolve() == ROOT.resolve() and Path(info["main"]).resolve() == MAIN.resolve(), "Delivery paths changed")
    require((ROOT / ".git").is_file(), "Expected a linked worktree")
    commons = [subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=p, text=True).strip() for p in (ROOT, MAIN)]
    require(Path(commons[0]).resolve() == Path(commons[1]).resolve(), "Worktree belongs to another repository")
    for path, digest in info["files"].items():
        require(sha256(ROOT / path) == digest, "Delivery file changed: " + path)
    return info


def duplicate_processes():
    found = []
    for item in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = item.read_bytes().split(b"\0")
            if int(item.parent.name) != os.getpid() and any(v.endswith(b"train_c17_lif_v1.py") for v in argv) and b"worker" in argv:
                found.append(int(item.parent.name))
        except (OSError, ValueError):
            continue
    return found


def start_direct(variant):
    p = paths(variant)
    delivery = verify_delivery()
    require(not duplicate_processes(), "Existing C17 + LIF v1 worker process protected")
    require(not p["launch"].exists() and not p["init"].exists(), "Existing C17 + LIF v1 launch/init protected")
    require(ROOT.resolve() != MAIN.resolve(), "Use the independent C17 + LIF v1 worktree, not the main checkout")
    require(shutil.which("tmux"), "tmux required for SSH-independent training")
    require(torch.cuda.is_available(), "AutoDL CUDA unavailable")
    require(os.environ.get("CONDA_DEFAULT_ENV") == "rtdetr", "Activate existing rtdetr environment")
    info = runtime()
    verify_server_environment(info)
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(),
            "Protect uncommitted source: commit/review changes before launch")
    require(not p["run"].exists(), f"Existing results protected: {p['run']}")
    session = "c17_lif_v1-training"
    require(subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0,
            f"Session already exists: {session}")
    # Atomic, persistent reservation shared by all worktrees; preserve failure evidence too.
    lock = p["run"].with_name(p["run"].name + ".c17_lif_v1.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir(exist_ok=False)
    write_json(lock / "owner.json", dict(pid=os.getpid(), worktree=str(ROOT), variant=variant, runtime=info))
    p["launch"].mkdir(parents=True, exist_ok=False)
    write_json(p["launch"] / "launch_state.json", dict(status="initializing",pid=os.getpid(),process_token=process_token(os.getpid()),full_server_preflight="PENDING"))
    try:
        require(p["c2_args"].is_file() and p["source"].is_file(), "C2 args/source missing")
        args, rows = recipe(p["c2_args"], variant, p["init"])
        require(Path(args["save_dir"]) == p["run"] and Path(args["data"]).is_file(), "Formal paths differ/missing")
        verify_data_config(args["data"])
        data = check_det_dataset(args["data"], autodownload=False)
        require(data["nc"] == 1, "Expected nc=1")
        for split in ("train", "val", "test"):
            locations = data.get(split)
            require(locations, f"Missing data split {split}")
            for location in locations if isinstance(locations, list) else [locations]:
                require(Path(location).exists(), f"Missing {split} path: {location}")
        record_source(p["launch"], p["c2_args"], Path(args["data"]))
        ensure_amp_resources(MAIN, p["launch"] / "amp_resources.json")
        print(json.dumps(dict(runtime=info,model_yaml=str(MODEL_DIR/VARIANTS[variant][0]),recipe_sha256=sha256(p['c2_args']),data_sha256=sha256(args['data']),source_sha256=sha256(p['source'])),indent=2),flush=True)
        bounded_preflight(p,data)
        write_json(p["launch"] / "initialization.json", initialize(p["source"], p["init"], variant))
        YAML.save(p["launch"] / "train_args.yaml", args)
        write_json(p["launch"] / "parameter_diff.json", rows)
        plan = dict(variant=variant, runtime=info, args=args, c2_args_sha256=sha256(p["c2_args"]),
                    init_sha256=sha256(p["init"]), run=str(p["run"]), session=session,
                    full_server_preflight="PASSED", full_server_preflight_note="Finite regression and native B16/640 AMP loss capacity passed; not full training",
                    mode="start-direct", delivery=delivery, cwd=str(ROOT), resolved_config_sha256=sha256(p["launch"] / "train_args.yaml"), created=datetime.now(timezone.utc).isoformat())
        write_json(p["launch"] / "plan.json", plan)
        worker = p["launch"] / "worker.sh"
        console = p["launch"] / "console.log"
        command = [sys.executable, "-u", str(ROOT / "tools/train_c17_lif_v1.py"), "worker"]
        plan["command"] = command
        write_json(p["launch"] / "plan.json", plan)
        # An existing tmux server may have another experiment's environment.
        env = {"PYTHONPATH": str(ROOT / "ultralytics-main"), "YOLO_AUTOINSTALL": "false", "C17_LIF_V1_MAIN": str(MAIN), "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV")}
        for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
            env[key] = os.environ.get(key)
        environment = "".join(("unset " + key if value is None else "export " + key + "=" + shlex.quote(value)) + "\n"
                              for key, value in env.items())
        environment += 'source /root/miniconda3/etc/profile.d/conda.sh || exit $?\nconda activate rtdetr || exit $?\nexport PYTHONUNBUFFERED=1\n'
        # Worker shell records even a Python startup failure; no pipe hides the exit code.
        worker.write_text("#!/usr/bin/env bash\nset -uo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n" +
            environment + " ".join(map(shlex.quote, command)) + " >> " + shlex.quote(str(console)) + " 2>&1\nrc=$?\n" +
            "printf '%s\\n' \"$rc\" > " + shlex.quote(str(p["launch"] / "process_exit_code.txt")) + "\nexit \"$rc\"\n", encoding="utf-8")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash " + shlex.quote(str(worker))], check=True)
        write_json(p["launch"] / "launch_state.json", dict(status="dispatched", session=session, log=str(console), full_server_preflight="PASSED"))
        print(f"Started {variant}: tmux {session}; log {console}; bounded preflight passed")
    except BaseException as error:
        write_json(p["launch"] / "launch_state.json", dict(status="failed", error=repr(error), full_server_preflight="FAILED"))
        raise


def worker(variant):
    p = paths(variant)
    code = 1
    try:
        plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
        verify_delivery()
        require(sha256(p["launch"] / "train_args.yaml") == plan["resolved_config_sha256"], "Resolved config changed")
        require(runtime()["commit"] == plan["runtime"]["commit"], "Worktree commit changed since dispatch")
        require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(), "Worktree changed since dispatch")
        require(sha256(p["init"]) == plan["init_sha256"], "Initialization changed")
        source_record = json.loads((p["launch"] / "source_record.json").read_text(encoding="utf-8"))
        require(sha256(plan["args"]["data"]) == source_record["data_sha256"] and
                sha256(p["c2_args"]) == plan["c2_args_sha256"], "Data config/C2 recipe changed since dispatch")
        require(not p["run"].exists(), "Results appeared after reservation; refusing overwrite")
        write_json(p["launch"] / "process.json", dict(pid=os.getpid(), process_token=process_token(os.getpid()),
                   session=plan["session"], started=datetime.now(timezone.utc).isoformat()))

        class RecordingTrainer(RTDETRTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model, audit = build_training_model(cfg, weights, self.data, variant)
                write_json(p["launch"] / "nc1_loading.json", audit)
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
            require(all(ids.count(id(v)) == 1 for v in trainer.model.parameters()), "Common/new parameters absent/duplicated in AdamW")
            require(type(trainer.optimizer) is torch.optim.AdamW, "Optimizer changed")
            require(all(count == 1 for count in added.values()), "New parameters absent/duplicated in optimizer")
            write_json(p["launch"] / "training_state.json", dict(status="training", pid=os.getpid()))
            write_json(p["launch"] / "training_setup.json", dict(amp=bool(trainer.amp), parameters=sum(v.numel() for v in trainer.model.parameters()),
                       optimizer=type(trainer.optimizer).__name__, added_optimizer_occurrences=added, optimizer_groups=optimizer_groups(trainer.model, trainer.optimizer), recipe_differences=differences,
                       batch=trainer.args.batch, full_server_preflight="PASSED"))

        model.add_callback("on_train_start", setup)
        warmup_rows=[]
        def record_warmup(trainer):
            if len(warmup_rows)<4:
                warmup_rows.append(dict(batch=len(warmup_rows),accumulate=trainer.accumulate,
                    groups=[dict(group=g['param_group'],lr=g['lr'],weight_decay=g['weight_decay']) for g in trainer.optimizer.param_groups],
                    parameter_norms={n:float(v.detach().float().norm()) for n,v in trainer.model.named_parameters() if is_added(n)}))
                write_json(p['launch']/'warmup_batches.json',warmup_rows)
        model.add_callback('on_train_batch_end',record_warmup)
        model.train(trainer=RecordingTrainer, **plan["args"])
        code = 0
    except KeyboardInterrupt:
        code = 130
        raise
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) and error.code != 0 else 1
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        write_json(p["launch"] / "exit_code.json", dict(exit_code=code, finished=datetime.now(timezone.utc).isoformat()))


def process_token(pid):
    """Linux PID start time avoids treating an unrelated reused PID as this worker."""
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


def run_state(p):
    def read(name):
        f = p["launch"] / name
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    launch, process, result = read("launch_state.json"), read("process.json"), read("exit_code.json")
    shell = p["launch"] / "process_exit_code.txt"
    shell_code = int(shell.read_text().strip()) if shell.is_file() else None
    code = result.get("exit_code")
    if (code is not None and code != 0) or (shell_code is not None and shell_code != 0) or launch.get("status") == "failed":
        return "FAILED"
    if code == shell_code == 0:
        required=('weights/best.pt','weights/last.pt','results.csv','args.yaml')
        return "SUCCESS" if all((p['run']/f).is_file() and (p['run']/f).stat().st_size>0 for f in required) else "FAILED"
    if launch.get('status')=='initializing':
        token=process_token(launch.get('pid',-1))
        return 'DISPATCHED' if token and token==launch.get('process_token') else 'FAILED'
    if process:
        token = process_token(process["pid"])
        return ("RUNNING" if (p["launch"] / "training_state.json").is_file() else "DISPATCHED") if token and token == process.get("process_token") else ("DISPATCHED" if code == 0 else "FAILED")
    if launch.get("status") == "dispatched" and shutil.which("tmux"):
        if subprocess.run(["tmux", "has-session", "-t", "c17_lif_v1-training"], capture_output=True).returncode:
            return "FAILED"
    return {"dispatched": "DISPATCHED", "initializing": "DISPATCHED"}.get(launch.get("status"), "NOT_STARTED")


def status(variant):
    p = paths(variant)
    print(f"Experiment: {variant}\nRun: {p['run']}\nLog: {p['launch'] / 'console.log'}")
    print("STATE:", run_state(p))
    if shutil.which("tmux"):
        active = subprocess.run(["tmux", "has-session", "-t", "c17_lif_v1-training"], capture_output=True).returncode == 0
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

    args = parser.parse_args()
    torch.set_num_threads(4)
    {"start-direct": start_direct, "worker": worker, "status": status}[args.mode]("c17_lif_v1")
