"""Pinned final-CBR FP32 evaluation, immutable val/test identity and LIGHT evidence archive."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from bmc_v1_common import (ROOT, OUT, MAIN, RUN, INIT, BASE, BRANCH, NAME, require, read, write, sha, git,
                           runtime, fingerprint, source_files, verify_delivery, utc)


def verify_selected_model(model, expected_name=NAME):
    from ultralytics.models.utils.bmc import BMCDetectionModel
    require(isinstance(model.model, BMCDetectionModel), "Weights are not from the BMC training model")
    # RTDETR._load deliberately reduces model.args to inference fields; identity lives in the checkpoint.
    require(model.ckpt["train_args"]["name"] == expected_name and model.model.model[-1].nc == 1,
            "Wrong experiment weight identity")


def evaluate(split):
    import numpy as np
    import torch
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.models.utils.bmc import BMCDetectionModel
    from c19_lif_v1_results import postprocess, image_record, POLICY, EVAL
    from bmc_v1 import processes, session_active
    require(split in ("val", "test"), "Explicit val/test only")
    verify_delivery()
    require(not processes() and not session_active(), "Wait until this experiment completes before locking evaluation weights")
    identity = fingerprint()
    prepared = read(OUT / "prepare.json")
    lock = OUT / "val_lock.json"
    weights = RUN / "weights/best.pt"
    require(weights.is_file(), "This experiment has no selected best.pt")
    weight_sha = sha(weights)
    selection = dict(checkpoint=str(weights), checkpoint_sha256=weight_sha, fingerprint=identity["sha256"],
                     policy=POLICY, source_sha=git("rev-parse", "HEAD"), settings=EVAL)
    if lock.exists():
        require(read(lock) == selection, "Validated selection is locked; changed weights/source/data cannot silently replace it")
    if split == "test":
        require(lock.is_file(), "Explicit test requires completed val and its locked weight hash")
        prior_val = read(OUT / "evaluation/val/metrics.json")
        require(prior_val["status"] == "COMPLETED" and prior_val["selection"] == selection,
                "Test requires completed val metrics for the locked identity")
    target = OUT / "evaluation" / split
    metric_path = target / "metrics.json"
    if metric_path.exists():
        old = read(metric_path)
        require(old.get("status") == "COMPLETED" and old.get("selection") == selection,
                "Previous evaluation incomplete/different; preserved for inspection")
        return dict(old, reused=True)
    require(not target.exists(), "Existing incomplete evaluation directory preserved")
    target.mkdir(parents=True)
    report = dict(status="FAIL", split=split, selection=selection, runtime=runtime(),
                  policy=POLICY, settings=EVAL, precision_recall_rule="Native ap_per_class: precision/recall at the maximum smoothed mean F1 point; conf=.001 is the prediction filter")
    counts = dict(images=0, ground_truth=0, predictions=0)
    try:
        model = RTDETR(str(weights))
        verify_selected_model(model)
        stream_path = target / "predictions_gt.jsonl.gz"
        with gzip.open(stream_path, "wt", encoding="utf-8") as stream:
            class CorrectedValidator(RTDETRValidator):
                def init_metrics(self, actual):
                    super().init_metrics(actual)
                    require(not self.training and not self.args.half, "Independent FP32 only")
                    require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()), "Evaluation config changed")
                    self.expected_files = {str(Path(p).resolve()) for p in self.dataloader.dataset.im_files}
                    self.observed_files = set()
                    require(len(self.expected_files) == prepared["data"]["inventory"][split]["images"], "Split count changed/empty")
                    report["actual_settings"] = vars(self.args).copy()

                def postprocess(self, predictions):
                    raw = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
                    require(torch.isfinite(raw).all(), "Nonfinite final CBR prediction")
                    # Use the mother's verified implementation, not a policy label on a different mask.
                    selected, _ = postprocess(predictions, self.args.imgsz, self.args.conf)
                    complete, _ = postprocess(predictions, self.args.imgsz, -float("inf"))
                    for a, b in zip(selected, complete):
                        a["_complete"] = b
                    return selected

                def update_metrics(self, predictions, batch):
                    for i, prediction in enumerate(predictions):
                        path = str(Path(batch["im_file"][i]).resolve())
                        require(path not in self.observed_files, "Duplicate evaluated image")
                        self.observed_files.add(path)
                        row = image_record(prediction.pop("_complete"), self._prepare_batch(i, batch), self.data["path"], self.args.conf)
                        counts["images"] += 1
                        counts["ground_truth"] += len(row["ground_truth"])
                        counts["predictions"] += len(row["predictions"])
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    return super().update_metrics(predictions, batch)

                def finalize_metrics(self):
                    require(self.observed_files == self.expected_files and counts["ground_truth"] > 0, "Incomplete/empty split")
                    return super().finalize_metrics()

            metrics = model.val(validator=CorrectedValidator, data=prepared["data"]["config"], split=split,
                device="0", project=str(target), name="plots", exist_ok=False, plots=True, save_json=False, **EVAL)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid AP array")
        require(sha(weights) == weight_sha, "Selected weights changed during evaluation")
        report.update(status="COMPLETED", mAP50_95=float(metrics.box.map), AP50=float(metrics.box.map50),
                      AP75=float(ap[:, 5].mean()), precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      ap_by_class=ap.tolist(), predictions_sha256=sha(stream_path), **counts)
        if split == "val":
            write(lock, selection)
    except BaseException as error:
        report.update(error=repr(error), **counts)
        raise
    finally:
        write(metric_path, report)
    return report


def validation_summary():
    checks = {}
    for name in ("checks_cpu", "checks_cuda_0", "lifecycle_cpu", "lifecycle_0", "operations"):
        path = OUT / (name + ".json")
        checks[name] = dict(status=read(path).get("status"), report=str(path), sha256=sha(path)) if path.is_file() else dict(status="PENDING")
    capacity = read(OUT / "preflight.json").get("status") if (OUT / "preflight.json").exists() else "PENDING"
    result = dict(source_sha=git("rev-parse", "HEAD"), base=BASE, branch=BRANCH, runtime=runtime(),
                  local_checks=checks, server_engineering_preflight=capacity,
                  probe=read(OUT / "probe.json").get("status") if (OUT / "probe.json").exists() else "PENDING",
                  formal_training="NOT_RUN" if not (RUN / "results.csv").exists() else "SEE_RESULTS_CSV",
                  final_test=read(OUT / "evaluation/test/metrics.json")["status"] if (OUT / "evaluation/test/metrics.json").exists() else "NOT_RUN")
    write(OUT / "validation.json", result)
    return result


def delivery_documents(full_sha):
    validation_summary()
    server = "/root/autodl-tmp/projects/Crack_RTDETR-bmc_v1"
    python = "/root/miniconda3/envs/rtdetr/bin/python"
    prefix = f"env PYTHONPATH={server}/ultralytics-main:{server}/tools PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false {python} {server}/tools/bmc_v1.py"
    sync = f"""set -euo pipefail
curl --fail --location https://raw.githubusercontent.com/supershaojie/concrete-crack-rtdetr/{full_sha}/tools/sync_bmc_v1.sh -o /tmp/sync_bmc_v1_{full_sha}.sh
BMC_PYTHON={python} bash /tmp/sync_bmc_v1_{full_sha}.sh {full_sha}"""
    sections = [f"# BMC v1 服务器命令\n\n交付 SHA：`{full_sha}`。每块独立执行；正式训练和 test 由用户显式启动。\n\n同步（保护主工作区）：\n\n```bash\n{sync}\n```\n"]
    for action in ("prepare", "preflight", "probe", "start", "status", "resume", "val", "test", "pack"):
        sections.append(f"\n{action}：\n\n```bash\n{prefix} {action}\n```\n")
    sections += ["\n查看 tmux：\n\n```bash\ntmux attach-session -t bmc-v1-training\n```\n\n脱离：先 Ctrl+B，再按 D。\n",
                 f"\n仅在无 checkpoint、无 results 数据行且无活跃进程的启动失败目录上归档后重启：\n\n```bash\n{prefix} start --archive-failed\n```\n",
                 f"\n日志与证据：`{server}/outputs/bmc_v1/`。LIGHT 归档绝对路径写入 `LATEST_LIGHT.txt`。\n"]
    (OUT / "server_commands.md").write_text("".join(sections), encoding="utf-8")
    (OUT / "DELIVERY.md").write_text(f"""# BMC v1 交付

母版：{BASE}
分支：{BRANCH}
交付：{full_sha}

仅训练 epoch>=20 的普通 decoder 第2层（loss slot 2）匹配可变。
固定预算 0.02*max(IQR,0.001)，按连通分量整体替换；原 LIF/CBR 文件不变，新增参数 0。
完整验证状态见 validation.json；初始化逐key证据见 trainer_initialization.json。
本地小测不能代替服务器 B16/640 preflight。训练/test 均须用户显式执行。
完整独立可复制命令见 server_commands.md。tmux：bmc-v1-training；Ctrl+B 后按 D 脱离。
LIGHT 默认不含数据集和 .pt，保留机制 epoch 汇总和详细事件。无 test 时记 NOT_RUN。
""", encoding="utf-8")


def package():
    OUT.mkdir(parents=True, exist_ok=True)
    full_sha = git("rev-parse", "HEAD")
    delivery_documents(full_sha)
    (OUT / "source_from_mother.patch").write_text(git("diff", BASE, "HEAD"), encoding="utf-8")
    (OUT / "working_diff.patch").write_text(git("diff", "HEAD"), encoding="utf-8")
    (OUT / "git_status.txt").write_text(git("status", "--short"), encoding="utf-8")
    staging = OUT / "light_metadata"
    staging.mkdir(exist_ok=True)
    logs = []
    for log in sorted(OUT.glob("console_*.log")):
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size-32768))
            tail = stream.read()
        name = staging / (log.stem + "_tail.txt")
        name.write_bytes(tail)
        logs.append(dict(path=str(log), bytes=log.stat().st_size, sha256=sha(log), tail=name.name))
    write(staging / "full_log_inventory.json", logs)
    weights = [p for p in (INIT, RUN / "weights/best.pt", RUN / "weights/last.pt") if p.is_file()]
    write(staging / "weights_manifest.json", [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p), included=False) for p in weights])
    write(staging / "source_identity.json", dict(sha=full_sha, base=BASE, branch=BRANCH, files=source_files()))
    entries = {}
    for name in source_files():
        entries["source/" + name] = ROOT / name
    for p in (ROOT / "docs/bmc_v1").rglob("*"):
        if p.is_file():
            entries["source/" + p.relative_to(ROOT).as_posix()] = p
    # Include mechanism data, bounded events, reports and metrics; leave all full logs/weights intact.
    for p in OUT.rglob("*"):
        if not p.is_file() or p.suffix not in {".json", ".jsonl", ".md", ".yaml", ".csv", ".txt", ".patch"}:
            continue
        if p.name in {"LATEST_LIGHT.txt"} or any(x.startswith(("capacity_", "lifecycle_")) for x in p.relative_to(OUT).parts[:-1]):
            continue
        entries["evidence/" + p.relative_to(OUT).as_posix()] = p
    for name in ("results.csv", "args.yaml"):
        if (RUN / name).is_file():
            entries["training/" + name] = RUN / name
    manifest = dict(created=utc(), source_sha=full_sha, test=validation_summary()["final_test"],
                    exclusions="datasets, large .pt, full console logs, full prediction streams", files=[])
    for name, path in sorted(entries.items()):
        manifest["files"].append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
    destination = OUT / ("BMC_v1_LIGHT_" + utc() + ".tar.gz")
    with tarfile.open(destination, "w:gz") as archive:
        for name, path in sorted(entries.items()):
            archive.add(path, arcname=name, recursive=False)
        raw = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(destination, "r:gz") as archive:
        require(len(archive.getnames()) == len(entries)+1, "Archive inventory differs")
        for row in manifest["files"]:
            raw = archive.extractfile(row["path"]).read()
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "Archive member corruption")
    (OUT / "LATEST_LIGHT.txt").write_text(str(destination.resolve())+"\n", encoding="utf-8")
    result = dict(status="PASS", archive=str(destination.resolve()), bytes=destination.stat().st_size,
                  sha256=sha(destination), readable_members=len(entries)+1, test=manifest["test"])
    write(OUT / "pack.json", result)
    return result
