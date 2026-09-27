"""Mother corrected_sorted_conf_mask_v1 evaluation, exact checkpoint locks and failure-capable LIGHT export."""
from __future__ import annotations

import gzip
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import traceback

import numpy as np
import torch
from rmd_v1_common import (ROOT, OUT, BASE, require, now, unique_name, sha256, digest, read_json, write_json,
                           runtime, code_identity, source_snapshot, inventory, verify_pair, git)
from c19_lif_v1_results import postprocess, image_record
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import init_seeds

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300,
            augment=False, rect=False, seed=42)
EVAL_ONLY_FILES = {"ultralytics-main/ultralytics/nn/autobackend.py", "tools/rmd_v1_eval.py"}


class OneBatch:
    def __init__(self, loader):
        self.loader, self.dataset = loader, loader.dataset
    def __len__(self):
        return 1
    def __iter__(self):
        yield next(iter(self.loader))


def evaluation_identity(training, allow_revision=False):
    current = code_identity(clean=True)
    previous = training["prepared"]["code"]
    differences = sorted(k for k in current["lf_files"].keys() | previous["lf_files"].keys()
                         if current["lf_files"].get(k) != previous["lf_files"].get(k))
    require(not differences or (allow_revision and set(differences) <= EVAL_ONLY_FILES),
            "Evaluation revision changes training code or lacks --allow-eval-revision: " + repr(differences))
    return dict(training_commit=previous["commit"], evaluation_commit=current["commit"],
                differences=differences, allowed_eval_revision=bool(differences), code=current)


def evaluate_checkpoint(weights, data, output, device="0", split="val", one_batch=False, identity=None):
    """Actual AutoBackend/fuse/warmup then val data; no synthetic success substitution."""
    weights, output = Path(weights).resolve(), Path(output).resolve()
    require(weights.is_file() and not output.exists(), "Missing checkpoint or existing evaluation output")
    output.mkdir(parents=True, exist_ok=False)
    digest_before = sha256(weights)
    settings = dict(EVAL, data=str(data), device=device, split=split, plots=not one_batch,
                    project=str(output), name="plots", exist_ok=False, save_json=False, save_txt=False)
    report = dict(status="FAIL", started=now(), scope="one_real_val_batch" if one_batch else "full_split",
                  split=split, checkpoint=str(weights), checkpoint_sha256=digest_before, settings=settings,
                  policy=POLICY, identity=identity, runtime=runtime(), images=0, ground_truth=0,
                  predictions=0, mask_affected_images=0, training_selection="original val fitness: mAP50-95 only")
    stream_path = output/f"{split}_predictions_gt.jsonl.gz"
    seen = set()
    try:
        init_seeds(42, deterministic=True)
        model = RTDETR(str(weights))
        report["parameters_unfused"] = verify_pair(model.model)
        require(all(not m._forward_pre_hooks for m in model.model.modules()), "Checkpoint serialized a capture hook")
        with gzip.open(stream_path, "xt", encoding="utf-8") as stream:
            class EvidenceValidator(RTDETRValidator):
                def get_dataloader(self, dataset_path, batch_size):
                    loader = super().get_dataloader(dataset_path, batch_size)
                    return OneBatch(loader) if one_batch else loader

                def init_metrics(self, backend):
                    super().init_metrics(backend)
                    require(not self.training and not self.args.half, "Independent FP32 evaluation required")
                    for k, v in EVAL.items():
                        require(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v, "Eval protocol changed: " + k)
                    report["actual_settings"] = vars(self.args).copy()
                    report["autobackend"] = dict(device=str(self.device), fp16=backend.fp16,
                                                warmup="zeros/actual CUDA path" if self.device.type != "cpu" else "native CPU warmup skipped",
                                                lif_bn_preserved=hasattr(backend.model.model[20], "bn"))
                    require(report["autobackend"]["lif_bn_preserved"], "Original LIF fuse exclusion lost")
                    report["native_speed_denominator_images"] = len(self.dataloader.dataset)
                    if one_batch:
                        baseline = deepcopy(backend.model)
                        baseline.__class__ = RTDETRDetectionModel
                        if hasattr(baseline, "criterion"):
                            del baseline.criterion
                        self.mother = AutoBackend(model=baseline, device=self.device, fp16=False, fuse=True)
                        self.mother.eval()
                        self.mother.warmup(imgsz=(1, 3, 640, 640))
                        report["mother_control"] = dict(same_zeros_warmup_fix=True, real_batch_max_abs=0.,
                            state_keys_equal=set(self.mother.model.state_dict()) == set(backend.model.state_dict()))
                        require(report["mother_control"]["state_keys_equal"], "Mother deployment state keys differ")

                def postprocess(self, preds):
                    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
                    require(bool(torch.isfinite(raw).all()), "Nonfinite raw evaluation output")
                    selected, affected = postprocess(preds, self.args.imgsz, self.args.conf)
                    full, _ = postprocess(preds, self.args.imgsz, -float("inf"))
                    report["mask_affected_images"] += affected
                    for chosen, all_queries, raw_image in zip(selected, full, raw):
                        require(len(all_queries["conf"]) == 300, "Expected all 300 regular queries")
                        all_queries["query_indices"] = raw_image[:, 4:].max(-1).values.argsort(descending=True).cpu().tolist()
                        chosen["_raw_queries"] = all_queries
                        if one_batch:
                            chosen["_raw_tensor"] = raw_image.detach()
                    return selected

                def update_metrics(self, preds, batch):
                    if one_batch:
                        mother = self.mother(batch["img"])
                        mother = mother[0] if isinstance(mother, (tuple, list)) else mother
                        for i, pred in enumerate(preds):
                            actual = pred.pop("_raw_tensor")
                            error = float((mother[i]-actual).abs().max())
                            report["mother_control"]["real_batch_max_abs"] = max(report["mother_control"]["real_batch_max_abs"], error)
                            require(torch.equal(mother[i], actual), "Original mother/RMD fused real-val predictions differ")
                    for i, pred in enumerate(preds):
                        all_queries = pred.pop("_raw_queries")
                        row = image_record(all_queries, self._prepare_batch(i, batch), self.data["path"], self.args.conf)
                        require(row["image"] not in seen, "Duplicate evaluation image")
                        seen.add(row["image"])
                        for prediction, index in zip(row["predictions"], all_queries["query_indices"]):
                            prediction["query_index"] = index
                        row.update(split=split, checkpoint_sha256=digest_before, settings_sha256=digest(settings))
                        report["images"] += 1
                        report["ground_truth"] += len(row["ground_truth"])
                        report["predictions"] += len(row["predictions"])
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    return super().update_metrics(preds, batch)

            metrics = model.val(validator=EvidenceValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid ten-IoU AP array")
        require(sha256(weights) == digest_before, "Evaluation changed checkpoint bytes")
        report.update(status="PASS", finished=now(), mAP50_95=float(metrics.box.map), mAP50=float(metrics.box.map50),
                      AP75=float(ap[:, 5].mean()), precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      ap_by_class=ap.tolist(), iou_thresholds=[round(.5+i*.05, 2) for i in range(10)],
                      native_speed=metrics.speed,
                      speed_ms_per_processed_image={k: v*report["native_speed_denominator_images"]/report["images"] for k, v in metrics.speed.items()},
                      predictions_sha256=sha256(stream_path), predictions_path=str(stream_path))
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output/"metrics.json", report)
    return report


def evaluate_formal(p, split, active_workers, allow_revision=False, recover_final=False):
    require(not active_workers, "Evaluation requires no active experiment worker")
    train = read_json(OUT/"training_identity.json")
    completion = read_json(OUT/"training_completed.json")
    require(train and completion and completion["status"] == "TRAINING_COMPLETED", "No genuine training completion evidence; resume interrupted training first")
    best = p["run"]/"weights/best.pt"
    require(best.is_file() and completion["training_binding"] == train["prepared"]["binding"], "Training identity/checkpoint missing")
    checkpoint_meta = read_json(OUT/"checkpoint_identity.json")
    require(checkpoint_meta and checkpoint_meta["best"]["sha256"] == sha256(best), "Best differs from the last native save/final-eval record")
    identity = evaluation_identity(train, allow_revision)
    data = inventory(p)
    require(data == train["prepared"]["data"], "Training/evaluation dataset identity changed")
    if recover_final:
        state = read_json(OUT/"final_eval.json", {})
        require(split == "val" and state.get("status") == "FINAL_EVAL_FAILED", "Only failed final evaluation can be recovered")
    binding = dict(checkpoint_sha256=sha256(best), data=data, eval_code=identity["code"], protocol=EVAL, policy=POLICY,
                   training_binding=train["prepared"]["binding"])
    if split == "test":
        lock = read_json(OUT/"val_lock.json")
        require(lock and lock["binding"] == binding and lock["status"] == "PASS", "Test requires the same best, full val lock, source, data and protocol")
        require(sha256(lock["report"]) == lock["report_sha256"], "Val lock report changed")
    folder = OUT/"evaluations"/(split + "_" + unique_name())
    report = evaluate_checkpoint(best, p["data"], folder, split=split, identity=identity)
    expected = data["splits"][split]
    require(report["images"] == expected["images"] and report["ground_truth"] == expected["boxes"], "Evaluation did not cover the frozen split")
    pointer = dict(status="PASS", binding=binding, report=str(folder/"metrics.json"), report_sha256=sha256(folder/"metrics.json"))
    write_json(OUT/("val_lock.json" if split == "val" else "test_result.json"), pointer)
    if recover_final:
        write_json(OUT/"recovery"/(unique_name()+".json"), dict(status="EVALUATION_RECOVERED", original_final_eval=read_json(OUT/"final_eval.json"),
                    original_exit=read_json(OUT/"latest_exit.json"), evaluation=pointer, identity=identity))
    return pointer


def pack(p, include_predictions=False, full=False):
    folder = OUT/"packages"; folder.mkdir(parents=True, exist_ok=True)
    target = folder/("rmd_v1_"+("FULL" if full else "LIGHT")+"_"+unique_name()+".tar.gz")
    snap = OUT/"pack_source"; source_snapshot(snap)
    files = {}
    for base, prefix in ((OUT, "evidence"), (p["run"], "run"), (ROOT/"docs/rmd_v1", "docs")):
        if not base.exists():
            continue
        for file in base.rglob("*"):
            if not file.is_file() or file.is_symlink() or folder in file.parents:
                continue
            if "native_resume_fixture" in file.parts:
                continue  # correctness report suffices; exclude its synthetic dataset/checkpoints
            if file.suffix in {".pt", ".pth"}:
                if not (full and base == p["run"] and file.name in {"best.pt", "last.pt"}):
                    continue
            if not include_predictions and file.name.endswith("predictions_gt.jsonl.gz"):
                continue
            files[prefix+"/"+file.relative_to(base).as_posix()] = file
    weights = {}
    for name in ("best", "last"):
        file = p["run"]/f"weights/{name}.pt"
        weights[name] = dict(exists=file.is_file(), included=full and file.is_file(),
                             bytes=file.stat().st_size if file.is_file() else None,
                             sha256=sha256(file) if file.is_file() else None)
    manifest = []
    with tarfile.open(target, "w:gz") as archive:
        for name, file in sorted(files.items()):
            row = dict(path=name, bytes=file.stat().st_size, sha256=sha256(file))
            archive.add(file, arcname=name, recursive=False); manifest.append(row)
        prediction_files = [name for name in files if name.endswith("predictions_gt.jsonl.gz")]
        metadata = dict(created=now(), format="FULL" if full else "LIGHT", weights=weights,
                        requested_predictions=include_predictions, includes_predictions=bool(prediction_files), prediction_files=prediction_files,
                        training=read_json(OUT/"training_completed.json"),
                        final_eval=read_json(OUT/"final_eval.json"), val=read_json(OUT/"val_lock.json"), test=read_json(OUT/"test_result.json"))
        for name, content in (("README.md", "# RMD v1 evidence\n\nFailure/PENDING evidence is retained. Inspect reports before interpreting metrics.\n"
                              + ("Includes best/last when available.\n" if full else "LIGHT excludes model weights and all dataset images.\n")),
                             ("PACKAGE_STATUS.json", json.dumps(metadata, indent=2, allow_nan=False))):
            raw = content.encode(); info = tarfile.TarInfo(name); info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
            import hashlib
            manifest.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        raw = json.dumps(manifest, indent=2).encode(); info = tarfile.TarInfo("MANIFEST.json"); info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    from c19_lif_v1_results import verify_archive
    verify_archive(target)
    return dict(path=str(target), sha256=sha256(target), files=len(manifest), **metadata)
