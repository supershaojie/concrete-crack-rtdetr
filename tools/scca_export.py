"""Streaming evaluation evidence and complete, verified archives; never runs inference."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import torch
from init_scca import ROOT, require, runtime, sha256, write_json
from ultralytics.utils import LOGGER, ops


@contextmanager
def evaluation_log(output):
    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log

        def write(self, value):
            self.terminal.write(value)
            self.log.write(value)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def isatty(self):
            return False

    with (output / "evaluation.log").open("x", encoding="utf-8") as log:
        # Ultralytics' existing terminal handler predates redirect_stdout.
        handler = logging.StreamHandler(log)
        LOGGER.addHandler(handler)
        try:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                yield
        finally:
            LOGGER.removeHandler(handler)


class PredictionStream:
    """One record per split image, including empty GT/predictions, no rounded floats."""

    def __init__(self, output, dataset_root, expected_files, split, conf):
        self.path = Path(output) / "predictions_gt.jsonl.gz"
        self.root = Path(dataset_root).resolve()
        self.expected = Counter(str(Path(p).resolve()) for p in expected_files)
        self.seen = Counter()
        self.split, self.conf = split, conf
        self.stream = gzip.open(self.path, "xt", encoding="utf-8")
        self.closed = False
        self.predictions = self.ground_truths = 0

    def write_batch(self, predictions, batch):
        require(len(predictions) == len(batch["im_file"]), "Prediction/image batch differs")
        for i, pred in enumerate(predictions):
            path = Path(batch["im_file"][i]).resolve()
            key = str(path)
            require(self.seen[key] < self.expected[key], f"Unexpected/repeated split image: {path}")
            h, w = map(int, batch["ori_shape"][i])
            ih, iw = batch["img"].shape[-2:]
            # RT-DETR stretches images: independent x/y gains, no letterbox subtraction.
            boxes = pred["bboxes"].detach().cpu().double() * torch.tensor([w/iw, h/ih, w/iw, h/ih], dtype=torch.float64)
            scores = pred["conf"].detach().cpu().tolist()
            classes = pred["cls"].detach().cpu().tolist()
            selected = batch["batch_idx"] == i
            gt = ops.xywh2xyxy(batch["bboxes"][selected].detach().cpu().double())
            gt *= torch.tensor([w, h, w, h], dtype=torch.float64)
            gt_cls = batch["cls"][selected].detach().cpu().flatten().tolist()
            record = dict(schema="scca_predictions_gt_v1", split=self.split,
                          image=os.path.relpath(path, self.root).replace("\\", "/"),
                          original_size=dict(height=h, width=w),
                          bbox_format="xyxy_original_pixels_unclipped", category_id_base=0,
                          predictions=[dict(category_id=int(c), score=s, bbox=b, used_for_metrics=s > self.conf)
                                       for b, s, c in zip(boxes.tolist(), scores, classes)],
                          ground_truth=[dict(category_id=int(c), bbox=b) for b, c in zip(gt.tolist(), gt_cls)])
            self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            self.seen[key] += 1
            self.predictions += len(scores)
            self.ground_truths += len(gt_cls)
        self.stream.flush()

    def close(self):
        if not self.closed:
            self.stream.close()
            self.closed = True

    def finish(self):
        self.close()
        require(self.seen == self.expected, "Incomplete split prediction/GT coverage")
        report = dict(schema="scca_predictions_gt_v1", path=self.path.name, sha256=sha256(self.path),
                      status="completed", images=sum(self.seen.values()), expected_images=sum(self.expected.values()),
                      predictions=self.predictions, ground_truths=self.ground_truths, dataset_root=str(self.root),
                      split=self.split, all_queries=True, metric_conf=self.conf,
                      coordinates="Unclipped continuous xyxy in original image pixels; x scales by original_width/input_width, y by original_height/input_height; no +1 convention.",
                      precision="FP32 inference values serialized without decimal rounding; coordinate conversion uses float64.",
                      coverage="Exact multiset match with validator dataset.im_files; includes empty images.")
        write_json(self.path.parent / "predictions_gt_manifest.json", report)
        return report


def verify_archive(path, manifest):
    expected = {row["path"]: row for row in manifest}
    seen = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            require(member.isfile() and member.name not in seen, "Invalid/duplicate archive member")
            seen.add(member.name)
            source = archive.extractfile(member)
            if member.name == "MANIFEST.json":
                require(json.load(source) == manifest, "Embedded manifest mismatch")
                continue
            digest, size = hashlib.sha256(), 0
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
            require(member.name in expected, f"Unexpected archive member: {member.name}")
            row = expected[member.name]
            require(size == row["bytes"] and digest.hexdigest() == row["sha256"], f"Corrupt archive member: {member.name}")
    require(seen == set(expected) | {"MANIFEST.json"}, "Archive inventory mismatch")


def package_complete(variant, destination=None):
    from train_scca import MAIN, paths
    p = paths(variant)
    destination = Path(destination or MAIN / "downloads/scca" / f"{variant}_complete_{datetime.now():%Y%m%d_%H%M%S_%f}.tar.gz").resolve()
    sidecars = [Path(str(destination) + suffix) for suffix in (".sha256", ".inventory.json", ".verification.json")]
    require(not any(path.exists() for path in [destination, *sidecars]), "Preserve old archive/sidecars")
    require(not any(destination.is_relative_to(p[k].resolve()) for k in ("run", "launch")), "Archive cannot be inside input folders")
    files, blobs, missing = {}, {}, []

    def add(path, name):
        require(not path.is_symlink(), f"Refuse symlink in evidence: {path}")
        require(name not in files and name not in blobs, f"Duplicate evidence: {name}")
        files[name] = path

    def tree(folder, prefix, skip_evaluation=False):
        if not folder.is_dir():
            missing.append(prefix)
            blobs[prefix + "/NOT_RUN.txt"] = b"No evidence available; no pass or completion claimed.\n"
            return
        for path in sorted(folder.rglob("*")):
            require(not path.is_symlink(), f"Refuse symlink in evidence: {path}")
            rel = path.relative_to(folder)
            if skip_evaluation and rel.parts[0].startswith("evaluation_"):
                continue
            if path.is_file():
                add(path, prefix + "/" + rel.as_posix())

    tree(p["run"], "training")
    tree(p["launch"], "metadata/launch", skip_evaluation=True)
    evaluation_state = {}
    for split in ("val", "test"):
        folder = p["launch"] / ("evaluation_" + split)
        tree(folder, "evaluation/" + split)
        metrics = folder / "metrics.json"
        evidence = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
        evaluation_state[split] = dict(status=evidence.get("status", "NOT_RUN"),
                                       checkpoint_sha256=evidence.get("checkpoint_sha256"),
                                       policy=evidence.get("policy"))
        for name in ("metrics.json", "predictions_gt.jsonl.gz", "predictions_gt_manifest.json", "evaluation.log", "exit_code.json"):
            if not (folder / name).is_file(): missing.append(f"evaluation/{split}/{name}")
        stream_manifest = folder / "predictions_gt_manifest.json"
        if stream_manifest.is_file():
            stream = json.loads(stream_manifest.read_text(encoding="utf-8"))
            require((folder / "predictions_gt.jsonl.gz").is_file() and
                    sha256(folder / "predictions_gt.jsonl.gz") == stream["sha256"], "Changed prediction stream")
        if "ap_by_class" not in evidence: missing.append(f"evaluation/{split}/AP50:0.05:AP95")
    for name in ("weights/best.pt", "weights/last.pt", "args.yaml", "results.csv", "results.png"):
        if not (p["run"] / name).is_file(): missing.append("training/" + name)
    for name in ("actual_train_args.yaml", "initialization.json", "parameter_diff.json", "nc1_loading.json",
                 "training_setup.json", "console.log", "exit_code.json", "process_exit_code.txt", "plan.json"):
        if not (p["launch"] / name).is_file(): missing.append("metadata/launch/" + name)
    for folder, prefix, patterns in ((p["run"], "training", ("*PR_curve.png", "*P_curve.png", "*R_curve.png", "*F1_curve.png", "confusion_matrix*.png", "labels*", "train_batch*", "val_batch*")),
                                     *[(p["launch"] / ("evaluation_" + s) / "plots", "evaluation/" + s + "/plots", ("*PR_curve.png", "*P_curve.png", "*R_curve.png", "*F1_curve.png", "confusion_matrix*.png", "val_batch*_pred.*")) for s in ("val", "test")]):
        for pattern in patterns:
            if not any(folder.glob(pattern)): missing.append(prefix + "/" + pattern)

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        path = ROOT / name
        if path.is_file() and ((name.startswith("ultralytics-main/") and path.suffix in {".py", ".yaml", ".toml", ".txt"}) or
                               (name.startswith("tools/") and ("scca" in name or "c25" in name or "cscef" in name)) or
                               name.startswith(("docs/scca/", "docs/c25/")) or name == "experiment_records/scca_aifi.md"):
            add(path, "metadata/source/" + name)
    blobs["metadata/source_working_tree.patch"] = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    blobs["metadata/source_c24_to_head.patch"] = subprocess.check_output(["git", "diff", "f6e9dfda765046ae7691302cf5ec89d3f76cec5d", "HEAD", "--binary"], cwd=ROOT)
    blobs["metadata/git_status.txt"] = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT)
    for key, path in (("c2_args", p["c2_args"]), ("data_config", MAIN / "configs/crack_autodl.yaml")):
        if path.is_file(): add(path, "metadata/" + key + ".yaml")
        else: missing.append("metadata/" + key)
    weight_hashes = {}
    for key, path in (("unified_initialization", p["source"]), ("controlled_initialization", p["init"])):
        if path.is_file(): weight_hashes[key] = dict(path=str(path), sha256=sha256(path))
        else: missing.append("metadata/hash/" + key)
    try:
        blobs["metadata/pip_freeze.txt"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        missing.append("metadata/pip_freeze: " + repr(error))
    training_exit = p["launch"] / "exit_code.json"
    record = dict(variant=variant, created=datetime.now(timezone.utc).isoformat(), runtime=runtime(),
                  training_exit=json.loads(training_exit.read_text(encoding="utf-8")) if training_exit.is_file() else "NOT_RECORDED",
                  evaluation=evaluation_state, initialization_hashes=weight_hashes, missing_evidence=sorted(set(missing)),
                  status="evidence_complete" if not missing else "missing_evidence", performed_training_or_evaluation=False,
                  note="Archive copies existing evidence only. Runtime is packaging environment; training/evaluation runtimes remain in their own reports. No dataset or environment copy.")
    blobs["metadata/package.json"] = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stream large checkpoints/logs directly from disk; verify archived bytes against pre-read hashes.
    with destination.open("xb") as raw, tarfile.open(fileobj=raw, mode="w|gz") as archive:
        for name in sorted(set(files) | set(blobs)):
            if name in files:
                path = files[name]
                size, digest = path.stat().st_size, sha256(path)
                info = tarfile.TarInfo(name)
                info.size = size
                with path.open("rb") as source: archive.addfile(info, source)
                require(path.stat().st_size == size and sha256(path) == digest, f"Evidence changed while packaging: {path}; retry with a new timestamp when idle")
            else:
                value = blobs[name]
                size, digest = len(value), hashlib.sha256(value).hexdigest()
                info = tarfile.TarInfo(name)
                info.size = size
                archive.addfile(info, io.BytesIO(value))
            manifest.append(dict(path=name, bytes=size, sha256=digest))
        value = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(value)
        archive.addfile(info, io.BytesIO(value))
    verify_archive(destination, manifest)
    write_json(sidecars[1], manifest)
    digest = sha256(destination)
    sidecars[0].write_text(digest + "  " + destination.name + "\n", encoding="utf-8")
    write_json(sidecars[2], dict(status="passed", archive_sha256=digest, files=len(manifest),
                               archive_integrity="Every member size/hash checked by streaming readback",
                               evidence_status=record["status"], missing_evidence=record["missing_evidence"]))
    print(destination, destination.stat().st_size, digest)
    print("Evidence:", record["status"], "; missing:", len(record["missing_evidence"]))
    return destination
