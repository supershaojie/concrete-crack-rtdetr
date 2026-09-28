"""Package existing evidence once. Never instantiate a model or access raw data."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from arg_v2_common import ROOT, OUT, RUN, FORMULA, git, read_json, write_json, digest, sha256, require


def verify_archive(path, expected):
    with tarfile.open(path) as archive:
        rows = json.load(archive.extractfile("manifest.json"))
        index = {row["path"]: row for row in rows}
        require(all(index.get(row["path"]) == row for row in expected), "Package inputs differ")
        for row in rows:
            stream = archive.extractfile(row["path"])
            count, hasher = 0, hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(block); hasher.update(block)
            require(count == row["bytes"] and hasher.hexdigest() == row["sha256"], "Package integrity failure")


def pack():
    from arg_v2 import file_info, status, code_identity, verify_preflight
    from arg_v2_evaluation import validate_record
    OUT.mkdir(parents=True, exist_ok=True)
    training = read_json(OUT / "training_identity.json")
    code = code_identity()
    checkpoints = [file_info(RUN / "weights" / name) for name in ("best.pt", "last.pt")]
    missing = [name for name in ("prepare.json", "data_snapshot.json", "initialization.json", "preflight.json",
        "training_identity.json", "training_completed.json", "epochs.jsonl", "train_args.yaml",
        "recipe_diff.json", "data_manifest.jsonl.gz", "source_snapshot.tar.gz", "actual_setup.json")
        if not (OUT / name).is_file()]
    try:
        require(training, "Missing training identity")
        snapshot = read_json(OUT / "data_snapshot.json", {})
        require(snapshot.get("status") == "PASS" and snapshot.get("snapshot_id") == digest(training["binding"]["data"]),
                "Saved data snapshot is missing, invalid or differs from training")
        verify_preflight(read_json(OUT / "preflight.json", {}), training["binding"])
        require(read_json(OUT / "training_completed.json", {}).get("status") == "TRAINING_COMPLETED",
                "Missing native training completion")
    except (RuntimeError, KeyError) as error:
        missing.append("training/preflight: " + str(error))
    dispatches = sorted((OUT / "dispatches").glob("*/dispatch.json"))
    for name in ("dispatch.json", "worker.json", "exit.json", "python_exit_code.txt"):
        if not dispatches or not (dispatches[-1].parent / name).is_file():
            missing.append("dispatch/" + name)
    metrics = {}
    for split in ("val", "test"):
        lock = read_json(OUT / (split + "_lock.json"))
        try:
            require(training and checkpoints[0]["exists"] and lock, "Missing training/best/lock")
            context = dict(checkpoint_sha256=checkpoints[0]["sha256"], data=training["binding"]["data"],
                           eval_identity=lock["eval_identity"])
            validate_record(lock, split, context)
            metrics[split] = {k: lock[k] for k in ("Precision", "Recall", "F1", "AP50", "AP75", "mAP50_95", "ap_by_class", "operating_point")}
        except (RuntimeError, KeyError, OSError) as error:
            missing.append(split + ": " + str(error))
    if not (RUN / "results.csv").is_file():
        missing.append("training/results.csv")
    for info in checkpoints:
        if not info["exists"]:
            missing.append(info["path"])
    summary = dict(status=status(), analysis_complete=not missing, missing=missing,
                   formal_metrics=metrics, formula=FORMULA, checkpoint_inventory=checkpoints,
                   includes_weights=False, includes_raw_dataset=False, includes_all_existing_predictions=True,
                   training_commit=training["binding"]["code"]["commit"] if training else None, current_code=code)
    files = {}
    for name in git("ls-files").splitlines():
        files["source/" + name] = ROOT / name
    for prefix, folder in (("evidence", OUT), ("training", RUN)):
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            relative = p.relative_to(folder)
            if (not p.is_file() or p.is_symlink() or "packages" in relative.parts
                    or p.suffix in {".pt", ".pth", ".tmp"} or p.name in {"operation.lock", "package.json", "finish.json"}):
                continue
            files[prefix + "/" + relative.as_posix()] = p
    manifest = [dict(path=name, bytes=p.stat().st_size, sha256=sha256(p)) for name, p in sorted(files.items())]
    # Volatile process status is included in the archive, not the reuse fingerprint.
    fingerprint = digest(dict(files=manifest, checkpoints=checkpoints, code=code, missing=missing))
    previous = read_json(OUT / "package.json", {})
    if previous.get("fingerprint") == fingerprint and Path(previous["package"]["path"]).is_file():
        require(file_info(previous["package"]["path"]) == previous["package"], "Previously completed package changed")
        return previous
    folder = OUT / "packages"
    folder.mkdir(exist_ok=True)
    filename = folder / ("ARG_v2_ANALYSIS_" + fingerprint[:20] + ".tar.gz")
    def receipt():
        value = dict(status="PASS", fingerprint=fingerprint, analysis_complete=not missing, missing=missing,
                     package=file_info(filename), includes_weights=False)
        write_json(OUT / "package.json", value)
        return value
    if filename.exists():
        verify_archive(filename, manifest)  # crash after completed archive, before receipt
        return receipt()
    partial = filename.with_suffix(".partial")
    if partial.exists():
        # Explicit pack/finish reentry resumes packaging and retains the interrupted bytes.
        partial.rename(partial.with_name(partial.name + ".interrupted." + str(time.time_ns())))
    archive_manifest = []
    with tarfile.open(partial, "w:gz") as archive:
        def add_bytes(name, content):
            info = tarfile.TarInfo(name); info.size = len(content); info.mtime = int(time.time())
            archive.addfile(info, io.BytesIO(content))
            archive_manifest.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        add_bytes("README.md", ("# ARG v2 analysis package\n"
            "ARG-v2-q2: lambda=0.20, epsilon_w=0.05, quality_gate=stopgrad(IoU)^2, denominator=M.\n"
            "Contains all existing query/GT exports and evidence; no raw dataset or weight bodies.\n"
            "See summary.json for completeness, missing material and true PASS/FAIL/PENDING scope.\n"
            "Precision is detection precision at the native max smoothed F1 operating point, not classification accuracy.\n"
            "Metric fractions are not percentages. Sampled scalar loss ratios are not gradient proportions.\n").encode())
        for row in manifest:
            p = files[row["path"]]
            info = tarfile.TarInfo(row["path"]); info.size = row["bytes"]; info.mtime = int(p.stat().st_mtime)
            with p.open("rb") as stream:
                archive.addfile(info, stream)
            archive_manifest.append(row)
        if training and not (OUT / "source_snapshot.tar.gz").is_file():
            import subprocess
            source = subprocess.check_output(["git", "archive", "--format=tar.gz", training["binding"]["code"]["commit"]],
                                             cwd=ROOT, timeout=60)
            add_bytes("training_source_snapshot.tar.gz", source)
        add_bytes("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False).encode())
        add_bytes("manifest.json", json.dumps(archive_manifest, indent=2).encode())
    verify_archive(partial, manifest)
    partial.rename(filename)
    return receipt()
