"""Verified CSR-P3 evidence package; exclude weights/data/predictions, retain bounded log tails."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from init_csr_p3 import ROOT, VARIANTS, require, runtime, sha256, write_json
from train_csr_p3 import paths, run_state

LIMIT = 20 * 1024 * 1024


def verify_archive(path):
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        require(len(names) == len(set(names)) and set(names) == set(manifest) | {"MANIFEST.json"}, "Duplicate/missing package member")
        for name, expected in manifest.items():
            path = PurePosixPath(name)
            member = archive.getmember(name)
            require(not path.is_absolute() and ".." not in path.parts and member.isfile(), "Unsafe archive member")
            value = archive.extractfile(member).read()
            require(len(value) == expected["bytes"] and hashlib.sha256(value).hexdigest() == expected["sha256"], "Archive integrity failure: " + name)
    return manifest


def package(variant, output, evidence=(), limit=LIMIT):
    require(0 < limit <= LIMIT, "Maximum lightweight package limit is 20 MiB")
    p = paths(variant)
    state = run_state(p)
    require(state["status"] not in {"RUNNING", "DISPATCHED"}, "Do not package changing active training")
    output = Path(output).resolve()
    require(not output.exists() and not output.with_name(output.name + ".partial").exists(), "Existing package preserved")
    entries, excluded = {}, []
    def allowed(file):
        return file.suffix.lower() in {".json", ".yaml", ".yml", ".csv", ".txt", ".md", ".log", ".sh"} or (
            file.suffix.lower() == ".png" and file.name.startswith(("results", "Box", "PR_", "P_", "R_", "F1_", "confusion_matrix")))
    def put(name, value):
        require(name not in entries, "Duplicate package name")
        entries[name] = value
    def add_file(file, name):
        require(not file.is_symlink(), "Symlink evidence not accepted")
        if file.suffix.lower() == ".log":
            with file.open("rb") as stream:
                stream.seek(max(0, file.stat().st_size - 65536))
                put(name + ".tail.txt", stream.read(65536))
            excluded.append(dict(path=str(file), reason="Full log replaced by final <=64 KiB .txt tail", bytes=file.stat().st_size))
        else:
            put(name, file.read_bytes())
    def tree(folder, prefix):
        if not folder.is_dir():
            return
        for file in sorted(folder.rglob("*")):
            if not file.is_file():
                continue
            rel = file.relative_to(folder)
            if "weights" in rel.parts or file.suffix.lower() in {".pt", ".pth", ".npy", ".npz", ".gz", ".zip"} or file.name.startswith("predictions"):
                excluded.append(dict(path=str(file), reason="Weights/large predictions/archive excluded", bytes=file.stat().st_size))
            elif allowed(file):
                add_file(file, prefix + "/" + rel.as_posix())
            else:
                excluded.append(dict(path=str(file), reason="Outside evidence allowlist", bytes=file.stat().st_size))
    tree(p["launch"], "lifecycle")
    tree(p["run"], "training")
    for i, folder in enumerate(evidence):
        folder = Path(folder)
        if folder.is_file():
            require(allowed(folder) and not folder.name.startswith("predictions"), "Direct evidence must obey allowlist; weights/archives forbidden")
            add_file(folder, f"preflight_{i}/" + folder.name)
        else:
            tree(folder, f"preflight_{i}")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        if name and (name.startswith(("ultralytics-main/ultralytics/", "tools/", "configs/", "docs/csr_p3/")) or name == ".gitattributes"):
            file = ROOT / name
            if file.is_file() and file.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".sh", ".json", ".txt"}:
                add_file(file, "source/" + name)
    missing = [str(p["run"] / n) for n in ("args.yaml", "results.csv", "results.png") if not (p["run"] / n).is_file()]
    missing += [str(p["launch"] / n) for n in ("evaluation_val/metrics.json", "evaluation_test/metrics.json") if not (p["launch"] / n).is_file()]
    issues = []
    if state["status"] not in {"COMPLETED_200", "EARLY_STOPPED"}:
        issues.append("Training lifecycle is " + state["status"])
    best = p["run"] / "weights/best.pt"
    best_hash = sha256(best) if best.is_file() else None
    for split in ("val", "test"):
        folder = p["launch"] / ("evaluation_" + split)
        metrics_path = folder / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "PASSED" or not best_hash or metrics.get("checkpoint_sha256") != best_hash:
                issues.append(split + " evaluation is not PASSED for the current training-selected best.pt")
        names = {f.name for f in folder.rglob("*.png")}
        for curve in ("PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"):
            if curve not in names and "Box" + curve not in names:
                missing.append(str(folder / curve))
        for matrix in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            if matrix not in names:
                missing.append(str(folder / matrix))
    complete = not missing and not issues
    metadata = dict(schema="csr_p3_light_v1", created=datetime.now(timezone.utc).isoformat(), runtime=runtime(), variant=variant,
                    training_state=state, training_best_sha256=best_hash, limit_bytes=limit,
                    missing_evidence=missing, evidence_issues=issues, evidence_complete=complete,
                    excluded=excluded, note="Does not execute training/val/test; absent evidence stays absent")
    put("PACKAGE.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode())
    manifest = {n: dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in sorted(entries.items())}
    entries["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("xb") as stream, tarfile.open(fileobj=stream, mode="w|gz") as archive:
        for name, value in sorted(entries.items()):
            item = tarfile.TarInfo(name)
            item.size = len(value)
            archive.addfile(item, io.BytesIO(value))
    verify_archive(partial)
    size = partial.stat().st_size
    if size >= limit:
        oversized = dict(status="OVERSIZE", archive=str(partial), bytes=size, limit=limit,
                         largest_entries=sorted([dict(path=n, **v) for n, v in manifest.items()], key=lambda x: x["bytes"], reverse=True)[:30],
                         note="All selected key evidence retained; inspect listed files, no silent dropping")
        write_json(output.with_name(output.name + ".oversize.json"), oversized)
        raise RuntimeError(json.dumps(oversized, ensure_ascii=False))
    # Exclusive creation, preserving any destination produced concurrently.
    with output.open("xb") as destination, partial.open("rb") as source:
        import shutil
        shutil.copyfileobj(source, destination)
    partial.unlink()
    digest = sha256(output)
    with output.with_name(output.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest + "  " + output.name + "\n")
    report = dict(status="PASSED", bytes=size, sha256=digest, output=str(output), evidence_complete=complete,
                  missing_evidence=missing, evidence_issues=issues)
    write_json(output.with_name(output.name + ".verification.json"), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_csr_p3_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[], help="Additional init/preflight evidence file/directory; repeatable")
    args = parser.parse_args()
    package(args.variant, args.output, args.evidence)
