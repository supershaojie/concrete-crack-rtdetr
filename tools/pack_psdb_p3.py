"""Verified lightweight evidence archive; never runs training or evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from init_psdb_p3 import ROOT, VARIANTS, require, sha256, git
from train_psdb_p3 import DEFAULT_VARIANT, paths, run_state, head, code_identity, write_json


def package(destination, variant=DEFAULT_VARIANT):
    p, destination = paths(variant), Path(destination).resolve()
    state = run_state(p)
    require(state["status"] not in {"RUNNING", "DISPATCHED", "FINAL_VALIDATING"}, "Do not package changing active run")
    require(not destination.exists() and not Path(str(destination) + ".partial").exists(), "Existing archive preserved")
    require(not destination.is_relative_to(p["run"].resolve()) and not destination.is_relative_to(p["launch"].resolve()), "Archive cannot contain itself")
    files, extra, missing, excluded = {}, {}, [], []
    allowed = {".json", ".yaml", ".csv", ".txt", ".md", ".py", ".sh", ".patch"}
    for prefix, folder in (("training", p["run"]), ("metadata/launch", p["launch"]),
                           ("metadata/source/docs/psdb_p3", ROOT / "docs/psdb_p3")):
        if not folder.is_dir():
            missing.append(str(folder))
            continue
        for file in sorted(folder.rglob("*")):
            if not file.is_file() or file.is_symlink():
                continue
            name = prefix + "/" + file.relative_to(folder).as_posix()
            if file.suffix == ".log":
                with file.open("rb") as stream:
                    stream.seek(max(0, file.stat().st_size - 64 * 1024))
                    extra[name + ".tail.txt"] = stream.read()
            elif file.suffix in allowed or (file.suffix.lower() in {".png", ".jpg"} and file.name.startswith(
                    ("results", "Box", "PR_", "P_", "R_", "F1_", "confusion_matrix"))):
                files[name] = file
            else:
                excluded.append(dict(path=str(file), bytes=file.stat().st_size, reason="weights/data/large predictions/archive excluded by light allowlist"))
    # Full tracked execution source/configuration is small and reconstructs what ran.
    for name in code_identity():
        files["metadata/source/" + name] = ROOT / name
    for path in (p["launch"] / "plan.json", p["launch"] / "evaluation_val/metrics.json",
                 p["launch"] / "evaluation_test/metrics.json", p["preflight"], p["run"] / "results.csv"):
        if not path.is_file():
            missing.append(str(path))
    metadata = dict(variant=variant, commit=head(), created=datetime.now(timezone.utc).isoformat(), state=state,
                    evidence_complete=not missing, missing_evidence=sorted(set(missing)), excluded=excluded,
                    scope="Existing light evidence only. No datasets, weights, full predictions or reference archives.",
                    git_status=git("status", "--porcelain"))
    extra["metadata/package.json"] = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = [dict(path=n, bytes=f.stat().st_size, sha256=sha256(f)) for n, f in sorted(files.items())]
    manifest += [dict(path=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n, b in sorted(extra.items())]
    extra["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + ".partial")
    with partial.open("xb") as stream, tarfile.open(fileobj=stream, mode="w|gz") as archive:
        for name, file in sorted(files.items()):
            archive.add(file, arcname=name, recursive=False)
        for name, value in extra.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    with tarfile.open(partial, "r:gz") as archive:
        require(set(archive.getnames()) == {v["path"] for v in manifest} | {"MANIFEST.json"}, "Archive inventory mismatch")
        for row in manifest:
            value = archive.extractfile(row["path"]).read()
            require(len(value) == row["bytes"] and hashlib.sha256(value).hexdigest() == row["sha256"], "Archive readback checksum failure")
    os.link(partial, destination)
    partial.unlink()
    report = dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha256(destination), integrity="PASSED",
                  evidence_complete=not missing, missing_evidence=sorted(set(missing)), target_bytes=20 * 1024 * 1024,
                  target_met=destination.stat().st_size < 20 * 1024 * 1024,
                  largest_members=sorted(manifest, key=lambda r: r["bytes"], reverse=True)[:20])
    with Path(str(destination) + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(report["sha256"] + "  " + destination.name + "\n")
    write_json(Path(str(destination) + ".verification.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    package(args.output, args.variant)
