"""Create a verified <20 MiB DCC evidence archive; never runs train, val or test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

from dcc_common import ROOT, VARIANTS, require, sha256, runtime
from train_dcc import code_identity

MAX_BYTES = 20 * 1024 * 1024
FILE_LIMIT = 3 * 1024 * 1024
TAIL_BYTES = 64 * 1024


def package(inputs, output, variant="cbr_lif_dcc_v1"):
    output = Path(output).resolve()
    require(not output.exists() and not Path(str(output) + ".sha256").exists(), "Existing archive/sidecar protected")
    entries, omitted = {}, []

    def add(path, name, log_tail=False):
        path = Path(path)
        if path.is_symlink():
            omitted.append(dict(path=str(path), reason="symlink"))
            return
        if path.stat().st_size > FILE_LIMIT and not log_tail:
            omitted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path), reason="oversized; retained on host"))
            return
        require(name not in entries, "Duplicate archive path: " + name)
        if log_tail:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - TAIL_BYTES))
                entries[name + ".tail.txt"] = stream.read(TAIL_BYTES)
        else:
            entries[name] = path.read_bytes()

    # Source-only tree: no assets, environments, checkpoints, reference ZIP or data.
    for path in sorted((ROOT / "ultralytics-main/ultralytics").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"} and "__pycache__" not in path.parts:
            add(path, "source/" + path.relative_to(ROOT).as_posix())
    for path in sorted(list((ROOT / "tools").glob("*dcc*.py")) + list((ROOT / "tools").glob("*dcc*.sh"))):
        add(path, "source/" + path.relative_to(ROOT).as_posix())
    for name in ("init_c19_lif_v1.py", "init_lif_down.py", "lif_down_topology.py", "c19_lif_v1_data.py",
                 "c19_lif_v1_probe.py", "c19_lif_v1_diagnostic.py", "c19_lif_v1_cutoff.py"):
        path = ROOT / "tools" / name
        if path.is_file():
            add(path, "source/tools/" + name)
    for path in sorted((ROOT / "docs/dcc").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".yaml", ".json", ".sh"}:
            add(path, "source/" + path.relative_to(ROOT).as_posix())
    folders = [Path(p).resolve() for p in inputs]
    for index, folder in enumerate(folders):
        require(folder.is_dir(), "Missing evidence directory: " + str(folder))
        require(folder != ROOT and folder != ROOT.parent, "Choose an experiment evidence directory, not a project tree")
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(folder)
            if any(part.lower() in {"weights", "datasets", "images", "labels", ".git", "__pycache__"} for part in relative.parts):
                continue
            if path.suffix.lower() in {".pt", ".pth", ".npz", ".npy", ".zip", ".gz", ".tar"} or "prediction" in path.stem.lower():
                omitted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path), reason="weights/predictions/large binary excluded"))
                continue
            if path.suffix.lower() not in {".json", ".yaml", ".yml", ".csv", ".txt", ".log", ".png", ".md"}:
                continue
            # Curves/results/confusion matrices only; never arbitrary dataset imagery.
            if path.suffix.lower() == ".png" and not ("curve" in path.stem.lower() or path.stem in
                                                       {"results", "confusion_matrix", "confusion_matrix_normalized"}):
                continue
            add(path, "evidence/%d_%s/%s" % (index, folder.name, relative.as_posix()), path.suffix.lower() == ".log")
    info = dict(created=datetime.now(timezone.utc).isoformat(), variant=variant, runtime=runtime(),
                code_identity=code_identity(), inputs=list(map(str, folders)), omitted=omitted,
                max_bytes_exclusive=MAX_BYTES, training="NOT_RUN_BY_PACK", test="NOT_RUN_BY_PACK",
                exclusions="weights, datasets, reference ZIP, arrays, full predictions, unbounded logs",
                evidence_status="See original reports; packaging does not promote PENDING/FAILED to PASSED")
    entries["PACKAGE_INFO.json"] = (json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    manifest = {n: dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n, b in sorted(entries.items())}
    entries["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()

    class CappedBuffer(io.BytesIO):
        def write(self, data):
            require(self.tell() + len(data) < MAX_BYTES, "Archive exceeds <20MiB target; no archive written")
            return super().write(data)

    buffer = CappedBuffer()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, value in sorted(entries.items()):
            item = tarfile.TarInfo(name)
            item.size = len(value)
            archive.addfile(item, io.BytesIO(value))
    raw = buffer.getvalue()
    require(len(raw) < MAX_BYTES, "Archive cap exceeded")
    # Read back every member before delivery without extracting filesystem paths.
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        require(set(archive.getnames()) == set(entries), "Archive member mismatch")
        for item in archive.getmembers():
            require(item.isfile() and archive.extractfile(item).read() == entries[item.name], "Archive verification failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with Path(str(output) + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest + "  " + output.name + "\n")
    result = dict(path=str(output), bytes=len(raw), sha256=digest, members=len(entries), verification="PASSED")
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="cbr_lif_dcc_v1")
    p.add_argument("--input", type=Path, action="append", required=True, help="Experiment evidence directories; repeatable")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    package(args.input, args.output, args.variant)


if __name__ == "__main__":
    main()
