"""Create and verify a capped SRE evidence archive; never evaluate, train or download."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 20 * 1024 * 1024 - 1
EXCLUDED = {".pt", ".pth", ".onnx", ".bin", ".npy", ".npz", ".zip", ".gz", ".tar", ".jpg", ".jpeg", ".bmp", ".cache"}
ALLOWED = {".json", ".yaml", ".yml", ".csv", ".md", ".txt", ".log", ".png"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate archive member")
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        if set(names) != set(manifest["files"]) | {"MANIFEST.json"}:
            raise RuntimeError("Archive manifest does not cover every member")
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise RuntimeError("Unsafe archive member")
            if member.name == "MANIFEST.json":
                continue
            data = archive.extractfile(member).read()
            expected = manifest["files"][member.name]
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise RuntimeError("Archive bytes/hash differ from manifest")
    return manifest


def package(metadata, output, run=None, evaluations=(), limit=LIMIT):
    if not 0 < limit <= LIMIT:
        raise ValueError("Limit must be positive and strictly below 20 MiB")
    output = Path(output).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError("Existing archive/checksum preserved")
    entries, sources, omitted = {}, {}, []

    def add_file(path, name, source_group):
        size = path.stat().st_size
        source = dict(path=str(path.resolve()), bytes=size, source_group=source_group)
        if path.is_symlink():
            omitted.append(dict(**source, reason="symlink; not followed"))
            return
        source["sha256"] = sha256(path)
        if path.suffix.lower() in EXCLUDED or path.suffix.lower() not in ALLOWED | {".py", ".sh"}:
            omitted.append(dict(**source, reason="weights/data/archive/raw predictions excluded"))
            return
        if path.suffix.lower() == ".png" and not any(word in path.stem.lower() for word in ("curve", "matrix", "results")):
            omitted.append(dict(**source, reason="image/preview excluded; only metric plots allowed"))
            return
        if path.suffix.lower() in {".log", ".txt"} and size > 65536:
            with path.open("rb") as stream:
                stream.seek(max(0, size-65536))
                content = stream.read(65536)
            name += ".tail.txt"
            source["transformation"] = "last 65536 bytes; original SHA256 retained"
        elif size > 2 * 1024 * 1024:
            omitted.append(dict(**source, reason="individual file exceeds 2 MiB; full original retained on server"))
            return
        else:
            content = path.read_bytes()
        if name in entries:
            raise RuntimeError("Duplicate archive destination: " + name)
        entries[name], sources[name] = content, source

    def add_folder(directory, prefix):
        directory = Path(directory).resolve(strict=True)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                # Scan only explicitly named metadata/run/evaluation folders; no dataset discovery.
                add_file(path, prefix + "/" + path.relative_to(directory).as_posix(), prefix)

    add_folder(metadata, "metadata")
    if run:
        add_folder(run, "run")
    for i, evaluation in enumerate(evaluations):
        add_folder(evaluation, f"evaluation_{i}")
    source_paths = list((ROOT / "tools").glob("*sre*.py")) + list((ROOT / "tools").glob("*sre*.sh"))
    source_paths += list((ROOT / "docs/sre").rglob("*")) if (ROOT / "docs/sre").exists() else []
    source_paths += list((ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr").glob("*sre*.yaml"))
    source_paths += [ROOT / "ultralytics-main/ultralytics/nn/modules" / name for name in ("sre.py", "cbr.py", "lif_down.py")]
    source_paths += [ROOT / "ultralytics-main/ultralytics/nn" / name for name in ("tasks.py", "modules/__init__.py", "autobackend.py")]
    source_paths += [ROOT / "tools" / name for name in ("c19_lif_v1_results.py", "init_c19_lif_v1.py", "init_lif_down.py")]
    for path in sorted(set(source_paths)):
        if path.is_file():
            add_file(path, "source/" + path.relative_to(ROOT).as_posix(), "source")
    commit = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--porcelain"], cwd=ROOT, text=True)
    manifest = dict(schema="sre_light_v1", created=datetime.now(timezone.utc).isoformat(), commit=commit,
                    git_status_porcelain=status, max_bytes=limit,
                    exclusions="weights, data, raw large predictions, archives and reference module bundles; no evaluation is run",
                    sources=sources, omitted=omitted,
                    files={name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in entries.items()})
    entries["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if sum(map(len, entries.values())) > 4 * LIMIT:
        raise ValueError("Evidence payload exceeds bounded memory budget; narrow input folders")
    class CappedBuffer(io.BytesIO):
        def write(self, data):
            if self.tell() + len(data) > limit:
                raise ValueError("Archive exceeds requested cap; narrow input folders")
            return super().write(data)
    buffer = CappedBuffer()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name,data in sorted(entries.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    raw = buffer.getvalue()
    if len(raw) > limit:
        raise ValueError("Compressed archive exceeds cap")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    verify_archive(output)
    digest = hashlib.sha256(raw).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    result = dict(status="PASSED", path=str(output), bytes=len(raw), sha256=digest, limit_bytes=limit,
                  commit=commit, files=len(entries), omitted_files=len(omitted))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=LIMIT)
    args = parser.parse_args()
    package(args.metadata, args.output, args.run, args.evaluation, args.max_bytes)
