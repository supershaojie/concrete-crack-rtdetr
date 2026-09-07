"""File provenance and bounded export for the read-only CBR rescue diagnosis."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "81d5f0175e0168e022f29feb1ceb54de8a52deb5"
C17 = "0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139"
C19 = "025997e3c51eaf6933534308a95da6ebf97bff53"
C20_SHA = "6eda2d56209a4e714490ab11297a9a92b9cbc509dc8baa646eb742eac3f208ba"
INIT_SHA = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"
MODULE_DIR = "ultralytics-main/ultralytics/nn/modules/"
TASK_FILES = ["tools/cbr_rescue_common.py", "tools/cbr_rescue_analysis.py",
              "tools/diagnose_cbr_rescue.py", "tools/autodl_cbr_rescue.sh",
              "tools/test_cbr_rescue.py", "docs/CBR_RESCUE_DIAGNOSTICS.md",
              "docs/cbr_rescue_local_validation.json"]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(data):
    return digest(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False,
                               default=lambda x: str(x) if isinstance(x, Path) else vars(x)) + "\n",
                    encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT)


def source_state():
    """Require all pre-existing tracked files unchanged, allowing platform line endings."""
    protected = {}
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", BASE))) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name, path = member.name, ROOT / member.name
            require(path.is_file(), f"C20 source missing: {name}")
            original = archive.extractfile(member).read()
            require(canonical(path.read_bytes()) == canonical(original), f"C20 source changed: {name}")
    for name, ref in (("cscef_v5.py", C17), ("cscef_v51.py", C17), ("cbr.py", C19)):
        path = MODULE_DIR + name
        data = (ROOT / path).read_bytes()
        expected = canonical(git("show", f"{ref}:{path}"))
        require(canonical(data) == expected, f"Module differs from source {ref}: {path}")
        protected[path] = {"raw_sha256": digest(data), "canonical_lf_sha256": expected, "source_commit": ref}
    files = git("ls-files").decode().splitlines()
    return {"commit": git("rev-parse", "HEAD").decode().strip(),
            "branch": git("branch", "--show-current").decode().strip(),
            "status": git("status", "--porcelain").decode(), "base": BASE, "protected_modules": protected,
            "code_hashes": {n: canonical((ROOT / n).read_bytes()) for n in files if (ROOT / n).is_file()}}


def package(folder, output):
    """Allowlisted evidence only; refuse incomplete runs, symlinks, overwrite and >20 MiB."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    require(report["status"] == "completed", "Only a completed diagnosis can be packaged.")
    names = report["artifacts"]
    require(len(names) == len(set(names)) and "summary.json" in names, "Invalid artifact list.")
    payloads = {}
    for name in names:
        rel = Path(name)
        path = folder / rel
        require(not rel.is_absolute() and ".." not in rel.parts and path.resolve().is_relative_to(folder),
                f"Unsafe artifact path: {name}")
        require(path.is_file() and not path.is_symlink(), f"Missing or linked artifact: {name}")
        require(path.suffix in {".json", ".csv", ".md", ".txt", ".log", ".patch"}, f"Excluded artifact: {name}")
        payloads[rel.as_posix()] = path.read_bytes()
    inventory = [{"name": n, "bytes": len(b), "sha256": digest(b)} for n, b in sorted(payloads.items())]
    payloads["CONTENTS.json"] = (json.dumps(inventory, indent=2) + "\n").encode()
    payloads["MANIFEST_SHA256.txt"] = "".join(f"{r['sha256']}  {r['name']}\n" for r in inventory).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    require(len(data) <= 20 * 1024 * 1024, f"Archive {len(data)} bytes exceeds 20 MiB; no evidence omitted.")
    sidecars = [output, output.with_name(output.name + ".sha256"), output.with_name(output.name + ".inventory.json")]
    require(not any(p.exists() for p in sidecars), "Preserve existing package and sidecars.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        f.write(data)
    sidecars[1].write_text(f"{digest(data)}  {output.name}\n", encoding="utf-8")
    write_json(sidecars[2], {"archive_bytes": len(data), "sha256": digest(data), "files": inventory})
    print(json.dumps({"archive": str(output), "bytes": len(data), "sha256": digest(data)}, indent=2))
