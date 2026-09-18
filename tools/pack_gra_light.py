"""Package existing GRA evidence/source, default <20 MiB; never runs train/eval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

from init_gra import ROOT, VARIANTS, require, sha256, runtime, write_json
from train_gra import source_manifest, MAIN


def package(variant, output, run=None, evidence=None):
    output = Path(output).resolve()
    require(not any(Path(str(output) + suffix).exists() for suffix in ("", ".sha256", ".partial", ".verification.json")),
            "Existing archive/sidecar protected")
    content, omitted = {}, []
    metadata = ROOT / "outputs/gra" / variant
    run = Path(run or MAIN / "runs/c_series" / VARIANTS[variant][2])

    def collect(folder, prefix, source=False):
        if not folder.exists():
            return
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(folder)
            name = prefix + "/" + rel.as_posix()
            if path == output or path.suffix.lower() in {".pt", ".pth", ".zip", ".npz", ".npy", ".tar", ".gz"} or "predictions" in path.name:
                omitted.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "reason": "weights/predictions/archive excluded"})
                continue
            if "__pycache__" in rel.parts or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md", ".csv", ".txt", ".log", ".png", ".sh"}:
                continue
            if path.suffix.lower() in {".log", ".txt"}:
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 64000))
                    content[name + ".tail.txt"] = stream.read()
            elif path.stat().st_size <= 4 * 1024 * 1024:
                content[name] = path.read_bytes()
            else:
                omitted.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "reason": "individual file >4 MiB; retained at source"})

    collect(metadata, "metadata")
    if evidence:
        collect(Path(evidence), "checks")
    collect(run, "training")
    collect(ROOT / "docs/gra", "docs")
    for name in source_manifest():
        content["source/" + name] = (ROOT / name).read_bytes()
    for path in (ROOT / "tools").glob("*gra*.sh"):
        content["source/tools/" + path.name] = path.read_bytes()
    summary = {"created": datetime.now(timezone.utc).isoformat(), "variant": variant, "runtime": runtime(),
               "source_manifest": source_manifest(), "omitted": omitted,
               "scope": "existing evidence only; no weights, dataset, reference ZIP, or complete predictions",
               "formal_training_state": "NOT_STARTED" if not (metadata / "training/training_state.json").exists() else
                   json.loads((metadata / "training/training_state.json").read_text(encoding="utf-8")),
               "missing": [str(p) for p in (run / "args.yaml", run / "results.csv", metadata / "training/training_state.json") if not p.exists()]}
    content["package.json"] = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = [{"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in sorted(content.items())]
    content["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    require(not partial.exists(), "Previous partial archive protected")
    with partial.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, raw in sorted(content.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    require(partial.stat().st_size < 20 * 1024 * 1024, "Light archive exceeded 20 MiB; partial retained, nothing deleted")
    with tarfile.open(partial, "r:gz") as archive:
        require(set(archive.getnames()) == set(content), "Archive members differ")
        for item in manifest:
            raw = archive.extractfile(item["path"]).read()
            require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "Archive integrity failure")
    # Exclusive link protects against races; never replace an existing user archive.
    import os
    os.link(partial, output)
    partial.unlink()
    with Path(str(output) + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha256(output) + "  " + output.name + "\n")
    verification = {"status": "PASSED_ARCHIVE_INTEGRITY", "bytes": output.stat().st_size, "sha256": sha256(output),
                    "members": len(content), "missing_evidence": summary["missing"], "not_a_claim_of_complete_training": True}
    write_json(Path(str(output) + ".verification.json"), verification)
    print(json.dumps({"archive": str(output), **verification}, indent=2))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_gra_v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--evidence", type=Path)
    args = p.parse_args()
    package(args.variant, args.output, args.run, args.evidence)


if __name__ == "__main__":
    main()
