"""Allowlisted existing BFR evidence only; <20 MiB, no weights/data/train/test/download."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from bfr_p4_lifecycle import ROOT, VARIANTS, paths, runtime, require, sha256, write_json

LIMIT = 20 * 1024 * 1024
EVIDENCE = {"args.yaml", "parent_args.yaml", "appendix_args.yaml", "actual_train_args.yaml",
            "results.csv", "results.png", "metrics.json", "training_state.json", "training_setup.json",
            "plan.json", "recipe_diff.json", "preflight.json", "trainer_loading.json", "init_report.json",
            "initialization.json", "checks.json", "local_validation.json", "math_checks.json",
            "parent_dataset_inventory.json", "PROVENANCE.json", "VALIDATION.md", "SERVER_COMMANDS.md",
            "branch_audit.json", "recipe_comparison.json", "source_record.json", "controlled_init_pair.json"}
EVIDENCE.update({"parent_recipe_audit.json", "reference_package.json", "README.md", "IMPLEMENTATION.md",
                 "PREFLIGHT.md", "SERVER_COMMANDS.template.md", "local_checks.json", "gate_checks.json",
                 "math_cpu.json", "math_cuda.json", "math_cpu_server.json", "math_cuda_server.json", "ops_cpu.json",
                 "cpu_fp32.json", "cuda_fp32.json", "cuda_amp.json", "initialization_negative_checks.json",
                 "FINAL_CODE_BINDING.json", "final_code_binding.json", "native_state_forensics.json",
                 "native_lifecycle_checks.json", "native_save_resume.json"})
EVIDENCE.update({"lock_checks.json", "gates_cpu.json", "LOCAL_VALIDATION.json",
                 "cbr_lif_bfr_p4_v1_recipe_diff.json", "bfr_p4_v1_recipe_diff.json", "cli_verification.json"})
PLOT_PREFIXES = ("Box", "PR_curve", "P_curve", "R_curve", "F1_curve", "confusion_matrix")


def package(variant, run, output):
    require(not output.exists(), "Existing package protected")
    entries, omitted = {}, []
    roots = [("metadata", paths(variant)["metadata"]), ("docs", ROOT / "docs/bfr_p4"), ("training", run)]
    for prefix, folder in roots:
        if not folder.is_dir():
            continue
        for file in sorted(folder.rglob("*")):
            if not file.is_file() or file.is_symlink():
                continue
            rel = file.relative_to(folder)
            if "weights" in rel.parts or file.suffix.lower() in {".pt", ".pth", ".zip", ".npz", ".gz"}:
                continue
            name = prefix + "/" + rel.as_posix()
            if file.suffix == ".log" or (file.suffix == ".txt" and file.name.startswith(("start_", "resume_"))):
                with file.open("rb") as stream:
                    stream.seek(max(0, file.stat().st_size - 32768))
                    entries[name + ".tail.txt"] = stream.read()
            elif file.name in EVIDENCE or (file.suffix == ".png" and file.name.startswith(PLOT_PREFIXES)):
                if file.stat().st_size > 4 * 1024 * 1024:
                    omitted.append({"path": name, "bytes": file.stat().st_size, "sha256": sha256(file),
                                    "reason": "single evidence file exceeds 4 MiB light limit"})
                else:
                    entries[name] = file.read_bytes()
    for name in ("math_cpu.json", "math_cuda.json", "math_cpu_server.json", "math_cuda_server.json", "gate_checks.json",
                 "ops_cpu.json", "initialization_negative_checks.json", "FINAL_CODE_BINDING.json", "final_code_binding.json",
                 "native_state_forensics.json", "native_lifecycle_checks.json", "native_save_resume.json", "lock_checks.json",
                 "gates_cpu.json"):
        file = ROOT / "outputs/bfr_p4" / name
        if file.is_file() and not file.is_symlink() and file.stat().st_size <= 4 * 1024 * 1024:
            entries["metadata/shared/" + name] = file.read_bytes()
    source_files = list((ROOT / "tools").glob("*bfr_p4*.py"))
    source_files += [ROOT / "ultralytics-main/ultralytics/nn/modules/bfr_p4.py",
                     ROOT / "ultralytics-main/ultralytics/nn/tasks.py",
                     ROOT / "ultralytics-main/ultralytics/nn/modules/cbr.py",
                     ROOT / "ultralytics-main/ultralytics/nn/modules/lif_down.py",
                     ROOT / "ultralytics-main/ultralytics/nn/modules/__init__.py",
                     ROOT / "ultralytics-main/ultralytics/nn/autobackend.py"]
    source_files += [ROOT / "tools" / name for name in ("init_lif_down.py", "init_c19_lif_v1.py",
                                                        "c19_lif_v1_probe.py", "c19_lif_v1_data.py")]
    source_files += list((ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr").glob("*bfr*p4*.yaml"))
    for file in source_files:
        if file.is_file():
            entries["source/" + file.relative_to(ROOT).as_posix()] = file.read_bytes()
    info = {"runtime": runtime(), "variant": variant, "omitted": omitted, "limit_bytes": LIMIT,
            "scope": "Existing evidence; never invokes train/evaluation/test or downloads",
            "excluded": "weights, datasets, large predictions, reference package",
            "files": {name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                      for name, value in sorted(entries.items())}}
    entries["MANIFEST.json"] = (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, value in sorted(entries.items()):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    blob = stream.getvalue()
    require(len(blob) < LIMIT, "Light package would exceed 20 MiB; no partial archive delivered")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        require(set(archive.getnames()) == set(entries), "Archive inventory mismatch")
        for name, value in entries.items():
            require(archive.extractfile(name).read() == value, "Archive member verification failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        target.write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    with Path(str(output) + ".sha256").open("x", encoding="utf-8") as sidecar:
        sidecar.write(digest + "  " + output.name + "\n")
    report = {"status": "PASSED", "path": str(output), "bytes": len(blob), "sha256": digest,
              "members": len(entries), "omitted": omitted}
    write_json(Path(str(output) + ".verification.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_bfr_p4_v1")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.variant, args.run or paths(args.variant)["run"], args.output)


if __name__ == "__main__":
    main()
