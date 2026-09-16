"""Bounded TRC evidence archive with per-file manifest and verified SHA256; no weights/data."""
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

from init_trc_v1 import ROOT, BASE_COMMIT, require, runtime, sha256
from train_trc_v1 import read_plan, status
from c19_lif_v1_results import verify_archive

MAX_BYTES = 32_000_000
MAX_FILE_BYTES = 4_000_000
TEXT_NAMES = {"plan.json", "training_state.json", "state.json", "args.yaml", "results.csv", "metrics.json",
              "actual_train_args.yaml", "authoritative_args.yaml", "data_config.yaml", "parameter_diff.json",
              "source_record.json", "dataset_inventory.json", "nc1_loading.json", "training_setup.json",
              "amp_resources.json", "pip_freeze.txt", "initialization.json", "preflight.json", "checks.json",
              "evaluation_val_pointer.json", "evaluation_test_pointer.json"}
PLOT_PREFIXES = ("results", "Box", "PR_curve", "P_curve", "R_curve", "F1_curve", "confusion_matrix")


def package(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = read_plan(plan_path)
    state = status(plan_path)
    require(state["status"] not in {"RUNNING", "STARTING"}, "Preserve a stable archive: training is active")
    require(all(not Path(str(output) + suffix).exists() for suffix in ("", ".partial", ".sha256")), "Existing archive/sidecar protected")
    require(not output.is_relative_to(Path(plan["args"]["save_dir"]).resolve()), "Archive cannot reside inside its formal run")
    entries, omitted = {}, []

    def add(path, name):
        path = Path(path)
        require(not path.is_symlink(), "Symlink is not an allowed archive input: " + str(path))
        if path.stat().st_size > MAX_FILE_BYTES:
            omitted.append(dict(path=name, bytes=path.stat().st_size, sha256=sha256(path), reason="per-file light limit"))
            return
        entries[name] = path.read_bytes()

    def evidence(folder, prefix):
        if not folder.exists():
            return
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(folder)
            if any(p in {"weights", "datasets", "data", "__pycache__"} for p in relative.parts):
                continue
            if path.name in TEXT_NAMES or (path.suffix.lower() == ".png" and path.name.startswith(PLOT_PREFIXES)):
                add(path, prefix + "/" + relative.as_posix())

    evidence(plan_path.parent, "audit")
    evidence(Path(plan["args"]["save_dir"]), "training")
    add(plan_path, "audit/plan.json")
    require("initialization_audit" in plan, "Selected initialization audit missing from plan")
    entries["audit/selected_initialization.json"] = (json.dumps(plan["initialization_audit"], ensure_ascii=False, indent=2) + "\n").encode()
    evaluations = {}
    for split in ("val", "test"):
        pointer_path = plan_path.parent / ("evaluation_" + split + "_pointer.json")
        if pointer_path.is_file():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            folder = Path(pointer["output"])
            require(sha256(folder / "metrics.json") == pointer["metrics_sha256"], "Selected evaluation report changed")
            evidence(folder, "evaluation_" + split)
            evaluations[split] = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    for key, name in (("preflight", "audit/selected_preflight.json"), ("authoritative_args", "audit/authoritative_args.yaml")):
        if Path(plan[key]).is_file():
            add(plan[key], name)
    initialized = Path(plan["initialized"])
    for path in (initialized.with_suffix(".json"), initialized.parent / "initialization.json"):
        if path.is_file():
            add(path, "audit/initialization/" + path.name)
    # Reproducible source scope: relevant Python, YAML and task documentation only.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        if name and ((name.startswith("tools/") and (name.endswith(".py") or ("trc" in name and name.endswith(".sh")))) or
                     (name.startswith("ultralytics-main/ultralytics/") and name.endswith((".py", ".yaml"))) or
                     (name.startswith("ultralytics-main/tests/") and "trc" in name and name.endswith(".py")) or
                     (name.startswith("docs/trc_v1/") and name.endswith((".md", ".txt", ".json", ".yaml", ".sh"))) or name == ".gitattributes"):
            add(ROOT / name, "source/" + name)
    for path in (ROOT / "outputs/trc_v1_delivery.json", ROOT / "outputs/trc_v1_delivery.txt"):
        if path.is_file():
            add(path, "audit/" + path.name)
    info = dict(schema="trc_v1_light_v1", created=datetime.now(timezone.utc).isoformat(), runtime=runtime(),
                base_commit=BASE_COMMIT, variant=plan["variant"], training_status=state["status"],
                test_status="NOT_RUN", size_limit=MAX_BYTES, omitted=omitted,
                excluded=["weights", "datasets", "reference module ZIP", "per-image predictions", "attention tensors"],
                evidence_complete=False, note="Archive records existing evidence; it never runs missing training/test")
    if "test" in evaluations:
        info["test_status"] = evaluations["test"].get("status", "UNKNOWN")
    required = {"training/args.yaml", "training/results.csv", "training/results.png", "audit/selected_preflight.json",
                "audit/selected_initialization.json", "evaluation_test/metrics.json"}
    missing = sorted(required - entries.keys())
    plot_names = {Path(name).name for name in entries if name.startswith("evaluation_test/")}
    for suffix in ("PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"):
        if suffix not in plot_names and "Box" + suffix not in plot_names:
            missing.append("evaluation_test/" + suffix)
    for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        if name not in plot_names:
            missing.append("evaluation_test/" + name)
    info["missing_evidence"] = missing
    info["evidence_complete"] = (state["status"] in {"COMPLETED_200_EPOCHS", "EARLY_STOPPED_PATIENCE"}
                                 and info["test_status"] == "PASSED" and not omitted and not missing)
    entries["PACKAGE_INFO.json"] = (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = [dict(path=n, bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for n, v in sorted(entries.items())]
    entries["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()

    class CappedBuffer(io.BytesIO):
        def write(self, data):
            require(self.tell() + len(data) <= MAX_BYTES, "Light package exceeds 32 MB; no archive delivered")
            return super().write(data)

    buffer = CappedBuffer()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in sorted(entries.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    with partial.open("xb") as f:
        f.write(buffer.getvalue())
    verify_archive(partial)
    os.link(partial, output)  # atomically refuse an output that appeared during packaging
    partial.unlink()
    digest = sha256(output)
    with Path(str(output) + ".sha256").open("x", encoding="utf-8") as f:
        f.write(digest + "  " + output.name + "\n")
    result = dict(path=str(output), sha256=digest, bytes=output.stat().st_size, files=len(manifest), **info)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.plan, args.output)
