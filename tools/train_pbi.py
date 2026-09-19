"""PBI's explicit plan/start/resume lifecycle; never launches as a side effect of preflight."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from pbi_common import (ROOT, MODEL_DIR, VARIANTS, BASE_COMMIT, SOURCE_SHA256, require,
                        sha256, write_json, runtime, verify_model, build_training_model, is_added)
from ultralytics import RTDETR
from ultralytics.data.utils import check_det_dataset, img2label_paths
from pbi_checkpoint import PBICheckpointTrainer, require_checkpoint_policy, optimizer_param_names
from ultralytics.utils import YAML

DEFAULT_MAIN = Path(os.environ.get("PBI_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
IDENTITY_FIELDS = {"model", "name", "project", "save_dir", "data"}


def git_head():
    head = subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(),
                                    "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(len(head) == 40 and all(c in "0123456789abcdef" for c in head), "Invalid full Git identity")
    return head


def server_environment():
    """Report the specified target rather than certify a different local GPU stack."""
    import sys
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    observed = dict(python=f"{sys.version_info.major}.{sys.version_info.minor}",
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=gpu)
    checks = dict(python=observed["python"] == "3.10", torch=str(torch.__version__) == "2.1.2+cu121",
                  cuda=torch.version.cuda == "12.1", gpu=gpu is not None and "4090" in gpu)
    return dict(status="PASSED" if all(checks.values()) else "PENDING", observed=observed, checks=checks)


def code_identity():
    """Content identity survives the final commit without rewriting old test evidence."""
    files = list((ROOT / "tools").glob("*pbi*.py")) + list((ROOT / "docs/pbi").glob("*.yaml"))
    files += list((ROOT / "tools").glob("*pbi*.sh"))
    files += [ROOT / "docs/pbi/parent_dataset_identity.json", ROOT / "docs/pbi/environment.sh"]
    files += list((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files += list(MODEL_DIR.glob("*.yaml"))
    files += [ROOT / "tools" / name for name in ("init_c19_lif_v1.py", "init_lif_down.py", "lif_down_topology.py",
              "c19_lif_v1_data.py", "c19_lif_v1_probe.py", "c19_lif_v1_diagnostic.py", "c19_lif_v1_cutoff.py")]
    files += [MODEL_DIR / item[0] for item in VARIANTS.values()]
    files += [ROOT / "ultralytics-main/ultralytics" / name for name in (
        "nn/modules/pbi.py", "nn/modules/__init__.py", "nn/tasks.py", "nn/autobackend.py",
        "engine/trainer.py", "models/rtdetr/train.py", "models/rtdetr/val.py",
        "nn/modules/cbr.py", "nn/modules/lif_down.py")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(set(files)) if p.is_file()}


def recipe(variant, init, main=DEFAULT_MAIN, data=None, parent_args=None):
    """Use all fields of the inspected successful parent, replacing identity only."""
    archived = ROOT / "docs/pbi/parent_args.yaml"
    require(archived.is_file(), "Missing inspected successful parent recipe archive")
    parent = YAML.load(archived)
    require(len(parent) == 109, "The complete 109-field successful recipe is required")
    if parent_args:
        actual = YAML.load(parent_args)
        require(actual == parent, "Actual successful parent args differ from the archived authority")
    main, init = Path(main).resolve(), Path(init).resolve()
    name = VARIANTS[variant][2]
    target = dict(parent)
    target.update(model=str(init), data=str(Path(data or main / "configs/crack_autodl.yaml").resolve()),
                  project=str(main / "runs/c_series"), name=name,
                  save_dir=str(main / "runs/c_series" / name))
    require(parent["epochs"] == 200 and parent["batch"] == 16 and parent["imgsz"] == 640 and
            parent["nbs"] == 64 and parent["amp"] is True and parent["freeze"] is None and
            parent["optimizer"] == "AdamW" and parent["lr0"] == .0005 and parent["exist_ok"] is False,
            "Archive is not the required successful 200-epoch online-augmentation recipe")
    differences = [{"field": k, "parent": parent[k], "target": target[k], "changed": parent[k] != target[k]}
                   for k in sorted(parent)]
    require({r["field"] for r in differences if r["changed"]} <= IDENTITY_FIELDS, "Recipe mutation")
    return target, differences


def dataset_identity(data):
    """Fingerprint actual split image paths and label bytes; never re-split or run inference."""
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "PBI requires one crack class")
    root = Path(resolved["path"]).resolve()
    result = {"config_sha256": sha256(data), "root": str(root),
              "names": {str(k): v for k, v in resolved["names"].items()}, "splits": {}}
    for split, expected in (("train", 6048), ("val", 1728), ("test", 864)):
        locations = resolved.get(split)
        require(locations, "Missing split " + split)
        files = []
        for location in locations if isinstance(locations, list) else [locations]:
            location = Path(location)
            if location.is_dir():
                files += [p.resolve() for p in location.rglob("*") if p.suffix.lower() in
                          {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
            elif location.is_file():
                for line in location.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        p = Path(line.strip())
                        files.append((location.parent / p if not p.is_absolute() else p).resolve())
        files = sorted(files)
        require(len(files) == len(set(files)) == expected, "Unexpected/duplicate image inventory for " + split)
        identities, labels, boxes = [], [], 0
        for image, label in zip(files, img2label_paths([str(p) for p in files])):
            require(image.is_file(), "Missing image " + str(image))
            label = Path(label)
            require(label.is_file(), "Missing label " + str(label))
            rows = [r.split() for r in label.read_text(encoding="utf-8").splitlines() if r.strip()]
            for row in rows:
                require(len(row) == 5 and float(row[0]) == 0 and
                        all(__import__("math").isfinite(float(v)) for v in row) and
                        all(0 <= float(v) <= 1 for v in row[1:]), "Invalid crack label " + str(label))
            boxes += len(rows)
            identities.append(image.relative_to(root).as_posix())
            labels.append(label.relative_to(root).as_posix() + ":" + sha256(label))
        result["splits"][split] = dict(images=len(files), boxes=boxes,
            paths_sha256=hashlib.sha256("\n".join(identities).encode()).hexdigest(),
            labels_sha256=hashlib.sha256("\n".join(labels).encode()).hexdigest())
    historical = ROOT / "docs/pbi/parent_dataset_identity.json"
    require(historical.is_file(), "Missing successful-parent actual dataset identity archive")
    authority = json.loads(historical.read_text(encoding="utf-8"))
    for split, identity in result["splits"].items():
        expected = authority[split]
        for target_key, parent_key in (("images", "images"), ("boxes", "boxes"),
                                       ("paths_sha256", "split_paths_sha256"),
                                       ("labels_sha256", "label_inventory_sha256")):
            require(identity[target_key] == expected[parent_key], "Dataset differs from historical parent: " + split + "." + target_key)
    return result


def checkpoint(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def verify_initialization(path, variant):
    ckpt = checkpoint(path)
    require(ckpt.get("epoch") == -1 and all(ckpt.get(k) is None for k in ("optimizer", "scaler", "ema")),
            "Start requires a fresh controlled initialization without learned training state")
    metadata = ckpt.get("pbi_provenance", {})
    require(metadata.get("variant") == variant and metadata.get("source_sha256") == SOURCE_SHA256 and
            metadata.get("base_commit") == BASE_COMMIT, "Controlled initialization identity missing/mismatched")
    source = Path(metadata.get("source", ""))
    require(source.is_file() and sha256(source) == SOURCE_SHA256, "Original public source is absent or changed")
    verify_model(ckpt["model"], variant, zero=True)
    return metadata


def optimizer_coverage(model, optimizer):
    ids = Counter(id(p) for group in optimizer.param_groups for p in group["params"])
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    require(set(ids) == expected and all(v == 1 for v in ids.values()), "Optimizer drops/duplicates trainable parameters")
    require(all(p.requires_grad for p in model.parameters()), "Unexpected frozen common/new parameter")
    return {"trainable_tensors": len(expected), "every_parameter_exactly_once": True,
            "new": {n: ids[id(p)] for n, p in model.named_parameters() if is_added(n)}}


def disable_oom_retry(trainer):
    # Preserve B16 even in the native first-epoch auto-recovery branch.
    trainer._oom_retries = 3


def _finite(value):
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_pass(row, name):
    require(isinstance(row, dict) and row.get("allclose") is True, "Comparison failed/missing: " + name)
    require(_finite(row.get("max_abs")) and row["max_abs"] >= 0, "Nonfinite metric: " + name)
    for key in ("relative_L2", "exceeded_fraction", "exceed_fraction"):
        if key in row:
            require(_finite(row[key]) and row[key] >= 0, "Invalid metric: " + name + "." + key)
    if "atol" in row or "rtol" in row:
        require(row.get("atol") == 2e-5 and row.get("rtol") == 2e-4, "Tolerance changed: " + name)


def _equivalence_pass(report, name):
    require(report.get("status") in {"PASSED", "PRECISION_NOTE"}, "Missing equivalence: " + name)
    required = {"scale_0", "scale_1", "scale_2", "projection_0", "projection_1", "projection_2",
                "encoder_features", "candidate_scores"}
    if ".fusion" in name:
        required |= {"pbi_input", "pbi_u", "pbi_v", "pbi_output", "pbi_residual"}
    continuous = report.get("continuous", {})
    require(set(continuous) == required, "Incomplete continuous path: " + name)
    for key, row in continuous.items():
        _metric_pass(row, name + "." + key)
    first, second = report.get("candidate_indices_a"), report.get("candidate_indices_b")
    require(isinstance(first, list) and first and isinstance(second, list) and second,
            "Missing candidate selection evidence: " + name)
    key = "native_outputs" if first == second else "fixed_candidate_diagnostic_only"
    outputs = report.get(key, {})
    needed = {"boxes", "scores", "raw_boxes", "raw_scores"}
    if first != second:
        needed |= {"selected_features", "reference_boxes"}
    require(set(outputs) == needed, "Missing selected/output path: " + name)
    for field, row in outputs.items():
        _metric_pass(row, name + "." + field)


def _controlled_pass(report, variant):
    from pbi_common import PBI_KEYS
    require(report.get("variant") == variant and report.get("source_sha256") == SOURCE_SHA256 and
            report.get("base_commit") == BASE_COMMIT and report.get("seed") == 42 and
            report.get("source_nc") == 80 and report.get("target_nc") == 80,
            "Controlled source/class/seed identity changed")
    require(report.get("new_parameters") == 24576 and set(report.get("NEW_TRAINABLE", [])) == set(PBI_KEYS),
            "PBI added parameter identity/count changed")
    for key in ("variant_new_state_exact", "public_constructor_equal", "no_parameter_storage_sharing"):
        require(report.get(key) is True, "Controlled initialization failed: " + key)
    for key in ("MISSING", "UNEXPECTED", "SHAPE_MISMATCH", "NEW_BUFFER"):
        require(report.get(key) == [], "Controlled initialization mismatch/missing: " + key)
    common = report.get("COMMON", [])
    require(common and all(row.get("equal") is True for row in common), "Missing/exact public initialization audit")
    rebuilt = report.get("native_nc1_rebuild", {})
    require(rebuilt.get("native_get_model") is True and rebuilt.get("parent_nc1_public_exact") is True and
            rebuilt.get("parent_initial_values_preserved") is True and rebuilt.get("COMMON") and
            len(rebuilt.get("ALLOWED_CLASS_ADAPTATION", [])) == 9, "Native nc1 reconstruction audit missing")
    for key in ("MISSING", "UNEXPECTED", "SHAPE_MISMATCH", "NEW_BUFFER"):
        require(rebuilt.get(key) == [], "Native nc1 state mismatch: " + key)


def _math_pass(report):
    require(report.get("tolerance") == {"atol": 2e-5, "rtol": 2e-4}, "Math tolerance changed")
    rng = report.get("rng_and_initialization", {})
    require(rng.get("status") == "PASSED" and rng.get("new_buffers") == [], "PBI RNG/state audit incomplete")
    for key in ("constructor_preserves_cpu_cuda_rng", "original_conv_state_and_later_cpu_rng_preserved",
                "xavier_gain1_exact", "independent_input_projections", "no_storage_sharing"):
        require(rng.get(key) is True, "Missing PBI initialization evidence: " + key)

    def compared(row, name):
        require(isinstance(row, dict) and row.get("raw_allclose") is True and row.get("finite") is True and
                _finite(row.get("max_abs")) and row["max_abs"] >= 0, "Explicit PBI comparison missing/failed: " + name)

    rows = report.get("devices", [])
    require(len(rows) == 2 and {d.get("device") for d in rows} == {"cpu", "cuda"}, "Math CPU/CUDA coverage missing")
    required = {"W1.weight", "W2.weight", "Wo.weight"}
    for device in rows:
        require(device.get("status") == "PASSED" and device.get("new_trainable_parameters") == 24576,
                "Math device PENDING/FAILED")
        shapes = device.get("shapes", [])
        require([r.get("shape") for r in shapes] == [[1, 256, 80, 80], [2, 256, 9, 11], [1, 256, 1, 1]] and
                all(r.get("initial_identity_exact") is True for r in shapes), "PBI identity/shape coverage incomplete")
        compared(device.get("nonzero_formula"), "nonzero formula")
        gradients = device.get("nonzero_gradients", {})
        require(set(gradients) == required | {"input"}, "Explicit three-projection/input gradient evidence missing")
        for key, row in gradients.items():
            compared(row, "gradient " + key)
        require(_finite(device.get("nonzero_residual_max")) and device["nonzero_residual_max"] > 0,
                "Nonzero PBI branch evidence missing")
        compared(device.get("nonzero_wrapper_fusion"), "learned wrapper fusion")
        require(device["nonzero_wrapper_fusion"].get("pbi_calls") == 1, "Fused PBI missing/duplicated")
        steps = device.get("gradient_steps", [])
        require(len(steps) == 2, "Missing two-step PBI startup")
        for index, step in enumerate(steps):
            norms = step.get("gradient_norms", {})
            require(_finite(step.get("loss")) and set(norms) == required and
                    all(_finite(v) and v >= 0 for v in norms.values()), "Nonfinite/missing mathematical gradients")
            require(norms["Wo.weight"] > 0 and (all(norms[k] == 0 for k in ("W1.weight", "W2.weight"))
                    if index == 0 else all(v > 0 for v in norms.values())), "PBI startup gradient contract failed")
        if device["device"] == "cuda":
            compared(device.get("native_amp"), "native AMP")
            compared(device.get("cuda_half"), "CUDA explicit half")
            steps = device.get("native_amp_optimizer_steps", [])
            require(len(steps) == 2 and all(s.get("effective_update") is True and _finite(s.get("loss")) and
                    s.get("scale_after", 0) >= s.get("scale_before", 1) > 0 for s in steps),
                    "Native GradScaler effective mathematical updates missing")
    structure = report.get("structure_and_counts", {})
    require(structure.get("status") == "PASSED" and structure.get("variant_initial_values_exact") is True and
            structure.get("cross_variant_no_storage_sharing") is True, "Two-variant initialization proof missing")
    variants = structure.get("variants", {})
    require(set(variants) == set(VARIANTS), "Both PBI configurations must be checked")
    from pbi_common import PARAM_COUNTS, PBI_KEYS
    hashes = structure.get("variant_initial_hashes", {})
    require(set(hashes) == set(VARIANTS) and hashes["pbi_v1"] == hashes["cbr_lif_pbi_v1"] and
            set(hashes["pbi_v1"]) == set(PBI_KEYS), "Variant PBI initialization hash mismatch")
    for variant, observed in variants.items():
        counts = observed.get("parameters", {})
        require(observed.get("status") == "PASSED" and observed.get("nc80_public_exact") is True and
                observed.get("full_model_initial_equality") is True and observed.get("pbi_calls") == 1,
                "Complete initialized-model equality missing: " + variant)
        require((counts.get("unfused"), counts.get("fused")) == tuple(PARAM_COUNTS[variant]) and
                counts.get("added") == 24576 and counts["unfused"] - counts.get("parent_unfused", 0) == 24576,
                "Parameter count contract changed: " + variant)
        trace = observed.get("node_trace", [])
        require(len(trace) == 27 and trace[17].get("source") == 5 and trace[17].get("type") == "PBIConv" and
                trace[17].get("shapes") == [[1, 256, 80, 80]], "PBI graph placement evidence missing")


def strict_gate(preflight_path, checks_path, variant, init, data):
    """Full, current-tree contract. Partial resume diagnostics can never authorize training."""
    from pbi_acceptance import CONTRACT_VERSION, evaluate_mode
    from pbi_common import PBI_KEYS
    require(preflight_path and checks_path, "start/resume require explicit capacity and full engineering reports")
    capacity = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    checks = json.loads(Path(checks_path).read_text(encoding="utf-8"))
    current_identity, current_runtime, init_hash = code_identity(), runtime(), sha256(init)

    def fresh(report, kind):
        require(report.get("report_kind") == kind and report.get("contract_version") == CONTRACT_VERSION,
                "Wrong/old report kind or acceptance contract: " + kind)
        require(report.get("status") in {"PASSED", "PRECISION_NOTE"} and not report.get("error") and
                not report.get("pending") and not report.get("failed"), "Incomplete report: " + kind)
        require(report.get("code_identity") == current_identity, "Code/config differs from tested tree: " + kind)
        for key in ("python", "torch", "cuda", "gpu", "ultralytics", "worktree"):
            require(key in report.get("runtime", {}) and report["runtime"][key] == current_runtime.get(key),
                    "Report must run in actual training environment: " + kind + "." + key)

    fresh(capacity, "native_capacity")
    fresh(checks, "full_preflight_engineering")
    require(capacity.get("git_head") == checks.get("git_head") == git_head(), "Preflight full HEAD differs")
    require(server_environment()["status"] == "PASSED" and
            capacity.get("server_environment") == server_environment(), "Specified server environment not verified")
    require(capacity.get("status") == "PASSED" and capacity.get("variant") == variant and
            capacity.get("init_sha256") == init_hash, "Capacity init/variant/status changed")
    require(checks.get("variant") == variant and checks.get("initialization_sha256") == init_hash,
            "Engineering initialization/variant changed")
    require(capacity.get("dataset_identity") == dataset_identity(data), "Actual data path/label identity changed")
    for split, recorded in capacity["dataset_identity"]["splits"].items():
        for key, check_key in (("images", "images"), ("boxes", "boxes"),
                               ("paths_sha256", "split_paths_sha256"),
                               ("labels_sha256", "label_inventory_sha256")):
            require(checks.get("data_identity", {}).get(split, {}).get(check_key) == recorded[key],
                    "Full engineering/capacity dataset differs: " + split + "." + key)
    c = capacity.get("capacity", {})
    require(c.get("batch") == 16 and c.get("imgsz") == 640 and c.get("AMP") is True and
            2 <= c.get("effective_updates", 0) <= c.get("observed_batches", 0) <= 16 and
            capacity.get("checkpoint_policy") == "optimizer_fp32_v1", "Missing B16/640 native AMP capacity evidence")
    expected_recipe, _ = recipe(variant, init, data=data)
    for key, value in expected_recipe.items():
        if key not in IDENTITY_FIELDS:
            require(capacity.get("actual_recipe", {}).get(key) == value and
                    capacity.get("recipe", {}).get(key) == value, "Capacity recipe changed: " + key)
    require(capacity.get("optimizer_coverage", {}).get("every_parameter_exactly_once") is True,
            "Capacity optimizer coverage missing")
    steps = capacity.get("steps", [])
    require(sum(s.get("effective") is True for s in steps) == c["effective_updates"], "Capacity update evidence missing")
    for name in PBI_KEYS:
        require(any(s.get("effective") is True and s.get("gradients", {}).get(name, {}).get("finite") is True and
                    _finite(s["gradients"][name].get("norm")) and s["gradients"][name]["norm"] > 0 for s in steps),
                "Capacity PBI gradient startup missing: " + name)

    prerequisite_hashes = {}
    prerequisites = {}
    for name, kind in (("initialization", "controlled_initialization_audit"), ("math", "pbi_math_audit")):
        reference = checks.get("prerequisites", {}).get(name, {})
        path = Path(reference.get("path", ""))
        require(path.is_absolute() and path.is_file() and sha256(path) == reference.get("sha256"),
                "Missing/changed bound prerequisite: " + name)
        report = json.loads(path.read_text(encoding="utf-8"))
        fresh(report, kind)
        require(report.get("status") == "PASSED", "Prerequisite not PASSED: " + name)
        prerequisites[name] = report
        prerequisite_hashes[name + "_report_sha256"] = reference["sha256"]
    initialization = prerequisites["initialization"]
    _controlled_pass(initialization, variant)
    require(initialization.get("output_sha256") == init_hash and initialization.get("reload_exact") is True,
            "Initialization output/reload changed")
    reconstruction = initialization.get("training_reconstruction", {})
    require(reconstruction.get("status") == "PASSED" and reconstruction.get("optimizer_steps") == 0 and
            all(reconstruction.get(k) is True for k in ("native_setup_model", "native_get_model",
                "actual_model_train_dispatch", "public_and_added_values_exact")), "Native initialization reconstruction missing")
    _math_pass(prerequisites["math"])
    _controlled_pass(checks.get("controlled_initialization", {}), variant)
    wiring = checks.get("wiring_640", {})
    require(wiring.get("status") == "PASSED" and wiring.get("nodes") == 27 and
            wiring.get("pbi_after_complete_projection") is True and wiring.get("concat") == [16, 17] and
            wiring.get("p3_downsample") == 20 and wiring.get("decoder_from") == [19, 22, 25], "Wiring contract incomplete")
    counts = checks.get("parameters", {})
    for kind in ("fused", "unfused"):
        require(counts.get("target", {}).get(kind, 0) - counts.get("parent", {}).get(kind, 0) == 24576,
                "PBI added parameter count changed: " + kind)
    _equivalence_pass(checks.get("initial_equivalence", {}), "initial")
    require(checks.get("tolerances") == {"atol": 2e-5, "rtol": 2e-4}, "Engineering tolerance changed")
    modes = {}
    precision_notes = []
    for mode in ("cpu_fp32", "cuda_fp32", "cuda_native_amp"):
        observed = checks.get("devices", {}).get(mode, {})
        lifecycle, controls = observed.get("lifecycle", {}), observed.get("live_controls", {})
        require(all(item.get("contract_version") == CONTRACT_VERSION for item in
                    (lifecycle, controls.get("parent", {}), controls.get("pbi", {}))),
                "Missing current-contract nested evidence: " + mode)
        acceptance = evaluate_mode(mode, lifecycle, controls.get("parent", {}), controls.get("pbi", {}))
        require(acceptance["status"] in {"PASSED", "PRECISION_NOTE"}, "Restore/trajectory evidence blocked: " +
                mode + " " + repr(acceptance.get("failures")))
        require(observed.get("acceptance") == acceptance, "Stored mode summary differs from raw evidence: " + mode)
        require(observed.get("optimizer_every_parameter_once") is True and observed.get("effective_updates", 0) >= 3,
                "Network optimizer/startup evidence missing: " + mode)
        steps = observed.get("steps", [])
        require(sum(s.get("skipped") is False for s in steps) == observed["effective_updates"] and
                all(_finite(s.get("loss")) for s in steps), "Invalid network update evidence: " + mode)
        require(any(not s.get("skipped", True) and set(s.get("pbi_grad_norms", {})) == set(PBI_KEYS) and
                    all(_finite(v) and v > 0 for v in s["pbi_grad_norms"].values()) for s in steps),
                "Network PBI gradient startup missing: " + mode)
        for key in ("state_dict_exact", "full_model_exact", "native_get_model_nonzero_preserved", "learned_nc80_to1"):
            require(lifecycle.get(key) is True, "Missing full lifecycle evidence: " + mode + "." + key)
        _metric_pass(lifecycle.get("ema_own_copy"), mode + ".ema_own_copy")
        fusion = observed.get("fusion", {})
        require(fusion.get("status") == "PASSED" and fusion.get("pbi_calls") == 1 and
                fusion.get("pbi_state_exact") is True, "Fusion evidence missing: " + mode)
        _equivalence_pass(fusion.get("strict_fp32", {}), mode + ".fusion")
        if mode != "cpu_fp32":
            half = fusion.get("cuda_half", {})
            require(half.get("status") in {"PASSED", "PRECISION_NOTE"} and half.get("pbi_calls") == 1 and
                    half.get("pbi_state_exact") is True and half.get("nonzero_residual") is True and
                    half.get("strict_fp32_continuous_passed") is True, "CUDA half lifecycle missing")
            for label in ("unfused", "fused"):
                reference = half.get("explicit_formula", {}).get(label, {})
                require(reference.get("exact") is True and reference.get("nonzero_residual") is True,
                        "Half explicit PBI formula/state evidence missing")
                _metric_pass(reference.get("comparison"), mode + ".half." + label)
            _metric_pass(half.get("identical_input_pbi"), mode + ".half.identical_input")
            require(set(half.get("pbi_same_precision", {})) ==
                    {"pbi_input", "pbi_u", "pbi_v", "pbi_output", "pbi_residual"}, "Half localization incomplete")
            raw = half.get("continuous_and_candidate_diagnostic", {})
            require(raw.get("continuous") and raw.get("candidate_indices_a") and raw.get("candidate_indices_b"),
                    "Half raw candidate/continuous evidence missing")
            for key, row in {**half["pbi_same_precision"], **raw["continuous"]}.items():
                require(isinstance(row.get("allclose"), bool) and _finite(row.get("max_abs")),
                        "Half raw finite comparison missing: " + key)
            if half["status"] == "PRECISION_NOTE":
                require(bool(half.get("note")), "Unexplained half precision difference")
                precision_notes.append(mode + ".cuda_half")
        mode_status = "PRECISION_NOTE" if acceptance["status"] == "PRECISION_NOTE" or mode + ".cuda_half" in precision_notes else "PASSED"
        require(observed.get("status") == mode_status, "Mode summary omits a precision note: " + mode)
        modes[mode] = acceptance
    expected_status = "PRECISION_NOTE" if precision_notes or any(v["status"] == "PRECISION_NOTE" for v in modes.values()) else "PASSED"
    require(checks["status"] == expected_status, "Full engineering summary disagrees with mode evidence")
    return dict(contract_version=CONTRACT_VERSION, status=expected_status, git_head=git_head(),
                code_identity=current_identity, variant=variant, init_sha256=init_hash,
                precision_notes=precision_notes,
                capacity_report_sha256=sha256(preflight_path), checks_report_sha256=sha256(checks_path),
                **prerequisite_hashes, modes=modes,
                components={name: "PASSED" for name in ("initialization", "mathematics", "wiring",
                    "gradient_startup", "checkpoint_correctness", "fusion", "cuda_half",
                    "native_B16_640_AMP_capacity", "source_runtime_data_init_identity")},
                trajectory_repeatability={name: result["trajectory"]["status"] for name, result in modes.items()})


def plan(args):
    recipe_args, differences = recipe(args.variant, args.init, args.main, args.data, args.parent_args)
    from pbi_acceptance import CONTRACT_VERSION
    return {"variant": args.variant, "runtime": runtime(), "code_identity": code_identity(),
            "git_head": git_head(), "server_environment": server_environment(),
            "contract_version": CONTRACT_VERSION,
            "args": recipe_args, "recipe_differences": differences,
            "formal_training": "NOT_STARTED", "final_test": "NOT_RUN",
            "checkpoint_policy": "optimizer_fp32_v1", "checkpoint_ema_dtype": "float16"}


def run(args):
    require(torch.cuda.is_available(), "CUDA unavailable; formal training not launched")
    require(ROOT.resolve() != args.main.resolve() and (ROOT / ".git").is_file(), "Use the independent PBI worktree")
    require(not subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(),
                                        "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                        text=True).strip(), "Commit tested source before formal start/resume")
    info = plan(args)
    train_args = info["args"]
    require(Path(train_args["data"]).is_file(), "Missing actual dataset config")
    info["initialization"] = verify_initialization(args.init, args.variant)
    info["init_sha256"] = sha256(args.init)
    info["gates"] = strict_gate(args.preflight, args.checks, args.variant, args.init, train_args["data"])
    out = ROOT / "outputs/pbi" / args.variant / "training"
    run_dir = Path(train_args["save_dir"])
    if args.mode == "start":
        require(not run_dir.exists(), "Existing formal run protected: " + str(run_dir))
        require(not out.exists(), "Existing launch metadata protected: " + str(out))
        reservation = run_dir.with_name(run_dir.name + ".pbi.lock")
        reservation.parent.mkdir(parents=True, exist_ok=True)
        reservation.mkdir(exist_ok=False)
        write_json(reservation / "owner.json", dict(variant=args.variant, worktree=str(ROOT),
                   pid=os.getpid(), init_sha256=info["init_sha256"], code_identity=info["code_identity"]))
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", info)
        selected, saved = args.init, None
    else:
        require(args.checkpoint and args.checkpoint.is_file(), "resume requires --checkpoint from unfinished training")
        require(out.is_dir(), "Missing original PBI launch identity")
        original = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        require(original["variant"] == args.variant and original["code_identity"] == code_identity() and
                original["init_sha256"] == info["init_sha256"], "Resume launch source/initialization changed")
        saved = checkpoint(args.checkpoint)
        # Validate source precision before native loading can upcast old moments.
        info["resume_checkpoint_policy"] = require_checkpoint_policy(saved)
        require(0 <= saved.get("epoch", -1) < 199 and saved.get("optimizer") is not None and
                saved.get("ema") is not None and saved.get("scaler") is not None,
                "Checkpoint is completed/stripped or lacks native optimizer/scaler/epoch/EMA")
        require(args.checkpoint.resolve().parent == (run_dir / "weights").resolve(), "Checkpoint is from another run")
        for k, value in train_args.items():
            if k not in {"model", "resume"}:
                require(saved["train_args"].get(k) == value, "Resume recipe differs: " + k)
        verify_model(saved["ema"], args.variant, zero=False)
        selected = args.checkpoint
        train_args["resume"] = str(selected.resolve())
    state_path = out / "state.json"
    state = {"status": "RUNNING", "pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat(),
             "completed_epochs": saved["epoch"] + 1 if saved else 0, "final_eval": "NOT_STARTED", "mode": args.mode}
    write_json(state_path, state)
    training_variant = args.variant

    class RecordingTrainer(PBICheckpointTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model, audit = build_training_model(cfg, weights, self.data, training_variant)
            write_json(out / ("resume_rebuild.json" if saved else "start_rebuild.json"), audit)
            return model

        def final_eval(self):
            state["final_eval"] = "RUNNING"
            write_json(state_path, state)
            try:
                result = super().final_eval()
                state["final_eval"] = "COMPLETED"
                return result
            except BaseException:
                state["final_eval"] = "FAILED"
                raise
            finally:
                write_json(state_path, state)

    def setup(trainer):
        verify_model(trainer.model, args.variant, zero=saved is None)
        require(trainer.amp and trainer.args.batch == 16 and trainer.args.imgsz == 640, "Native AMP/B16/640 changed")
        require(type(trainer.optimizer) is torch.optim.AdamW, "Optimizer changed")
        actual = vars(trainer.args)
        ignored = {"model", "resume"} if saved else set()
        changes = {k: [v, actual.get(k)] for k, v in train_args.items() if k not in ignored and actual.get(k) != v}
        require(not changes, "Actual native recipe changed: " + repr(changes))
        if saved:
            require(optimizer_param_names(trainer.model, trainer.optimizer) ==
                    saved["pbi_checkpoint"]["optimizer_param_names"],
                    "Native optimizer parameter-name/group-order mapping differs")
            require(trainer.start_epoch == saved["epoch"] + 1 and trainer.scaler.state_dict() == saved["scaler"] and
                    trainer.ema.updates == saved["updates"], "Native resume did not restore epoch/scaler/EMA updates")
            require(len(trainer.optimizer.state) == len(saved["optimizer"]["state"]), "Native optimizer restore mismatch")
            restored = trainer.optimizer.state_dict()
            require(restored["param_groups"] == saved["optimizer"]["param_groups"],
                    "Native optimizer group hyperparameters/order differ")
            require(set(restored["state"]) == set(saved["optimizer"]["state"]),
                    "Native optimizer state IDs differ")
            for key, parameters in saved["optimizer"]["state"].items():
                require(set(restored["state"][key]) == set(parameters), "Native optimizer state fields differ")
                for field, value in parameters.items():
                    observed = restored["state"][key][field]
                    equal = (observed.dtype == value.dtype and torch.equal(observed.detach().cpu(), value.detach().cpu())
                             if torch.is_tensor(value) else observed == value)
                    require(equal, "Native optimizer state restore mismatch: %s.%s" % (key, field))
            saved_ema = {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                         for k, v in saved["ema"].state_dict().items()}
            for name, model in (("model", trainer.model), ("EMA", trainer.ema.ema)):
                actual_state = model.state_dict()
                require(set(actual_state) == set(saved_ema) and
                        all(torch.equal(v.detach().cpu(), saved_ema[k]) for k, v in actual_state.items()),
                        "Native " + name + " weight/buffer restore mismatch")
        write_json(out / "setup.json", {"coverage": optimizer_coverage(trainer.model, trainer.optimizer),
                   "native_resume": bool(saved), "start_epoch": trainer.start_epoch, "runtime": runtime()})
        YAML.save(out / "actual_train_args.yaml", actual)

    def epoch_complete(trainer):
        # The native final_eval also emits on_fit_epoch_end; epoch itself is not advanced there.
        state["completed_epochs"] = int(trainer.epoch) + 1
        write_json(state_path, state)

    try:
        model = RTDETR(str(selected))
        model.add_callback("on_train_batch_start", disable_oom_retry)
        model.add_callback("on_train_start", setup)
        model.add_callback("on_fit_epoch_end", epoch_complete)
        model.train(trainer=RecordingTrainer, **train_args)
        state["status"] = "COMPLETED_200" if state["completed_epochs"] >= 200 else "EARLY_STOPPED"
    except KeyboardInterrupt:
        state["status"] = "INTERRUPTED"
        raise
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["finished"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "start", "resume", "status"))
    p.add_argument("--variant", choices=VARIANTS, default="cbr_lif_pbi_v1")
    p.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    p.add_argument("--data", type=Path)
    p.add_argument("--init", type=Path)
    p.add_argument("--parent-args", type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--checks", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, help="Optional plan JSON; exclusive creation")
    return p


def main():
    args = parser().parse_args()
    args.init = args.init or ROOT / "weights" / (args.variant + "_controlled_init.pt")
    if args.mode == "status":
        path = ROOT / "outputs/pbi" / args.variant / "training/state.json"
        print(path.read_text(encoding="utf-8") if path.is_file() else json.dumps({"status": "NOT_STARTED", "final_test": "NOT_RUN"}))
    elif args.mode == "plan":
        result = plan(args)
        if args.output:
            require(not args.output.exists(), "Existing plan protected")
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
