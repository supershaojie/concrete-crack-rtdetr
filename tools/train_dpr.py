"""DPR formal training admission, unchanged parent recipe, and explicit resume.

Preflight never calls start. Formal start accepts only current, complete server
evidence. A completed training run is never silently restarted after final val.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

from init_dpr import (ROOT, MAIN, VARIANTS, SOURCE_SHA256, build_training_model,
                      require, sha256, verify_model, write_json)
import torch
import ultralytics
from ultralytics import RTDETR
from ultralytics.utils import YAML
from dpr_checkpoint import DPRCheckpointTrainer, optimizer_param_names, require_checkpoint_policy
from dpr_data import dataset_identity

from dpr_acceptance import CONTRACT, FULL_SCHEMA, policy as acceptance_policy, scope as capability_scope
BASE_COMMIT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
SERVER_MAIN = Path(os.environ.get("DPR_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))
LIF_SHA = "26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7"
CBR_SHA = "d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787"


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def atomic_json(path, payload):
    def encode(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    temporary.write_text(json.dumps(encode(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def paths(variant=MAIN):
    name = VARIANTS[variant][2]
    return dict(init=ROOT / "weights" / (variant + "_controlled_init.pt"),
                source=SERVER_MAIN / "weights/rtdetr_r18_lite_imagenet_backbone_init.pt",
                data=SERVER_MAIN / "configs/crack_autodl.yaml", meta=ROOT / "outputs/dpr" / variant,
                project=SERVER_MAIN / "runs/c_series", run=SERVER_MAIN / "runs/c_series" / name)


def environment():
    expected = (ROOT / "ultralytics-main/ultralytics/__init__.py").resolve()
    actual = Path(ultralytics.__file__).resolve()
    require(actual == expected, "Imported ultralytics is outside this DPR worktree")
    info = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), ultralytics=str(actual),
                platform=platform.system(), executable=sys.executable,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    info["server_environment"] = (info["platform"] == "Linux" and info["python"].startswith("3.10.")
                                    and info["torch"] == "2.1.2+cu121"
                                    and info["cuda_available"] and "4090" in info["gpu"])
    return info


def code_identity():
    """Bind all production Python, model YAML, DPR tools, and parent contracts."""
    files = list((ROOT / "ultralytics-main/ultralytics").rglob("*.py"))
    files += list((ROOT / "ultralytics-main/ultralytics/cfg").rglob("*.yaml"))
    files += list((ROOT / "tools").glob("*dpr*"))
    files += [ROOT / "tools/c19_lif_v1_data.py", ROOT / "tools/init_c19_lif_v1.py",
              ROOT / "tools/init_lif_down.py", ROOT / "docs/dpr/parent_args.yaml",
              ROOT / "docs/dpr/parent_dataset_inventory.json", ROOT / "docs/dpr/acceptance_review_bb768787.md"]
    entries = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(files)) if p.is_file()}
    for name, expected in (("lif_down.py", LIF_SHA), ("cbr.py", CBR_SHA)):
        raw = (ROOT / "ultralytics-main/ultralytics/nn/modules" / name).read_bytes().replace(b"\r\n", b"\n")
        require(hashlib.sha256(raw).hexdigest() == expected, "Original module changed: " + name)
    return dict(head=git("rev-parse", "HEAD"), files=entries,
                content_sha256=hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest())


def recipe(variant=MAIN, initialized=None, data=None):
    p = paths(variant)
    parent = YAML.load(ROOT / "docs/dpr/parent_args.yaml")
    require(len(parent) == 109, "Parent recipe must contain its complete 109-field snapshot")
    target = dict(parent)
    target.update(model=str(Path(initialized or p["init"]).resolve()),
                  data=str(Path(data).resolve()) if data is not None else p["data"].as_posix(),
                  name=VARIANTS[variant][2],
                  project=p["project"].as_posix(), save_dir=p["run"].as_posix())
    allowed = {"model", "data", "project", "name", "save_dir"}
    diff = [dict(field=k, parent=parent[k], candidate=target[k], changed=parent[k] != target[k],
                 permitted_identity_change=k in allowed) for k in parent]
    require(all(not r["changed"] or r["permitted_identity_change"] for r in diff), "Training recipe changed")
    return target, diff


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evidence_file(record):
    require(isinstance(record, dict) and {"path", "sha256"} <= set(record), "Missing evidence file identity")
    path = Path(record["path"])
    require(path.is_file() and sha256(path) == record["sha256"], "Evidence file missing or changed: " + str(path))
    return load_json(path)


def strict_gate(variant, source, initialized, data, report_path):
    """Revalidate substantive leaves; a passed marker or partial audit is insufficient."""
    from check_dpr import validate_math_report, TARGET, DPRConvNormLayer, FP32_ATOL, FP32_RTOL, HALF_ATOL, HALF_RTOL

    accepted = {"PASSED", "PRECISION_NOTE"}
    new_names = {TARGET + "." + name for name in DPRConvNormLayer.parameter_names}

    def rebuild_evidence(entry, actual_entry=False, adaptation=False):
        require(isinstance(entry, dict) and entry.get("status") == "PASSED" and entry.get("variant") == variant
                and entry.get("native_get_model") is True and entry.get("target_nc") == 1
                and entry.get("parent_nc1_public_exact") is True and entry.get("learned_dpr_preserved") is True,
                "Actual native Trainer reconstruction evidence incomplete")
        require(set(entry.get("NEW_TRAINABLE", [])) == new_names and entry.get("NEW_BUFFER") == []
                and all(entry.get(key) == [] for key in ("MISSING", "UNEXPECTED", "SHAPE_MISMATCH")),
                "Trainer reconstruction inventory differs")
        rows = entry.get("COMMON", [])
        require(rows and len(rows) == entry.get("parent_nc1_public_states") and all(row.get("equal") is True
                and len(row.get("sha256", "")) == 64 for row in rows), "Trainer common-state equality evidence incomplete")
        if actual_entry:
            require(entry.get("actual_train_entry") is True and entry.get("optimizer_steps") == 0
                    and entry.get("batches") == 0, "Actual train entry reconstruction was not isolated")
        if adaptation:
            rows = entry.get("ALLOWED_CLASS_ADAPTATION", [])
            require(len(rows) == 9 and all(row.get("exact_parent_nc1") is True for row in rows),
                    "Nine nc80 to nc1 adaptation states were not audited")

    def tensor_evidence(entry, label, exact=False, allow_raw_failure=False):
        require(isinstance(entry, dict) and entry.get("tensors") and entry.get("finite") is True
                and entry.get("metadata_errors") == [], label + ": missing/nonfinite tensor evidence")
        rows = entry["tensors"]
        require(all(row.get("finite") is True for row in rows.values()), label + ": nonfinite tensor")
        if exact:
            require(entry.get("exact") is True and entry.get("status") == "PASSED"
                    and all(row.get("equal") is True for row in rows.values()), label + ": not exact")
        if not allow_raw_failure:
            require(entry.get("raw_allclose") is True and entry.get("failed_tensors") == []
                    and all(row.get("raw_allclose") is True for row in rows.values()), label + ": raw threshold failed")
        else:
            require(set(entry.get("failed_tensors", [])) == {name for name, row in rows.items() if not row.get("raw_allclose")},
                    label + ": failed tensor list inconsistent")

    def inference_leaf(entry, label, half=False):
        require(isinstance(entry, dict) and entry.get("status") in accepted, "Inference leaf incomplete: " + label)
        atol, rtol = (HALF_ATOL, HALF_RTOL) if half else (FP32_ATOL, FP32_RTOL)
        # Backend raw equality is a hard gate even if a report also has details.
        if label.endswith(("autobackend_saved_best", "autobackend_half")):
            from dpr_diagnostics import validate_backend_record
            validate_backend_record(entry, atol, rtol)
            return
        if "details" in entry:
            rows = entry["details"]
            for name in ("target", "P3", "P4", "P5", "encoder_features", "candidate_scores"):
                value = rows.get(name, {})
                require(value.get("raw_allclose") is True and value.get("finite") is True and
                        value.get("atol") == atol and value.get("rtol") == rtol, "Inference continuous path failed: " + label + ":" + name)
            output = rows.get("output", {})
            require(output.get("finite") is True, "Nonfinite detector output: " + label)
            if entry["status"] == "PASSED":
                require(output.get("raw_allclose") is True, "Detector output raw comparison failed: " + label)
            else:
                replay = rows.get("fixed_candidate_replay_output", {})
                require(rows.get("candidate_index_changes", 0) > 0 and replay.get("raw_allclose") is True
                        and replay.get("finite") is True and replay.get("atol") == atol and replay.get("rtol") == rtol,
                        "Candidate precision note lacks continuous/fixed-candidate evidence: " + label)
        elif "comparison" in entry:
            value = entry["comparison"]
            require(value.get("raw_allclose") is True and value.get("finite") is True and
                    value.get("atol") == atol and value.get("rtol") == rtol, "Backend output comparison failed: " + label)
            require(entry.get("deployed") is True, "Backend did not perform DPR deployment: " + label)
        elif label.endswith("ema_retains_nonzero"):
            require(entry.get("updates", 0) > 0, "EMA nonzero update evidence missing")
        else:
            raise RuntimeError("Unrecognized/incomplete inference evidence: " + label)

    report = load_json(report_path)
    require(report.get("contract") == CONTRACT and report.get("report_kind") == "full_preflight_engineering",
            "Only this complete DPR preflight contract is eligible")
    require(report.get("report_schema") == FULL_SCHEMA and
             report.get("statistics_version") == "allclose_right_reference_v2", "Stale preflight evidence format")
    from dpr_acceptance import validate_scope, validate_b, validate_half
    require(report.get("policy") == acceptance_policy(), "Unapproved/unknown R1 policy")
    validate_scope(report.get("capability_scope"))
    require(report.get("variant") == variant, "Wrong preflight variant")
    require(report.get("status") in accepted, "Preflight contains failed or pending checks")
    env = environment()
    require(env["server_environment"], "Formal start requires existing Python3.10 / torch2.1.2+cu121 / RTX4090 server")
    require(report.get("environment") == env, "Preflight runtime differs from this server environment")
    identity = code_identity()
    require(report.get("code_identity") == identity, "Preflight HEAD/code/config contents are stale")
    require(report.get("source_stable_during_checks") is True and report.get("code_identity_at_end") == identity,
            "Preflight source changed during checks")
    from supplement_dpr import verify as verify_supplement
    from types import SimpleNamespace
    supplement = report.get("supplement", {})
    evidence_file(supplement)
    verify_supplement(supplement["path"], SimpleNamespace(variant=variant, source=source, initialized=initialized, data=data), server=True)
    require(not git("status", "--porcelain", "--untracked-files=no"), "Tracked source edits block formal start")
    require(sha256(source) == SOURCE_SHA256 == report.get("source_sha256"), "Public source identity mismatch")
    require(sha256(initialized) == report.get("initialization_sha256"), "Controlled initialization changed")
    require(dataset_identity(Path(data)) == report.get("dataset_identity"), "Dataset identity changed")
    current_recipe, _ = recipe(variant, initialized, data)
    require(report.get("recipe") == current_recipe, "Complete parent recipe or run identity changed")
    init = evidence_file(report.get("initialization"))
    require(init.get("status") == "PASSED" and init.get("variant") == variant and init.get("reload_exact") is True,
            "Initialization evidence incomplete")
    require(init.get("output_sha256") == sha256(initialized) and init.get("source_sha256") == SOURCE_SHA256,
            "Initialization evidence binds different source or bytes")
    require(init.get("new_parameters") == 3072 and init.get("public_constructor_equal") is True
            and set(init.get("NEW_TRAINABLE", [])) == new_names and init.get("NEW_BUFFER") == []
            and all(init.get(key) == [] for key in ("MISSING", "UNEXPECTED", "SHAPE_MISMATCH")),
            "Controlled initialization state inventory incomplete")
    require(set(init.get("new_initial_values", {})) == new_names and
            all(row.get("nonzero") == 0 for row in init["new_initial_values"].values()), "DPR initial state was not exactly zero")
    rebuild_evidence(init.get("native_trainer_rebuild"), adaptation=True)
    rebuild_evidence(init.get("train_entry_rebuild"), actual_entry=True, adaptation=True)
    common_state_names = {row["name"] for row in init["native_trainer_rebuild"]["COMMON"]}
    def complete_state_inventory(entry, label, candidate=True):
        expected = common_state_names | (new_names if candidate else set())
        names = set(entry.get("tensors", {}))
        for prefix in ("state.model.", "state.ema."):
            require({name[len(prefix):] for name in names if name.startswith(prefix)} == expected,
                    label + ": incomplete model/EMA tensor inventory")
    mathematical = evidence_file(report.get("mathematics"))
    validate_math_report(mathematical, variant)
    require(report.get("mathematics_verified") is True, "Required mathematical/structure evidence not verified")
    leaves = report.get("checks", {})
    required = {"capacity", "cpu_fp32", "cuda_fp32", "cuda_native_amp", "learned_inference"}
    require(required <= set(leaves), "Missing complete preflight check categories")
    capacity = leaves["capacity"]
    require(capacity.get("status") == "PASSED" and capacity.get("batch") == 16 and capacity.get("imgsz") == 640
            and capacity.get("native_amp") is True and 2 <= capacity.get("effective_updates", 0)
            and 0 < capacity.get("observed_batches", 0) <= 16 and capacity.get("gt_instances", 0) > 0
            and capacity.get("dn_seen") is True and capacity.get("optimizer_exact_coverage") is True,
            "Native B16/640/AMP bounded real-data capacity evidence incomplete")
    rebuild_evidence(capacity.get("actual_trainer_rebuild"), adaptation=True)
    optimizer_names = [name for group in capacity.get("optimizer_names", []) for name in group]
    require(optimizer_names and len(optimizer_names) == len(set(optimizer_names)) and new_names <= set(optimizer_names),
            "Capacity optimizer parameter names are missing/duplicated")
    steps = capacity.get("steps", [])
    require(sum(row.get("effective_update") is True for row in steps) == capacity["effective_updates"],
            "Capacity effective updates disagree with actual step evidence")
    losses = capacity.get("losses", [])
    require(len(losses) == capacity["observed_batches"] and all(isinstance(value, (int, float)) and
            -float("inf") < value < float("inf") for value in losses), "Capacity real loss evidence missing/nonfinite")
    require(capacity.get("peak_memory_bytes", 0) > 0 and capacity.get("elapsed_seconds", 0) > 0,
            "Capacity resource measurements missing")
    for mode in ("cpu_fp32", "cuda_fp32", "cuda_native_amp"):
        item = leaves[mode]
        require(item.get("status") in {"PASSED", "PRECISION_NOTE"}, "Lifecycle mode incomplete: " + mode)
        a = item.get("A", {})
        for name in ("saved_live_optimizer_exact", "saved_byte_restoration_exact", "optimizer_names_groups_exact",
                     "no_shared_storage", "same_gradient_complete_state_exact", "nonzero_dpr_preserved",
                     "state_dict_reload_exact", "full_model_reload_exact", "native_trainer_rebuild_exact"):
            require(a.get(name) is True, "Checkpoint A hard gate missing/failed: " + mode + ":" + name)
        require(a.get("status") == "PASSED", "Checkpoint A did not pass")
        require(a.get("policy", {}).get("policy") == "optimizer_fp32_v1" and
                a["policy"].get("original_tensors_inspected_before_cast") is True and
                a["policy"].get("fp32_moment_tensors", 0) > 0, "Original saved FP32 moment audit missing")
        tensor_evidence(a.get("saved_live_optimizer_comparison"), mode + ":saved live FP32 optimizer", exact=True)
        if mode == "cpu_fp32":
            require(a.get("cpu_native_continuation_exact") is True, "CPU actual native saved-state continuation differs")
        tensor_evidence(a.get("restoration"), mode + ":saved byte restoration", exact=True)
        tensor_evidence(a.get("replay"), mode + ":same gradient replay", exact=True)
        complete_state_inventory(a["restoration"], mode + ":saved byte restoration")
        complete_state_inventory(a["replay"], mode + ":same gradient replay")
        require(len(a.get("replay_steps", [])) == 2 and all(row.get("effective_update") is True for row in a["replay_steps"]),
                "Actual effective replay steps missing")
        if mode == "cuda_native_amp":
            require(a.get("overflow_skip_exact") is True and a.get("effective_step_exact") is True,
                    "Native AMP effective and overflow replay incomplete")
            require(len(a.get("overflow_rows", [])) == 2 and all(row.get("effective_update") is False
                    and row.get("scale_after", 0) < row.get("scale_before", 0) for row in a["overflow_rows"]),
                    "Actual native AMP overflow steps missing")
        b = item.get("B", {})
        assessment = validate_b(item, mode)
        require(item.get("assessment") == assessment, "B8 stored assessment disagrees with independently validated facts")
        require(item.get("binding", {}).get("code_identity") == identity and item["binding"].get("session") == report.get("output"),
                "B evidence belongs to another full preflight")
        require(b.get("atol") == 2e-5 and b.get("rtol") == 2e-4,
                 "Independent backward B evidence missing")
        require(b.get("parent") and b.get("candidate"), "Missing matched live parent/candidate controls")
        for control in ("parent", "candidate"):
            evidence = b[control]
            require(all(evidence.get(key) is True for key in ("initial_exact", "storage_independent", "forward_exact",
                    "loss_exact", "exact_metadata_buffers")), "Independent backward prerequisites failed: " + mode + ":" + control)
            tensor_evidence(evidence.get("forward"), mode + ":" + control + ":forward", exact=True)
            for key in ("state", "gradients"):
                tensor_evidence(evidence.get(key), mode + ":" + control + ":" + key,
                                allow_raw_failure=assessment["status"] == "EXPLAINED_BACKWARD_VARIATION")
            complete_state_inventory(evidence["state"], mode + ":" + control, candidate=control == "candidate")
            require(len(evidence.get("steps", [])) == 2 and all(row.get("effective_update") is True and
                    row.get("finite_gradients") is True for row in evidence["steps"]), "Live independent updates missing")
        # R1 B1--B8 replace the old subset/raw-added-gradient restrictions only
        # after full causal and functional proof. Raw B is never rewritten.
        require(all(item.get("gradients", {}).get(k, {}).get("finite_nonzero") is True
                    for k in ("dpr_cd", "dpr_hd", "dpr_vd", "dpr_ad", "conv.weight")), "Missing actual DPR/W gradients")
        updates = item.get("updates", {})
        require(updates.get("status") == "PASSED" and 2 <= updates.get("effective_updates", 0)
                and 0 < updates.get("observed_batches", 0) <= 16 and all(updates.get("parameter_changes", {}).get(name) is True
                for name in new_names | {TARGET + ".conv.weight"}), "Actual DPR/W updates incomplete")
    inference = leaves["learned_inference"]
    for key in ("cpu_fp32", "cuda_fp32", "ema", "fold_norm_retained", "native_fuse",
                "fuse_idempotent", "deploy_reload", "autobackend"):
        require(inference.get(key, {}).get("status") in {"PASSED", "PRECISION_NOTE"}, "Learned inference gate incomplete: " + key)
    required_inference = {"training_state_reload", "training_full_reload", "ema_retains_nonzero", "dpr_fold_only",
                          "deploy_state_reload", "deploy_full_reload_idempotent", "native_fuse", "native_fuse_again",
                          "autobackend_saved_best", "native_fused_saved_reload"}
    for mode in ("cpu_fp32", "cuda_fp32"):
        details = report.get("learned_inference_details", {}).get(mode, {})
        require(details.get("status") in accepted and required_inference <= set(details.get("checks", {})),
                "Actual full-model learned inference evidence missing: " + mode)
        require(str(details.get("device", "")).startswith("cuda" if mode == "cuda_fp32" else "cpu")
                and details.get("dtype") == "torch.float32", "Learned inference device/precision mismatch: " + mode)
        for key in required_inference:
            inference_leaf(details["checks"][key], mode + ":" + key)
        require(details["checks"]["autobackend_saved_best"].get("norm_retained") is True, "Native inference removed original norm")
    half = inference["cuda_half"]
    from dpr_acceptance import validate_original_half_finiteness
    validate_original_half_finiteness(half)
    require(str(half.get("device", "")).startswith("cuda") and half.get("dtype") == "float16"
            and half.get("atol") == HALF_ATOL and half.get("rtol") == HALF_RTOL,
            "Explicit CUDA half evidence/threshold missing")
    for key in ("half_deploy_reload", "autobackend_half"):
        inference_leaf(half.get("checks", {}).get(key), "cuda_half:" + key, half=True)
    validate_half(report.get("half_R1", {}))
    return dict(status="PASSED", contract=CONTRACT, variant=variant, head=identity["head"],
                policy=acceptance_policy(), capability_scope=capability_scope(),
                report=str(Path(report_path).resolve()), report_sha256=sha256(report_path),
                initialization_sha256=sha256(initialized), source_sha256=sha256(source), environment=env)


def disable_oom_retry(trainer):
    # Native first-epoch fallback would silently halve B16. Preserve the recipe.
    trainer._oom_retries = 3


class DPRTrainer(DPRCheckpointTrainer):
    """The same class is used by formal train, native capacity, and restore audits."""
    variant = MAIN

    def get_model(self, cfg=None, weights=None, verbose=True):
        model, audit = build_training_model(cfg, weights, self.data, variant=self.variant)
        self.dpr_rebuild_audit = audit
        return model

    def _load_checkpoint_state(self, ckpt):
        require_checkpoint_policy(ckpt)
        expected_names = ckpt["DPR_checkpoint"]["optimizer_param_names"]
        require(optimizer_param_names(self.model, self.optimizer) == expected_names,
                "Checkpoint optimizer order/name binding differs from reconstructed model")
        super()._load_checkpoint_state(ckpt)


class DPRSingleTrainer(DPRTrainer):
    variant = "dpr_v1"


def trainer_class(variant):
    return DPRTrainer if variant == MAIN else DPRSingleTrainer


def invoke_train(variant, initialized, overrides, callbacks=None):
    model = RTDETR(str(initialized))
    model.add_callback("on_train_batch_start", disable_oom_retry)
    for name, fn in (callbacks or {}).items():
        model.add_callback(name, fn)
    return model.train(trainer=trainer_class(variant), **overrides)


def revoke_permit(variant):
    permit = paths(variant)["meta"] / "start_permit.json"
    if permit.exists():
        os.replace(str(permit), str(permit.with_name("start_permit.revoked." + timestamp() + ".json")))


def start_or_resume(args, resume=False):
    revoke_permit(args.variant)
    p = paths(args.variant)
    initialized, source, data = args.initialized or p["init"], args.source or p["source"], args.data or p["data"]
    report_path = args.report or Path(load_json(p["meta"] / "latest_preflight.json")["report"])
    gate = strict_gate(args.variant, source, initialized, data, report_path)
    overrides, diff = recipe(args.variant, initialized, data)
    checkpoint_path = initialized
    if resume:
        require(args.checkpoint is not None, "Resume requires an explicit unfinished DPR last.pt")
        checkpoint_path = args.checkpoint.resolve()
        require(checkpoint_path == (p["run"] / "weights/last.pt").resolve(), "Resume only the exact experiment run last.pt")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require_checkpoint_policy(ckpt)
        require(ckpt["DPR_checkpoint"]["purpose"] == "formal_training", "Diagnostic/capacity checkpoint cannot resume formal training")
        require(ckpt["DPR_checkpoint"]["scaler_enabled"] is True, "Formal resume must retain native CUDA AMP scaler")
        require(ckpt["DPR_checkpoint"].get("code_head") == gate["head"],
                "Resume checkpoint was produced by a different or unresolved worktree HEAD")
        require(0 <= ckpt.get("epoch", -1) < overrides["epochs"] - 1, "Initialization/completed checkpoint cannot resume")
        require(ckpt.get("train_args", {}).get("name") == VARIANTS[args.variant][2], "Wrong checkpoint variant/run")
        verify_model(ckpt["ema"], args.variant, zero=False)
        original = dict(ckpt["train_args"])
        for key, expected in overrides.items():
            if key not in {"model", "resume", "save_dir"}:
                require(original.get(key) == expected, "Resume recipe differs: " + key)
        overrides["resume"] = True
    else:
        require(not p["run"].exists(), "Formal output already exists; inspect it and explicitly resume the unfinished checkpoint")
    record = dict(gate, created=timestamp(), action="resume" if resume else "start", recipe=overrides, recipe_diff=diff,
                  completed_epochs=0, final_eval="NOT_RUN", status="STARTING")
    launch = p["meta"] / ("launch_" + timestamp() + ".json")
    atomic_json(p["meta"] / "start_permit.json", record)
    atomic_json(launch, record)

    def completed(trainer):
        record["completed_epochs"] = int(trainer.epoch) + 1
        record["status"] = "TRAINING"
        atomic_json(launch, record)

    try:
        invoke_train(args.variant, checkpoint_path, overrides, {"on_fit_epoch_end": completed})
        record.update(status="COMPLETED", final_eval="COMPLETED", exit_code=0)
    except BaseException as error:
        record.update(status="FAILED", error=repr(error), final_eval="FAILED_OR_NOT_REACHED", exit_code=3)
        raise
    finally:
        atomic_json(launch, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("environment", "plan", "verify", "start", "resume"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=MAIN)
    for key in ("source", "initialized", "data", "report", "checkpoint"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    p = paths(args.variant)
    if args.action == "environment":
        print(json.dumps(environment(), indent=2))
    elif args.action == "plan":
        config, differences = recipe(args.variant, args.initialized, args.data)
        print(json.dumps(dict(variant=args.variant, recipe=config, recipe_diff=differences,
                              formal_training="NOT_STARTED", final_test="NOT_RUN"), indent=2))
    elif args.action == "verify":
        require(args.report is not None, "verify requires --report")
        result = strict_gate(args.variant, args.source or p["source"], args.initialized or p["init"],
                             args.data or p["data"], args.report)
        print(json.dumps(result, indent=2))
    else:
        start_or_resume(args, resume=args.action == "resume")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
