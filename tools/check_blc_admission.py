"""Fast synthetic FIXTURE tests only. Never evidence of server admission."""
from copy import deepcopy
import ast
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import blc_admission as admission
import blc_admission_identity as identity

ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64


def difference(exact=True):
    return dict(max_abs=0 if exact else .86, relative_L2=0 if exact else .1,
                allclose=exact, finite=True, atol=2e-5, rtol=2e-4)


def fixture():
    """Invented metadata for contract tests; no outputs from a model."""
    branch = dict(calls=1, completed=True, increment_norms=[.01])
    def step(i):
        return dict(batch=i, scale_before=65536., scale_after=65536., optimizer_state_step=float(i),
                    scaler_skipped=False, effective_update=True, all_gradients_finite=True,
                    new_gradient_norms={k: .1 if i > 1 or "Wo." in k else 0. for k in admission.KEYS},
                    parameter_delta={k: .01 if i > 1 or "Wo." in k else 0. for k in admission.KEYS})
    inits = {}
    audit = dict(NEW_BUFFER=[], MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[], NEW_TRAINABLE=sorted(admission.KEYS),
                 COMMON=dict(count=552, unequal_count=0, unequal_samples=[], inventory_sha256=HASH),
                 native_get_model=True, nc=1, classification_parent_candidate_exact=True,
                 new_state_exact=True, ALLOWED_CLASS_ADAPTATION=[])
    for variant in admission.VARIANTS:
        inits[variant] = dict(fixture=True, status="PASSED", phase="init-preflight", variant=variant,
            context=dict(source_sha256=HASH),
            current_source_audit=dict(status="PASSED", reload_exact=True, variant=variant, source_sha256=HASH,
                                      constructor=deepcopy(audit), transfer=deepcopy(audit), nc1_rebuild=deepcopy(audit)),
            structure=dict(status="PASSED", exact_saved_state=True, topology_parameters_and_zero_initialization=True, tensors=556),
            actual_train_api=dict(status="PASSED", actual_RTDERT_train_api=True, all_states_exact=True,
                                  batch_count=0, audit=deepcopy(audit)))
    paths = {}
    for mode in ("FP32", "AMP", "half"):
        paths[mode] = dict(status="PRECISION_NOTE" if mode == "FP32" else "PENDING", stage="complete",
            dtype="torch.float16" if mode == "half" else "torch.float32", autocast=mode == "AMP",
            device="cuda:0", input_shape=[1, 3, 160, 160], input_sha256=HASH,
            independent_state=dict(exact=True, shared_storage=False, tensors=364),
            direct_blc=deepcopy(branch), backend_blc=deepcopy(branch), autobackend=dict(status="PASSED", **difference()),
            fusion=dict(status="PRECISION_NOTE" if mode == "FP32" else "PENDING", natural=difference(False),
                        features={str(k): difference(k != "scores") for k in (5, 6, 7, "scores")},
                        fixed_indices_diagnostic_only=difference(), forward_count=3,
                        differing_candidate_positions=136, same_candidate_set=False, indices_sha256=[HASH, HASH]))
    resumed_step = step(3)
    resumed_step["batch"] = 1
    report = dict(fixture=True, status="PENDING", phase="preflight", variant=admission.VARIANTS[0], stage="complete",
        context=dict(runtime=dict(torch="2.1.2+cu121")),
        formal_init_untouched=True, batch_budget=16, validation_batch_budget=1, total_attempted_training_batches=3,
        capacity=dict(status="PASSED", batch=16, imgsz=640, AMP=True, attempted_batches=2, effective_updates=2,
                      batches=[dict(batch=i, loss=2., gt=5, amp=True) for i in (1, 2)], steps=[step(1), step(2)]),
        native_resume=dict(status="PASSED", attempted_batches=1, effective_updates=1, epoch=1,
                           batches=[dict(loss=1., gt=3)], steps=[resumed_step],
                           restored_state=dict(epoch=True, optimizer_moments_steps=True, scaler=True, ema=True,
                                               updates=True, shared_storage=False)),
        bounded_checkpoint=dict(epoch=0, incomplete_epoch=True, native_resume_next_epoch=1, sha256=HASH),
        native_half_ema_epoch_val=dict(status="PASSED", batches=1, half=True, metrics={"metrics/mAP50(B)": .01}),
        lifecycle=dict(status="PENDING", stage="complete", state_dict_and_full_model_exact=True,
                       serialization=dict(state_dict=difference(), full_model=difference()),
                       live_blc=deepcopy(branch), nonzero_increment_norm=.01, live_vs_file_quantization=difference(False),
                       native_storage=dict(model=None, ema_dtype="torch.float16",
                                           optimizer="native convert_optimizer_state_dict_to_fp16", updates=2), paths=paths))
    return report, inits


def amp_backoff_fixture():
    """SYNTHETIC: three overflow skips, two real updates, then a restored update."""
    report, inits = fixture()
    capacity = report["capacity"]
    skips = []
    scale = admission.NATIVE_SCALER["init_scale"]
    for batch in (1, 2, 3):
        skips.append(dict(batch=batch, scale_before=scale, scale_after=scale * .5, optimizer_state_step=0.,
                          scaler_skipped=True, effective_update=False, all_gradients_finite=False,
                          new_gradient_norms={k: .1 if "Wo." in k else 0. for k in admission.KEYS},
                          parameter_delta={k: 0. for k in admission.KEYS}))
        scale *= .5
    for i, step in enumerate(capacity["steps"], 4):
        step.update(batch=i, scale_before=scale, scale_after=scale)
    capacity.update(steps=skips + capacity["steps"], attempted_batches=5,
                    batches=[dict(batch=i, loss=2., gt=5, amp=True) for i in range(1, 6)])
    report["native_resume"]["steps"][0].update(scale_before=scale, scale_after=scale)
    report["total_attempted_training_batches"] = 6
    report["lifecycle"]["native_storage"]["updates"] = 5  # Native EMA ticks also on scaler skips.
    return report, inits


class AdmissionTests(unittest.TestCase):
    def test_three_backoffs_then_updates_and_restore(self):
        raw, inits = amp_backoff_fixture()
        before = deepcopy(raw)
        deployed = types.ModuleType("deployed_admission_FIXTURE_only")
        old_source = identity.tree(str(ROOT), identity.DEPLOYED_ADMISSION)["tools/blc_admission.py"]
        exec(compile(old_source, "3906d102:tools/blc_admission.py", "exec"), deployed.__dict__)
        old_result = deployed.evaluate(raw, inits)["training_admission"]
        self.assertEqual(old_result["status"], "FAILED")
        self.assertEqual(old_result["errors"], [message for i in range(3) for message in (
            f"capacity.steps[{i}]: gradients finite", f"capacity.steps[{i}].optimizer_state_step")])
        result = admission.evaluate(raw, inits)
        self.assertEqual(result["training_admission"]["errors"], [])
        self.assertEqual(result["training_admission"]["status"], "PASSED")
        self.assertEqual(result["fusion_diagnostic"]["status"], "PENDING")
        phases = result["training_admission"]["optimizer_attempts"]
        self.assertEqual([x["classification"] for x in phases["capacity"]["attempts"]],
                         ["AMP_BACKOFF"] * 3 + ["EFFECTIVE_UPDATE"] * 2)
        self.assertEqual(phases["capacity"]["amp_backoff_count"], 3)
        self.assertEqual(phases["capacity"]["effective_updates"], 2)
        self.assertEqual(phases["native_resume"]["effective_updates"], 1)
        self.assertEqual(phases["native_resume"]["initial_state"], phases["capacity"]["final_state"])
        self.assertEqual(phases["native_resume"]["attempts"][0]["expected_previous_optimizer_state_step"], 2.)
        captured = io.StringIO()
        with redirect_stdout(captured):
            admission.print_result("SYNTHETIC FIXTURE", result)
        self.assertIn("scale=65536.0 -> 32768.0 optimizer_state_step=0.0", captured.getvalue())
        self.assertIn("training_admission=PASSED; fusion_diagnostic=PENDING", captured.getvalue())
        self.assertEqual(raw, before)

    def test_backoff_forever_is_insufficient(self):
        raw, inits = amp_backoff_fixture()
        raw["capacity"].update(steps=raw["capacity"]["steps"][:3], batches=raw["capacity"]["batches"][:3],
                               attempted_batches=3, effective_updates=0)
        raw["total_attempted_training_batches"] = 4
        result = admission.evaluate(raw, inits)["training_admission"]
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(any("insufficient" in e for e in result["errors"]))
        self.assertEqual(result["optimizer_attempts"]["capacity"]["amp_backoff_count"], 3)

    def test_skip_contradictions_and_missing_fields(self):
        changes = [
            ("optimizer_state_step", 1.), ("scale_after", 100.), ("scale_before", 100.),
            ("scaler_skipped", False), ("effective_update", True), ("all_gradients_finite", True),
            ("batch", 0), ("batch", 6),
            ("parameter_delta", {k: .1 for k in admission.KEYS}),
            ("new_gradient_norms", {k: float("inf") for k in admission.KEYS}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                raw, inits = amp_backoff_fixture()
                raw["capacity"]["steps"][0][key] = value
                result = admission.evaluate(raw, inits)["training_admission"]
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["optimizer_attempts"]["capacity"]["attempts"][0]["classification"], "INVALID")
        original, _ = amp_backoff_fixture()
        for key in original["capacity"]["steps"][0]:
            with self.subTest(missing=key):
                raw, inits = amp_backoff_fixture()
                del raw["capacity"]["steps"][0][key]
                self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")
        raw, inits = amp_backoff_fixture()
        del raw["capacity"]["steps"][0]["parameter_delta"]["model.5.blc.Wg.bias"]
        self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")

    def test_batch_loss_and_attempt_chronology(self):
        for mutation in ("loss_nan", "loss_inf_string", "gt_missing", "duplicate_batch", "out_of_order", "broken_scale"):
            with self.subTest(mutation=mutation):
                raw, inits = amp_backoff_fixture()
                if mutation == "loss_nan":
                    raw["capacity"]["batches"][0]["loss"] = float("nan")
                elif mutation == "loss_inf_string":
                    raw["capacity"]["batches"][0]["loss"] = "inf"
                elif mutation == "gt_missing":
                    del raw["capacity"]["batches"][0]["gt"]
                elif mutation in ("duplicate_batch", "out_of_order"):
                    raw["capacity"]["steps"][1]["batch"] = 1 if mutation == "duplicate_batch" else 0
                else:
                    raw["capacity"]["steps"][1].update(scale_before=65536., scale_after=32768.)
                self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")

    def test_applied_update_flags_and_nonfinite_rejection(self):
        for mutation in ("nonfinite_updated", "skip_with_finite", "wo_zero", "false_effective",
                         "step_static", "step_jump", "backoff_on_success", "wrong_torch"):
            with self.subTest(mutation=mutation):
                raw, inits = amp_backoff_fixture()
                step = raw["capacity"]["steps"][3]
                if mutation == "nonfinite_updated":
                    step["all_gradients_finite"] = False
                elif mutation == "skip_with_finite":
                    step["scaler_skipped"] = True
                elif mutation == "wo_zero":
                    step["parameter_delta"]["model.5.blc.Wo.weight"] = 0
                elif mutation == "false_effective":
                    step["effective_update"] = False
                elif mutation == "step_static":
                    step["optimizer_state_step"] = 0
                elif mutation == "step_jump":
                    step["optimizer_state_step"] = 2
                elif mutation == "backoff_on_success":
                    step["scale_after"] = step["scale_before"] * .5
                else:
                    raw["context"]["runtime"]["torch"] = "UNKNOWN"
                self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")

    def test_applied_without_wo_increment_is_not_backoff_or_effective(self):
        raw, inits = fixture()
        capacity = raw["capacity"]
        ineffective = deepcopy(capacity["steps"][0])
        ineffective.update(effective_update=False, parameter_delta={k: 0. for k in admission.KEYS})
        for step in capacity["steps"]:
            step["batch"] += 1
            step["optimizer_state_step"] += 1
        capacity.update(steps=[ineffective] + capacity["steps"], attempted_batches=3,
                        batches=[dict(batch=i, loss=2., gt=5, amp=True) for i in range(1, 4)])
        raw["native_resume"]["steps"][0]["optimizer_state_step"] = 4.
        raw["total_attempted_training_batches"] = 4
        raw["lifecycle"]["native_storage"]["updates"] = 3
        result = admission.evaluate(raw, inits)["training_admission"]
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["optimizer_attempts"]["capacity"]["applied_no_wo_change_count"], 1)
        self.assertEqual(result["optimizer_attempts"]["capacity"]["effective_updates"], 2)

    def test_resume_origin_and_resume_backoff(self):
        for mutation in ("zero_origin", "old_step", "reset_scale", "unproven_scaler", "unproven_optimizer"):
            with self.subTest(mutation=mutation):
                raw, inits = amp_backoff_fixture()
                resume = raw["native_resume"]
                if mutation in ("zero_origin", "old_step"):
                    resume["steps"][0]["optimizer_state_step"] = 1. if mutation == "zero_origin" else 2.
                elif mutation == "reset_scale":
                    resume["steps"][0].update(scale_before=65536., scale_after=65536.)
                else:
                    resume["restored_state"]["scaler" if mutation == "unproven_scaler" else "optimizer_moments_steps"] = False
                self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")
        raw, inits = amp_backoff_fixture()
        skip = deepcopy(raw["capacity"]["steps"][2])
        skip.update(batch=1, optimizer_state_step=2., scale_before=8192., scale_after=4096.)
        resumed = raw["native_resume"]
        resumed["steps"][0].update(batch=2, scale_before=4096., scale_after=4096.)
        resumed.update(steps=[skip] + resumed["steps"], attempted_batches=2,
                       batches=[dict(loss=1., gt=3)] * 2)
        raw["total_attempted_training_batches"] = 7
        result = admission.evaluate(raw, inits)["training_admission"]
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["optimizer_attempts"]["native_resume"]["amp_backoff_count"], 1)

    def test_source_scaler_configuration_and_scoped_history(self):
        source = identity.tree(str(ROOT), identity.SOURCE)
        trainer = ast.parse(source["ultralytics-main/ultralytics/engine/trainer.py"])
        calls = [n for n in ast.walk(trainer) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == "torch.cuda.amp.GradScaler"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, [])
        self.assertEqual([(kw.arg, ast.unparse(kw.value)) for kw in calls[0].keywords], [("enabled", "self.amp")])
        updates = [n for n in ast.walk(trainer) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.scaler.update"]
        self.assertTrue(updates)
        self.assertTrue(all(not c.args and not c.keywords for c in updates))
        deployed = identity.tree(str(ROOT), identity.DEPLOYED_ADMISSION)
        changed = [r["path"] for r in identity.changed_files(ROOT, identity.SOURCE, identity.DEPLOYED_ADMISSION)]
        identity.scope_check(source, deployed, changed)
        current = dict(deployed)
        for name in identity.AMP_FIX_FILES:
            path = ROOT / name
            if path.is_file() and identity.tracked_code(name):
                current[name] = path.read_bytes().replace(b"\r\n", b"\n")
        identity.amp_fix_scope_check(deployed, current, identity.AMP_FIX_FILES)
        identity.scope_check(source, current, identity.ALLOWED_FILES | identity.AMP_FIX_FILES,
                             identity.ALLOWED_FILES | identity.AMP_FIX_FILES)
        with self.assertRaisesRegex(ValueError, "Unknown AMP-fix"):
            identity.amp_fix_scope_check(deployed, current, {"tools/blc_preflight.py"})
        corrupted = dict(current)
        corrupted["ultralytics-main/ultralytics/engine/trainer.py"] += b"\n# FIXTURE outside scope\n"
        with self.assertRaisesRegex(ValueError, "protected source"):
            identity.amp_fix_scope_check(deployed, corrupted, identity.AMP_FIX_FILES)

    def test_finite_fusion_difference_admits_without_relabeling(self):
        raw, inits = fixture()
        before = deepcopy(raw)
        result = admission.evaluate(raw, inits)
        self.assertEqual(result["training_admission"]["errors"], [])
        self.assertEqual(result["training_admission"]["status"], "PASSED")
        self.assertEqual(result["fusion_diagnostic"]["status"], "PENDING")
        for mode in ("FP32", "AMP", "half"):
            self.assertEqual({k: v for k, v in result["fusion_diagnostic"]["modes"][mode].items()
                              if k != "original_path_status"}, raw["lifecycle"]["paths"][mode]["fusion"])
        self.assertEqual(raw, before)

    def test_required_updates_validation_resume_and_branch(self):
        for field in ("updates", "validation", "resume", "branch", "gradient", "serialization", "budget", "error", "failed_status"):
            with self.subTest(field=field):
                raw, inits = fixture()
                if field == "updates":
                    raw["capacity"]["effective_updates"] = 1
                elif field == "validation":
                    raw["native_half_ema_epoch_val"]["batches"] = 0
                elif field == "resume":
                    raw["native_resume"]["restored_state"]["scaler"] = False
                elif field == "branch":
                    raw["lifecycle"]["paths"]["AMP"]["backend_blc"]["calls"] = 0
                elif field == "gradient":
                    for s in raw["capacity"]["steps"]:
                        s["new_gradient_norms"]["model.5.blc.Wd.weight"] = 0
                elif field == "serialization":
                    raw["lifecycle"]["serialization"]["state_dict"]["max_abs"] = .000001
                elif field == "budget":
                    raw["total_attempted_training_batches"] = 17
                elif field == "error":
                    raw["probe_error"] = "FIXTURE RuntimeError"
                else:
                    raw["status"] = "FAILED"
                self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")

    def test_same_file_nonfinite_and_missing_diagnostic(self):
        for mode in ("FP32", "AMP", "half"):
            for fault in ("same_file", "nan", "inf_string", "not_run", "missing_scores", "missing_replay", "shared_storage"):
                with self.subTest(mode=mode, fault=fault):
                    raw, inits = fixture()
                    row = raw["lifecycle"]["paths"][mode]
                    if fault == "same_file":
                        row["autobackend"]["status"] = "FAILED"
                    elif fault == "nan":
                        row["fusion"]["features"]["scores"]["max_abs"] = float("nan")
                    elif fault == "inf_string":
                        row["fusion"]["fixed_indices_diagnostic_only"]["relative_L2"] = "inf"
                    elif fault == "not_run":
                        row["fusion"]["status"] = "NOT_RUN"
                    elif fault == "missing_scores":
                        del row["fusion"]["features"]["scores"]
                    elif fault == "missing_replay":
                        del row["fusion"]["fixed_indices_diagnostic_only"]
                    else:
                        row["independent_state"]["shared_storage"] = True
                    self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")

    def test_initialization_missing_or_inconsistent(self):
        for variant in admission.VARIANTS:
            raw, inits = fixture()
            inits[variant]["actual_train_api"]["audit"]["MISSING"] = ["FIXTURE.parameter"]
            self.assertEqual(admission.evaluate(raw, inits)["training_admission"]["status"], "FAILED")
        raw, _ = fixture()
        errors = admission.evaluate(raw, {})["training_admission"]["errors"]
        self.assertTrue(any("actual Trainer reconstruction" in e for e in errors))

    def test_context_fields_are_not_exempted(self):
        context = dict(variant="FIXTURE", code={"sha256": HASH}, runtime=dict(commit="old", python="FIXTURE"),
                       data=dict(inventory={"images": ["x"], "labels": HASH}, data_yaml={"names": {"0": "crack"}, "train": "images/train"},
                                 resolved_root="/fixture", data_sha256=HASH), recipe={"batch": 16},
                       source_sha256=HASH, init_sha256=HASH)
        for path in (("data", "inventory", "images"), ("data", "inventory", "labels"),
                     ("data", "data_yaml", "names"), ("data", "data_yaml", "train"),
                     ("data", "resolved_root"), ("data", "data_sha256"),
                     ("recipe", "batch"), ("init_sha256",), ("source_sha256",), ("runtime", "python")):
            changed = deepcopy(context)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = "MUTATED FIXTURE"
            with self.assertRaises(ValueError):
                identity.context_relation(context, changed)
        current = deepcopy(context)
        current["runtime"]["commit"] = "new"
        identity.context_relation(context, current)

    def test_raw_hash_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory(prefix="blc-admission-fixture-") as folder:
            path = Path(folder) / "raw.json"
            path.write_text('{"fixture": true}', encoding="utf-8")
            bindings = {key: identity.binding(path) for key in ("preflight", *admission.VARIANTS)}
            identity.verify_bindings(bindings)
            path.write_text('{"fixture": false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                identity.verify_bindings(bindings)
            path.write_text('{"fixture": true, "fixture": false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                admission.read_report(path)

    def test_real_local_init_schema_and_source_manifests_without_models(self):
        # Read already-existing LOCAL metadata only; this does not admit the server.
        for variant, name in zip(admission.VARIANTS, ("init-main.json", "init-single.json")):
            row = admission.read_report(ROOT / "docs/blc_v1/light_preflight_fix" / name)
            checks = admission.Checks()
            checks.initialization(row, variant)
            self.assertEqual(checks.errors, [])
            self.assertEqual(row["context"]["code"], identity.manifest(identity.tree(str(ROOT), identity.SOURCE)))
            relation = identity.evidence_proof(ROOT, row, row["context"], "init")
            self.assertEqual(relation["verified_source_revision"], identity.SOURCE)
        for revision in identity.INIT_SOURCES:
            identity.init_semantics(ROOT, revision)

    def test_exact_scope_rejects_computation_and_unknown_paths(self):
        before = identity.tree(str(ROOT), identity.SOURCE)
        after = dict(before)
        for name in identity.ALLOWED_FILES:
            p = ROOT / name
            if p.is_file() and identity.tracked_code(name):
                after[name] = p.read_bytes().replace(b"\r\n", b"\n")
        identity.scope_check(before, after, identity.ALLOWED_FILES)
        bad = dict(after)
        bad["tools/blc_server.py"] = bad["tools/blc_server.py"].replace(b"wrapper.train(trainer=BLCTrainer, **kwargs)",
                                                                      b"wrapper.train(trainer=BLCTrainer, batch=1)")
        with self.assertRaisesRegex(ValueError, "computation protected"):
            identity.scope_check(before, bad, identity.ALLOWED_FILES)
        with self.assertRaisesRegex(ValueError, "Unknown changed files"):
            identity.scope_check(before, after, {"tools/blc_preflight.py"})
        old = admission.read_report(ROOT / "docs/blc_v1/light_preflight_fix/init-main.json")
        altered = deepcopy(old)
        altered["context"]["code"]["files"]["tools/blc_preflight.py"] = HASH
        with self.assertRaisesRegex(ValueError, "source tree"):
            identity.evidence_proof(ROOT, altered, old["context"], "init")

    def test_revision_limit_and_current_manifest(self):
        def response(root, *args, **kwargs):
            if args[0] == "rev-parse":
                return b"FIXTURE_HEAD\n"
            if args[0] == "rev-list":
                if args[-1] == identity.DEPLOYED_ADMISSION:
                    return (identity.DEPLOYED_ADMISSION + " " + identity.SOURCE + "\n").encode()
                return b"FIXTURE_HEAD UNKNOWN_FUTURE_PARENT\n"
            self.fail("Unexpected Git call")
        with patch.object(identity, "git", side_effect=response):
            with self.assertRaisesRegex(ValueError, "exact SOURCE"):
                identity.current_proof(ROOT, {"fixture": True})
        def source_parent(root, *args, **kwargs):
            if args[0] == "rev-parse":
                return b"FIXTURE_HEAD\n"
            if args[0] == "rev-list":
                child = args[-1]
                parent = identity.SOURCE if child == identity.DEPLOYED_ADMISSION else identity.DEPLOYED_ADMISSION
                return (child + " " + parent + "\n").encode()
            if args[0] == "status":
                return b""
            self.fail("Unexpected Git call")
        with patch.object(identity, "git", side_effect=source_parent), patch.object(identity, "tree", return_value={}):
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                identity.current_proof(ROOT, {"code": {"fixture": True}})

    def test_metadata_reassess_and_start_gate_recompute(self):
        raw, inits = fixture()
        with tempfile.TemporaryDirectory(prefix="blc-admission-fixture-") as folder:
            root = Path(folder)
            src = root / "preflight.json"
            src.write_text(json.dumps(raw), encoding="utf-8")
            bindings = {key: identity.binding(src) for key in ("preflight", *admission.VARIANTS)}
            contexts = {v: {"fixture": True} for v in admission.VARIANTS}
            proof = {"fixture": True}
            common = types.ModuleType("blc_common")
            common.paths = lambda variant: {"evidence": root}
            common.stamp = lambda: "FIXTURE"
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(value, stream)
            common.write_json = write
            result = (raw, inits, bindings, contexts, proof, [])
            with patch.dict(sys.modules, {"blc_common": common}), patch.object(identity, "collect", return_value=result), redirect_stdout(io.StringIO()):
                output = admission.reassess(admission.VARIANTS[0], src)
                admission.require_admission(admission.VARIANTS[0])
                with self.assertRaises(FileExistsError):
                    admission.reassess(admission.VARIANTS[0], src)
                changed_contexts = {**contexts, "changed": True}
                with patch.object(identity, "collect", return_value=(raw, inits, bindings, changed_contexts, proof, [])):
                    with self.assertRaises(RuntimeError):
                        admission.require_admission(admission.VARIANTS[0])
                raw["native_half_ema_epoch_val"]["batches"] = 0
                with self.assertRaises(RuntimeError):
                    admission.require_admission(admission.VARIANTS[0])
                raw["native_half_ema_epoch_val"]["batches"] = 1
                saved = admission.read_report(output)
                saved["training_admission"]["status"] = "EDITED FIXTURE"
                Path(output).write_text(json.dumps(saved), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    admission.require_admission(admission.VARIANTS[0])
                src.write_text("{}", encoding="utf-8")
                with self.assertRaises(ValueError):
                    admission.require_admission(admission.VARIANTS[0])


if __name__ == "__main__":
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "safe.directory"
    os.environ["GIT_CONFIG_VALUE_0"] = ROOT.as_posix()
    unittest.main(verbosity=2)
