"""Fast synthetic FIXTURE tests only. Never evidence of server admission."""
from copy import deepcopy
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
        return dict(batch=i, scale_before=1024., scale_after=1024., optimizer_state_step=float(i),
                    scaler_skipped=False, effective_update=True, all_gradients_finite=True,
                    new_gradient_norms={k: .1 for k in admission.KEYS}, parameter_delta={k: .01 for k in admission.KEYS})
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
    report = dict(fixture=True, status="PENDING", phase="preflight", variant=admission.VARIANTS[0], stage="complete",
        formal_init_untouched=True, batch_budget=16, validation_batch_budget=1, total_attempted_training_batches=3,
        capacity=dict(status="PASSED", batch=16, imgsz=640, AMP=True, attempted_batches=2, effective_updates=2,
                      batches=[dict(loss=2., gt=5, amp=True)] * 2, steps=[step(1), step(2)]),
        native_resume=dict(status="PASSED", attempted_batches=1, effective_updates=1, epoch=1,
                           batches=[dict(loss=1., gt=3)], steps=[step(1)],
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


class AdmissionTests(unittest.TestCase):
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
                return b"FIXTURE_HEAD UNKNOWN_FUTURE_PARENT\n"
            self.fail("Unexpected Git call")
        with patch.object(identity, "git", side_effect=response):
            with self.assertRaisesRegex(ValueError, "single admission revision"):
                identity.current_proof(ROOT, {"fixture": True})
        def source_parent(root, *args, **kwargs):
            if args[0] == "rev-parse":
                return b"FIXTURE_HEAD\n"
            if args[0] == "rev-list":
                return ("FIXTURE_HEAD " + identity.SOURCE + "\n").encode()
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
