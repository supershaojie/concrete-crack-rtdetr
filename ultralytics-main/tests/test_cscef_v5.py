"""Focused CSCEF-v5 regression tests; synthetic inputs only, no training or dataset evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import py_compile
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from audit_rtdetr_r18_lite_cscef_v5 import audit_module, audit_structure, verify_topology
from init_rtdetr_r18_lite_cscef_v5_controlled import NEW_SUFFIXES, SOURCE_SHA256, initialize, read_source, sha256
from train_rtdetr_r18_lite_cscef_v5 import C2_KEY_FIELDS, build_locked_args, execute, prepare
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import CSCEFv5
from ultralytics.utils import YAML


class TestCSCEFv5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_syntax_and_registration(self):
        paths = [ROOT / "tools" / name for name in (
            "init_rtdetr_r18_lite_cscef_v5_controlled.py", "audit_rtdetr_r18_lite_cscef_v5.py",
            "train_rtdetr_r18_lite_cscef_v5.py")]
        paths += [ROOT / "ultralytics-main/ultralytics/nn/modules/cscef_v5.py", Path(__file__)]
        with TemporaryDirectory(dir=ROOT / "outputs") as directory:
            for i, path in enumerate(paths):
                py_compile.compile(str(path), cfile=str(Path(directory) / f"{i}.pyc"), doraise=True)
        self.assertTrue(verify_topology()["v4_only_class_changed"])

    def test_architecture_rng_and_initializers(self):
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        a, b = CSCEFv5(256, 256), CSCEFv5(256, 256)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 26912)
        self.assertEqual(set(a.state_dict()), NEW_SUFFIXES)
        self.assertFalse(torch.equal(a.lateral_projection.weight, a.semantic_projection.weight))
        for name, parameter in a.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(b.named_parameters())[name]))
            self.assertEqual(torch.count_nonzero(parameter).item() == 0, name == "output_projection.weight")
        for child in a.modules():
            if isinstance(child, torch.nn.Conv2d):
                self.assertIsNone(child.bias)
            elif isinstance(child, torch.nn.GroupNorm):
                self.assertEqual(child.num_groups, 8)
                self.assertFalse(child.affine)
                self.assertEqual(child.eps, 1e-6)

    def test_v4_structure_and_boundaries_exact(self):
        self.assertEqual(audit_structure()["max_abs"], 0)

    def test_cpu_two_stage_gradients_and_semantic_dependence(self):
        self.assertEqual(audit_module()["status"], "passed")

    def test_zero_constant_random_and_small_spatial(self):
        module = CSCEFv5(256, 256)
        for height, width in ((1, 1), (1, 9), (7, 1), (9, 13)):
            for factory in (torch.zeros, torch.ones, torch.randn):
                with self.subTest(height=height, width=width, factory=factory):
                    lateral = factory(1, 256, height, width)
                    semantic = factory(1, 256, max(1, height // 2), max(1, width // 2))
                    with torch.no_grad():
                        output = module([lateral, semantic])
                    self.assertTrue(torch.equal(output, lateral))
        with torch.no_grad():
            torch.nn.init.normal_(module.output_projection.weight, std=0.01)
        for shape in ((1, 1), (1, 7), (5, 1), (9, 13)):
            for factory in (torch.zeros, torch.ones, torch.randn):
                output = module([factory(1, 256, *shape), factory(1, 256, *shape)])
                self.assertTrue(torch.isfinite(output).all())

    def test_invalid_construction_and_inputs(self):
        for channels in ((0, 256), (256, -1), (True, 256), (1.5, 256)):
            with self.assertRaises(ValueError):
                CSCEFv5(*channels)
        for kwargs in ({"eps": 0}, {"eps": float("nan")}, {"eps": float("inf")},
                       {"hidden_channels": 31}, {"num_groups": 0}, {"hidden_channels": 8}):
            with self.assertRaises(ValueError):
                CSCEFv5(256, 256, **kwargs)
        module = CSCEFv5(256, 256)
        x = torch.ones(1, 256, 3, 5)
        invalid = (x, [x], [x, "bad"], [x[0], x], [x, torch.ones(2, 256, 3, 5)],
                   [x, torch.ones(1, 255, 3, 5)], [x, torch.ones(1, 256, 0, 5)],
                   [x.long(), x], [x, torch.empty(1, 256, 3, 5, device="meta")],
                   [torch.ones(0, 256, 3, 5), torch.ones(0, 256, 3, 5)])
        for inputs in invalid:
            with self.assertRaises(ValueError):
                module(inputs)

    def test_activated_state_and_full_module_roundtrip(self):
        module = CSCEFv5(256, 256)
        with torch.no_grad():
            module.output_projection.weight.normal_(0, 0.01)
        x, s = torch.randn(1, 256, 7, 9), torch.randn(1, 256, 3, 5)
        restored = CSCEFv5(256, 256)
        restored.load_state_dict(module.state_dict(), strict=True)
        buffer = BytesIO()
        torch.save(module, buffer)
        buffer.seek(0)
        full = torch.load(buffer, weights_only=False)
        with torch.no_grad():
            expected = module([x, s])
            self.assertTrue(torch.equal(expected, restored([x, s])))
            self.assertTrue(torch.equal(expected, full([x, s])))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_amp_mixed_input_dtype_from_real_neck(self):
        module = CSCEFv5(256, 256).cuda()
        lateral = torch.randn(1, 256, 9, 13, device="cuda", dtype=torch.float16, requires_grad=True)
        semantic = torch.randn(1, 256, 9, 13, device="cuda", dtype=torch.float32, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = module([lateral, semantic])
        self.assertEqual(output.dtype, lateral.dtype)
        self.assertTrue(torch.equal(output, lateral))
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(module.output_projection.weight.grad).all())
        self.assertGreater(torch.count_nonzero(module.output_projection.weight.grad).item(), 0)

    def test_init_rejects_wrong_hash_and_overwrite(self):
        with TemporaryDirectory(dir=ROOT / "outputs") as directory:
            path = Path(directory) / "untrusted.pt"
            path.write_bytes(b"not the hash-verified C2 initialization")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                read_source(path)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                initialize(path, path)
            self.assertEqual(path.read_bytes(), b"not the hash-verified C2 initialization")


class TestC2RecipeLock(unittest.TestCase):
    def fixture(self):
        """Synthetic configuration for unit tests ONLY; never presented as original C2 evidence."""
        return {**deepcopy(DEFAULT_CFG_DICT), **C2_KEY_FIELDS, "model": "c2_init.pt", "task": "detect", "mode": "train",
                "project": "old/results", "name": "c2", "save_dir": "old/results/c2", "data": "original_data.yaml",
                "augmentations": None}

    def test_all_fields_preserved_except_model_and_outputs(self):
        baseline = self.fixture()
        original = deepcopy(baseline)
        target, rows = build_locked_args(baseline, Path("v5_init.pt"), ROOT / "outputs/test_recipe", "c15_test")
        self.assertEqual(baseline, original)
        self.assertEqual(set(target), set(baseline))
        self.assertEqual(len(rows), len(baseline))
        self.assertEqual({row["field"] for row in rows if not row["equal"]}, {"model", "project", "name", "save_dir"})
        self.assertEqual(target["data"], original["data"])
        self.assertIs(target["pretrained"], True)

    def test_missing_or_wrong_c2_fields_fail_without_defaults(self):
        for key, bad in (("lr0", 0.01), ("epochs", 150), ("mosaic", 0), ("amp", False), ("optimizer", "SGD")):
            baseline = self.fixture()
            baseline[key] = bad
            with self.assertRaisesRegex(RuntimeError, "sanity check failed"):
                build_locked_args(baseline, Path("v5.pt"), Path("new"), "c15")
        incomplete = self.fixture()
        del incomplete["save_period"]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            build_locked_args(incomplete, Path("v5.pt"), Path("new"), "c15")

    def test_invalid_output_identifier_and_unknown_field_rejected(self):
        for name in ("../c13", "a/b", "..", "/tmp", ""):
            with self.assertRaises(RuntimeError):
                build_locked_args(self.fixture(), Path("v5.pt"), Path("new"), name)
        bad = self.fixture()
        bad["typo_lr"] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown"):
            build_locked_args(bad, Path("v5.pt"), Path("new"), "c15")

    def test_missing_authoritative_file_does_not_start_training(self):
        with TemporaryDirectory(dir=ROOT / "outputs") as directory, patch("train_rtdetr_r18_lite_cscef_v5.RTDETR") as api:
            args = argparse.Namespace(c2_args=Path(directory) / "missing_C2_args.yaml")
            with self.assertRaisesRegex(FileNotFoundError, "No fallback"):
                prepare(args)
            api.assert_not_called()

    def test_prepare_roundtrip_and_stale_audit_or_existing_output_rejected(self):
        with TemporaryDirectory(dir=ROOT / "outputs") as directory, patch("train_rtdetr_r18_lite_cscef_v5.RTDETR") as api:
            root = Path(directory)
            data = root / "synthetic_data.yaml"
            data.write_text("# Synthetic fixture only; no dataset is read.\n", encoding="utf-8")
            baseline = self.fixture()
            baseline["data"] = str(data)
            original = root / "synthetic_args.yaml"
            YAML.save(original, baseline)
            initialized = root / "synthetic_init.pt"
            initialized.write_bytes(b"synthetic fixture; not loaded")
            audit_file = root / "synthetic_audit.json"
            audit = {"status": "passed", "initialization": {"source_sha256": SOURCE_SHA256,
                     "initialized_sha256": sha256(initialized)}, "runtime": {"code_sha256": {"fixture": "1"}}}
            audit_file.write_text(json.dumps(audit), encoding="utf-8")
            args = argparse.Namespace(c2_args=original, initialized=initialized, project=root / "runs", name="c15",
                                      audit_report=audit_file, report_dir=root / "reports")
            with patch("train_rtdetr_r18_lite_cscef_v5.runtime_info", return_value={"code_sha256": {"fixture": "1"}}):
                plan, path = prepare(args)
                self.assertEqual(plan["target_args"], YAML.load(root / "reports/train_args.yaml"))
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "prepared_no_training")
                self.assertEqual(plan["c2_args_sha256"], sha256(original))
                (root / "runs/c15").mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    prepare(args)
                args.name = "c15_new"
                initialized.write_bytes(b"changed fixture")
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    prepare(args)
            api.assert_not_called()

    def test_future_training_log_and_exit_code_with_mock_only(self):
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory(dir=ROOT / "outputs") as directory:
                root = Path(directory)
                source, weights = root / "synthetic_args.yaml", root / "synthetic_init.pt"
                source.write_bytes(b"synthetic args")
                weights.write_bytes(b"synthetic weights")
                plan = {"runtime": {"git_status": [], "git_commit": "fixture", "ultralytics_file": "fixture"},
                        "c2_args": str(source), "c2_args_sha256": sha256(source), "initialized": str(weights),
                        "initialized_sha256": sha256(weights), "target_args": {"synthetic": True}}

                def mock_train(**kwargs):
                    self.assertEqual(kwargs, {"synthetic": True})
                    print("Synthetic logging test; no model, optimizer or dataset was constructed.")
                    if fail:
                        raise RuntimeError("synthetic failure")

                with patch("train_rtdetr_r18_lite_cscef_v5.RTDETR") as api:
                    api.return_value.train.side_effect = mock_train
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                            execute(plan, root)
                    else:
                        execute(plan, root)
                    api.return_value.train.assert_called_once()
                code = json.loads((root / "exit_code.json").read_text(encoding="utf-8"))
                self.assertEqual(code["exit_code"], int(fail))
                self.assertIn("Synthetic logging test", (root / "console.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
