"""C17 mean_structure and launch-lock regressions. Synthetic tensors; no training or dataset evaluation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import sys
import shlex
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from audit_rtdetr_r18_lite_cscef_v51 import (
    audit_autocast_context, audit_intervention, audit_module, audit_structure, verify_topology,
)
from init_rtdetr_r18_lite_cscef_v51_controlled import BASE_CFG, NEW_SUFFIXES, SOURCE_SHA256, sha256
import train_rtdetr_r18_lite_cscef_v51 as launch
from ultralytics import RTDETR
from ultralytics.nn.modules import CSCEFv5, CSCEFv51
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML


class TestCSCEFv51(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        (ROOT / "outputs").mkdir(exist_ok=True)

    def test_only_override_and_exact_state_contract(self):
        self.assertEqual(CSCEFv51.__bases__, (CSCEFv5,))
        self.assertIs(CSCEFv51.forward, CSCEFv5.forward)
        self.assertIs(CSCEFv51.__init__, CSCEFv5.__init__)
        before = torch.get_rng_state().clone()
        module = CSCEFv51(256, 256)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(set(module.state_dict()), NEW_SUFFIXES)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 26912)
        self.assertEqual(len(list(module.parameters())), 5)
        self.assertEqual(len(list(module.buffers())), 2)
        self.assertTrue(verify_topology()["v5_only_class_changed"])

    def test_nonlinear_order_boundaries_detach_and_sample_independence(self):
        self.assertTrue(audit_structure()["sample_independent"])

    def test_nonzero_full_forward_matches_diagnostic(self):
        self.assertGreater(audit_intervention()["nonidentity_cases"], 0)

    def test_gradient_opening_and_semantic_dependence(self):
        self.assertEqual(audit_module()["status"], "passed")

    def test_legacy_cpu_autocast_compatibility(self):
        real_autocast = torch.autocast

        def legacy_autocast(device_type, dtype=None, **kwargs):
            if torch.device(device_type).type == "cpu" and dtype == torch.float16:
                raise RuntimeError("PyTorch 2.1 CPU FP16 autocast constructor is unsupported")
            return real_autocast(device_type=device_type, dtype=dtype, **kwargs)

        with patch("audit_rtdetr_r18_lite_cscef_v51.torch.autocast", side_effect=legacy_autocast):
            self.assertIsInstance(audit_autocast_context("cpu", False), nullcontext)
            self.assertEqual(audit_intervention()["max_abs"], 0)
        with self.assertRaises(ValueError):
            audit_autocast_context("cpu", True)

    def test_baseline_smoke_and_v5_parser_regression(self):
        baseline = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=1, verbose=False).eval()
        v5 = RTDETRDetectionModel(str(BASE_CFG.with_name("rtdetr-resnet18-lite-cscef-v5.yaml")),
                                 ch=3, nc=1, verbose=False).eval()
        self.assertIs(type(v5.model[18]), CSCEFv5)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 20082772)
        self.assertEqual(sum(p.numel() for p in v5.parameters()), 20109684)
        with torch.inference_mode():
            for model in (baseline, v5):
                output = model(torch.rand(1, 3, 320, 320))[0]
                self.assertEqual(output.shape, (1, 300, 5))
                self.assertTrue(torch.isfinite(output).all())


class TestC17RecipeLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "outputs").mkdir(exist_ok=True)

    def baseline(self):
        # Exact attachment copy as a TEST fixture; production always reads the server's original file.
        return YAML.load(Path(__file__).parent / "fixtures/c17_original_c2_args.yaml")

    def test_actual_c2_all_fields_and_project_preserved(self):
        baseline = self.baseline()
        original = deepcopy(baseline)
        target, rows = launch.build_locked_args(baseline, PurePosixPath("/server/v51_init.pt"), None, launch.DEFAULT_NAME)
        self.assertEqual(baseline, original)
        self.assertEqual(set(target), set(baseline))
        self.assertEqual({r["field"] for r in rows if not r["equal"]}, {"model", "name", "save_dir"})
        self.assertEqual(launch.ALLOWED_CHANGES, {"model", "name", "save_dir"})
        self.assertEqual(target["project"], original["project"])
        self.assertEqual(target["data"], original["data"])

    def test_no_project_changes_or_defaults_or_unknown_fields(self):
        with self.assertRaisesRegex(RuntimeError, "project must equal"):
            launch.build_locked_args(self.baseline(), Path("v51.pt"), Path("other"), "c17")
        for field in self.baseline():
            if field not in launch.DEFAULT_CFG_DICT:
                continue
            incomplete = self.baseline()
            del incomplete[field]
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "incomplete"):
                launch.build_locked_args(incomplete, Path("v51.pt"), None, "c17")
        unknown = self.baseline()
        unknown["typo"] = 1
        with self.assertRaisesRegex(RuntimeError, "Unknown"):
            launch.build_locked_args(unknown, Path("v51.pt"), None, "c17")

    def fixture(self, root):
        """Synthetic file/runtime evidence for launch logic only; never a passing model audit."""
        baseline = self.baseline()
        baseline.update(project=str(root / "runs"), data=str(root / "data.yaml"))
        (root / "data.yaml").write_text("# fixture; dataset never loaded\n", encoding="utf-8")
        YAML.save(root / "args.yaml", baseline)
        initialized = root / "init.pt"
        initialized.write_bytes(b"synthetic initialization placeholder")
        runtime = {"git_status": [], "git_commit": "fixture", "code_sha256": {"fixture": "1"},
                   "torch_version": "fixture", "ultralytics_version": "fixture", "python": "fixture",
                   "ultralytics_file": "fixture"}
        audit = {"status": "passed", "initialization": {"source_sha256": SOURCE_SHA256,
                 "initialized_sha256": sha256(initialized)}, "runtime": runtime}
        (root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
        args = argparse.Namespace(c2_args=root / "args.yaml", initialized=initialized, project=None,
                                  name=launch.DEFAULT_NAME, audit_report=root / "audit.json", report_dir=root / "reports")
        return args, runtime

    def test_prepare_is_inert_immutable_and_plan_rejects_changes(self):
        with TemporaryDirectory(dir=ROOT / "outputs") as directory, patch.object(launch, "RTDETR") as api:
            args, runtime = self.fixture(Path(directory))
            with patch.object(launch, "runtime_info", return_value=runtime):
                plan, path = launch.prepare(args)
                self.assertEqual(plan["target_args"], YAML.load(args.report_dir / "train_args.yaml"))
                launch.validate_plan(plan, args.report_dir)
                api.assert_not_called()
                before = path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "not empty"):
                    launch.prepare(args)
                self.assertEqual(path.read_bytes(), before)
                for field, value in (("project", "other"), ("freeze", 1), ("lr0", 0.2)):
                    changed = deepcopy(plan)
                    changed["target_args"][field] = value
                    with self.assertRaisesRegex(RuntimeError, "recipe was modified"):
                        launch.validate_plan(changed, args.report_dir)
                args.initialized.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "initialized changed"):
                    launch.validate_plan(plan, args.report_dir)

    def test_missing_data_remains_server_path_and_no_artifacts(self):
        with TemporaryDirectory(dir=ROOT / "outputs") as directory, patch.object(launch, "RTDETR") as api:
            args, runtime = self.fixture(Path(directory))
            baseline = YAML.load(args.c2_args)
            baseline["data"] = str(Path(directory) / "missing_server_data.yaml")
            YAML.save(args.c2_args, baseline)
            with patch.object(launch, "runtime_info", return_value=runtime):
                with self.assertRaisesRegex(RuntimeError, "data field will not be rewritten"):
                    launch.prepare(args)
            self.assertEqual(YAML.load(args.c2_args)["data"], baseline["data"])
            self.assertFalse(args.report_dir.exists())
            api.assert_not_called()

    def test_success_failure_logs_and_exclusive_launch_with_mock_training(self):
        for fail in (False, True):
            with TemporaryDirectory(dir=ROOT / "outputs") as directory, self.subTest(fail=fail):
                args, runtime = self.fixture(Path(directory))
                with patch.object(launch, "runtime_info", return_value=runtime):
                    plan, _ = launch.prepare(args)

                    def mock_train(**kwargs):
                        trainer = kwargs.pop("trainer")
                        self.assertTrue(issubclass(trainer, launch.RTDETRTrainer))
                        self.assertEqual(kwargs, plan["target_args"])
                        print("Synthetic launch test; no training or dataset evaluation.")
                        if fail:
                            raise RuntimeError("synthetic failure")

                    with patch.object(launch, "RTDETR") as api:
                        api.return_value.train.side_effect = mock_train
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                                launch.execute(plan, args.report_dir)
                        else:
                            launch.execute(plan, args.report_dir)
                    code = json.loads((args.report_dir / "exit_code.json").read_text(encoding="utf-8"))
                    self.assertEqual(code["exit_code"], int(fail))
                    self.assertIn("Synthetic launch test", (args.report_dir / "console.log").read_text(encoding="utf-8"))
                    with self.assertRaisesRegex(RuntimeError, "already been launched"):
                        launch.validate_plan(plan, args.report_dir)

    def test_real_api_passes_complete_recipe_to_trainer_without_dataset(self):
        target, _ = launch.build_locked_args(self.baseline(), Path("v51.pt"), None, launch.DEFAULT_NAME)
        cls = launch.locked_trainer(target)

        class StopBeforeDataset(Exception):
            pass

        def stop_trainer(self, cfg=None, overrides=None, _callbacks=None):
            self.args = SimpleNamespace(**deepcopy(cfg))
            raise StopBeforeDataset

        model = RTDETR(str(BASE_CFG))
        with patch("ultralytics.engine.model.checks.check_pip_update_available"), \
                patch.object(launch.RTDETRTrainer, "__init__", stop_trainer):
            with self.assertRaises(StopBeforeDataset):
                model.train(trainer=cls, **deepcopy(target))
            bad = deepcopy(target)
            bad["nbs"] = 32
            with self.assertRaisesRegex(RuntimeError, "before Trainer"):
                cls(overrides=bad)

    def test_tmux_reuses_plan_and_current_conda_environment_without_training(self):
        with TemporaryDirectory(dir=ROOT / "outputs") as directory:
            args, runtime = self.fixture(Path(directory))
            with patch.object(launch, "runtime_info", return_value=runtime):
                plan, path = launch.prepare(args)
            # Rebind only this module's os reference; do not change pathlib's platform behavior.
            fake_os = SimpleNamespace(name="posix", environ=dict(os.environ))
            with patch.object(launch, "os", fake_os), patch.object(launch, "validate_plan") as validate, \
                    patch.object(launch.subprocess, "run") as run, patch.object(launch, "execute") as execute, \
                    patch.object(sys, "argv", ["train", "--plan", str(path), "--tmux"]):
                launch.main()
            self.assertGreaterEqual(validate.call_count, 1)
            execute.assert_not_called()
            command = run.call_args.args[0]
            self.assertEqual(command[:5], ["tmux", "new-session", "-d", "-s",
                                           "cscef_v51_" + plan["target_args"]["name"]])
            worker = shlex.split(command[-1])
            self.assertIn(sys.executable, worker)
            self.assertIn("PYTHONPATH=" + str(ROOT / "ultralytics-main"), worker)
            self.assertEqual(worker[-3:], ["--execute", "--plan", str(path.resolve())])


if __name__ == "__main__":
    unittest.main()
