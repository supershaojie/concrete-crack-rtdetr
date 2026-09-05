"""Negative-path recipe/launch tests and real RTDETR.train -> get_model integration, without training."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import train_rtdetr_r18_lite_gsdr_aifi_v3 as launch
from gsdr_aifi_v3_tmux_worker import supervise
from audit_rtdetr_r18_lite_gsdr_aifi_v3 import state_hashes, trainer_build, real_optimizer, optimizer_rows
from init_rtdetr_r18_lite_gsdr_aifi_v3_controlled import V3_CFG
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds

FIXTURE = ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml"


class GSDRV3ToolsTest(unittest.TestCase):
    def setUp(self):
        self.baseline = YAML.load(FIXTURE)
        (ROOT / "weights").mkdir(exist_ok=True)

    def test_diagnostic_defines_zero_and_per_image_constant_energy_honestly(self):
        from diagnose_gsdr_aifi_v3 import activation_statistics
        from ultralytics.nn.modules import QueryLocalDeformableRelationV3
        relation = QueryLocalDeformableRelationV3(8, 8, 2)
        x = torch.randn(1, 8, 5, 7)
        _, state = relation.relation_features(x)
        zero = activation_statistics(x, torch.zeros_like(x), state)
        self.assertIsNone(zero["spatial_constant_energy_fraction"])
        self.assertTrue(zero["zero_delta"])
        constant = activation_statistics(x, torch.ones_like(x), state)
        self.assertAlmostEqual(constant["spatial_constant_energy_fraction"], 1.0)
        self.assertAlmostEqual(constant["normalized_point_weight_entropy"], 1.0)
        self.assertEqual(constant["outside_valid_pixel_centers_fraction"], 0)
        self.assertEqual(len(constant["per_query"]), 35)
        self.assertEqual(constant["per_query"][8]["xy"], [1, 1])

    def test_all_109_fields_only_three_identifiers_change(self):
        before = deepcopy(self.baseline)
        target, rows = launch.build_locked_args(self.baseline, ROOT / "weights/example.pt", launch.DEFAULT_NAME)
        self.assertEqual(self.baseline, before)
        self.assertEqual(set(target), set(before))
        self.assertEqual(len(rows), 109)
        self.assertEqual({r["field"] for r in rows if not r["equal"]}, {"model", "name", "save_dir"})
        self.assertEqual(target["project"], before["project"])
        self.assertEqual(launch.ALLOWED_CHANGES, {"model", "name", "save_dir"})
        for key in before:
            self.assertIs(type(target[key]), type(before[key]))

    def test_missing_or_unknown_fields_and_recipe_changes_fail_closed(self):
        for key in self.baseline:
            incomplete = deepcopy(self.baseline)
            incomplete.pop(key)
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                launch.build_locked_args(incomplete, Path("init.pt"), "c18")
        for key, value in (("resume", True), ("batch", 8), ("amp", False), ("warmup_bias_lr", 0.0),
                           ("project", "different"), ("device", 0), ("fraction", 1),
                           ("plots", 1), ("degrees", 5.0), ("iou", 0.5), ("tracker", "different")):
            modified = {**self.baseline, key: value}
            with self.subTest(modified=key), self.assertRaises(RuntimeError):
                launch.build_locked_args(modified, Path("init.pt"), "c18")
        with self.assertRaises(RuntimeError):
            launch.build_locked_args({**self.baseline, "unknown": 1}, Path("init.pt"), "c18")

    def test_actual_args_drift_is_rejected_even_outside_sanity_subset(self):
        for key in ("iou", "plots", "fraction", "tracker", "data", "project"):
            actual = deepcopy(self.baseline)
            actual[key] = "changed"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launch.actual_args_check(actual, self.baseline)

    def test_actual_args_equal_values_with_different_types_are_rejected(self):
        for key, value in (("plots", 1), ("fraction", 1), ("degrees", 5.0), ("device", 0)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launch.actual_args_check({**self.baseline, key: value}, self.baseline)

    def test_existing_run_prevents_preparation_before_checkpoint_io(self):
        with TemporaryDirectory() as directory, patch.object(launch, "verify_protected"), patch.object(launch.YAML, "load", return_value=self.baseline):
            config = Path(directory) / "args.yaml"
            config.write_text("fixture")
            target = {**self.baseline, "project": directory, "name": launch.DEFAULT_NAME}
            (Path(directory) / launch.DEFAULT_NAME).mkdir()
            with patch.object(launch, "build_locked_args", return_value=(target, [])), self.assertRaisesRegex(RuntimeError, "already exists"):
                launch.prepare(SimpleNamespace(c2_args=config, initialized=Path("unused.pt"), name=launch.DEFAULT_NAME))

    def test_bootstrap_failure_logs_actual_exit_code(self):
        with TemporaryDirectory() as directory:
            code = supervise(Path(directory), [sys.executable, "-c", "print('controlled bootstrap failure'); raise SystemExit(7)"])
            self.assertEqual(code, 7)
            self.assertEqual(json.loads((Path(directory) / "exit_code.json").read_text())["exit_code"], 7)
            self.assertIn("controlled bootstrap failure", (Path(directory) / "bootstrap.log").read_text())

    def test_missing_original_c2_file_never_falls_back(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                launch.prepare(SimpleNamespace(c2_args=Path(directory) / "missing.yaml"))

    def test_one_atomic_claim_per_experiment_across_reports(self):
        with TemporaryDirectory() as directory, patch.object(launch, "recheck"):
            plan = {"target_args": {"save_dir": str(Path(directory) / "run")}, "runtime": {"git_commit": "test"}}
            token = launch.claim_launch(plan)
            launch.verify_claim(plan, token)
            with self.assertRaises(FileExistsError):
                launch.claim_launch(plan)
            with self.assertRaises(RuntimeError):
                launch.verify_claim(plan, "wrong-token")

    def test_preflight_rejects_actual_initialized_state_mutation(self):
        with TemporaryDirectory() as directory:
            model = RTDETR(str(V3_CFG))
            target, _ = trainer_build(model.model.yaml, model.model, 1)
            report_dir = Path(directory)
            audit_path = report_dir / "audit.json"
            optimizer = real_optimizer(target)
            audit_path.write_text(json.dumps({"initialization": {"classes": {"1": {
                "state_sha256": state_hashes(target)}}}, "optimizer": {"all_parameters": optimizer_rows(target, optimizer)}}), encoding="utf-8")
            plan = {"target_args": {"resume": False}, "audit_report": str(audit_path), "runtime": {"git_commit": "test"}}
            launch.install_preflight(model, plan, report_dir)
            trainer = SimpleNamespace(args=SimpleNamespace(resume=False), start_epoch=0, amp=True,
                                      data={"nc": 1}, model=target, optimizer=optimizer, ema=SimpleNamespace(updates=0))
            callback = model.callbacks["on_train_start"][-1]
            callback(trainer)
            with torch.no_grad():
                target.model[26].enc_score_head.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, "initialization differs"):
                callback(trainer)

    def test_real_train_api_preserves_locked_args_and_uses_real_get_model(self):
        """Stop in a dataset-free trainer fixture immediately after the real API builds/loads its model."""
        with TemporaryDirectory(dir=ROOT / "weights") as directory:
            model = RTDETR(str(V3_CFG))
            weights = Path(directory) / "clean.pt"
            torch.save({"model": model.model.float(), "train_args": {"task": "detect"}}, weights)
            initialized = RTDETR(str(weights))
            expected, _ = trainer_build(initialized.model.yaml, initialized.model, 1)
            expected_hashes = state_hashes(expected)
            locked, _ = launch.build_locked_args(self.baseline, weights, "api_probe")

            class AuditStopped(Exception):
                pass

            class DatasetFreeTrainer(RTDETRTrainer):
                def __init__(self, overrides, _callbacks):
                    self.args = SimpleNamespace(**{k: v for k, v in overrides.items() if k != "session"})
                    self.data = {"nc": 1, "channels": 3}
                    init_seeds(42, deterministic=True)

                def train(self):
                    launch.actual_args_check(vars(self.args), locked)
                    if state_hashes(self.model) != expected_hashes:
                        raise RuntimeError("Real train API changed the initialized model")
                    raise AuditStopped("No dataset or optimizer step is executed")

            self.assertIs(DatasetFreeTrainer.get_model, RTDETRTrainer.get_model)
            with patch("ultralytics.engine.model.checks.check_pip_update_available"), self.assertRaises(AuditStopped):
                initialized.train(trainer=DatasetFreeTrainer, **locked)


if __name__ == "__main__":
    unittest.main(verbosity=2)
