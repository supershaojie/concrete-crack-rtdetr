"""Bounded lifecycle fault checks; no training, model inference or final test execution."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import train_trc_v1 as train
import trc_v1_common as common
import pack_trc_v1_light as pack
import ultralytics.engine.trainer as native_trainer
from ultralytics.utils import YAML


class LifecycleChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trc_lifecycle_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_recipe_and_reject_unapproved_change(self):
        target, rows = train.recipe(train.ARCHIVE, train.DEFAULT_VARIANT, self.root / "init.pt",
                                    data=self.root / "data.yaml", project=self.root / "runs")
        expected = YAML.load(train.ARCHIVE)
        self.assertEqual(set(target), set(expected))
        self.assertEqual(len(target), 109)
        self.assertLessEqual({r["field"] for r in rows if r["changed"]}, train.IDENTITY_FIELDS)
        for key in set(expected) - train.IDENTITY_FIELDS:
            self.assertEqual(target[key], expected[key])
            self.assertIs(type(target[key]), type(expected[key]))
        changed = self.root / "changed.yaml"
        YAML.save(changed, {**expected, "batch": 8})
        with self.assertRaises(RuntimeError):
            train.recipe(changed, train.DEFAULT_VARIANT, self.root / "init.pt")

    def test_plan_never_reserves_run_and_never_overwrites(self):
        source, init, data = (self.root / n for n in ("source.pt", "init.pt", "data.yaml"))
        source.write_bytes(b"source")
        init.write_bytes(b"init")
        config = YAML.load(train.ROOT / "docs/c19_lif_v1/c2_data.yaml")
        config["path"] = str(self.root / "dataset")
        YAML.save(data, config)
        args = SimpleNamespace(plan=self.root / "audit/plan.json", source=source, initialized=init,
                               data=data, c2_args=train.ARCHIVE, variant=train.DEFAULT_VARIANT,
                               project=self.root / "formal", preflight=self.root / "pending.json")
        with patch.object(train, "runtime", return_value={"commit": "test"}), \
                patch.object(train, "initialization_audit", return_value={"mock": "audit"}), redirect_stdout(io.StringIO()):
            plan = train.create_plan(args)
            self.assertFalse(Path(plan["args"]["save_dir"]).exists())
            self.assertFalse(args.project.exists())
            with self.assertRaises(RuntimeError):
                train.create_plan(args)

    def test_pending_or_stale_preflight_rejected(self):
        report = self.root / "preflight.json"
        fp = dict(variant=train.DEFAULT_VARIANT, initialized_sha256="same")
        base = dict(status="PASSED", server_capacity={"status": "PASSED"}, formal_optimizer_steps=0, fingerprint=fp)
        with patch.object(common, "fingerprint", return_value=fp):
            for changed in ({"status": "PENDING"}, {"server_capacity": {"status": "PENDING"}},
                            {"fingerprint": {"initialized_sha256": "changed"}}, {"formal_optimizer_steps": 1}):
                report.write_text(json.dumps({**base, **changed}), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    common.verify_preflight(report, train.DEFAULT_VARIANT, "source", "init")
            report.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(common.verify_preflight(report, train.DEFAULT_VARIANT, "source", "init"), base)

    def test_resume_keeps_learned_state_and_rejects_recipe_or_stripped_checkpoint(self):
        target, _ = train.recipe(train.ARCHIVE, train.DEFAULT_VARIANT, self.root / "init.pt",
                                 data=self.root / "data.yaml", project=self.root / "runs")
        checkpoint = Path(target["save_dir"]) / "weights/last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fake")
        model = torch.nn.Linear(2, 1)
        before = model.weight.detach().clone()
        value = dict(epoch=3, optimizer={"state": {}}, scaler={"scale": 65536}, train_args=target, ema=model)
        plan = dict(args=target, variant=train.DEFAULT_VARIANT)
        with patch.object(train, "torch_load", return_value=value), patch.object(train, "verify_model") as verify:
            train.validate_resume_checkpoint(checkpoint, plan)
            verify.assert_called_once_with(model, train.DEFAULT_VARIANT, zero=False)
            self.assertTrue(torch.equal(before, model.weight))
            value["train_args"] = {**target, "batch": 8}
            with self.assertRaises(RuntimeError):
                train.validate_resume_checkpoint(checkpoint, plan)
            value.update(train_args=target, optimizer=None)
            with self.assertRaises(RuntimeError):
                train.validate_resume_checkpoint(checkpoint, plan)

    def test_terminal_classifications(self):
        trainer = SimpleNamespace(epoch=199, args=SimpleNamespace(epochs=200), stop=True,
                                  stopper=SimpleNamespace(best_epoch=10, patience=50))
        self.assertEqual(train.completion_state(trainer), "COMPLETED_200_EPOCHS")
        trainer.epoch = 59
        self.assertEqual(train.completion_state(trainer), "EARLY_STOPPED_PATIENCE")
        trainer.epoch = 15
        self.assertEqual(train.completion_state(trainer), "INTERRUPTED")

    def test_native_trainer_constructor_start_and_resume_identity(self):
        """Exercise native argument plumbing only, with data/device setup stubbed; no train call."""
        target, _ = train.recipe(train.ARCHIVE, train.DEFAULT_VARIANT, self.root / "init.pt",
                                 data=self.root / "data.yaml", project=self.root / "runs")
        Path(target["data"]).write_text("names: [crack]\n", encoding="utf-8")
        with patch.object(native_trainer, "select_device", return_value=torch.device("cuda:0")), \
                patch.object(native_trainer, "init_seeds"), \
                patch.object(native_trainer, "print_args"), \
                patch.object(native_trainer.callbacks, "add_integration_callbacks"), \
                patch.object(native_trainer.BaseTrainer, "run_callbacks"), \
                patch.object(native_trainer.BaseTrainer, "get_dataset", return_value={"nc": 1, "channels": 3}), \
                patch.dict(train.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            constructor = native_trainer.BaseTrainer(overrides=dict(target))
            self.assertEqual(train.arguments_differ(vars(constructor.args), target), {})
            self.assertEqual(constructor.save_dir, Path(target["save_dir"]))
            checkpoint = constructor.save_dir / "weights/last.pt"
            checkpoint.write_bytes(b"checkpoint placeholder")
            loaded = SimpleNamespace(args=dict(target))
            with patch.object(native_trainer, "load_checkpoint", return_value=(loaded, {})):
                resumed = native_trainer.BaseTrainer(overrides={**target, "model": str(checkpoint), "resume": str(checkpoint)})
            self.assertEqual(train.arguments_differ(vars(resumed.args), target, resume=checkpoint), {})
            self.assertEqual(resumed.save_dir, constructor.save_dir)
            self.assertFalse(constructor.save_dir.with_name(constructor.save_dir.name + "2").exists())

    def test_light_archive_excludes_weights_and_verifies_hashes(self):
        audit, run = self.root / "audit", self.root / "run"
        audit.mkdir()
        (run / "weights").mkdir(parents=True)
        (run / "weights/best.pt").write_bytes(b"must not ship")
        (run / "predictions_gt.jsonl.gz").write_bytes(b"must not ship")
        (run / "results.csv").write_text("epoch,loss\n1,1\n", encoding="utf-8")
        plan = dict(schema="trc_v1_plan_v1", variant=train.DEFAULT_VARIANT, args={"save_dir": str(run)},
                    initialization_audit={"origin": "test audit"},
                    initialized=str(self.root / "init.pt"), preflight=str(audit / "preflight.json"),
                    authoritative_args=str(train.ARCHIVE))
        plan_path = audit / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        output = self.root / "LIGHT.tar.gz"
        with patch.object(pack, "runtime", return_value={"commit": "test"}), \
                patch.object(pack.subprocess, "check_output", return_value=b""), redirect_stdout(io.StringIO()):
            result = pack.package(plan_path, output)
            names = {row["path"] for row in pack.verify_archive(output)}
            self.assertIn("training/results.csv", names)
            self.assertFalse(any(n.endswith((".pt", ".jsonl.gz")) for n in names))
            self.assertEqual(result["sha256"], train.sha256(output))
            self.assertFalse(result["evidence_complete"])
            with self.assertRaises(RuntimeError):
                pack.package(plan_path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleChecks))
    if args.output:
        train.atomic_json(args.output, dict(status="PASSED" if result.wasSuccessful() else "FAILED", checks=result.testsRun,
                          failures=len(result.failures), errors=len(result.errors), formal_training="NOT_STARTED", test="NOT_RUN"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
