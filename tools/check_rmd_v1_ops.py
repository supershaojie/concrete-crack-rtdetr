"""Fault tests for fail-closed launch gates, final-eval failure preservation and incomplete LIGHT archives."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rmd_v1 as ops
import rmd_v1_eval as evaluation
from rmd_v1_common import ROOT, write_json, read_json, sha256, strict_record
from ultralytics.models.rtdetr.train import RTDETRTrainer
import torch


class Operations(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rmd_ops_")
        self.folder = Path(self.temporary.name)
        self.out = self.folder/"evidence"; self.out.mkdir()
        self.run = self.folder/"run"; (self.run/"weights").mkdir(parents=True)
        self.patch_ops = patch.object(ops, "OUT", self.out); self.patch_ops.start()
        self.patch_eval = patch.object(evaluation, "OUT", self.out); self.patch_eval.start()
        parent = self.folder/"mother.pt"; parent.write_bytes(b"synthetic gate fixture; not a checkpoint")
        self.report = dict(status="PASS", binding="fixture", micro_batches=16, elapsed_seconds=100, correctness={"status": "PASS"},
                           capacity=dict(status="PASS", effective_updates=1, batch=16, imgsz=640),
                           resume={"status": "PASS"}, lifecycle={"status": "PASS"},
                           applicability=dict(status="APPLICABILITY_PASS", valid_batches=8, gt_coverage_k_ge_2=.6,
                             mean_abs_weight_minus_one=.02, trained_mother=dict(path=str(parent), sha256=sha256(parent))))

    def tearDown(self):
        self.patch_eval.stop(); self.patch_ops.stop(); self.temporary.cleanup()

    def publish(self, report):
        path = self.out/"report.json"; write_json(path, report)
        write_json(self.out/"preflight.json", dict(report=str(path), report_sha256=sha256(path),
            status=report["status"], process_exit_code=0, timed_out=False))

    def test_every_gate_and_binding_is_required(self):
        self.publish(self.report)
        ops.gates({"binding": "fixture"})
        for component in ("correctness", "capacity", "resume", "lifecycle"):
            broken = deepcopy(self.report); broken[component]["status"] = "PENDING"
            self.publish(broken)
            with self.assertRaises(RuntimeError): ops.gates({"binding": "fixture"})
        for field, value in (("valid_batches", 7), ("gt_coverage_k_ge_2", .499), ("mean_abs_weight_minus_one", .00999),
                             ("status", "NOT_APPLICABLE"), ("status", "APPLICABILITY_PENDING")):
            broken = deepcopy(self.report); broken["applicability"][field] = value
            self.publish(broken)
            with self.assertRaises(RuntimeError): ops.gates({"binding": "fixture"})
        broken = deepcopy(self.report); broken["capacity"]["effective_updates"] = 0
        self.publish(broken)
        with self.assertRaises(RuntimeError): ops.gates({"binding": "fixture"})
        broken = deepcopy(self.report); broken["status"] = "FAIL"
        self.publish(broken)
        with self.assertRaises(RuntimeError): ops.gates({"binding": "fixture"})
        self.publish(self.report)
        with self.assertRaises(RuntimeError): ops.gates({"binding": "changed"})
        write_json(self.out/"not_applicable.json", dict(binding="fixture", parent_sha256=self.report["applicability"]["trained_mother"]["sha256"]))
        with self.assertRaisesRegex(RuntimeError, "NOT_APPLICABLE is retained"):
            ops.gates({"binding": "fixture"})

    def test_final_eval_failure_retains_completed_training_and_exception(self):
        for name in ("best", "last"):
            (self.run/f"weights/{name}.pt").write_bytes(b"isolated lifecycle fixture")
        trainer = ops.RunTrainer.__new__(ops.RunTrainer)
        trainer.stop = True; trainer.epoch = 199; trainer.epochs = 200
        trainer.args = SimpleNamespace(patience=50)
        trainer.stopper = SimpleNamespace(best_epoch=160)
        trainer.rmd_paths = {"run": self.run}
        trainer.rmd_identity = {"prepared": {"binding": "original", "code": {"commit": "original training SHA"}}}
        with patch.object(RTDETRTrainer, "final_eval", side_effect=RuntimeError("injected actual warmup failure")):
            with self.assertRaisesRegex(RuntimeError, "injected actual warmup failure"):
                trainer.final_eval()
        self.assertEqual(read_json(self.out/"training_completed.json")["status"], "TRAINING_COMPLETED")
        self.assertEqual(read_json(self.out/"final_eval.json")["status"], "FINAL_EVAL_FAILED")
        self.assertIn("injected actual warmup failure", read_json(self.out/"final_eval.json")["traceback"])
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            ops.valid_resume({"run": self.run}, {})
        self.assertFalse((self.out/"latest_exit.json").exists())

    def test_test_requires_completion_and_val_lock(self):
        with self.assertRaisesRegex(RuntimeError, "completion evidence"):
            evaluation.evaluate_formal({"run": self.run}, "test", [])
        with self.assertRaisesRegex(RuntimeError, "active experiment worker"):
            evaluation.evaluate_formal({"run": self.run}, "val", [{"pid": 123}])

    def test_strict_nonfinite_json_and_failure_pack(self):
        write_json(self.out/"failure.json", {"status": "FAIL", "loss": float("nan"), "error": "original exception"})
        record = read_json(self.out/"failure.json")
        self.assertIsNone(record["loss"]); self.assertEqual(record["nonfinite_paths"], ["$.loss"])
        (self.run/"weights/best.pt").write_bytes(b"fixture weight excluded from LIGHT")
        report = evaluation.pack({"run": self.run})
        self.assertEqual(report["format"], "LIGHT")
        self.assertFalse(report["weights"]["best"]["included"])
        self.assertIsNone(report["test"])
        import tarfile
        with tarfile.open(report["path"], "r:gz") as archive:
            names = archive.getnames()
            self.assertIn("evidence/failure.json", names)
            self.assertFalse(any(name.endswith(".pt") for name in names))

    def test_corrected_sorted_confidence_mask(self):
        preds = torch.tensor([[[.1, .2, .1, .1, .0001], [.7, .2, .1, .1, .9], [.5, .2, .1, .1, .5]]])
        rows, affected = evaluation.postprocess(preds, 640, .001)
        self.assertEqual(affected, 1)
        torch.testing.assert_close(rows[0]["conf"], torch.tensor([.9, .5]))
        torch.testing.assert_close(rows[0]["bboxes"][:, [0, 2]].mean(1)/640, torch.tensor([.7, .5]))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Operations)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(ROOT/"outputs/rmd_v1/local_checks/operations.json", dict(status="PASS" if result.wasSuccessful() else "FAIL",
               tests=result.testsRun, failures=[str(x) for x in result.failures+result.errors],
               scope="isolated fault fixtures, not a formal run"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
