from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from init_rtdetr_r18_lite_cbr_controlled import DEFAULT_OUTPUT, verify_protected
from train_rtdetr_r18_lite_cbr import build_locked_args, actual_args_check, DEFAULT_NAME
from ultralytics.utils import YAML


class CBRToolTests(unittest.TestCase):
    def test_full_recipe_and_reject_mutations(self):
        recipe = YAML.load(ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml")
        target, rows = build_locked_args(recipe, DEFAULT_OUTPUT, DEFAULT_NAME)
        self.assertEqual(len(rows), 109)
        self.assertEqual({r["field"] for r in rows if not r["equal"]}, {"model", "name", "save_dir"})
        self.assertEqual(target["project"], recipe["project"])
        actual_args_check(target, deepcopy(target))
        for key, value in (("amp", 1), ("epochs", 100), ("project", "runs/new"), ("hsv_s", .6)):
            bad = deepcopy(recipe); bad[key] = value
            with self.assertRaises(RuntimeError): build_locked_args(bad, DEFAULT_OUTPUT, DEFAULT_NAME)
        bad = deepcopy(recipe); bad.pop("erasing")
        with self.assertRaises(RuntimeError): build_locked_args(bad, DEFAULT_OUTPUT, DEFAULT_NAME)

    def test_protected_c2_and_supervisor_exit(self):
        self.assertEqual(verify_protected()["status"], "passed")
        from cbr_tmux_worker import supervise
        import json
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(supervise(folder, [sys.executable, "-c", "raise SystemExit(7)"]), 7)
            report = json.loads((Path(folder) / "process_exit_code.json").read_text())
            self.assertEqual(report["exit_code"], 7)

    def test_fixed_pairing_and_package_exclusions(self):
        import hashlib
        import json
        import tarfile
        import torch
        from cbr_results import matches, package
        before = torch.tensor([[0., 0., 1., 1.], [2., 2., 3., 3.]])
        gi, pi = matches(before, before.flip(0))
        self.assertEqual((gi, pi), ([0, 1], [1, 0]))
        self.assertEqual(matches(before[:0], before), ([], []))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root / "run"; launch = root / "launch"; evaluation = root / "evaluation"
            (run / "weights").mkdir(parents=True); launch.mkdir(); evaluation.mkdir()
            (run / "weights/best.pt").write_bytes(b"test checkpoint placeholder")
            digest = hashlib.sha256(b"test checkpoint placeholder").hexdigest()
            for n in ("args.yaml", "results.csv"): (run / n).write_text("test fixture")
            for n in ("initialization.json", "audit.json", "launch_plan.json", "train_args.yaml", "actual_train_args.yaml",
                      "preflight.json", "tmux.json", "exit_code.json", "process_exit_code.json"):
                (launch / n).write_text("{}")
            (evaluation / "metrics_summary.json").write_text(json.dumps({"status":"completed", "models":{"C19":{"checkpoint_sha256":digest}}}))
            for n in ("C2_predictions_gt.jsonl.gz", "C19_predictions_gt.jsonl.gz", "fixed_val_diagnostics.json", "fixed_val_samples.json"):
                (evaluation / n).write_text("test fixture")
            package(run, launch, evaluation, root / "small.tar.gz")
            with tarfile.open(root / "small.tar.gz") as tar:
                self.assertFalse(any(n.endswith(".pt") for n in tar.getnames()))
                self.assertIn("training/results.csv", tar.getnames())
            with self.assertRaises(RuntimeError): package(run, launch, evaluation, root / "small.tar.gz")
            (launch / "preflight.json").unlink()
            with self.assertRaises(RuntimeError): package(run, launch, evaluation, root / "missing.tar.gz")


if __name__ == "__main__":
    unittest.main()
