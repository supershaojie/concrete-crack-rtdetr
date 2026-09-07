"""Direct-launch reservations, failure policy, recipe and evaluation/export regression tests."""
from pathlib import Path
import json
import os
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import train_scca as train
import scca_results as results


class TestSCCATools(unittest.TestCase):
    def test_recipe_all_109_fields_and_reject_drift(self):
        fixture = ROOT / "docs/scca/c2_args.yaml"
        for variant in ("c24", "c25"):
            target, rows = train.recipe(fixture, variant, ROOT / "weights" / (variant + ".pt"))
            self.assertEqual(len(target), 109)
            self.assertEqual({r["field"] for r in rows if r["changed"]}, {"model", "name", "save_dir"})
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            path = Path(d) / "args.yaml"
            data = train.YAML.load(fixture)
            data["lr0"] = .01
            train.YAML.save(path, data)
            with self.assertRaises(RuntimeError): train.recipe(path, "c24", Path(d) / "init.pt")

    def test_original_oom_guard_is_exhausted_before_batch(self):
        trainer = SimpleNamespace(_oom_retries=0, batch_size=16)
        train.disable_oom_retry(trainer)
        self.assertGreaterEqual(trainer._oom_retries, 3)
        self.assertEqual(trainer.batch_size, 16)
        # Lock this adapter to the actual native catch branch instead of assuming its behavior.
        native = (ROOT / "ultralytics-main/ultralytics/engine/trainer.py").read_text(encoding="utf-8")
        self.assertIn('if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:', native)
        self.assertLess(native.index('self.run_callbacks("on_train_batch_start")'), native.index('except torch.cuda.OutOfMemoryError:'))

    def test_direct_dispatch_no_prepare_and_duplicate_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            p = dict(name="c24", run=root / "run", launch=root / "launch", init=root / "init.pt",
                     source=root / "source.pt", c2_args=root / "c2.yaml")
            p["source"].write_bytes(b"source"); p["c2_args"].write_text("test")
            data = root / "data.yaml"; data.write_text("test")
            args = dict(save_dir=str(p["run"]), data=str(data))
            dispatch = []
            def run(cmd, **kwargs):
                dispatch.append(cmd)
                return SimpleNamespace(returncode=1 if "has-session" in cmd else 0)
            def initialize(source, output, variant):
                output.write_bytes(b"fresh")
                return {"status": "passed"}
            with patch.object(train, "paths", return_value=p), patch.object(train, "ROOT", root), \
                 patch.object(train, "runtime", return_value={"commit": "a" * 40}), \
                 patch.object(train, "recipe", return_value=(args, [])), \
                 patch.object(train, "ensure_amp_resources"), patch.object(train, "initialize", side_effect=initialize), \
                 patch.object(train, "check_det_dataset", return_value=dict(nc=1, train=str(root), val=str(root), test=str(root))), \
                 patch.object(train.shutil, "which", return_value="tmux"), patch.object(train.torch.cuda, "is_available", return_value=True), \
                 patch.object(train.subprocess, "check_output", return_value=""), patch.object(train.subprocess, "run", side_effect=run), \
                 patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "rtdetr"}):
                train.start_direct("c24")
                plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
                self.assertEqual(plan["full_server_preflight"], "NOT_RUN")
                self.assertTrue(any("new-session" in c for c in dispatch))
                self.assertIn("process_exit_code.txt", (p["launch"] / "worker.sh").read_text())
                script = (p["launch"] / "worker.sh").read_text()
                self.assertIn("export PYTHONPATH=", script)
                self.assertIn("export YOLO_AUTOINSTALL=false", script)
                with self.assertRaises(FileExistsError): train.start_direct("c24")
                self.assertEqual(sum("new-session" in c for c in dispatch), 1)

    def test_corrected_sort_threshold(self):
        pred = torch.tensor([[[.5,.5,.2,.2,.05], [.5,.5,.2,.2,.9], [.5,.5,.2,.2,.01]]])
        result, affected = results.postprocess(pred, 640, .1)
        self.assertEqual(affected, 1)
        self.assertEqual(len(result[0]["conf"]), 1)
        self.assertAlmostEqual(float(result[0]["conf"][0]), .9, places=6)

    def test_pack_excludes_weights_and_images_and_protects_old(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            run, launch = root / "run", root / "launch"
            run.mkdir(); launch.mkdir()
            (run / "weights").mkdir()
            (run / "weights/best.pt").write_bytes(b"weights")
            (run / "train_batch0.jpg").write_bytes(b"dataset")
            (run / "results.csv").write_text("epoch,loss\n1,2\n")
            (launch / "console.log").write_text("test fixture\n")
            target = root / "result.tar.gz"
            with patch.object(results, "paths", return_value=dict(run=run, launch=launch)), \
                 patch.object(results, "runtime", return_value=dict(commit="a" * 40)), \
                 patch.object(results.subprocess, "check_output", return_value=b""):
                results.package("c24", target)
                with tarfile.open(target) as archive:
                    names = archive.getnames()
                    self.assertFalse(any(n.endswith((".pt", ".jpg")) for n in names))
                    self.assertIn("evaluation/test/NOT_RUN.txt", names)
                    self.assertIn("MANIFEST.json", names)
                self.assertLess(target.stat().st_size, 20 * 1024 * 1024)
                with self.assertRaises(RuntimeError): results.package("c24", target)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    unittest.main()
