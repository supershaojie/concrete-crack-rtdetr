"""C26 direct dispatch, fixed recipe, streaming coordinates and complete archive regressions."""
from pathlib import Path
import gzip
import json
import os
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import train_c26 as train
import c26_results as results
from init_c26 import SOURCE_SHA256, write_json


class TestC26Tools(unittest.TestCase):
    def test_109_field_recipe_and_type_drift(self):
        source = ROOT / "docs/scca/c2_args.yaml"
        target, rows = train.recipe(source, "c26", ROOT / "weights/c26.pt")
        self.assertEqual(len(rows), 109)
        self.assertEqual({r["field"] for r in rows if r["changed"]}, {"model", "name", "save_dir"})
        self.assertFalse(target["augment"])
        self.assertEqual(target["mosaic"], .8)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            path = Path(d) / "c2.yaml"
            args = train.YAML.load(source)
            args["warmup_epochs"] = 5.0  # equal value but wrong archived type
            train.YAML.save(path, args)
            with self.assertRaises(RuntimeError):
                train.recipe(path, "c26", Path(d) / "init.pt")

    def test_direct_c25_independence_duplicate_guard_and_failure_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            p = dict(name="c26", run=root / "run", launch=root / "launch", init=root / "init.pt",
                     source=root / "source.pt", c2_args=root / "c2.yaml")
            p["source"].write_bytes(b"source")
            p["c2_args"].write_text("test")
            data = root / "data.yaml"
            data.write_text("test")
            args = dict(save_dir=str(p["run"]), data=str(data))
            dispatch = []
            def run(cmd, **kwargs):
                dispatch.append(cmd)
                return SimpleNamespace(returncode=int("has-session" in cmd and cmd[-1] == "c26-training"))
            def initialize(source, output, variant):
                output.write_bytes(b"fresh")
                return {"status": "passed"}
            with patch.object(train, "paths", return_value=p), patch.object(train, "ROOT", root), \
                 patch.object(train, "runtime", return_value={"commit": "a" * 40}), patch.object(train, "verify_server_environment"), \
                 patch.object(train, "recipe", return_value=(args, [])), patch.object(train, "record_source"), \
                 patch.object(train, "ensure_amp_resources"), patch.object(train, "initialize", side_effect=initialize), \
                 patch.object(train, "check_det_dataset", return_value=dict(nc=1, train=str(root), val=str(root), test=str(root))), \
                 patch.object(train.shutil, "which", return_value="tmux"), patch.object(train.torch.cuda, "is_available", return_value=True), \
                 patch.object(train.subprocess, "check_output", return_value=""), patch.object(train.subprocess, "run", side_effect=run), \
                 patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "rtdetr"}):
                train.start_direct("c26")
                plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
                self.assertEqual(plan["full_server_preflight"], "NOT_RUN")
                self.assertEqual(plan["session"], "c26-training")
                script = (p["launch"] / "worker.sh").read_text()
                self.assertIn(str(root / "tools/train_c26.py"), script)
                self.assertIn("process_exit_code.txt", script)
                self.assertIn("export C26_MAIN=", script)
                self.assertNotIn("c25", script.lower())
                with self.assertRaises(FileExistsError):
                    train.start_direct("c26")
                self.assertEqual(sum("new-session" in c for c in dispatch), 1)
                # Missing init after dispatch must produce a nonzero worker record, no retry or batch change.
                p["init"].unlink()
                with self.assertRaises(FileNotFoundError):
                    train.worker("c26")
                self.assertEqual(json.loads((p["launch"] / "exit_code.json").read_text())["exit_code"], 1)
        trainer = SimpleNamespace(_oom_retries=0, batch_size=16)
        train.disable_oom_retry(trainer)
        self.assertEqual((trainer._oom_retries, trainer.batch_size), (3, 16))

    def test_stream_full_precision_anisotropic_unclipped_and_empty_gt(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            pred = dict(bboxes=torch.tensor([[-64., 128., 704., 512.], [1., 2., 3., 4.]]),
                        conf=torch.tensor([.123456789, .00001]), cls=torch.zeros(2))
            gt = dict(im_file=str(root / "images/x.jpg"), ori_shape=(320, 1280), imgsz=(640, 640),
                      bboxes=torch.empty(0, 4), cls=torch.empty(0))
            row = results.image_record(pred, gt, root, .001)
            self.assertEqual(row["image"], "images/x.jpg")
            self.assertEqual(row["predictions"][0]["bbox"], [-128., 64., 1408., 256.])
            self.assertEqual(row["predictions"][0]["score"], float(pred["conf"][0]))
            self.assertEqual(len(row["predictions"]), 2)
            self.assertFalse(row["predictions"][1]["used_for_metrics"])
            self.assertEqual(row["ground_truth"], [])

    def test_complete_pack_over_20mib_with_weights_images_streams_and_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            run, launch = root / "run", root / "launch"
            (run / "weights").mkdir(parents=True)
            launch.mkdir()
            (run / "weights/best.pt").write_bytes(os.urandom(21 * 1024 * 1024))
            for name in ("weights/last.pt", "args.yaml", "results.csv", "results.png", "train_batch0.jpg", "labels.jpg"):
                (run / name).write_bytes(b"fixture")
            plots = ("BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
                     "confusion_matrix.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg")
            for name in plots:
                (run / name).write_bytes(b"fixture")
            required = ("nc1_loading.json", "training_setup.json", "actual_train_args.yaml", "train_args.yaml", "parameter_diff.json",
                        "authoritative_c2_args.yaml", "data_config.yaml", "pip_freeze.txt", "source_snapshot.tar.gz", "source_from_c24.patch",
                        "source_record.json", "process.json", "console.log", "worker.sh", "launch_state.json")
            for name in required:
                (launch / name).write_bytes(b"fixture")
            info = {"commit": "a" * 40}
            write_json(launch / "plan.json", dict(runtime=info, full_server_preflight="NOT_RUN"))
            write_json(launch / "initialization.json", dict(source_sha256=SOURCE_SHA256))
            write_json(launch / "exit_code.json", dict(exit_code=0))
            (launch / "process_exit_code.txt").write_text("0\n")
            digest = results.sha256(run / "weights/best.pt")
            for split in ("val", "test"):
                folder = launch / ("evaluation_" + split)
                folder.mkdir()
                row = dict(image="images/" + split + ".jpg", predictions=[{}] * 300, ground_truth=[{}])
                stream = folder / "predictions_gt.jsonl.gz"
                with gzip.open(stream, "wt", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                write_json(folder / "metrics.json", dict(status="completed", split=split, policy=results.POLICY, checkpoint_sha256=digest,
                           runtime=info, data_sha256=results.sha256(launch / "data_config.yaml"), export_complete=True,
                           settings=results.EVAL, ap_by_class=[[.1] * 10], predictions_gt_sha256=results.sha256(stream),
                           images=1, expected_images=1, predictions=300, ground_truth=1,
                           split_paths_sha256=results.hashlib.sha256(row["image"].encode()).hexdigest()))
                for name in plots:
                    (folder / name).write_bytes(b"prediction")
            target = root / "result.tar.gz"
            with patch.object(results, "ROOT", root), patch.object(results, "paths", return_value=dict(run=run, launch=launch)), \
                 patch.object(results, "runtime", return_value=info), patch.object(results.subprocess, "check_output", return_value=b""), \
                 patch.object(results, "evaluate", side_effect=AssertionError("Packaging must not evaluate")):
                results.package(target)
                self.assertGreater(target.stat().st_size, 20 * 1024 * 1024)
                with tarfile.open(target) as archive:
                    names = archive.getnames()
                    for name in ("training/weights/best.pt", "training/weights/last.pt", "training/train_batch0.jpg",
                                 "evaluation/test/predictions_gt.jsonl.gz", "evaluation/val/val_batch0_pred.jpg", "metadata/launch/console.log"):
                        self.assertIn(name, names)
                    self.assertFalse(any("audit.passed" in n for n in names))
                results.verify_archive(target)
                self.assertTrue(Path(str(target) + ".sha256").read_text().startswith(results.sha256(target)))
                with self.assertRaises(RuntimeError):
                    results.package(target)
                write_json(launch / "exit_code.json", dict(exit_code=1))
                with self.assertRaises(RuntimeError):
                    results.package(root / "failed.tar.gz")


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    unittest.main()
