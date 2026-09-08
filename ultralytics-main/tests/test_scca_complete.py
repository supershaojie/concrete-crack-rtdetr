"""C25 optimizer, per-image export and complete-package failure/coverage regressions."""
import gzip
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import scca_export as export
import scca_results as results
import train_scca as train
from init_scca import build
from ultralytics.models.rtdetr.train import RTDETRTrainer


class TestCompleteEvidence(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)

    def test_stream_non_square_full_precision_empty_and_coverage(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            images = [str(root / "images/a.jpg"), str(root / "images/b.jpg")]
            stream = export.PredictionStream(root, root, images, "val", .001)
            score = float(torch.tensor(.123456789))
            preds = [dict(bboxes=torch.tensor([[64., 128., 320., 512.]]), conf=torch.tensor([score]), cls=torch.tensor([0.])),
                     dict(bboxes=torch.empty(0, 4), conf=torch.empty(0), cls=torch.empty(0))]
            batch = dict(im_file=images, ori_shape=[(100, 200), (300, 100)], img=torch.empty(2, 3, 640, 640),
                         batch_idx=torch.tensor([0]), bboxes=torch.tensor([[.5, .5, .5, .5]]), cls=torch.tensor([[0.]]))
            stream.write_batch(preds, batch)
            report = stream.finish()
            self.assertEqual(report["images"], 2)
            with gzip.open(stream.path, "rt", encoding="utf-8") as f: rows = [json.loads(line) for line in f]
            self.assertEqual(rows[0]["image"], "images/a.jpg")
            self.assertEqual(rows[0]["predictions"][0]["score"], score)
            self.assertEqual(rows[0]["predictions"][0]["bbox"], [20., 20., 100., 80.])
            self.assertEqual(rows[0]["ground_truth"][0]["bbox"], [50., 25., 150., 75.])
            self.assertEqual(rows[1]["predictions"], [])
            self.assertEqual(rows[1]["ground_truth"], [])
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            stream = export.PredictionStream(d, d, [Path(d) / "missing.jpg"], "test", .001)
            with self.assertRaisesRegex(RuntimeError, "Incomplete split"): stream.finish()

    def test_stream_rejects_duplicate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            path = str(Path(d) / "a.jpg")
            stream = export.PredictionStream(d, d, [path], "val", .001)
            batch = dict(im_file=[path], ori_shape=[(100, 200)], img=torch.empty(1, 3, 640, 640),
                         batch_idx=torch.empty(0), bboxes=torch.empty(0, 4), cls=torch.empty(0, 1))
            pred = dict(bboxes=torch.empty(0, 4), conf=torch.empty(0), cls=torch.empty(0))
            stream.write_batch([pred], batch)
            with self.assertRaisesRegex(RuntimeError, "Unexpected/repeated"): stream.write_batch([pred], batch)
            stream.close()

    def test_all_queries_preserves_corrected_metrics(self):
        raw = torch.tensor([[[.5, .5, .1, .1, .0001], [.5, .5, .2, .2, .9], [.5, .5, .3, .3, .5]]])
        old = raw.clone()
        rows, affected = results.postprocess(raw, 640, .001, all_queries=True)
        filtered, affected2 = results.postprocess(raw, 640, .001)
        self.assertTrue(torch.equal(raw, old))
        self.assertEqual(len(rows[0]["conf"]), 3)
        self.assertEqual(affected, affected2)
        for k, value in rows[0].items():
            torch.testing.assert_close(value[rows[0]["conf"] > .001], filtered[0][k], rtol=0, atol=0)

    def test_test_rejects_changed_workers_or_source_before_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            weights, data, prior = root / "best.pt", root / "data.yaml", root / "val.json"
            weights.write_bytes(b"fixture")
            data.write_text("nc: 1\n")
            report = dict(status="completed", split="val", checkpoint_sha256=results.sha256(weights),
                          data_sha256=results.sha256(data), policy="corrected_sorted_conf_mask_v1",
                          runtime={"commit": "a" * 40}, settings=dict(imgsz=640, batch=16, workers=1, half=False,
                          conf=.001, iou=.7, max_det=300, augment=False, rect=False))
            with patch.object(results, "RTDETR", side_effect=AssertionError("Must reject before inference")), \
                 patch.object(results, "runtime", return_value={"commit": "b" * 40}):
                prior.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "workers"):
                    results.evaluate(weights, data, "test", root / "test", val_report=prior)
                report["settings"]["workers"] = 0
                prior.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "source commit"):
                    results.evaluate(weights, data, "test", root / "test", val_report=prior)
                self.assertFalse((root / "test").exists())

    def test_both_modules_in_native_optimizer_once_reject_missing_duplicate(self):
        torch.set_num_threads(4)
        model = build("c25", nc=1)
        trainer = object.__new__(RTDETRTrainer)
        optimizer = trainer.build_optimizer(model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
        result = train.optimizer_audit(model, optimizer, "c25")
        self.assertEqual(len(result["cscef"]), 5)
        self.assertTrue(all(count == 1 for group in result.values() for count in group.values()))
        param = model.model[18].output_projection.weight
        optimizer.param_groups[0]["params"].append(param)
        with self.assertRaisesRegex(RuntimeError, "absent or duplicated"): train.optimizer_audit(model, optimizer, "c25")
        optimizer.param_groups[0]["params"].pop()
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p is not param]
        with self.assertRaisesRegex(RuntimeError, "absent or duplicated"): train.optimizer_audit(model, optimizer, "c25")

    def test_complete_pack_over_20mib_weights_images_missing_and_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as d:
            root = Path(d)
            run, launch = root / "run", root / "launch"
            (run / "weights").mkdir(parents=True)
            launch.mkdir()
            # Incompressible fixture demonstrates that the old compressed-size cap is absent.
            with (run / "weights/best.pt").open("wb") as f:
                for _ in range(21): f.write(os.urandom(1024 * 1024))
            (run / "weights/last.pt").write_bytes(b"last")
            (run / "train_batch0.jpg").write_bytes(b"visualization")
            (launch / "console.log").write_text("full log\n")
            p = dict(run=run, launch=launch, source=root / "init.pt", init=root / "controlled.pt", c2_args=root / "args.yaml")
            target = root / "complete.tar.gz"
            with patch.object(train, "paths", return_value=p), patch.object(train, "MAIN", root), \
                 patch.object(export, "runtime", return_value={"commit": "a" * 40}), \
                 patch.object(export.subprocess, "check_output", return_value=b""), \
                 patch.object(results, "evaluate", side_effect=AssertionError("Packaging must never evaluate")):
                export.package_complete("c25", target)
                self.assertGreater(target.stat().st_size, 20 * 1024 * 1024)
                inventory = json.loads(Path(str(target) + ".inventory.json").read_text())
                with tarfile.open(target) as archive:
                    self.assertIn("training/weights/best.pt", archive.getnames())
                    self.assertIn("training/train_batch0.jpg", archive.getnames())
                    meta = json.load(archive.extractfile("metadata/package.json"))
                    self.assertEqual(meta["status"], "missing_evidence")
                    self.assertEqual(meta["evaluation"]["test"]["status"], "NOT_RUN")
                inventory[0]["sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "Corrupt archive"): export.verify_archive(target, inventory)
                with self.assertRaisesRegex(RuntimeError, "Preserve old"): export.package_complete("c25", target)


if __name__ == "__main__":
    unittest.main()
