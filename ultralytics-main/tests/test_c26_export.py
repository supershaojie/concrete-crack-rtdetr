"""Three-image synthetic end-to-end export fixture; never evaluates the crack dataset."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from init_c26 import build, require, write_json
from c26_results import evaluate, sha256
from ultralytics.utils import YAML


class TestC26Export(unittest.TestCase):
    def test_three_image_stream_same_pass_and_test_selection(self):
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory(prefix="c26_export_", dir=ROOT / "outputs") as d:
            root = Path(d)
            for kind in ("images", "labels"):
                (root / kind).mkdir()
            for i, (h, w) in enumerate(((96, 160), (160, 96), (128, 128))):
                image = np.full((h, w, 3), 150, np.uint8)
                cv2.line(image, (w // 3, h // 4), (w // 2, h * 3 // 4), (30, 30, 30), 2)
                cv2.imwrite(str(root / f"images/{i}.jpg"), image)
                (root / f"labels/{i}.txt").write_text("0 0.5 0.5 0.3 0.5\n" if i < 2 else "", encoding="utf-8")
            data = root / "fixture.yaml"
            YAML.save(data, dict(path=str(root), train="images", val="images", test="images", names={0: "crack"}))
            model = build(nc=1).eval()
            with torch.no_grad():
                model.model[-1].cbr.offset_out.bias.fill_(.25)
                model.model[9].scca_o.weight.normal_(std=.001)
            weights = root / "fixture.pt"
            torch.save(dict(epoch=-1, model=deepcopy(model), train_args={"task": "detect"}), weights)
            val = evaluate(weights, data, "val", root / "val", device="cpu")
            test = evaluate(weights, data, "test", root / "test", device="cpu", val_report=root / "val/metrics.json")
            for split, report in (("val", val), ("test", test)):
                self.assertEqual(report["status"], "completed")
                self.assertEqual((report["images"], report["predictions"], report["ground_truth"]), (3, 900, 2))
                self.assertEqual(report["parameters_unfused"], 20194201)
                self.assertEqual(np.asarray(report["ap_by_class"]).shape, (1, 10))
                self.assertEqual(report["actual_settings"]["batch"], 16)
                with gzip.open(root / split / "predictions_gt.jsonl.gz", "rt", encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f]
                self.assertEqual(len(rows), 3)
                self.assertEqual({r["image"] for r in rows}, {f"images/{i}.jpg" for i in range(3)})
                self.assertTrue(all(len(r["predictions"]) == 300 for r in rows))
                self.assertTrue(list((root / split / "plots").glob("*pred.jpg")))
                self.assertTrue((root / split / "plots/BoxPR_curve.png").is_file())
            with self.assertRaises(RuntimeError):
                evaluate(weights, data, "val", root / "val", device="cpu")
            modified = root / "modified.pt"
            modified.write_bytes(weights.read_bytes() + b"different hash")
            with self.assertRaises(RuntimeError):
                evaluate(modified, data, "test", root / "wrong_test", device="cpu", val_report=root / "val/metrics.json")
            require(not (root / "wrong_test").exists(), "Mismatched test must fail before inference/output creation")
            evidence = dict(scope="Synthetic 3-image fixture, same images for fake val/test solely to exercise plumbing; NOT formal metrics",
                            val=val, test=test, temporary_outputs_discarded=True)
            # Report only; no debug weight or dataset can enter Git/formal initialization.
            write_json(ROOT / "outputs/c26_export_fixture.json", evidence)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    unittest.main()
