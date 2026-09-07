"""Targeted unittest checks; synthetic fixtures are NOT trained-checkpoint experiments."""
from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cbr_rescue_common import ROOT, sha256, write_json, package, canonical, git, C17, C19, MODULE_DIR
TEST_OUTPUT = ROOT / "outputs/cbr_rescue_local_tests"
TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
(TEST_OUTPUT / "settings").mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(TEST_OUTPUT / "settings")
os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import numpy as np
import torch
from PIL import Image
from ultralytics.models.rtdetr.val import RTDETRValidator, RTDETRDataset
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules.cbr import RTDETRDecoderCBR
from ultralytics.utils.metrics import box_iou
from cbr_rescue_analysis import (AlignedValidator, MaskObserver, raw_tensor, fixed_pairs, cscef_observer,
                                  cbr_rows, distribution, numeric_summary)
from diagnose_cbr_rescue import SETTINGS, build_dataset, manifest, select_fixed, evaluate_model, metric_result, conclusion

torch.set_num_threads(4)


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=TEST_OUTPUT)
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def validator(self, cls=RTDETRValidator):
        return cls(save_dir=self.root / cls.__name__, args=SETTINGS)

    def test_mask_known_mismatch_and_nonmutation(self):
        raw = torch.tensor([[[.5, .5, .2, .2, .9], [.4, .4, .1, .1, .0005],
                             [.2, .2, .1, .1, .8], [.3, .3, .2, .2, .001]]])
        untouched = raw.clone()
        observer = MaskObserver()
        observer.observe((raw, None), ["image"])
        torch.testing.assert_close(raw, untouched, rtol=0, atol=0)
        self.assertEqual(observer.low_queries, 2)
        self.assertEqual(observer.different_queries, 2)
        self.assertEqual(observer.examples[0]["query_indices"], [2, 3])
        native, aligned = self.validator(), self.validator(AlignedValidator)
        old, new = native.postprocess(raw.clone()), aligned.postprocess(raw)
        self.assertEqual(len(old[0]["conf"]), 2)
        self.assertLess(float(old[0]["conf"][-1]), .0011)
        torch.testing.assert_close(new[0]["conf"], torch.tensor([.9, .8]))
        torch.testing.assert_close(raw, untouched, rtol=0, atol=0)
        again = aligned.postprocess(raw)
        torch.testing.assert_close(new[0]["bboxes"], again[0]["bboxes"], rtol=0, atol=0)

    def test_unaffected_native_aligned_and_independent_scaling(self):
        raw = torch.tensor([[[.5, .5, .2, .2, .2], [.4, .3, .1, .1, .9]]])
        before = raw.clone()
        before[..., 0] -= .01
        observer = MaskObserver()
        observer.observe(raw, ["x"])
        self.assertEqual(observer.different_queries, 0)
        native, aligned = self.validator(), self.validator(AlignedValidator)
        for value in (raw, before):
            old, new = native.postprocess(value.clone()), aligned.postprocess(value)
            for k in old[0]:
                torch.testing.assert_close(old[0][k], new[0][k], rtol=0, atol=0)
        a, b = aligned.postprocess(raw)[0], aligned.postprocess(before)[0]
        torch.testing.assert_close(a["bboxes"][:, 0] - b["bboxes"][:, 0], torch.full((2,), 6.4), atol=3e-5, rtol=0)
        self.assertLess(float(raw[..., :4].max()), 1)

    def test_fixed_pairing_survives_swapped_after(self):
        gt = torch.tensor([[0., 0., .2, .2], [.8, .8, 1., 1.]])
        before, after = gt.clone(), gt.flip(0)
        pairs = fixed_pairs(before, gt, [0, 1], torch.zeros(2), torch.zeros(2))
        self.assertEqual([(g, q) for g, q, _ in pairs], [(0, 0), (1, 1)])
        self.assertTrue(all(float(box_iou(gt[g:g+1], after[q:q+1])) == 0 for g, q, _ in pairs))
        rematched = fixed_pairs(after, gt, [0, 1], torch.zeros(2), torch.zeros(2))
        self.assertNotEqual([(g, q) for g, q, _ in pairs], [(g, q) for g, q, _ in rematched])
        self.assertEqual(fixed_pairs(before, gt[:0], [0], torch.zeros(2), torch.zeros(0)), [])
        self.assertEqual(fixed_pairs(before, gt, [], torch.zeros(2), torch.zeros(2)), [])
        self.assertEqual(fixed_pairs(before, gt, [0, 1], torch.ones(2), torch.zeros(2)), [])
        low = fixed_pairs(before + 2, gt, [0, 1], torch.zeros(2), torch.zeros(2))
        self.assertEqual(len(low), 2)
        self.assertTrue(all(v == 0 for _, _, v in low))

    def test_mask_examples_are_bounded(self):
        raw = torch.zeros(20, 3, 5)
        raw[..., 4] = torch.tensor([.0001, .8, .9])
        observer = MaskObserver()
        observer.observe(raw, list(map(str, range(20))))
        self.assertEqual(observer.different_images, 20)
        self.assertEqual(len(observer.examples), 16)

    def test_official_metrics_are_used(self):
        raw = torch.tensor([[[.5, .5, .2, .2, .9]]])
        batch = {"img": torch.zeros(1, 3, 640, 640), "batch_idx": torch.tensor([0]),
                 "cls": torch.tensor([[0.]]), "bboxes": raw[0, :, :4].clone(),
                 "ori_shape": [(100, 200)], "ratio_pad": [(6.4, 3.2)], "im_file": ["synthetic"]}
        results = []
        for cls in (RTDETRValidator, AlignedValidator):
            v = self.validator(cls)
            v.device, v.training, v.data = torch.device("cpu"), False, {"val": "fixture"}
            v.init_metrics(type("ModelNames", (), {"names": {0: "crack"}})())
            v.update_metrics(v.postprocess(raw.clone()), copy.deepcopy(batch))
            results.append(metric_result(v))
        self.assertEqual(results[0], results[1])
        self.assertGreater(results[0]["mAP50_95"], .99)

    def make_data(self):
        (self.root / "images/val").mkdir(parents=True)
        (self.root / "labels/val").mkdir(parents=True)
        rng = np.random.default_rng(42)
        for i in range(2):
            Image.fromarray(rng.integers(0, 255, (55 + i, 91, 3), dtype=np.uint8)).save(self.root / f"images/val/{i}.png")
            (self.root / f"labels/val/{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n" if i == 0 else "", encoding="utf-8")
        return {"names": {0: "crack"}, "nc": 1, "channels": 3, "val": str(self.root / "images/val")}

    def test_native_preprocessing_and_fixed_evidence(self):
        data = self.make_data()
        settings = {**SETTINGS, "imgsz": 128, "batch": 2, "device": "cpu"}
        dataset = build_dataset(data, "val", self.root / "diagnosis", settings)
        hashes = manifest(dataset)
        fixed = select_fixed(hashes, self.root / "missing.gz", count=2)
        self.assertIn("new stable", fixed["origin"])
        evidence = self.root / "evidence.gz"
        with gzip.open(evidence, "wt", encoding="utf-8") as f:
            for row in reversed(hashes):
                f.write(json.dumps(row) + "\n")
        restored = select_fixed(hashes, evidence, count=2)
        self.assertEqual(restored["rows"], list(reversed(hashes)))
        self.assertFalse(list(self.root.rglob("*.cache")))
        # Native loader on the SAME files: disable only cache persistence, not transforms.
        with patch("ultralytics.data.dataset.save_dataset_cache_file", lambda prefix, path, x, version: x.update(version=version)):
            native = RTDETRDataset(img_path=data["val"], imgsz=128, batch_size=2, augment=False,
                                   hyp=self.validator().args, rect=False, cache=False, data=data)
        for i in range(2):
            a, b = dataset[i], native[i]
            for key in ("img", "bboxes", "cls"):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(manifest(dataset), hashes)
        hashes[0] = {**hashes[0], "label_sha256": "changed"}
        with self.assertRaisesRegex(RuntimeError, "changed"):
            select_fixed(hashes, evidence, count=2)

    def test_unfused_real_model_interfaces_and_pipeline(self):
        """Random 128px fixtures only; build no optimizer, loss, backward, train or smoke."""
        data = self.make_data()
        settings = {**SETTINGS, "imgsz": 128, "batch": 2, "device": "cpu"}
        dataset = build_dataset(data, "val", self.root / "fixture", settings)
        fixed = select_fixed(manifest(dataset), None, count=2)
        torch.manual_seed(42)
        for name, config in (("C2", "rtdetr-resnet18-lite.yaml"),
                             ("C17", "rtdetr-resnet18-lite-cscef-v51.yaml"),
                             ("C19", "rtdetr-resnet18-lite-cbr.yaml"),
                             ("C20", "rtdetr-resnet18-lite-cscef-cbr.yaml")):
            model = RTDETRDetectionModel(str(ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr" / config),
                                        nc=1, verbose=False).float().eval()
            # Make synthetic effects nonzero so clone/parity/IoU checks cannot pass on identity alone.
            with torch.no_grad():
                if name in {"C19", "C20"}:
                    head = next(m for m in model.modules() if isinstance(m, RTDETRDecoderCBR))
                    head.cbr.offset_out.bias.fill_(.3)
                if name in {"C17", "C20"}:
                    from ultralytics.nn.modules.cscef_v51 import CSCEFv51
                    next(m for m in model.modules() if isinstance(m, CSCEFv51)).output_projection.weight.normal_(0, .01)
            hooks = sum(len(m._forward_hooks) for m in model.modules())
            if name == "C20":
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with cscef_observer(model, [0], ["x"], []):
                        raise RuntimeError("intentional")
                self.assertEqual(sum(len(m._forward_hooks) for m in model.modules()), hooks)
            checkpoint = self.root / f"{name}_synthetic.pt"
            torch.save({"model": model, "train_args": {}}, checkpoint)
            old_hash = sha256(checkpoint)
            report = evaluate_model(name, checkpoint, "val", dataset, data, fixed, self.root / "pipeline", settings)
            self.assertEqual(report["paired_same_forward"], name in {"C19", "C20"})
            self.assertEqual(report["images"], 2)
            self.assertEqual(len(report["metrics"]), 4 if name in {"C19", "C20"} else 2)
            if name in {"C19", "C20"}:
                self.assertEqual(report["cbr"]["query_summaries"]["all"]["n_rows"], 600)
            self.assertTrue(all(p["hooks_removed"] for p in report["output_parity"]))
            self.assertEqual(sha256(checkpoint), old_hash)
            if name in {"C17", "C20"}:
                self.assertEqual(report["cscef"]["n_rows"], 2)
            if name == "C20" and torch.cuda.is_available():
                model.to("cuda")
                image = torch.rand(1, 3, 640, 640, device="cuda")
                rows = []
                with torch.inference_mode():
                    ordinary = raw_tensor(model.predict(image)).clone()
                    with cscef_observer(model, [0], ["synthetic_640_cuda"], rows):
                        diagnostic, details = model.predict(image, cbr_diagnostics=True)
                    torch.testing.assert_close(ordinary, raw_tensor(diagnostic), atol=1e-6, rtol=1e-5)
                    self.assertEqual(ordinary.shape, (1, 300, 5))
                    self.assertEqual(len(rows), 1)
                    self.assertGreater(rows[0]["residual_ratio"], 0)
                model.cpu()
                del image, ordinary, diagnostic, details
                torch.cuda.empty_cache()
            del model

    def test_sources_equal_c17_c19(self):
        for name, ref in (("cscef_v5.py", C17), ("cscef_v51.py", C17), ("cbr.py", C19)):
            self.assertEqual(canonical((ROOT / MODULE_DIR / name).read_bytes()), canonical(git("show", f"{ref}:{MODULE_DIR}{name}")))

    def test_package_hashes_allowlist_size_and_overwrite(self):
        run = self.root / "run"
        run.mkdir()
        write_json(run / "summary.json", {"status": "completed", "artifacts": ["summary.json", "metrics.json"]})
        write_json(run / "metrics.json", {"fixture_only": True})
        (run / "exclude.pt").write_bytes(b"never package")
        output = self.root / "result.tar.gz"
        package(run, output)
        with tarfile.open(output) as tar:
            self.assertNotIn("exclude.pt", tar.getnames())
            for row in json.load(tar.extractfile("CONTENTS.json")):
                import hashlib
                self.assertEqual(hashlib.sha256(tar.extractfile(row["name"]).read()).hexdigest(), row["sha256"])
        self.assertTrue(output.with_name(output.name + ".sha256").read_text().startswith(sha256(output)))
        with self.assertRaisesRegex(RuntimeError, "existing"):
            package(run, output)
        write_json(run / "summary.json", {"status": "completed", "artifacts": ["summary.json", "exclude.pt"]})
        with self.assertRaisesRegex(RuntimeError, "Excluded"):
            package(run, self.root / "bad.tar.gz")
        write_json(run / "summary.json", {"status": "completed", "artifacts": ["summary.json", "../secret.txt"]})
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            package(run, self.root / "bad.tar.gz")
        (run / "large.log").write_bytes(os.urandom(21 * 1024 * 1024))
        write_json(run / "summary.json", {"status": "completed", "artifacts": ["summary.json", "large.log"]})
        with self.assertRaisesRegex(RuntimeError, "20 MiB"):
            package(run, self.root / "large.tar.gz")
        self.assertFalse((self.root / "large.tar.gz").exists())

    def test_statistics_empty_and_conclusion_no_test_dependency(self):
        self.assertIsNone(distribution([])["mean"])
        self.assertEqual(numeric_summary([])["n_rows"], 0)
        reports = {}
        for name in ("C17", "C19", "C20"):
            reports[name + "_val"] = {"metrics": {route + "_aligned": {"mAP50_95": .5} for route in ("before", "after")}}
        first = conclusion(reports, "aligned")
        reports["C20_test"] = {"metrics": {"mAP50_95": 1}}
        self.assertEqual(first, conclusion(reports, "aligned"))

    def test_cbr_saturation_direction_and_empty_high_score_scope(self):
        before = torch.tensor([.5, .5, .2, .1]).repeat(300, 1)
        tanh = torch.tensor([.96, -.96, 0., .5]).repeat(300, 1)
        displacement = .1 * before[:, [2, 2, 3, 3]] * tanh
        dl, dr, dt, db = displacement.unbind(-1)
        after = before + torch.stack(((dl + dr) / 2, (dt + db) / 2, dr - dl, db - dt), -1)
        details = dict(before=before, after=after, tanh_offsets=tanh, displacement=displacement,
                       aggregation_weights=torch.full((300, 4, 3), 1/3))
        raw = torch.cat((after, torch.full((300, 1), .1)), -1)
        rows, pairs, images = cbr_rows(details, raw, torch.tensor([[.4, .45, .6, .55]]), torch.tensor([0]), "fixture")
        self.assertAlmostEqual(rows[0]["left_relative"], .096, places=6)
        self.assertEqual(rows[0]["left_saturated"], 1)
        self.assertEqual(rows[0]["right_negative"], 1)
        self.assertEqual(rows[0]["top_zero"], 1)
        self.assertEqual(images[1]["no_candidates"], 1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["query"], 0)  # stable tie resolved by original query index
        self.assertEqual(pairs[0]["worsened"], 1)

    def test_dataset_never_repairs_jpeg_or_deletes_npy_cache(self):
        data = self.make_data()
        cache = self.root / "images/val/0.npy"
        cache.write_bytes(b"invalid npy must not be removed")
        with self.assertRaisesRegex(RuntimeError, "npy"):
            build_dataset(data, "val", self.root / "npy_check", SETTINGS)
        self.assertEqual(cache.read_bytes(), b"invalid npy must not be removed")
        cache.unlink()  # remove test-owned fixture only
        image = self.root / "images/val/bad.jpg"
        Image.new("RGB", (20, 20)).save(image)
        image.write_bytes(image.read_bytes()[:-2])
        before = sha256(image)
        with self.assertRaisesRegex(RuntimeError, "JPEG needs repair"):
            build_dataset(data, "val", self.root / "jpeg_check", SETTINGS)
        self.assertEqual(sha256(image), before)

    def test_orchestration_catches_late_model_mismatch_and_exports(self):
        """Mock ONLY inference for scheduler/export coverage; real inference tested separately."""
        import diagnose_cbr_rescue as driver
        from ultralytics.utils import YAML
        data = self.make_data()
        for i in range(2, 32):
            Image.new("RGB", (20, 30), (i, i, i)).save(self.root / f"images/val/{i:02}.png")
            (self.root / f"labels/val/{i:02}.txt").write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
        (self.root / "images/test").mkdir()
        Image.new("RGB", (20, 30)).save(self.root / "images/test/test.png")
        data_file = self.root / "data.yaml"
        YAML.save(data_file, {"path": str(self.root), "train": "images/val", "val": "images/val", "test": "images/test", "names": {0: "crack"}})
        weights = {}
        for name in ("c2", "c17", "c19", "c20", "initialization"):
            weights[name] = self.root / f"{name}_scheduler_fixture.pt"
            weights[name].write_bytes(b"scheduler test placeholder, never loaded as a checkpoint")
        for mismatch in (None, "C17", "C19"):
            calls = []
            def fake_evaluate(name, weights_path, split, dataset, *args):
                calls.append((name, split))
                stats = numeric_summary([{"residual_ratio": .1}])
                return {"split": split, "mask": {"different_queries": int(name == mismatch)},
                        "metrics": {route + "_" + policy: {"mAP50_95": .5} for route in ("after", "before")
                                    for policy in ("legacy", "aligned")}, "cscef": stats,
                        "cbr": {"query_summaries": {scope: stats for scope in ("all", "score_ge_025")}}}
            output = self.root / f"orchestration_{mismatch}"
            args = SimpleNamespace(output=output, main=self.root, data=data_file, evidence=None, device="cpu", **weights)
            with patch.object(driver, "import_runtime", return_value={"synthetic_scheduler_fixture": True}), \
                 patch.object(driver, "source_state", return_value={"status": "", "fixture_only": True}), \
                 patch.object(driver, "C20_SHA", sha256(weights["c20"])), \
                 patch.object(driver, "INIT_SHA", sha256(weights["initialization"])), \
                 patch.object(driver, "evaluate_model", side_effect=fake_evaluate):
                driver.run(args)
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["comparison_policy"], "legacy" if mismatch is None else "aligned")
            self.assertEqual(len(calls), 4 if mismatch is None else 8)
            self.assertEqual(calls[:4], [("C20", "val"), ("C20", "test"), ("C17", "val"), ("C19", "val")])
            if mismatch:
                self.assertEqual(set(calls), {(n, s) for n in ("C2", "C17", "C19", "C20") for s in ("val", "test")})
            self.assertTrue(report["dataset_content_unchanged"])
            (output / "console.log").write_text("Synthetic scheduler fixture only.\n", encoding="utf-8")
            package(output, self.root / f"scheduler_{mismatch}.tar.gz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
