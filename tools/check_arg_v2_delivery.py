"""Real temporary export/lock/package IO; synthetic evaluator and completed-run fixtures.

No actual formal val/test inference or training is performed by these tests.
"""
from contextlib import ExitStack
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import arg_v2 as app
import arg_v2_common as common
import arg_v2_evaluation as evaluation
import arg_v2_package as packaging
from ultralytics.models.rtdetr.arg_v2_val import ARGv2Validator, EVAL
from ultralytics.models.rtdetr.val import RTDETRValidator


class Completion(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.out, self.run = self.root / "out", self.root / "run"
        self.out.mkdir(); (self.run / "weights").mkdir(parents=True)
        (self.run / "weights/best.pt").write_bytes(b"locked best fixture")
        (self.run / "weights/last.pt").write_bytes(b"last fixture")
        (self.run / "results.csv").write_text("epoch,metrics\n1,fixture\n")
        (self.root / "source.py").write_text("# source fixture\n")
        import tarfile
        with tarfile.open(self.out / "source_snapshot.tar.gz", "w:gz") as archive:
            archive.add(self.root / "source.py", arcname="source.py")
        code = dict(commit="a" * 40, files={"source.py": common.sha256(self.root / "source.py")})
        self.training = dict(binding=dict(code=code, data={"snapshot": "frozen fixture"}))
        for name in ("prepare.json", "data_snapshot.json", "initialization.json", "recipe_diff.json", "actual_setup.json"):
            common.write_json(self.out / name, dict(status="PASS", scope="fixture"))
        common.write_json(self.out / "data_snapshot.json", dict(status="PASS", snapshot_id=common.digest(self.training["binding"]["data"])))
        (self.out / "train_args.yaml").write_text("seed: 42\n")
        (self.out / "data_manifest.jsonl.gz").write_bytes(gzip.compress(b"fixture manifest\n"))
        checks = {k: dict(status="PASS") for k in ("cpu", "cuda_b16_amp", "mechanism", "new_process_val", "resume", "native_scale")}
        checks["cuda_b16_amp"].update(batch=16, imgsz=640, amp=True, effective_updates=1)
        common.write_json(self.out / "preflight.json", dict(status="PASS", start_eligible=True,
            binding=self.training["binding"], checks=checks, gpu=dict(micro_batches=4), scope="fixture only"))
        for name in ("dispatch.json", "worker.json", "exit.json"):
            common.write_json(self.out / "dispatches/fixture" / name, dict(scope="fixture"))
        (self.out / "dispatches/fixture/python_exit_code.txt").write_text("0")
        common.write_json(self.out / "training_identity.json", self.training)
        common.write_json(self.out / "training_completed.json", dict(status="TRAINING_COMPLETED"))
        (self.out / "epochs.jsonl").write_text('{"epoch":0}\n')
        self.context = dict(training=self.training, completion={"status": "TRAINING_COMPLETED"},
            eval_identity=dict(training_commit=code["commit"], eval_commit=code["commit"]),
            data=self.training["binding"]["data"], data_path=str(self.root / "never_read.yaml"),
            best=str(self.run / "weights/best.pt"), checkpoint_sha256=common.sha256(self.run / "weights/best.pt"))
        for module in (app, packaging):
            self.stack.enter_context(patch.multiple(module, ROOT=self.root, OUT=self.out, RUN=self.run))
        self.stack.enter_context(patch.multiple(evaluation, OUT=self.out, RUN=self.run, COUNTS=dict(val=(2, 2), test=(1, 1))))
        self.stack.enter_context(patch.object(app, "active_workers", return_value=[]))
        self.stack.enter_context(patch.object(app, "has_tmux", return_value=False))
        self.stack.enter_context(patch.object(app, "code_identity", return_value=code))
        self.stack.enter_context(patch.object(app, "git", return_value="source.py"))
        self.stack.enter_context(patch.object(packaging, "git", return_value="source.py"))
        self.stack.enter_context(patch.object(evaluation, "evaluation_context", return_value=self.context))
        self.stack.enter_context(patch.object(app, "checkpoint_identity", return_value={}))
        self.stack.enter_context(patch.object(common, "inventory", side_effect=AssertionError("Unexpected raw inventory")))
        self.calls, self.fail_test = [], False
        owner = self

        class Evaluator:
            def __init__(self, args, save_dir):
                self.args = args
                self.actual_settings = dict(args)
                self.export_path = None
                self.metrics = SimpleNamespace(box=SimpleNamespace(mp=.5, mr=.25,
                    all_ap=np.arange(10).reshape(1, 10) / 20, px=np.linspace(0, 1, 1000),
                    p_curve=np.full((1, 1000), .5), r_curve=np.full((1, 1000), .25),
                    f1_curve=np.full((1, 1000), 1/3), prec_values=np.full((1, 1000), .5)))

            def __call__(self, model):
                split = self.args["split"]
                owner.calls.append(split)
                if split == "test" and owner.fail_test:
                    owner.fail_test = False
                    raise RuntimeError("injected test-stage failure")
                count = evaluation.COUNTS[split][0]
                self.arg_seen = set(str(i) for i in range(count))
                self.arg_gt_count = count
                with gzip.open(self.export_path, "xt", encoding="utf-8") as stream:
                    for i in range(count):
                        stream.write(json.dumps(dict(image=str(i), query_indices=list(range(300)),
                            scores=[0.00001] * 300, gt_boxes=[[1, 1, 2, 2]], identity=self.export_identity)) + "\n")
                return {"metrics/precision(B)": .5, "metrics/recall(B)": .25}

        self.stack.enter_context(patch.object(evaluation, "ARGv2Validator", Evaluator))

    def test_finish_once_and_reentry_reuses_evaluation_and_package(self):
        first = app.finish()
        second = app.finish()
        self.assertEqual(self.calls, ["val", "test"])
        self.assertEqual(first["package"], second["package"])
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(len(list((self.out / "packages").glob("*.tar.gz"))), 1)
        import tarfile
        with tarfile.open(first["package"]["path"]) as archive:
            summary = json.load(archive.extractfile("summary.json"))
            self.assertTrue(summary["analysis_complete"])
            self.assertEqual(summary["formal_metrics"]["val"]["AP75"], .25)
            self.assertAlmostEqual(summary["formal_metrics"]["val"]["F1"], 1/3)
            self.assertEqual(len([p for p in archive.getnames() if p.endswith("predictions_gt.jsonl.gz")]), 2)
            self.assertFalse(any(p.endswith(".pt") for p in archive.getnames()))
        app.status(); app.pack()
        self.assertEqual(self.calls, ["val", "test"])

    def test_final_eval_val_lock_reused_by_finish(self):
        from ultralytics.models.rtdetr.arg_v2_model import ARGv2Trainer
        from ultralytics.utils import torch_utils
        trainer = SimpleNamespace(arg_output=self.out, epoch=0, epochs=200,
            best=self.run / "weights/best.pt", last=self.run / "weights/last.pt",
            run_callbacks=lambda event: None)
        with patch.multiple(common, OUT=self.out, RUN=self.run), \
             patch.object(torch_utils, "strip_optimizer", return_value={"train_results": "fixture"}) as strip:
            ARGv2Trainer.final_eval(trainer)
        self.assertEqual(strip.call_count, 2)
        self.assertEqual(common.read_json(self.out / "final_eval.json")["status"], "PASS")
        self.assertEqual(self.calls, ["val"])
        app.finish()
        self.assertEqual(self.calls, ["val", "test"])

    def test_failure_resume_only_missing_stage_preserves_error(self):
        self.fail_test = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            app.finish()
        failure = common.read_json(self.out / "finish.json")
        self.assertEqual(failure["phase"], "test")
        self.assertEqual(failure["status"], "FAIL")
        app.finish()
        self.assertEqual(self.calls, ["val", "test", "test"])
        self.assertTrue(list((self.out / "recoveries").glob("*_finish_failure.json")))
        self.assertTrue(any(common.read_json(p)["status"] == "FAIL" for p in (self.out / "evaluations").glob("*/metrics.json")))

    def test_complete_report_recovers_missing_lock_without_inference(self):
        first = evaluation.evaluate("val")
        (self.out / "val_lock.json").unlink()
        recovered = evaluation.evaluate("val")
        self.assertEqual(recovered, first)
        self.assertEqual(self.calls, ["val"])

    def test_lock_write_failure_preserves_complete_report_and_does_not_reinfer(self):
        def publish(path, value):
            if Path(path) == self.out / "val_lock.json":
                raise OSError("injected lock publication failure")
            return common.write_json(path, value)
        with patch.object(evaluation, "write_json", side_effect=publish):
            with self.assertRaisesRegex(OSError, "lock publication failure"):
                app.finish()
        self.assertFalse((self.out / "val_lock.json").exists())
        reports = list((self.out / "evaluations").glob("*/metrics.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(common.read_json(reports[0])["status"], "PASS")
        failure = common.read_json(reports[0].with_name("publication_failure.json"))
        self.assertEqual(failure["status"], "FAIL")
        self.assertTrue(failure["evaluation_completed"])
        self.assertEqual(app.finish()["status"], "PASS")
        self.assertEqual(self.calls, ["val", "test"])
        self.assertEqual(common.read_json(reports[0].with_name("publication_failure.json")), failure)

    def test_missing_export_refuses_implicit_replay_and_pack_reports_gap(self):
        first = evaluation.evaluate("val")
        Path(first["predictions"]["path"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "recover-export"):
            evaluation.evaluate("val")
        result = app.pack()
        app.status()
        self.assertFalse(result["analysis_complete"])
        self.assertEqual(self.calls, ["val"])
        evaluation.evaluate("val", recover_export=True)
        self.assertEqual(self.calls, ["val", "val"])
        self.assertTrue(list((self.out / "evaluations").glob("*/previous_lock.json")))

    def test_test_requires_same_best_as_val_and_never_reselects(self):
        evaluation.evaluate("val")
        self.context["checkpoint_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "identity/protocol differs"):
            evaluation.evaluate("test")
        self.assertEqual(self.calls, ["val"])

    def test_completed_package_recovers_missing_receipt_without_repack(self):
        first = app.finish()["package"]
        (self.out / "package.json").unlink()
        second = app.pack()["package"]
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.out / "packages").glob("*.tar.gz"))), 1)
        self.assertEqual(self.calls, ["val", "test"])

    def test_interrupted_pack_retained_and_only_pack_retried(self):
        evaluation.evaluate("val"); evaluation.evaluate("test")
        with patch.object(packaging, "verify_archive", side_effect=RuntimeError("injected packaging interruption")):
            with self.assertRaisesRegex(RuntimeError, "packaging interruption"):
                app.pack()
        self.assertTrue(list((self.out / "packages").glob("*.partial")))
        result = app.pack()
        self.assertTrue(result["analysis_complete"])
        self.assertTrue(list((self.out / "packages").glob("*.interrupted.*")))
        self.assertEqual(self.calls, ["val", "test"])


class QueryExport(unittest.TestCase):
    def test_real_exporter_keeps_all_queries_gt_and_original_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            validator = ARGv2Validator.__new__(ARGv2Validator)
            validator.args = SimpleNamespace(imgsz=640, conf=.001)
            validator.data = {"path": str(root)}
            validator.arg_seen, validator.arg_gt_count = set(), 0
            validator.export_path = root / "queries.jsonl.gz"
            validator.export_identity = {"scope": "synthetic exporter test"}
            raw = torch.full((1, 300, 5), .5)
            raw[0, 0, 4] = .00001
            gt = dict(im_file=str(root / "images/one.jpg"), ori_shape=(100, 200), imgsz=(640, 640),
                      bboxes=torch.tensor([[160., 160., 480., 480.]]), cls=torch.tensor([0.]))
            with gzip.open(validator.export_path, "xt", encoding="utf-8") as stream:
                validator.arg_stream = stream
                with patch.object(validator, "_prepare_batch", return_value=gt), \
                     patch.object(RTDETRValidator, "update_metrics", return_value=None):
                    filtered = validator.postprocess(raw)
                    self.assertEqual(len(filtered[0]["conf"]), 299)
                    validator.update_metrics(filtered, {})
            with gzip.open(validator.export_path, "rt", encoding="utf-8") as stream:
                row = json.loads(stream.readline())
            self.assertEqual(row["query_indices"], list(range(300)))
            self.assertEqual(len(row["boxes"]), 300)
            self.assertLess(row["scores"][0], .001)
            self.assertEqual(row["gt_boxes"], [[50., 25., 150., 75.]])
            self.assertEqual(row["boxes"][0], [50., 25., 150., 75.])
            self.assertEqual(validator.arg_gt_count, 1)


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (Completion, QueryExport))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    common.write_json(common.OUT / "local_delivery.json", dict(status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, scope="synthetic evaluator/completed-run fixtures; real exporter, locks, archives, hashes and recovery IO; no formal val/test inference"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
