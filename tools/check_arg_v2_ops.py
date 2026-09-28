"""Fail-closed lifecycle control tests; mocked server controls, real local package IO."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys
import time
import unittest
from unittest.mock import patch

import arg_v2 as app
import arg_v2_package as packaging
from arg_v2_common import OUT, write_json


class Lifecycle(unittest.TestCase):
    def test_real_child_deadline(self):
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            app.run_bounded([sys.executable, "-c", "import time; time.sleep(10)"], .3, "owned-child deadline fixture")
        self.assertLess(time.monotonic() - start, 8)

    def preflight(self):
        checks = {k: dict(status="PASS") for k in ("cpu", "cuda_b16_amp", "mechanism", "new_process_val", "resume", "native_scale")}
        checks["cuda_b16_amp"].update(batch=16, imgsz=640, amp=True, effective_updates=1)
        return dict(status="PASS", start_eligible=True, binding={"code": "fixture"}, checks=checks, gpu=dict(micro_batches=4))

    def test_all_gates_independent(self):
        pre = self.preflight()
        app.verify_preflight(pre, pre["binding"])
        for key in ("cpu", "cuda_b16_amp", "mechanism", "new_process_val", "resume"):
            broken = deepcopy(pre); broken["checks"][key]["status"] = "PENDING"
            with self.assertRaises(RuntimeError):
                app.verify_preflight(broken, pre["binding"])
        for field, value in (("batch", 2), ("imgsz", 160), ("amp", False), ("effective_updates", 0)):
            broken = deepcopy(pre); broken["checks"]["cuda_b16_amp"][field] = value
            with self.assertRaises(RuntimeError):
                app.verify_preflight(broken, pre["binding"])
        with self.assertRaises(RuntimeError):
            app.verify_preflight(pre, {"code": "changed"})

    def test_fallback_is_explicit(self):
        pre = self.preflight(); pre["status"] = "PENDING"
        pre["checks"]["native_scale"]["status"] = "PENDING"
        with self.assertRaises(RuntimeError):
            app.verify_preflight(pre, pre["binding"])
        pre["checks"]["fallback_scale"] = dict(status="PASS")
        pre["gpu"]["effective_update_arm"] = "diagnostic_init_scale_128"
        app.verify_preflight(pre, pre["binding"])
        self.assertEqual(pre["checks"]["native_scale"]["status"], "PENDING")

    def test_lock_rejects_live_owner(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app, "OUT", Path(folder)):
            with app.operation_lock():
                with self.assertRaises(RuntimeError):
                    with app.operation_lock():
                        self.fail("Live lock accepted")
            self.assertFalse((Path(folder) / "operation.lock").exists())

    def test_eval_repairs_preserve_training_identity(self):
        old = dict(commit="training-sha", files={"ultralytics-main/ultralytics/nn/autobackend.py": "before", "tools/train.py": "same"})
        training = dict(binding=dict(code=deepcopy(old)))
        current = dict(commit="eval-sha", files=dict(old["files"]))
        current["files"]["ultralytics-main/ultralytics/nn/autobackend.py"] = "after"
        with patch.object(app, "code_identity", return_value=current):
            self.assertEqual(app.evaluation_source(training)["training_commit"], "training-sha")
        current["files"]["tools/train.py"] = "changed"
        with patch.object(app, "code_identity", return_value=current), self.assertRaises(RuntimeError):
            app.evaluation_source(training)
        self.assertEqual(training["binding"]["code"], old)

    def test_failure_pack_without_test_or_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root / "outputs"; out.mkdir()
            (root / "source.py").write_text("# fixture\n")
            write_json(out / "failure.json", {"status": "FAIL", "value": float("nan")})
            with patch.multiple(app, ROOT=root, OUT=out, RUN=root / "absent_run"), \
                 patch.multiple(packaging, ROOT=root, OUT=out, RUN=root / "absent_run"), \
                 patch.object(packaging, "git", return_value="source.py"), \
                 patch.object(app, "git", return_value="source.py"), patch.object(app, "status", return_value={"training": "FAILED"}), \
                 patch.object(app, "code_identity", return_value={"commit": "fixture"}):
                result = app.pack()
                self.assertTrue(Path(result["package"]["path"]).exists())
                self.assertFalse(result["analysis_complete"])
                import tarfile
                with tarfile.open(result["package"]["path"]) as archive:
                    failure = json.load(archive.extractfile("evidence/failure.json"))
                    self.assertIsNone(failure["value"]["value"])
                    self.assertFalse(any(name.endswith(".pt") for name in archive.getnames()))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Lifecycle))
    write_json(OUT / "local_ops.json", dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
        scope="mocked server control and gates; real temporary lock/package/hash IO; no server dispatch"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
