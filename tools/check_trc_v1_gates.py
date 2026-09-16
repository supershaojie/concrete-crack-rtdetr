"""Exercise TRC start gates with disposable fixtures, without model construction or training.

Run: python tools/check_trc_v1_gates.py --report outputs/trc_v1/gate_checks.json
The report path is exclusive: existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import c19_lif_v1_data
import trc_v1_common as common
from preflight_trc_v1 import strict_fp32


class PreflightGateTests(unittest.TestCase):
    """Mock expensive inventories, while exercising the real report acceptance gate."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="trc-v1-gates-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.initialized = self.root / "source.pt", self.root / "initialized.pt"
        # Deliberately tiny bytes, never loaded as model weights.
        self.source.write_bytes(b"fixed source fixture")
        self.initialized.write_bytes(b"zero-update initialization fixture")
        self.dataset = self.root / "dataset"
        self.data = self.root / "data.yaml"
        self.data.write_text("path: " + json.dumps(self.dataset.as_posix()) + "\n", encoding="utf-8")
        self.path = self.root / "preflight.json"
        self.variant = "cbr_lif_trc_v1"
        self.bound = dict(
            variant=self.variant,
            source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            initialized_sha256=hashlib.sha256(self.initialized.read_bytes()).hexdigest(),
            source_files={"module.py": "module-before", "model.yaml": "yaml-before"},
            data_config_sha256=hashlib.sha256(self.data.read_bytes()).hexdigest(),
        )
        self.inventory = {split: dict(images=2, boxes=3, split_paths_sha256=split + "-paths",
                                     label_inventory_sha256=split + "-labels")
                          for split in ("train", "val", "test")}
        self.report = dict(status="PASSED", server_capacity=dict(status="PASSED", batch=16,
                           imgsz=[640, 640], amp=True), fingerprint=deepcopy(self.bound),
                           formal_optimizer_steps=0, dataset=deepcopy(self.inventory))
        fingerprint_patch = patch.object(common, "fingerprint", return_value=deepcopy(self.bound))
        self.fingerprint_mock = fingerprint_patch.start()
        self.addCleanup(fingerprint_patch.stop)
        inventory_patch = patch.object(c19_lif_v1_data, "dataset_inventory", return_value=deepcopy(self.inventory))
        self.inventory_mock = inventory_patch.start()
        self.addCleanup(inventory_patch.stop)

    def verify(self):
        self.path.write_text(json.dumps(self.report), encoding="utf-8")
        return common.verify_preflight(self.path, self.variant, self.source, self.initialized, self.data)

    def reject(self, message):
        with self.assertRaisesRegex(RuntimeError, message):
            self.verify()

    def test_complete_bound_report_is_accepted(self):
        self.assertEqual(self.verify(), self.report)
        self.fingerprint_mock.assert_called_once_with(self.variant, self.source, self.initialized, self.data)
        self.inventory_mock.assert_called_once_with(self.dataset)

    def test_pending_preflight_cannot_authorize_start(self):
        self.report["status"] = "PENDING"
        self.reject("Preflight is not PASSED")
        self.fingerprint_mock.assert_not_called()

    def test_failed_preflight_cannot_authorize_start(self):
        self.report["status"] = "FAILED"
        self.reject("Preflight is not PASSED")
        self.fingerprint_mock.assert_not_called()

    def test_pending_failed_or_missing_server_capacity_cannot_authorize_start(self):
        for capacity in ({"status": "PENDING"}, {"status": "FAILED"}, {}):
            with self.subTest(capacity=capacity):
                self.report["server_capacity"] = capacity
                self.reject("Real B16/640/AMP server capacity is mandatory")
        self.fingerprint_mock.assert_not_called()

    def changed_fingerprint(self, key, value):
        current = deepcopy(self.bound)
        current[key] = value
        self.fingerprint_mock.return_value = current
        self.reject("Source/config/weights/init changed after preflight")
        self.inventory_mock.assert_not_called()

    def test_changed_fixed_source_weight_is_rejected(self):
        self.source.write_bytes(b"different source")
        self.changed_fingerprint("source_sha256", hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_changed_initialized_weight_is_rejected(self):
        self.initialized.write_bytes(b"updated initialization")
        self.changed_fingerprint("initialized_sha256", hashlib.sha256(self.initialized.read_bytes()).hexdigest())

    def test_changed_source_code_is_rejected(self):
        self.changed_fingerprint("source_files", {**self.bound["source_files"], "module.py": "module-after"})

    def test_changed_model_configuration_is_rejected(self):
        self.changed_fingerprint("source_files", {**self.bound["source_files"], "model.yaml": "yaml-after"})

    def test_changed_data_configuration_is_rejected(self):
        self.data.write_text(self.data.read_text(encoding="utf-8") + "nc: 2\n", encoding="utf-8")
        self.changed_fingerprint("data_config_sha256", hashlib.sha256(self.data.read_bytes()).hexdigest())

    def test_other_variant_report_is_rejected(self):
        self.changed_fingerprint("variant", "trc_v1")

    def test_updated_or_missing_formal_optimizer_count_is_rejected(self):
        for count in (1, None):
            with self.subTest(count=count):
                self.report["formal_optimizer_steps"] = count
                self.reject("Formal initialization was updated")

    def test_changed_dataset_split_or_labels_is_rejected(self):
        for field in ("split_paths_sha256", "label_inventory_sha256"):
            with self.subTest(field=field):
                changed = deepcopy(self.inventory)
                changed["train"][field] = "changed"
                self.inventory_mock.return_value = changed
                self.reject("Dataset split/labels changed after preflight")


class FingerprintAndPrecisionTests(unittest.TestCase):
    def test_actual_fingerprint_detects_fixture_edits_and_normalizes_text_line_endings(self):
        # Exercise the hashing implementation itself, independently of the gate's
        # mocked fingerprint. Restrict the inventory to three disposable files.
        with tempfile.TemporaryDirectory(prefix="trc-v1-fingerprint-") as directory:
            root = Path(directory)
            source, initialized = root / "source.pt", root / "initialized.pt"
            code, model_yaml, data = root / "module.py", root / "model.yaml", root / "data.yaml"
            source.write_bytes(b"source")
            initialized.write_bytes(b"initial")
            code.write_bytes(b"value = 1\n")
            model_yaml.write_bytes(b"nc: 1\n")
            data.write_bytes(b"path: /fixture\n")
            with patch.object(common, "ROOT", root), patch.object(common, "source_files", return_value=[code, model_yaml]):
                def fingerprint():
                    return common.fingerprint("cbr_lif_trc_v1", source, initialized, data)
                baseline = fingerprint()
                for path in (code, model_yaml, data):
                    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
                self.assertEqual(fingerprint(), baseline)
                for path, key in ((source, "source_sha256"), (initialized, "initialized_sha256"),
                                  (code, "source_files"), (model_yaml, "source_files"), (data, "data_config_sha256")):
                    with self.subTest(path=path.name):
                        before = path.read_bytes()
                        path.write_bytes(before + b"changed")
                        self.assertNotEqual(fingerprint()[key], baseline[key])
                        path.write_bytes(before)
                self.assertEqual(fingerprint(), baseline)

    def test_strict_fp32_restores_tf32_after_success_and_exception(self):
        # Backend flags are process-local; this allocates no CUDA tensors or model.
        original = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            for previous in ((False, True), (True, False)):
                for raises in (False, True):
                    with self.subTest(previous=previous, raises=raises):
                        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
                        try:
                            with strict_fp32() as audit:
                                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                                self.assertFalse(torch.backends.cudnn.allow_tf32)
                                self.assertEqual(audit["restore"], list(previous))
                                if raises:
                                    raise RuntimeError("diagnostic fixture exception")
                        except RuntimeError as error:
                            self.assertTrue(raises)
                            self.assertEqual(str(error), "diagnostic fixture exception")
                        self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), previous)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id().split(".", 1)[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=common.ROOT / "docs/trc_v1/gate_checks.json")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Existing gate report protected; select a new --report path")
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (PreflightGateTests, FingerprintAndPrecisionTests))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    report = dict(status="PASSED" if result.wasSuccessful() and not result.skipped else "FAILED",
                  created_utc=datetime.now(timezone.utc).isoformat(), checks=result.testsRun,
                  passed=result.passed,
                  failures=[dict(test=test.id(), traceback=trace) for test, trace in result.failures + result.errors],
                  skipped=[dict(test=test.id(), reason=reason) for test, reason in result.skipped],
                  scope="Temporary fixtures, mocked gate inventories plus real isolated hashes; no models or CUDA tensors",
                  python=sys.version.split()[0], torch=str(torch.__version__), formal_optimizer_steps=0,
                  formal_training="NOT_STARTED", test="NOT_RUN")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], checks=result.testsRun, report=str(args.report))))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
