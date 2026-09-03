"""CPU tests for the standalone CSCEF-v2 validation diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import diagnose_cscef_v2 as diagnostic  # noqa: E402
from ultralytics import RTDETR  # noqa: E402
from ultralytics.models.rtdetr.val import RTDETRValidator  # noqa: E402
from ultralytics.models.yolo.detect import DetectionValidator  # noqa: E402
from ultralytics.nn.modules import CSCEFv2  # noqa: E402


class DiagnosticHelpersTest(unittest.TestCase):
    """Exercise diagnostic primitives without a dataset, GPU, or real checkpoint."""

    def setUp(self):
        torch.manual_seed(7)

    def test_find_cscef_v2_rejects_missing_module(self):
        with self.assertRaisesRegex(RuntimeError, "no CSCEFv2"):
            diagnostic.find_unique_cscef_v2(nn.Sequential(nn.Conv2d(3, 4, 1)))

    def test_find_cscef_v2_rejects_multiple_modules(self):
        FakeCSCEFv2 = type("CSCEFv2", (nn.Module,), {"forward": lambda self, value: value})
        with self.assertRaisesRegex(RuntimeError, "2 CSCEFv2"):
            diagnostic.find_unique_cscef_v2(nn.Sequential(FakeCSCEFv2(), FakeCSCEFv2()))

    def test_find_cscef_v2_accepts_one_module_without_layer_index(self):
        module = CSCEFv2(8, 8, hidden_channels=4)
        self.assertIs(diagnostic.find_unique_cscef_v2(nn.Sequential(nn.Identity(), module)), module)

    def test_constant_half_gate_has_dynamic_shape_dtype_and_device(self):
        module = CSCEFv2(8, 8, hidden_channels=4)
        lateral = torch.randn(3, 4, 5, 7, dtype=torch.float64)
        semantic = torch.randn_like(lateral)
        gate = diagnostic.constant_gate_05(module, lateral, semantic)
        self.assertEqual(gate.shape, (3, 1, 5, 7))
        self.assertEqual(gate.dtype, torch.float32)
        self.assertEqual(gate.device, lateral.device)
        torch.testing.assert_close(gate, torch.full_like(gate, 0.5))

    def test_constant_one_gate_is_exact(self):
        module = CSCEFv2(8, 8, hidden_channels=4)
        lateral = torch.randn(2, 4, 3, 6)
        gate = diagnostic.constant_gate_1(module, lateral, torch.randn_like(lateral))
        torch.testing.assert_close(gate, torch.ones_like(gate), rtol=0, atol=0)

    def test_layer_scale_zero_makes_effective_delta_zero_and_restores(self):
        module = CSCEFv2(8, 8, hidden_channels=4).eval()
        lateral = torch.randn(2, 8, 5, 6)
        semantic = torch.randn(2, 8, 3, 4)
        original_raw = module.layer_scale_raw.detach().clone()
        with diagnostic.temporary_intervention(module, "layer_scale_zero"):
            self.assertEqual(module._effective_layer_scale().abs().max().item(), 0.0)
            output = module([lateral, semantic])
            torch.testing.assert_close(output, lateral, rtol=0, atol=0)
        torch.testing.assert_close(module.layer_scale_raw, original_raw, rtol=0, atol=0)

    def test_shared_norm_bias_intervention_changes_only_target_then_restores(self):
        module = CSCEFv2(8, 8, hidden_channels=4)
        with torch.no_grad():
            module.shared_norm.bias.copy_(torch.arange(4, dtype=torch.float32) + 1)
        before = {name: value.detach().clone() for name, value in module.state_dict().items()}
        with diagnostic.temporary_intervention(module, "shared_norm_bias_zero"):
            for name, value in module.state_dict().items():
                if name == "shared_norm.bias":
                    self.assertEqual(value.abs().sum().item(), 0.0)
                else:
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_edge_and_both_norm_bias_targeting(self):
        module = CSCEFv2(8, 8, hidden_channels=4)
        edge_norm = diagnostic.find_unique_edge_group_norm(module)
        with torch.no_grad():
            module.shared_norm.bias.fill_(2)
            edge_norm.bias.fill_(3)
        with diagnostic.temporary_intervention(module, "edge_norm_bias_zero"):
            self.assertEqual(edge_norm.bias.abs().sum().item(), 0.0)
            self.assertTrue((module.shared_norm.bias == 2).all().item())
        with diagnostic.temporary_intervention(module, "both_norm_bias_zero"):
            self.assertEqual(edge_norm.bias.abs().sum().item(), 0.0)
            self.assertEqual(module.shared_norm.bias.abs().sum().item(), 0.0)
        self.assertTrue((edge_norm.bias == 3).all().item())
        self.assertTrue((module.shared_norm.bias == 2).all().item())

    def test_gate_method_intervention_is_instance_local_and_restored(self):
        first = CSCEFv2(8, 8, hidden_channels=4)
        second = CSCEFv2(8, 8, hidden_channels=4)
        embeddings = (torch.randn(2, 4, 3, 5), torch.randn(2, 4, 3, 5))
        expected_second = second._semantic_gate(*embeddings)
        with diagnostic.temporary_intervention(first, "gate_const_05"):
            torch.testing.assert_close(first._semantic_gate(*embeddings), torch.full((2, 1, 3, 5), 0.5))
            torch.testing.assert_close(second._semantic_gate(*embeddings), expected_second)
        self.assertNotIn("_semantic_gate", first.__dict__)

    def test_raw_cosine_gate_matches_requested_formula(self):
        module = CSCEFv2(8, 8, hidden_channels=4)
        lateral = torch.randn(2, 4, 5, 6)
        semantic = torch.randn_like(lateral)
        consistency = F.cosine_similarity(lateral.float(), semantic.float(), dim=1, eps=module.eps).unsqueeze(1)
        expected = torch.sigmoid(module._effective_temperature() * consistency)
        actual = diagnostic.raw_cosine_gate(module, lateral, semantic)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_online_stats_handles_normal_constant_and_tiny_variance(self):
        normal = diagnostic.OnlineTensorStats()
        normal.update(torch.tensor([1.0, 2.0, 3.0]))
        summary = normal.summary()
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["std"], (2 / 3) ** 0.5)
        constant = diagnostic.OnlineTensorStats()
        constant.update(torch.full((10,), 4.0))
        self.assertEqual(constant.summary()["std"], 0.0)
        tiny = diagnostic.OnlineTensorStats()
        tiny.update(torch.tensor([1.0, 1.0 + 1e-7, 1.0 - 1e-7]))
        self.assertTrue(torch.isfinite(torch.tensor(tiny.summary()["std"])).item())

    def test_online_stats_counts_nan_and_infinities(self):
        stats = diagnostic.OnlineTensorStats()
        stats.update(torch.tensor([1.0, float("nan"), float("inf"), float("-inf")]))
        summary = stats.summary()
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["finite_count"], 1)
        self.assertEqual(summary["nan_count"], 1)
        self.assertEqual(summary["positive_infinity_count"], 1)
        self.assertEqual(summary["negative_infinity_count"], 1)

    def test_large_activation_sampling_indices_never_round_out_of_bounds(self):
        length = 16 * 256 * 80 * 80
        indices = diagnostic.evenly_spaced_integer_indices(length, 4096)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[-1].item(), length - 1)
        self.assertGreaterEqual(indices.min().item(), 0)
        self.assertLess(indices.max().item(), length)

    def test_gt_box_mapping_uses_expected_gate_cells(self):
        masks, counts, invalid = diagnostic.build_gt_masks(
            torch.tensor([[0.5, 0.5, 0.5, 0.5]]), torch.tensor([0]), 1, 4, 4
        )
        expected = torch.zeros(1, 4, 4, dtype=torch.bool)
        expected[0, 1:3, 1:3] = True
        torch.testing.assert_close(masks, expected)
        self.assertEqual(counts, [1])
        self.assertEqual(invalid, 0)

    def test_multiple_gt_boxes_are_unioned_and_clipped(self):
        boxes = torch.tensor([[0.125, 0.125, 0.25, 0.25], [0.875, 0.875, 0.50, 0.50]])
        masks, counts, invalid = diagnostic.build_gt_masks(boxes, torch.tensor([0, 0]), 1, 4, 4)
        self.assertTrue(masks[0, 0, 0].item())
        self.assertTrue(masks[0, 3, 3].item())
        self.assertEqual(counts, [2])
        self.assertEqual(invalid, 0)

    def test_empty_gt_batch_returns_empty_masks(self):
        masks, counts, invalid = diagnostic.build_gt_masks(torch.empty(0, 4), torch.empty(0), 2, 3, 5)
        self.assertEqual(masks.shape, (2, 3, 5))
        self.assertFalse(masks.any().item())
        self.assertEqual(counts, [0, 0])
        self.assertEqual(invalid, 0)

    def test_batch_index_equal_to_batch_size_is_rejected_on_cpu(self):
        with self.assertRaisesRegex(diagnostic.InvalidBatchIndexError, "outside"):
            diagnostic.build_gt_masks(torch.ones(1, 4), torch.tensor([2], device="cpu"), 2, 4, 4)

    def test_negative_batch_index_is_rejected_on_cpu(self):
        with self.assertRaisesRegex(diagnostic.InvalidBatchIndexError, "batch_idx.*-1"):
            diagnostic.build_gt_masks(torch.ones(1, 4), torch.tensor([-1], device="cpu"), 2, 4, 4)

    def test_out_of_range_batch_index_error_records_paths_and_index(self):
        with self.assertRaises(diagnostic.InvalidBatchIndexError) as caught:
            diagnostic.build_gt_masks(
                torch.ones(1, 4), torch.tensor([3]), 2, 4, 4, ["first.png", "second.png"]
            )
        message = str(caught.exception)
        self.assertIn("'batch_idx': 3", message)
        self.assertIn("first.png", message)
        self.assertIn("second.png", message)

    def test_gt_coordinates_are_clipped_to_gate_extent(self):
        boxes = torch.tensor([[-0.1, -0.1, 1.0, 1.0], [1.1, 1.1, 1.0, 1.0]])
        masks, counts, invalid = diagnostic.build_gt_masks(boxes, torch.tensor([0, 0]), 1, 4, 4)
        self.assertTrue(masks[0, 0, 0].item())
        self.assertTrue(masks[0, 3, 3].item())
        self.assertEqual(counts, [2])
        self.assertEqual(invalid, 0)

    def test_warmup_forward_is_ignored_by_collector(self):
        module = CSCEFv2(8, 8, hidden_channels=4).eval()
        collector = diagnostic.ActivationCollector(module)
        try:
            with torch.no_grad():
                module([torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)])
            self.assertEqual(collector.stats["gate"].count, 0)
            self.assertIsNone(collector._pending)
            self.assertIsNone(collector.runtime_context["current_batch_index"])
        finally:
            collector.close()

    def test_forward_hook_returns_none_and_does_not_replace_output(self):
        module = CSCEFv2(8, 8, hidden_channels=4).eval()
        collector = diagnostic.ActivationCollector(module)
        lateral = torch.randn(1, 8, 4, 4)
        semantic = torch.randn(1, 8, 2, 2)
        try:
            with torch.no_grad():
                output = module([lateral, semantic])
            expected = output.clone()
            collector.begin_batch(
                0,
                {"img": torch.zeros(1, 3, 32, 32), "batch_idx": torch.empty(0), "im_file": ["image.png"]},
            )
            returned = collector._forward_hook(module, ([lateral, semantic],), output)
            self.assertIsNone(returned)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        finally:
            collector.close()

    def test_gate_cannot_be_joined_to_the_next_batch(self):
        module = CSCEFv2(8, 8, hidden_channels=4).eval()
        collector = diagnostic.ActivationCollector(module)
        try:
            collector.begin_batch(0)
            with torch.no_grad():
                module([torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)])
            with self.assertRaisesRegex(RuntimeError, "batch mismatch"):
                collector.finish_batch(
                    1,
                    {"bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0), "im_file": ["next.png"]},
                )
        finally:
            collector.close()

    def test_collector_aligns_one_gate_with_same_batch_labels(self):
        module = CSCEFv2(8, 8, hidden_channels=4).eval()
        collector = diagnostic.ActivationCollector(module)
        try:
            collector.begin_batch(0)
            lateral = torch.randn(2, 8, 4, 4)
            semantic = torch.randn(2, 8, 2, 2)
            with torch.no_grad():
                module([lateral, semantic])
            collector.finish_batch(
                0,
                {
                    "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
                    "batch_idx": torch.tensor([0]),
                    "im_file": ["positive.png", "empty.png"],
                },
            )
            result = collector.as_dict()
            self.assertEqual(result["collection"]["completed_stat_batches"], 1)
            self.assertEqual(result["gate_inside_vs_outside_gt"]["valid_images"], 1)
            self.assertEqual(result["gate_inside_vs_outside_gt"]["skipped_no_gt"], 1)
            self.assertEqual([row["image_path"] for row in collector.per_image_rows], ["positive.png", "empty.png"])
        finally:
            collector.close()

    def test_official_rtdetr_task_map_does_not_select_generic_validator(self):
        selected = RTDETR.task_map.fget(None)["detect"]["validator"]
        self.assertIs(selected, RTDETRValidator)
        self.assertIsNot(selected, DetectionValidator)

    def test_diagnostic_validator_must_be_rtdetr_validator_subclass(self):
        with self.assertRaisesRegex(TypeError, "RTDETRValidator subclass"):
            diagnostic.require_rtdetr_validator(DetectionValidator, RTDETRValidator)
        validator = diagnostic.make_diagnostic_validator(RTDETRValidator, None, RTDETRValidator)
        diagnostic.require_rtdetr_validator(validator, RTDETRValidator)
        self.assertTrue(issubclass(validator, RTDETRValidator))

    def test_cuda_device_assert_is_marked_unrecoverable(self):
        self.assertTrue(diagnostic.is_unrecoverable_cuda_error("CUDA error: device-side assert triggered"))
        self.assertTrue(diagnostic.is_unrecoverable_cuda_error("IndexKernel.cu: index out of bounds"))
        self.assertFalse(diagnostic.is_unrecoverable_cuda_error("ordinary validation error"))

    def test_metric_extraction_accepts_rtdetr_validator_results_object(self):
        metrics = SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.1,
                "metrics/recall(B)": 0.2,
                "metrics/mAP50(B)": 0.3,
                "metrics/mAP50-95(B)": 0.4,
                "fitness": 0.5,
            }
        )
        self.assertEqual(
            diagnostic.extract_metrics(metrics),
            {"precision": 0.1, "recall": 0.2, "mAP50": 0.3, "mAP50-95": 0.4, "fitness": 0.5},
        )

    def test_original_failure_still_writes_terminal_artifacts(self):
        class FakeUltralytics:
            __version__ = "8.4.21"
            __file__ = str(ROOT / "ultralytics-main" / "ultralytics" / "__init__.py")

        class FakeWrapper:
            def __init__(self, _weights):
                self.model = nn.Sequential(CSCEFv2(8, 8, hidden_channels=4))

        class FakeRTDETRValidator:
            pass

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            weights = directory / "best.pt"
            data = directory / "data.yaml"
            weights.write_bytes(b"read-only checkpoint")
            data.write_text("names: [crack]\n", encoding="utf-8")
            args = argparse.Namespace(
                weights=weights,
                data=data,
                split="val",
                imgsz=640,
                batch=16,
                workers=8,
                device="cpu",
                project=directory / "diagnostics",
                name="failed_original",
                seed=42,
                stats_max_batches=0,
                variants="original",
                plots=False,
                overwrite=False,
            )
            failure = diagnostic.failed_metric_row(
                "original", diagnostic.sha256_file(weights), diagnostic.sha256_file(data), "synthetic traceback"
            )
            audit = {
                "variant": "original",
                "wrapper_class": "RTDETR",
                "model_class": "RTDETRDetectionModel",
                "validator_class": "CSCEFDiagnosticValidator",
                "validator_module": "ultralytics.models.rtdetr.val",
                "task": "detect",
                "model_nc": 1,
                "dataset_nc": 1,
                "cscef_v2_instance_count": 1,
            }
            with mock.patch.object(
                diagnostic,
                "import_local_ultralytics",
                return_value=(FakeUltralytics, FakeWrapper, FakeRTDETRValidator),
            ), mock.patch.object(diagnostic, "dataset_class_count", return_value=1), mock.patch.object(
                diagnostic, "run_variant", return_value=(failure, None, audit, False)
            ):
                self.assertEqual(diagnostic.run(args), 1)
            output = args.project / args.name
            for filename in (
                "run_manifest.json",
                "variant_metrics.json",
                "variant_metrics.csv",
                "diagnostic_report.md",
                "diagnostic.log",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            saved = json.loads((output / "variant_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["status"], "failed")
            self.assertEqual(saved[0]["error"], "synthetic traceback")

    def test_metric_csv_and_json_have_complete_fields(self):
        row = {field: None for field in diagnostic.METRIC_FIELDS}
        row.update({"variant": "original", "status": "failed", "error": "traceback"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            diagnostic.write_metric_artifacts(output, [row])
            with (output / "variant_metrics.csv").open(newline="", encoding="utf-8") as stream:
                csv_row = next(csv.DictReader(stream))
            json_row = json.loads((output / "variant_metrics.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(tuple(csv_row), diagnostic.METRIC_FIELDS)
            self.assertEqual(tuple(json_row), diagnostic.METRIC_FIELDS)
            self.assertEqual(json_row["status"], "failed")

    def test_split_test_is_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            weights = directory / "best.pt"
            data = directory / "data.yaml"
            weights.write_bytes(b"checkpoint")
            data.write_text("path: .\n", encoding="utf-8")
            args = argparse.Namespace(
                weights=weights,
                data=data,
                split="test",
                imgsz=640,
                batch=16,
                workers=8,
                device="cpu",
                project=directory,
                name="diagnosis",
                seed=42,
                stats_max_batches=0,
                variants="all",
                plots=False,
                overwrite=False,
            )
            with self.assertRaisesRegex(ValueError, "Only --split val"):
                diagnostic.validate_args(args)

    def test_checkpoint_file_bytes_are_unchanged_by_interventions(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"immutable checkpoint bytes")
            before = diagnostic.sha256_file(checkpoint)
            module = CSCEFv2(8, 8, hidden_channels=4)
            with diagnostic.temporary_intervention(module, "both_norm_bias_zero"):
                pass
            after = diagnostic.sha256_file(checkpoint)
            self.assertEqual(before, after)
            self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_variant_parser_supports_all_and_lists(self):
        self.assertEqual(diagnostic.parse_variants("all"), list(diagnostic.VARIANTS))
        self.assertEqual(diagnostic.parse_variants("original,gate_const_1"), ["original", "gate_const_1"])
        with self.assertRaises(ValueError):
            diagnostic.parse_variants("original,unknown")
        with self.assertRaises(ValueError):
            diagnostic.parse_variants("original,original")

    def test_output_directory_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            output = diagnostic.prepare_output_directory(project, "run", False)
            marker = output / "marker.txt"
            marker.write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                diagnostic.prepare_output_directory(project, "run", False)
            replaced = diagnostic.prepare_output_directory(project, "run", True)
            self.assertFalse((replaced / "marker.txt").exists())

    def test_parameter_audit_contains_requested_sections(self):
        audit = diagnostic.parameter_audit(CSCEFv2(8, 8, hidden_channels=4))
        self.assertEqual(
            set(audit),
            {
                "temperature",
                "effective_layer_scale",
                "shared_group_norm",
                "edge_calibration_group_norm",
                "projections",
                "cscef_v2_total_parameters",
            },
        )
        self.assertIn("effective_temperature", audit["temperature"])
        self.assertIn("negative_fraction", audit["effective_layer_scale"])
        self.assertIn("l2_norm", audit["shared_group_norm"]["bias"])


if __name__ == "__main__":
    unittest.main()
