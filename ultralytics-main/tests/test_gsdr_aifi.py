# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the Global-Sparse Deformable Relation AIFI."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import math
from pathlib import Path
import py_compile
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import AIFI, GSDRAIFI, SparseDeformableRelation
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ultralytics-main" / "ultralytics" / "cfg" / "models" / "rt-detr"
MODULE_DIR = ROOT / "ultralytics-main" / "ultralytics" / "nn" / "modules"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
GSDR_CFG = MODEL_DIR / "rtdetr-resnet18-lite-gsdr-aifi.yaml"
PROTECTED_SHA256 = {
    BASE_CFG: "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    MODULE_DIR / "transformer.py": "8986f3f58a5e30a42545f544f1a35f5e5861b13ee5d4507d8b2afab1c41df782",
    ROOT / "tools" / "init_rtdetr_r18_lite_imagenet_backbone.py": (
        "1ee77782f64b7e310a14c436a15807ffdca57725a4f1cd9b5a684e9d054df87a"
    ),
}
EXPECTED_SPARSE_PARAMETERS = 117_644

sys.path.insert(0, str(ROOT / "tools"))
from init_rtdetr_r18_lite_gsdr_aifi_controlled import (  # noqa: E402
    build_strict_mapping,
    original_aifi_keys,
    verify_zero_initialized_boundaries,
)


def canonical_lf_sha256(path: Path) -> str:
    """Return a line-ending-independent SHA256 digest."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def finite_nonzero_gradient(parameter: torch.Tensor) -> bool:
    """Return whether a parameter gradient exists, is finite, and is nonzero."""
    return (
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        and parameter.grad.abs().sum().item() > 0
    )


class GSDRAIFITest(unittest.TestCase):
    """Exercise module math, parser integration, checkpoints, and complete RT-DETR models."""

    def test_py_compile_and_public_import(self):
        """Compile every GSDR entry point and verify public imports."""
        files = (
            MODULE_DIR / "gsdr_aifi.py",
            ROOT / "tools" / "init_rtdetr_r18_lite_gsdr_aifi_controlled.py",
            ROOT / "tools" / "audit_rtdetr_r18_lite_gsdr_aifi.py",
            Path(__file__),
        )
        (ROOT / "weights").mkdir(exist_ok=True)
        with TemporaryDirectory(dir=ROOT / "weights") as directory:
            for index, path in enumerate(files):
                with self.subTest(path=path):
                    py_compile.compile(str(path), cfile=str(Path(directory) / f"compiled_{index}.pyc"), doraise=True)
        from ultralytics.nn.modules import GSDRAIFI as PublicGSDRAIFI
        from ultralytics.nn.modules import SparseDeformableRelation as PublicSparseDeformableRelation

        self.assertIs(PublicGSDRAIFI, GSDRAIFI)
        self.assertIs(PublicSparseDeformableRelation, SparseDeformableRelation)

    def test_constructor_validation(self):
        """Reject invalid channels, heads, groups, stride, kernel, range, and dropout."""
        invalid_relation = (
            ((0,), {}),
            ((16,), {"aux_dim": 0}),
            ((16,), {"aux_dim": 30, "aux_heads": 4}),
            ((16,), {"aux_dim": 32, "offset_groups": 3}),
            ((16,), {"aux_dim": 32, "aux_heads": 2, "offset_groups": 4}),
            ((16,), {"sample_stride": 0}),
            ((16,), {"offset_kernel": 2}),
            ((16,), {"offset_range_factor": -1}),
            ((16,), {"offset_range_factor": float("inf")}),
            ((16,), {"dropout": 1}),
        )
        for args, kwargs in invalid_relation:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                SparseDeformableRelation(*args, **kwargs)
        for args in ((0,), (30, 32, 8), (32, 32, 0)):
            with self.subTest(gsdr_args=args), self.assertRaises(ValueError):
                GSDRAIFI(*args)

    def test_forward_input_validation(self):
        """Reject non-image tensors and incorrect channel counts."""
        relation = SparseDeformableRelation(16, 16, 4, 4)
        module = GSDRAIFI(16, 32, 4, aux_dim=16, aux_heads=4, offset_groups=4)
        invalid = (torch.randn(2, 16, 5), torch.randn(2, 8, 5, 7))
        for tensor in invalid:
            with self.subTest(shape=tuple(tensor.shape)), self.assertRaises(ValueError):
                relation(tensor)
            with self.subTest(gsdr_shape=tuple(tensor.shape)), self.assertRaises(ValueError):
                module(tensor)

    def test_dynamic_shapes_offsets_reference_grid_and_zero_output(self):
        """Check square, non-square, and singleton-axis sampling without non-finite values."""
        torch.manual_seed(30)
        relation = SparseDeformableRelation(256, 128, 4, 4, 2, 2.0, 3).eval()
        for height, width in ((20, 20), (16, 24), (1, 1), (1, 5), (5, 1)):
            with self.subTest(shape=(height, width)), torch.no_grad():
                tensor = torch.randn(1, 256, height, width)
                projected = relation.input_proj(tensor)
                reference, offsets, deformed = relation._sampling_grid(projected)
                output = relation(tensor)
                sample_h, sample_w = math.ceil(height / 2), math.ceil(width / 2)
                self.assertEqual(reference.shape, (1, sample_h, sample_w, 2))
                self.assertEqual(offsets.shape, (1, 4, sample_h, sample_w, 2))
                self.assertEqual(deformed.shape, offsets.shape)
                self.assertEqual(output.shape, tensor.shape)
                tensors = (reference, offsets, deformed, output)
                self.assertTrue(all(torch.isfinite(item).all().item() for item in tensors))
                self.assertGreaterEqual(reference.min().item(), -1.0)
                self.assertLessEqual(reference.max().item(), 1.0)
                torch.testing.assert_close(offsets, torch.zeros_like(offsets), rtol=0, atol=0)
                torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    def test_offset_bounds_and_xy_coordinate_order(self):
        """Drive dx/dy logits to saturation and verify runtime bounds and coordinate ordering."""
        relation = SparseDeformableRelation(16, 16, 4, 4, 2, 2.0, 3).eval()
        tensor = torch.randn(1, 16, 8, 12)
        with torch.no_grad():
            relation.offset_out.bias.zero_()
            relation.offset_out.bias[0] = 100.0  # group-0 x
            relation.offset_out.bias[1] = -100.0  # group-0 y
            reference, offsets, deformed = relation._sampling_grid(relation.input_proj(tensor))
        self.assertTrue((offsets[:, 0, ..., 0] > 0).all().item())
        self.assertTrue((offsets[:, 0, ..., 1] < 0).all().item())
        self.assertLessEqual(offsets[..., 0].abs().max().item(), 4.0 / 6.0 + 1e-6)
        self.assertLessEqual(offsets[..., 1].abs().max().item(), 4.0 / 4.0 + 1e-6)
        self.assertGreaterEqual(deformed.min().item(), -1.0)
        self.assertLessEqual(deformed.max().item(), 1.0)
        torch.testing.assert_close(reference[0, 0, 0], reference.new_tensor([-5.0 / 6.0, -0.75]))

    def test_initialization_boundaries_and_parameter_budget(self):
        """Require zero offset/bias/output boundaries and the controlled sparse parameter count."""
        module = GSDRAIFI(256, 1024, 8)
        verify_zero_initialized_boundaries(module)
        relation = module.sparse_relation
        self.assertEqual(sum(parameter.numel() for parameter in relation.parameters()), EXPECTED_SPARSE_PARAMETERS)
        self.assertEqual(relation.output_proj.weight.count_nonzero().item(), 0)
        self.assertEqual(relation.offset_out.weight.count_nonzero().item(), 0)
        for mlp in relation.relative_bias:
            self.assertEqual(mlp.fc2.weight.count_nonzero().item(), 0)
            self.assertEqual(mlp.fc2.bias.count_nonzero().item(), 0)

    def test_continuous_bias_uses_actual_deformed_positions(self):
        """Require dynamic bias shape and sensitivity to the actual post-offset sample coordinates."""
        relation = SparseDeformableRelation(16, 16, 4, 4).eval()
        query_grid = relation._reference_grid(4, 6, torch.device("cpu"), torch.float32)
        sample_grid = relation._reference_grid(2, 3, torch.device("cpu"), torch.float32).repeat(1, 4, 1, 1, 1)
        with torch.no_grad():
            for mlp in relation.relative_bias:
                mlp.fc1.weight.fill_(0.25)
                mlp.fc1.bias.fill_(0.1)
                mlp.fc2.weight.fill_(0.125)
                mlp.fc2.bias.zero_()
        original = relation._relative_displacement_bias(query_grid, sample_grid, torch.float32)
        shifted_grid = sample_grid.clone()
        shifted_grid[:, 0, ..., 0].add_(0.2).clamp_(-1, 1)
        shifted = relation._relative_displacement_bias(query_grid, shifted_grid, torch.float32)
        self.assertEqual(original.shape, (1, 4, 24, 6))
        self.assertTrue(torch.isfinite(original).all().item())
        self.assertFalse(torch.equal(original[:, 0], shifted[:, 0]))
        torch.testing.assert_close(original[:, 1:], shifted[:, 1:], rtol=0, atol=0)

    def test_baseline_equivalence_post_and_pre_norm(self):
        """Load original AIFI tensors and require elementwise-exact initialized output equality."""
        torch.manual_seed(31)
        tensor = torch.randn(2, 256, 16, 24)
        for normalize_before in (False, True):
            with self.subTest(normalize_before=normalize_before):
                baseline = AIFI(256, 1024, 8, dropout=0, normalize_before=normalize_before).eval()
                gsdr = GSDRAIFI(256, 1024, 8, dropout=0, normalize_before=normalize_before).eval()
                result = gsdr.load_state_dict(baseline.state_dict(), strict=False)
                self.assertFalse(result.unexpected_keys)
                self.assertTrue(result.missing_keys)
                self.assertTrue(all(key.startswith("sparse_relation.") for key in result.missing_keys))
                with torch.no_grad():
                    expected = baseline(tensor)
                    actual = gsdr(tensor)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual((actual - expected).abs().max().item(), 0.0)

    def test_staged_gradient_unblocking(self):
        """Prove first-step boundary learning and eventual finite gradients for every sparse parameter."""
        torch.manual_seed(32)
        relation = SparseDeformableRelation(32, 32, 4, 4)
        tensor = torch.randn(2, 32, 8, 10, requires_grad=True)
        (relation(tensor) * torch.randn_like(tensor)).mean().backward()
        self.assertTrue(finite_nonzero_gradient(relation.output_proj.weight))
        self.assertTrue(finite_nonzero_gradient(relation.output_proj.bias))

        relation.zero_grad(set_to_none=True)
        tensor.grad = None
        with torch.no_grad():
            torch.nn.init.normal_(relation.output_proj.weight, std=0.02)
        (relation(tensor) * torch.randn_like(tensor)).mean().backward()
        for parameter in (
            relation.q_proj.weight,
            relation.k_proj.weight,
            relation.v_proj.weight,
            relation.offset_out.weight,
            relation.relative_bias[0].fc2.weight,
        ):
            self.assertTrue(finite_nonzero_gradient(parameter))

        relation.zero_grad(set_to_none=True)
        tensor.grad = None
        with torch.no_grad():
            torch.nn.init.normal_(relation.offset_out.weight, std=0.02)
            for mlp in relation.relative_bias:
                torch.nn.init.normal_(mlp.fc2.weight, std=0.02)
        (relation(tensor) * torch.randn_like(tensor)).mean().backward()
        for name, parameter in relation.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(finite_nonzero_gradient(parameter))
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all().item())
        self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

    def test_yaml_differs_from_baseline_only_at_layer9(self):
        """Keep every baseline layer, index, decoder input, class count, and scale unchanged."""
        baseline = YAML.load(BASE_CFG)
        gsdr = YAML.load(GSDR_CFG)
        expected = deepcopy(baseline["backbone"] + baseline["head"])
        expected[9] = [-1, 1, "GSDRAIFI", [1024, 8, 128, 4, 4, 2, 2.0, 3]]
        self.assertEqual(gsdr["backbone"] + gsdr["head"], expected)
        self.assertEqual(gsdr["nc"], baseline["nc"])
        self.assertEqual(gsdr["scales"], baseline["scales"])
        self.assertEqual(expected[-1][0], [19, 22, 25])

    def test_parser_nc1_nc80_and_baseline_regression(self):
        """Build both class counts through the parser and retain the original baseline model."""
        baseline = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=1, verbose=False).eval()
        self.assertIs(type(baseline.model[9]), AIFI)
        self.assertFalse(any(isinstance(item, GSDRAIFI) for item in baseline.modules()))
        for nc in (1, 80):
            with self.subTest(nc=nc):
                model = RTDETRDetectionModel(str(GSDR_CFG), ch=3, nc=nc, verbose=False).eval()
                self.assertIsInstance(model.model[9], GSDRAIFI)
                self.assertEqual(model.model[9].c1, 256)
                with torch.no_grad():
                    output = model(torch.zeros(1, 3, 128, 128))
                self.assertEqual(output[0].shape, (1, 300, nc + 4))
                self.assertTrue(torch.isfinite(output[0]).all().item())

    def test_strict_mapping_contract_and_original_aifi_coverage(self):
        """Require identity mapping for every baseline key and sparse-only target additions."""
        baseline = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
        gsdr = RTDETRDetectionModel(str(GSDR_CFG), ch=3, nc=80, verbose=False)
        baseline_state = baseline.state_dict()
        plan = build_strict_mapping(baseline_state, baseline_state, gsdr.state_dict())
        self.assertEqual(len(plan.state), len(baseline_state))
        self.assertFalse(plan.shape_mismatches)
        self.assertTrue(plan.new_target_keys)
        self.assertTrue(all(key.startswith("model.9.sparse_relation.") for key in plan.new_target_keys))
        aifi_keys = original_aifi_keys(baseline_state)
        self.assertEqual(len(aifi_keys), 12)
        self.assertTrue(all(key in plan.source_to_target for key in aifi_keys))
        self.assertTrue(all(plan.source_to_target[key] == key for key in aifi_keys))

    def test_protected_baseline_files_unchanged(self):
        """Keep the baseline YAML, AIFI source, and baseline initialization script byte-content unchanged."""
        for path, expected in PROTECTED_SHA256.items():
            with self.subTest(path=path):
                self.assertEqual(canonical_lf_sha256(path), expected)

    def test_direct_checkpoint_and_yaml_load_compatibility(self):
        """Reload a direct GSDR checkpoint and load it into a fresh YAML model."""
        model = RTDETRDetectionModel(str(GSDR_CFG), ch=3, nc=1, verbose=False).eval()
        (ROOT / "weights").mkdir(exist_ok=True)
        with TemporaryDirectory(dir=ROOT / "weights") as directory:
            checkpoint_path = Path(directory) / "gsdr_aifi_test.pt"
            model.args = {**DEFAULT_CFG_DICT, "model": str(GSDR_CFG), "task": "detect"}
            model.task = "detect"
            torch.save({"epoch": -1, "model": deepcopy(model).half(), "train_args": model.args}, checkpoint_path)
            direct = RTDETR(str(checkpoint_path))
            self.assertIsInstance(direct.model.model[9], GSDRAIFI)
            yaml_loaded = RTDETR(str(GSDR_CFG))
            yaml_loaded.load(str(checkpoint_path))
            self.assertIsInstance(yaml_loaded.model.model[9], GSDRAIFI)

    def test_complete_cpu_640_forward_and_128_backward(self):
        """Run full RT-DETR CPU FP32 at 640 and propagate a finite 128 backward."""
        model = RTDETRDetectionModel(str(GSDR_CFG), ch=3, nc=1, verbose=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 640, 640))
        self.assertEqual(output[0].shape, (1, 300, 5))
        self.assertTrue(torch.isfinite(output[0]).all().item())
        del output, model
        gc.collect()

        model = RTDETRDetectionModel(str(GSDR_CFG), ch=3, nc=1, verbose=False).eval()
        image = torch.randn(1, 3, 128, 128, requires_grad=True)
        output = model(image)
        (output[0].float() * torch.randn_like(output[0].float())).mean().backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all().item())
        self.assertGreater(image.grad.abs().sum().item(), 0.0)
        self.assertTrue(finite_nonzero_gradient(model.model[9].sparse_relation.output_proj.weight))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_module_fp32_amp_and_explicit_half(self):
        """Run real CUDA FP32/AMP and model.cuda().half() module forward/backward checks."""
        for mode in ("fp32", "amp", "half"):
            with self.subTest(mode=mode):
                module = GSDRAIFI(256, 1024, 8).cuda()
                if mode == "half":
                    module = module.half()
                dtype = torch.float16 if mode == "half" else torch.float32
                tensor = torch.randn(1, 256, 16, 24, device="cuda", dtype=dtype, requires_grad=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mode == "amp"):
                    output = module(tensor)
                    loss = (output.float() * torch.randn_like(output.float())).mean()
                self.assertEqual(output.shape, tensor.shape)
                self.assertTrue(torch.isfinite(output).all().item())
                loss.backward()
                self.assertTrue(torch.isfinite(tensor.grad).all().item())
                self.assertTrue(finite_nonzero_gradient(module.sparse_relation.output_proj.weight))
                del loss, output, tensor, module
                gc.collect()
                torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_complete_model_cuda_fp32_amp_and_half_640(self):
        """Run complete CUDA FP32/AMP forwards and explicit-half 640 inference."""
        for use_amp in (False, True):
            with self.subTest(amp=use_amp):
                model = RTDETRDetectionModel(str(GSDR_CFG), ch=3, nc=1, verbose=False).eval().cuda()
                image = torch.zeros(1, 3, 128, 128, device="cuda")
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    output = model(image)
                self.assertEqual(output[0].shape, (1, 300, 5))
                self.assertTrue(torch.isfinite(output[0]).all().item())
                del output, image, model
                gc.collect()
                torch.cuda.empty_cache()

        model = RTDETRDetectionModel(str(GSDR_CFG), ch=3, nc=1, verbose=False).eval().cuda().half()
        image = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            output = model(image)
        self.assertEqual(output[0].shape, (1, 300, 5))
        self.assertEqual(output[0].dtype, torch.float16)
        self.assertTrue(torch.isfinite(output[0]).all().item())
        del output, image, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
