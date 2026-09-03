# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for bias-free discrepancy-gated cross-scale edge fusion."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import inspect
from io import BytesIO
from pathlib import Path
import py_compile
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn.functional as F

from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import CSCEFv3
from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ultralytics-main" / "ultralytics" / "cfg" / "models" / "rt-detr"
MODULE_DIR = ROOT / "ultralytics-main" / "ultralytics" / "nn" / "modules"
V3_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v3.yaml"
PROTECTED_SHA256 = {
    MODEL_DIR / "rtdetr-resnet18-lite.yaml": "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    MODEL_DIR
    / "rtdetr-resnet18-lite-cscef.yaml": "27545ed78ed8308f8c0e5f18abfecdcfc5a0ef69cdf1c605467d564e432bf895",
    MODEL_DIR
    / "rtdetr-resnet18-lite-cscef-v2.yaml": "8275efea9212131799508c57da8c411baf89625cac0be05f9a5c42c3410b906e",
    MODULE_DIR / "cscef.py": "6826911701ce14b08bb99945fa7790897d9389838a78cd3227c5514c0d945cda",
    MODULE_DIR / "cscef_v2.py": "050aa1c9e2a2fc5ffd83bd217cf859fb9607c9f00b7ba405af4e78faf625b9aa",
}


def canonical_lf_sha256(data: bytes) -> str:
    """Hash text bytes after normalizing CRLF and CR line endings to canonical LF."""
    canonical = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def channel_direction(values: list[float], height: int = 3, width: int = 4) -> torch.Tensor:
    """Create a spatially repeated, non-degenerate channel direction."""
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1).expand(1, -1, height, width).clone()


class CSCEFv3Test(unittest.TestCase):
    """Exercise CSCEF-v3 modules and complete RT-DETR models without training."""

    def test_py_compile_and_public_import(self):
        """Compile every new Python entry point and import CSCEFv3 through the public module package."""
        files = (
            MODULE_DIR / "cscef_v3.py",
            ROOT / "tools" / "init_rtdetr_r18_lite_cscef_v3_from_baseline.py",
            ROOT / "tools" / "audit_rtdetr_r18_lite_cscef_v3.py",
        )
        with TemporaryDirectory(dir=ROOT) as directory:
            for index, path in enumerate(files):
                with self.subTest(path=path):
                    py_compile.compile(str(path), cfile=str(Path(directory) / f"compiled_{index}.pyc"), doraise=True)
        from ultralytics.nn.modules import CSCEFv3 as PublicCSCEFv3

        self.assertIs(PublicCSCEFv3, CSCEFv3)

    def test_constructor_validation(self):
        """Reject invalid shared projection, GroupNorm, alpha, and epsilon configurations."""
        invalid_arguments = (
            ((0, 16), {}),
            ((16, 0), {}),
            ((16, 32), {}),
            ((16, 16), {"hidden_channels": 0}),
            ((16, 16), {"hidden_channels": 10}),
            ((16, 16), {"num_groups": 0}),
            ((16, 16), {"eps": 0}),
            ((16, 16), {"alpha_init": 0}),
            ((16, 16), {"alpha_init": 0.05}),
            ((16, 16), {"alpha_init": 0.1, "alpha_max": 0.05}),
        )
        for args, kwargs in invalid_arguments:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                CSCEFv3(*args, **kwargs)

    def test_exact_parameterization_normalization_and_bias_constraints(self):
        """Enforce the exact default parameter count and removal of all forbidden v2 parameters."""
        module = CSCEFv3(256, 256)
        named_parameters = dict(module.named_parameters())
        self.assertEqual(sum(parameter.numel() for parameter in named_parameters.values()), 16673)
        self.assertEqual(
            set(named_parameters),
            {"shared_projection.weight", "depthwise_conv.weight", "output_projection.weight", "raw_alpha"},
        )
        self.assertEqual(module.raw_alpha.numel(), 1)
        self.assertFalse(module.shared_norm.affine)
        self.assertFalse(module.edge_norm.affine)
        self.assertIsNone(module.shared_norm.weight)
        self.assertIsNone(module.shared_norm.bias)
        self.assertIsNone(module.edge_norm.weight)
        self.assertIsNone(module.edge_norm.bias)
        self.assertEqual(module.shared_norm.num_groups, 8)
        self.assertEqual(module.edge_norm.num_groups, 8)
        self.assertIsNone(module.shared_projection.bias)
        self.assertIsNone(module.depthwise_conv.bias)
        self.assertIsNone(module.output_projection.bias)
        forbidden = ("temperature", "similarity_bias", "layer_scale")
        self.assertFalse(any(token in name for name in named_parameters for token in forbidden))

    def test_alpha_initialization_and_strict_bounds(self):
        """Check scalar inverse-logit initialization and the strict open effective-alpha interval."""
        module = CSCEFv3(16, 16)
        self.assertAlmostEqual(module.raw_alpha.item(), -1.38629436112, places=6)
        self.assertAlmostEqual(module._effective_alpha().item(), 0.01, places=7)
        for raw in (-float("inf"), -100.0, 0.0, 100.0, float("inf")):
            with self.subTest(raw=raw), torch.no_grad():
                module.raw_alpha.fill_(raw)
                alpha = module._effective_alpha().item()
                self.assertGreater(alpha, 0.0)
                self.assertLess(alpha, 0.05)

    def test_discrepancy_gate_boundaries_shape_and_gradient(self):
        """Verify the mathematical gate boundaries, single-channel shape, FP32 dtype, and attached gradient."""
        module = CSCEFv3(8, 8, hidden_channels=8)
        direction = channel_direction([1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.25, -0.25])
        orthogonal = channel_direction([0.0, 0.0, 1.0, -1.0, -0.25, 0.25, 0.5, -0.5])
        cases = ((direction, 0.0), (orthogonal, 0.5), (-direction, 1.0))
        for semantic, expected in cases:
            with self.subTest(expected=expected):
                gate = module._compute_discrepancy_gate(direction, semantic)
                self.assertEqual(gate.shape, (1, 1, 3, 4))
                self.assertEqual(gate.dtype, torch.float32)
                self.assertTrue(torch.isfinite(gate).all().item())
                torch.testing.assert_close(gate, torch.full_like(gate, expected), rtol=0, atol=1e-6)

        torch.manual_seed(0)
        lateral = torch.randn(2, 8, 3, 4, requires_grad=True)
        semantic = torch.randn(2, 8, 3, 4, requires_grad=True)
        module._compute_discrepancy_gate(lateral, semantic).sum().backward()
        self.assertIsNotNone(lateral.grad)
        self.assertIsNotNone(semantic.grad)
        self.assertGreater(lateral.grad.abs().sum().item(), 0.0)
        self.assertGreater(semantic.grad.abs().sum().item(), 0.0)

    def test_gate_has_no_forced_per_image_half_mean(self):
        """Show that gate spatial means follow discrepancies instead of per-image standardization."""
        module = CSCEFv3(8, 8, hidden_channels=8)
        direction = channel_direction([1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.25, -0.25], 2, 4)
        semantic = direction.clone()
        semantic[:, :, :, -1] = -direction[:, :, :, -1]
        gate = module._compute_discrepancy_gate(direction, semantic)
        self.assertAlmostEqual(gate.mean().item(), 0.25, places=6)
        self.assertNotAlmostEqual(gate.mean().item(), 0.5, places=3)

    def test_scharr_buffers_fp32_values_padding_and_finiteness(self):
        """Check normalized fixed kernels, FP32 magnitude, reflect padding, and singleton fallback."""
        module = CSCEFv3(8, 8, hidden_channels=8)
        buffers = dict(module.named_buffers())
        parameters = dict(module.named_parameters())
        self.assertIn("scharr_x", buffers)
        self.assertIn("scharr_y", buffers)
        self.assertNotIn("scharr_x", parameters)
        self.assertNotIn("scharr_y", parameters)
        expected_x = torch.tensor(((3.0, 0.0, -3.0), (10.0, 0.0, -10.0), (3.0, 0.0, -3.0))) / 32.0
        torch.testing.assert_close(module.scharr_x.squeeze(), expected_x, rtol=0, atol=0)
        torch.testing.assert_close(module.scharr_y.squeeze(), expected_x.T, rtol=0, atol=0)

        feature = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        padded = F.pad(feature, (1, 1, 1, 1), mode="reflect")
        expected = torch.sqrt(
            F.conv2d(padded, module.scharr_x).square()
            + F.conv2d(padded, module.scharr_y).square()
            + module.eps
        )
        actual = module._compute_scharr_magnitude(feature)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

        for shape in ((1, 1), (1, 5), (5, 1)):
            with self.subTest(shape=shape):
                magnitude = module._compute_scharr_magnitude(torch.ones(2, 8, *shape))
                self.assertEqual(magnitude.shape, (2, 8, *shape))
                self.assertEqual(magnitude.dtype, torch.float32)
                self.assertTrue(torch.isfinite(magnitude).all().item())
                torch.testing.assert_close(
                    magnitude, torch.full_like(magnitude, module.eps**0.5), atol=1e-7, rtol=0
                )

    def test_cpu_forward_rms_order_bound_shape_dtype_and_finiteness(self):
        """Verify gate-before-injection math, gate-free RMS matching, and the 5% residual bound on CPU."""
        torch.manual_seed(1)
        module = CSCEFv3(16, 16, hidden_channels=8).eval()
        lateral = torch.randn(2, 16, 10, 12)
        semantic = torch.randn(2, 16, 5, 6)
        output = module([lateral, semantic])
        self.assertEqual(output.shape, lateral.shape)
        self.assertEqual(output.dtype, lateral.dtype)
        self.assertTrue(torch.isfinite(output).all().item())

        lateral_embedding, semantic_embedding = module._project_features(lateral, semantic)
        gate = module._compute_discrepancy_gate(lateral_embedding, semantic_embedding)
        edge = module._compute_scharr_magnitude(lateral_embedding)
        edge_feature = module.edge_activation(module.edge_norm(module.depthwise_conv(edge)))
        residual_raw = module.output_projection(edge_feature)
        residual_normalized = module._match_residual_rms(lateral, residual_raw)
        expected = lateral + (module._effective_alpha() * gate * residual_normalized).to(lateral.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        raw_rms = torch.sqrt(residual_raw.float().square().mean(dim=(1, 2, 3)))
        normalized_rms = torch.sqrt(residual_normalized.square().mean(dim=(1, 2, 3)))
        lateral_rms = torch.sqrt(lateral.float().square().mean(dim=(1, 2, 3)))
        self.assertTrue((normalized_rms <= lateral_rms + 2e-6).all().item())
        self.assertTrue((raw_rms > 0).all().item())
        delta_rms = torch.sqrt((output - lateral).float().square().mean(dim=(1, 2, 3)))
        ratio = delta_rms / lateral_rms
        self.assertLessEqual(ratio.max().item(), module.alpha_max + 1e-6)

        source = inspect.getsource(CSCEFv3.forward)
        self.assertIn("_match_residual_rms(x_lateral, residual_raw)", source)
        self.assertLess(source.index("residual_normalized ="), source.index("delta ="))

    def test_first_backward_reaches_every_new_trainable_branch(self):
        """Verify finite, nonzero gradients for every trainable v3 parameter and both inputs."""
        torch.manual_seed(2)
        module = CSCEFv3(16, 16, hidden_channels=8)
        lateral = torch.randn(2, 16, 8, 10, requires_grad=True)
        semantic = torch.randn(2, 16, 4, 5, requires_grad=True)
        output = module([lateral, semantic])
        (output * torch.randn_like(output)).mean().backward()
        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(lateral.grad)
        self.assertIsNotNone(semantic.grad)
        self.assertTrue(torch.isfinite(lateral.grad).all().item())
        self.assertTrue(torch.isfinite(semantic.grad).all().item())

    def test_module_state_dict_round_trip(self):
        """Reload all v3 parameters and fixed buffers without changing output."""
        torch.manual_seed(3)
        module = CSCEFv3(16, 16, hidden_channels=8).eval()
        lateral = torch.randn(2, 16, 8, 10)
        semantic = torch.randn(2, 16, 4, 5)
        expected = module([lateral, semantic])
        buffer = BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        restored = CSCEFv3(16, 16, hidden_channels=8).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        torch.testing.assert_close(restored([lateral, semantic]), expected, rtol=0, atol=0)

    def test_parser_yaml_topology_and_complete_models_nc1_nc80_cpu(self):
        """Build v3 through the parser and run nc=1/nc=80 complete CPU FP32 forwards."""
        for nc in (1, 80):
            with self.subTest(nc=nc):
                model = RTDETRDetectionModel(str(V3_CFG), ch=3, nc=nc, verbose=False).eval()
                self.assertIsInstance(model.model[18], CSCEFv3)
                self.assertEqual(model.model[18].f, [17, 16])
                self.assertEqual(model.model[19].f, [16, 18])
                self.assertEqual(model.model[-1].f, [20, 23, 26])
                with torch.no_grad():
                    output = model(torch.zeros(1, 3, 128, 128))
                self.assertIsInstance(output, tuple)
                self.assertEqual(output[0].shape, (1, 300, nc + 4))
                self.assertTrue(torch.isfinite(output[0]).all().item())
                del output, model
                gc.collect()

    def test_direct_checkpoint_and_yaml_load_compatibility(self):
        """Reload a direct model checkpoint and load it into a fresh RTDETR(YAML) instance."""
        model = RTDETRDetectionModel(str(V3_CFG), ch=3, nc=1, verbose=False).eval()
        with TemporaryDirectory(dir=ROOT) as directory:
            checkpoint_path = Path(directory) / "cscef_v3_test.pt"
            model.args = {**DEFAULT_CFG_DICT, "model": str(V3_CFG), "task": "detect"}
            model.task = "detect"
            torch.save({"model": deepcopy(model).half(), "train_args": model.args}, checkpoint_path)
            self.assertTrue(checkpoint_path.is_file())
            direct = RTDETR(str(checkpoint_path))
            self.assertIsInstance(direct.model.model[18], CSCEFv3)
            yaml_loaded = RTDETR(str(V3_CFG))
            yaml_loaded.load(str(checkpoint_path))
            self.assertIsInstance(yaml_loaded.model.model[18], CSCEFv3)

    def test_baseline_and_v2_regression_builds_and_protected_files(self):
        """Keep canonical-LF baseline/v1/v2 content unchanged and ensure baseline and CSCEF-v2 still parse."""
        for path, expected in PROTECTED_SHA256.items():
            with self.subTest(path=path):
                self.assertEqual(canonical_lf_sha256(path.read_bytes()), expected)
        baseline = RTDETRDetectionModel(str(MODEL_DIR / "rtdetr-resnet18-lite.yaml"), ch=3, nc=1, verbose=False)
        v2 = RTDETRDetectionModel(str(MODEL_DIR / "rtdetr-resnet18-lite-cscef-v2.yaml"), ch=3, nc=1, verbose=False)
        self.assertNotIsInstance(baseline.model[18], CSCEFv3)
        self.assertEqual(type(v2.model[18]).__name__, "CSCEFv2")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_module_fp32_and_amp_fp16(self):
        """Run real CUDA FP32 and AMP FP16 module forward/backward checks."""
        for use_amp in (False, True):
            with self.subTest(amp=use_amp):
                module = CSCEFv3(16, 16, hidden_channels=8).cuda()
                dtype = torch.float16 if use_amp else torch.float32
                lateral = torch.randn(2, 16, 8, 10, device="cuda", dtype=dtype, requires_grad=True)
                semantic = torch.randn(2, 16, 4, 5, device="cuda", dtype=dtype, requires_grad=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    output = module([lateral, semantic])
                    loss = output.float().square().mean()
                self.assertEqual(output.shape, lateral.shape)
                self.assertEqual(output.dtype, lateral.dtype)
                self.assertTrue(torch.isfinite(output).all().item())
                loss.backward()
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_complete_model_cuda_fp32_and_amp(self):
        """Run complete nc=1 RT-DETR CUDA FP32 and AMP inference."""
        for use_amp in (False, True):
            with self.subTest(amp=use_amp):
                model = RTDETRDetectionModel(str(V3_CFG), ch=3, nc=1, verbose=False).eval().cuda()
                image = torch.zeros(1, 3, 128, 128, device="cuda")
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    output = model(image)
                self.assertEqual(output[0].shape, (1, 300, 5))
                self.assertTrue(torch.isfinite(output[0]).all().item())
                del output, image, model
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
