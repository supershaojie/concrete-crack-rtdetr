# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for constrained cross-scale semantic-consistency edge fusion."""

import gc
import hashlib
from io import BytesIO
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import CSCEFv2
from ultralytics.nn.tasks import RTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ultralytics-main" / "ultralytics" / "cfg" / "models" / "rt-detr"
V2_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v2.yaml"
PROTECTED_SHA256 = {
    MODEL_DIR / "rtdetr-resnet18-lite.yaml": "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    MODEL_DIR
    / "rtdetr-resnet18-lite-cscef.yaml": "863ed9f2786737bbc0f96879b5f79c65b8a72fba2f65094cd64624ff54f70820",
    ROOT
    / "ultralytics-main"
    / "ultralytics"
    / "nn"
    / "modules"
    / "cscef.py": "9c6ade426fa468915e8325fb88b40e31facc3bff41472866e96aefdc11ae3cbf",
}


class CSCEFv2Test(unittest.TestCase):
    """Exercise CSCEF-v2 modules and complete RT-DETR models without training."""

    def test_constructor_validation_and_group_selection(self):
        """Reject invalid shared-projection configurations and select valid GroupNorm groups."""
        invalid_arguments = (
            ((0, 16), {}),
            ((16, 0), {}),
            ((16, 32), {}),
            ((16, 16), {"hidden_channels": 0}),
            ((16, 16), {"eps": 0}),
            ((16, 16), {"temperature_init": 0.5}),
            ((16, 16), {"temperature_min": 4.0, "temperature_max": 0.5}),
            ((16, 16), {"layer_scale_init": 0}),
            ((16, 16), {"layer_scale_init": 0.1, "layer_scale_max": 0.1}),
            ((16, 16), {"similarity_clip": 0}),
        )
        for args, kwargs in invalid_arguments:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                CSCEFv2(*args, **kwargs)

        expected_groups = {32: 8, 12: 4, 10: 2, 7: 1}
        for hidden_channels, groups in expected_groups.items():
            with self.subTest(hidden_channels=hidden_channels):
                module = CSCEFv2(16, 16, hidden_channels=hidden_channels)
                self.assertEqual(module.shared_norm.num_groups, groups)
                self.assertEqual(module.edge_calibration[1].num_groups, groups)

    def test_cpu_forward_alignment_shape_and_finiteness(self):
        """Run FP32 forwards for aligned and resized semantic inputs."""
        torch.manual_seed(0)
        module = CSCEFv2(16, 16, hidden_channels=8)
        lateral = torch.randn(2, 16, 12, 14)
        for semantic_size in ((12, 14), (6, 7)):
            with self.subTest(semantic_size=semantic_size):
                semantic = torch.randn(2, 16, *semantic_size)
                output = module([lateral, semantic])
                self.assertEqual(output.shape, lateral.shape)
                self.assertEqual(output.dtype, lateral.dtype)
                self.assertTrue(torch.isfinite(output).all().item())

    def test_scharr_buffers_values_reflect_padding_and_fallback(self):
        """Check normalized fixed kernels, reflect padding, and singleton-dimension fallback."""
        module = CSCEFv2(8, 8, hidden_channels=4)
        buffers = dict(module.named_buffers())
        parameters = dict(module.named_parameters())
        self.assertIn("scharr_x", buffers)
        self.assertIn("scharr_y", buffers)
        self.assertNotIn("scharr_x", parameters)
        self.assertNotIn("scharr_y", parameters)
        expected_x = torch.tensor(((3.0, 0.0, -3.0), (10.0, 0.0, -10.0), (3.0, 0.0, -3.0))) / 32.0
        torch.testing.assert_close(module.scharr_x.squeeze(), expected_x, rtol=0, atol=0)
        torch.testing.assert_close(module.scharr_y.squeeze(), expected_x.T, rtol=0, atol=0)
        self.assertEqual(module.scharr_x.abs().sum().item(), 1.0)
        self.assertEqual(module.scharr_y.abs().sum().item(), 1.0)

        feature = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        padded = F.pad(feature, (1, 1, 1, 1), mode="reflect")
        gradient_x = F.conv2d(padded, module.scharr_x)
        gradient_y = F.conv2d(padded, module.scharr_y)
        expected_magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + module.eps)
        torch.testing.assert_close(module._scharr_magnitude(feature), expected_magnitude)

        for shape in ((1, 1), (1, 5), (5, 1)):
            with self.subTest(shape=shape):
                singleton = torch.ones(2, 4, *shape)
                magnitude = module._scharr_magnitude(singleton)
                self.assertEqual(magnitude.shape, singleton.shape)
                self.assertTrue(torch.isfinite(magnitude).all().item())
                torch.testing.assert_close(magnitude, torch.full_like(magnitude, module.eps**0.5), atol=1e-7, rtol=0)

    def test_temperature_and_layer_scale_constraints(self):
        """Check inverse-mapped initial values and effective bounds."""
        module = CSCEFv2(16, 16)
        self.assertAlmostEqual(module._effective_temperature().item(), 1.0, places=6)
        torch.testing.assert_close(
            module._effective_layer_scale(), torch.full((1, 16, 1, 1), 0.01), rtol=1e-6, atol=1e-7
        )

        with torch.no_grad():
            module.raw_temperature.fill_(-100)
            module.layer_scale_raw.fill_(-100)
        self.assertGreaterEqual(module._effective_temperature().item(), 0.5)
        self.assertLessEqual(module._effective_temperature().item(), 4.0)
        self.assertTrue((module._effective_layer_scale().abs() <= 0.1).all().item())
        with torch.no_grad():
            module.raw_temperature.fill_(100)
            module.layer_scale_raw.fill_(100)
        self.assertGreaterEqual(module._effective_temperature().item(), 0.5)
        self.assertLessEqual(module._effective_temperature().item(), 4.0)
        self.assertTrue((module._effective_layer_scale().abs() <= 0.1).all().item())

    def test_semantic_gate_range_distribution_and_zero_variance(self):
        """Keep random gates non-saturated and constant-similarity gates centered at one half."""
        torch.manual_seed(1)
        module = CSCEFv2(16, 16, hidden_channels=8)
        lateral = torch.randn(2, 8, 10, 12)
        semantic = torch.randn(2, 8, 10, 12)
        gate = module._semantic_gate(lateral, semantic)
        self.assertEqual(gate.dtype, torch.float32)
        self.assertTrue(torch.isfinite(gate).all().item())
        self.assertTrue(((gate > 0) & (gate < 1)).all().item())
        self.assertGreater(gate.std().item(), 0.02)
        self.assertGreater(gate.mean().item(), 0.35)
        self.assertLess(gate.mean().item(), 0.65)
        self.assertLess((gate < 0.05).float().mean().item(), 0.02)
        self.assertLess((gate > 0.95).float().mean().item(), 0.02)

        constant = torch.ones(2, 8, 5, 7)
        constant_gate = module._semantic_gate(constant, constant)
        self.assertTrue(torch.isfinite(constant_gate).all().item())
        self.assertLess((constant_gate - 0.5).abs().max().item(), 5e-4)
        near_constant = constant + torch.randn_like(constant) * 1e-8
        near_constant_gate = module._semantic_gate(constant, near_constant)
        self.assertTrue(torch.isfinite(near_constant_gate).all().item())
        self.assertAlmostEqual(near_constant_gate.mean().item(), 0.5, places=3)

    def test_first_backward_reaches_every_new_trainable_branch(self):
        """Verify finite, nonzero first-step gradients throughout the new path."""
        torch.manual_seed(2)
        module = CSCEFv2(16, 16, hidden_channels=8)
        lateral = torch.randn(2, 16, 8, 10, requires_grad=True)
        semantic = torch.randn(2, 16, 8, 10, requires_grad=True)
        output = module([lateral, semantic])
        weights = torch.randn_like(output)
        (output * weights).mean().backward()

        required_names = {
            "shared_projection.weight",
            "shared_norm.weight",
            "shared_norm.bias",
            "edge_calibration.0.weight",
            "edge_calibration.1.weight",
            "edge_calibration.1.bias",
            "output_projection.weight",
            "raw_temperature",
            "layer_scale_raw",
        }
        named_parameters = dict(module.named_parameters())
        self.assertTrue(required_names.issubset(named_parameters))
        for name in required_names:
            with self.subTest(parameter=name):
                gradient = named_parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_state_dict_round_trip(self):
        """Restore every parameter and buffer without changing module output."""
        torch.manual_seed(3)
        module = CSCEFv2(16, 16, hidden_channels=8).eval()
        lateral = torch.randn(2, 16, 8, 10)
        semantic = torch.randn(2, 16, 4, 5)
        expected = module([lateral, semantic])
        buffer = BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        restored = CSCEFv2(16, 16, hidden_channels=8).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        torch.testing.assert_close(restored([lateral, semantic]), expected, rtol=0, atol=0)

    def test_complete_models_nc1_nc80_cpu(self):
        """Build complete models, verify topology, and run nc=1/nc=80 CPU FP32 inference."""
        for nc in (1, 80):
            with self.subTest(nc=nc):
                model = RTDETRDetectionModel(str(V2_CFG), ch=3, nc=nc, verbose=False).eval()
                self.assertIsInstance(model.model[18], CSCEFv2)
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

    def test_protected_baseline_and_cscef_v1_files_unchanged(self):
        """Pin the exact baseline and CSCEF-v1 source/config bytes from fa829d3."""
        for path, expected in PROTECTED_SHA256.items():
            with self.subTest(path=path):
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_module_fp32_and_amp(self):
        """Run CSCEF-v2 CUDA FP32 and AMP forward/backward checks."""
        for use_amp in (False, True):
            with self.subTest(amp=use_amp):
                module = CSCEFv2(16, 16, hidden_channels=8).cuda()
                lateral = torch.randn(2, 16, 8, 10, device="cuda", requires_grad=True)
                semantic = torch.randn(2, 16, 4, 5, device="cuda", requires_grad=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    output = module([lateral, semantic])
                    loss = output.float().square().mean()
                self.assertEqual(output.shape, lateral.shape)
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
                model = RTDETRDetectionModel(str(V2_CFG), ch=3, nc=1, verbose=False).eval().cuda()
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
