# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for structure-consistent, conflict-suppressed cross-scale edge fusion."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import inspect
from io import BytesIO
from pathlib import Path
import py_compile
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn.functional as F

from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import CSCEFv3, CSCEFv4
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ultralytics-main" / "ultralytics" / "cfg" / "models" / "rt-detr"
MODULE_DIR = ROOT / "ultralytics-main" / "ultralytics" / "nn" / "modules"
V4_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v4.yaml"
V3_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v3.yaml"
PROTECTED_SHA256 = {
    MODEL_DIR / "rtdetr-resnet18-lite.yaml": "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    MODEL_DIR
    / "rtdetr-resnet18-lite-cscef.yaml": "27545ed78ed8308f8c0e5f18abfecdcfc5a0ef69cdf1c605467d564e432bf895",
    MODEL_DIR
    / "rtdetr-resnet18-lite-cscef-v2.yaml": "8275efea9212131799508c57da8c411baf89625cac0be05f9a5c42c3410b906e",
    V3_CFG: "cf679dc1cde990419ca549299be0f991b002cc31d64fdfb9b44e50a3ea10cdbc",
    MODULE_DIR / "cscef.py": "6826911701ce14b08bb99945fa7790897d9389838a78cd3227c5514c0d945cda",
    MODULE_DIR / "cscef_v2.py": "050aa1c9e2a2fc5ffd83bd217cf859fb9607c9f00b7ba405af4e78faf625b9aa",
    MODULE_DIR / "cscef_v3.py": "40f5fae030d21b9eeee67385ebdd058f2c1d1e283396c76d2915136ee8f62143",
    ROOT / "ultralytics-main" / "tests" / "test_cscef_v2.py": (
        "834ab9fa70415a358ae0036c764f5ee50d6abaa6f9a35d12556a5dce95e2ae12"
    ),
    ROOT / "ultralytics-main" / "tests" / "test_cscef_v3.py": (
        "995659103686f97f2603c7a1c08443a339fd6885bb97539cd246a52cd5c1c211"
    ),
    ROOT / "tools" / "audit_rtdetr_r18_lite_cscef_v2.py": (
        "fe1e86d2312c50375c293784d11852db4faa21ab1e2c6538e76facf8b8006310"
    ),
    ROOT / "tools" / "audit_rtdetr_r18_lite_cscef_v3.py": (
        "254500dffdde0b93c7a4bde92e2fac7f1f428ebc20958266d9572f3872d316d3"
    ),
}


def canonical_lf_sha256(data: bytes) -> str:
    """Hash text bytes after normalizing CRLF and CR line endings to canonical LF."""
    canonical = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def channel_direction(values: list[float], height: int = 3, width: int = 4) -> torch.Tensor:
    """Create a spatially repeated, non-degenerate channel direction."""
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1).expand(1, -1, height, width).clone()


class CSCEFv4Test(unittest.TestCase):
    """Exercise CSCEF-v4 math, module behavior, parser integration, and complete RT-DETR models."""

    def test_py_compile_and_public_import(self):
        """Compile all V4 entry points and import CSCEFv4 through the public module package."""
        files = (
            MODULE_DIR / "cscef_v4.py",
            ROOT / "tools" / "init_rtdetr_r18_lite_cscef_v4_controlled.py",
            ROOT / "tools" / "audit_rtdetr_r18_lite_cscef_v4.py",
            Path(__file__),
        )
        with TemporaryDirectory(dir=ROOT) as directory:
            for index, path in enumerate(files):
                with self.subTest(path=path):
                    py_compile.compile(str(path), cfile=str(Path(directory) / f"compiled_{index}.pyc"), doraise=True)
        from ultralytics.nn.modules import CSCEFv4 as PublicCSCEFv4

        self.assertIs(PublicCSCEFv4, CSCEFv4)

    def test_constructor_and_input_validation(self):
        """Reject invalid channels, grouping, alpha, epsilon, and forward inputs."""
        invalid_arguments = (
            ((0, 16), {}),
            ((16, 0), {}),
            ((16, 32), {}),
            ((16, 16), {"hidden_channels": 0}),
            ((16, 16), {"hidden_channels": 10}),
            ((16, 16), {"num_groups": 0}),
            ((16, 16), {"eps": 0}),
            ((16, 16), {"eps": float("inf")}),
            ((16, 16), {"alpha_init": 0}),
            ((16, 16), {"alpha_init": 0.05}),
            ((16, 16), {"alpha_init": 0.1, "alpha_max": 0.05}),
        )
        for args, kwargs in invalid_arguments:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                CSCEFv4(*args, **kwargs)
        module = CSCEFv4(8, 8, hidden_channels=8)
        with self.assertRaises(ValueError):
            module(torch.zeros(1, 8, 3, 3))
        with self.assertRaises(ValueError):
            module([torch.zeros(1, 8, 3, 3)])
        with self.assertRaises(ValueError):
            module([torch.zeros(1, 7, 3, 3), torch.zeros(1, 7, 3, 3)])

    def test_parameter_names_shapes_count_and_initialization_match_v3(self):
        """Keep every trainable parameter name, shape, count, and seeded initialization identical to V3."""
        torch.manual_seed(7)
        v3 = CSCEFv3(256, 256)
        torch.manual_seed(7)
        v4 = CSCEFv4(256, 256)
        v3_parameters = dict(v3.named_parameters())
        v4_parameters = dict(v4.named_parameters())
        expected_names = {
            "shared_projection.weight",
            "depthwise_conv.weight",
            "output_projection.weight",
            "raw_alpha",
        }
        self.assertEqual(set(v4_parameters), expected_names)
        self.assertEqual(set(v4_parameters), set(v3_parameters))
        self.assertEqual(sum(parameter.numel() for parameter in v4_parameters.values()), 16673)
        for name in expected_names:
            with self.subTest(parameter=name):
                self.assertEqual(v4_parameters[name].shape, v3_parameters[name].shape)
                torch.testing.assert_close(v4_parameters[name], v3_parameters[name], rtol=0, atol=0)

    def test_bias_normalization_buffers_and_forbidden_parameters(self):
        """Enforce bias-free convolutions, affine-free normalization, fixed Scharr buffers, and no extra parameters."""
        module = CSCEFv4(256, 256)
        self.assertFalse(module.shared_norm.affine)
        self.assertFalse(module.edge_norm.affine)
        self.assertIsNone(module.shared_norm.weight)
        self.assertIsNone(module.shared_norm.bias)
        self.assertIsNone(module.edge_norm.weight)
        self.assertIsNone(module.edge_norm.bias)
        for convolution in (module.shared_projection, module.depthwise_conv, module.output_projection):
            self.assertIsNone(convolution.bias)
        parameters = dict(module.named_parameters())
        buffers = dict(module.named_buffers())
        self.assertNotIn("scharr_x", parameters)
        self.assertNotIn("scharr_y", parameters)
        self.assertIn("scharr_x", buffers)
        self.assertIn("scharr_y", buffers)
        forbidden = ("attention", "temperature", "similarity_bias", "gate_floor", "layer_scale")
        self.assertFalse(any(token in name for name in parameters for token in forbidden))

    def test_alpha_initialization_and_strict_bounds(self):
        """Check inverse-logit initialization and the strict open effective-alpha interval."""
        module = CSCEFv4(16, 16)
        self.assertAlmostEqual(module.raw_alpha.item(), -1.38629436112, places=6)
        self.assertAlmostEqual(module._effective_alpha().item(), 0.01, places=7)
        for raw in (-float("inf"), -100.0, 0.0, 100.0, float("inf")):
            with self.subTest(raw=raw), torch.no_grad():
                module.raw_alpha.fill_(raw)
                alpha = module._effective_alpha().item()
                self.assertGreater(alpha, 0.0)
                self.assertLess(alpha, module.alpha_max)

    def test_semantic_bandpass_boundaries_shape_range_and_gradient(self):
        """Verify identical/orthogonal/opposite boundaries, FP32 invariants, and gradients to both inputs."""
        module = CSCEFv4(8, 8, hidden_channels=8)
        direction = channel_direction([1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.25, -0.25])
        orthogonal = channel_direction([0.0, 0.0, 1.0, -1.0, -0.25, 0.25, 0.5, -0.5])
        for semantic, expected in ((direction, 0.0), (orthogonal, 1.0), (-direction, 0.0)):
            with self.subTest(expected=expected):
                bandpass = module._compute_semantic_bandpass(direction, semantic)
                self.assertEqual(bandpass.shape, (1, 1, 3, 4))
                self.assertEqual(bandpass.dtype, torch.float32)
                self.assertTrue(torch.isfinite(bandpass).all().item())
                self.assertTrue(((0.0 <= bandpass) & (bandpass <= 1.0)).all().item())
                torch.testing.assert_close(bandpass, torch.full_like(bandpass, expected), rtol=0, atol=1e-6)

        torch.manual_seed(8)
        lateral = torch.randn(2, 8, 3, 4, requires_grad=True)
        semantic = torch.randn(2, 8, 3, 4, requires_grad=True)
        module._compute_semantic_bandpass(lateral, semantic).sum().backward()
        for tensor in (lateral, semantic):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

    def test_gate_has_no_forced_per_image_half_mean(self):
        """Show that the semantic bandpass and final gate are not standardized to a mean of 0.5."""
        module = CSCEFv4(8, 8, hidden_channels=8)
        direction = channel_direction([1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.25, -0.25], 4, 4)
        orthogonal = channel_direction([0.0, 0.0, 1.0, -1.0, -0.25, 0.25, 0.5, -0.5], 4, 4)
        semantic = direction.clone()
        semantic[:, :, :, -1] = orthogonal[:, :, :, -1]
        bandpass = module._compute_semantic_bandpass(direction, semantic)
        gx = torch.ones_like(direction)
        gy = torch.zeros_like(direction)
        final_gate = bandpass * module._compute_structure_confidence(gx, gy)
        self.assertAlmostEqual(bandpass.mean().item(), 0.25, places=6)
        self.assertNotAlmostEqual(final_gate.mean().item(), 0.5, places=3)

    def test_scharr_components_match_v3_and_small_spatial_fallback(self):
        """Check exact V3 kernels, component numerics, one Scharr call site, and singleton spatial fallback."""
        v3 = CSCEFv3(8, 8, hidden_channels=8)
        v4 = CSCEFv4(8, 8, hidden_channels=8)
        torch.testing.assert_close(v4.scharr_x, v3.scharr_x, rtol=0, atol=0)
        torch.testing.assert_close(v4.scharr_y, v3.scharr_y, rtol=0, atol=0)
        feature = torch.arange(96, dtype=torch.float32).reshape(1, 8, 3, 4)
        padded = F.pad(feature, (1, 1, 1, 1), mode="reflect")
        expected_x = F.conv2d(padded, v4.scharr_x.expand(8, 1, 3, 3), groups=8)
        expected_y = F.conv2d(padded, v4.scharr_y.expand(8, 1, 3, 3), groups=8)
        actual_x, actual_y = v4._compute_scharr_components(feature)
        self.assertEqual(actual_x.dtype, torch.float32)
        self.assertEqual(actual_y.dtype, torch.float32)
        torch.testing.assert_close(actual_x, expected_x)
        torch.testing.assert_close(actual_y, expected_y)
        source = inspect.getsource(CSCEFv4.forward)
        self.assertEqual(source.count("_compute_scharr_components("), 1)
        scharr_source = inspect.getsource(CSCEFv4._compute_scharr_components)
        self.assertEqual(scharr_source.count("F.conv2d("), 1)

        for shape in ((1, 1), (1, 5), (5, 1)):
            with self.subTest(shape=shape):
                gx, gy = v4._compute_scharr_components(torch.ones(2, 8, *shape))
                self.assertEqual(gx.shape, (2, 8, *shape))
                self.assertEqual(gy.shape, (2, 8, *shape))
                self.assertTrue(torch.isfinite(gx).all().item())
                self.assertTrue(torch.isfinite(gy).all().item())
                torch.testing.assert_close(gx, torch.zeros_like(gx), rtol=0, atol=0)
                torch.testing.assert_close(gy, torch.zeros_like(gy), rtol=0, atol=0)

    def test_structure_tensor_controlled_boundaries_and_detachment(self):
        """Exercise zero, directional, isotropic, transpose, and diagonal structure tensors."""
        module = CSCEFv4(8, 8, hidden_channels=8)
        zero = torch.zeros(1, 8, 5, 7)
        zero_confidence = module._compute_structure_confidence(zero, zero)
        self.assertEqual(zero_confidence.shape, (1, 1, 5, 7))
        self.assertEqual(zero_confidence.dtype, torch.float32)
        torch.testing.assert_close(zero_confidence, torch.zeros_like(zero_confidence), rtol=0, atol=0)

        gx = torch.ones(1, 8, 5, 7, requires_grad=True)
        gy = torch.zeros_like(gx)
        coherence_x, _ = module._compute_structure_terms(gx, gy)
        coherence_y, _ = module._compute_structure_terms(gy, gx)
        torch.testing.assert_close(coherence_x, torch.ones_like(coherence_x), rtol=0, atol=3e-6)
        torch.testing.assert_close(coherence_y, torch.ones_like(coherence_y), rtol=0, atol=3e-6)

        isotropic_x = torch.zeros(1, 8, 5, 7)
        isotropic_y = torch.zeros_like(isotropic_x)
        isotropic_x[:, :4] = 1.0
        isotropic_y[:, 4:] = 1.0
        isotropic_coherence, _ = module._compute_structure_terms(isotropic_x, isotropic_y)
        torch.testing.assert_close(isotropic_coherence, torch.zeros_like(isotropic_coherence), rtol=0, atol=1e-7)

        ramp = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7).expand(1, 8, 5, 7)
        horizontal = module._compute_structure_confidence(ramp, torch.zeros_like(ramp))
        vertical_transposed = module._compute_structure_confidence(
            torch.zeros_like(ramp.transpose(-2, -1)), ramp.transpose(-2, -1)
        )
        torch.testing.assert_close(horizontal.transpose(-2, -1), vertical_transposed)

        diagonal = module._compute_structure_confidence(gx, gx)
        self.assertTrue(torch.isfinite(diagonal).all().item())
        self.assertTrue(((0.0 <= diagonal) & (diagonal <= 1.0)).all().item())
        confidence = module._compute_structure_confidence(gx, gy)
        self.assertFalse(confidence.requires_grad)

    def test_small_spatial_complete_module_forward(self):
        """Run 1x1, 1x5, and 5x1 module forwards without errors or non-finite outputs."""
        module = CSCEFv4(8, 8, hidden_channels=8, num_groups=4).eval()
        for shape in ((1, 1), (1, 5), (5, 1)):
            with self.subTest(shape=shape), torch.no_grad():
                lateral = torch.randn(2, 8, *shape)
                semantic = torch.randn(2, 8, *shape)
                output = module([lateral, semantic])
                self.assertEqual(output.shape, lateral.shape)
                self.assertEqual(output.dtype, lateral.dtype)
                self.assertTrue(torch.isfinite(output).all().item())

    def test_cpu_forward_rms_order_bound_shape_dtype_and_finiteness(self):
        """Verify gate-free RMS matching, final gate math, and the alpha_max residual bound on CPU."""
        torch.manual_seed(9)
        module = CSCEFv4(16, 16, hidden_channels=8).eval()
        lateral = torch.randn(2, 16, 10, 12)
        semantic = torch.randn(2, 16, 5, 6)
        output = module([lateral, semantic])
        self.assertEqual(output.shape, lateral.shape)
        self.assertEqual(output.dtype, lateral.dtype)
        self.assertEqual(output.device, lateral.device)
        self.assertTrue(torch.isfinite(output).all().item())

        lateral_embedding, semantic_embedding = module._project_features(lateral, semantic)
        bandpass = module._compute_semantic_bandpass(lateral_embedding, semantic_embedding)
        gx, gy = module._compute_scharr_components(lateral_embedding)
        confidence = module._compute_structure_confidence(gx, gy)
        gate = bandpass * confidence
        self.assertEqual(gate.shape, (2, 1, 10, 12))
        self.assertEqual(gate.dtype, torch.float32)
        self.assertTrue(torch.isfinite(gate).all().item())
        self.assertTrue(((0.0 <= gate) & (gate <= 1.0)).all().item())
        edge = torch.sqrt(gx.square() + gy.square() + module.eps)
        edge_feature = module.edge_activation(module.edge_norm(module.depthwise_conv(edge)))
        residual_raw = module.output_projection(edge_feature)
        residual_normalized = module._match_residual_rms(lateral, residual_raw)
        expected = lateral + (module._effective_alpha() * gate * residual_normalized).to(lateral.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        lateral_rms = torch.sqrt(lateral.float().square().mean(dim=(1, 2, 3)))
        normalized_rms = torch.sqrt(residual_normalized.square().mean(dim=(1, 2, 3)))
        self.assertTrue((normalized_rms <= lateral_rms + 2e-6).all().item())
        delta_rms = torch.sqrt((output - lateral).float().square().mean(dim=(1, 2, 3)))
        self.assertLessEqual((delta_rms / lateral_rms).max().item(), module.alpha_max + 1e-6)
        source = inspect.getsource(CSCEFv4.forward)
        self.assertLess(source.index("residual_normalized ="), source.index("delta ="))

    def test_first_backward_reaches_every_trainable_parameter(self):
        """Require finite nonzero first-backward gradients for all V4 parameters and both inputs."""
        torch.manual_seed(10)
        module = CSCEFv4(16, 16, hidden_channels=8)
        lateral = torch.randn(2, 16, 8, 10, requires_grad=True)
        semantic = torch.randn(2, 16, 4, 5, requires_grad=True)
        output = module([lateral, semantic])
        (output * torch.randn_like(output)).mean().backward()
        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        for tensor in (lateral, semantic):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

    def test_module_state_dict_round_trip(self):
        """Reload all V4 parameters and fixed buffers without changing output."""
        torch.manual_seed(11)
        module = CSCEFv4(16, 16, hidden_channels=8).eval()
        lateral = torch.randn(2, 16, 8, 10)
        semantic = torch.randn(2, 16, 4, 5)
        expected = module([lateral, semantic])
        buffer = BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        restored = CSCEFv4(16, 16, hidden_channels=8).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        torch.testing.assert_close(restored([lateral, semantic]), expected, rtol=0, atol=0)

    def test_v4_yaml_differs_from_v3_only_at_layer_18(self):
        """Enforce identical parsed V3/V4 topology except for the layer-18 module name."""
        v3 = YAML.load(V3_CFG)
        v4 = YAML.load(V4_CFG)
        self.assertEqual(v3["backbone"], v4["backbone"])
        v3_layers = v3["backbone"] + v3["head"]
        v4_layers = v4["backbone"] + v4["head"]
        expected = [[*layer] for layer in v3_layers]
        expected[18] = [[17, 16], 1, "CSCEFv4", []]
        self.assertEqual(v4_layers, expected)
        self.assertEqual(v4_layers[19][0], [16, 18])
        self.assertEqual(v4_layers[-1][0], [20, 23, 26])

    def test_parser_topology_and_complete_models_nc1_nc80_cpu(self):
        """Build V4 through the parser and run nc=1/nc=80 complete CPU FP32 forwards."""
        for nc in (1, 80):
            with self.subTest(nc=nc):
                model = RTDETRDetectionModel(str(V4_CFG), ch=3, nc=nc, verbose=False).eval()
                instances = [module for module in model.modules() if isinstance(module, CSCEFv4)]
                self.assertEqual(len(instances), 1)
                self.assertIs(model.model[18], instances[0])
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
        """Reload a direct V4 checkpoint and load it into a fresh RTDETR(YAML) instance."""
        model = RTDETRDetectionModel(str(V4_CFG), ch=3, nc=1, verbose=False).eval()
        with TemporaryDirectory(dir=ROOT) as directory:
            checkpoint_path = Path(directory) / "cscef_v4_test.pt"
            model.args = {**DEFAULT_CFG_DICT, "model": str(V4_CFG), "task": "detect"}
            model.task = "detect"
            torch.save({"model": deepcopy(model).half(), "train_args": model.args}, checkpoint_path)
            direct = RTDETR(str(checkpoint_path))
            self.assertIsInstance(direct.model.model[18], CSCEFv4)
            yaml_loaded = RTDETR(str(V4_CFG))
            yaml_loaded.load(str(checkpoint_path))
            self.assertIsInstance(yaml_loaded.model.model[18], CSCEFv4)

    def test_baseline_v1_v2_v3_regression_builds_and_protected_files(self):
        """Keep all protected files unchanged and ensure baseline plus V1/V2/V3 still build."""
        for path, expected in PROTECTED_SHA256.items():
            with self.subTest(path=path):
                self.assertEqual(canonical_lf_sha256(path.read_bytes()), expected)
        configurations = (
            "rtdetr-resnet18-lite.yaml",
            "rtdetr-resnet18-lite-cscef.yaml",
            "rtdetr-resnet18-lite-cscef-v2.yaml",
            "rtdetr-resnet18-lite-cscef-v3.yaml",
        )
        models = [
            RTDETRDetectionModel(str(MODEL_DIR / configuration), ch=3, nc=1, verbose=False)
            for configuration in configurations
        ]
        self.assertEqual(type(models[1].model[18]).__name__, "CSCEF")
        self.assertEqual(type(models[2].model[18]).__name__, "CSCEFv2")
        self.assertEqual(type(models[3].model[18]).__name__, "CSCEFv3")
        self.assertFalse(any(isinstance(module, CSCEFv4) for model in models for module in model.modules()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_module_fp32_and_amp_fp16(self):
        """Run real CUDA FP32 and AMP FP16 module forward/backward checks."""
        for use_amp in (False, True):
            with self.subTest(amp=use_amp):
                module = CSCEFv4(16, 16, hidden_channels=8).cuda()
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
    def test_cuda_half_module_direct_forward_without_autocast(self):
        """Run an explicitly half V4 module and inputs without autocast."""
        module = CSCEFv4(16, 16, hidden_channels=8).cuda().half()
        lateral = torch.randn(2, 16, 8, 10, device="cuda", dtype=torch.float16, requires_grad=True)
        semantic = torch.randn(2, 16, 4, 5, device="cuda", dtype=torch.float16, requires_grad=True)
        gx, gy = module._compute_scharr_components(torch.randn(1, 8, 8, 10, device="cuda", dtype=torch.float16))
        self.assertEqual(gx.dtype, torch.float32)
        self.assertEqual(gy.dtype, torch.float32)
        output = module([lateral, semantic])
        self.assertEqual(output.shape, lateral.shape)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all().item())
        output.float().square().mean().backward()
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_complete_model_cuda_fp32_and_amp_fp16(self):
        """Run complete nc=1 RT-DETR CUDA FP32 and AMP FP16 inference."""
        for use_amp in (False, True):
            with self.subTest(amp=use_amp):
                model = RTDETRDetectionModel(str(V4_CFG), ch=3, nc=1, verbose=False).eval().cuda()
                image = torch.zeros(1, 3, 128, 128, device="cuda")
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    output = model(image)
                self.assertEqual(output[0].shape, (1, 300, 5))
                self.assertTrue(torch.isfinite(output[0]).all().item())
                del output, image, model
                gc.collect()
                torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_complete_model_cuda_half_640_without_autocast(self):
        """Run explicit half RT-DETR at 640x640 without autocast."""
        model = RTDETRDetectionModel(str(V4_CFG), ch=3, nc=1, verbose=False).eval().cuda().half()
        image = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            output = model(image)
        self.assertEqual(output[0].shape, (1, 300, 5))
        self.assertTrue(torch.isfinite(output[0]).all().item())
        del output, image, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
