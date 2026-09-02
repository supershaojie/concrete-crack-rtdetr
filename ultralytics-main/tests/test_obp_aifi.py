"""Unit and integration tests for Orthogonal-Branch Perception AIFI."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ULTRALYTICS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ULTRALYTICS_ROOT.parent
BASE_CFG = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite.yaml"
OBP_CFG = (
    ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite-obp-aifi.yaml"
)
WEIGHTS = PROJECT_ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

from ultralytics.nn.modules import AdaptiveOrthogonalMixer, OBPAIFI  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402


def assert_finite_tree(test_case: unittest.TestCase, value) -> None:
    """Assert that every tensor in a nested model output contains only finite values."""
    if isinstance(value, torch.Tensor):
        test_case.assertTrue(torch.isfinite(value).all().item())
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_finite_tree(test_case, item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite_tree(test_case, item)


def tensor_tree_shapes(value):
    """Convert a nested model output into a comparable type-and-shape description."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape))
    if isinstance(value, tuple):
        return ("tuple", tuple(tensor_tree_shapes(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(tensor_tree_shapes(item) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple((key, tensor_tree_shapes(item)) for key, item in value.items()))
    return (type(value).__name__, value)


class TestAdaptiveOrthogonalMixer(unittest.TestCase):
    """Test the standalone directional mixer."""

    def test_parallel_branches_and_adaptive_weights(self):
        """All three branches should run and use normalized sample-level weights."""
        mixer = AdaptiveOrthogonalMixer(256, 1024, kernel_size=7).eval()
        called = {"horizontal": 0, "vertical": 0, "local": 0}
        handles = []
        for name in called:
            handles.append(
                getattr(mixer, name).register_forward_hook(
                    lambda _module, _inputs, _output, key=name: called.__setitem__(key, called[key] + 1)
                )
            )

        x = torch.randn(2, 256, 15, 19)
        try:
            with torch.no_grad():
                output = mixer(x)
                projected = mixer.act(mixer.input_proj(x))
                weights = mixer.branch_weights(projected)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(called, {"horizontal": 1, "vertical": 1, "local": 1})
        self.assertEqual(weights.shape, (2, 3, 1, 1, 1))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 1, 1, 1), rtol=0, atol=1e-6)

    def test_initial_gate_is_balanced(self):
        """Zero gate logits should initialize all three branch weights to one third."""
        mixer = AdaptiveOrthogonalMixer(256, 1024, kernel_size=7).eval()
        x = torch.randn(2, 256, 7, 11)
        with torch.no_grad():
            projected = mixer.act(mixer.input_proj(x))
            weights = mixer.branch_weights(projected)
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 3.0), rtol=0, atol=1e-6)


class TestOBPAIFI(unittest.TestCase):
    """Test OBP-AIFI shape, initialization, gradients, serialization, and AMP behavior."""

    def test_cpu_fp32_shape_and_finite(self):
        """The required 20x20 CPU input should preserve shape and remain finite."""
        module = OBPAIFI(256, 1024, 8, 7).eval()
        x = torch.randn(2, 256, 20, 20)
        with torch.no_grad():
            output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_dynamic_non_square_shape(self):
        """The module should infer non-square spatial dimensions dynamically."""
        module = OBPAIFI(256, 1024, 8, 7).eval()
        x = torch.randn(2, 256, 15, 19)
        with torch.no_grad():
            output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_gamma_initialization(self):
        """The learnable residual scale should start at exactly zero."""
        module = OBPAIFI(256, 1024, 8, 7)
        torch.testing.assert_close(module.gamma, torch.zeros_like(module.gamma), rtol=0, atol=0)

    def test_backward_reaches_attention_gamma_and_branches(self):
        """Attention and gamma should train immediately, and branches should train once gamma departs zero."""
        module = OBPAIFI(256, 1024, 8, 7)
        x = torch.randn(1, 256, 8, 10, requires_grad=True)
        target = torch.randn_like(x)
        (module(x) * target).mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertIsNotNone(module.gamma.grad)
        self.assertTrue(torch.isfinite(module.gamma.grad).all().item())
        self.assertGreater(module.gamma.grad.abs().sum().item(), 0)
        self.assertIsNotNone(module.ma.in_proj_weight.grad)
        self.assertTrue(torch.isfinite(module.ma.in_proj_weight.grad).all().item())
        self.assertGreater(module.ma.in_proj_weight.grad.abs().sum().item(), 0)

        module.zero_grad(set_to_none=True)
        with torch.no_grad():
            module.gamma.fill_(1e-3)
        second_input = torch.randn(1, 256, 8, 10, requires_grad=True)
        (module(second_input) * target).mean().backward()
        for branch in (module.mixer.horizontal, module.mixer.vertical, module.mixer.local):
            self.assertIsNotNone(branch.weight.grad)
            self.assertTrue(torch.isfinite(branch.weight.grad).all().item())
            self.assertGreater(branch.weight.grad.abs().sum().item(), 0)

    def test_state_dict_round_trip(self):
        """Saving and restoring a state dict should reproduce the output."""
        source = OBPAIFI(256, 1024, 8, 7).eval()
        restored = OBPAIFI(256, 1024, 8, 7).eval()
        restored.load_state_dict(source.state_dict())
        x = torch.randn(1, 256, 9, 13)
        with torch.no_grad():
            expected = source(x)
            actual = restored(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_head_divisibility_is_clear(self):
        """Invalid attention head divisibility should fail with an explicit message."""
        with self.assertRaisesRegex(ValueError, "must be divisible by num_heads"):
            OBPAIFI(250, 1024, 8, 7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_amp(self):
        """CUDA automatic mixed precision should preserve shape and finite values."""
        device = torch.device("cuda")
        module = OBPAIFI(256, 1024, 8, 7).to(device).eval()
        x = torch.randn(2, 256, 20, 20, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all().item())


class TestOBPAIFIModelIntegration(unittest.TestCase):
    """Test parser registration and full RT-DETR model behavior."""

    def test_structure_s5_path_and_cpu_forward(self):
        """The new YAML should only replace layer 9 and preserve decoder output structure."""
        baseline = RTDETRDetectionModel(str(BASE_CFG), ch=3, verbose=False).eval()
        obp = RTDETRDetectionModel(str(OBP_CFG), ch=3, verbose=False).eval()
        baseline_types = [type(layer).__name__ for layer in baseline.model]
        obp_types = [type(layer).__name__ for layer in obp.model]
        self.assertEqual(len(baseline_types), len(obp_types))
        self.assertEqual(baseline_types[:9], obp_types[:9])
        self.assertEqual(baseline_types[10:], obp_types[10:])
        self.assertEqual(baseline_types[9], "AIFI")
        self.assertEqual(obp_types[9], "OBPAIFI")
        self.assertEqual(sum(isinstance(module, OBPAIFI) for module in obp.modules()), 1)
        self.assertFalse(any("CSCEF" in type(module).__name__ for module in obp.modules()))
        self.assertEqual(obp.model[-1].f, [19, 22, 25])

        shapes = {}
        handles = [
            obp.model[index].register_forward_hook(
                lambda _module, inputs, output, layer_index=index: shapes.__setitem__(
                    layer_index, (tuple(inputs[0].shape), tuple(output.shape))
                )
            )
            for index in (7, 8, 9, 10)
        ]
        image = torch.randn(1, 3, 128, 160)
        try:
            with torch.no_grad():
                baseline_output = baseline(image)
                obp_output = obp(image)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(shapes[7][1], (1, 512, 4, 5))
        self.assertEqual(shapes[8], ((1, 512, 4, 5), (1, 256, 4, 5)))
        self.assertEqual(shapes[9], ((1, 256, 4, 5), (1, 256, 4, 5)))
        self.assertEqual(shapes[10], ((1, 256, 4, 5), (1, 256, 4, 5)))
        self.assertEqual(tensor_tree_shapes(obp_output), tensor_tree_shapes(baseline_output))
        assert_finite_tree(self, obp_output)

    def test_nc1_build_and_cpu_forward(self):
        """A one-class crack detector should build and run without changing decoder queries."""
        model = RTDETRDetectionModel(str(OBP_CFG), ch=3, nc=1, verbose=False).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 128, 128))
        self.assertEqual(model.model[-1].nc, 1)
        self.assertEqual(model.model[-1].num_queries, 300)
        self.assertEqual(output[0].shape, (1, 300, 5))
        assert_finite_tree(self, output)

    def test_existing_checkpoint_load_interface(self):
        """The ImageNet-initialized R18-Lite checkpoint should load through the existing model interface."""
        self.assertTrue(WEIGHTS.is_file(), f"Missing checkpoint: {WEIGHTS}")
        checkpoint = torch_load(str(WEIGHTS), map_location="cpu")
        model = RTDETRDetectionModel(str(OBP_CFG), ch=3, nc=80, verbose=False)
        model.load(checkpoint, verbose=False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_fp32_full_model_640(self):
        """The full OBP-AIFI detector should run a 640x640 CUDA FP32 inference."""
        device = torch.device("cuda")
        model = RTDETRDetectionModel(str(OBP_CFG), ch=3, nc=80, verbose=False).to(device).eval()
        image = torch.randn(1, 3, 640, 640, device=device)
        layer9_shape = {}
        handle = model.model[9].register_forward_hook(
            lambda _module, inputs, output: layer9_shape.update(
                {"input": tuple(inputs[0].shape), "output": tuple(output.shape)}
            )
        )
        try:
            with torch.no_grad():
                output = model(image)
        finally:
            handle.remove()
        self.assertEqual(layer9_shape, {"input": (1, 256, 20, 20), "output": (1, 256, 20, 20)})
        self.assertEqual(output[0].shape, (1, 300, 84))
        assert_finite_tree(self, output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_amp_full_model_640(self):
        """The full OBP-AIFI detector should run a 640x640 CUDA AMP inference."""
        device = torch.device("cuda")
        model = RTDETRDetectionModel(str(OBP_CFG), ch=3, nc=80, verbose=False).to(device).eval()
        image = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(image)
        self.assertEqual(output[0].shape, (1, 300, 84))
        assert_finite_tree(self, output)


if __name__ == "__main__":
    unittest.main()
