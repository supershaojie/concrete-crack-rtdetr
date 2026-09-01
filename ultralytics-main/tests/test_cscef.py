# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for cross-scale semantic-consistency edge fusion."""

import unittest
from io import BytesIO
from pathlib import Path

import torch

from ultralytics.nn.modules import CSCEF
from ultralytics.nn.tasks import RTDETRDetectionModel


class CSCEFTest(unittest.TestCase):
    """Exercise CSCEF on the locally available devices without third-party test dependencies."""

    def test_cpu_forward_backward_and_state_dict(self):
        """Check alignment, identity initialization, gradients, fixed kernels, and state restoration on CPU."""
        for semantic_size in ((16, 20), (8, 10)):
            with self.subTest(semantic_size=semantic_size):
                torch.manual_seed(0)
                module = CSCEF(64, 96)
                lateral = torch.randn(2, 64, 16, 20, requires_grad=True)
                semantic = torch.randn(2, 96, *semantic_size, requires_grad=True)

                output = module([lateral, semantic])
                self.assertEqual(output.shape, lateral.shape)
                self.assertTrue(torch.isfinite(output).all().item())
                torch.testing.assert_close(output, lateral, rtol=0, atol=0)

                output.square().mean().backward()
                self.assertIsNotNone(lateral.grad)
                self.assertIsNotNone(module.gamma.grad)
                self.assertFalse(module.scharr_x.requires_grad)
                self.assertFalse(module.scharr_y.requires_grad)
                parameter_ids = {id(parameter) for parameter in module.parameters()}
                self.assertNotIn(id(module.scharr_x), parameter_ids)
                self.assertNotIn(id(module.scharr_y), parameter_ids)

                with torch.no_grad():
                    module.gamma.fill_(0.25)
                buffer = BytesIO()
                torch.save(module.state_dict(), buffer)
                buffer.seek(0)
                restored = CSCEF(64, 96)
                restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
                restored_output = restored([lateral.detach(), semantic.detach()])
                torch.testing.assert_close(restored_output, module([lateral.detach(), semantic.detach()]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_amp(self):
        """Check CUDA autocast forward and backward stability."""
        module = CSCEF(64, 96).cuda()
        lateral = torch.randn(2, 64, 16, 20, device="cuda", requires_grad=True)
        semantic = torch.randn(2, 96, 8, 10, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = module([lateral, semantic])
            loss = output.float().square().mean()
        self.assertEqual(output.shape, lateral.shape)
        self.assertTrue(torch.isfinite(output).all().item())
        loss.backward()
        self.assertIsNotNone(lateral.grad)
        self.assertIsNotNone(module.gamma.grad)

    def test_rtdetr_model_parsing_and_cpu_forward(self):
        """Build the CSCEF YAML, verify its topology, and preserve the RT-DETR output interface on CPU."""
        cfg = (
            Path(__file__).resolve().parents[1]
            / "ultralytics"
            / "cfg"
            / "models"
            / "rt-detr"
            / "rtdetr-resnet18-lite-cscef.yaml"
        )
        model = RTDETRDetectionModel(str(cfg), ch=3, nc=80, verbose=False).eval()
        self.assertIsInstance(model.model[18], CSCEF)
        self.assertEqual(model.model[18].f, [17, 16])
        self.assertEqual(model.model[18].gamma.item(), 0.0)
        self.assertEqual(model.model[19].f, [16, 18])
        self.assertEqual(model.model[-1].f, [20, 23, 26])

        with torch.no_grad():
            output = model(torch.zeros(1, 3, 128, 128))
        self.assertIsInstance(output, tuple)
        self.assertEqual(output[0].shape, (1, 300, 84))
        self.assertTrue(torch.isfinite(output[0]).all().item())


if __name__ == "__main__":
    unittest.main()
