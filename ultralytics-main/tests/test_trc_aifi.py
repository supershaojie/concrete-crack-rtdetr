"""TRC v1 mathematical, inheritance, mask and native dtype regression checks.

Run with the repository Python: python -m unittest discover -s ultralytics-main/tests -p test_trc_aifi.py -v
CUDA checks are explicitly skipped (PENDING, not PASS) if CUDA is unavailable.
"""
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.nn.modules import AIFI, AIFI_TRC, TokenRedundancyCalibration
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import ModelEMA


class TRCAIFITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)

    @staticmethod
    def pair(**kwargs):
        torch.manual_seed(11)
        parent = AIFI(256, 1024, 8, **kwargs)
        torch.manual_seed(11)
        candidate = AIFI_TRC(256, 1024, 8, **kwargs)
        return parent, candidate

    def test_parameters_public_state_rng_and_inherited_spatial_forward(self):
        torch.manual_seed(81)
        parent = AIFI(256, 1024, 8)
        after_parent = torch.get_rng_state().clone()
        torch.manual_seed(81)
        candidate = AIFI_TRC(256, 1024, 8)
        self.assertTrue(torch.equal(after_parent, torch.get_rng_state()))
        self.assertIs(AIFI_TRC.forward, AIFI.forward)
        self.assertIs(AIFI_TRC.build_2d_sincos_position_embedding, AIFI.build_2d_sincos_position_embedding)
        for key, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)
        self.assertEqual(set(candidate.state_dict()) - set(parent.state_dict()), {
            "trc.descriptor.weight", "trc.coefficient.weight", "trc.coefficient.bias"})
        self.assertEqual(sum(p.numel() for p in candidate.trc.parameters()), 10248)
        self.assertGreater(candidate.trc.descriptor.weight.abs().sum().item(), 0)
        self.assertEqual(candidate.trc.coefficient.weight.count_nonzero().item(), 0)
        self.assertEqual(candidate.trc.coefficient.bias.count_nonzero().item(), 0)

    def test_independent_small_reference_and_batch_head_order(self):
        module = TokenRedundancyCalibration()
        with torch.no_grad():
            module.coefficient.weight.normal_(std=0.025)
            module.coefficient.bias.copy_(torch.linspace(-0.3, 0.3, 8))
        x = torch.randn(2, 5, 256)
        x[:, 1] = x[:, 0]
        bias, got = module(x, return_diagnostics=True)
        # Explicit loops provide an independent check of axis reductions, signs,
        # diagonal inclusion and head layout (not merely another broadcast).
        u = F.layer_norm(x, (256,), eps=1e-5)
        z = F.normalize(u @ module.descriptor.weight.T, dim=-1, eps=1e-6)
        rho = torch.empty(2, 5)
        lam = torch.empty(2, 8, 5)
        expected = torch.empty(2, 8, 5, 5)
        for b in range(2):
            for j in range(5):
                rho[b, j] = sum(torch.exp(8 * ((z[b, j] * z[b, k]).sum().clamp(-1, 1) - 1)) for k in range(5))
                for h in range(8):
                    lam[b, h, j] = 0.5 * torch.tanh(
                        (u[b, j] * module.coefficient.weight[h]).sum() + module.coefficient.bias[h])
        r = (rho.clamp_min(1e-6).log() - rho.clamp_min(1e-6).log().mean(-1, keepdim=True)).clamp(-2, 2)
        for b in range(2):
            for h in range(8):
                for i in range(5):
                    for j in range(5):
                        expected[b, h, i, j] = -lam[b, h, i] * r[b, j]
        for actual, reference in ((got["rho"], rho), (got["r"], r), (got["lambda"], lam), (bias, expected)):
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=3e-6)
        flattened = bias.reshape(16, 5, 5)
        for b in range(2):
            for h in range(8):
                torch.testing.assert_close(flattened[b * 8 + h], expected[b, h], rtol=2e-5, atol=3e-6)

    def test_degenerate_content_self_term_and_single_token(self):
        module = TokenRedundancyCalibration()
        with torch.no_grad():
            module.coefficient.bias.fill_(0.2)
        for x in (torch.zeros(2, 6, 256), torch.randn(2, 1, 256).expand(-1, 6, -1), torch.randn(2, 1, 256)):
            bias, diag = module(x, True)
            self.assertTrue(torch.isfinite(bias).all())
            torch.testing.assert_close(diag["r"], torch.zeros_like(diag["r"]), atol=1e-6, rtol=0)
            torch.testing.assert_close(bias, torch.zeros_like(bias), atol=1e-6, rtol=0)
        _, diag = module(torch.zeros(2, 6, 256), True)
        torch.testing.assert_close(diag["rho"], torch.full((2, 6), 6 * torch.exp(torch.tensor(-8.)).item()))
        self.assertTrue((diag["rho"] < 1).all())
        for pre in (False, True):
            _, model = self.pair(normalize_before=pre)
            self.assertTrue(torch.isfinite(model(torch.zeros(2, 256, 2, 3))).all())

    def test_copy_multiplicity_and_signed_logit_direction(self):
        module = TokenRedundancyCalibration()
        x = torch.randn(1, 3, 256)
        _, before = module(x, True)
        repeated = x[:, [0, 0, 0, 1, 2]]
        with torch.no_grad():
            module.coefficient.bias[:4].fill_(0.7)
            module.coefficient.bias[4:].fill_(-0.7)
        bias, after = module(repeated, True)
        self.assertGreater(after["rho"][0, 0].item(), before["rho"][0, 0].item() + 1.99)
        self.assertGreater(after["r"][0, 0].item(), 0)
        self.assertLess(after["r"][0, -1].item(), 0)
        self.assertTrue((bias[:, :4, :, :3] < 0).all())
        self.assertTrue((bias[:, 4:, :, :3] > 0).all())
        self.assertTrue((after["lambda"].abs() <= 0.5).all())
        self.assertTrue((after["r"].abs() <= 2).all())

    def test_zero_equivalence_pre_post_and_rectangular_position(self):
        x = torch.randn(2, 256, 3, 5)
        for pre in (False, True):
            parent, candidate = self.pair(normalize_before=pre)
            for training in (True, False):
                parent.train(training)
                candidate.train(training)
                torch.testing.assert_close(candidate(x), parent(x), rtol=1e-5, atol=1e-6)

    def test_trc_uses_content_before_position_and_pre_norm(self):
        _, candidate = self.pair(normalize_before=True)
        with torch.no_grad():
            candidate.norm1.weight.copy_(torch.linspace(0.2, 2.0, 256))
            candidate.norm1.bias.normal_(std=0.3)
        src, pos = torch.randn(2, 6, 256), torch.randn(1, 6, 256)
        seen = []
        handle = candidate.trc.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        candidate.forward_pre(src, pos=pos)
        handle.remove()
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], src)

    def test_bool_float_2d_3d_masks_and_padding_match_parent(self):
        src, pos = torch.randn(2, 6, 256), torch.randn(1, 6, 256)
        mask = torch.zeros(6, 6, dtype=torch.bool)
        mask[:, -1] = True
        float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, -float("inf"))
        float_mask[:, 2] -= 0.7
        padding = torch.zeros(2, 6, dtype=torch.bool)
        padding[1, 0] = True
        float_padding = torch.zeros_like(padding, dtype=torch.float32).masked_fill(padding, -float("inf"))
        for pre in (False, True):
            parent, candidate = self.pair(normalize_before=pre)
            for attn in (mask, float_mask, mask.expand(16, -1, -1), float_mask.expand(16, -1, -1)):
                for pad in (None, padding, float_padding):
                    kwargs = dict(src_mask=attn, src_key_padding_mask=pad, pos=pos)
                    method = "forward_pre" if pre else "forward_post"
                    torch.testing.assert_close(getattr(candidate, method)(src, **kwargs),
                                               getattr(parent, method)(src, **kwargs), rtol=1e-5, atol=1e-6)

    def test_nonzero_mask_addition_and_invalid_shapes(self):
        parent, candidate = self.pair()
        src = torch.randn(2, 6, 256)
        with torch.no_grad():
            candidate.trc.coefficient.weight.normal_(std=0.02)
        mask = torch.randn(16, 6, 6) * 0.1
        actual, _ = candidate._attention_masks(src, mask, None, None)
        torch.testing.assert_close(actual, candidate.trc(src).reshape(16, 6, 6) + mask)
        torch.testing.assert_close(candidate.forward_post(src, src_mask=mask),
                                   parent.forward_post(src, src_mask=actual), rtol=1e-5, atol=1e-6)
        for invalid in (torch.zeros(1, 6, 6), torch.zeros(2, 8, 6, 6), torch.zeros(5, 6)):
            with self.assertRaisesRegex(ValueError, "src_mask shape"):
                candidate.forward_post(src, src_mask=invalid)
        with self.assertRaisesRegex(TypeError, "bool or floating"):
            candidate.forward_post(src, src_mask=torch.zeros(6, 6, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "src_key_padding_mask"):
            candidate.forward_post(src, src_key_padding_mask=torch.zeros(6))

    def test_fully_masked_rows_keep_original_behavior(self):
        parent, candidate = self.pair()
        src = torch.randn(2, 6, 256)
        mask = torch.zeros(6, 6, dtype=torch.bool)
        mask[1, :] = True
        expected = parent.forward_post(src, src_mask=mask)
        actual = candidate.forward_post(src, src_mask=mask)
        self.assertTrue(torch.isnan(expected[:, 1]).all())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)

    def test_first_coefficient_gradient_and_nonzero_descriptor_gradient(self):
        _, candidate = self.pair()
        src = torch.randn(2, 256, 3, 4, requires_grad=True)
        with torch.no_grad():
            src[:, :, 0, 1:3] = src[:, :, 0, 0:1]
        projection = torch.randn_like(src)
        (candidate(src) * projection).sum().backward()
        for parameter in (candidate.trc.coefficient.weight, candidate.trc.coefficient.bias):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(candidate.trc.descriptor.weight.grad.count_nonzero().item(), 0)
        diagnostic = deepcopy(candidate)
        diagnostic.zero_grad(set_to_none=True)
        with torch.no_grad():
            diagnostic.trc.coefficient.weight.normal_(std=0.01)
        (diagnostic(src.detach()) * projection).sum().backward()
        grad = diagnostic.trc.descriptor.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_zero_nonzero_reload_and_ema_preserve_learned_state(self):
        _, candidate = self.pair()
        for nonzero in (False, True):
            if nonzero:
                with torch.no_grad():
                    candidate.trc.coefficient.weight.normal_(std=0.01)
                    candidate.trc.coefficient.bias.fill_(0.2)
            buffer = io.BytesIO()
            torch.save(candidate.state_dict(), buffer)
            buffer.seek(0)
            _, restored = self.pair()
            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            for key, value in candidate.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            ema = ModelEMA(candidate)
            ema.update(candidate)
            for key, value in candidate.trc.state_dict().items():
                torch.testing.assert_close(ema.ema.trc.state_dict()[key], value, rtol=1e-7, atol=1e-7)

    def test_yaml_topology_and_all_public_initial_states(self):
        folder = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/rt-detr"
        trc_states = []
        for stem in ("rtdetr-resnet18-lite", "rtdetr-resnet18-lite-cbr-lif"):
            parent_file = stem + ("-down.yaml" if stem.endswith("lif") else ".yaml")
            torch.manual_seed(42)
            parent = RTDETRDetectionModel(str(folder / parent_file), nc=1, verbose=False)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(42)
            candidate = RTDETRDetectionModel(str(folder / (stem + "-trc-v1.yaml")), nc=1, verbose=False)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in parent.state_dict().items():
                self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)
            self.assertEqual(set(candidate.state_dict()) - set(parent.state_dict()), {
                "model.9.trc.descriptor.weight", "model.9.trc.coefficient.weight", "model.9.trc.coefficient.bias"})
            self.assertEqual(sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in parent.parameters()), 10248)
            self.assertIsInstance(candidate.model[9], AIFI_TRC)
            self.assertEqual(candidate.model[8].conv.out_channels, 256)
            self.assertEqual(candidate.model[19].__class__.__name__, "RepC3")
            self.assertEqual(candidate.model[26].f, [19, 22, 25])
            self.assertEqual(candidate.model[20].__class__.__name__, "LIFDown" if stem.endswith("lif") else "Conv")
            self.assertEqual(candidate.model[26].__class__.__name__, "RTDETRDecoderCBR" if stem.endswith("lif") else "RTDETRDecoder")
            if not stem.endswith("lif"):
                self.assertFalse(any("cbr" in key.lower() or "O_proj" in key for key in candidate.state_dict()))
            trc_states.append(deepcopy(candidate.model[9].trc.state_dict()))
        for key in trc_states[0]:
            self.assertTrue(torch.equal(trc_states[0][key], trc_states[1][key]), key)

    @unittest.skipUnless(torch.cuda.is_available(), "PENDING: CUDA FP32 requires CUDA")
    def test_cuda_fp32_zero_equivalence_and_backward(self):
        old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            parent, candidate = (m.cuda() for m in self.pair())
            src = torch.randn(2, 256, 20, 20, device="cuda")
            torch.testing.assert_close(candidate(src), parent(src), rtol=1e-5, atol=2e-6)
            (candidate(src) * torch.randn_like(src)).mean().backward()
            self.assertTrue(torch.isfinite(candidate.trc.coefficient.weight.grad).all())
            self.assertGreater(candidate.trc.coefficient.weight.grad.abs().sum().item(), 0)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

    @unittest.skipUnless(torch.cuda.is_available(), "PENDING: CUDA AMP/half requires CUDA")
    def test_cuda_native_amp_half_and_mask_dtype(self):
        for half_model in (False, True):
            for pre in (False, True):
                parent, candidate = self.pair(normalize_before=pre)
                parent = parent.cuda()
                candidate = candidate.cuda()
                if half_model:
                    parent.half()
                    candidate.half()
                spatial = torch.randn(2, 256, 3, 4, device="cuda", dtype=torch.float16 if half_model else torch.float32)
                with torch.autocast("cuda", dtype=torch.float16, enabled=not half_model):
                    torch.testing.assert_close(candidate(spatial), parent(spatial), rtol=2e-3, atol=2e-3)
                with torch.no_grad():
                    candidate.trc.coefficient.weight.normal_(std=0.01)
                src = torch.randn(2, 12, 256, device="cuda", dtype=torch.float16 if half_model else torch.float32)
                pos = torch.randn(1, 12, 256, device="cuda", dtype=src.dtype)
                mask = torch.zeros(12, 12, device="cuda", dtype=torch.bool)
                mask[:, -1] = True
                padding = torch.zeros(2, 12, device="cuda", dtype=torch.bool)
                padding[1, 0] = True
                before = {key: (id(value), value.dtype) for key, value in candidate.named_parameters()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=not half_model):
                    bias, diag = candidate.trc(src, True)
                    self.assertEqual(bias.dtype, torch.float32)
                    self.assertTrue(all(t.dtype == torch.float32 for t in diag.values()))
                    merged, pad = candidate._attention_masks(src, mask, padding, pos)
                    self.assertEqual(merged.dtype, torch.float16)
                    self.assertEqual(pad.dtype, torch.float16)
                    method = candidate.forward_pre if pre else candidate.forward_post
                    out = method(src, src_mask=mask, src_key_padding_mask=padding, pos=pos)
                    self.assertTrue(torch.isfinite(out).all())
                    loss = (out.float() * torch.randn_like(out).float()).mean()
                loss.backward()
                for parameter in candidate.trc.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                self.assertEqual(before, {key: (id(value), value.dtype) for key, value in candidate.named_parameters()})

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "PENDING: CUDA BF16 requires supported CUDA hardware")
    def test_autocast_bfloat16_uses_actual_projected_dtype(self):
        _, candidate = self.pair()
        candidate.cuda()
        src = torch.randn(2, 6, 256, device="cuda")
        mask = torch.zeros(6, 6, device="cuda", dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            merged, _ = candidate._attention_masks(src, mask, None, None)
            self.assertEqual(merged.dtype, torch.bfloat16)
            self.assertEqual(candidate.trc(src).dtype, torch.float32)
            self.assertTrue(torch.isfinite(candidate.forward_post(src, src_mask=mask)).all())


if __name__ == "__main__":
    unittest.main()
