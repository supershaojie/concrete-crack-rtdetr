"""Focused SCCA formula, RNG, precision and gradient tests; run with unittest."""
import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from ultralytics.nn.modules import AIFI, SCCAAIFI


class TestSCCAAIFI(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(4)

    def test_rng_and_public_states(self):
        base = AIFI(256, 1024, 8)
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        new = SCCAAIFI(256, 1024, 8)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in base.state_dict().items()))
        self.assertEqual(sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in base.parameters()), 65540)
        self.assertEqual(new.scca_x_norm.eps, 1e-5)
        self.assertFalse(new.scca_s_norm.elementwise_affine)

    def test_zero_alignment_and_single_mha(self):
        for pre in (False, True):
            for shape in ((2, 256, 5, 7), (1, 256, 1, 1), (1, 256, 20, 20)):
                base = AIFI(256, 1024, 8, dropout=.1, normalize_before=pre)
                new = SCCAAIFI(256, 1024, 8, dropout=.1, normalize_before=pre)
                new.load_state_dict({**new.state_dict(), **base.state_dict()}, strict=True)
                x = torch.randn(shape)
                # Training dropout RNG must also stay unchanged by the added computation.
                state = torch.get_rng_state()
                expected = base(x)
                after = torch.get_rng_state()
                torch.set_rng_state(state)
                with patch.object(new.ma, "forward", wraps=new.ma.forward) as mha:
                    actual = new(x)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                self.assertEqual(mha.call_count, 1)
                self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_literal_formula_and_conditioning(self):
        m = SCCAAIFI(256)
        torch.nn.init.xavier_uniform_(m.scca_o.weight)
        x, s = torch.randn(2, 35, 256), torch.randn(2, 35, 256)
        d, a = m.scca_channel(x, s)
        xn, sn = F.layer_norm(x, (256,), eps=1e-5), F.layer_norm(s, (256,), eps=1e-5)
        q, k, v = [(z @ projection.weight.T).reshape(2, 35, 4, 16).permute(0, 2, 3, 1)
                   for z, projection in ((sn, m.scca_q), (xn, m.scca_k), (xn, m.scca_v))]
        q, k = q - q.mean(-1, keepdim=True), k - k.mean(-1, keepdim=True)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        expected_a = ((math.log(4) * m.scca_temperature_raw.tanh()).exp() * (q @ k.transpose(-2, -1))).softmax(-1)
        expected_d = (expected_a @ v).permute(0, 3, 1, 2).reshape(2, 35, 64) @ m.scca_o.weight.T
        torch.testing.assert_close(a, expected_a)
        torch.testing.assert_close(d, expected_d)
        self.assertEqual(tuple(a.shape), (2, 4, 16, 16))
        torch.testing.assert_close(a.sum(-1), torch.ones(2, 4, 16))
        self.assertGreater(float((a - m.scca_channel(x, s.flip(1))[1]).abs().max()), 1e-4)
        for n in (1, 9):
            d, a = m.scca_channel(torch.ones(1, n, 256), torch.ones(1, n, 256))
            self.assertTrue(torch.isfinite(d).all())
            torch.testing.assert_close(a, torch.full_like(a, 1 / 16), atol=0, rtol=0)

    def test_two_updates_and_input_gradients(self):
        m = SCCAAIFI(256, 1024)
        opt = torch.optim.SGD(m.parameters(), lr=.1)
        x, s = torch.randn(2, 35, 256, requires_grad=True), torch.randn(2, 35, 256, requires_grad=True)
        target = torch.randn_like(x)
        for step in range(2):
            opt.zero_grad()
            d, _ = m.scca_channel(x, s)
            (d * target).sum().backward()
            self.assertGreater(float(m.scca_o.weight.grad.abs().max()), 0)
            for p in (m.scca_q.weight, m.scca_k.weight, m.scca_v.weight, m.scca_temperature_raw):
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertEqual(float(p.grad.abs().max()) == 0, step == 0)
            opt.step()
        self.assertGreater(float(x.grad.abs().max()), 0)
        self.assertGreater(float(s.grad.abs().max()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_amp_and_real_half(self):
        for pre in (False, True):
            m = SCCAAIFI(256, 1024, normalize_before=pre).cuda()
            torch.nn.init.xavier_uniform_(m.scca_o.weight)
            x = torch.randn(1, 256, 5, 7, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                y = m(x)
                _, a = m.scca_channel(x.flatten(2).transpose(1, 2), torch.randn(1, 35, 256, device="cuda"))
            self.assertEqual(a.dtype, torch.float32)
            self.assertTrue(torch.isfinite(y).all())
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            with torch.no_grad():
                y = copy.deepcopy(m).half()(x.half())
            self.assertEqual(y.dtype, torch.float16)
            self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
