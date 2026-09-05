"""Behavioral tests for C18 local coordinates, content, axes, normalization and AIFI integration."""
from copy import deepcopy
import math
import unittest

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import AIFI, GSDRAIFIV3, QueryLocalDeformableRelationV3


SHAPES = ((20, 20), (17, 23), (1, 7), (7, 1), (1, 1))


def bilinear_oracle(value, pixels, weights, heads):
    """Independent pixel-coordinate interpolation; intentionally no grid_sample or production helpers."""
    batch, channels, height, width = value.shape
    result = torch.zeros_like(value)
    for b in range(batch):
        for q in range(height * width):
            for head in range(heads):
                section = slice(head * (channels // heads), (head + 1) * (channels // heads))
                for point in range(4):
                    x, y = pixels[b, q, head, point].tolist()
                    x0, y0 = math.floor(x), math.floor(y)
                    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
                    dx, dy = x - x0, y - y0
                    v = ((1-dx)*(1-dy)*value[b, section, y0, x0] + dx*(1-dy)*value[b, section, y0, x1]
                         + (1-dx)*dy*value[b, section, y1, x0] + dx*dy*value[b, section, y1, x1])
                    result[b, section, q // width, q % width] += v * weights[b, q, head, point]
    return result


class GSDRV3BehaviorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_fixed_budget_bias_free_initialization(self):
        m = QueryLocalDeformableRelationV3(256)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 88064)
        self.assertEqual(len(list(m.parameters())), 5)
        self.assertFalse(any("bias" in n for n, _ in m.named_parameters()))
        self.assertEqual(list(m.named_buffers())[0][0], "anchors")
        self.assertFalse(m.token_norm.elementwise_affine)
        self.assertEqual(m.token_norm.eps, 1e-5)
        self.assertEqual(m.token_norm.normalized_shape, (128,))
        for name in ("offset_proj", "weight_proj", "output_proj"):
            self.assertEqual(torch.count_nonzero(getattr(m, name).weight), 0)
        for name in ("input_proj", "value_proj"):
            self.assertGreater(torch.count_nonzero(getattr(m, name).weight), 0)
        with torch.no_grad():
            x = torch.randn(2, 256, 5, 7)
            self.assertEqual(torch.count_nonzero(m(x)), 0)
            _, state = m.relation_features(x)
        self.assertTrue(torch.equal(state["weights"], torch.full_like(state["weights"], .25)))
        self.assertEqual(torch.unique(state["pixels"][0, 17, 0], dim=0).shape[0], 4)

    def test_coordinates_extremes_locality_xy_and_constant_edges(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            m = QueryLocalDeformableRelationV3(8, 8, 2).to(device)
            for height, width in SHAPES:
                for raw_value in (0., 1e6, -1e6):
                    with self.subTest(device=device, shape=(height, width), raw=raw_value):
                        raw = torch.full((2, height * width, 2, 4, 2), raw_value, device=device)
                        q, pixels, grid = m.sampling_geometry(raw, height, width)
                        self.assertEqual(tuple(grid.shape), (2, height * width, 2, 4, 2))
                        self.assertEqual(grid.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(grid).all())
                        self.assertLessEqual((pixels - q[None, :, None, None]).abs().max(), 2.00001)
                        self.assertGreaterEqual(pixels.min(), -1e-6)
                        self.assertLessEqual(pixels[..., 0].max(), width - 1 + 1e-6)
                        self.assertLessEqual(pixels[..., 1].max(), height - 1 + 1e-6)
                        for axis, size in enumerate((width, height)):
                            # CUDA tensor division and scalar reciprocal multiplication can differ by one FP32 ULP.
                            torch.testing.assert_close(grid[..., axis], 2*(pixels[..., axis]+.5)/size-1, rtol=0, atol=2e-7)
                            if size == 1:
                                self.assertEqual(torch.count_nonzero(grid[..., axis]), 0)
                        # Independent scalar coordinate formula, including non-square xy indexing.
                        for qi in (0, height * width // 2, height * width - 1):
                            self.assertEqual(q[qi].tolist(), [qi % width, qi // width])
                            for point, anchor in enumerate(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5))):
                                expected = []
                                for axis, (pos, size) in enumerate(zip((qi % width, qi // width), (width, height))):
                                    lo, hi = max(0,pos-2), min(size-1,pos+2)
                                    expected.append((lo+hi)/2 + (hi-lo)/2 * math.tanh(raw_value+math.atanh(anchor[axis])))
                                torch.testing.assert_close(pixels[0, qi, 0, point].cpu(), torch.tensor(expected), rtol=0, atol=2e-6)
                        value = torch.ones(2, 8, height, width, device=device)
                        weights = torch.full((2, height*width, 2, 4), .25, device=device)
                        actual = m.aggregate(value, grid, weights)
                        torch.testing.assert_close(actual, value, rtol=0, atol=3e-6)

    def test_nonconstant_values_do_not_broadcast_or_mix_batch_heads(self):
        m = QueryLocalDeformableRelationV3(8, 8, 2)
        height, width = 5, 7
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        value = torch.stack([torch.stack([(b+1)*1000 + c*100 + xx.float() + (c+1)*yy.float()
                                         for c in range(8)]) for b in range(2)])
        raw = torch.randn(2, height*width, 2, 4, 2) * .4
        _, pixels, grid = m.sampling_geometry(raw, height, width)
        weights = torch.softmax(torch.randn(2, height*width, 2, 4), -1)
        actual = m.aggregate(value, grid, weights)
        expected = bilinear_oracle(value, pixels, weights, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=.001)
        self.assertFalse(torch.allclose(actual[:, :, 1, 1], actual[:, :, 3, 5]))
        self.assertFalse(torch.allclose(actual[0], actual[1]))
        changed = value.clone()
        changed[1, 4:] += 17
        difference = m.aggregate(changed, grid, weights) - actual
        torch.testing.assert_close(difference[0], torch.zeros_like(difference[0]), rtol=0, atol=0)
        torch.testing.assert_close(difference[1, :4], torch.zeros_like(difference[1, :4]), rtol=0, atol=0)
        torch.testing.assert_close(difference[1, 4:], torch.full_like(difference[1, 4:], 17), rtol=0, atol=.001)

    def test_end_to_end_known_content_nonzero_output_and_token_norm(self):
        m = QueryLocalDeformableRelationV3(4, 4, 2)
        with torch.no_grad():
            for name in ("input_proj", "value_proj", "output_proj"):
                getattr(m, name).weight.copy_(torch.eye(4).reshape(4, 4, 1, 1))
        yy, xx = torch.meshgrid(torch.arange(5), torch.arange(7), indexing="ij")
        x = torch.stack([torch.stack((xx.float(), yy.float(), xx*yy*.2+b, torch.full((5, 7), float(2+b))))
                         for b in (0, 1)])
        z = (x-x.mean(1, keepdim=True)) / (x.var(1, unbiased=False, keepdim=True)+1e-5).sqrt()
        pixels = torch.zeros(2, 35, 2, 4, 2)
        for q in range(35):
            for point, anchor in enumerate(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5))):
                for axis, (pos, size) in enumerate(((q % 7, 7), (q // 7, 5))):
                    lo, hi = max(0,pos-2), min(size-1,pos+2)
                    pixels[:, q, :, point, axis] = (lo+hi)/2 + (hi-lo)/2*anchor[axis]
        expected = bilinear_oracle(z, pixels, torch.full((2, 35, 2, 4), .25), 2)
        torch.testing.assert_close(m(x), expected, rtol=2e-5, atol=2e-6)
        self.assertGreater(m(x).std((-2, -1)).mean(), .1)

    def test_offset_weight_value_projections_affect_actual_output(self):
        m = QueryLocalDeformableRelationV3(16, 16, 4)
        with torch.no_grad():
            m.output_proj.weight.normal_(std=.1)
        x = torch.randn(2, 16, 7, 9)
        expected = m(x)
        for name in ("offset_proj", "weight_proj", "value_proj"):
            changed = deepcopy(m)
            with torch.no_grad():
                getattr(changed, name).weight.normal_(std=.3)
            actual = changed(x)
            self.assertGreater((actual-expected).abs().max(), 1e-3, name)
            _, state = changed.relation_features(x)
            torch.testing.assert_close(state["weights"].sum(-1), torch.ones(2, 63, 4), rtol=0, atol=2e-7)
            self.assertEqual(tuple(state["weights"].shape), (2, 63, 4, 4))
            if name == "offset_proj":
                self.assertFalse(torch.equal(state["grid"][0], state["grid"][1]))
                self.assertFalse(torch.equal(state["grid"][:, :, 0], state["grid"][:, :, 1]))

    def test_aifi_pre_post_norm_zero_equivalence_and_dropout_rng(self):
        for pre in (False, True):
            for train in (False, True):
                torch.manual_seed(5)
                baseline = AIFI(32, 64, 4, dropout=.2, normalize_before=pre).train(train)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(5)
                target = GSDRAIFIV3(32, 64, 4, aux_dim=32, dropout=.2, normalize_before=pre).train(train)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key, value in baseline.state_dict().items():
                    self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
                x = torch.randn(2, 32, 5, 7)
                rng = torch.get_rng_state().clone()
                expected = baseline(x)
                after = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                self.assertTrue(torch.equal(expected, target(x)), (pre, train))
                self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_open_branch_integration_matches_original_residual_order(self):
        for pre in (False, True):
            m = GSDRAIFIV3(32, 64, 4, aux_dim=32, normalize_before=pre).eval()
            with torch.no_grad():
                m.sparse_relation.output_proj.weight.normal_(std=.05)
            x = torch.randn(2, 32, 5, 7)
            tokens = x.flatten(2).transpose(1, 2)
            normalized = m.norm1(tokens) if pre else tokens
            pos = m.build_2d_sincos_position_embedding(7, 5, 32)
            q = normalized + pos
            dense = m.ma(q, q, normalized)[0]
            delta = m.sparse_relation(normalized.transpose(1, 2).reshape_as(x)).flatten(2).transpose(1, 2)
            residual = (tokens + dense) + delta
            if pre:
                expected = residual + m.fc2(m.act(m.fc1(m.norm2(residual))))
            else:
                residual = m.norm1(residual)
                expected = m.norm2(residual + m.fc2(m.act(m.fc1(residual))))
            self.assertTrue(torch.equal(m(x), expected.transpose(1, 2).reshape_as(x)))

    def test_half_anchors_geometry_remains_fp32_and_exact(self):
        m = QueryLocalDeformableRelationV3(8, 8, 2)
        raw = torch.randn(2, 17*23, 2, 4, 2)
        before = m.sampling_geometry(raw, 17, 23)
        m.half()
        after = m.sampling_geometry(raw, 17, 23)
        for a, b in zip(before, after):
            self.assertEqual(b.dtype, torch.float32)
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
