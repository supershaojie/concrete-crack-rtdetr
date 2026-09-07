"""Behavioral contracts for the complete SALA attention replacement (no training claims)."""
import unittest
from unittest.mock import patch

import torch

from ultralytics.nn.modules import SALAMSDeformAttn
from ultralytics.nn.modules.transformer import MSDeformAttn, DeformableTransformerDecoderLayer
from ultralytics.nn.modules.utils import multi_scale_deformable_attn_pytorch as kernel


class TestSALA(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.base = MSDeformAttn(256, 3, 8, 4)
        state = torch.get_rng_state().clone()
        torch.manual_seed(42)
        self.sala = SALAMSDeformAttn()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key, value in self.base.state_dict().items():
            self.assertTrue(torch.equal(value, self.sala.state_dict()[key]), key)

    def capture(self, module, query, boxes, value, shapes, mask=None):
        observed = {}
        def spy(v, s, locations, weights):
            observed.update(locations=locations.detach(), weights=weights.detach(), value=v.detach())
            return kernel(v, s, locations, weights)
        package = 'sala' if isinstance(module, SALAMSDeformAttn) else 'transformer'
        with patch(f'ultralytics.nn.modules.{package}.multi_scale_deformable_attn_pytorch', spy):
            output = module(query, boxes, value, shapes, mask)
        return output, observed

    def test_zero_and_nonzero_allocation_sampling_mask(self):
        shapes = [(6, 10), (3, 5), (2, 3)]
        for batch, queries, levels in ((1, 7, 1), (2, 13, 3)):
            query = torch.randn(batch, queries, 256)
            boxes = torch.rand(batch, queries, levels, 4)
            value = torch.randn(batch, 81, 256)
            mask = torch.zeros(batch, 81, dtype=torch.bool); mask[:, -4:] = True
            # Nonuniform base point logits detect accidental double softmax.
            with torch.no_grad():
                self.base.attention_weights.weight.normal_(std=.02)
                self.sala.attention_weights.load_state_dict(self.base.attention_weights.state_dict())
                self.sala.sala_level_head.weight.zero_(); self.sala.sala_level_head.bias.zero_()
            a, original = self.capture(self.base, query, boxes, value, shapes, mask)
            b, initial = self.capture(self.sala, query, boxes, value, shapes, mask)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(original['locations'], initial['locations']))
            self.assertTrue(torch.equal(original['weights'], initial['weights']))
            self.assertEqual(initial['value'][:, -4:].count_nonzero().item(), 0)
            with torch.no_grad(): self.sala.sala_level_head.weight.normal_(std=.15)
            c, changed = self.capture(self.sala, query, boxes, value, shapes, mask)
            self.assertFalse(torch.equal(b, c))
            self.assertTrue(torch.equal(original['locations'], changed['locations']))
            wa, wb = original['weights'], changed['weights']
            self.assertGreater((wa.sum(-1) - wb.sum(-1)).abs().max().item(), 1e-4)
            torch.testing.assert_close(wa / wa.sum(-1, keepdim=True), wb / wb.sum(-1, keepdim=True), atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(wb.sum((-1, -2)), torch.ones(batch, queries, 8))
            # Query AND box size must affect the actual bias after opening the zero head.
            delta = self.sala.level_bias(query, boxes, shapes)
            self.assertFalse(torch.equal(delta, self.sala.level_bias(query + 1, boxes, shapes)))
            smaller = boxes.clone(); smaller[..., 2:] *= .5
            self.assertFalse(torch.equal(delta, self.sala.level_bias(query, smaller, shapes)))

    def test_geometry_broadcast_clamp_detach_and_hw_order(self):
        boxes = torch.tensor([[[[.5,.5,.1,.5]]]], requires_grad=True)
        shapes = [(10, 40), (4, 10), (2, 2)]
        expected = torch.tensor([[[[4.,5.],[1.,2.],[.25,1.]]]]).log2()
        self.assertTrue(torch.equal(self.sala.geometry_descriptor(boxes, shapes), expected))
        self.assertTrue(torch.equal(self.sala.geometry_descriptor(boxes.expand(-1,-1,3,-1), shapes), expected))
        self.assertFalse(self.sala.geometry_descriptor(boxes, shapes).requires_grad)
        extreme = torch.tensor([[[[0.,0.,0.,10000.]]]])
        actual = self.sala.geometry_descriptor(extreme, shapes)
        self.assertEqual(actual.min().item(), -2.)
        self.assertEqual(actual.max().item(), 7.)
        with torch.no_grad(): self.sala.sala_level_head.weight.normal_()
        output = self.sala(torch.randn(1,1,256), boxes, torch.randn(1,444,256), shapes)
        output.square().sum().backward()
        self.assertIsNotNone(boxes.grad)
        self.assertTrue(torch.isfinite(boxes.grad).all())
        self.assertGreater(boxes.grad.abs().sum().item(), 0.)

    def test_2d_baseline_unchanged_sala_rejects(self):
        q,v,b = torch.randn(1,5,256),torch.randn(1,21,256),torch.rand(1,5,1,2)
        self.assertEqual(self.base(q,b,v,[(4,4),(2,2),(1,1)]).shape, q.shape)
        with self.assertRaisesRegex(ValueError, '4D boxes'):
            self.sala(q,b,v,[(4,4),(2,2),(1,1)])
        torch.manual_seed(123)
        first = DeformableTransformerDecoderLayer(n_levels=3)
        rng = torch.get_rng_state()
        torch.manual_seed(123)
        second = DeformableTransformerDecoderLayer(n_levels=3, sala=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(type(second.cross_attn), MSDeformAttn)
        for k,v in first.state_dict().items(): self.assertTrue(torch.equal(v, second.state_dict()[k]))

    def test_logging_preserves_output_and_gradient(self):
        q = torch.randn(1,9,256, requires_grad=True)
        boxes,value,shapes = torch.rand(1,9,1,4),torch.randn(1,21,256),[(4,4),(2,2),(1,1)]
        first = self.sala(q,boxes,value,shapes)
        first.square().sum().backward()
        gradients = {n:p.grad.clone() for n,p in self.sala.named_parameters()}
        self.sala.zero_grad(); q.grad = None
        self.sala.sala_stats_interval = 1
        second = self.sala(q,boxes,value,shapes)
        second.square().sum().backward()
        self.assertTrue(torch.equal(first, second))
        for n,p in self.sala.named_parameters(): self.assertTrue(torch.equal(gradients[n], p.grad), n)
        self.assertEqual(len(self.sala.sala_last_stats['level_mass_by_head']), 8)
        self.assertFalse(any('stats' in k for k in self.sala.state_dict()))


if __name__ == '__main__':
    unittest.main()
