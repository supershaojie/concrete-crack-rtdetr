"""Risk-focused CBR checks; run with unittest (no additional test dependency)."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.nn.modules import CrackBoundaryRefinement, RTDETRDecoder, RTDETRDecoderCBR


class CBRTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)

    def test_coordinates_signed_evidence_and_bounds(self):
        module = CrackBoundaryRefinement(2, 8)
        # Rectangular 40x80 map with channels equal to normalized pixel-center x,y.
        y, x = torch.meshgrid((torch.arange(40) + .5) / 40, (torch.arange(80) + .5) / 80, indexing="ij")
        feat = torch.stack((x, y))[None]
        boxes = torch.tensor([[[.5, .5, .8, .1], [.5, .5, .1, .8], [.01, .99, .08, .06]]])
        grid = module.sampling_grid(boxes)
        self.assertEqual(grid.shape, (1, 3, 4, 3, 3, 2))
        sampled = F.grid_sample(feat, grid.reshape(1, 3, 36, 2), padding_mode="border", align_corners=False)
        got = sampled.permute(0, 2, 3, 1).reshape_as(grid)
        expected = (grid + 1) / 2
        expected[..., 0].clamp_(.5 / 80, 1 - .5 / 80)
        expected[..., 1].clamp_(.5 / 40, 1 - .5 / 40)
        torch.testing.assert_close(got, expected, rtol=0, atol=1.2e-7)  # <2 FP32 ULP at unit coordinates
        diff = got[:, :2, :, :, 0] - got[:, :2, :, :, 2]
        self.assertTrue((diff[:, :, 0, :, 0] > 0).all())
        self.assertTrue((diff[:, :, 1, :, 0] < 0).all())
        self.assertTrue((diff[:, :, 2, :, 1] > 0).all())
        self.assertTrue((diff[:, :, 3, :, 1] < 0).all())
        query = torch.randn(1, 3, 8)
        self.assertTrue(torch.equal(module(feat, query, boxes), boxes))
        with torch.no_grad():
            module.offset_out.weight.normal_(std=2.)
        out, d = module(feat, query, boxes, True)
        self.assertTrue((out[..., 2:] >= .8 * boxes[..., 2:] - 1e-7).all())
        self.assertTrue((d["displacement"].abs() <= .1 * boxes[..., [2, 2, 3, 3]]).all())
        self.assertFalse(torch.equal(out, boxes))

    def test_query_feature_sensitivity_and_gradients_after_update(self):
        module = CrackBoundaryRefinement(16, 32)
        p3 = torch.randn(2, 16, 9, 13, requires_grad=True)
        query = torch.randn(2, 11, 32, requires_grad=True)
        boxes = torch.rand(2, 11, 4, requires_grad=True)
        opt = torch.optim.AdamW(module.parameters(), lr=.0005)
        probe = torch.randn_like(boxes)
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            out = module(p3, query, boxes)
            (out * probe).sum().backward()
            for name, p in module.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                if step == 0 and not name.startswith("offset_out"):
                    self.assertEqual(p.grad.abs().sum().item(), 0, name)
                if step > 0:
                    self.assertGreater(p.grad.abs().sum().item(), 0, name)
            opt.step()
        for value in (p3, query, boxes):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        out, details = module(p3, query, boxes, True)
        out_q, details_q = module(p3, query + torch.randn_like(query) * 3, boxes, True)
        self.assertFalse(torch.equal(out, out_q))
        self.assertGreater((details["aggregation_weights"] - details_q["aggregation_weights"]).abs().max().item(), 1e-5)
        self.assertFalse(torch.equal(out, module(p3.flip(-1), query, boxes)))

    def test_head_zero_equivalence_dn_gradient_and_nonzero_return(self):
        opts = dict(nc=1, ch=(16, 16, 16), hd=32, nq=7, ndp=4, nh=4, ndl=3, d_ffn=64)
        torch.manual_seed(8)
        base = RTDETRDecoder(**opts)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(8)
        target = RTDETRDecoderCBR(**opts)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in base.state_dict().items():
            self.assertTrue(torch.equal(v, target.state_dict()[k]), k)
        features = [torch.randn(2, 16, h, w) for h, w in ((12, 16), (6, 8), (3, 4))]
        for groups in ([0, 0], [1, 3]):
            n = sum(groups)
            batch = {"cls": torch.zeros(n, dtype=torch.long), "bboxes": torch.rand(n, 4),
                     "batch_idx": torch.tensor([0] * groups[0] + [1] * groups[1], dtype=torch.long), "gt_groups": groups}
            base.train(); target.train()
            torch.manual_seed(12); a = base(features, batch)
            torch.manual_seed(12); b = target(features, batch)
            for aa, bb in zip(a[:4], b[:4]):
                self.assertTrue(torch.equal(aa, bb))
            if n:
                self.assertEqual(a[4]["dn_num_split"], b[4]["dn_num_split"])
                self.assertGreater(b[0].shape[2], 7)
            # Preserve last_refined_bbox vs detached reference's original training gradient path.
            base.zero_grad(); target.zero_grad()
            a[0].sum().backward(); b[0].sum().backward()
            for k, p in base.named_parameters():
                q = dict(target.named_parameters())[k]
                if p.grad is None:
                    self.assertIsNone(q.grad, k)
                else:
                    self.assertTrue(torch.equal(p.grad, q.grad), k)
        base.eval(); target.eval()
        with torch.no_grad():
            a = base(features)
            target.cbr.offset_out.bias.fill_(.5)
            b, d = target.forward_with_diagnostics(features)
        self.assertTrue(torch.equal(a[0][..., 4:], b[0][..., 4:]))
        self.assertFalse(torch.equal(a[0][..., :4], b[0][..., :4]))
        self.assertTrue(torch.equal(b[0][..., :4], d["after"]))
        # Full-object FP32 round-trip retains type and actually changed behavior.
        buffer = io.BytesIO(); torch.save(target, buffer); buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        self.assertIs(type(restored), RTDETRDecoderCBR)
        with torch.no_grad():
            self.assertTrue(torch.equal(restored(features)[0], b[0]))


if __name__ == "__main__":
    unittest.main()
