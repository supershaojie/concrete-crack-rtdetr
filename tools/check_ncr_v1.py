"""Bounded NCR unit/integration checks, comparing against the exact parent Git source."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
from ultralytics.models.utils.ncr import NCRConfig, NCRDetectionLoss, configure_ncr, ncr_terms
from ultralytics.nn.modules.cbr import CrackBoundaryRefinement
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.metrics import bbox_iou

PARENT = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"


def parent_loss():
    name = "ultralytics.models.utils._ncr_reference"
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    source = subprocess.check_output(["git", "show", PARENT + ":ultralytics-main/ultralytics/models/utils/loss.py"], cwd=ROOT)
    exec(compile(source, "parent_loss.py", "exec"), module.__dict__)
    return module.RTDETRDetectionLoss


def fixture(dn=True):
    torch.manual_seed(91)
    boxes = torch.tensor([.43, .50, .30, .24]).repeat(4, 2, 5, 1).requires_grad_()
    scores = torch.randn(4, 2, 5, 1, requires_grad=True)
    batch = dict(bboxes=torch.tensor([[.5, .5, .8, .7], [.5, .5, .8, .7]]),
                 cls=torch.zeros(2, dtype=torch.long), gt_groups=[1, 1])
    db = torch.tensor([.43, .5, .3, .24]).repeat(3, 2, 2, 1).requires_grad_()
    ds = torch.randn(3, 2, 2, 1, requires_grad=True)
    meta = dict(dn_pos_idx=[torch.tensor([0]), torch.tensor([0])], dn_num_group=1, dn_num_split=[2, 5])
    return (boxes, scores), batch, dict(dn_bboxes=db, dn_scores=ds, dn_meta=meta) if dn else {}


class MathChecks(unittest.TestCase):
    def test_interval_plateau(self):
        # Units normalized by 10; y is identical. GIoU center gradients vanish, L1 does not.
        b = torch.tensor([[.4, .5, .6, 1.], [.5, .5, .6, 1.], [.6, .5, .6, 1.]], requires_grad=True)
        g = torch.tensor([[.5, .5, 1., 1.]]).repeat(3, 1)
        raw, h, u, s = ncr_terms(b, g, (640, 640))
        grad = torch.autograd.grad(raw, b)[0]
        self.assertLess(float(grad[0, 0]), 0)
        self.assertEqual(float(grad[1, 0]), 0)
        self.assertGreater(float(grad[2, 0]), 0)
        torch.testing.assert_close(grad[:, 2:], torch.zeros_like(grad[:, 2:]), rtol=0, atol=0)
        giou = torch.autograd.grad((1 - bbox_iou(b, g, xywh=True, GIoU=True)).sum(), b)[0]
        torch.testing.assert_close(giou[:, :2], torch.zeros_like(giou[:, :2]), atol=1e-6, rtol=0)
        l1 = torch.autograd.grad((b - g).abs().sum(), b)[0]
        self.assertNotEqual(float(l1[0, 0]), 0)

    def test_geometry_and_all_matches_denominator(self):
        b = torch.tensor([[.45, .5, .9, .8], [.45, .65, .2, .4], [.15, .15, .2, .2],
                          [.3, .5, .4, .8], [.5, .5, .8, .8], [.501, .501, .1, .1]], requires_grad=True)
        g = torch.tensor([[.5, .5, .4, .4], [.5, .5, .8, .4], [.5, .5, .4, .4],
                          [.5, .5, .8, .8], [.5, .5, .8, .8], [.5, .5, 1e-9, 1e-9]], requires_grad=True)
        raw, h, u, scale = ncr_terms(b, g, (320, 640))
        self.assertTrue(bool((h[0] > 0).all()))  # prediction contains GT
        self.assertTrue(h[1, 0] > 0 and h[1, 1] == 0)
        torch.testing.assert_close(h[2:5], torch.zeros_like(h[2:5]), atol=1e-6, rtol=0)
        torch.testing.assert_close(scale[-1], torch.tensor([4/640, 4/320]))
        expected = h * u.clamp(-1, 1) / (2 * len(b) * scale)
        grads = torch.autograd.grad(raw, (b, g), allow_unused=True)
        torch.testing.assert_close(grads[0][:, :2], expected)
        torch.testing.assert_close(grads[0][:, 2:], torch.zeros_like(b[:, 2:]))
        self.assertIsNone(grads[1])
        single = ncr_terms(b[:1], g[:1], (320, 640))[0]
        rest = ncr_terms(b[1:], g[1:], (320, 640))[0]
        torch.testing.assert_close(raw, (single + rest * 5) / 6)

    def test_frozen_gate_finite_difference(self):
        b = torch.tensor([[.43, .54, .2, .3]], requires_grad=True)
        g = torch.tensor([[.5, .5, .8, .8]])
        raw, gate, _, scale = ncr_terms(b, g, (640, 640))
        grad = torch.autograd.grad(raw, b)[0]
        def frozen(value):
            u = (value[:, :2] - g[:, :2]) / scale
            return (gate * torch.where(u.abs() <= 1, .5*u.square(), u.abs()-.5)).sum()/2
        for a in range(4):
            plus, minus = b.detach().clone(), b.detach().clone()
            plus[0, a] += 1e-3; minus[0, a] -= 1e-3
            torch.testing.assert_close((frozen(plus)-frozen(minus))/2e-3, grad[0, a], atol=1e-6, rtol=1e-4)

    def test_huber_linear_branch(self):
        b = torch.tensor([[.6, .4, .9, .9]], requires_grad=True)
        g = torch.tensor([[.5, .5, .02, .02]])
        raw, h, u, s = ncr_terms(b, g, (640, 640))
        self.assertTrue(bool((u.abs() > 1).all()))
        torch.testing.assert_close(torch.autograd.grad(raw, b)[0][:, :2], h*u.sign()/(2*s))

    def test_epoch_ramp(self):
        cfg = NCRConfig()
        for epoch, expected in ((0, 0), (4, 0), (5, .25/15), (19, .25), (20, .25), (87, .25)):
            self.assertAlmostEqual(cfg.coefficient(epoch), expected)
        with self.assertRaises(ValueError): cfg.coefficient(None)

    def test_amp_fp32_terms(self):
        b = torch.tensor([[.43, .5, .2, .3]], dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            raw = ncr_terms(b, torch.tensor([[.5, .5, .8, .8]]), (640, 640))[0]
        self.assertEqual(raw.dtype, torch.float32)
        raw.backward()
        self.assertTrue(bool(torch.isfinite(b.grad).all()))


class WiringChecks(unittest.TestCase):
    def test_zero_matches_exact_parent_loss_gradient_rng(self):
        for dn in (False, True):
            for epoch, coefficient in ((20, 0.), (0, .25), (4, .25)):
                preds, batch, kwargs = fixture(dn)
                inputs = (*preds, *[kwargs[k] for k in ("dn_bboxes", "dn_scores")]) if dn else preds
                parent = parent_loss()(nc=1, use_vfl=True)
                new = NCRDetectionLoss(nc=1, use_vfl=True, config=dict(lambda_ncr=coefficient))
                before = torch.get_rng_state().clone()
                a = parent(preds, batch, **kwargs)
                with patch("ultralytics.models.utils.ncr.ncr_terms", side_effect=AssertionError("zero path built NCR")):
                    b = new(preds, batch, **kwargs, input_hw=(640, 640), epoch=epoch)
                self.assertEqual(set(b), set(a) | {"loss_ncr"})
                for k in a: torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
                ga = torch.autograd.grad(sum(a.values()), inputs, retain_graph=True)
                gb = torch.autograd.grad(sum(b.values()), inputs)
                for x, y in zip(ga, gb): torch.testing.assert_close(x, y, rtol=0, atol=0)
                torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
                self.assertEqual(float(b["loss_ncr"]), 0)

    def test_final_regular_scope_dn_and_match_count(self):
        for dn in (False, True):
            preds, batch, kwargs = fixture(dn)
            old = parent_loss()(nc=1, use_vfl=True)
            new = NCRDetectionLoss(nc=1, use_vfl=True)
            a = old(preds, batch, **kwargs)
            with patch.object(new.matcher, "forward", wraps=new.matcher.forward) as matcher:
                b = new(preds, batch, **kwargs, input_hw=(640, 640), epoch=19, collect_diagnostics=True)
                self.assertEqual(matcher.call_count, 4)  # encoder + 3 decoders, DN has its native pairs
            self.assertGreater(float(b["loss_ncr"]), 0)
            self.assertFalse(any("ncr_dn" in k or "ncr_aux" in k for k in b))
            for k in a: torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
            inputs = (*preds, kwargs["dn_bboxes"], kwargs["dn_scores"]) if dn else preds
            ga = torch.autograd.grad(sum(a.values()), inputs, retain_graph=True)
            gb = torch.autograd.grad(sum(b.values()), inputs, retain_graph=True)
            torch.testing.assert_close(ga[0][:-1], gb[0][:-1], atol=0, rtol=0)
            torch.testing.assert_close(ga[0][-1, ..., 2:], gb[0][-1, ..., 2:], atol=0, rtol=0)
            for x, y in zip(ga[1:], gb[1:]): torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(new.last_diagnostics["matches"], 2)
            json.dumps(new.last_diagnostics, allow_nan=False)
            torch.testing.assert_close(sum(b.values())-sum(a.values()), b["loss_ncr"], atol=2e-6, rtol=1e-4)

    def test_empty_gt(self):
        preds, batch, _ = fixture(False)
        batch.update(bboxes=torch.empty(0, 4), cls=torch.empty(0, dtype=torch.long), gt_groups=[0, 0])
        loss = NCRDetectionLoss(nc=1, use_vfl=True)(preds, batch, input_hw=(640, 640), epoch=19)
        self.assertEqual(float(loss["loss_ncr"]), 0)
        self.assertEqual(loss["loss_ncr"].device, preds[0].device)
        self.assertFalse(loss["loss_ncr"].requires_grad)
        self.assertTrue(torch.isfinite(sum(loss.values())))

    def test_model_total_and_original_cbr_gradient_path(self):
        # Actual model.loss assembly with an activated synthetic CBR output; independent leaves for other branches.
        model = RTDETRDetectionModel(str(MODEL), nc=1, verbose=False).train()
        model.nc = 1  # Native DetectionTrainer.set_model_attributes does this during training.
        p3, query = torch.randn(1, 16, 8, 8), torch.randn(1, 5, 16)
        cbr = CrackBoundaryRefinement(16, 16)
        boxes = torch.tensor([.43, .50, .30, .24]).repeat(1, 5, 1).requires_grad_()
        refined = cbr(p3, query, boxes)
        dec = torch.stack([boxes, boxes, refined])
        scores = torch.zeros(3, 1, 5, 1, requires_grad=True)
        preds = (dec, scores, boxes.clone(), scores[0].clone(), None)
        batch = dict(img=torch.zeros(1, 3, 320, 640), batch_idx=torch.zeros(1),
                     bboxes=torch.tensor([[.5, .5, .8, .7]]), cls=torch.zeros(1, 1))
        total0, items0 = model.loss(batch, preds)
        configure_ncr(model); model.ncr_epoch = 19; model.ncr_collect_diagnostics = True
        total1, items1 = model.loss(batch, preds)
        self.assertEqual(items0.shape, (3,)); self.assertEqual(items1.shape, (4,))
        torch.testing.assert_close(total1-total0, items1[-1], atol=2e-6, rtol=1e-4)
        extra = total1-total0
        box_grad, cbr_grad = torch.autograd.grad(extra, (boxes, cbr.offset_out.weight))
        self.assertGreater(float(box_grad[:, :, :2].abs().sum()), 0)
        self.assertGreater(float(cbr_grad.abs().sum()), 0)

    def test_topology_inference_and_fresh_process_reload(self):
        from init_c19_lif_v1 import verify_model
        torch.manual_seed(42)
        model = RTDETRDetectionModel(str(MODEL), nc=1, verbose=False).eval()
        model.nc = 1
        verify_model(model)
        inputs = torch.rand(1, 3, 128, 128)
        with torch.no_grad(): before = model(inputs)[0]
        keys = {k: v.shape for k, v in model.state_dict().items()}
        configure_ncr(model); model.ncr_epoch = 19
        model.criterion = model.init_criterion()
        with torch.no_grad(): after = model(inputs)[0]
        self.assertEqual(keys, {k: v.shape for k, v in model.state_dict().items()})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20149765)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/"best.pt"
            torch.save(dict(model=model, image=inputs, predictions=after), file)
            subprocess.run([sys.executable, __file__, "--reload-child", str(file)], cwd=ROOT, check=True, timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reload-child", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.reload_child:
        saved = torch.load(args.reload_child, map_location="cpu", weights_only=False)
        with torch.no_grad(): value = saved["model"](saved["image"])[0]
        torch.testing.assert_close(saved["predictions"], value, atol=0, rtol=0)
        assert isinstance(saved["model"].criterion, NCRDetectionLoss)
        return 0
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    report = dict(status="PASSED" if result.wasSuccessful() else "FAILED", tests=result.testsRun,
                  failures=[str(x) for x in result.failures], errors=[str(x) for x in result.errors],
                  skipped=result.skipped, python=sys.version, torch=torch.__version__, reference_commit=PARENT,
                  scope="synthetic mathematical/wiring/inference/reload checks; not real-data effectiveness")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
