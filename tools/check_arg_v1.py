"""Bounded CPU math, routing, native reconstruction/update and serialization checks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

from arg_v1_common import ROOT, OUT, SOURCE, MODEL, runtime, write_json, source_contract
import torch
from torch import nn
from ultralytics.models.rtdetr.arg_loss import ARGDetectionLoss, ARGGeometryError, geometry, interval_loss, ramp, xyxy
from ultralytics.models.rtdetr.arg_model import ARGDetectionModel, ARGTrainer, groups
from ultralytics.models.rtdetr.arg_val import postprocess
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load


def cxcywh(boxes):
    b = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return torch.cat(((b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]), -1)


class Mathematics(unittest.TestCase):
    def test_teaching_and_normalized(self):
        for scale in (1, 1 / 128):
            b = cxcywh([[5, 2, 105, 12]]) * scale
            g = (cxcywh([[0, 0, 100, 10]]) * scale).requires_grad_()
            b.requires_grad_()
            raw, d = geometry(b, g)
            expected = dict(q=.6129032258, qx=.6666666667, qy=.9047619048, wx=.4657014822, wy=1.5342985178)
            for key, value in expected.items():
                self.assertAlmostEqual(float(d[key]), value, delta=2e-6)
            for key in ("q", "qx", "qy", "dx", "dy", "wx", "wy"):
                self.assertFalse(d[key].requires_grad)
            self.assertTrue(raw.requires_grad)
            grad, gt_grad = torch.autograd.grad(raw, (b, g), allow_unused=True)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertIsNone(gt_grad)

    def test_geometries(self):
        gt = cxcywh([[0, 0, 1, 1]])
        cases = [[0, 0, 1, 1], [.2, 0, 1.2, 1], [0, .2, 1, 1.2], [-.5, -.5, 1.5, 1.5],
                 [.2, .2, .8, .8], [2, .1, 3, .9], [.49, .1, .4901, .9]]
        for case in cases:
            b = cxcywh([case]).requires_grad_()
            raw, d = geometry(b, gt)
            torch.testing.assert_close(d["wx"] + d["wy"], torch.full_like(d["wx"], 2))
            self.assertTrue(bool(((d["wx"] > 0) & (d["wx"] < 2) & (d["wy"] > 0) & (d["wy"] < 2)).all()))
            self.assertTrue(bool(((d["qx"] + 1e-6 >= d["q"]) & (d["qy"] + 1e-6 >= d["q"])).all()))
            # At valid scales repair-x IoU equals vertical 1D IoU, not horizontal.
            bb, gg = xyxy(b), xyxy(gt)
            for key, axis in (("qx", [1, 3]), ("qy", [0, 2])):
                lo = torch.maximum(bb[:, axis[0]], gg[:, axis[0]])
                hi = torch.minimum(bb[:, axis[1]], gg[:, axis[1]])
                inter = (hi - lo).clamp_min(0)
                union = bb[:, axis[1]] - bb[:, axis[0]] + gg[:, axis[1]] - gg[:, axis[0]] - inter
                torch.testing.assert_close(d[key], inter / union)
            swapped, ds = geometry(b[:, [1, 0, 3, 2]], gt[:, [1, 0, 3, 2]])
            torch.testing.assert_close(raw, swapped)
            torch.testing.assert_close(d["wx"], ds["wy"])
        self.assertEqual(float(geometry(gt, gt)[0]), 0)
        b = cxcywh([[2, .1, 3, .9]]).requires_grad_()
        gradient = torch.autograd.grad(geometry(b, gt)[0], b)[0]
        self.assertGreater(float(gradient[0, 0]), 0)  # gradient descent moves left toward GT
        self.assertLess(float(gradient[0, 2]), 0)

    def test_frozen_weight_finite_difference(self):
        b = cxcywh([[.05, .2, 1.05, 1.2]]).requires_grad_()
        g = cxcywh([[0, 0, 1, 1]])
        raw, details = geometry(b, g)
        auto = torch.autograd.grad(raw, b)[0]
        def objective(v):
            bb, gg = xyxy(v), xyxy(g)
            ex = interval_loss(bb[:, [0, 2]], gg[:, [0, 2]])[0]
            ey = interval_loss(bb[:, [1, 3]], gg[:, [1, 3]])[0]
            return (0.5 * (details["wx"] * ex + details["wy"] * ey)).mean()
        eps = 1e-3
        finite = torch.zeros_like(b)
        for j in range(4):
            offset = torch.zeros_like(b); offset[0, j] = eps
            finite[0, j] = (objective(b.detach() + offset) - objective(b.detach() - offset)) / (2 * eps)
        torch.testing.assert_close(auto, finite, atol=2e-4, rtol=2e-3)

    def test_empty_tiny_invalid_schedule(self):
        raw, _ = geometry(torch.zeros(0, 4, requires_grad=True), torch.zeros(0, 4))
        self.assertEqual(float(raw), 0)
        tiny = cxcywh([[0, 0, 1e-5, 1e-5]])
        raw, details = geometry(tiny, tiny)
        self.assertTrue(torch.isfinite(raw))
        self.assertTrue(details["protected"].all())
        self.assertLess(float(details["q"]), 1)  # area protection is intentionally biased
        for value in (float("nan"), 0, -1):
            with self.assertRaises(ARGGeometryError):
                geometry(torch.tensor([[.5, .5, value, .2]]), tiny)
        self.assertEqual([ramp(x) for x in (0, 5, 6, 19, 20, 100)], [0, 0, 1/15, 14/15, 1, 1])

    def test_sorted_mask(self):
        raw = torch.tensor([[[.2, .2, .1, .1, .0001], [.5, .5, .3, .3, .9], [.8, .8, .2, .2, .01]]])
        output = postprocess(raw, 640, .001)[0]
        torch.testing.assert_close(output["conf"], torch.tensor([.9, .01]))
        torch.testing.assert_close(output["bboxes"][0], torch.tensor([224., 224., 416., 416.]))


class MatchRecorder(nn.Module):
    def __init__(self, matcher):
        super().__init__(); self.matcher = matcher; self.calls = []

    def forward(self, *args, **kwargs):
        result = self.matcher(*args, **kwargs)
        self.calls.append([(a.tolist(), b.tolist()) for a, b in result])
        return result


def routing_check():
    torch.manual_seed(42)
    boxes = (torch.rand(4, 2, 6, 4) * .5 + .1).requires_grad_()
    scores = torch.randn(4, 2, 6, 1, requires_grad=True)
    batch = dict(bboxes=torch.tensor([[.4, .4, .2, .2], [.7, .6, .1, .3]]), cls=torch.zeros(2, dtype=torch.long), gt_groups=[2, 0])
    evidence = []
    for dn in (False, True):
        kwargs = dict(dn_bboxes=boxes[:3, :, :4], dn_scores=scores[:3, :, :4],
                      dn_meta=dict(dn_num_group=1, dn_pos_idx=[torch.tensor([0, 1]), torch.tensor([], dtype=torch.long)])) if dn else {}
        mother = RTDETRDetectionLoss(nc=1, use_vfl=True)
        criterion = ARGDetectionLoss(nc=1, use_vfl=True)
        mother.matcher = MatchRecorder(mother.matcher)
        criterion.matcher = MatchRecorder(criterion.matcher)
        expected = mother((boxes, scores), batch, **kwargs)
        for epoch in (0, 20):
            criterion.epoch, criterion.sample = epoch, True
            criterion.matcher.calls = []
            actual = criterion((boxes, scores), batch, **kwargs)
            assert all(torch.equal(value, actual[key]) for key, value in expected.items())
            assert criterion.matcher.calls == mother.matcher.calls
            assert len(criterion.matcher.calls) == 4
            assert set(actual) - set(expected) == ({"loss_arg"} if epoch == 20 else set())
            if epoch == 20:
                grad, cls_grad = torch.autograd.grad(actual["loss_arg"], (boxes, scores), retain_graph=True, allow_unused=True)
                assert cls_grad is None and torch.count_nonzero(grad[:-1]) == 0
                used = {i for i in criterion.last_matches[0][0]}
                assert all(torch.count_nonzero(grad[-1, 0, i]) == 0 for i in range(6) if i not in used)
                assert torch.count_nonzero(grad[:, 1]) == 0
            evidence.append(dict(dn=dn, epoch=epoch, calls=len(criterion.matcher.calls), keys=list(actual)))
        criterion.enabled = False
        validation = criterion((boxes, scores), batch, **kwargs)
        assert set(validation) == set(expected) and all(torch.equal(v, validation[k]) for k, v in expected.items())
        criterion.enabled = True
    empty = dict(bboxes=torch.zeros(0, 4), cls=torch.zeros(0, dtype=torch.long), gt_groups=[0, 0])
    losses = criterion((boxes, scores), empty)
    assert losses["loss_arg"] == 0 and torch.isfinite(losses["loss_class"])
    return evidence


def optimizer(model):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    return trainer.build_optimizer(model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)


def model_check(output):
    torch.set_num_threads(1)  # repeated-index CPU embedding backward reduction order
    from init_c19_lif_v1 import controlled_models, build_training_model
    _, weights, init_report = controlled_models(SOURCE)
    torch.manual_seed(42)
    mother, mapping = build_training_model(str(MODEL), weights, dict(nc=1, channels=3))
    trainer = ARGTrainer.__new__(ARGTrainer); trainer.data = dict(nc=1, channels=3)
    torch.manual_seed(42)
    arg = trainer.get_model(str(MODEL), weights, False)
    assert all(torch.equal(v, arg.state_dict()[k]) for k, v in mother.state_dict().items())
    del weights
    batch = dict(img=torch.rand(2, 3, 160, 160), bboxes=torch.tensor([[.4, .4, .2, .2]]),
                 cls=torch.zeros(1, 1), batch_idx=torch.tensor([0]))
    mother.nc = arg.nc = 1  # normally supplied by native Trainer.set_model_attributes
    mother.train(); arg.train()
    left, right = optimizer(mother), optimizer(arg)
    assert groups(mother, left) == groups(arg, right)
    def detached(value):
        if torch.is_tensor(value): return value.detach().clone()
        if isinstance(value, dict): return {k: detached(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [detached(v) for v in value]
        return value
    def same(a, b):
        if torch.is_tensor(a): return torch.equal(a, b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    def run(model):
        torch.manual_seed(734)
        output = []
        hook = model.model[-1].register_forward_hook(lambda m, a, y: output.append(detached(y)))
        try:
            loss, items = model(batch)
        finally:
            hook.remove()
        loss.backward()
        return loss.detach(), items, output
    a, b = run(mother), run(arg)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert same(a[2], b[2]), "Native raw forward/DN metadata changed at r=0"
    for (n, p), (k, q) in zip(mother.named_parameters(), arg.named_parameters()):
        assert n == k and ((p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)), n
    for model, opt in ((mother, left), (arg, right)):
        native = RTDETRTrainer.__new__(RTDETRTrainer)
        native.model, native.optimizer, native.ema = model, opt, None
        native.scaler = torch.cuda.amp.GradScaler(enabled=False)
        native.optimizer_step()
    assert all(torch.equal(v, arg.state_dict()[k]) for k, v in mother.state_dict().items())
    # Nonzero original output layers after the update allow ARG to reach shared paths.
    arg.arg_epoch, arg.arg_sample = 20, True
    target = dict(bboxes=batch["bboxes"], cls=batch["cls"].long().view(-1), batch_idx=batch["batch_idx"], gt_groups=[1, 0])
    torch.manual_seed(123)
    dec, score, enc, enc_score, dn = arg.predict(batch["img"], batch=target)
    db, boxes = torch.split(dec, dn["dn_num_split"], dim=2)
    ds, scores = torch.split(score, dn["dn_num_split"], dim=2)
    criterion = ARGDetectionLoss(nc=1, use_vfl=True); criterion.epoch = 20
    losses = criterion((torch.cat((enc[None], boxes)), torch.cat((enc_score[None], scores))), target, db, ds, dn)
    losses["loss_arg"].backward()
    grad = {}
    for key in ("model.26.cbr.offset_out.weight", "model.26.cbr.query_proj.weight", "model.26.dec_bbox_head.2.layers.2.weight", "model.19.cv1.conv.weight"):
        value = dict(arg.named_parameters())[key].grad
        assert value is not None and torch.isfinite(value).all() and torch.count_nonzero(value), key
        grad[key] = float(value.norm())
    assert all(p.grad is None for p in arg.model[-1].dec_score_head[-1].parameters())
    arg.zero_grad(set_to_none=True)
    ema = ModelEMA(arg); ema.updates = 1
    args = dict(model=str(MODEL), data="diagnostic-only", epochs=200)
    arg.args = args; ema.ema.args = args
    checkpoint = output / "diagnostic_resume.pt"
    torch.save(dict(model=None, ema=deepcopy(ema.ema), optimizer=right.state_dict(), scaler={}, updates=1,
                    epoch=19, best_fitness=.1, train_args=args), checkpoint)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reload", str(checkpoint)], cwd=ROOT, check=True, timeout=90)
    return dict(status="PASS", shape="synthetic B2/160 CPU; not formal B16 capacity", states=len(arg.state_dict()),
                parameters=sum(p.numel() for p in arg.parameters()), r0_loss=float(a[0]),
                r0_forward_exact=True, r0_gradients_exact=True, r0_native_optimizer_update_exact=True, optimizer_groups=groups(arg, right),
                init_mapping=mapping, source_sha256=init_report["source_sha256"], arg_only_gradients=grad,
                new_process_resume=read_json_local(output / "diagnostic_resume.reload.json"))


def read_json_local(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reload_check(path):
    ckpt = torch_load(path, map_location="cpu")
    t = ARGTrainer.__new__(ARGTrainer); t.data = dict(nc=1, channels=3)
    t.model = t.get_model(str(MODEL), ckpt["ema"].float(), False)
    t.optimizer = optimizer(t.model)
    t.scaler = torch.cuda.amp.GradScaler(enabled=False)
    t.ema = ModelEMA(t.model); t.resume = True; t.epochs = 200
    t.args = SimpleNamespace(model=str(path), close_mosaic=10)
    t.resume_training(ckpt)
    assert t.start_epoch == 20 and ramp(t.start_epoch) == 1 and t.ema.updates == 1
    for k, state in ckpt["optimizer"]["state"].items():
        for name, value in state.items():
            actual = t.optimizer.state_dict()["state"][k][name]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    assert t.scaler.state_dict() == ckpt["scaler"]
    assert all(torch.equal(v, t.ema.ema.state_dict()[k]) for k, v in ckpt["ema"].state_dict().items())
    t.model.eval()
    with torch.no_grad():
        predictions = t.model(torch.zeros(1, 3, 160, 160))[0]
    assert predictions.shape == (1, 300, 5) and torch.isfinite(predictions).all()
    write_json(path.with_suffix(".reload.json"), dict(status="PASS", epoch=t.start_epoch, ramp=ramp(t.start_epoch),
        optimizer_exact=True, ema_exact=True, scaler_exact=True, scaler_scope="CPU disabled scaler; CUDA state tested by server preflight",
        validation_scope="synthetic inference; real val pending server"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "local_checks")
    parser.add_argument("--math-only", action="store_true")
    parser.add_argument("--reload", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.reload:
        reload_check(args.reload); return
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status="FAIL", runtime=runtime(), modules=source_contract(), started=time.time())
    try:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Mathematics))
        assert result.wasSuccessful(), "Math checks failed"
        report["math"] = dict(status="PASS", tests=result.testsRun)
        report["routing"] = dict(status="PASS", evidence=routing_check())
        if not args.math_only:
            report["model"] = model_check(args.output)
        report["status"] = "PASS"
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        write_json(args.output / "checks.json", report)


if __name__ == "__main__":
    main()
