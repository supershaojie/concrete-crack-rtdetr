"""Bounded CPU correctness tests, including the actual original decoder and native optimizer."""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.rmd_v1 import (RMDLoss, RMDDetectionModel, RMDTrainer, CONTEXT,
    centered_weights, residual_weights, weighted_box_loss, promote, ramp, sync_epoch)
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils import ops as dn_ops
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load

MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml"
EVIDENCE = {}


def targets(groups):
    torch.manual_seed(19)
    n = sum(groups)
    return dict(cls=torch.zeros(n, dtype=torch.long),
                bboxes=torch.cat((torch.rand(n, 2) * .5 + .25, torch.rand(n, 2) * .12 + .08), 1),
                batch_idx=torch.cat([torch.full((g,), i, dtype=torch.long) for i, g in enumerate(groups)]),
                gt_groups=groups)


def fixture(k=3):
    batch = targets([2, 1])
    g = batch["bboxes"]
    split = [4 * k, 300]
    dn = [torch.arange(2 * k), torch.arange(k) * 2]
    meta = dict(dn_pos_idx=dn, dn_num_group=k, dn_num_split=split)
    match = [(torch.tensor([8, 4]), torch.tensor([0, 1])), (torch.tensor([9]), torch.tensor([2]))]
    dm = RMDLoss.get_dn_match_indices(dn, k, batch["gt_groups"])
    initial = torch.full((2, sum(split), 4), float("nan"))  # padding/negative must never be touched
    desired = torch.tensor([.9, .4, .1]) if k == 3 else torch.linspace(.1, .9, k)
    for image, ((q, gi), (p, dg)) in enumerate(zip(match, dm)):
        initial[image, split[0] + q] = g[gi]
        for j, (pi, gj) in enumerate(zip(p, dg)):
            s = desired[j // len(q)] if image == 0 else desired.flip(0)[j]
            noisy = g[gj].clone()
            noisy[0] += torch.sqrt(-.5 * torch.log(s)) * (g[gj, 2] + 1e-7)
            initial[image, pi] = noisy
    return batch, meta, match, dm, initial


def prediction_fixture(k):
    batch, meta, match, dm, initial = fixture(k)
    torch.manual_seed(97)
    boxes = torch.cat((torch.rand(4, 2, 300, 2) * .7 + .15,
                       torch.rand(4, 2, 300, 2) * .2 + .04), -1).requires_grad_()
    scores = torch.randn(4, 2, 300, 1, requires_grad=True)
    dnboxes = torch.cat((torch.rand(3, 2, 4*k, 2) * .7 + .15,
                         torch.rand(3, 2, 4*k, 2) * .2 + .04), -1).requires_grad_()
    dnscores = torch.randn(3, 2, 4*k, 1, requires_grad=True)
    # Make actual Hungarian choices predictable, with differing auxiliary matches.
    with torch.no_grad():
        for image, (q, gi) in enumerate(match):
            boxes[-1, image, q] = batch["bboxes"][gi]
            scores[-1, image, q] = 15
    return batch, {**meta, CONTEXT: initial}, (boxes, scores), dnboxes, dnscores


class Correctness(unittest.TestCase):
    def test_formula_edges(self):
        s = torch.tensor([.9, .4, .1], requires_grad=True)
        w = centered_weights(s, torch.tensor([0, 0, 0]), 1)
        torch.testing.assert_close(w, torch.tensor([1.2166667, .9666667, .8166667]))
        self.assertFalse(w.requires_grad)
        self.assertEqual(float(w.mean()), 1)
        for s in (torch.zeros(3), torch.full((3,), .4)):
            self.assertTrue(torch.equal(centered_weights(s, torch.zeros(3, dtype=torch.long), 1), torch.ones(3)))
        self.assertTrue(torch.equal(centered_weights(torch.rand(4), torch.arange(4), 1), torch.ones(4)))
        self.assertEqual([ramp(e) for e in (0, 5, 6, 20, 199)], [0., 0., 1/15, 1., 1.])
        with self.assertRaisesRegex(RuntimeError, "invalid similarity"):
            centered_weights(torch.tensor([float("nan")]), torch.tensor([0]), 1)

    def test_index_alignment_and_shuffle(self):
        for k in (1, 3, 7):
            batch, meta, match, dm, initial = fixture(k)
            w, stats = residual_weights(initial, batch, match, dm, meta, 1)
            self.assertEqual(len(w), 3*k)
            self.assertEqual(stats["positives"], 3*k)
            ids = torch.cat([d for _, d in dm])
            for identity in range(3):
                torch.testing.assert_close(w[ids == identity].mean(), torch.tensor(1.))
            if k == 3:
                torch.testing.assert_close(w[:6:2], torch.tensor([1.2166667, .9666667, .8166667]))
                torch.testing.assert_close(w[6:], torch.tensor([.8166667, .9666667, 1.2166667]))
            perm = [torch.randperm(len(p)) for p, _ in dm]
            shuffled = [(p[ix], d[ix]) for (p, d), ix in zip(dm, perm)]
            ws, _ = residual_weights(initial, batch, match, shuffled, meta, 1)
            expected = torch.cat([part[ix] for part, ix in zip(w.split([2*k, k]), perm)])
            # FP32 scatter-add changes reduction order under a permutation.
            torch.testing.assert_close(ws, expected, rtol=0, atol=1e-7)
            idx, gi = RMDLoss._get_index(dm)
            idx_s, gi_s = RMDLoss._get_index(shuffled)
            pred = torch.nan_to_num(initial[:, :meta["dn_num_split"][0]], nan=.2).clone()
            pred[..., 0] += .04
            l1 = weighted_box_loss(pred[idx], batch["bboxes"][gi], w, {"bbox": 5, "giou": 2})
            l2 = weighted_box_loss(pred[idx_s], batch["bboxes"][gi_s], ws, {"bbox": 5, "giou": 2})
            for key in l1:
                torch.testing.assert_close(l1[key], l2[key])
            crossed = [(dm[0][0], dm[0][1]), (dm[1][0], torch.zeros_like(dm[1][1]))]
            with self.assertRaisesRegex(RuntimeError, "crossed image"):
                residual_weights(initial, batch, match, crossed, meta, 1)
            if k > 1:
                mixed = [(dm[0][0], 1-dm[0][1]), dm[1]]
                wrong, _ = residual_weights(initial, batch, match, mixed, meta, 1)
                self.assertFalse(torch.allclose(w, wrong), "Test must detect within-image cross-GT misalignment")
        incomplete = [(match[0][0][:1], match[0][1][:1]), match[1]]
        w, stats = residual_weights(initial, batch, incomplete, dm, meta, 1)
        self.assertEqual(stats["unmatched_gt"], 1)
        self.assertTrue(torch.equal(w[:2*k][1::2], torch.ones(k)))
        bad = initial.clone(); bad[0, 0, 2] = 0
        with self.assertRaisesRegex(RuntimeError, "width/height"):
            residual_weights(bad, batch, match, dm, meta, 1)

    def test_loss_replacement_matching_and_gradients(self):
        changed = {"loss_bbox_dn", "loss_giou_dn", "loss_bbox_aux_dn", "loss_giou_aux_dn"}
        for k, epoch in ((3, 0), (1, 20), (3, 20)):
            batch, meta, preds, db, ds = prediction_fixture(k)
            native = RTDETRDetectionLoss(nc=1, use_vfl=True)
            loss = RMDLoss(nc=1, use_vfl=True); loss.enabled = True; loss.epoch = epoch
            original = native(preds, batch, db, ds, meta)
            with patch.object(loss.matcher, "forward", wraps=loss.matcher.forward) as matcher:
                actual = loss(preds, batch, db, ds, meta)
                self.assertEqual(matcher.call_count, 4)  # final + encoder + 2 auxiliary; no extra final matching
            for key in original:
                if epoch == 0 or k == 1 or key not in changed:
                    torch.testing.assert_close(original[key], actual[key], rtol=0, atol=0)
            source_grads = torch.autograd.grad(sum(original.values()), (*preds, db, ds), retain_graph=True)
            target_grads = torch.autograd.grad(sum(actual.values()), (*preds, db, ds))
            for i, (a, b) in enumerate(zip(source_grads, target_grads)):
                if epoch == 0 or k == 1 or i != 2:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            if k > 1 and epoch > 0:
                self.assertGreater(sum(abs(float(original[c]-actual[c])) for c in changed), 0)
                replacement = sum(original.values()) - sum(original[c] for c in changed) + sum(actual[c] for c in changed)
                torch.testing.assert_close(sum(actual.values()), replacement)
                self.assertEqual(len(loss.statistics["dn_layers"]), 3)
            self.assertFalse(hasattr(loss, "_layer_log"))
        with self.assertRaisesRegex(RuntimeError, "capture missing"):
            loss(preds, batch, db, ds, {k: v for k, v in meta.items() if k != CONTEXT})
        with self.assertRaisesRegex(RuntimeError, "metadata/predictions missing"):
            loss(preds, batch)

    def test_empty_and_missing_image(self):
        criterion = RMDLoss(nc=1, use_vfl=True); criterion.enabled = True; criterion.epoch = 20
        batch = targets([0, 0]); preds = (torch.rand(4, 2, 300, 4), torch.randn(4, 2, 300, 1))
        actual = criterion(preds, batch)
        native = RTDETRDetectionLoss(nc=1, use_vfl=True)(preds, batch)
        for key in native:
            torch.testing.assert_close(actual[key], native[key], rtol=0, atol=0)
        batch = targets([0, 1])
        _, _, _, meta = dn_ops.get_cdn_group(batch, 1, 300, torch.rand(1, 256), training=True)
        dm = criterion.get_dn_match_indices(meta["dn_pos_idx"], meta["dn_num_group"], batch["gt_groups"])
        normal = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)), (torch.tensor([0]), torch.tensor([0]))]
        initial = torch.full((2, sum(meta["dn_num_split"]), 4), .3)
        w, stats = residual_weights(initial, batch, normal, dm, meta, 1)
        self.assertEqual(len(w), 100)
        self.assertEqual(stats["unmatched_gt"], 0)

    def test_real_forward_rng_optimizer_and_hook_cleanup(self):
        torch.manual_seed(42)
        base = RTDETRDetectionModel(str(MODEL), nc=1, verbose=False)
        base.nc = 1  # Native Trainer.set_model_attributes supplies this during real training.
        for epoch, groups in ((0, [2, 1]), (20, [51, 50])):
            native, model = deepcopy(base).train(), promote(deepcopy(base)).train()
            model.rmd_epoch = epoch
            batch = targets(groups)
            img = torch.rand(2, 3, 160, 160)
            data = dict(batch, img=img, cls=batch["cls"].view(-1, 1))
            state = torch.get_rng_state()
            out = native.predict(img, batch=batch)
            after_native = torch.get_rng_state()
            native_loss = native.loss(data, out)[0]
            native_loss.backward()
            torch.set_rng_state(state)
            seen_refs = []
            hook = model.model[-1].decoder.layers[0].register_forward_pre_hook(lambda module, args: seen_refs.append(args[1].detach().clone()))
            with patch.object(dn_ops, "get_cdn_group", wraps=dn_ops.get_cdn_group) as cdn:
                actual = model.predict(img, batch=batch)
                self.assertEqual(cdn.call_count, 1)
            hook.remove()
            self.assertTrue(torch.equal(after_native, torch.get_rng_state()))
            for before, after in zip(out[:4], actual[:4]):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(len(model.model[-1].decoder._forward_pre_hooks), 0)
            if epoch:
                torch.testing.assert_close(actual[-1][CONTEXT], seen_refs[0], rtol=0, atol=0)
                self.assertFalse(actual[-1][CONTEXT].requires_grad)
                self.assertEqual(actual[-1]["dn_num_group"], 1)
            else:
                self.assertNotIn(CONTEXT, actual[-1])
            loss = model.loss(data, actual)[0]
            torch.testing.assert_close(loss, native_loss, rtol=0, atol=0)
            loss.backward()
            for (name, p), (name2, q) in zip(native.named_parameters(), model.named_parameters()):
                self.assertEqual(name, name2)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            trainer = RTDETRTrainer.__new__(RTDETRTrainer)
            optimizers = [trainer.build_optimizer(m, name="AdamW", lr=.0005, momentum=.937, decay=.0001) for m in (native, model)]
            for m, optimizer in zip((native, model), optimizers):
                torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
                optimizer.step()
            for name, p in native.state_dict().items():
                torch.testing.assert_close(p, model.state_dict()[name], rtol=0, atol=0)
            EVIDENCE[f"real_e{epoch}_K{out[-1]['dn_num_group']}"] = dict(outputs_exact=True, rng_exact=True,
                gradients_exact=True, native_adamw_update_exact=True, forward_calls=1, cdn_calls=1, hooks_after=0)
        model.rmd_epoch = 20
        before_failure = deepcopy(model.state_dict())
        with patch.object(model.model[-1].decoder, "forward", side_effect=RuntimeError("injected decoder failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                model.predict(img, batch=batch)
        self.assertEqual(len(model.model[-1].decoder._forward_pre_hooks), 0)
        # A decoder exception still follows the original backbone BN forward.
        # Restore the fault fixture before comparing eval to the untouched control.
        model.load_state_dict(before_failure)
        model.eval()
        native.eval()
        with torch.no_grad():
            eval_out = model(img)
            native_eval_out = native(img)
        torch.testing.assert_close(eval_out[0], native_eval_out[0], rtol=0, atol=0)
        self.assertIsNone(eval_out[1][-1])
        self.assertEqual(set(native.state_dict()), set(model.state_dict()))
        EVIDENCE["architecture"] = dict(parameters=sum(p.numel() for p in model.parameters()), added=0,
                                        state_keys=len(model.state_dict()), exception_hook_cleanup=True)

    def test_native_trainer_and_new_process(self):
        trainer = RMDTrainer.__new__(RMDTrainer)
        trainer.data = dict(nc=1, channels=3)
        model = trainer.get_model(str(MODEL), verbose=False)
        self.assertIsInstance(model, RMDDetectionModel)
        model.args = dict(DEFAULT_CFG_DICT, model=str(MODEL), task="detect")
        model.names = {0: "crack"}
        trainer.model, trainer.ema, trainer.epoch = model, ModelEMA(model), 17
        sync_epoch(trainer)
        self.assertEqual(model.rmd_epoch, 17)
        self.assertEqual(trainer.ema.ema.rmd_epoch, 17)
        trainer.epoch = 21; sync_epoch(trainer)
        self.assertEqual(ramp(model.rmd_epoch), 1)
        output = ROOT / "outputs/rmd_v1/local_checks/importable.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=deepcopy(model), ema=None, train_args=model.args, epoch=16), output)
        script = """import sys, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from ultralytics.utils.patches import torch_load
from ultralytics.models.rtdetr.rmd_v1 import RMDDetectionModel
torch.set_num_threads(4)
c = torch_load(sys.argv[2], map_location='cpu'); m = c['model'].float().eval()
assert isinstance(m, RMDDetectionModel)
assert all(not x._forward_pre_hooks for x in m.modules())
with torch.no_grad():
    y = m(torch.zeros(1, 3, 160, 160))[0]
assert torch.isfinite(y).all()
print('FRESH_PROCESS_PASS', tuple(y.shape))
"""
        result = subprocess.run([sys.executable, "-c", script, str(ROOT/"ultralytics-main"), str(output)],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        EVIDENCE["new_process"] = dict(status="PASS", output=result.stdout.strip(),
                                        real_val="PENDING: separate actual-data lifecycle check")
        EVIDENCE["native_rebuild"] = trainer.rmd_rebuild_audit

    def test_native_checkpoint_and_resumed_training_in_new_process(self):
        """Small CPU fixture proves lifecycle only, never B16/640 capacity or applicability."""
        import numpy as np
        from PIL import Image
        from ultralytics.utils import YAML
        from types import SimpleNamespace
        root = ROOT/"outputs/rmd_v1/local_checks/native_resume_fixture"
        for split, count in (("train", 4), ("val", 2)):
            (root/"images"/split).mkdir(parents=True, exist_ok=True)
            (root/"labels"/split).mkdir(parents=True, exist_ok=True)
            for i in range(count):
                rng = np.random.default_rng(i)
                Image.fromarray(rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)).save(root/f"images/{split}/{i}.png")
                (root/f"labels/{split}/{i}.txt").write_text("0 0.45 0.55 0.15 0.2\n", encoding="utf-8")
        data = root/"data.yaml"
        YAML.save(data, dict(path=str(root), train="images/train", val="images/val", names={0: "crack"}))
        # Unique diagnostic directory also makes this repeatable without touching formal results.
        import uuid
        name = "cpu_"+uuid.uuid4().hex[:8]
        args = dict(model=str(MODEL), data=str(data), project=str(root), name=name, epochs=30,
                    device="cpu", batch=2, imgsz=160, workers=0, amp=False, plots=False,
                    optimizer="AdamW", lr0=.0005, momentum=.937, weight_decay=.0001, seed=42)
        trainer = RMDTrainer(overrides=args)
        trainer._setup_train(); trainer.epoch = 20; sync_epoch(trainer)
        trainer.model.train()
        batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
        loss, trainer.loss_items = trainer.model(batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        trainer.loss = loss
        loss.backward(); trainer.optimizer_step()
        trainer.fitness = trainer.best_fitness = 0.
        trainer.save_model()
        checkpoint = torch_load(trainer.last, map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 20)
        self.assertIsNotNone(checkpoint["optimizer"])
        script = """import sys, torch
sys.path.insert(0, sys.argv[1])
from ultralytics.models.rtdetr.rmd_v1 import RMDTrainer, RMDDetectionModel, sync_epoch
from ultralytics.utils.patches import torch_load
torch.set_num_threads(1)
checkpoint = torch_load(sys.argv[2], map_location='cpu')
t = RMDTrainer(overrides=dict(model=sys.argv[2], resume=sys.argv[2], device='cpu', workers=0))
t._setup_train()
assert t.start_epoch == 21 and isinstance(t.model, RMDDetectionModel)
assert t.scaler.state_dict() == checkpoint['scaler']
restored = t.optimizer.state_dict()
assert restored['param_groups'] == checkpoint['optimizer']['param_groups']
for key, row in checkpoint['optimizer']['state'].items():
    for field, value in row.items():
        actual = restored['state'][key][field]
        assert torch.equal(actual.cpu(), value.to(actual.dtype).cpu()) if isinstance(value, torch.Tensor) else actual == value
for key, value in checkpoint['ema'].float().state_dict().items():
    assert torch.equal(t.ema.ema.state_dict()[key].cpu(), value)
assert t.ema.updates == checkpoint['updates']
t.epoch = t.start_epoch; sync_epoch(t); assert t.model.rmd_epoch == 21
t.model.train()
batch = t.preprocess_batch(next(iter(t.train_loader)))
loss, t.loss_items = t.model(batch); assert torch.isfinite(loss)
t.loss = loss; t.scaler.scale(loss).backward()
before = t.model.model[-1].cbr.offset_out.weight.detach().clone()
t.optimizer_step()
assert not torch.equal(before, t.model.model[-1].cbr.offset_out.weight)
assert all(torch.isfinite(p).all() for p in t.model.parameters())
assert all(not m._forward_pre_hooks for m in t.model.modules())
print('NATIVE_RESUME_TRAIN_PASS epoch=21 optimizer/EMA/scaler restored; CPU fixture only')
"""
        result = subprocess.run([sys.executable, "-c", script, str(ROOT/"ultralytics-main"), str(trainer.last)],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        EVIDENCE["native_resume"] = dict(status="PASS", epoch=21, checkpoint_schema="original trainer.save_model",
            optimizer_ema_scaler_restored=True, new_process_training_update=True, fixture="CPU B2/160; not capacity evidence")


def run(output):
    torch.set_num_threads(1)
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Correctness)
    with torch.autograd.set_multithreading_enabled(False):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
                  elapsed_seconds=time.monotonic()-started, evidence=EVIDENCE,
                  failures=[str(f) for f in result.failures + result.errors],
                  executable=sys.executable, torch=str(torch.__version__),
                  criterion_source=inspect.getfile(RMDLoss),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    from rmd_v1_common import code_identity
    report["tested_code"] = code_identity()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/rmd_v1/local_checks/correctness.json")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
