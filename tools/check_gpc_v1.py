"""Bounded PyTorch GPC contracts. CPU/small synthetic CUDA are not formal preflight."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import numpy as np
import torch
from torch import nn
from ultralytics.models.rtdetr.gpc import (GPCConfig, GPCDetectionModel, auxiliary_loss, consistency_xyxy,
    matched_consistency, paired_forward_scope, paired_targets, select_views, translate)
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh

DEVICE = "cpu"
EVIDENCE = {}
MODEL = str(ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml")


def sample(device="cpu"):
    return dict(img=torch.rand(2, 3, 160, 192, device=device),
                bboxes=torch.tensor([[.5, .5, .2, .1], [.3, .4, .1, .2], [.6, .5, .12, .15]], device=device),
                cls=torch.zeros(3, 1, device=device), batch_idx=torch.tensor([0, 0, 1], device=device),
                im_file=["a.jpg", "b.jpg"])


def context(batch, epoch=20):
    return dict(batch, gpc_context=dict(epoch=epoch, batch_index=2, seed=42, image_ids=batch["im_file"], metrics={}))


class RecordingMatcher(HungarianMatcher):
    def forward(self,*args,**kwargs):
        result=super().forward(*args,**kwargs)
        self.records.append([(a.clone(),b.clone()) for a,b in result])
        return result


class RecordingCriterion(RTDETRDetectionLoss):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.matcher=RecordingMatcher(cost_gain=self.matcher.cost_gain)

    def forward(self, *args, **kwargs):
        self.matcher.records=[]
        losses = super().forward(*args, **kwargs)
        self.observed = {k: v.detach().clone() for k, v in losses.items()}
        self.observed_input=tuple(x.detach().clone() for x in args[0])
        self.dn = deepcopy(kwargs.get("dn_meta"))
        return losses


class FormulaTests(unittest.TestCase):
    def test_integer_translation_all_directions_non_square(self):
        x = torch.arange(3 * 19 * 29).reshape(3, 19, 29).float()
        original = x.clone()
        for dx, dy in GPCConfig().shift_choices_xy:
            y = translate(x, dx, dy)
            for j in range(19):
                for i in range(29):
                    expected = x[:, j - dy, i - dx] if 0 <= i - dx < 29 and 0 <= j - dy < 19 else torch.full((3,), 114 / 255)
                    self.assertTrue(torch.equal(y[:, j, i], expected))
        self.assertTrue(torch.equal(original, x))
        self.assertTrue(torch.equal(translate(x, 0, 0), x))

    def test_rng_ramp_and_identity(self):
        config = GPCConfig()
        self.assertEqual([config.ramp(e) for e in (0, 5, 6, 20, 200)], [0, 0, 1 / 15, 1, 1])
        state, py, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
        first = select_views(16, list(map(str, range(16))), 20, 10)
        self.assertEqual(first, select_views(16, list(map(str, range(16))), 20, 10))
        self.assertNotEqual(first, select_views(16, list(map(str, range(16))), 21, 10))
        self.assertEqual(len(set(first[0])), 4)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        with self.assertRaises(ValueError):
            GPCConfig.from_dict(dict(shift_pixels=16))

    def test_visibility_and_parent_offsets(self):
        batch = dict(img=torch.zeros(2, 3, 100, 200),
                     bboxes=xyxy2xywh(torch.tensor([[.2,.2,.4,.5], [.94,.2,1.,.5], [.97,.2,.99,.5],
                                                   [0.,0.,.1,.1], [.4,.4,.6,.6]])),
                     cls=torch.zeros(5, 1), batch_idx=torch.tensor([0,0,0,1,1]))
        frozen = batch["bboxes"].clone()
        targets = paired_targets(batch, [1,0], [(-8,-8),(8,0)])
        self.assertEqual(set(targets["eligible"]), {(1,1),(0,0)})
        self.assertEqual(targets["gt_groups"], [2,3,2,2])
        self.assertEqual(targets["counts"]["clipped_gt"], 2)
        self.assertEqual(targets["counts"]["vanished_gt"], 1)
        self.assertTrue(torch.equal(batch["bboxes"], frozen))
        # Real Hungarian, shuffled queries; shifted labels have different global offsets/GT counts.
        boxes = torch.full((4, 8, 4), .01)
        offset = 0
        for row, size in enumerate(targets["gt_groups"]):
            boxes[row, torch.arange(size - 1, -1, -1)] = targets["bboxes"][offset:offset+size]
            offset += size
        loss, stats = matched_consistency(boxes, torch.zeros(4,8,1), targets, RTDETRDetectionLoss(1).matcher, True)
        self.assertLess(float(loss), 1e-12)
        self.assertEqual(stats["pairs"], 2)
        self.assertEqual({tuple(x["parent_gt_id"]) for x in stats["per_gt"]}, {(1,1),(0,0)})
        # Fewer queries than GT: intersection, not fabricated matches.
        loss, stats = matched_consistency(boxes[:,:1], torch.zeros(4,1,1), targets, RTDETRDetectionLoss(1).matcher)
        self.assertLessEqual(stats["pairs"], 2)
        empty = dict(batch, bboxes=torch.empty(0,4), cls=torch.empty(0,1), batch_idx=torch.empty(0))
        self.assertFalse(paired_targets(empty,[0],[(8,8)])["eligible"])

    def test_zero_perfect_wrong_and_gradients(self):
        gt = torch.tensor([[.1,.2,.5,.3]], requires_grad=True)
        delta = torch.tensor([[8/200,8/100,8/200,8/100]])
        a = (gt.detach() + .1).requires_grad_()  # consistently wrong predictions are an acknowledged degeneracy
        b = (a.detach() + delta).requires_grad_()
        loss, _ = consistency_xyxy(a,b,gt,delta,(100,200))
        self.assertLess(float(loss), 1e-12)
        b = (a.detach() + delta + torch.tensor([[.02,-.03,.04,-.02]])).requires_grad_()
        loss, residual = consistency_xyxy(a,b,gt,delta,(100,200))
        loss.backward()
        scale = torch.tensor([[.4,.1,.4,.1]])
        expected = residual.detach().clamp(-1,1) / scale / 4
        torch.testing.assert_close(a.grad,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(b.grad,-expected,rtol=1e-5,atol=1e-6)
        self.assertIsNone(gt.grad)
        fixed, _ = consistency_xyxy(a,a,gt,delta,(100,200))
        self.assertGreater(float(fixed),0)
        roundtrip = xywh2xyxy(xyxy2xywh(gt.detach()))
        torch.testing.assert_close(roundtrip,gt.detach())
        with self.assertRaises(FloatingPointError):
            consistency_xyxy(a * float("nan"), b, gt, delta, (100,200))

    def test_bn_scope_restores_exception_rng(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleList([nn.BatchNorm2d(2), nn.Identity()])
        model = Toy().train()
        bn = model.model[0]
        before = {k:v.clone() for k,v in bn.state_dict().items()}
        rng = torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with paired_forward_scope(model,torch.device("cpu")):
                self.assertTrue(model.training and model.model[-1].training)
                self.assertFalse(bn.training)
                bn(torch.rand(2,2,3,3)).sum().backward()
                raise RuntimeError("injected")
        self.assertTrue(bn.training)
        self.assertIsNotNone(bn.weight.grad)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,bn.state_dict()[k]) for k,v in before.items()))


class ModelTests(unittest.TestCase):
    def test_native_equivalence_active_bn_and_serialization(self):
        torch.manual_seed(42)
        base = RTDETRDetectionModel(MODEL,nc=1,verbose=False).to(DEVICE).train()
        new = GPCDetectionModel(MODEL,nc=1,verbose=False,gpc_config=dict(enabled=False)).to(DEVICE).train()
        new.load_state_dict(base.state_dict(),strict=True)
        self.assertEqual(list(base.state_dict()),list(new.state_dict()))
        self.assertEqual(sum(p.numel() for p in new.parameters()),20149765)
        self.assertIs(GPCDetectionModel.predict,RTDETRDetectionModel.predict)
        batch = sample(DEVICE)
        state = deepcopy(base.state_dict())
        base.criterion = RecordingCriterion(nc=1,use_vfl=True)
        new.criterion = RecordingCriterion(nc=1,use_vfl=True)
        opt0,opt1 = (torch.optim.AdamW(m.parameters(),lr=.0005) for m in (base,new))
        repeat = deepcopy(base) if DEVICE == "cuda" else None
        opt_repeat = torch.optim.AdamW(repeat.parameters(),lr=.0005) if repeat is not None else None
        # Exact CPU equality is the declared tolerance; CUDA uses 1e-6 absolute / 1e-5 relative.
        atol,rtol = (0.,0.) if DEVICE == "cpu" else (1e-6,1e-5)
        torch.manual_seed(111)
        loss0,items0 = base.loss(batch)
        loss0.backward()
        torch.manual_seed(111)
        loss1,items1 = new.loss(batch)
        loss1.backward()
        native_jitter=0.
        if repeat is not None:
            torch.manual_seed(111)
            repeat_loss,_=repeat.loss(batch)
            repeat_loss.backward()
            native_jitter=max(float((a.grad-b.grad).abs().max()) for a,b in zip(base.parameters(),repeat.parameters()) if a.grad is not None)
        torch.testing.assert_close(loss0,loss1,atol=atol,rtol=rtol)
        self.assertEqual(set(base.criterion.observed),set(new.criterion.observed))
        for key,value in base.criterion.observed.items():
            torch.testing.assert_close(value,new.criterion.observed[key],atol=atol,rtol=rtol)
        self.assertEqual(base.criterion.dn["dn_num_split"],new.criterion.dn["dn_num_split"])
        for a,b in zip(base.criterion.observed_input,new.criterion.observed_input):
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        for a,b in zip(base.criterion.dn["dn_pos_idx"],new.criterion.dn["dn_pos_idx"]):
            self.assertTrue(torch.equal(a,b))
        self.assertEqual(len(base.criterion.matcher.records),len(new.criterion.matcher.records))
        for a,b in zip(base.criterion.matcher.records,new.criterion.matcher.records):
            for (aq,ag),(bq,bg) in zip(a,b):
                self.assertTrue(torch.equal(aq,bq) and torch.equal(ag,bg))
        max_grad = 0.
        for a,b in zip(base.parameters(),new.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                if DEVICE=="cpu":
                    torch.testing.assert_close(a.grad,b.grad,atol=atol,rtol=rtol)
                max_grad=max(max_grad,float((a.grad-b.grad).abs().max()))
        if repeat is not None:
            # Keep the original strict tolerance in evidence; use a separate, declared
            # native-repeat envelope (3x measured maximum) for nondeterministic CUDA atomics.
            self.assertLessEqual(max_grad,max(1e-6,3*native_jitter))
        opt0.step();opt1.step()
        if opt_repeat is not None:opt_repeat.step()
        parameter_delta=max(float((a-b).abs().max()) for a,b in zip(base.parameters(),new.parameters()))
        update_jitter=max(float((a-b).abs().max()) for a,b in zip(base.parameters(),repeat.parameters())) if repeat is not None else 0.
        for a,b in zip(base.parameters(),new.parameters()):
            if DEVICE=="cpu":torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        if repeat is not None:self.assertLessEqual(parameter_delta,max(1e-6,3*update_jitter))
        new.zero_grad(set_to_none=True)
        del opt0,opt1,opt_repeat,repeat
        # Inference contract is compared at identical weights, not after divergent CUDA atomic updates.
        new.load_state_dict(base.state_dict(),strict=True)
        base.eval();new.eval()
        with torch.no_grad():
            torch.testing.assert_close(base(batch["img"])[0],new(batch["img"])[0],atol=atol,rtol=rtol)
            base.model[-1].export = new.model[-1].export = True
            torch.testing.assert_close(base(batch["img"]),new(batch["img"]),atol=atol,rtol=rtol)
            base.model[-1].export = new.model[-1].export = False
        new.gpc_config=asdict(GPCConfig())
        new.train()
        with self.assertRaisesRegex(RuntimeError,"context"):
            new.loss(batch)
        # Off schedule/lambda/validator must not enter an auxiliary or RNG scope.
        with patch("ultralytics.models.rtdetr.gpc.auxiliary_loss",side_effect=AssertionError("unexpected auxiliary")):
            new.loss(context(batch,5))
            new.gpc_config["loss_weight"]=0.0
            new.loss(batch)
            new.gpc_config["loss_weight"]=0.1
            new.eval()
            with torch.no_grad():
                preds=new.predict(batch["img"])
                new.loss(batch,preds=preds)
        # Active branch's main loss is exactly the native loss at the same RNG/BN state.
        base.train();new.train()
        base.load_state_dict(state,strict=True);new.load_state_dict(state,strict=True)
        torch.manual_seed(222)
        main,_=base.loss(batch)
        rng_after_main=torch.get_rng_state()
        cuda_rng_after_main=torch.cuda.get_rng_state() if DEVICE=="cuda" else None
        torch.manual_seed(222)
        active_batch=context(batch)
        total,_=new.loss(active_batch)
        self.assertTrue(torch.equal(rng_after_main,torch.get_rng_state()))
        if cuda_rng_after_main is not None:self.assertTrue(torch.equal(cuda_rng_after_main,torch.cuda.get_rng_state()))
        for key,value in base.criterion.observed.items():
            torch.testing.assert_close(value,new.criterion.observed[key],atol=atol,rtol=rtol)
        self.assertFalse(any("gpc" in k for k in new.criterion.observed))
        for (name,a),(_,b) in zip(base.named_buffers(),new.named_buffers()):
            if "running_" in name or "num_batches_tracked" in name:
                self.assertTrue(torch.equal(a,b),name)
        stats=active_batch["gpc_context"]["metrics"]
        self.assertEqual(stats["auxiliary_images"],4)
        self.assertTrue(stats["aux_dn_none"])
        torch.testing.assert_close(total.detach()-main.detach(),total.new_tensor(stats["weighted_gpc"]),atol=2e-6,rtol=1e-4)
        total.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in new.named_parameters() if "cbr.offset_out" in n))
        # Explicit engineering-only zero shift, same joint auxiliary batch/BN statistics.
        with torch.no_grad():
            zero,zero_stats,zero_raw=auxiliary_loss(new,batch,[0,1],[(0,0),(0,0)],new.criterion.matcher,return_predictions=True)
        self.assertIsNone(zero_raw[-1])
        self.assertEqual(tuple(zero_raw[0].shape),(3,4,300,4))
        self.assertLess(float(zero),1e-10)
        # No eligible GT must skip auxiliary predict, without resampling.
        empty=dict(batch,bboxes=batch["bboxes"][:0],cls=batch["cls"][:0],batch_idx=batch["batch_idx"][:0])
        with patch.object(new,"predict",side_effect=AssertionError("unexpected forward")):
            zero,_=auxiliary_loss(new,empty,[0,1],[(8,8),(8,8)],new.criterion.matcher)
            self.assertEqual(float(zero),0.)
        # Remove test-only recorder; saved model resolves from a real importable module.
        new.criterion=RTDETRDetectionLoss(nc=1,use_vfl=True)
        from ultralytics.utils.torch_utils import ModelEMA
        ema=ModelEMA(new)
        ema.update(new)
        buffer=io.BytesIO()
        torch.save(dict(model=ema.ema,epoch=6,algorithm=new.gpc_config),buffer)
        buffer.seek(0)
        restored=torch.load(buffer,map_location=DEVICE,weights_only=False)["model"]
        self.assertIs(type(restored),GPCDetectionModel)
        self.assertEqual(restored.gpc_config,new.gpc_config)
        self.assertEqual(list(restored.state_dict()),list(base.state_dict()))
        EVIDENCE.update(parameters=20149765,added_parameters=0,closed_max_gradient_difference=max_grad,
                        native_repeat_max_gradient_difference=native_jitter,
                        closed_max_optimizer_parameter_difference=parameter_delta,native_repeat_max_optimizer_parameter_difference=update_jitter,
                        cuda_comparison="3x native-repeat maximum error envelope; not strict deterministic equality" if DEVICE=="cuda" else "exact",
                        original_loss_keys=sorted(base.criterion.observed),dn_num_split=base.criterion.dn["dn_num_split"],
                        active=stats,zero_shift_loss=float(zero_stats["loss_gpc"]),tolerance=dict(atol=atol,rtol=rtol),
                        checkpoint_model_ema_reload="PASS",inference_export="native_contract_equal")

    def test_native_checkpoint_resume(self):
        from gpc_trainer import GPCTrainer
        from ultralytics.utils.torch_utils import ModelEMA
        from ultralytics.utils.patches import torch_load
        # Native save_model and resume_training are exercised on a real updated model/AdamW.
        with tempfile.TemporaryDirectory(prefix="gpc_resume_") as temporary:
            folder=Path(temporary)
            trainer=GPCTrainer.__new__(GPCTrainer)
            trainer.algorithm=asdict(GPCConfig())
            trainer.gpc_identity=dict(test="synthetic_native_checkpoint",run=str(folder))
            trainer.data=dict(nc=1,channels=3)
            trainer.resume=False
            trainer.save_dir=folder
            trainer.model=trainer.get_model(cfg=MODEL,verbose=False).to(DEVICE).train()
            trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=.0005)
            trainer.scaler=torch.cuda.amp.GradScaler(enabled=False)
            trainer.ema=ModelEMA(trainer.model)
            batch=sample(DEVICE)
            loss,_=trainer.model.loss(context(batch,6))
            loss.backward();trainer.optimizer.step();trainer.optimizer.zero_grad()
            trainer.ema.update(trainer.model)
            trainer.epoch=6;trainer.epochs=200;trainer.start_epoch=0
            trainer.best_fitness=trainer.fitness=.25
            trainer.metrics={"metrics/mAP50-95(B)":.25}
            trainer.args=SimpleNamespace(epochs=200,close_mosaic=10,model=MODEL)
            trainer.wdir=folder/"weights";trainer.last=trainer.wdir/"last.pt";trainer.best=trainer.wdir/"best.pt"
            trainer.save_period=-1;trainer.csv=folder/"results.csv"
            trainer.csv.write_text("epoch,metrics/mAP50-95(B)\n7,0.25\n")
            trainer.save_model()
            saved=torch_load(trainer.last,map_location=DEVICE)
            self.assertIsNotNone(saved["optimizer"])
            self.assertIn("scaler",saved)
            trainer.resume=True
            trainer.model=trainer.get_model(cfg=MODEL,weights=saved["ema"],verbose=False).to(DEVICE)
            trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=.0005)
            trainer.resume_training(saved)
            self.assertEqual(trainer.start_epoch,7)
            self.assertEqual(trainer.ema.updates,1)
            self.assertEqual(trainer.best_fitness,.25)
            self.assertEqual(trainer.scaler.state_dict(),saved["scaler"])
            self.assertGreater(len(trainer.optimizer.state),0)
            self.assertTrue(all(float(s["step"])==1 for s in trainer.optimizer.state.values()))
            trainer.model.train()
            continued=context(batch,trainer.start_epoch)
            next_loss,_=trainer.model.loss(continued)
            self.assertTrue(torch.isfinite(next_loss))
            self.assertEqual(continued["gpc_context"]["metrics"]["ramp"],2/15)
            with self.assertRaisesRegex(ValueError,"stripped"):
                trainer.resume_training(dict(saved,optimizer=None))
            EVIDENCE["native_resume"]=dict(epoch_restored=7,optimizer_steps_restored=1,ema_updates=1,scaler="native_state_restored",
                                           limitation="Parent epoch checkpoint uses half EMA/optimizer serialization; no exact microbatch replay claim")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_small_cuda_amp(self):
        if DEVICE!="cuda":self.skipTest("Separate small CUDA/AMP engineering check")
        model=GPCDetectionModel(MODEL,nc=1,verbose=False).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
        scaler=torch.cuda.amp.GradScaler(enabled=True)
        rows=[]
        batch=sample("cuda")
        for index in range(8):
            optimizer.zero_grad(set_to_none=True)
            before=float(scaler.get_scale())
            with torch.autocast("cuda"):
                loss,_=model.loss(context(batch,20))
            self.assertTrue(bool(torch.isfinite(loss)))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            old_steps=sum(float(s.get("step",0)) for s in optimizer.state.values())
            scaler.step(optimizer);scaler.update()
            updated=sum(float(s.get("step",0)) for s in optimizer.state.values())>old_steps
            rows.append(dict(scale_before=before,scale_after=float(scaler.get_scale()),finite_gradients=grad_finite,effective_update=updated,loss=float(loss.detach())))
            if updated:break
        self.assertTrue(any(row["effective_update"] for row in rows))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        EVIDENCE["small_cuda_amp"]=dict(rows=rows,main_B=2,aux_B=4,shape_hw=[160,192],formal_preflight=False)


def run(device="cpu", output=None, formulas_only=False):
    global DEVICE
    DEVICE=device
    EVIDENCE.clear()
    previous_threads=torch.get_num_threads()
    previous_autograd_threads=torch.autograd.is_multithreading_enabled()
    # Fix reduction scheduling for the exact-equality CPU reference experiment.
    torch.set_num_threads(1)
    torch.autograd.set_multithreading_enabled(False)
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(42,deterministic=True)
    start=time.monotonic()
    suite=unittest.TestLoader().loadTestsFromTestCase(FormulaTests)
    if not formulas_only:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ModelTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    torch.set_num_threads(previous_threads)
    torch.autograd.set_multithreading_enabled(previous_autograd_threads)
    report=dict(status="PASS" if result.wasSuccessful() else "FAIL",device=device,torch=str(torch.__version__),
                python=sys.version,cuda=torch.version.cuda,seconds=time.monotonic()-start,tests=result.testsRun,
                failures=[dict(test=str(test),trace=trace) for test,trace in result.failures+result.errors],
                skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],
                evidence=deepcopy(EVIDENCE),formal_B16_640_real_data_AMP="PENDING",AP_performance="NOT_TESTED")
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--formulas-only",action="store_true")
    args=parser.parse_args()
    sys.exit(0 if run(args.device,args.output,args.formulas_only)["status"]=="PASS" else 1)
