"""Focused QCC formula, native loss routing and real model equivalence checks."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from qcc_v1_common import ROOT, OUT, SOURCE, MODEL, runtime, write_json
import torch
from ultralytics.models.rtdetr import qcc_loss
from ultralytics.models.rtdetr.qcc_loss import QCCDetectionLoss, competition, select_groups, group_kl, ramp
from ultralytics.models.rtdetr.qcc_model import QCCTrainer, QCCDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.qcc_io import aggregate
from ultralytics.models.rtdetr.qcc_val import postprocess
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou


def small():
    gt = torch.tensor([[.3,.3,.2,.2], [.7,.7,.2,.2], [.3,.3,.2,.2]])
    boxes = torch.tensor([[[.3,.3,.2,.2], [.7,.7,.2,.2], [.32,.3,.2,.2], [.33,.3,.2,.2],
                           [.34,.3,.2,.2], [.35,.3,.2,.2], [.9,.1,.1,.1]],
                          [[.3,.3,.2,.2], [.32,.3,.2,.2], [.33,.3,.2,.2], [.9,.9,.1,.1],
                           [.9,.9,.1,.1], [.9,.9,.1,.1], [.9,.9,.1,.1]]])
    z = torch.zeros(2,7,1)
    matches = [(torch.tensor([0,1]),torch.tensor([0,1])), (torch.tensor([0]),torch.tensor([2]))]
    q = torch.zeros(2,7); q[0,:2] = .99; q[1,0] = .99
    return boxes, z, gt, [2,1], matches, q


class FormulaTests(unittest.TestCase):
    def test_reference_and_cap(self):
        # Independent Python scalar softmax + explicit KL, not the implementation expression.
        q, h, a = .82, .49, [.7,.4]
        odds = [.55/.45, .7/.3*.7, .4/.6*.4, 1.0]
        probs = [v/sum(odds) for v in odds]
        expected = q*math.log(q/probs[0]) + (1-q)*math.log((1-q)/probs[-1])
        z = torch.logit(torch.tensor([.55,.7,.4], dtype=torch.double)).requires_grad_()
        loss, group = group_kl(z, torch.tensor(q,dtype=torch.double), torch.tensor(a,dtype=torch.double))
        self.assertAlmostEqual(float(loss), expected, places=12)
        self.assertAlmostEqual(float(loss), .7804489352, places=9)
        grad, = torch.autograd.grad(loss*h,z)
        torch.testing.assert_close(grad, torch.tensor([-.2565169811,.1941509434,.0316981132],dtype=torch.double), atol=1e-10, rtol=0)
        z = torch.logit(torch.tensor([.90,.7,.4],dtype=torch.double)).requires_grad_()
        loss,_ = group_kl(z,torch.tensor(q,dtype=torch.double),torch.tensor(a,dtype=torch.double))
        self.assertEqual(float(torch.autograd.grad(loss,z)[0][0]),0)

    def test_gradcheck_frozen_selection_and_weights(self):
        for p in (.2,.9):
            z = torch.logit(torch.tensor([p,.7,.4],dtype=torch.double)).requires_grad_()
            q = torch.tensor(.82,dtype=torch.double,requires_grad=True)
            a = torch.tensor([.7,.4],dtype=torch.double,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda v: group_kl(v,q,a)[0]*.49,(z,)))
            loss, logits = group_kl(z,q,a)
            grad = torch.autograd.grad(loss,(z,q,a),allow_unused=True)
            self.assertIsNone(grad[1]); self.assertIsNone(grad[2])
            self.assertLessEqual(float(grad[0][0]),0)
            self.assertTrue((grad[0][1:] >= 0).all())
            self.assertAlmostEqual(float(logits.softmax(0).sum()),1)
            perm = torch.tensor([0,2,1])
            torch.testing.assert_close(group_kl(z[perm],q,a.flip(0))[0],loss)

    def test_equality_q_one_extremes(self):
        for q in (.5, 1.):
            z = torch.tensor([0., 1000., -1000.],requires_grad=True)
            l,_=group_kl(z,torch.tensor(q),torch.tensor([1e-35,.4]))
            l.backward()
            self.assertTrue(torch.isfinite(l) and torch.isfinite(z.grad).all())
            if q == .5: self.assertEqual(float(z.grad[0]),0)
        for pos in (-1000.,1000.):
            z=torch.tensor([pos,1000.,-1000.],requires_grad=True)
            l,_=group_kl(z,torch.tensor(.82),torch.tensor([1e-35,.1])); l.backward()
            self.assertTrue(torch.isfinite(z.grad).all()); self.assertGreaterEqual(float(l),-1e-5)

    def test_ownership_isolation_exclusion_and_k(self):
        boxes,z,gt,sizes,matches,q=small()
        groups,counts=select_groups(boxes,z,gt,sizes,matches,q)
        self.assertEqual([(x['batch'],int(x['positive'])) for x in groups],[(0,0),(1,0)])
        self.assertEqual(groups[0]['competitors'].tolist(),[2,3,4])
        self.assertEqual(groups[1]['competitors'].tolist(),[1,2])
        self.assertEqual(counts['no_candidates'],1)
        for b in range(2):
            selected=[int(c) for x in groups if x['batch']==b for c in x['competitors']]
            self.assertEqual(len(selected),len(set(selected)))
            self.assertFalse(set(selected)&set(matches[b][0].tolist()))
        q[0,0]=.5
        groups,counts=select_groups(boxes,z,gt,sizes,matches,q)
        self.assertEqual(counts['excluded_better'],4)
        self.assertFalse(any(g['batch']==0 for g in groups))

    def test_ties_and_unmatched_gt_owner(self):
        boxes,z,gt,sizes,matches,q=small()
        boxes[0,2:6]=boxes[0,2]
        groups,_=select_groups(boxes,z,gt,sizes,matches,q)
        self.assertEqual(groups[0]['competitors'].tolist(),[2,3,4])
        gt[1]=gt[0]
        groups,_=select_groups(boxes,z,gt,sizes,matches,q)
        self.assertFalse(any(g['batch']==0 for g in groups))  # all top-IoU ties have a=0
        gt[1]=boxes[0,2]
        matches[0]=(torch.tensor([0]),torch.tensor([0]))  # unmatched GT still wins ownership
        groups,_=select_groups(boxes,z,gt,sizes,matches,q)
        self.assertFalse(any(g['batch']==0 for g in groups))

    def test_empty_paths_and_geometry_detach(self):
        boxes,z,gt,sizes,matches,q=small()
        boxes.requires_grad_(); z.requires_grad_(); gt.requires_grad_(); q.requires_grad_()
        raw,stats=competition(boxes,z,gt,sizes,matches,q,True)
        grads=torch.autograd.grad(raw,(boxes,z,gt,q),allow_unused=True)
        self.assertIsNone(grads[0]); self.assertIsNone(grads[2]); self.assertIsNone(grads[3])
        self.assertGreater(float(grads[1][0,2,0]),0)
        self.assertLess(float(grads[1][0,0,0]),0)
        self.assertEqual(stats['M'],3)
        for which in ('q0','no_gt','no_unmatched','no_overlap'):
            b,v,g,n,match,qual=small(); v.requires_grad_()
            if which=='q0': qual.zero_()
            if which=='no_gt':
                g=g[:0]; n=[0,0]; match=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]*2
            if which=='no_unmatched':
                b=b[:,:1]; v=v[:,:1].detach().requires_grad_(); g=g[:2]; n=[1,1]
                match=[(torch.tensor([0]),torch.tensor([i])) for i in range(2)]; qual=qual[:,:1]
            if which=='no_overlap': b[:,:,0]=.99; b[:,:,2:]=.001
            loss,s=competition(b,v,g,n,match,qual,True); loss.backward()
            self.assertEqual(float(loss),0); self.assertTrue(torch.isfinite(v.grad).all())

    def test_native_iou_epsilon_and_bad_values(self):
        boxes,z,gt,sizes,matches,q=small()
        boxes[:,:,2:]=1e-5; gt[:,2:]=1e-5
        raw,stat=competition(boxes,z,gt,sizes,matches,q,True)
        self.assertTrue(torch.isfinite(raw))
        for bad in (float('nan'),float('inf'),1.1,-.1):
            q[0,0]=bad
            with self.assertRaises(ValueError): competition(boxes,z,gt,sizes,matches,q)
        with self.assertRaises(NotImplementedError): QCCDetectionLoss(nc=2)
        self.assertEqual([ramp(e) for e in (0,5,6,20,99)],[0,0,1/15,1,1])

    def test_fp32_autocast_and_total_match_normalization(self):
        b,z,g,n,match,q=small()
        groups,_=select_groups(b,z,g,n,match,q)
        expected=0.0
        for item in groups:
            pos=item['positive']; comp=item['competitors']; bi=item['batch']
            logits=torch.cat((z[bi,pos,0].reshape(1),z[bi,comp,0]))
            expected+=float(item['h']*group_kl(logits,item['q'],item['a'])[0])/3
        z=z.to(torch.bfloat16).requires_grad_()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            raw,stats=competition(b,z,g,n,match,q,True)
        self.assertEqual(raw.dtype,torch.float32)
        self.assertAlmostEqual(float(raw),expected,places=7)
        self.assertEqual(stats['M'],3); self.assertEqual(stats['active_groups'],2)
        raw.backward(); self.assertTrue(torch.isfinite(z.grad).all())

    def test_log_aggregation_and_postprocess(self):
        a=aggregate([dict(M=1,p_lt_q=1,moments={'q':dict(count=1,sum=1,sumsq=1,min=1,max=1)}),
                     dict(M=3,p_ge_q=3,moments={'q':dict(count=3,sum=0,sumsq=0,min=0,max=0)})])
        self.assertEqual(a['moments']['q']['mean'],.25)
        self.assertEqual(a['p_lt_q_fraction'],.25)
        raw=torch.tensor([[[.1,.1,.1,.1,.1],[.3,.3,.1,.1,.9],[.7,.7,.1,.1,.2]]])
        before=raw.clone(); output=postprocess(raw,640,.15)
        self.assertTrue(torch.equal(raw,before))
        torch.testing.assert_close(output[0]['conf'],torch.tensor([.9,.2]))


class RoutingTests(unittest.TestCase):
    def test_l0_parity_matches_and_direct_gradients(self):
        torch.manual_seed(42)
        b,z,g,n,match,q=small()
        boxes=b.unsqueeze(0).repeat(4,1,1,1).requires_grad_()
        logits=(z.unsqueeze(0).repeat(4,1,1,1)+torch.randn(4,2,7,1)*.05).requires_grad_()
        batch=dict(bboxes=g,cls=torch.zeros(3,dtype=torch.long),gt_groups=n)
        db=b[:,:4].unsqueeze(0).repeat(3,1,1,1).clone().requires_grad_()
        dz=torch.zeros(3,2,4,1,requires_grad=True)
        for dn in (False,True):
            kwargs=dict(dn_bboxes=db,dn_scores=dz,dn_meta=dict(dn_pos_idx=[torch.tensor([0,1]),torch.tensor([0])],dn_num_group=1)) if dn else {}
            native=RTDETRDetectionLoss(nc=1,use_vfl=True)
            with patch.object(native.matcher,'forward',wraps=native.matcher.forward) as counter:
                l0=native((boxes,logits),batch,**kwargs)
                count=counter.call_count
            for epoch,enabled in ((0,True),(20,False),(20,True)):
                criterion=QCCDetectionLoss(nc=1,use_vfl=True); criterion.epoch=epoch; criterion.enabled=enabled; criterion.sample=True
                state=torch.get_rng_state()
                with patch.object(criterion.matcher,'forward',wraps=criterion.matcher.forward) as counter:
                    losses=criterion((boxes,logits),batch,**kwargs)
                self.assertEqual(counter.call_count,count)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
                self.assertIsNone(criterion._captured); self.assertIsNone(criterion._native_q)
                self.assertTrue(all(torch.equal(l0[k],losses[k]) for k in l0))
                if epoch==20 and enabled:
                    self.assertEqual(set(losses)-set(l0),{'loss_qcc'})
                    grads=torch.autograd.grad(losses['loss_qcc'],(boxes,logits,db,dz),retain_graph=True,allow_unused=True)
                    self.assertIsNone(grads[0]); self.assertIsNone(grads[2]); self.assertIsNone(grads[3])
                    self.assertEqual(int(torch.count_nonzero(grads[1][:-1])),0)
                    self.assertGreater(float(grads[1][-1].abs().sum()),0)
                else:
                    self.assertEqual(set(losses),set(l0))
                    for x,y in zip(torch.autograd.grad(sum(losses.values()),(boxes,logits),retain_graph=True),
                                   torch.autograd.grad(sum(l0.values()),(boxes,logits),retain_graph=True)):
                        self.assertTrue(torch.equal(x,y))

    def test_native_q_capture_full_queries_and_disabled_fast_path(self):
        b,z,g,n,match,q=small()
        batch=dict(bboxes=g,cls=torch.zeros(3,dtype=torch.long),gt_groups=n)
        criterion=QCCDetectionLoss(nc=1,use_vfl=True); criterion.epoch=20
        original=qcc_loss.competition
        captured=[]
        def observed(*args,**kwargs):
            captured.append(args)
            return original(*args,**kwargs)
        with patch.object(qcc_loss,'competition',side_effect=observed):
            criterion((b[None],z[None]),batch)
        full,logits,gt,sizes,matches,quality=captured[0]
        self.assertEqual(full.shape,(2,7,4))
        idx,dst=criterion._get_index(matches)
        expected=bbox_iou(full[idx].detach(),gt[dst],xywh=True).squeeze(-1)
        self.assertTrue(torch.equal(quality[idx],expected))
        criterion.epoch=5
        with patch.object(qcc_loss,'select_groups',side_effect=AssertionError('extra grouping')):
            criterion((b[None],z[None]),batch)


class ModelTests(unittest.TestCase):
    def test_native_rebuild_real_forward_rng_gradients_export_and_serialization(self):
        torch.set_num_threads(1)  # deterministic CPU embedding/scatter accumulation for bitwise gradient audit
        from init_c19_lif_v1 import initialize, build_training_model, source_contract
        from ultralytics.utils.patches import torch_load
        source_contract()
        init_path=OUTPUT/'public_init.pt'
        if not init_path.exists():
            write_json(OUTPUT/'initialization.json',initialize(SOURCE,init_path))
        weights=torch_load(init_path,map_location='cpu')['model']
        data=dict(nc=1,channels=3)
        trainer=QCCTrainer.__new__(QCCTrainer); trainer.data=data
        torch.manual_seed(42); state=torch.get_rng_state()
        native,mapping=build_training_model(str(MODEL),weights,data)
        torch.set_rng_state(state)
        model=trainer.get_model(str(MODEL),weights,verbose=False)
        # Native _setup_train normally calls set_model_attributes after get_model.
        native.nc=model.nc=1
        self.assertIs(type(model),QCCDetectionModel)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in native.state_dict().items()))
        self.assertEqual({k:tuple(v.shape) for k,v in native.named_parameters()}, {k:tuple(v.shape) for k,v in model.named_parameters()})
        model.train(); native.train()
        batch=dict(img=torch.rand(2,3,160,160),bboxes=torch.tensor([[.4,.4,.25,.15],[.6,.6,.2,.3]]),
                   cls=torch.zeros(2,1),batch_idx=torch.tensor([0,1]))
        state=torch.get_rng_state()
        l0,items0=native(batch); l0.backward()
        torch.set_rng_state(state)
        model.qcc_epoch=5
        loss,items=model(batch); loss.backward()
        self.assertTrue(torch.equal(loss,l0) and torch.equal(items,items0))
        for (name,p),(other,p0) in zip(model.named_parameters(),native.named_parameters()):
            self.assertEqual(name,other)
            self.assertTrue((p.grad is None and p0.grad is None) or torch.equal(p.grad,p0.grad),name)
        left=RTDETRTrainer.build_optimizer(trainer,native,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
        right=RTDETRTrainer.build_optimizer(trainer,model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
        left.step(); right.step()
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in native.state_dict().items()))
        native.zero_grad(); model.zero_grad()
        model.qcc_epoch=20; model.qcc_sample=True
        # Capture actual head output and actual QCC input around the native DN split.
        head=[]; observed=[]
        hook=model.model[-1].register_forward_hook(lambda m,a,o: head.append(o))
        orig=qcc_loss.competition
        def capture(*args,**kwargs):
            observed.append(args[0].detach().clone())
            return orig(*args,**kwargs)
        with patch.object(qcc_loss,'competition',side_effect=capture):
            loss,_=model(batch)
        hook.remove()
        dec,_,_,_,dn=head[0]
        ordinary=dec[-1,:,dn['dn_num_split'][0]:] if dn else dec[-1]
        self.assertTrue(torch.equal(observed[0],ordinary))
        loss.backward()
        self.assertTrue(torch.isfinite(model.model[-1].cbr.offset_out.weight.grad).all())
        head.clear(); observed.clear()
        model.eval(); native.load_state_dict(model.state_dict()); native.eval()
        with torch.no_grad():
            prediction=model(batch['img'])[0]
            self.assertTrue(torch.equal(prediction,native(batch['img'])[0]))
            model.qcc_enabled=False
            self.assertTrue(torch.equal(prediction,model(batch['img'])[0]))
            model.model[-1].export=native.model[-1].export=True
            self.assertTrue(torch.equal(model(batch['img']),native(batch['img'])))
            model.model[-1].export=native.model[-1].export=False
        self.assertEqual(set(model.state_dict()),set(native.state_dict()))
        model.args=deepcopy(weights.args); model.args.update(model=str(MODEL),data=str(ROOT/'configs/crack.yaml'))
        checkpoint=OUTPUT/'qcc_serialization.pt'
        torch.save(dict(model=model,ema=None,epoch=-1,train_args=model.args),checkpoint)
        reloaded=torch_load(checkpoint,map_location='cpu')['model']
        self.assertIs(type(reloaded),QCCDetectionModel)
        with torch.no_grad(): self.assertTrue(torch.equal(prediction,reloaded(batch['img'])[0]))
        # Actual native serializer, nonempty optimizer state, disabled CPU scaler.
        # CUDA scaler/full native setup restoration is a separate server gate.
        from types import SimpleNamespace
        from ultralytics.utils.torch_utils import ModelEMA
        trainer.model=model; trainer.optimizer=right; trainer.ema=ModelEMA(model); trainer.ema.updates=1
        trainer.scaler=torch.amp.GradScaler('cuda',enabled=False)
        trainer.epoch=19; trainer.best_fitness=trainer.fitness=.1; trainer.metrics={}
        trainer.args=SimpleNamespace(**dict(model.args,epochs=200,close_mosaic=10))
        trainer.wdir=OUTPUT/'native_save'; trainer.last=trainer.wdir/'last.pt'; trainer.best=trainer.wdir/'best.pt'
        trainer.csv=OUTPUT/'diagnostic_results.csv'; trainer.save_period=-1
        trainer.save_model()
        write_json(OUTPUT/'model_evidence.json',dict(status='PASS',mapping=mapping,rebuild=trainer.qcc_rebuild,
            real_forward='B2/160 CPU, restored RNG; exact original loss and all parameter gradients',
            cbr_final_routing=True,state_keys=len(model.state_dict()),serialization=str(checkpoint),
            native_optimizer_step_equal=True,native_saved_checkpoint=str(trainer.last),
            inference_equal=True,export_mode_equal=True,formal_capacity='PENDING'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--math-only',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT/'local_checks')
    args=parser.parse_args(); OUTPUT=args.output; OUTPUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    suite=unittest.TestSuite()
    for cls in ([FormulaTests,RoutingTests] if args.math_only else [FormulaTests,RoutingTests,ModelTests]):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(OUTPUT/'checks.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=[str(x) for x in result.failures+result.errors],scope='formula/routing'+('' if args.math_only else '/native model'),runtime=runtime()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
