"""Executable CPU mathematical and native model/routing checks; no formal training/test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"ultralytics-main"))
os.environ["YOLO_AUTOINSTALL"] = "false"
import torch
from ultralytics.models.rtdetr.geo_loss import (GEODetectionLoss, raw_geo_xyxy, pair_overflow,
    envelope, intersection, ramp, GEOGeometryError)
from ultralytics.models.rtdetr.geo_model import GEOTrainer, GEODetectionModel, aggregate, write_json
from ultralytics.models.rtdetr.geo_val import postprocess
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.ops import xyxy2xywh


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


class MathChecks(unittest.TestCase):
    def test_fixed_example(self):
        b = tensor([[[0,.2,1.2,.8]]]).requires_grad_()
        gt = tensor([[0,0,1,1],[.8,0,1.8,1]]).requires_grad_()
        raw, stats = raw_geo_xyxy(b, gt, [2], [(torch.tensor([0]), torch.tensor([0]))], True)
        self.assertAlmostEqual(raw.item(), .2)
        torch.testing.assert_close(torch.autograd.grad(raw, b, retain_graph=True)[0], tensor([[[0,0,1,0]]]))
        weighted = .2*ramp(20)*raw
        self.assertAlmostEqual(weighted.item(), .04)
        weighted.backward()
        torch.testing.assert_close(b.grad, tensor([[[0,0,.2,0]]]))
        self.assertIsNone(gt.grad)
        self.assertEqual(stats["M"], 1)
        self.assertEqual([ramp(e) for e in (0,5,6,19,20,200)], [0,0,1/15,14/15,1,1])

    def test_zero_cases_and_contact(self):
        own = tensor([[0,0,1,1]])
        for box in ([0,0,1,1],[.1,.2,.9,.8],[0,.1,1,.9]):
            b = tensor([box]).requires_grad_()
            e = pair_overflow(b, own, tensor([[-.2,0,0,1],[1,0,2,1],[.1,.1,.9,.9],[-1,-1,2,2]]))
            self.assertEqual(e.sum().item(), 0)
            e.sum().backward()
            self.assertTrue(torch.equal(b.grad, torch.zeros_like(b)))
        b = tensor([[[-.2,0,1.2,1]]]).requires_grad_()
        for gt, groups, matches in ((own,[1],[(torch.tensor([0]),torch.tensor([0]))]),
            (torch.empty(0,4,dtype=torch.float64),[0],[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))])):
            raw, _ = raw_geo_xyxy(b, gt, groups, matches)
            self.assertEqual(raw.item(), 0)
            self.assertTrue(raw.requires_grad)
        empty = torch.empty(0,2,4, requires_grad=True)
        raw,_ = raw_geo_xyxy(empty, torch.empty(0,4), [], [])
        raw.backward()
        self.assertEqual(raw.item(), 0)

    def test_random_equivalence_bounds_signs(self):
        gen = torch.Generator().manual_seed(13)
        def boxes(n):
            low = torch.rand(n,2,generator=gen,dtype=torch.float64)*2-1
            return torch.cat((low, low+torch.rand(n,2,generator=gen,dtype=torch.float64)*1.3+.002),1)
        b, own, neighbor = boxes(83).requires_grad_(), boxes(83), boxes(97)
        e = pair_overflow(b, own, neighbor)
        h = envelope(b, own)
        denominator = torch.maximum((own[:,2:]-own[:,:2]).prod(1)[:,None], (neighbor[:,2:]-neighbor[:,:2]).prod(1)[None]).clamp_min(1e-9)
        difference = (intersection(h[:,None],neighbor[None])-intersection(own[:,None],neighbor[None]))/denominator
        torch.testing.assert_close(e,difference,atol=1e-13,rtol=1e-12)
        self.assertTrue(bool(((e>=-1e-12)&(e<=1+1e-12)).all()))
        e.sum().backward()
        inside = torch.cat((b[:,:2]>=own[:,:2], b[:,2:]<=own[:,2:]),1)
        self.assertTrue(bool((b.grad[inside]==0).all()))
        self.assertTrue(bool((b.grad[:,:2]<=0).all() and (b.grad[:,2:]>=0).all()))

    def test_finite_difference_narrow_and_platform(self):
        b=tensor([[-.17,.12,1.19,1.08]]).requires_grad_()
        own=tensor([[0,0,1,1]])
        neighbor=tensor([[-.3,-.4,.11,.79],[.83,.23,1.41,.97]])
        self.assertTrue(torch.autograd.gradcheck(lambda x: pair_overflow(x,own,neighbor), (b,), eps=1e-6))
        b=tensor([[0,0,2e-6,1]]).requires_grad_()
        self.assertTrue(bool(torch.isfinite(pair_overflow(b,tensor([[0,0,1e-6,1]]),tensor([[1e-6,0,2e-6,1]]))).all()))
        b=tensor([[-10,-10,10,10]]).requires_grad_()
        pair_overflow(b,own,tensor([[2,2,3,3]])).sum().backward()
        self.assertTrue(bool((b.grad==0).all()))

    def test_topk_ties_zero_neighbors_and_M(self):
        own=[0,0,1,1]
        gt=tensor([own,[-1,0,0,1],[-1,0,0,1],[1,0,2,1],[1,0,2,1]])
        b=tensor([[[-.25,0,1.25,1]]]).requires_grad_()
        matches=[(torch.tensor([0]),torch.tensor([0]))]
        raw,_=raw_geo_xyxy(b,gt,[5],matches)
        raw.backward()
        torch.testing.assert_close(b.grad,tensor([[[-2/3,0,1/3,0]]]))
        gt=tensor([own,[1,0,2,1],[3,3,4,4],[5,5,6,6]])
        b=tensor([[[0,0,1.3,1]]]).requires_grad_()
        raw,_=raw_geo_xyxy(b,gt,[4],matches)
        self.assertAlmostEqual(raw.item(),.1)
        raw2,_=raw_geo_xyxy(b,gt[:3],[3],matches)
        self.assertAlmostEqual(raw2.item(),.15)
        joined=torch.cat((b,tensor([[[0,0,1,1]]])),0)
        raw3,_=raw_geo_xyxy(joined,torch.cat((gt,tensor([own]))),[4,1],matches+[(torch.tensor([0]),torch.tensor([4]))])
        self.assertAlmostEqual(raw3.item(),.05)
        # Duplicate coordinates on another image never create a neighbor.
        no,_=raw_geo_xyxy(joined,tensor([own,own]),[1,1],[(torch.tensor([0]),torch.tensor([0])),(torch.tensor([0]),torch.tensor([1]))])
        self.assertEqual(no.item(),0)

    def test_invalid_gt_reported(self):
        with self.assertRaises(GEOGeometryError):
            raw_geo_xyxy(tensor([[[0,0,1,1]]]),tensor([[0,0,0,1]]),[1],[(torch.tensor([0]),torch.tensor([0]))])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; only a tiny local precision test")
    def test_cuda_autocast_geometry_is_fp32(self):
        device="cuda:0"
        boxes=(torch.rand(4,1,5,4,device=device,dtype=torch.float16)*.4+.2).requires_grad_()
        scores=torch.randn(4,1,5,1,device=device,dtype=torch.float16,requires_grad=True)
        gt=torch.tensor([[.4,.4,.5,.4],[.6,.5,.5,.4]],device=device)
        batch=dict(bboxes=gt,cls=torch.zeros(2,device=device,dtype=torch.long),gt_groups=[2])
        geo=GEODetectionLoss(nc=1,use_vfl=True); geo.epoch=20
        with torch.autocast(device_type="cuda",dtype=torch.float16): result=geo((boxes,scores),batch)
        self.assertEqual(result["loss_geo"].dtype,torch.float32)
        g=torch.autograd.grad(result["loss_geo"],(boxes,scores),allow_unused=True)
        self.assertTrue(bool(torch.isfinite(g[0]).all()))
        self.assertIsNone(g[1])

    def test_sorted_mask_and_input_unchanged(self):
        raw=torch.tensor([[[.5,.5,.1,.1,.0001],[.2,.2,.1,.1,.9],[.8,.8,.1,.1,.1]]])
        saved=raw.clone()
        out=postprocess(raw,640,.001)[0]
        self.assertEqual(len(out["conf"]),2)
        torch.testing.assert_close(out["conf"],torch.tensor([.9,.1]))
        self.assertTrue(torch.equal(raw,saved))
        self.assertEqual(len(postprocess(torch.empty(1,0,5),640,.001)[0]["conf"]),0)

    def test_loss_routing_with_and_without_DN(self):
        torch.manual_seed(47)
        gt=tensor([[0,0,1,1],[.8,0,1.8,1],[2,2,3,3],[0,0,1,1]]).float()
        target=dict(bboxes=xyxy2xywh(gt),cls=torch.zeros(4,dtype=torch.long),gt_groups=[3,1])
        boxes=(torch.rand(4,2,5,4)*.5+.3).requires_grad_()
        scores=torch.randn(4,2,5,1,requires_grad=True)
        db=(torch.rand(3,2,6,4)*.4+.3).requires_grad_()
        ds=torch.randn(3,2,6,1,requires_grad=True)
        meta=dict(dn_pos_idx=[torch.tensor([0,1,2,3,4,5]),torch.tensor([0,1])],dn_num_group=2)
        for dn in (None,meta):
            native=RTDETRDetectionLoss(nc=1,use_vfl=True)
            geo=GEODetectionLoss(nc=1,use_vfl=True); geo.epoch=20; geo.sample=True
            with patch.object(native.matcher,"forward",wraps=native.matcher.forward) as nm, patch.object(geo.matcher,"forward",wraps=geo.matcher.forward) as gm:
                args=dict(dn_bboxes=db,dn_scores=ds,dn_meta=dn)
                old=native((boxes,scores),target,**args); new=geo((boxes,scores),target,**args)
                self.assertEqual(nm.call_count,4); self.assertEqual(gm.call_count,4)
                for na,ga in zip(nm.call_args_list,gm.call_args_list):
                    self.assertTrue(torch.equal(na.args[0],ga.args[0]))
            self.assertEqual(set(new)-set(old),{"loss_geo"})
            for key in old: self.assertTrue(torch.equal(old[key],new[key]),key)
            grad=torch.autograd.grad(new["loss_geo"],(boxes,scores,db,ds),allow_unused=True,retain_graph=True)
            self.assertTrue(bool((grad[0][:-1]==0).all()))
            self.assertEqual(grad[1:],(None,None,None))
            golden_matches=native.matcher(boxes[-1],scores[-1],target["bboxes"],target["cls"],target["gt_groups"])
            for image,(src,_) in enumerate(golden_matches):
                unmatched=torch.ones(boxes.shape[2],dtype=torch.bool); unmatched[src]=False
                self.assertTrue(bool((grad[0][-1,image,unmatched]==0).all()))
            self.assertIsNone(geo._geo_context)
            for enabled,epoch in ((False,20),(True,5),(True,0)):
                geo.enabled,geo.epoch=enabled,epoch
                rng=torch.get_rng_state().clone()
                with patch("ultralytics.models.rtdetr.geo_loss.raw_geo_xyxy",side_effect=AssertionError("geometry must be skipped")):
                    zero=geo((boxes,scores),target,**args)
                self.assertEqual(set(zero),set(old))
                for key in old: self.assertTrue(torch.equal(old[key],zero[key]))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                a=torch.autograd.grad(sum(old.values()),(boxes,scores),retain_graph=True)
                z=torch.autograd.grad(sum(zero.values()),(boxes,scores),retain_graph=True)
                for x,y in zip(a,z): self.assertTrue(torch.equal(x,y))
        self.assertEqual((geo.vfl.alpha,geo.vfl.gamma,geo.matcher.alpha,geo.matcher.gamma),(.25,1.5,.25,2.0))
        self.assertEqual(geo.matcher.cost_gain,{"class":2,"bbox":5,"giou":2})
        with self.assertRaises(ValueError): GEODetectionLoss(nc=2)


def model_checks(source, folder):
    # CPU embedding scatter-add can reorder reductions across worker threads.
    # A single thread makes the identical native path bitwise comparable.
    torch.set_num_threads(1)
    from init_c19_lif_v1 import initialize, native_rebuild, source_contract, MODEL_DIR, CONFIGS
    from ultralytics.utils.patches import torch_load
    from ultralytics.nn.tasks import RTDETRDetectionModel
    init=folder/"public_init.pt"
    provenance=initialize(source,init)
    weights=torch_load(init,map_location="cpu")["model"]
    cfg=str(MODEL_DIR/CONFIGS["pair"])
    torch.manual_seed(42)
    rng=torch.get_rng_state().clone()
    mother=native_rebuild(cfg,weights)
    after=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    t=GEOTrainer.__new__(GEOTrainer); t.data=dict(nc=1,channels=3)
    model=t.get_model(cfg,weights,verbose=False)
    mother.nc=model.nc=1  # native Trainer.set_model_attributes normally sets this
    assert torch.equal(after,torch.get_rng_state())
    assert set(mother.state_dict())==set(model.state_dict())
    assert [(k,tuple(p.shape),p.numel()) for k,p in mother.named_parameters()]==[(k,tuple(p.shape),p.numel()) for k,p in model.named_parameters()]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in mother.state_dict().items())
    batch=dict(img=torch.rand(2,3,160,160),bboxes=torch.tensor([[.4,.4,.4,.4],[.55,.45,.4,.3],[.5,.5,.5,.4]]),
        cls=torch.zeros(3,1),batch_idx=torch.tensor([0,0,1]))
    mother.train(); model.train()
    rows=[]
    for enabled,epoch in ((False,20),(True,5)):
        model.geo_enabled,model.geo_epoch=enabled,epoch
        mother.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
        state=torch.get_rng_state().clone()
        a,ai=mother(batch); a.backward()
        torch.set_rng_state(state)
        b,bi=model(batch); b.backward()
        assert torch.equal(a,b) and torch.equal(ai,bi)
        for (name,p),(_,q) in zip(mother.named_parameters(),model.named_parameters()):
            assert (p.grad is None)==(q.grad is None),name
            if p.grad is not None: assert torch.equal(p.grad,q.grad),name
        rows.append(dict(enabled=enabled,epoch=epoch,loss=float(b.detach()),forward_loss_gradients_exact=True))
    model.geo_enabled=True; model.geo_epoch=20; model.geo_sample=True
    captured={}
    hook=model.criterion.register_forward_hook(lambda m,a,o: captured.update(o))
    loss,items=model(batch)
    assert "loss_geo" in captured and torch.equal(loss,sum(captured.values()))
    assert not any("geo" in k for k in captured if k!="loss_geo")
    hook.remove(); captured.clear()
    loss.backward()
    model.eval(); mother.eval()
    unfused_parameters=sum(p.numel() for p in model.parameters())
    unfused_states=len(model.state_dict())
    # BN statistics followed identical DN forwards except the final GEO probe; sync weights for inference parity.
    mother.load_state_dict(model.state_dict())
    with torch.no_grad():
        x=batch["img"]; a=mother(x)[0]; b=model(x)[0]
        assert torch.equal(a,b)
        mother.fuse(verbose=False); model.fuse(verbose=False)
        af=mother(x)[0]; bf=model(x)[0]
        assert torch.equal(af,bf)
        torch.testing.assert_close(bf,b,atol=2e-5,rtol=2e-4)
    model.geo_sample=False
    checkpoint=folder/"model_checkpoint.pt"
    torch.save(dict(model=model,epoch=19),checkpoint)
    return dict(status="PASS",source=provenance,module_hashes=source_contract(),parameters=unfused_parameters,
        states=unfused_states,fused_parameters=sum(p.numel() for p in model.parameters()),rng_exact=True,
        disabled_and_zero=rows,inference_exact=True,fused_inference_exact=True,optimized_total_contains_geo=True,
        checkpoint=str(checkpoint),scope="CPU B2/160 native reconstruction and real random-DN forward, synthetic images; not server B16")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-only",action="store_true")
    parser.add_argument("--source",type=Path,default=Path("D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt") if os.name=="nt" else Path("/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"))
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/geo_v1/local_checks")
    parser.add_argument("--reload",type=Path)
    args=parser.parse_args(); torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.reload:
        from ultralytics.utils.patches import torch_load
        model=torch_load(args.reload,map_location="cpu")["model"]
        assert type(model) is GEODetectionModel and isinstance(model.criterion,GEODetectionLoss)
        with torch.no_grad(): assert torch.isfinite(model(torch.zeros(1,3,160,160))[0]).all()
        write_json(args.output/"reload.json",dict(status="PASS",model=inspect.getfile(type(model)),criterion=inspect.getfile(type(model.criterion))))
        return
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MathChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status="PASS" if result.wasSuccessful() else "FAIL",math_tests=result.testsRun,
        torch=torch.__version__,python=sys.executable,criterion=inspect.getfile(GEODetectionLoss))
    if result.wasSuccessful() and not args.math_only:
        report["model"]=model_checks(args.source,args.output)
    write_json(args.output/"checks.json",report)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=="__main__": main()
