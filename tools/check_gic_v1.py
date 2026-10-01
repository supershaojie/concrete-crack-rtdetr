"""Bounded GIC mathematical, scope, assembly and checkpoint tests (stdlib unittest)."""
from __future__ import annotations
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
import torch.nn.functional as F
from ultralytics.models.rtdetr.gic_loss import (CONFIG, GICDetectionLoss, eta_at_epoch, entropy,
                                              shifted_boxes, geometry_interval, positive_delta)
from ultralytics.models.utils.loss import DETRLoss
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.metrics import bbox_iou

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
source = subprocess.check_output(["git", "show", BASE + ":ultralytics-main/ultralytics/models/utils/loss.py"], cwd=ROOT).decode()
mother_module = types.ModuleType("ultralytics.models.utils.gic_test_mother")
exec(compile(source, "mother_loss_at_" + BASE, "exec"), mother_module.__dict__)
Mother = mother_module.RTDETRDetectionLoss


def fixture(empty=False):
    torch.manual_seed(42)
    gt = torch.tensor([[.40,.42,.012,.10], [.73,.67,.11,.012], [.25,.30,.017,.12]])
    boxes = torch.rand(4,2,6,4) * .3 + .2
    boxes[:,0,:2] = gt[:2]
    boxes[:,1,0] = gt[2]
    boxes[...,0] += .001
    scores = torch.randn(4,2,6,3)
    batch = dict(bboxes=gt[:0] if empty else gt, cls=torch.tensor([],dtype=torch.long) if empty else torch.tensor([0,2,1]),
                 gt_groups=[0,0] if empty else [2,1])
    dn_boxes = torch.rand(3,2,4,4)*.4+.1
    dn_scores = torch.randn(3,2,4,3)
    dn_meta = None if empty else dict(dn_pos_idx=[torch.tensor([0,1]),torch.tensor([0])],dn_num_group=1)
    return (boxes,scores,dn_boxes,dn_scores),batch,dn_meta


def run_loss(criterion, values, batch, meta):
    leaves = [x.clone().requires_grad_() for x in values]
    loss = criterion(tuple(leaves[:2]),batch,*leaves[2:],meta)
    gradients = torch.autograd.grad(sum(loss.values()),leaves,allow_unused=True)
    return loss,gradients,leaves


class GICTests(unittest.TestCase):
    def test_ramp_and_context(self):
        self.assertEqual([eta_at_epoch(e) for e in [0,4,5,19,20]],[0,0,.5/15,.5,.5])
        with self.assertRaises(ValueError): eta_at_epoch(-1)
        with self.assertRaises(ValueError): GICDetectionLoss(use_vfl=True,config=dict(CONFIG,shift_px=2))

    def test_zero_exact_mother_loss_grad_rng(self):
        values,batch,meta = fixture()
        mother=Mother(nc=3,use_vfl=True)
        for e in (0,4,20):
            gic=GICDetectionLoss(nc=3,use_vfl=True,config=dict(CONFIG,eta_max=0 if e==20 else .5))
            gic.set_context(e,(320,640),True)
            a,ga,_=run_loss(mother,values,batch,meta)
            rng=torch.get_rng_state().clone()
            with patch("ultralytics.models.rtdetr.gic_loss.geometry_interval",side_effect=AssertionError("zero path used interval")):
                b,gb,_=run_loss(gic,values,batch,meta)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(set(a),set(b))
            for k in a: self.assertTrue(torch.equal(a[k],b[k]),k)
            for x,y in zip(ga,gb): self.assertTrue(torch.equal(x,y))

    def test_rectangular_offsets_edges_iou(self):
        boxes=torch.tensor([[0.,0.,.02,.04],[1.,1.,.02,.04],[.5,.5,1e-6,1e-6]])
        shifted=shifted_boxes(boxes,(320,640))
        self.assertEqual(tuple(shifted.shape),(3,9,4))
        self.assertTrue(torch.equal(shifted[:,4],boxes))
        self.assertAlmostEqual(float(shifted[0,0,0]),-1/640)
        self.assertAlmostEqual(float(shifted[0,0,1]),-1/320)
        self.assertTrue((shifted[1,8,:2]>1).all())
        self.assertTrue(torch.equal(shifted[...,2:],boxes[:,None,2:].expand(-1,9,-1)))
        q=bbox_iou(boxes,boxes,xywh=True).squeeze(-1)
        lo,hi=geometry_interval(boxes.requires_grad_(),boxes,q,(320,640))
        self.assertTrue(((lo<=q)&(q<=hi)).all())
        self.assertFalse(lo.requires_grad or hi.requires_grad)
        # Explicit q must remain in the interval even if precision differs.
        _,hi=geometry_interval(boxes,boxes,torch.ones(3),(320,640))
        self.assertTrue(torch.equal(hi,torch.ones(3)))
        with self.assertRaises(ValueError): geometry_interval(boxes*torch.tensor([1,1,-1,1]),boxes,q,(320,640))

    def test_formula_gradient_and_finite_difference(self):
        torch.manual_seed(9)
        q=torch.cat((torch.rand(512),torch.tensor([0,1,0,.25,.5,.75,1,.5,.5])))
        lo=q*torch.rand_like(q); hi=q+(1-q)*torch.rand_like(q)
        lo[-4:]=torch.tensor([.75,1,0,.5]);hi[-4:]=torch.tensor([.75,1,1,.5])
        z=torch.cat((torch.randn(512)*6,torch.tensor([-80.,80.,80.,-80.,0.,1.1,80.,0.,0.]))).requires_grad_()
        delta,t=positive_delta(z,q,lo,hi,.5)
        old=q*F.binary_cross_entropy_with_logits(z,q,reduction="none")
        new=old+delta
        direct=q*(entropy(q)+.5*(F.binary_cross_entropy_with_logits(z,q,reduction="none")-entropy(q))+
                  .5*(F.binary_cross_entropy_with_logits(z,t,reduction="none")-entropy(t)))
        torch.testing.assert_close(new,direct,atol=1e-5,rtol=2e-6)
        g=torch.autograd.grad(new.sum(),z)[0]
        p=z.detach().sigmoid();gold=q*(p-q);analytic=q*(.5*(p-q)+.5*(p-t))
        torch.testing.assert_close(g,analytic,atol=8e-8,rtol=1e-5)
        self.assertTrue((g*gold>=-1e-9).all())
        self.assertTrue((g.abs()>=.5*gold.abs()-1e-7).all())
        self.assertTrue((g.abs()<=gold.abs()+1e-7).all())
        self.assertFalse(t.requires_grad)
        self.assertTrue(torch.isfinite(new).all())
        self.assertTrue((delta<0).any())
        # Double-precision reference finite differences through the envelope.
        qd,ld,hd,zd=[x.detach().double() for x in (q,lo,hi,z)]
        def reference(v):
            td=torch.minimum(torch.maximum(v.sigmoid(),ld),hd)
            return qd*(entropy(qd)+.5*(F.binary_cross_entropy_with_logits(v,qd,reduction="none")-entropy(qd))+
                       .5*(F.binary_cross_entropy_with_logits(v,td,reduction="none")-entropy(td)))
        numeric=(reference(zd+1e-4)-reference(zd-1e-4))/2e-4
        torch.testing.assert_close(g.double(),numeric,atol=1e-7,rtol=1e-4)

    def test_interval_in_out_boundary_singleton(self):
        q=torch.tensor([.5]*5);lo=torch.tensor([.3]*5);hi=torch.tensor([.7]*5)
        p=torch.tensor([.1,.3,.5,.7,.9]);z=p.logit().requires_grad_()
        d,t=positive_delta(z,q,lo,hi,.5)
        torch.testing.assert_close(t,torch.tensor([.3,.3,.5,.7,.7]))
        d,t=positive_delta(z,q,q,q,.5)
        self.assertTrue(torch.equal(d,torch.zeros_like(d)))
        self.assertTrue(torch.equal(torch.autograd.grad(d.sum(),z)[0],torch.zeros_like(z)))

    def test_active_scope_no_extra_matching_and_bbox_grad(self):
        values,batch,meta=fixture()
        gains=dict(class_=2.7,bbox=5,giou=2);gains['class']=gains.pop('class_')
        mother=Mother(nc=3,use_vfl=True,loss_gain=gains)
        gic=GICDetectionLoss(nc=3,use_vfl=True,loss_gain=gains);gic.set_context(19,(320,640),True)
        with patch.object(mother.matcher,"forward",wraps=mother.matcher.forward) as m:
            a,ga,_=run_loss(mother,values,batch,meta); count=m.call_count
        with patch.object(gic.matcher,"forward",wraps=gic.matcher.forward) as m:
            b,gb,leaves=run_loss(gic,values,batch,meta);self.assertEqual(m.call_count,count)
        self.assertEqual(set(a),set(b));self.assertNotEqual(float(a['loss_class']),float(b['loss_class']))
        for k in a:
            if k!='loss_class': self.assertTrue(torch.equal(a[k],b[k]),k)
        for i in (0,2,3): self.assertTrue(torch.equal(ga[i],gb[i]),str(i))
        self.assertTrue(torch.equal(ga[1][:-1],gb[1][:-1]))
        indices=mother.matcher(values[0][-1],values[1][-1],batch['bboxes'],batch['cls'],batch['gt_groups'])
        idx,gtidx=mother._get_index(indices)
        mask=torch.ones_like(ga[1],dtype=torch.bool);mask[-1,idx[0],idx[1],batch['cls'][gtidx]]=False
        self.assertTrue(torch.equal(ga[1][mask],gb[1][mask]))
        torch.testing.assert_close(sum(b.values())-sum(a.values()),torch.tensor(gic.diagnostics['delta_reduced']),atol=3e-6,rtol=1e-4)
        # Independent delta has NO bbox gradient and a nonzero z gradient.
        boxes=values[0][-1].clone().requires_grad_();z=values[1][-1].clone().requires_grad_()
        q=bbox_iou(boxes[idx].detach(),batch['bboxes'][gtidx],xywh=True).squeeze(-1)
        lo,hi=geometry_interval(boxes[idx],batch['bboxes'][gtidx],q,(320,640))
        d,t=positive_delta(z[idx[0],idx[1],batch['cls'][gtidx]],q,lo,hi,.5)
        bz,zz=torch.autograd.grad(d.sum(),(boxes,z),allow_unused=True)
        self.assertIsNone(bz);self.assertTrue((zz!=0).any())

    def test_empty_and_aux_scope_empty_postfix(self):
        values,batch,meta=fixture(True)
        m=Mother(nc=3,use_vfl=True);g=GICDetectionLoss(nc=3,use_vfl=True);g.set_context(20,(320,640))
        a,ga,_=run_loss(m,values,batch,meta);b,gb,_=run_loss(g,values,batch,meta)
        for k in a:self.assertTrue(torch.equal(a[k],b[k]))
        values,batch,_=fixture()
        m.device=g.device=torch.device('cpu')
        with patch.object(g,"_adjust_final_normal_class",side_effect=AssertionError("aux leaked GIC")):
            b=g._get_loss_aux(*values[:2],batch['bboxes'],batch['cls'],batch['gt_groups'],postfix='')
        a=m._get_loss_aux(*values[:2],batch['bboxes'],batch['cls'],batch['gt_groups'],postfix='')
        for k in a:self.assertTrue(torch.equal(a[k],b[k]))

    def test_model_assembly_inference_serialization(self):
        model=RTDETRDetectionModel(str(ROOT/'ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml'),nc=3,verbose=False)
        model.nc=3
        baseline=deepcopy(model)
        model.gic_config=deepcopy(CONFIG);model.gic_epoch=19;model.gic_collect_diagnostics=True
        values,targets,meta=fixture()
        values=[x.clone().requires_grad_() for x in values]
        b,s,db,ds=values
        meta['dn_num_split']=[4,6]
        preds=(torch.cat((db,b[1:]),dim=2),torch.cat((ds,s[1:]),dim=2),b[0],s[0],meta)
        batch=dict(img=torch.zeros(2,3,320,640),batch_idx=torch.tensor([0,0,1]),bboxes=targets['bboxes'],cls=targets['cls'][:,None])
        l0,_=baseline.loss(batch,preds);l1,items=model.loss(batch,preds)
        torch.testing.assert_close(l1-l0,torch.tensor(model.criterion.diagnostics['delta_reduced']),atol=3e-6,rtol=1e-4)
        self.assertTrue((torch.autograd.grad(l1-l0,s)[0][-1]!=0).any())
        self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in baseline.parameters()))
        self.assertEqual(set(model.state_dict()),set(baseline.state_dict()))
        baseline.eval();model.eval()
        with torch.no_grad():
            x=torch.rand(1,3,160,160)
            self.assertTrue(torch.equal(baseline(x)[0],model(x)[0]))
        buf=io.BytesIO();torch.save(dict(model=model,epoch=19),buf);buf.seek(0)
        from ultralytics.utils.patches import torch_load
        restored=torch_load(buf,map_location='cpu')['model']
        self.assertEqual(restored.gic_config,CONFIG);self.assertEqual(restored.gic_epoch,19)
        self.assertEqual(eta_at_epoch(restored.gic_epoch+1),.5)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    torch.set_num_threads(4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GICTests))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                failures=[(str(t),v) for t,v in result.failures+result.errors],mother_sha=BASE,
                python=sys.version,torch=torch.__version__,cuda_available=torch.cuda.is_available(),
                scope='synthetic/math plus real model assembly; NOT a real-data B16 GPU preflight')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__': sys.exit(main())
