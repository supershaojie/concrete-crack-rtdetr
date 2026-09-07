"""ACR formulas, DN/mask contracts, native MHA equivalence and real gradient paths."""
import math
import unittest
from copy import deepcopy

import torch
from torch import nn
from ultralytics.nn.modules.acr import CoverageRelationSelfAttention as ACR
from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer


class ACRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(4)

    def test_geometry_fixed_order_direction_bounds(self):
        boxes = torch.tensor([[[.5,.5,.8,.2],[.5,.5,.2,.2],[1.5,.5,.2,.2],[.5,1.5,.2,.2],[0.,0.,0.,1e-9]]], requires_grad=True)
        g = ACR.geometry(boxes)
        self.assertEqual(g.shape, (1,5,5,10))
        self.assertFalse(g.requires_grad)
        torch.testing.assert_close(g[0,0,1], torch.tensor([0,0,math.log(4),0,0,0,.25,1,1,1]))
        torch.testing.assert_close(g[0,1,0,6:], torch.tensor([1.,1.,.25,1.]))
        torch.testing.assert_close(g[0,0,0],torch.tensor([0.]*6+[1.]*4))
        self.assertEqual(float(g[0,1,2,6]),0.)
        self.assertEqual(float(g[0,1,2,7]),1.)
        self.assertEqual(float(g[0,1,3,6]),1.)
        self.assertEqual(float(g[0,1,3,7]),0.)
        self.assertGreater(float(g[0,1,2,4]),0.)
        self.assertGreater(float(g[0,1,3,5]),0.)
        self.assertTrue(torch.isfinite(g).all())
        self.assertTrue((g[...,:4].abs()<=4).all() and (g[...,4:6]<=4).all() and (g[...,6:]<=1).all())
        # Independent scalar oracle for all ordered pairs, including the size floor.
        for i,a in enumerate(boxes[0].detach().tolist()):
            for j,b in enumerate(boxes[0].detach().tolist()):
                a[2:]=[max(v,1e-4) for v in a[2:]]; b[2:]=[max(v,1e-4) for v in b[2:]]
                overlap=[];gap=[]
                for d in range(2):
                    l=max(a[d]-a[d+2]/2,b[d]-b[d+2]/2);r=min(a[d]+a[d+2]/2,b[d]+b[d+2]/2)
                    overlap.append(max(0,r-l));gap.append(max(0,l-r))
                expected=[max(-4,min(4,math.asinh((b[d]-a[d])/(.5*(a[d+2]+b[d+2]))))) for d in range(2)]
                expected += [max(-4,min(4,math.log(a[d]/b[d]))) for d in (2,3)]
                expected += [min(4,math.log1p(gap[d]/min(a[d+2],b[d+2]))) for d in range(2)]
                expected += [min(1,overlap[d]/box[d+2]) for box in (a,b) for d in range(2)]
                torch.testing.assert_close(g[0,i,j],torch.tensor(expected),atol=1e-6,rtol=1e-5)
        translated=boxes.detach()[:,:4].clone();translated[...,:2]+=7
        torch.testing.assert_close(ACR.geometry(translated),g[:,:4,:4],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(ACR.geometry(boxes.detach()[:,:4]*2),g[:,:4,:4])

    def test_constructor_rng_common_parameters_and_independence(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(42)
            a=nn.MultiheadAttention(256,8,device=device)
            cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            torch.manual_seed(42);b=ACR(256,8,device=device)
            self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
            for x,y in zip(cuda,torch.cuda.get_rng_state_all() if cuda else []): self.assertTrue(torch.equal(x,y))
            for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]))
            self.assertEqual(sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters()),616)
            self.assertNotEqual(b.acr_head.weight.data_ptr(),deepcopy(b).acr_head.weight.data_ptr())

    def test_native_mha_equivalence_dropout_masks_and_dtypes(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for dropout in (0.,.2):
                native=nn.MultiheadAttention(32,4,dropout=dropout).to(device)
                acr=ACR(32,4,dropout=dropout).to(device)
                acr.load_state_dict({**acr.state_dict(),**native.state_dict()},strict=True)
                q=torch.randn(7,2,32,device=device);v=torch.randn_like(q)
                boxes=torch.rand(2,7,4,device=device)
                for kind in ('none','bool','float','expanded'):
                    mask=None
                    if kind!='none':
                        mask=torch.zeros(7,7,dtype=torch.bool,device=device);mask[:,0]=True
                        if kind in ('float','expanded'):
                            mask=torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,-torch.inf);mask[:,2]=-.35
                        if kind=='expanded':mask=mask.expand(8,7,7).clone()
                    torch.manual_seed(123);a=native(q,q,v,attn_mask=mask)
                    torch.manual_seed(123);b=acr(q,q,v,attn_mask=mask,refer_bbox=boxes,num_queries=7)
                    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
                if device=='cuda':
                    half=deepcopy(acr).half().eval()
                    out=half(q.half(),q.half(),v.half(),refer_bbox=boxes.half(),num_queries=7)
                    self.assertTrue(torch.isfinite(out[0]).all())
                    with torch.autocast('cuda'):
                        out=acr(q,q,v,refer_bbox=boxes,num_queries=7)
                    self.assertTrue(torch.isfinite(out[0]).all())

    def test_nonzero_bias_dn_masks_padding_and_entropy(self):
        model=ACR(32,4,dropout=.4).eval()
        nn.init.normal_(model.acr_head.weight,std=.7)
        model.acr_stats_interval=1
        for dn in (0,2,6):
            n=dn+5; q=torch.randn(n,2,32);boxes=torch.rand(2,n,4)
            meta={'dn_num_split':[dn,5]} if dn else None
            mask=torch.zeros(n,n,dtype=torch.bool);mask[dn:,:dn]=True
            if dn:mask[:dn,0]=True
            pad=torch.zeros(2,n,dtype=torch.bool);pad[:,-1]=True
            bias=model.relation_bias(boxes[:,dn:]); self.assertTrue((bias.diagonal(dim1=-2,dim2=-1)==0).all())
            self.assertFalse(torch.equal(bias,bias.transpose(-1,-2)))
            expected=nn.functional.pad(bias,(dn,0,dn,0))
            expected=expected+torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,-torch.inf)
            expected=expected.masked_fill(pad[:,None,None,:],-torch.inf)
            native=nn.MultiheadAttention(32,4);native.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('acr_')});native.eval()
            reference=native(q,q,q,attn_mask=expected.reshape(8,n,n),average_attn_weights=False)
            for layout in (mask,mask.expand(2,n,n),mask.expand(8,n,n),mask.expand(2,4,n,n)):
                out,attention=model(q,q,q,attn_mask=layout,key_padding_mask=pad,refer_bbox=boxes,num_queries=5,dn_meta=meta,average_attn_weights=False)
                torch.testing.assert_close(out,reference[0]);torch.testing.assert_close(attention,reference[1])
                torch.testing.assert_close(attention.sum(-1),torch.ones(2,4,n)) # eval: pre-dropout weights
                self.assertTrue((attention[..., -1]==0).all())
                if dn:self.assertTrue((attention[:,:,dn:,:dn]==0).all())
            self.assertTrue(math.isfinite(model.acr_last_stats['allowed_regular_row_entropy']))
            if dn:
                with self.assertRaises(ValueError):model(q,q,q,refer_bbox=boxes,num_queries=5,dn_meta=None)
                with self.assertRaises(ValueError):model(q,q,q,attn_mask=mask,refer_bbox=boxes,num_queries=5,dn_meta={'dn_num_split':[dn+1,5]})
                with self.assertRaises(ValueError):model(q,q,q,refer_bbox=boxes,num_queries=5,dn_meta=meta)

    def test_geometry_detach_and_two_step_learning(self):
        m=ACR(32,4);q=torch.randn(6,2,32,requires_grad=True);boxes=torch.rand(2,6,4,requires_grad=True)
        optimizer=torch.optim.SGD(m.parameters(),lr=.1)
        for step in range(2):
            optimizer.zero_grad();q.grad=None
            out=m(q,q,q,refer_bbox=boxes,num_queries=6)[0]
            (out*torch.randn_like(out)).sum().backward()
            self.assertIsNone(boxes.grad)
            self.assertTrue(torch.isfinite(q.grad).all() and q.grad.abs().sum()>0)
            for n,p in m.named_parameters():self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),n)
            self.assertGreater(float(m.acr_head.weight.grad.abs().sum()),0)
            geometry_grad=float(m.acr_geometry_proj.weight.grad.abs().sum())
            self.assertEqual(geometry_grad,0) if step==0 else self.assertGreater(geometry_grad,0)
            optimizer.step()

    def test_original_reference_and_feature_gradients_retained(self):
        torch.manual_seed(91);base=DeformableTransformerDecoderLayer(32,4,64,n_levels=3)
        torch.manual_seed(91);acr=DeformableTransformerDecoderLayer(32,4,64,n_levels=3,acr=True)
        query=torch.randn(2,5,32);boxes=torch.rand(2,5,4);feat=torch.randn(2,14,32)
        gradients=[]
        for layer in (base,acr):
            q=query.clone().requires_grad_();b=boxes.clone().requires_grad_();f=feat.clone().requires_grad_()
            out=layer(q,b,f,[(2,4),(2,2),(1,2)],query_pos=b.repeat(1,1,8),num_queries=5)
            (out.square().sum()).backward();gradients.append((q.grad,b.grad,f.grad))
        for a,b in zip(*gradients):
            self.assertGreater(float(a.abs().sum()),0);torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
