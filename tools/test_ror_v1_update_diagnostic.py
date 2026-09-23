"""Targeted checks for zero-weight graphs and the independent-update diagnostic."""
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
import ror_v1_update_diagnostic as d
from ultralytics.models.utils.ror import RORLoss


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.,2.]))
        self.register_buffer('persistent',torch.ones(2))
        self.register_buffer('nonpersistent',torch.ones(1),persistent=False)
        self.criterion = torch.nn.Identity()


def snapshot(model, optimizer):
    value = dict(state=d.cpu_copy(model.state_dict()),buffers=d.cpu_copy(dict(model.named_buffers())),
                 rng=d.rng_state(),optimizer=deepcopy(optimizer.state_dict()),groups=d.group_spec(model,optimizer))
    value['identity'] = dict(state=d.digest_tree(value['state']),buffers=d.digest_tree(value['buffers']),
        rng=d.digest_tree(value['rng']),optimizer=d.digest_tree(value['optimizer']),groups=d.digest_tree(value['groups']))
    return value


class UpdateDiagnosticTests(unittest.TestCase):
    def test_complete_reset_including_nonpersistent_buffers_and_optimizer(self):
        model = Toy(); optimizer = torch.optim.AdamW(model.parameters(),lr=.0005)
        model.weight.sum().backward(); optimizer.step()
        saved = snapshot(model,optimizer)
        with torch.no_grad(): model.weight.add_(1); model.persistent.add_(2); model.nonpersistent.add_(3)
        optimizer.param_groups[0]['lr'] = .1
        optimizer.state[model.weight]['exp_avg'].add_(4)
        random.random(); np.random.rand(); torch.rand(3)
        self.assertEqual(d.restore_trial(model,optimizer,saved),saved['identity'])
        self.assertIsNone(model.weight.grad)
        self.assertEqual(float(model.nonpersistent[0]),1.)

    def test_same_gradient_optimizer_is_separate_and_exact(self):
        a,b=Toy(),Toy(); models=dict(mother=a,ror=b)
        opts={k:torch.optim.AdamW(v.parameters(),lr=.0005,weight_decay=.0001) for k,v in models.items()}
        report={}
        d.identical_gradient_control(models,opts,snapshot(a,opts['mother']),{'weight':torch.tensor([1e-8,-20.])},report,lambda:None)
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['parameter_max_abs'],0.)
        self.assertLess(report['trials']['mother']['clip_coefficient'],1.)
        self.assertIn('cannot pass independent',report['scope'])

    def test_graph_fingerprint_detects_connected_zero(self):
        p=torch.ones(2,requires_grad=True); names={id(p):'p'}
        plain=d.graph_signature(p.square().sum(),names)
        connected=d.graph_signature(p.square().sum()+0*p.sum(),names)
        self.assertNotEqual(plain['sha256'],connected['sha256'])

    def test_real_zero_criterion_graph_losses_and_gradients_match_mother(self):
        parent=d.parent_module('ultralytics-main/ultralytics/models/utils/loss.py','ultralytics.models.utils._ror_zero_test_parent')
        torch.manual_seed(7)
        boxes=torch.rand(4,2,5,4)*.7+.1; scores=torch.randn(4,2,5,1)
        batch=dict(cls=torch.zeros(3,dtype=torch.long),bboxes=torch.tensor([[.4,.5,.2,.3],[.7,.2,.1,.1],[.2,.4,.1,.2]]),gt_groups=[2,1])
        # Native DN positives: two GTs in image 0 and one in image 1.
        meta=dict(dn_pos_idx=[torch.tensor([0,1]),torch.tensor([0])],dn_num_group=1)
        for epoch in (0,5):
            outputs=[]
            for criterion in (parent.RTDETRDetectionLoss(nc=1,use_vfl=True),RORLoss()):
                if isinstance(criterion,RORLoss):criterion.set_epoch(epoch)
                b=boxes.clone().requires_grad_();s=scores.clone().requires_grad_()
                db=boxes[:3,:,:2].clone().requires_grad_();ds=scores[:3,:,:2].clone().requires_grad_()
                calls=[];handle=criterion.matcher.register_forward_hook(lambda m,a,o:calls.append(d.digest_tree(o)))
                kw=dict(ror_enabled=True) if isinstance(criterion,RORLoss) else {}
                with patch('ultralytics.models.utils.ror.matched_loss',side_effect=AssertionError('Zero path created ROR')):
                    loss=criterion((b,s),batch,dn_bboxes=db,dn_scores=ds,dn_meta=meta,**kw)
                handle.remove(); total=sum(loss.values())
                graph=d.graph_signature(total,{id(x):n for x,n in zip((b,s,db,ds),('boxes','scores','dn_boxes','dn_scores'))})
                total.backward()
                outputs.append((d.digest_tree(loss),graph,calls,d.digest_tree([x.grad for x in (b,s,db,ds)])))
                self.assertNotIn('loss_ror',loss)
            self.assertEqual(outputs[0],outputs[1])

    def test_coordinate_sign_amplification_is_measured_without_new_bound(self):
        initial=torch.tensor([1.,2.]); group=dict(lr=.0005,eps=1e-8,weight_decay=.0001,betas=(.937,.999))
        data={};trials={}
        for label,value in zip(('mother_0','mother_1','mother_2','ror_0','ror_1','ror_2'),(-1e-6,1e-6,2e-6,1e-6,-1e-6,-2e-6)):
            g=torch.tensor([value,1.]);data[label]=dict(raw={'p':g},clipped={'p':g},after={'p':d.first_adam(initial,g,group)})
            trials[label]=dict(clip_coefficient=1.,total_gradient_norm=1.)
        row=d.coordinate_details('p',0,data,trials,initial,group)
        self.assertTrue(row['worst_cross']['gradient_sign_flip'])
        self.assertTrue(row['mother_repeat_spans_both_signs'])
        self.assertAlmostEqual(abs(row['worst_cross']['formula_predicted_difference']),.001,delta=1e-5)
        self.assertLess(abs(row['worst_cross']['difference_residual']),2e-7)
        json.dumps(row,allow_nan=False)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
