"""Real native matrix/convolution updates, group warmup and state continuation."""
from __future__ import annotations
import ast
from copy import deepcopy
import inspect
import textwrap
import unittest
import numpy as np
import torch
torch.set_num_threads(2)
from support import configure, ROOT, native_recipe
configure(ROOT/'outputs/yolo26m-configurable-validation/optimizer-runtime')
from adapters import ComparisonTrainer, model_identity
from configuration import resolve_config, cli_overrides
from optimizer import optimizer_identity, verify_buffers
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from run import load_initial_model


class TinyBranches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([torch.nn.BatchNorm2d(4), *[torch.nn.Identity() for _ in range(22)]])
        head = torch.nn.Module()
        head.cv3 = torch.nn.Conv2d(4,4,1)
        head.one2one_cv3 = torch.nn.Conv2d(4,4,1)
        head.project = torch.nn.Linear(4,4)
        self.model.append(head)


def build(model, name='MuSGD'):
    config=resolve_config({},cli_overrides([f'optimizer={name}']))
    trainer=ComparisonTrainer.__new__(ComparisonTrainer)
    trainer.args=get_cfg(overrides=native_recipe(config))
    trainer.comparison_config=config
    optimizer=trainer.build_optimizer(model,name=name,lr=config['lr0'],momentum=config['momentum'],decay=config['weight_decay'])
    for g in optimizer.param_groups: g['initial_lr']=g['lr']
    return trainer,optimizer,config


class OptimizerTests(unittest.TestCase):
    def test_musgd_two_buffers_real_2d_4d_updates_and_exact_optimizer_resume(self):
        torch.manual_seed(42)
        model=TinyBranches(); _,optimizer,cfg=build(model,'MuSGD')
        initial={n:p.detach().clone() for n,p in model.named_parameters()}
        for p in model.parameters(): p.grad=torch.randn_like(p)
        optimizer.step()
        report=optimizer_identity(optimizer,model,cfg)
        self.assertEqual(report['class'],'ultralytics.optim.muon.MuSGD')
        self.assertEqual((optimizer.muon,optimizer.sgd),(.5,.5))
        self.assertEqual(verify_buffers(optimizer)['populated_mixed_tensors'],3)
        for n,p in model.named_parameters():
            self.assertFalse(torch.equal(initial[n],p),n)
            self.assertTrue(torch.isfinite(p).all())
            if p.ndim>=2:
                self.assertEqual(set(optimizer.state[p]),{'momentum_buffer','momentum_buffer_SGD'})
        saved=deepcopy(optimizer.state_dict())
        resumed=deepcopy(model); _,restored,_=build(resumed)
        restored.load_state_dict(saved)
        self.assertEqual(optimizer_identity(restored,resumed,cfg),report)
        # The next actual update must match after re-creating the native optimizer.
        for left,right in zip(model.parameters(),resumed.parameters()):
            grad=torch.randn_like(left); left.grad=grad.clone(); right.grad=grad.clone()
        optimizer.step(); restored.step()
        for left,right in zip(model.parameters(),resumed.parameters()):
            self.assertTrue(torch.equal(left,right))
            for key in optimizer.state[left]:
                self.assertTrue(torch.equal(optimizer.state[left][key],restored.state[right][key]))
        restored.muon=.6
        with self.assertRaisesRegex(ValueError,'coefficients'): optimizer_identity(restored,resumed,cfg)

    def test_native_group_coverage_regex_lr_and_actual_patched_warmup(self):
        for name in ('MuSGD','SGD','Adam','AdamW'):
            model=TinyBranches(); trainer,optimizer,cfg=build(model,name)
            groups=optimizer.param_groups
            self.assertEqual(len(groups),8 if name=='MuSGD' else 3)
            for g in groups:
                if not g['params']: continue
                boosted=name=='MuSGD' and all('23' in n and 'cv3' in n for n in g['comparison_names'])
                self.assertEqual(g['initial_lr'],.03 if boosted else .01)
            # Execute the exact warmup AST from our hash-locked native trainer.
            tree=ast.parse(textwrap.dedent(inspect.getsource(BaseTrainer._do_train)))
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                and isinstance(n.test.left,ast.Name) and n.test.left.id=='ni'
                and isinstance(n.test.ops[0],ast.LtE))
            code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'native_warmup','exec')
            trainer.optimizer=optimizer; trainer.batch_size=16; trainer.lf=lambda epoch:1.
            for ni in (0,50,100):
                exec(code,{'self':trainer,'ni':ni,'nw':100,'epoch':0,'np':np})
                for g in groups:
                    start=.1 if g['comparison_bias'] else 0.
                    self.assertAlmostEqual(g['lr'],start+(g['initial_lr']-start)*ni/100)
                    if 'momentum' in g: self.assertAlmostEqual(g['momentum'],.8+(.937-.8)*ni/100)
            self.assertTrue(any(g['comparison_bias'] and g['params'] for g in groups))
            self.assertFalse(groups[0]['comparison_bias'])

    def test_original_coco_yaml_tensor_protection_dual_loss_and_l1_slot_gain(self):
        from ultralytics.utils.loss import E2ELoss
        from adapters import criterion_state, restore_criterion
        model,initialization=load_initial_model()
        self.assertEqual(initialization['transferred_tensors'],756)
        self.assertEqual(len(initialization['missing_or_reshaped_keys']),12)
        self.assertTrue(initialization['coco']['one_to_many_present'])
        self.assertTrue(initialization['crack']['one_to_one_present'])
        model.args=get_cfg(overrides=native_recipe())
        model.train(); model.zero_grad()
        batch={'img':torch.rand(2,3,64,64),'batch_idx':torch.tensor([0,1]),'cls':torch.zeros(2,1),
            'bboxes':torch.tensor([[.5,.5,.3,.3],[.4,.4,.2,.2]]),'resized_shape':[(64,64)]*2}
        output=model(batch['img'])
        self.assertEqual(set(output),{'one2many','one2one'})
        loss,items=model.loss(batch,output)
        self.assertIs(type(model.criterion),E2ELoss)
        self.assertIsNone(model.criterion.one2one.bbox_loss.dfl_loss)
        self.assertGreater(float(items[2]),0.)
        loss.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.cv2.' in n))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'one2one_cv2' in n))
        baseline=items[2].clone()
        model.args.dfl=0.
        _,zero_items=model.loss(batch,output)
        self.assertEqual(float(zero_items[2]),0.)
        model.args.dfl=1.5
        _,same_items=model.loss(batch,output)
        self.assertTrue(torch.equal(same_items[2],baseline))
        initial=criterion_state(model.criterion)
        model.criterion.update(); saved=criterion_state(model.criterion)
        self.assertEqual(saved['updates'],1)
        replacement=model.init_criterion(); restore_criterion(replacement,saved)
        self.assertEqual(criterion_state(replacement),saved)
        replacement.update(); model.criterion.update()
        self.assertEqual(criterion_state(replacement),criterion_state(model.criterion))
        self.assertEqual(initial['o2m'],.8)
        head=model.model[-1]; head.cv2=None
        with self.assertRaisesRegex(ValueError,'complete unfused'): model_identity(model,1)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
