"""Focused boundary, FP32, clipping and inference checks; no dataset inference."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from lbc_v1_checks import bare_trainer, fixture
from lbc_v1_training import LBCTrainer, native_copy, HEAD_KEYS
from init_c19_lif_v1 import ROOT, require, write_json
from ultralytics.models.rtdetr.lbc import LBCHead, regions, score_loss
from ultralytics.utils.patches import torch_load
from lbc_v1_precision import strict_fusion_precision, precision_settings


def run(folder):
    torch.set_num_threads(1)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report={}
    head=LBCHead();f=torch.randn(2,128,8,8,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):z=head(f.to(torch.bfloat16))
    require(z.dtype==torch.float32,'Head must stay FP32 inside external autocast')
    with torch.no_grad():head.prototype.zero_()
    head(f).sum().backward();require(torch.isfinite(f.grad).all(),'Zero prototype epsilon branch')
    precision={};before=precision_settings()
    try:
        with strict_fusion_precision(precision):raise RuntimeError('intentional exception fixture')
    except RuntimeError as error:require(str(error)=='intentional exception fixture','Wrong precision exception')
    require(before==precision_settings() and precision['restored'],'Exceptional exit changed precision')
    report['precision']=dict(head_FP32_under_autocast=True,zero_prototype_finite=True,exception_scope=precision)
    model=torch_load(ROOT/'outputs/lbc_v1/final_checks/lifecycle/last.pt',map_location='cpu')['model'].float().train()
    batch=fixture();model.lbc_epoch=5
    with patch.object(model.lbc_head,'forward',side_effect=AssertionError('Head called while disabled')):
        loss,_=model(batch)
    model.lbc_epoch=6;loss,_=model(batch)
    stats=deepcopy(model.lbc_last)
    require(stats['pairs']>0 and abs(stats['weighted']-stats['raw_total']*.05/15)<1e-12,'e6 integration weight')
    model.zero_grad(set_to_none=True)
    targets=dict(cls=batch['cls'].long().flatten(),bboxes=batch['bboxes'],batch_idx=batch['batch_idx'].long(),gt_groups=[2,1])
    _,f3=model.predict_with_s3(batch['img'],targets)
    pairs,_,_=regions(batch,f3.shape[-2:]);raw,_=score_loss(model.lbc_head(f3),pairs)
    named=list(model.named_parameters())
    gradients=torch.autograd.grad(raw,[p for _,p in named],allow_unused=True)
    report['direct_gradient_norms']={}
    for group in ('head','S3','earlier'):
        values=[g.float().square().sum() for (n,_),g in zip(named,gradients) if g is not None and
                (n.startswith('lbc_head.') if group=='head' else
                 n.startswith('model.5.') if group=='S3' else n.startswith('model.') and int(n.split('.')[1])<5)]
        report['direct_gradient_norms'][group]=float(torch.stack(values).sum().sqrt())
    trainer=bare_trainer(model);model.lbc_epoch=20;loss,_=model(batch);loss.backward()
    with patch.object(trainer.scaler,'unscale_',wraps=trainer.scaler.unscale_) as unscale, \
         patch('torch.nn.utils.clip_grad_norm_',wraps=torch.nn.utils.clip_grad_norm_) as clip:
        trainer.optimizer_step()
        require(unscale.call_count==1 and clip.call_count==2,'Unscale/clip call counts')
        groups=[{id(p) for p in call.args[0]} for call in clip.call_args_list]
        all_ids={id(p) for p in model.parameters()};head_ids={id(p) for p in model.lbc_head.parameters()}
        require(groups[1]==head_ids and groups[0]==all_ids-head_ids,'Clipping groups overlap/omit parameters')
    model.eval()
    with torch.no_grad(),patch.object(model.lbc_head,'forward',side_effect=AssertionError('Head called in eval')):
        predictions=model(batch['img']);v,items=model.loss(batch,predictions)
    require(torch.isfinite(v) and len(items)==3,'Validation L0 contract')
    empty=deepcopy(batch);empty.update(bboxes=torch.empty(0,4),cls=torch.empty(0,1),batch_idx=torch.empty(0))
    model.train();model.lbc_epoch=20;model.zero_grad(set_to_none=True)
    with patch.object(model.lbc_head,'forward',side_effect=AssertionError('Head called without pairs')):
        e,_=model(empty);e.backward()
    require(all(p.grad is None for p in model.lbc_head.parameters()),'No GT head gradients')
    from lbc_v1 import working_points
    p=np.linspace(.2,1,1000)[None];r=np.linspace(1,0,1000)[None];f1=2*p*r/(p+r+1e-16)
    point=working_points(SimpleNamespace(box=SimpleNamespace(p_curve=p,r_curve=r,f1_curve=f1,px=np.linspace(0,1,1000))))
    require(point['recall_at_minimum_precision']['0.9']['precision']>=.9,'Fixed precision point')
    report.update(status='PASSED',e5_head_skipped=True,e6_weight=stats['weight'],
        unscale_count=1,clip_count=2,disjoint_clip_groups=True,val_predict_head_skipped=True,
        no_GT_native_loss=True,working_point_curve_fixture=point)
    write_json(folder/'contracts.json',report)
    return report


if __name__=='__main__':run(ROOT/'outputs/lbc_v1/contracts')
