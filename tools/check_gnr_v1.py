"""Bounded PyTorch mathematical, same-forward integration and checkpoint checks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from gnr_v1_common import ROOT, MODEL, write_json, compare_recipe, load_yaml, source_identity
import torch
from ultralytics.models.rtdetr.gnr_loss import GNRDetectionLoss, negative_weights, negative_derivative, ordered_weights, ramp, FORMULA
from ultralytics.models.rtdetr.gnr_model import GNRDetectionModel, forward_with_features, criterion_inputs, targets_from_batch
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), (a.shape, (a - b).abs().max())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def mathematics(device):
    mother = RTDETRDetectionLoss(nc=1, use_vfl=True).to(device)
    z = torch.linspace(-8, 8, 101, device=device, requires_grad=True)
    negative_loss = mother.vfl(z.reshape(1, -1, 1), torch.zeros(1, len(z), 1, device=device), torch.zeros(1, len(z), 1, device=device)) * len(z)
    derivative = torch.autograd.grad(negative_loss, z)[0]
    analytical = negative_derivative(z.detach(), mother.vfl.alpha, mother.vfl.gamma)
    torch.testing.assert_close(derivative, analytical, rtol=2e-6, atol=1e-7)
    assert torch.isfinite(negative_derivative(torch.tensor([-1000., 1000.], device=device), .25, 1.5)).all()
    linear = torch.nn.Linear(3, 1).to(device)
    h = torch.tensor([[.2, -.5, 1.3]], device=device)
    logits = linear(h).view(1, 1, 1)
    loss = mother.vfl(logits, torch.zeros_like(logits), torch.zeros_like(logits))
    grad_w, grad_b = torch.autograd.grad(loss, (linear.weight, linear.bias))
    d = negative_derivative(logits.detach().flatten(), mother.vfl.alpha, mother.vfl.gamma)
    torch.testing.assert_close(grad_w, d[:, None] * h, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(grad_b, d, rtol=2e-6, atol=1e-7)
    ones = torch.ones(8, device=device)
    w, r, order = ordered_weights(ones, ones, torch.ones(8, 8, device=device), torch.ones(8, 8, device=device), ones, .5, 1)
    expected = torch.tensor([1, .75, 2/3, .625, .6, 7/12, 4/7, .5625], device=device)
    torch.testing.assert_close(w, expected, rtol=0, atol=6e-8)
    assert abs(float(w.double().sum()) - 5.358928571428571) < 2e-7
    exact(order, torch.arange(8, device=device))
    a = torch.tensor([.2, .8, .6], device=device)
    _, r, order = ordered_weights(a, torch.ones_like(a), torch.ones(3, 3, device=device), torch.ones(3, 3, device=device), torch.tensor([3., 2., 1.], device=device), .5, 1)
    torch.testing.assert_close(r, torch.tensor([0., .2, 1.], device=device))
    assert [ramp(e) for e in (0, 5, 6, 20, 57)] == [0, 0, 1/15, 1, 1]

    boxes = torch.tensor([[[.2,.2,.2,.2]]*4 + [[.8,.8,.2,.2]]*4,
                          [[.5,.5,.4,.4]]*8], device=device, requires_grad=True)
    logits = torch.full((2, 8, 1), -2., device=device, requires_grad=True)
    features = torch.ones((2, 8, 3), device=device, requires_grad=True)
    gt = torch.tensor([[.2,.2,.2,.2],[.8,.8,.2,.2],[.5,.5,.4,.4]], device=device)
    quality = torch.zeros((2,8), device=device)
    quality[0, [0,4]], quality[1,0] = 1, 1
    matches = [(torch.tensor([0,4]), torch.tensor([0,1])), (torch.tensor([0]), torch.tensor([2]))]
    def weights(b=boxes, z=logits, h=features, g=gt, groups=(2,1), m=matches, q=quality):
        return negative_weights(b,z,h,g,list(groups),m,q,alpha=mother.vfl.alpha,gamma=mother.vfl.gamma,beta=.5,epoch=20)
    w, detail = weights()
    assert not w.requires_grad and w.grad_fn is None
    assert bool((w[detail['first']] == 1).all()) and bool((w >= .5).all())
    assert bool((w[0,[0,1,4,5]] == 1).all()) and bool((w[1,:2] == 1).all())
    assert detail['ownership'][1,1:].eq(0).all()  # correct flattened GT offset
    assert int((w < 1).sum()) == 10
    empty = [(torch.zeros(0,dtype=torch.long),torch.zeros(0,dtype=torch.long))]*2
    exact(weights(g=gt[:0],groups=(0,0),m=empty,q=quality*0)[0],torch.ones_like(w))
    exact(weights(q=quality*0)[0],torch.ones_like(w))
    exact(weights(z=torch.full_like(logits, 100.))[0],torch.ones_like(w))
    # GT tie, nonintersection, only negative, no negative and absent GT positive.
    tie_gt = gt[[0,0]]
    tie_matches = [(torch.tensor([0,1]), torch.tensor([0,1]))]
    kwargs = dict(b=boxes[:1],z=logits[:1],h=features[:1],groups=(2,),g=tie_gt,m=tie_matches,q=quality[:1])
    exact(weights(**kwargs)[0],torch.ones_like(w[:1]))
    single = dict(b=boxes[:1,:2],z=logits[:1,:2],h=features[:1,:2],g=gt[:1],groups=(1,),m=matches[:1],q=quality[:1,:2])
    single['m']=[(torch.tensor([0]),torch.tensor([0]))]
    exact(weights(**single)[0],torch.ones_like(w[:1,:2]))
    single['m']=[(torch.tensor([0,1]),torch.tensor([0,1]))]
    single.update(g=gt[:2],groups=(2,))
    exact(weights(**single)[0],torch.ones_like(w[:1,:2]))
    opposite = features.detach().clone()
    opposite[0,0] = -2
    opposite[0,4] = -2
    opposite[1,0] = -2
    exact(weights(h=opposite)[0],torch.ones_like(w))
    missing = weights(m=[(torch.tensor([0]),torch.tensor([0])), matches[1]])[1]
    assert missing['unmatched_gt_negative_queries'] == 4
    return {"actual_vfl_alpha":mother.vfl.alpha,"actual_vfl_gamma":mother.vfl.gamma,
            "derivative_max_abs_error":float((derivative-analytical).abs().max()),"eight_weight_sum":float(expected.double().sum())}


def criterion_checks(device):
    torch.manual_seed(42)
    b = torch.rand(4,2,8,4,device=device,requires_grad=True)
    z = torch.randn(4,2,8,1,device=device,requires_grad=True)
    gt = {"bboxes":torch.tensor([[.5,.5,.7,.7],[.2,.2,.2,.2],[.5,.5,.5,.5]],device=device),
          "cls":torch.zeros(3,dtype=torch.long,device=device),"gt_groups":[2,1]}
    db = torch.rand(3,2,6,4,device=device,requires_grad=True)
    dz = torch.randn(3,2,6,1,device=device,requires_grad=True)
    dm = {"dn_pos_idx":[torch.tensor([0,1,3,4]),torch.tensor([0,3])],"dn_num_group":2}
    mother=RTDETRDetectionLoss(nc=1,use_vfl=True)
    gnr=GNRDetectionLoss(nc=1,use_vfl=True)
    reference=mother((b,z),gt,db,dz,dm)
    for epoch,beta in ((0,.5),(5,.5),(20,0)):
        actual=gnr((b,z),gt,db,dz,dm,epoch=epoch,beta=beta)
        exact(reference,actual)
        exact(torch.autograd.grad(sum(reference.values()),(b,z,db,dz),retain_graph=True),torch.autograd.grad(sum(actual.values()),(b,z,db,dz),retain_graph=True))
    actual,details=gnr((b,z),gt,db,dz,dm,features=torch.ones(2,8,256,device=device),epoch=20,return_details=True)
    assert reference.keys()==actual.keys()
    for key in reference:
        if key!='loss_class': exact(reference[key],actual[key])
    target=details['quality'].unsqueeze(-1)
    labels=(~details['negative']).unsqueeze(-1).long()
    modulation=mother.vfl.alpha*z[-1].sigmoid().pow(mother.vfl.gamma)*(1-labels)+target*labels
    terms=torch.nn.functional.binary_cross_entropy_with_logits(z[-1],target,reduction='none')*modulation
    expected=(terms*details['w'].unsqueeze(-1)).sum()/3
    torch.testing.assert_close(actual['loss_class'],expected,rtol=1e-6,atol=1e-7)
    grad=torch.autograd.grad(actual['loss_class'],z,retain_graph=True)[0]
    assert grad[-1].abs().sum()>0 and not grad[:-1].any()
    assert torch.autograd.grad(actual['loss_class'],b,allow_unused=True,retain_graph=True)[0] is None
    try:
        gnr((b,z),gt,epoch=20)
    except ValueError as error:
        assert 'requires explicit' in str(error)
    else: raise AssertionError('Active missing features silently accepted')
    return {"loss_keys":list(actual),"protected_losses_exact":True,"off_loss_and_gradients_exact":True}


def integration():
    torch.manual_seed(42)
    mother=RTDETRDetectionModel(str(MODEL),nc=1,verbose=False)
    candidate=GNRDetectionModel(str(MODEL),nc=1,verbose=False)
    candidate.load_state_dict(mother.state_dict(),strict=True)
    for m in (mother,candidate): m.nc=1
    state=deepcopy(mother.state_dict())
    batch={"img":torch.rand(2,3,160,160),"batch_idx":torch.tensor([0,0,1]),"cls":torch.zeros(3,1),
           "bboxes":torch.tensor([[.5,.5,.7,.7],[.2,.2,.2,.2],[.5,.5,.5,.5]])}
    targets=targets_from_batch(batch)
    rng=torch.random.get_rng_state()
    raw=mother.predict(batch['img'],batch=targets)
    torch.random.set_rng_state(rng)
    new,h=forward_with_features(candidate,batch['img'],targets)
    exact(raw,new)
    exact(new[1][-1],candidate.model[-1].dec_score_head[-1](h))
    inputs,ordinary,_,_,meta=criterion_inputs(new,h)
    assert ordinary.shape==(2,300,256) and h.shape[1]==sum(meta['dn_num_split'])
    for m in (mother,candidate): m.load_state_dict(state); m.zero_grad(set_to_none=True)
    torch.random.set_rng_state(rng)
    loss,items=mother.loss(batch);loss.backward()
    reference={n:None if p.grad is None else p.grad.clone() for n,p in mother.named_parameters()}
    torch.random.set_rng_state(rng)
    off,off_items=candidate.loss(batch);off.backward()
    exact(loss,off);exact(items,off_items)
    for n,p in candidate.named_parameters():
        try: exact(reference[n],p.grad)
        except AssertionError as error: raise AssertionError(n) from error
    mother.eval();candidate.eval()
    # BN updates are identical in both paths.
    with torch.no_grad(): exact(mother.predict(batch['img']),candidate.predict(batch['img']))
    candidate.train(); candidate.gnr_epoch=20;candidate.zero_grad(set_to_none=True)
    on,_=candidate.loss(batch);on.backward()
    assert torch.isfinite(on) and candidate.gnr_diagnostics['ramp']==1
    assert set(candidate.state_dict())==set(mother.state_dict())
    return candidate,{"same_forward_outputs_and_linear_inputs_exact":True,"ordinary_feature_shape":list(ordinary.shape),
                      "off_model_loss_and_all_parameter_gradients_exact":True,"inference_exact":True,"active_backward_finite":True,
                      "parameters":sum(p.numel() for p in candidate.parameters()),"state_keys":len(candidate.state_dict())}


def amp_criterion_check():
    """Exercise real CUDA autocast outputs, separately from the real-update preflight."""
    device = 'cuda:0'
    head = torch.nn.Linear(256, 1).to(device)
    with torch.no_grad():
        head.weight.fill_(.001)
        head.bias.fill_(-2)
    h = torch.ones(1, 8, 256, device=device)
    boxes = torch.tensor([.5, .5, .6, .6], device=device).expand(4, 1, 8, 4).clone().requires_grad_()
    gt = {'bboxes': boxes[0, 0, :1].detach().clone(), 'cls': torch.zeros(1, dtype=torch.long, device=device), 'gt_groups': [1]}
    mother, candidate = RTDETRDetectionLoss(nc=1, use_vfl=True), GNRDetectionLoss(nc=1, use_vfl=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        z = head(h).unsqueeze(0).expand(4, -1, -1, -1)
        original = mother((boxes, z), gt)
        off = candidate((boxes, z), gt, epoch=0)
        active, detail = candidate((boxes, z), gt, features=h, epoch=20, return_details=True)
    assert z.dtype == torch.float16 and detail['w'].dtype == torch.float32
    exact(original, off)
    for key in original:
        if key != 'loss_class':
            exact(original[key], active[key])
    before = torch.autograd.grad(original['loss_class'], z, retain_graph=True)[0]
    after = torch.autograd.grad(active['loss_class'], z, retain_graph=True)[0]
    positive = ~detail['negative']
    exact(before[-1, ..., 0][positive], after[-1, ..., 0][positive])
    assert (detail['w'] < 1).any() and torch.isfinite(after).all()
    assert torch.autograd.grad(active['loss_class'], boxes, allow_unused=True, retain_graph=True)[0] is None
    sum(active.values()).backward()
    assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    return {'logits_dtype': str(z.dtype), 'weights_dtype': str(detail['w'].dtype),
            'off_and_protected_losses_exact': True, 'positive_logit_gradients_exact': True,
            'active_negative_count': int((detail['w'] < 1).sum()), 'finite_nonzero_head_gradient': True,
            'scope': 'fixed tensor CUDA autocast criterion; no optimizer or real B16 preflight claim'}


def checkpoint_check(model):
    from gnr_v1_training import GNRTrainer
    bind={"fixture":"temporary_native_resume"}
    model.gnr_identity=bind;model.gnr_epoch=19
    model.zero_grad(set_to_none=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
    scaler=torch.amp.GradScaler('cpu',init_scale=65536)
    head=model.model[-1].dec_score_head[-1]
    loss=head(torch.ones(2,256)).square().mean()
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
    ema=ModelEMA(model);ema.update(model)
    with tempfile.TemporaryDirectory(prefix='gnr-resume-') as folder:
        path=Path(folder)/'last.pt'
        torch.save(dict(epoch=19,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),ema=ema.ema,updates=ema.updates,best_fitness=.4),path)
        saved=torch_load(path,map_location='cpu')
        trainer=object.__new__(GNRTrainer)
        trainer.model=deepcopy(model)
        trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=.001)
        trainer.scaler=torch.amp.GradScaler('cpu',init_scale=16)
        trainer.ema=ModelEMA(trainer.model)
        trainer.gnr_binding=bind;trainer.resume=True;trainer.epochs=200
        trainer.args=SimpleNamespace(model=str(path),close_mosaic=10)
        trainer.resume_training(saved)
        exact(trainer.optimizer.state_dict(),optimizer.state_dict())
        exact(trainer.scaler.state_dict(),scaler.state_dict())
        exact(trainer.ema.ema.state_dict(),ema.ema.state_dict())
        assert trainer.start_epoch==20 and trainer.ema.updates==ema.updates
        assert ramp(trainer.start_epoch)==1
    return {"optimizer_exact":True,"enabled_cpu_gradscaler_exact":True,"ema_exact":True,"restored_start_epoch":20,"restored_ramp":1}


def infrastructure():
    from gnr_v1_eval import postprocess, threshold_counts, verified_success, artifact_record
    preds=torch.tensor([[[.5,.5,.5,.5,.2],[.5,.5,.5,.5,.0001],[.5,.5,.5,.5,.9]]])
    filtered=postprocess(preds,640,.001)[0]
    assert filtered['query_ids'].tolist()==[2,0]
    assert torch.equal(preds[0,:,4],torch.tensor([.2,.0001,.9]))
    with tempfile.TemporaryDirectory(prefix='gnr-offline-') as tmp:
        file=Path(tmp)/'predictions.jsonl.gz'
        with gzip.open(file,'wt') as stream:
            stream.write(json.dumps(dict(scores=[.9,.1],boxes=[[0,0,.7,1],[0,0,1,1]],classes=[0,0],gt_boxes=[[0,0,1,1]],gt_classes=[0]))+'\n')
        counts=threshold_counts(file,.5)
        assert counts['TP']==1 and counts['FP']==counts['FN']==0
        folder = Path(tmp) / 'reuse'; folder.mkdir()
        for name in ('queries_gt.jsonl.gz', 'raw_stats.npz', 'curves.npz'):
            (folder / name).write_bytes(b'hashed artifact fixture')
        record = {'status': 'COMPLETE', 'exit_code': 0, 'identity': {'fixture': True}, 'artifacts': artifact_record(folder)}
        write_json(folder / 'metrics.json', record)
        assert verified_success(folder, record['identity']) == record  # success index absent
        assert verified_success(folder, {'fixture': False}) is None
        (folder / 'curves.npz').write_bytes(b'corrupted')
        assert verified_success(folder, record['identity']) is None
    expected=load_yaml(ROOT/'docs/c19_lif_v1/resolved_formal_config.yaml')
    actual=dict(expected,model=expected['model'].replace('-v1/','-v1-gatefix/'))
    original=deepcopy(actual)
    assert compare_recipe(actual,expected)['alias']
    assert actual==original
    actual['batch']=16.0
    try: compare_recipe(actual,expected)
    except RuntimeError: pass
    else: raise AssertionError('Type mismatch accepted')
    return {"sorted_conf_mask_query_mapping":True,"offline_rematches_after_threshold":True,"recipe_alias_and_type_guard":True,
            "reuse_without_success_index_and_corruption_rejection": True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'docs/gnr_v1/local_validation.json');p.add_argument('--cuda',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);torch.autograd.set_multithreading_enabled(False);torch.use_deterministic_algorithms(True);started=time.monotonic()
    result={"status":"RUNNING","torch":torch.__version__,"cpu":{},"cuda":"PENDING","formal_preflight":"Separate required server check; development B16 resource failure is in local_preflight.json",
            "determinism":{"cpu_threads":1,"autograd_multithreading":False,"strict_equal_tolerance":0,
                           "development_observation":"Earlier multithreaded CPU accumulation differed by 2.7940e-09; single-thread deterministic comparison uses exact equality without relaxed tolerance"}}
    try:
        result['cpu']['math']=mathematics('cpu');result['cpu']['criterion']=criterion_checks('cpu')
        model,result['cpu']['integration']=integration();result['cpu']['checkpoint']=checkpoint_check(model)
        result['cpu']['infrastructure']=infrastructure()
        if args.cuda:
            assert torch.cuda.is_available()
            result['cuda']={"device":torch.cuda.get_device_name(0),"math":mathematics('cuda:0'),"criterion":criterion_checks('cuda:0'),
                            "autocast":amp_criterion_check(),"scope":"fixed tensor checks; not real training preflight"}
        result['status']='PASS'
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['code'] = source_identity(clean=False)
        write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
