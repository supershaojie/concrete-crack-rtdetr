"""Small independent CQS risk checks, plus optional native lifecycle and GPU measurements."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import random
import time

from cqs_v1_common import *
from ultralytics.nn.modules.cqs import candidate_pool, pairwise_relation, greedy_select, gather_positions


def reference_relation(features, boxes):
    """Independent scalar CPU oracle: no production geometry/reduction helper."""
    def one(f, g, a, b):
        dot = sum(x*y for x,y in zip(f,g))
        norm = max(math.sqrt(sum(x*x for x in f)), 1e-12) * max(math.sqrt(sum(x*x for x in g)), 1e-12)
        cosine = max(0., min(1., max(-1., dot/norm)))
        aa = [a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2]
        bb = [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]
        inter = max(0., min(aa[2],bb[2])-max(aa[0],bb[0])) * max(0., min(aa[3],bb[3])-max(aa[1],bb[1]))
        union = a[2]*a[3]+b[2]*b[3]-inter
        iou = min(1., max(0., inter/max(union,1e-12)))
        return iou**2*cosine**2
    return [[one(f,g,a,b) for g,b in zip(features,boxes)] for f,a in zip(features,boxes)]


def reference_greedy(scores, relations, core, queries):
    selected = list(range(core))
    factors = []
    while len(selected) < queries:
        values = [(scores[i]*(1-.5*max(relations[i][j] for j in selected)), -i, i)
                  for i in range(len(scores)) if i not in selected]
        _, _, i = max(values)
        factors.append(1-.5*max(relations[i][j] for j in selected))
        selected.append(i)
    return selected, factors


def algorithm_checks(device='cpu'):
    torch.manual_seed(42)
    cases = []
    for n in (300, 301, 525, 599, 600, 8400):
        # All ties intentionally make top600[:300] unsafe on real torch topk implementations.
        logits = torch.ones(2,n,1,device=device)
        native, pool = candidate_pool(logits)
        expected = torch.topk(logits.max(-1).values,300,dim=1).indices
        require(torch.equal(native,expected) and torch.equal(pool[:,:300],expected), 'Native top300 tie ordering changed')
        require(pool.shape == (2,min(n,600)) and all(len(set(r)) == min(n,600) for r in pool.tolist()), 'Duplicate/wrong pool')
        features = torch.randn(2,pool.shape[1],8,device=device)
        boxes = torch.rand(2,pool.shape[1],4,device=device)
        relation = pairwise_relation(features,boxes)
        ids, factors = greedy_select(torch.ones(2,pool.shape[1],device=device)*.8,relation)
        require(torch.equal(ids[:,:240],torch.arange(240,device=device).expand(2,-1)), 'Core changed')
        require(all(len(set(r)) == 300 for r in ids.tolist()), 'Final duplicates')
        require(bool(((factors>=.5)&(factors<=1)).all()), 'Discount out of bounds')
        cases.append(dict(N=n, pool=pool.shape[1], native_prefix_exact=True, unique=True))
    try:
        candidate_pool(torch.ones(2,299,1,device=device))
    except ValueError as e:
        underflow = str(e)
    else:
        raise AssertionError('N<300 did not fail')
    fixtures = [
        ([[1,0],[1,0],[0,1],[1,0],[0,0],[1,0]],
         [[.5,.5,.2,.2],[.5,.5,.2,.2],[.5,.5,.2,.2],[.9,.9,.1,.1],[.5,.5,.2,.2],[.1,.1,1e-9,1e-9]],
         [.99,.98,.8,.8,.7,.6],1,4),
        ([[1,0],[0,1],[1,0],[0,1],[1,0]],
         [[.2,.2,.2,.2],[.8,.8,.2,.2],[.8,.8,.2,.2],[.2,.2,.2,.2],[.2,.2,.2,.2]],
         [.99,.98,.9,.9,.85],2,4),
        ([[0,0]]*6, [[1.1,.5,.4,.2]]*6, [.7]*6,2,5),
    ]
    oracle = []
    for f,b,s,c,q in fixtures:
        rr = reference_relation(f,b)
        expected, discounts = reference_greedy(s,rr,c,q)
        rel = pairwise_relation(torch.tensor([f],device=device,dtype=torch.float32), torch.tensor([b],device=device))
        ids, fac = greedy_select(torch.tensor([s],device=device),rel,c,q)
        require(torch.allclose(rel.cpu(),torch.tensor([rr]),atol=2e-6), 'Independent relation oracle differs')
        require(ids.tolist()[0] == expected and torch.allclose(fac.cpu(),torch.tensor([discounts]),atol=2e-6), 'Greedy full sequence differs')
        oracle.append(dict(sequence=expected,discounts=discounts))
    require(oracle[0]['sequence'][1] == 2, 'Complementary candidate failed to replace duplicate')
    # Candidate 2 shares feature with core0 and geometry with core1, but no pair has both.
    anti = reference_relation(fixtures[1][0],fixtures[1][1])
    require(max(anti[2][:2]) == 0, 'Two-max multiplication bug')
    # Explicit sequential update: choosing a duplicate must change the next choice.
    rel = torch.tensor([[[1,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,1]]],device=device,dtype=torch.float32)
    ids,_ = greedy_select(torch.tensor([[.99,.9,.89,.8]],device=device),rel,1,3)
    require(ids.tolist() == [[0,1,3]], 'Selection did not update after each addition')
    return dict(status='PASS', device=device, pool_cases=cases, underflow=underflow, independent_oracle=oracle,
                sequential_update=[0,1,3], pair_product_before_max=True, ties='first pool index')


def mapping_and_gradient(device='cpu'):
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    from ultralytics.models.utils.ops import get_cdn_group
    torch.manual_seed(71)
    head = RTDETRDecoderCBRCQS(nc=1).to(device).train()
    parent = RTDETRDecoderCBR(nc=1).to(device).train()
    parent.load_state_dict(head.state_dict(),strict=True)
    features = torch.randn(2,525,256,device=device,requires_grad=True)
    shapes = [[20,20],[10,10],[5,5]]
    targets = dict(cls=torch.zeros(3,device=device,dtype=torch.long),
                   bboxes=torch.tensor([[.3,.3,.15,.2],[.7,.6,.2,.1],[.5,.5,.3,.2]],device=device),
                   batch_idx=torch.tensor([0,1,1],device=device),gt_groups=[1,2])
    state = torch.get_rng_state()
    if device != 'cpu':
        cuda_state = torch.cuda.get_rng_state()
    dn = get_cdn_group(targets,1,300,head.denoising_class_embed.weight,100,.5,1.,True)
    torch.set_rng_state(state)
    if device != 'cpu':
        torch.cuda.set_rng_state(cuda_state)
    other = get_cdn_group(targets,1,300,parent.denoising_class_embed.weight,100,.5,1.,True)
    require(all(torch.equal(a,b) for a,b in zip(dn[:3],other[:3])), 'DN content/mask changed')
    with head.capture_selection(full=True) as rows:
        embed,refs,boxes,logits = head._get_decoder_input(features,shapes,dn[0],dn[1])
    record = rows[0]; ids = record['indices']
    ef = head.enc_output(head.valid_mask*features)
    es = head.enc_score_head(ef)
    expected_features = gather_positions(ef,ids)
    expected_anchor = gather_positions(head.anchors.expand(2,-1,-1),ids)
    expected_refs = head.enc_bbox_head(expected_features)+expected_anchor
    require(torch.equal(embed[:,dn[0].shape[1]:],expected_features.detach()), 'Content gather mismatch')
    require(torch.equal(refs[:,dn[1].shape[1]:],expected_refs.detach()), 'Reference gather mismatch')
    require(torch.equal(boxes,expected_refs.sigmoid()) and torch.equal(logits,gather_positions(es,ids)), 'Encoder loss gather mismatch')
    require(torch.equal(embed[:,:dn[0].shape[1]],dn[0]) and torch.equal(refs[:,:dn[1].shape[1]],dn[1]), 'DN prefix changed')
    require(not refs.requires_grad and not record['relation'].requires_grad and boxes.requires_grad and logits.requires_grad,
            'Reference/selection/encoder detach contract broken')
    normal = head._get_decoder_input(features,shapes)
    require(not normal[0].requires_grad and not normal[1].requires_grad, 'Ordinary initial embeddings/references not detached')
    criterion = RTDETRDetectionLoss(nc=1,use_vfl=True)
    losses = criterion((boxes.unsqueeze(0),logits.unsqueeze(0)),targets)
    sum(losses.values()).backward()
    gradients = {}
    for name,values in [('encoder_class',head.enc_score_head.parameters()), ('encoder_bbox',head.enc_bbox_head.parameters())]:
        grads = [p.grad for p in values if p.grad is not None]
        require(grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0, name+' gradient invalid')
        gradients[name] = sum(float(g.float().norm()) for g in grads)
    require(torch.isfinite(features.grad).all() and bool(features.grad.abs().sum()>0),'No finite feature gradient')
    gradients['features'] = float(features.grad.norm())
    degradation = {}
    for mode in ('beta0','disabled'):
        head.cqs_config = {**CQS_CONFIG, **({'beta':0} if mode=='beta0' else {'enabled':False})}
        calls = head.cqs_calls
        a = head._get_decoder_input(features,shapes,dn[0],dn[1])
        b = parent._get_decoder_input(features,shapes,dn[0],dn[1])
        require(all(torch.equal(x,y) for x,y in zip(a,b)) and head.cqs_calls == calls,'Degradation not exact parent path')
        degradation[mode] = dict(all_four_outputs_exact=True, no_cqs_calls=True)
    head.cqs_config = dict(CQS_CONFIG)
    # Diagnostic failure retains context and permits valid infinity anchor sentinels.
    broken = features.detach().clone();broken[1,300,0] = float('nan')
    try:
        head._get_decoder_input(broken,shapes)
    except FloatingPointError as e:
        nonfinite_context = str(e)
    else:
        raise AssertionError('NaN not detected')
    return dict(status='PASS',device=device,unique_batch_mapping=True,ordinary_queries=300,dn_split=dn[3]['dn_num_split'],
                attention_mask_shape=list(dn[2].shape),gradients=gradients,losses=losses,
                degraded=degradation,nonfinite_context=nonfinite_context)


def benchmark(model, device='cuda', size=640, repeats=3):
    require(torch.cuda.is_available(), 'CUDA benchmark unavailable')
    model = deepcopy(model).to(device).eval()
    # Same source, exactly same weights; beta0 delegates to native parent forward input path.
    x = torch.rand(1,3,size,size,device=device)
    report = dict(scope='B1 short forward only; not formal B16 capacity or FPS',size=size,repeats=repeats)
    for amp in (False,True):
        rows = {}
        for enabled in (False,True):
            model.model[-1].cqs_config = {**CQS_CONFIG,'enabled':enabled}
            with torch.no_grad(),torch.autocast(device_type='cuda',enabled=amp,dtype=torch.float16):
                for _ in range(2):
                    y = model(x)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(repeats):
                    y = model(x)
                torch.cuda.synchronize()
                rows['cqs' if enabled else 'parent'] = dict(ms_per_forward=(time.perf_counter()-start)*1000/repeats,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),output_finite=bool(torch.isfinite(y[0]).all()))
        report['AMP' if amp else 'FP32'] = rows
    model.model[-1].cqs_config = dict(CQS_CONFIG)
    neck=[]
    def shape_observer(module,inputs):
        neck.extend([list(v.shape) for v in inputs[0]])
    handle=model.model[-1].register_forward_pre_hook(shape_observer)
    with torch.no_grad(),model.model[-1].capture_selection(full=True) as traces:
        model(x)
    handle.remove()
    trace=traces[0]
    require(neck==[[1,256,80,80],[1,256,40,40],[1,256,20,20]],'Real 640 neck shape changed')
    require(trace['pool_size']==600 and sum(h*w for h,w in trace['shapes'])==8400,'Real 640 pool/N changed')
    report['real_640_shapes']=neck
    report['real_encoder_positions']=8400
    report['real_pool_size']=600
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--lifecycle',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = dict(status='RUNNING',runtime=runtime(),formal_training='NOT_RUN',formal_test='NOT_RUN')
    try:
        report['algorithm'] = algorithm_checks(args.device)
        report['mapping_gradient'] = mapping_and_gradient(args.device)
        report['constructor_nc1'] = compare_states(parent_build(nc=1),build())
        if args.source:
            init = args.output.parent/'controlled_init.pt'
            report['initialization'] = initialize(args.source,init)
        if args.lifecycle:
            from cqs_v1_training import local_lifecycle
            report['lifecycle'] = local_lifecycle(args.output.parent,args.device,args.source)
        if args.device == 'cuda':
            report['timing'] = benchmark(build())
        report['status'] = 'PASS'
    except BaseException as error:
        import traceback
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output,report)


if __name__ == '__main__':
    main()
