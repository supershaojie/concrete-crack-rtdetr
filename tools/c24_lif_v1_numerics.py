"""Bounded candidate-aware diagnostics. No production forward or sorting changes.

Tolerances declared before measurements: original LIF FP32 fusion 2e-5/2e-4;
AMP/half 8e-3/4e-2 from C17 pair checker 1e927b8 (test policy only).
Unexplained natural SET_DRIFT remains REQUIRES_REVIEW, even if replay passes.
"""
from __future__ import annotations
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch
import torch
from c24_lif_v1_common import require
from c24_lif_v1_topology import locate

SCHEMA=4
CONTINUOUS=('down','p3','p4','p5','encoder_input','encoder_features','encoder_logits','valid_mask','anchors_valid')
SELECTED=('query','reference','encoder_boxes','encoder_scores','decoder_query_0','decoder_query_1','decoder_query_2','decoder_boxes','decoder_logits','output_boxes','output_scores')
KEYS=CONTINUOUS+('candidate_ids',)+SELECTED
TOLERANCES=dict(fp32=(2e-5,2e-4),amp=(.008,.04),half=(.008,.04))

def compare(a,b,key,atol,rtol):
    row=dict(key=key,shape_a=list(a.shape),shape_b=list(b.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype),
             device_a=str(a.device),device_b=str(b.device),atol=atol,rtol=rtol,status='BLOCKED')
    if a.shape!=b.shape:return dict(row,error='SHAPE_MISMATCH')
    if not a.is_floating_point() or not b.is_floating_point():
        return dict(row,status='PASSED' if a.dtype==b.dtype and torch.equal(a,b) else 'BLOCKED',exact=True)
    a,b=a.detach().double(),b.detach().double()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all());row['finite']=finite
    if not finite:return dict(row,error='NONFINITE')
    d=(a-b).abs();bad=d>atol+rtol*b.abs()
    i=int(d.argmax()) if d.numel() else 0
    coord=[];flat=i
    for dim in reversed(d.shape):coord.insert(0,flat%dim);flat//=dim
    row.update(max_abs=float(d.max()) if d.numel() else 0.,max_rel=float((d/b.abs().clamp_min(1e-12)).max()) if d.numel() else 0.,
        max_coordinate=coord,a=float(a.flatten()[i]) if a.numel() else None,b=float(b.flatten()[i]) if b.numel() else None,
        over_tolerance=int(bad.sum()),status='BLOCKED' if bool(bad.any()) else 'PASSED')
    return row

def require_compare(a,b,key='tensor',atol=2e-6,rtol=2e-5):
    row=compare(a,b,key,atol,rtol);require(row['status']=='PASSED',str(row));return row

def capture(model,x,fixed=None):
    """Record the return indices consumed by native _get_decoder_input's gather.

    The scoped topk wrapper delegates to the original operation exactly once. Replay
    substitutes only its indices in disposable test models; no activations are copied.
    All instance wrappers and hooks are restored even on exceptions.
    """
    top=locate(model.yaml);head=model.model[top['decoder']];records={};handles=[]
    original=head._get_decoder_input;prior=head.__dict__.get('_get_decoder_input');calls=[]
    def save(key):
        def hook(m,args,out):records[key]=out.detach().clone()
        return hook
    for key,index in [('down',top['p3_to_p4']['downsample']),*zip(['p3','p4','p5'],top['decoder_inputs'])]:
        handles.append(model.model[index].register_forward_hook(save(key)))
    handles += [head.enc_output.register_forward_hook(save('encoder_features')),
                head.enc_score_head.register_forward_hook(save('encoder_logits'))]
    handles += [layer.register_forward_hook(save('decoder_query_'+str(i))) for i,layer in enumerate(head.decoder.layers)]
    native_topk=torch.topk
    def selection(scores,k,*args,**kwargs):
        result=native_topk(scores,k,*args,**kwargs)
        require(scores.ndim==2 and k==head.num_queries and kwargs.get('dim')==1,'Unexpected topk schema')
        calls.append(True)
        if fixed is not None:
            ids=fixed.to(device=result.indices.device)
            require(ids.dtype==torch.int64 and ids.shape==result.indices.shape,'Replay ID shape/dtype')
            require(bool(((ids>=0)&(ids<scores.shape[1])).all()),'Illegal replay ID')
            require(all(len(set(r.tolist()))==k for r in ids),'Duplicate replay ID')
            result=type(result)((scores.gather(1,ids),ids))
        records['candidate_ids']=result.indices.detach().clone()
        return result
    def decoder_input(feats,shapes,*args,**kwargs):
        records['encoder_input']=feats.detach().clone()
        with patch('torch.topk',selection):out=original(feats,shapes,*args,**kwargs)
        for key,value in zip(['query','reference','encoder_boxes','encoder_scores'],out):records[key]=value.detach().clone()
        records['valid_mask']=head.valid_mask.detach().clone()
        records['anchors_valid']=head.anchors[head.valid_mask.expand_as(head.anchors)].detach().clone()
        invalid=head.anchors[~head.valid_mask.expand_as(head.anchors)]
        require(bool(torch.isposinf(invalid).all()),'Invalid anchor sentinel changed')
        return out
    head._get_decoder_input=decoder_input
    try:
        with torch.no_grad():out=model(x)
        require(len(calls)==1,'Must observe one actually consumed encoder topk')
        final,raw=out
        records.update(decoder_boxes=raw[0].squeeze(0).detach(),decoder_logits=raw[1].squeeze(0).detach(),
            output_boxes=final[...,:4].detach(),output_scores=final[...,4:].detach())
        require(set(records)==set(KEYS),'Capture schema missing/unexpected keys')
        return records
    finally:
        if prior is None:head.__dict__.pop('_get_decoder_input',None)
        else:head._get_decoder_input=prior
        for h in handles:h.remove()

def identity(a,b,candidates):
    if a.dtype!=torch.int64 or b.dtype!=torch.int64 or a.shape!=b.shape or a.ndim!=2:return 'ILLEGAL_ID'
    for ids in (a,b):
        if not bool(((ids>=0)&(ids<candidates)).all()) or any(len(set(row.tolist()))!=len(row) for row in ids):return 'ILLEGAL_ID'
    if torch.equal(a,b):return 'IDENTICAL'
    if torch.equal(a.sort(1).values,b.sort(1).values):return 'PERMUTATION'
    return 'SET_DRIFT'

def aligned(value,ids,target):
    orders=torch.stack([torch.tensor([row.tolist().index(i) for i in wanted.tolist()],device=value.device) for row,wanted in zip(ids,target)])
    return value.gather(1,orders.unsqueeze(-1).expand(-1,-1,value.shape[-1]))

def compare_records(a,b,mode):
    atol,rtol=TOLERANCES[mode];rows=[]
    missing=[k for k in KEYS if k not in a or k not in b]
    if missing:return dict(schema=SCHEMA,status='BLOCKED',first_error=dict(key=missing[0],error='MISSING_KEY'),missing=missing)
    for key in CONTINUOUS:rows.append(compare(a[key],b[key],key,atol,rtol))
    relation=identity(a['candidate_ids'],b['candidate_ids'],a['encoder_logits'].shape[1])
    natural=[compare(a[k],b[k],k,atol,rtol) for k in SELECTED]
    aligned_rows=[]
    if relation in ('IDENTICAL','PERMUTATION'):
        for key in SELECTED:
            right=b[key] if relation=='IDENTICAL' else aligned(b[key],b['candidate_ids'],a['candidate_ids'])
            aligned_rows.append(compare(a[key],right,key,atol,rtol))
    errors=[r for r in rows+aligned_rows if r['status']!='PASSED']
    if relation=='ILLEGAL_ID':errors.append(dict(key='candidate_ids',error=relation))
    return dict(schema=SCHEMA,mode=mode,continuous=rows,natural_relation=relation,natural_row_comparison=natural,
        id_aligned=aligned_rows,candidate_ids_a=a['candidate_ids'].tolist(),candidate_ids_b=b['candidate_ids'].tolist(),
        first_error=errors[0] if errors else None,
        operator_status='BLOCKED' if errors else 'PASSED',
        status='BLOCKED' if errors else 'REQUIRES_REVIEW' if relation=='SET_DRIFT' else 'PASSED')

def boundary(a,b):
    sa=a['encoder_logits'].detach().double().max(-1).values;sb=b['encoder_logits'].detach().double().max(-1).values
    output=[]
    for x,y,ia,ib in zip(sa,sb,a['candidate_ids'],b['candidate_ids']):
        aa,bb=set(ia.tolist()),set(ib.tolist());change=sorted(aa^bb);error=float((x-y).abs().max())
        ca=float(x[ia].min());cb=float(y[ib].min())
        margins=[dict(id=i,score_a=float(x[i]),score_b=float(y[i]),distance_a=float(abs(x[i]-ca)),distance_b=float(abs(y[i]-cb))) for i in change]
        swaps=[]
        for i in sorted(aa-bb):
            for j in sorted(bb-aa):
                ma=float(x[i]-x[j]);mb=float(y[j]-y[i]);local=float(abs(x[i]-y[i])+abs(x[j]-y[j]))
                swaps.append(dict(a_only=i,b_only=j,margin_a=ma,margin_b=mb,local_perturbation=local,explained=0<=ma<=local and 0<=mb<=local))
        sx=x.sort(descending=True).values;sy=y.sort(descending=True).values;k=len(ia)
        output.append(dict(union=sorted(aa|bb),only_a=sorted(aa-bb),only_b=sorted(bb-aa),cutoff_a=ca,cutoff_b=cb,
            next_a=float(sx[k]) if k<len(sx) else None,next_b=float(sy[k]) if k<len(sy) else None,
            ties_at_cutoff_a=int((x==ca).sum()),ties_at_cutoff_b=int((y==cb).sum()),swaps=swaps,
            max_score_perturbation=error,changed_scores=margins,within_band=all(r['distance_a']<=error and r['distance_b']<=error for r in margins) and all(r['explained'] for r in swaps)))
    return output

def cutoff_acceptance(report,parent=None):
    """Conservative narrow exception; absent or merely similar parent evidence cannot pass."""
    if report.get('operator_status')!='PASSED':return 'BLOCKED'
    if report.get('natural_relation') not in ('IDENTICAL','PERMUTATION','SET_DRIFT'):return 'BLOCKED'
    if report.get('natural_relation')!='SET_DRIFT':return report.get('status','BLOCKED')
    replay=report.get('replay',{})
    if any(replay.get(k,{}).get('status')!='PASSED' for k in ['A','B']):return 'BLOCKED'
    if report.get('mode')!='half':return 'REQUIRES_REVIEW'
    if not report.get('boundary') or not all(r.get('within_band') for r in report['boundary']):return 'REQUIRES_REVIEW'
    if not parent or parent.get('operator_status')!='PASSED' or parent.get('comparability')!='VERIFIED':return 'REQUIRES_REVIEW'
    if any(parent.get('replay',{}).get(k,{}).get('status')!='PASSED' for k in ['A','B']):return 'REQUIRES_REVIEW'
    if any(parent.get(k)!=report.get(k) for k in ['mode','candidate_ids_a','candidate_ids_b']):return 'REQUIRES_REVIEW'
    if not parent.get('boundary') or not all(r.get('within_band') for r in parent['boundary']):return 'REQUIRES_REVIEW'
    if not parent.get('comparison_fingerprints') or parent['comparison_fingerprints']!=report.get('comparison_fingerprints'):return 'REQUIRES_REVIEW'
    return 'ACCEPTED_WITH_BASELINE_CUTOFF_WARNING'

def compare_pair(a,b,x,mode):
    left,right=capture(a,x),capture(b,x)
    report=compare_records(left,right,mode)
    if report.get('natural_relation')=='SET_DRIFT':
        report['boundary']=boundary(left,right);report['replay']={}
        for side,ids in [('A',left['candidate_ids']),('B',right['candidate_ids'])]:
            report['replay'][side]=compare_records(capture(a,x,ids),capture(b,x,ids),mode)
        report['natural_outputs']={side:{k:value[k].detach().float().cpu().tolist() for k in ['output_boxes','output_scores']} for side,value in [('A',left),('B',right)]}
        report['parent_control']=dict(status='NOT_AVAILABLE',reason='Both nonzero modules alter encoder features; arbitrary C2 is not a comparable cutoff control.')
        report['status']=cutoff_acceptance(report)
    return report

def activate(model,scca=True,lif=True,bn=False):
    top=locate(model.yaml)
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(812)
        for m in model.modules():
            if hasattr(m,'scca_o'):
                value=torch.randn(m.scca_o.weight.shape)*.01 if scca else torch.zeros(m.scca_o.weight.shape)
                m.scca_o.weight.copy_(value)
            if hasattr(m,'O_proj'):
                value=torch.randn(m.O_proj.weight.shape)*.01 if lif else torch.zeros(m.O_proj.weight.shape)
                m.O_proj.weight.copy_(value)
        if bn:
            m=model.model[top['p3_to_p4']['downsample']]
            m.bn.running_mean.copy_(torch.linspace(-.3,.4,len(m.bn.running_mean)))
            m.bn.running_var.copy_(torch.linspace(.6,1.5,len(m.bn.running_var)))
            m.bn.weight.copy_(torch.linspace(.8,1.2,len(m.bn.weight)))
            m.bn.bias.copy_(torch.linspace(-.1,.15,len(m.bn.bias)))

def negative_checks():
    def fixture():
        r={k:torch.randn(1,4,2) for k in CONTINUOUS}
        r['encoder_logits']=torch.randn(1,6,1);r['candidate_ids']=torch.tensor([[1,2,3,4]])
        r.update({k:torch.randn(1,4,2) for k in SELECTED});return r
    torch.manual_seed(22);a=fixture();b=deepcopy(a);order=torch.tensor([2,0,3,1])
    b['candidate_ids']=b['candidate_ids'][:,order]
    for k in SELECTED:b[k]=b[k][:,order]
    cases={'permutation':compare_records(a,b,'fp32')['status']=='PASSED'}
    for name,mutate in [('missing_key',lambda b:b.pop('p4')),('illegal_id',lambda b:b['candidate_ids'].fill_(99)),
        ('NaN',lambda b:b['p3'].fill_(float('nan'))),('lost_residual',lambda b:b['down'].add_(1)),('wrong_bn',lambda b:b['down'].mul_(2))]:
        b=deepcopy(a);mutate(b);cases[name]=compare_records(a,b,'fp32')['status']=='BLOCKED'
    cases['empty_report']=compare_records({}, {},'fp32')['status']=='BLOCKED' and cutoff_acceptance({})=='BLOCKED' and cutoff_acceptance(dict(status='PASSED',operator_status='PASSED'))=='BLOCKED'
    report=dict(operator_status='PASSED',natural_relation='SET_DRIFT',mode='half',status='REQUIRES_REVIEW',
        replay={'A':{'status':'PASSED'},'B':{'status':'BLOCKED'}},boundary=[dict(within_band=True)],candidate_ids_a=[[1,2]],candidate_ids_b=[[1,3]])
    cases['only_A_replay']=cutoff_acceptance(report)=='BLOCKED'
    report['replay']['B']['status']='PASSED';parent=deepcopy(report);parent.update(comparability='VERIFIED',candidate_ids_b=[[1,4]])
    cases['different_parent_set']=cutoff_acceptance(report,parent)=='REQUIRES_REVIEW'
    require(all(cases.values()),'Negative gate regression: '+str(cases));return cases
