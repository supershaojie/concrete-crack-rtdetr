"""Ordered tensor semantics and conservative, candidate-aware fusion diagnostics.

Only engineering probes import this module. Model code and evaluation do not.
Allclose direction is a=unfused, b=fused; rtol is relative to abs(b).
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import tempfile
import traceback

import torch
import numpy as np

LIF_KEYS=['lif_input','lif_conv','lif_residual','lif_pre_bn','lif_post_bn']
PRE_KEYS=['scale_0','downsample','scale_1','scale_2','projection_0','projection_1','projection_2',
          'flattened_features','encoder_features','candidate_scores','anchors','valid_mask']
SELECTED_KEYS=['selected_features','selected_anchors','selected_bbox_delta','reference_boxes','decoder_embeddings','enc_boxes','enc_scores']
CBR_KEYS=['cbr_p3','cbr_query','cbr_before','cbr_offsets','cbr_residual','cbr_after']
MODES={'C2':(False,False),'LIF':(True,False),'C19':(False,True),'pair':(True,True)}
PASS={'PASSED','PASS_WITH_CANDIDATE_PERMUTATION'}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class DiagnosticError(RuntimeError):
    def __init__(self,detail):
        self.detail=detail
        super().__init__(json.dumps(detail,ensure_ascii=False,allow_nan=False))


def fail(kind,**detail):
    raise DiagnosticError(dict(status=kind,**detail))


def schema(records):
    mode=records.get('mode');layers=records.get('layers')
    if mode not in MODES or not isinstance(layers,int) or layers<1:
        fail('FAILED_INCOMPLETE_DIAGNOSTIC',key='schema',mode=mode,layers=layers)
    lif,cbr=MODES[mode]
    return ((LIF_KEYS if lif else [])+PRE_KEYS+['candidate_indices']+SELECTED_KEYS+
            [f'query_layer_{i}' for i in range(layers)]+['raw_boxes','raw_scores']+
            (['returned_query']+CBR_KEYS if cbr else [])+['boxes','scores'])


def validate(records,stage):
    keys=schema(records)
    for key in keys+['native_candidate_indices']:
        if key not in records or not isinstance(records[key],torch.Tensor):
            fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key=key,reason='Required tensor missing')
    if not records.get('actual_gather_verified') or records.get('topk_calls')!=1:
        fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key='candidate_indices',reason='Actual gather/count not verified')
    if not records.get('shapes') or 'dn_split' not in records:
        fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key='shapes/dn_split',reason='Metadata missing')
    for key in ('candidate_indices','native_candidate_indices'):
        if records[key].dtype!=torch.int64:
            fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key=key,reason='IDs must be int64')
    for key in keys:
        expected=torch.bool if key=='valid_mask' else (torch.int64 if key=='candidate_indices' else None)
        value=records[key]
        if (expected is not None and value.dtype!=expected) or (expected is None and not value.is_floating_point()):
            fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key=key,reason='Wrong semantic dtype',dtype=str(value.dtype))
    return keys


def tensor_compare(a,b,key,stage,atol,rtol,device='cpu',precision='fp32',candidate_ids_available=False):
    info=dict(key=key,stage=stage,device=device,precision=precision,shape=list(a.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype),
              atol=atol,rtol=rtol,candidate_ids_available=candidate_ids_available,relative_reference='b (fused/comparison target); denominator=max(abs(b),1e-12)')
    floating=a.is_floating_point() and b.is_floating_point()
    if a.shape!=b.shape or (a.dtype!=b.dtype and not (precision=='amp' and floating)):
        fail('FAILED_INCOMPLETE_DIAGNOSTIC',**info,shape_b=list(b.shape),reason='Shape/dtype mismatch')
    if not floating:
        bad=a!=b
        exact_deltas=[abs(int(x)-int(y)) for x,y in zip(a[bad].tolist(),b[bad].tolist())]
        result=dict(info,max_abs_error=max(exact_deltas,default=0),max_rel_error=None,allclose_failed_count=int(bad.sum()))
    else:
        # Float64 is only for reporting arithmetic; allclose itself uses original tensors/dtypes.
        finite_a=torch.isfinite(a);finite_b=torch.isfinite(b)
        if key in ('anchors','selected_anchors'):
            if torch.isnan(a).any() or torch.isnan(b).any() or not torch.equal(finite_a,finite_b) or not torch.equal(a[~finite_a],b[~finite_b]):
                fail('FAILED_REAL_NUMERICAL_MISMATCH',**info,reason='Masked anchor nonfinite mask/value mismatch')
        elif not finite_a.all() or not finite_b.all():
            fail('FAILED_REAL_NUMERICAL_MISMATCH',**info,reason='Unexpected NaN/Inf')
        aa=a.double();bb=b.double();diff=torch.where(finite_a,(aa-bb).abs(),0.)
        relative=diff/bb.abs().clamp_min(1e-12)
        # AMP can legitimately yield FP32/FP16 at a hook boundary; compare in promoted dtype.
        ca,cb=(a.float(),b.float()) if a.dtype!=b.dtype else (a,b)
        bad=~torch.isclose(ca,cb,atol=atol,rtol=rtol,equal_nan=False)
        bad &= finite_a
        result=dict(info,max_abs_error=float(diff.max()) if diff.numel() else 0.,max_rel_error=float(relative.max()) if relative.numel() else 0.,
                    allclose_failed_count=int(bad.sum()),masked_nonfinite_count=int((~finite_a).sum()))
    if bad.any():
        # Report largest absolute failure, not an arbitrary set iteration entry.
        if floating:
            ranked=(a.double()-b.double()).abs().masked_fill(~bad,-1)
            flat=int(ranked.flatten().argmax())
        else:flat=int(bad.flatten().nonzero()[exact_deltas.index(max(exact_deltas))])
        index=[];rest=flat
        for size in reversed(a.shape):index.insert(0,rest%size);rest//=size
        av=a.flatten()[flat].item();bv=b.flatten()[flat].item()
        fail('FAILED_REAL_NUMERICAL_MISMATCH',**result,index=index,a=av,b=bv,reference_denominator=max(abs(bv),1e-12),
             reason='Exact identity mismatch' if not floating else 'Original allclose tolerance exceeded')
    return result


def compare_records(a,b,atol=2e-6,rtol=2e-5,stage='parent',device='cpu',precision='fp32',keys=None,collect=False):
    ak=validate(a,stage);bk=validate(b,stage)
    if a['layers']!=b['layers'] or a['shapes']!=b['shapes'] or a['dn_split']!=b['dn_split']:
        fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key='metadata',reason='Layer/shape/DN mismatch')
    if keys is None:
        # Explicit mode projection: validate each full schema first, then omit only
        # the known absent module taps in a parent-vs-pair degeneration.
        omit=[]
        if not (MODES[a['mode']][0] and MODES[b['mode']][0]):omit+=LIF_KEYS
        if not (MODES[a['mode']][1] and MODES[b['mode']][1]):omit+=['returned_query']+CBR_KEYS
        keys=[k for k in ak if k not in omit]
        if keys!=[k for k in bk if k not in omit]:fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key='schema',reason='Unexpected mode difference')
    report={}
    for key in keys:
        if key not in a or key not in b:fail('FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,key=key,reason='Required comparison key missing')
        try:report[key]=tensor_compare(a[key],b[key],key,stage,atol,rtol,device,precision,True)
        except DiagnosticError as error:
            report[key]=error.detail
            if not collect:
                error.partial=report
                raise
    return report


def coordinates(ids,shapes):
    result=[]
    for row in ids.tolist():
        mapped=[]
        for value in row:
            offset=value
            for level,(h,w) in enumerate(shapes):
                if offset<h*w:mapped.append([value,level,offset//w,offset%w]);break
                offset-=h*w
            else:fail('FAILED_INCOMPLETE_DIAGNOSTIC',key='candidate_indices',reason='Out of grid ID')
        result.append(mapped)
    return result


def selection_report(a,b):
    ia,ib=a['candidate_indices'],b['candidate_indices'];sa=a['candidate_scores'].max(-1).values;sb=b['candidate_scores'].max(-1).values
    if ia.shape!=ib.shape:fail('FAILED_INCOMPLETE_DIAGNOSTIC',key='candidate_indices',reason='ID shape changed')
    rows=[];same_set=True;order=True
    for i,(ra,rb) in enumerate(zip(ia.tolist(),ib.tolist())):
        if len(set(ra))!=len(ra) or len(set(rb))!=len(rb):fail('FAILED_INCOMPLETE_DIAGNOSTIC',key='candidate_indices',reason='Duplicate ID')
        left=sorted(set(ra)-set(rb));right=sorted(set(rb)-set(ra));same_set &= not left and not right;order &= ra==rb
        e=float((sa[i]-sb[i]).abs().max());slack=4*torch.finfo(sa.dtype).eps*max(float(sa[i].abs().max()),float(sb[i].abs().max()),1.)
        margins=[dict(a_only=x,b_only=y,margin_a=float(sa[i,x]-sa[i,y]),margin_b=float(sb[i,y]-sb[i,x])) for x in left for y in right]
        explained=all(-slack<=r['margin_a']<=2*e+slack and -slack<=r['margin_b']<=2*e+slack for r in margins)
        def gap(scores,ids):
            mask=torch.ones(scores.numel(),dtype=torch.bool);mask[ids]=False
            return float(scores[ids].min()-scores[mask].max()) if mask.any() else None
        rows.append(dict(image=i,natural_a=ra,natural_b=rb,changed_positions=sum(x!=y for x,y in zip(ra,rb)),a_only=left,b_only=right,
                         common_count=len(set(ra)&set(rb)),score_disturbance_e=e,margin_bound=2*e+slack,floating_slack=slack,
                         exchanges=margins,margin_explained=explained,cutoff_gap_a=gap(sa[i],ra),cutoff_gap_b=gap(sb[i],rb)))
    return dict(kind='IDENTICAL' if order else ('PERMUTATION' if same_set else 'SET_DRIFT'),images=rows,
                coordinates_a=coordinates(ia,a['shapes']),coordinates_b=coordinates(ib,b['shapes']))


def align_to_ids(a,b):
    result=dict(b);ids=a['candidate_indices'];other=b['candidate_indices']
    orders=[]
    for ra,rb in zip(ids.tolist(),other.tolist()):
        if set(ra)!=set(rb) or len(set(rb))!=len(rb):raise RuntimeError('Alignment requires a candidate ID bijection')
        lookup={value:j for j,value in enumerate(rb)};orders.append([lookup[value] for value in ra])
    order=torch.tensor(orders,dtype=torch.long)
    # Eval-only: decoder/CBR train DN queries are deliberately outside this protocol.
    if a['dn_split'] is not None or b['dn_split'] is not None:raise RuntimeError('Fusion alignment requires eval without DN')
    for key in schema(b):
        if key in LIF_KEYS+PRE_KEYS+['cbr_p3']:continue
        value=b[key]
        axis=2 if key in ('raw_boxes','raw_scores','boxes','scores') else 1
        if axis==2:
            index=order[None,:,:,None].expand(value.shape[0],-1,-1,value.shape[-1])
        else:index=order.reshape(*order.shape,*([1]*(value.ndim-2))).expand(*order.shape,*value.shape[2:])
        result[key]=value.gather(axis,index)
    return result


def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(),numpy=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    if 'python' in state:random.setstate(state['python'])
    if 'numpy' in state:np.random.set_state(state['numpy'])


def fusion_protocol(unfused,fused,image,folder,device,precision='fp32',parent_factory=None):
    from c19_lif_v1_probe import capture
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',acceptance='BLOCKED',device=device,precision=precision,formal_training='NOT_RUN',
                runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,threads=torch.get_num_threads(),
                             tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                             deterministic=torch.are_deterministic_algorithms_enabled()),
                pre_selection={},selection={},natural_outputs={},common_id_alignment={},fixed_query_replay={},failure=None)
    report['natural_outputs_scope']='Unaligned row comparisons; leaf mismatches do not diagnose an operator until candidate identity/alignment/replay is checked'
    atol,rtol=(2e-5,2e-4) if precision=='fp32' else (3e-3,3e-2)
    report['tolerances']=dict(atol=atol,rtol=rtol)
    state=rng_state();records={};stage='fixture'
    def persist():atomic_json(folder/'fuse_diagnostic.json',report)
    def run(model,ids=None):
        restore_rng(state)
        with torch.no_grad(),(torch.autocast(device_type=device,dtype=torch.float16) if precision=='amp' else nullcontext()):
            return capture(model,image,fixed_ids=ids)[1]
    try:
        # Save before instrumenting. Only state tensors, not patched model objects.
        fixture=folder/'fixture.pt'
        torch.save(dict(image=image.detach().cpu(),unfused={k:v.detach().cpu().clone() for k,v in unfused.state_dict().items()},
                        fused={k:v.detach().cpu().clone() for k,v in fused.state_dict().items()},rng=state,yaml=unfused.yaml,
                        nc=1,precision=precision,device=device),fixture)
        report['fixture']=dict(path=str(fixture),sha256=digest(fixture))
        source=Path(__file__).resolve().parents[1]
        report['source_hashes']={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in [Path(__file__),source/'tools/c19_lif_v1_probe.py']+
                                 [source/'ultralytics-main/ultralytics/nn/modules'/n for n in ('lif_down.py','cbr.py','head.py','transformer.py')]+[source/'ultralytics-main/ultralytics/nn/tasks.py']}
        persist();stage='natural_trace'
        records['natural_a']=a=run(unfused);records['natural_b']=b=run(fused)
        validate(a,stage);validate(b,stage)
        if a['mode']!=b['mode']:fail('FAILED_INCOMPLETE_DIAGNOSTIC',key='mode',reason='Fusion changed topology')
        pre=(LIF_KEYS if MODES[a['mode']][0] else [])+PRE_KEYS
        downstream=[k for k in schema(a) if k not in pre+['candidate_indices']]
        stage='pre_selection'
        report[stage]=compare_records(a,b,atol,rtol,device+'.'+precision+'.fuse.'+stage,device,precision,pre);persist()
        stage='selection';report[stage]=selection_report(a,b);persist()
        stage='natural_outputs'
        report[stage]=compare_records(a,b,atol,rtol,device+'.'+precision+'.fuse.natural',device,precision,downstream,collect=True);persist()
        if report['selection']['kind']!='SET_DRIFT':
            stage='common_id_alignment';aligned=align_to_ids(a,b);records['aligned_b']=aligned
            report[stage]=compare_records(a,aligned,atol,rtol,device+'.'+precision+'.fuse.aligned',device,precision,downstream);persist()
        else:report['common_id_alignment']=dict(status='NOT_APPLICABLE',reason='Different candidate sets; no intersection-only acceptance')
        stage='fixed_query_replay'
        records['replay_a']=ra=run(unfused,a['candidate_indices']);records['replay_b']=rb=run(fused,a['candidate_indices'])
        report[stage]=compare_records(ra,rb,atol,rtol,device+'.'+precision+'.fuse.replay',device,precision);persist()
        if report['selection']['kind']=='SET_DRIFT':
            stage='parent_selection'
            if parent_factory is not None:
                # Same input and RNG, with parent's own continuous operator path.
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
                    parent=parent_factory().eval().to(device)
                    parent_fused=deepcopy(parent.float()).fuse(verbose=False)
                    if precision=='half':parent.half();parent_fused.half()
                    pa=run(parent);pb=run(parent_fused)
                    records['parent_a']=pa;records['parent_b']=pb
                    report[stage]=dict(selection=selection_report(pa,pb),pre_selection=compare_records(pa,pb,atol,rtol,device+'.'+precision+'.parent',device,precision,PRE_KEYS,collect=True),
                                       natural_outputs=compare_records(pa,pb,atol,rtol,device+'.'+precision+'.parent.natural',device,precision,SELECTED_KEYS+['raw_boxes','raw_scores','boxes','scores'],collect=True))
            else:report[stage]=dict(status='NOT_RUN',reason='Parent factory unavailable')
            report.update(status='REQUIRES_REVIEW',acceptance='BLOCKED',operator_status='OPERATOR_CHECK_PASSED',natural_status='NATURAL_SELECTION_DRIFT',
                          reason='Changed candidate membership requires review even when margins are explainable')
        else:
            report.update(status='PASSED' if report['selection']['kind']=='IDENTICAL' else 'PASS_WITH_CANDIDATE_PERMUTATION',acceptance='PASSED',operator_status='OPERATOR_CHECK_PASSED')
    except BaseException as error:
        if hasattr(error,'trace_records'):records['incomplete_trace']=error.trace_records
        report['failure']=getattr(error,'detail',dict(status='FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,error=repr(error),traceback=traceback.format_exc()))
        report['failure']['phase']=stage
        if hasattr(error,'partial'):report[stage]=error.partial
        report.update(status=report['failure']['status'],acceptance='BLOCKED')
        raise
    finally:
        restore_rng(state)
        # These records preserve natural differences, not just the matched subset.
        try:
            evidence=folder/'records.pt';torch.save(records,evidence)
            report['records']=dict(path=str(evidence),sha256=digest(evidence))
        except BaseException as error:
            report.update(status='FAILED_INCOMPLETE_DIAGNOSTIC',acceptance='BLOCKED',evidence_error=repr(error))
            persist();raise
        persist()
    return report
