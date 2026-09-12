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
    from c19_lif_v1_cutoff import selection_from_scores
    return selection_from_scores(a['candidate_indices'].tolist(),b['candidate_indices'].tolist(),
        a['candidate_scores'].max(-1).values.double().tolist(),b['candidate_scores'].max(-1).values.double().tolist(),
        a['shapes'],str(a['candidate_scores'].dtype),str(b['candidate_scores'].dtype))


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


def fusion_protocol(unfused,fused,image,folder,device,precision='fp32',parent_factory=None,parent_source=None,fixture_reference=None):
    from c19_lif_v1_probe import capture
    from c19_lif_v1_cutoff import VERSION,review_evidence,fusion_accepted
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=VERSION,status='RUNNING',acceptance='BLOCKED',device=device,precision=precision,formal_training='NOT_RUN',
                runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,threads=torch.get_num_threads(),
                             tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                             deterministic=torch.are_deterministic_algorithms_enabled()),
                pre_selection={},selection={},natural_outputs={},common_id_alignment={},fixed_query_replay={},failure=None)
    report['natural_outputs_scope']='Unaligned row comparisons; leaf mismatches do not diagnose an operator until candidate identity/alignment/replay is checked'
    atol,rtol=(2e-5,2e-4) if precision=='fp32' else (3e-3,3e-2)
    report['tolerances']=dict(atol=atol,rtol=rtol)
    state=rng_state();records={};stage='fixture';report['traces']={}
    report['record_schema']=dict(mode='pair',layers=len(unfused.model[-1].decoder.layers))
    report['precision_path']=dict(model_a_dtype=str(next(unfused.parameters()).dtype),model_b_dtype=str(next(fused.parameters()).dtype),
                                  input_dtype=str(image.dtype),autocast=precision=='amp')
    def persist():atomic_json(folder/'fuse_diagnostic.json',report)
    def trace(record):
        return {k:record[k].tolist() if isinstance(record[k],torch.Tensor) else record[k] for k in
                ('actual_gather_verified','topk_calls','candidate_indices','native_candidate_indices')}
    def run(model,ids=None):
        restore_rng(state)
        with torch.no_grad(),(torch.autocast(device_type=device,dtype=torch.float16) if precision=='amp' else nullcontext()):
            return capture(model,image,fixed_ids=ids)[1]
    try:
        # A normal full preflight shares one FP32 fixture per device across modes.
        # Successes retain compact JSON only; failed records remain on the server.
        if fixture_reference:report['fixture']=fixture_reference
        lif_a=next(m for m in unfused.modules() if type(m).__name__=='LIFDown')
        lif_b=next(m for m in fused.modules() if type(m).__name__=='LIFDown')
        report['fusion_invariants']=dict(lif_bn_retained=hasattr(lif_b,'bn'),
            lif_state_exact=set(lif_a.state_dict())==set(lif_b.state_dict()) and all(torch.equal(v,lif_b.state_dict()[k]) for k,v in lif_a.state_dict().items()),
            other_bn_fused=sum(isinstance(m,torch.nn.BatchNorm2d) for m in fused.modules())<sum(isinstance(m,torch.nn.BatchNorm2d) for m in unfused.modules()))
        if not all(report['fusion_invariants'].values()):
            fail('FAILED_REAL_NUMERICAL_MISMATCH',stage='fusion_invariants',key='lif_bn/state',reason='LIF residual/BN state or ordinary fusion changed')
        source=Path(__file__).resolve().parents[1]
        report['source_hashes']={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in [Path(__file__),source/'tools/c19_lif_v1_probe.py',source/'tools/c19_lif_v1_cutoff.py']+
                                 [source/'ultralytics-main/ultralytics/nn/modules'/n for n in ('lif_down.py','cbr.py','head.py','transformer.py')]+[source/'ultralytics-main/ultralytics/nn/tasks.py']}
        persist();stage='natural_trace'
        records['natural_a']=a=run(unfused);records['natural_b']=b=run(fused)
        validate(a,stage);validate(b,stage)
        report['traces'].update(natural_a=trace(a),natural_b=trace(b))
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
        report[stage]=compare_records(ra,rb,atol,rtol,device+'.'+precision+'.fuse.replay',device,precision)
        report['traces'].update(replay_A_a=trace(ra),replay_A_b=trace(rb));persist()
        if report['selection']['kind']=='SET_DRIFT':
            report['natural_status']='NATURAL_SELECTION_DRIFT'
            stage='fixed_query_replay_B'
            records['replay_B_a']=ra=run(unfused,b['candidate_indices']);records['replay_B_b']=rb=run(fused,b['candidate_indices'])
            report[stage]=compare_records(ra,rb,atol,rtol,device+'.'+precision+'.fuse.replay_B',device,precision)
            report['traces'].update(replay_B_a=trace(ra),replay_B_b=trace(rb));persist()
            stage='parent_selection';parent_report=report[stage]={}
            if parent_factory is not None:
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
                    parent=parent_factory().eval().float()
                    source_state=(parent_source if parent_source is not None else unfused).state_dict()
                    ps=parent.state_dict();us=unfused.state_dict()
                    shape_ok=set(ps)<=set(source_state) and all(v.shape==source_state[k].shape for k,v in ps.items())
                    values_ok=shape_ok and all(torch.equal(v.cpu(),source_state[k].cpu()) for k,v in ps.items())
                    cast_ok=shape_ok and all(torch.equal(v.to(us[k].device,us[k].dtype),us[k]) for k,v in ps.items())
                    h=hashlib.sha256()
                    for key,v in sorted(ps.items()):
                        h.update(key.encode());h.update(str(tuple(v.shape)).encode());h.update(v.cpu().contiguous().numpy().tobytes())
                    parent_report['mapping']=dict(status='PASSED' if shape_ok and values_ok and cast_ok else 'FAILED',common_states=len(ps),
                        all_shapes_exact=shape_ok,all_values_exact=values_ok,pair_cast_exact=cast_ok,
                        source_dtype=str(next((parent_source if parent_source is not None else unfused).parameters()).dtype),
                        same_input_rng_precision=True,source_state_sha256=h.hexdigest(),fuse_order='float32_fuse_then_half' if precision=='half' else 'float32_fuse')
                    parent=parent.to(device);parent_fused=deepcopy(parent).fuse(verbose=False)
                    if precision=='half':parent.half();parent_fused.half()
                    records['parent_a']=pa=run(parent);records['parent_b']=pb=run(parent_fused)
                    if pa['mode']!='C2' or pb['mode']!='C2':raise RuntimeError('Parent must be original C2')
                    parent_report['selection']=selection_report(pa,pb)
                    parent_report['traces']=dict(natural_a=trace(pa),natural_b=trace(pb))
                    # Strict, deliberately not collect=True: parent failure is not evidence of equivalence.
                    parent_report['pre_selection']=compare_records(pa,pb,atol,rtol,device+'.'+precision+'.parent',device,precision,PRE_KEYS)
                    parent_post=[k for k in schema(pa) if k not in PRE_KEYS+['candidate_indices']]
                    parent_report['natural_outputs']=compare_records(pa,pb,atol,rtol,device+'.'+precision+'.parent.natural',device,precision,parent_post,collect=True)
                    for label,ids in [('A',a['candidate_indices']),('B',b['candidate_indices'])]:
                        left=run(parent,ids);right=run(parent_fused,ids)
                        records['parent_replay_'+label+'_a']=left;records['parent_replay_'+label+'_b']=right
                        parent_report['replay_'+label]=compare_records(left,right,atol,rtol,device+'.'+precision+'.parent.replay_'+label,device,precision)
                        parent_report['traces']['replay_'+label+'_a']=trace(left);parent_report['traces']['replay_'+label+'_b']=trace(right)
                        persist()
            else:parent_report.update(status='NOT_RUN',reason='Parent factory unavailable')
        else:report['natural_status']='IDENTICAL_CANDIDATES' if report['selection']['kind']=='IDENTICAL' else 'CANDIDATE_PERMUTATION'
        report['operator_status']='OPERATOR_CHECK_PASSED'
        report['review']=review_evidence(report)
        report.update({k:v for k,v in report['review'].items() if k!='accepted'})
    except BaseException as error:
        if hasattr(error,'trace_records'):records['incomplete_trace']=error.trace_records
        report['failure']=getattr(error,'detail',dict(status='FAILED_INCOMPLETE_DIAGNOSTIC',stage=stage,error=repr(error),traceback=traceback.format_exc()))
        report['failure']['phase']=stage
        if hasattr(error,'partial'):
            if stage=='parent_selection':report[stage]['failed_comparison']=error.partial
            else:report[stage]=error.partial
        report.update(status=report['failure']['status'],acceptance='BLOCKED')
        raise
    finally:
        restore_rng(state)
        # Full tensors are local failure evidence, never part of LIGHT uploads.
        try:
            if not fusion_accepted(report):
                if not fixture_reference:
                    fixture=folder/'fixture.pt'
                    torch.save(dict(image=image.detach().cpu(),unfused={k:v.detach().cpu() for k,v in unfused.state_dict().items()},
                        fused={k:v.detach().cpu() for k,v in fused.state_dict().items()},rng=state,yaml=unfused.yaml,nc=1,precision=precision,device=device),fixture)
                    report['fixture']=dict(path=str(fixture),sha256=digest(fixture))
                evidence=folder/'records.pt';torch.save(records,evidence)
                report['records']=dict(path=str(evidence),sha256=digest(evidence),upload='EXCLUDED; use LIGHT JSON only')
            else:report['records']=dict(status='COMPACT_JSON_ONLY',reason='Successful diagnostic; no redundant full model/tensor copies')
        except BaseException as error:
            report.update(status='FAILED_INCOMPLETE_DIAGNOSTIC',acceptance='BLOCKED',evidence_error=repr(error))
            persist();raise
        persist()
    return report
