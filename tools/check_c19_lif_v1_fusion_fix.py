"""Positive and injected-failure tests; no training or complete evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import torch
from init_c19_lif_v1 import build,require,runtime
from check_c19_lif_v1 import activate
from c19_lif_v1_probe import capture
from c19_lif_v1_diagnostic import (DiagnosticError,atomic_json,compare_records,tensor_compare,selection_report,
                                   align_to_ids,fusion_protocol,schema,PASS)
from train_c19_lif_v1 import require_preflight
from ultralytics.utils.patches import torch_load


def rejected(action,kind=None):
    try:action()
    except (RuntimeError,AssertionError,AttributeError) as error:
        if kind:require(isinstance(error,DiagnosticError) and error.detail['status']==kind,'Wrong diagnostic error: '+repr(error))
        return getattr(error,'detail',dict(error=repr(error)))
    raise AssertionError('Fault was accepted')


def run(output,fixture):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report=dict(status='FAILED',runtime=runtime(),formal_training='NOT_RUN',full_val_test='NOT_RUN',tests={})
    rows=report['tests']
    try:
        f=torch_load(fixture,map_location='cpu');require(f['precision']=='fp32' and f['device']=='cpu','Use the unchanged CPU fixture')
        model=build('pair',nc=1).eval();model.load_state_dict(f['unfused'],strict=True);x=f['image']
        with torch.no_grad():_,a=capture(model,x)
        for key in ('lif_conv','candidate_scores','query_layer_0','cbr_after'):
            other=dict(a);other[key]=a[key].clone();other[key].flatten()[5]+=.1
            detail=rejected(lambda:compare_records(a,other,stage='fault.'+key),'FAILED_REAL_NUMERICAL_MISMATCH')
            require(detail['key']==key and detail['device']=='cpu' and detail['allclose_failed_count']==1 and 'index' in detail,'Missing error context')
            rows['tensor_'+key]=detail
        missing=dict(a);missing.pop('raw_boxes')
        rows['missing_required']=rejected(lambda:compare_records(a,missing),'FAILED_INCOMPLETE_DIAGNOSTIC')
        rows['integer_identity']=rejected(lambda:tensor_compare(torch.tensor([2**30]),torch.tensor([2**30+1]),'candidate_indices','identity',2e-5,2e-4),'FAILED_REAL_NUMERICAL_MISMATCH')
        from check_lif_down import compare as legacy_compare
        rows['legacy_float_ID_false_pass']=legacy_compare(torch.tensor([2**30]),torch.tensor([2**30+1]))
        wrong=dict(a);wrong['candidate_indices']=a['candidate_indices'].float()
        rows['wrong_id_dtype']=rejected(lambda:compare_records(a,wrong),'FAILED_INCOMPLETE_DIAGNOSTIC')
        shape=dict(a);shape['raw_scores']=a['raw_scores'][...,:0]
        rows['wrong_shape']=rejected(lambda:compare_records(a,shape),'FAILED_INCOMPLETE_DIAGNOSTIC')
        nan=dict(a);nan['scale_0']=a['scale_0'].clone();nan['scale_0'].flatten()[0]=float('nan')
        rows['nonfinite']=rejected(lambda:compare_records(a,nan),'FAILED_REAL_NUMERICAL_MISMATCH')
        anchor=torch.tensor([0.,float('inf')]);rows['legal_masked_anchor']=tensor_compare(anchor,anchor,'anchors','mask',2e-5,2e-4)
        rows['anchor_mask_changed']=rejected(lambda:tensor_compare(anchor,torch.zeros_like(anchor),'anchors','mask',2e-5,2e-4),'FAILED_REAL_NUMERICAL_MISMATCH')
        require(selection_report(a,a)['kind']=='IDENTICAL','Same order not recognized')
        permuted=align_to_ids(dict(a,candidate_indices=a['candidate_indices'].flip(1)),a)
        require(selection_report(a,permuted)['kind']=='PERMUTATION','Permutation not recognized')
        compare_records(a,align_to_ids(a,permuted));rows['same_set_bijection']='PASSED; all post-selection records restored by actual ID'
        rows['legacy_row_permutation_false_failure']=rejected(lambda:legacy_compare(a['raw_boxes'],permuted['raw_boxes'],2e-5,2e-4))
        drift=dict(a);drift['candidate_indices']=a['candidate_indices'].clone()
        unused=next(i for i in range(a['candidate_scores'].shape[1]) if i not in set(a['candidate_indices'][0].tolist()))
        drift['candidate_indices'][0,0]=unused
        require(selection_report(a,drift)['kind']=='SET_DRIFT','Membership drift hidden')
        rows['different_set_rejected']=rejected(lambda:align_to_ids(a,drift))
        original_topk=torch.topk
        with torch.no_grad():
            original=model(x)[0].clone();_,observed=capture(model,x);after=model(x)[0]
        require(torch.equal(original,after) and observed['actual_gather_verified'] and torch.topk is original_topk,'Probe changed production output/topk')
        require(not model.model[-1]._forward_hooks and '_get_decoder_input' not in model.model[-1].__dict__,'Hook/method leaked')
        # Exceptions during setup and during forward both restore global and instance state.
        damaged=deepcopy(model);del damaged.model[20].bn
        rejected(lambda:capture(damaged,x))
        require(torch.topk is original_topk and '_get_decoder_input' not in damaged.model[-1].__dict__,'Setup exception leaked patch')
        hook=model.model[-1].enc_output.register_forward_hook(lambda *args:(_ for _ in ()).throw(RuntimeError('Injected trace failure')))
        try:rejected(lambda:capture(model,x))
        finally:hook.remove()
        with torch.no_grad():require(torch.equal(original,model(x)[0]),'Exception cleanup changed production output')
        rows['native_gather_and_cleanup']='PASSED: actual selected features and reference anchors verified; normal/setup/forward exception cleanup exact'
        for label in ('lif_residual_disabled','lif_bn_broken','encoder_score','query','cbr_output','trace_exception'):
            fused=deepcopy(model).fuse(verbose=False);handles=[];context=None
            if label=='lif_residual_disabled':
                native=fused.model[20].residual
                context=patch.object(fused.model[20],'residual',lambda image:native(image)*0)
            elif label=='lif_bn_broken':
                with torch.no_grad():fused.model[20].bn.running_mean.add_(.1)
            elif label=='encoder_score':handles.append(fused.model[-1].enc_score_head.register_forward_hook(lambda m,args,v:v+.1))
            elif label=='query':handles.append(fused.model[-1].decoder.layers[0].register_forward_hook(lambda m,args,v:v+.1))
            elif label=='cbr_output':handles.append(fused.model[-1].cbr.register_forward_hook(lambda m,args,v:(v[0]+.1,v[1])))
            else:handles.append(fused.model[-1].enc_output.register_forward_hook(lambda *args:(_ for _ in ()).throw(RuntimeError('Injected trace failure'))))
            try:
                if context:context.__enter__()
                detail=rejected(lambda:fusion_protocol(model,fused,x,output/label,'cpu'))
            finally:
                if context:context.__exit__(None,None,None)
                for handle in handles:handle.remove()
            diagnostic=json.loads((output/label/'fuse_diagnostic.json').read_text(encoding='utf-8'))
            expected='FAILED_INCOMPLETE_DIAGNOSTIC' if label=='trace_exception' else 'FAILED_REAL_NUMERICAL_MISMATCH'
            require(diagnostic['status']==expected and diagnostic['acceptance']=='BLOCKED','Injected fault misclassified: '+label)
            require((output/label/'records.pt').is_file() and (output/label/'fixture.pt').is_file(),'Failure tensors not saved')
            require(torch.topk is original_topk,'Protocol exception leaked topk')
            rows[label]=dict(status='PASSED (injected failure detected)',failure=diagnostic['failure'],partial_keys=list(diagnostic['pre_selection']))
        # End-to-end membership drift with real native gather, via explicit test-only IDs.
        # Scores/features/model are untouched, so an unexplained switch cannot be excused.
        fused=deepcopy(model).fuse(verbose=False)
        from c19_lif_v1_probe import capture as native_capture
        count=[0]
        def force_drift(m,image,batch=None,fixed_ids=None):
            count[0]+=1
            ids=drift['candidate_indices'] if count[0]==2 else fixed_ids
            return native_capture(m,image,batch,fixed_ids=ids)
        def parent():
            p=build('C2',nc=1).eval();p.load_state_dict({k:model.state_dict()[k] for k in p.state_dict()},strict=True);return p
        with patch('c19_lif_v1_probe.capture',side_effect=force_drift):
            d=fusion_protocol(model,fused,x,output/'set_drift','cpu',parent_factory=parent)
        require(d['status']=='REQUIRES_REVIEW' and d['operator_status']=='OPERATOR_CHECK_PASSED' and d['acceptance']=='BLOCKED','Set drift incorrectly accepted')
        require(d['parent_selection']['selection'],'Parent sensitivity absent')
        rows['end_to_end_set_drift']=dict(status=d['status'],selection=d['selection']['kind'],margin_explained=d['selection']['images'][0]['margin_explained'])
        # A fault exclusively in fixed-query replay must still fail that stage.
        count[0]=0
        def replay_fault(m,image,batch=None,fixed_ids=None):
            count[0]+=1
            handle=None
            if count[0]==4:handle=m.model[-1].decoder.layers[0].register_forward_hook(lambda mod,args,v:v+.1)
            try:return native_capture(m,image,batch,fixed_ids=fixed_ids)
            finally:
                if handle:handle.remove()
        with patch('c19_lif_v1_probe.capture',side_effect=replay_fault):
            rejected(lambda:fusion_protocol(model,fused,x,output/'replay_fault','cpu'),'FAILED_REAL_NUMERICAL_MISMATCH')
        replay=json.loads((output/'replay_fault/fuse_diagnostic.json').read_text(encoding='utf-8'))
        require(replay['failure']['phase']=='fixed_query_replay','Replay failure misclassified')
        rows['fixed_query_fault']=replay['failure']
        # Exercise the real checker's failure/finally writer without expensive repeats.
        import check_c19_lif_v1 as checker
        checker_out=output/'partial_checker'
        argv=['check_c19_lif_v1.py','--source',str(fixture),'--initialized',str(fixture),'--real-dataset',str(output),'--output',str(checker_out)]
        with patch('sys.argv',argv),patch.object(checker,'module_checks',side_effect=RuntimeError('Injected initial-stage failure')):
            rejected(checker.main)
        partial=json.loads((checker_out/'checks.json').read_text(encoding='utf-8'))
        require(partial['status']=='FAILED' and partial['capacity']['status']=='NOT_RUN' and 'failure' in partial,'finally hid failure')
        rows['partial_checks_json']='PASSED: atomic real checker failure report; failure remains FAILED'
        valid=dict(status='PASSED',capacity=dict(status='PASSED',batch=16,imgsz=640,AMP=True,loss=1.,optimizer_steps=0),
                   cpu=dict(fuse={'FP32':dict(status='PASSED',acceptance='PASSED')}),
                   cuda=dict(fuse={k:dict(status='PASSED',acceptance='PASSED') for k in ('FP32','amp_fuse','half_fuse')}))
        rows['status_only_proof_rejected']=rejected(lambda:require_preflight(valid))
        for status in ('REQUIRES_REVIEW','FAILED_REAL_NUMERICAL_MISMATCH','FAILED_INCOMPLETE_DIAGNOSTIC'):
            invalid=deepcopy(valid);invalid['cpu']['fuse']['FP32']['status']=status
            rows['launch_blocks_'+status]=rejected(lambda:require_preflight(invalid))
        invalid=deepcopy(valid);invalid['capacity']['status']='NOT_RUN'
        rows['capacity_not_substituted']=rejected(lambda:require_preflight(invalid))
        report['status']='PASSED'
    finally:atomic_json(output/'tests.json',report)
    print(json.dumps(dict(status=report['status'],tests=len(rows),report=str(output/'tests.json')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.fixture)
