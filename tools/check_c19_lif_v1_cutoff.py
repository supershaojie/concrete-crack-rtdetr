"""Focused cutoff gate tests, including native gather/replays under injected logits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import torch
from init_c19_lif_v1 import build,require,runtime
from c19_lif_v1_diagnostic import fusion_protocol,atomic_json,rng_state,restore_rng
from c19_lif_v1_cutoff import WARNING,fusion_accepted,review_evidence,selection_from_scores
from c19_lif_v1_probe import capture
from ultralytics.utils.patches import torch_load


def run(folder,fixture,light):
    folder.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report=dict(status='FAILED',runtime=runtime(),formal_training='NOT_RUN',tests={});tests=report['tests']
    try:
        # The real LIGHT reports remain byte-for-byte untouched, including BLOCKED.
        old={p.parent.name:json.loads(p.read_text(encoding='utf-8')) for p in light.rglob('fuse_diagnostic.json')}
        half=old['cuda_half_fuse_fusion'];cpu=old['cpu_FP32_fusion']
        require(half['status']=='REQUIRES_REVIEW' and half['selection']['images'][0]['common_count']==299,'Wrong LIGHT fixture')
        require(cpu['natural_outputs']['raw_boxes']['max_abs_error']==0.20000004768371582 and cpu['common_id_alignment']['raw_boxes']['max_abs_error']==0,'CPU row mismatch evidence differs')
        pair=half['selection']['images'];parent=half['parent_selection']['selection']['images']
        require(all(a[k]==b[k] for a,b in zip(pair,parent) for k in ('natural_a','natural_b')),'LIGHT parent IDs differ')
        tests['LIGHT_original_review']=dict(original_status=half['status'],review=review_evidence(half),
            missing_dynamic=['B-set pair replay','A/B C2 replay','FP64 local cutoff statistics','strict source mapping/trace proof'],cpu_raw_0_2_explained=True)
        require(not fusion_accepted(half),'Legacy incomplete evidence accepted')
        require(torch.cuda.is_available(),'Native half gate test requires available CUDA')
        f=torch_load(fixture,map_location='cpu');source=build('pair',nc=1).eval();source.load_state_dict(f['unfused'],strict=True)
        source=source.cuda();x=f['image'].cuda().half()
        left=deepcopy(source).half();right=deepcopy(source).fuse(verbose=False).half()
        def parent_factory():
            m=build('C2',nc=1).eval();m.load_state_dict({k:source.state_dict()[k].cpu() for k in m.state_dict()},strict=True);return m
        # Test-only logits synthesize an adjacent-ULP cutoff tie. IDs are chosen
        # by native topk and verified native gather, never forced candidate IDs.
        n=sum((h*w) for h,w in ((20,24),(10,12),(5,6)));k=left.model[-1].num_queries
        template=torch.full((1,n),-.75,device='cuda',dtype=torch.float16);template[:,:k-1]=.75;template[:,k-1:k+1]=.5
        native_topk=torch.topk
        chosen=native_topk(template,k,dim=1).indices[0].tolist();boost=next(i for i in (k-1,k) if i not in chosen)
        def injected_capture(model,image,batch=None,fixed_ids=None):
            unfused=any(type(m).__name__=='Conv' and hasattr(m,'bn') for m in model.modules())
            def logits(module,args,value):
                v=template.expand(value.shape[0],-1).clone()
                if unfused:v[:,boost]=torch.nextafter(v[:,boost],torch.full_like(v[:,boost],float('inf')))
                return v.unsqueeze(-1).expand_as(value)
            hook=model.model[-1].enc_score_head.register_forward_hook(logits)
            try:return capture(model,image,batch,fixed_ids=fixed_ids)
            finally:hook.remove()
        initial=rng_state()
        with torch.no_grad():production_before=left(x)[0].clone()
        with patch('c19_lif_v1_probe.capture',side_effect=injected_capture):
            accepted=fusion_protocol(left,right,x,folder/'native_cutoff','cuda','half',parent_factory,source)
        require(accepted['status']==WARNING and fusion_accepted(accepted),'Evidence-backed native cutoff was not accepted: '+str(accepted.get('review')))
        with torch.no_grad():require(torch.equal(production_before,left(x)[0]),'Production forward changed after diagnostics')
        final=rng_state();require(torch.equal(initial['cpu'],final['cpu']) and all(torch.equal(a,b) for a,b in zip(initial['cuda'],final['cuda'])) and initial['python']==final['python'],'RNG leaked')
        require(initial['numpy'][0]==final['numpy'][0] and (initial['numpy'][1]==final['numpy'][1]).all() and initial['numpy'][2:]==final['numpy'][2:],'NumPy RNG leaked')
        require(torch.topk is native_topk and '_get_decoder_input' not in left.model[-1].__dict__,'Trace patch leaked')
        tests['native_cutoff_acceptance']=dict(status=accepted['status'],scope='test-only injected logits; actual original model operators and native topk/gather',
            pair_B=accepted['fixed_query_replay_B']['boxes'],parent_A=accepted['parent_selection']['replay_A']['boxes'],parent_B=accepted['parent_selection']['replay_B']['boxes'])
        tests['cleanup']='PASSED: topk/hooks/forward/CPU-CUDA-Python-NumPy RNG restored'
        def blocked(name,mutate):
            r=deepcopy(accepted);mutate(r);decision=review_evidence(r)
            require(not fusion_accepted(r) and not decision['accepted'],name+' accepted')
            tests[name]=decision
        def parent_other(r):
            s=r['parent_selection']['selection'];ids=[row['natural_a'][:] for row in s['images']];ids[0][0],ids[0][1]=ids[0][1],ids[0][0]
            r['parent_selection']['selection']=selection_from_scores(ids,[row['natural_b'] for row in s['images']],s['scores_a'],s['scores_b'],s['shapes'],s['score_dtype_a'],s['score_dtype_b'])
        blocked('different_C2_full_IDs',parent_other)
        blocked('C2_continuous_failure',lambda r:r['parent_selection']['pre_selection']['encoder_features'].update(allclose_failed_count=1))
        blocked('only_A_passes_B_fails',lambda r:r['fixed_query_replay_B']['raw_boxes'].update(allclose_failed_count=1))
        blocked('parent_B_failure',lambda r:r['parent_selection']['replay_B']['raw_boxes'].update(allclose_failed_count=1))
        blocked('missing_parent',lambda r:r.pop('parent_selection'))
        blocked('missing_field',lambda r:r['fixed_query_replay_B'].pop('cbr_query'))
        blocked('wrong_semantic_dtype',lambda r:r['fixed_query_replay_B']['valid_mask'].update(dtype_a='torch.float32'))
        blocked('wrong_floating_dtype',lambda r:r['fixed_query_replay_B']['raw_boxes'].update(dtype_a='torch.int64'))
        blocked('missing_finite_evidence',lambda r:r['fixed_query_replay_B']['raw_boxes'].pop('masked_nonfinite_count'))
        blocked('empty_tensor_shape',lambda r:r['fixed_query_replay_B']['raw_boxes'].update(shape=[]))
        blocked('bad_BN',lambda r:r['fusion_invariants'].update(lif_bn_retained=False))
        blocked('LIF_residual_error',lambda r:r['pre_selection']['lif_residual'].update(allclose_failed_count=1))
        blocked('CBR_output_error',lambda r:r['fixed_query_replay_B']['cbr_after'].update(allclose_failed_count=1))
        blocked('parent_mapping_missing',lambda r:r['parent_selection']['mapping'].pop('source_state_sha256'))
        blocked('illegal_ID',lambda r:r['selection']['images'][0]['natural_a'].__setitem__(0,-1))
        blocked('NaN_scores',lambda r:r['selection']['scores_a'][0].__setitem__(0,float('nan')))
        blocked('CPU_exception_forbidden',lambda r:r.update(device='cpu'))
        blocked('AMP_exception_forbidden',lambda r:r.update(precision='amp'))
        blocked('FP32_exception_forbidden',lambda r:r.update(precision='fp32'))
        blocked('wrong_tolerance',lambda r:r['fixed_query_replay_B']['boxes'].update(atol=.2))
        # A stable, high-scoring member cannot be described as a boundary tie.
        s=accepted['selection'];ia=[r['natural_a'] for r in s['images']];ib=[r['natural_b'] for r in s['images']]
        scores=deepcopy(s['scores_a']);scores[0][boost]=.75
        stable=selection_from_scores(ia,ib,scores,s['scores_b'],s['shapes'],s['score_dtype_a'],s['score_dtype_b'])
        require(not stable['images'][0]['cutoff_eligible'],'Stable member classified as cutoff')
        tests['stable_member_blocked']=True
        from train_c19_lif_v1 import require_preflight
        for status in ('PASSED',WARNING,'REQUIRES_REVIEW'):
            try:require_preflight(dict(status='PASSED',capacity=dict(status='PASSED',batch=16,imgsz=640,AMP=True,loss=1.,optimizer_steps=0),cpu=dict(fuse=dict(FP32=dict(status=status,acceptance='PASSED')))))
            except RuntimeError:tests['empty_gate_'+status]=True
            else:raise AssertionError('Status-only launch proof accepted')
        # Incomplete evidence retains local tensors; success is compact JSON only.
        require(not (folder/'native_cutoff/fixture.pt').exists() and not (folder/'native_cutoff/records.pt').exists(),'Success duplicated large tensors')
        report['status']='PASSED'
    finally:atomic_json(folder/'tests.json',report)
    print(json.dumps(dict(status=report['status'],tests=len(tests)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--light',type=Path,required=True)
    a=p.parse_args();run(a.output,a.fixture,a.light)
