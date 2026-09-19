"""Admission logic unit fixtures only: never real preflight or start permission."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import train_dpr as t
from check_dpr import TARGET,HALF_ATOL,HALF_RTOL

variant='cbr_lif_dpr_v1'
init=json.loads((ROOT/'outputs/dpr/cbr_lif_dpr_v1/init.json').read_text())
math=json.loads((ROOT/'outputs/dpr/module_check_03.json').read_text())
base_capture=math['checks'][variant]['learned_lifecycle']
names={row['name'] for row in init['native_trainer_rebuild']['COMMON']}
new={TARGET+'.'+n for n in ('dpr_cd','dpr_hd','dpr_vd','dpr_ad')}
row=dict(equal=True,raw_allclose=True,finite=True,max_abs=0.,relative_L2=0.,exceeded_fraction=0.)
def metrics(keys):
    return dict(status='PASSED',exact=True,raw_allclose=True,finite=True,failed_tensors=[],metadata_errors=[],
                tensors={k:deepcopy(row) for k in keys})
def state(candidate):
    return metrics({prefix+k for prefix in ('state.model.','state.ema.') for k in names|(new if candidate else set())})
def bside(candidate):
    return dict(initial_exact=True,storage_independent=True,forward_exact=True,loss_exact=True,exact_metadata_buffers=True,
                raw_allclose=True,state=state(candidate),gradients=metrics(['state.'+TARGET+'.conv.weight']),
                forward=metrics(['state.decoder']),steps=[dict(effective_update=True,finite_gradients=True)]*2)
def a(amp):
    result={k:True for k in ('saved_live_optimizer_exact','saved_byte_restoration_exact','optimizer_names_groups_exact',
            'no_shared_storage','same_gradient_complete_state_exact','nonzero_dpr_preserved','state_dict_reload_exact',
            'full_model_reload_exact','native_trainer_rebuild_exact','overflow_skip_exact','effective_step_exact')}
    result.update(status='PASSED',policy=dict(policy='optimizer_fp32_v1',original_tensors_inspected_before_cast=True,
              fp32_moment_tensors=100),restoration=state(True),replay=state(True),
              saved_live_optimizer_comparison=metrics(['state.optimizer.exp_avg','state.optimizer.exp_avg_sq']),
              cpu_native_continuation_exact=True,
              replay_steps=[dict(effective_update=True)]*2,
              overflow_rows=[dict(effective_update=False,scale_before=65536,scale_after=32768)]*2 if amp else [])
    return result
def lifecycle(amp):
    return dict(status='PASSED',A=a(amp),B=dict(status='PASSED',atol=2e-5,rtol=2e-4,raw_allclose=True,
                parent=bside(False),candidate=bside(True)),
                gradients={k:dict(finite_nonzero=True) for k in ('dpr_cd','dpr_hd','dpr_vd','dpr_ad','conv.weight')},
                updates=dict(status='PASSED',effective_updates=2,observed_batches=2,
                             parameter_changes={k:True for k in new|{TARGET+'.conv.weight'}}))
def halfleaf(key):
    v=deepcopy(base_capture['checks'][key])
    for value in v.get('details',{}).values():
        if isinstance(value,dict): value.update(atol=HALF_ATOL,rtol=HALF_RTOL)
    if 'comparison' in v: v['comparison'].update(atol=HALF_ATOL,rtol=HALF_RTOL)
    return v
half=dict(status='PASSED',dtype='float16',device='cuda:0',atol=HALF_ATOL,rtol=HALF_RTOL,checks={
    'native_half_vs_fp32':halfleaf('dpr_fold_only'),'fp32_fold_then_half':halfleaf('native_fuse'),
    'half_deploy_reload':halfleaf('deploy_full_reload_idempotent'),'autobackend_half':halfleaf('autobackend_saved_best')})
environment=dict(server_environment=True,fixture='UNIT_TEST_NOT_ACTUAL_CUDA')
identity=dict(head='f'*40)
report=dict(contract=t.CONTRACT,report_kind='full_preflight_engineering',variant=variant,status='PASSED',
    environment=environment,code_identity=identity,source_sha256=t.SOURCE_SHA256,initialization_sha256=init['output_sha256'],
    dataset_identity={'fixture':True},recipe={'fixture':True},initialization={'kind':'init'},mathematics={'kind':'math'},
    mathematics_verified=True,checks={
    'capacity':dict(status='PASSED',batch=16,imgsz=640,native_amp=True,effective_updates=2,observed_batches=2,gt_instances=1,
        dn_seen=True,optimizer_exact_coverage=True,actual_trainer_rebuild=init['native_trainer_rebuild'],
        optimizer_names=[sorted(new)],steps=[{'effective_update':True}]*2,losses=[1.,1.],peak_memory_bytes=1,elapsed_seconds=1.),
    'cpu_fp32':lifecycle(False),'cuda_fp32':lifecycle(False),'cuda_native_amp':lifecycle(True),
    'learned_inference':{k:{'status':'PASSED'} for k in ('cpu_fp32','cuda_fp32','ema','fold_norm_retained','native_fuse','fuse_idempotent','deploy_reload','autobackend')}},
    learned_inference_details={'cpu_fp32':deepcopy(base_capture),'cuda_fp32':deepcopy(base_capture)})
report['learned_inference_details']['cuda_fp32']['device']='cuda:0'
report['checks']['learned_inference']['cuda_half']=half

def invoke(value):
    with patch.object(t,'load_json',return_value=value),patch.object(t,'environment',return_value=environment),\
         patch.object(t,'code_identity',return_value=identity),patch.object(t,'git',return_value=''),\
         patch.object(t,'sha256',side_effect=lambda p:t.SOURCE_SHA256 if str(p)=='source' else init['output_sha256']),\
         patch.object(t,'dataset_identity',return_value={'fixture':True}),patch.object(t,'recipe',return_value=({'fixture':True},[])),\
         patch.object(t,'evidence_file',side_effect=lambda record:init if record['kind']=='init' else math):
        return t.strict_gate(variant,Path('source'),Path('init'),Path('data'),Path('report'))

results=[]
assert invoke(report)['status']=='PASSED'
results.append(dict(case='complete_mock_schema',accepted=True,scope='Only unit fixture; not actual preflight'))
faults=[
('missing_capacity',lambda r:r['checks'].pop('capacity')),
('missing_A_tensor_inventory',lambda r:r['checks']['cpu_fp32']['A']['restoration']['tensors'].pop('state.model.'+TARGET+'.dpr_cd')),
('missing_saved_optimizer_raw_evidence',lambda r:r['checks']['cpu_fp32']['A'].pop('saved_live_optimizer_comparison')),
('failed_cpu_native_continuation',lambda r:r['checks']['cpu_fp32']['A'].update(cpu_native_continuation_exact=False)),
('false_A_summary',lambda r:r['checks']['cpu_fp32']['A']['replay']['tensors']['state.model.'+TARGET+'.dpr_cd'].update(equal=False)),
('missing_B_control',lambda r:r['checks']['cuda_fp32']['B'].pop('parent')),
('fake_B_precision_note',lambda r:r['checks']['cuda_fp32']['B'].update(status='PRECISION_NOTE',raw_allclose=False)),
('fake_fusion_precision_note',lambda r:r['learned_inference_details']['cuda_fp32']['checks']['native_fuse'].update(status='PRECISION_NOTE')),
('cpu_as_cuda_inference',lambda r:r['learned_inference_details']['cuda_fp32'].update(device='cpu')),
('missing_half_reload',lambda r:r['checks']['learned_inference']['cuda_half']['checks'].pop('half_deploy_reload')),
]
for name,fn in faults:
    value=deepcopy(report);fn(value)
    try:invoke(value)
    except (RuntimeError,KeyError) as error:results.append(dict(case=name,rejected=True,reason=str(error)))
    else:raise AssertionError('Fault accepted: '+name)

# A documented candidate-index precision note remains admissible when raw output
# is false but every continuous path and fixed-index output actually pass schema.
value=deepcopy(report)
entry=value['learned_inference_details']['cuda_fp32']['checks']['native_fuse']
entry['status']='PRECISION_NOTE'
entry['details']['fixed_candidate_replay_output']=deepcopy(entry['details']['output'])
entry['details']['output']['raw_allclose']=False
entry['details']['candidate_index_changes']=2
assert invoke(value)['status']=='PASSED'
results.append(dict(case='supported_candidate_precision_note',accepted=True,scope='Synthetic note used only for gate logic'))

# Finite, non-identical gradients below raw tolerance can yield inherited state
# failures with AdamW near zero. Every state failure needs its own mapped row.
value=deepcopy(report)
b=value['checks']['cuda_fp32']['B']
b.update(status='PRECISION_NOTE',raw_allclose=False,explanation='Unit fixture of inherited AdamW amplification')
for side in ('parent','candidate'):
    control=b[side]
    name='model.12.bn.bias'
    failure='state.model.'+name
    control['state'].update(status='FAILED',exact=False,raw_allclose=False,failed_tensors=[failure])
    control['state']['tensors'][failure].update(equal=False,raw_allclose=False,max_abs=3e-5)
    control['gradients']['tensors']['state.'+name]=dict(row,equal=False,max_abs=1e-8)
assert invoke(value)['status']=='PASSED'
results.append(dict(case='supported_subtolerance_gradient_precision_note',accepted=True,scope='Synthetic gate fixture, original raw_allclose remains false'))
value['checks']['cuda_fp32']['B']['candidate']['gradients']['tensors']['state.model.12.bn.bias']['equal']=True
try:invoke(value)
except RuntimeError as error:results.append(dict(case='state_difference_without_gradient_support',rejected=True,reason=str(error)))
else:raise AssertionError('Unsupported state precision note accepted')

out=dict(report_kind='dpr_gate_fault_unit_test',scope='Mocked runtime and metrics; NO server/CUDA/preflight/permit evidence',
         status='PASSED',cases=results)
(ROOT/'docs/dpr/admission_fault_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
