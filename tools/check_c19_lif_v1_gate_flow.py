"""Real launch predicate over complete local evidence; capacity is explicitly mocked."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from c19_lif_v1_diagnostic import atomic_json
from train_c19_lif_v1 import require_preflight


def run(checks,cutoff,output):
    actual=json.loads(checks.read_text(encoding='utf-8'))
    warning=json.loads(cutoff.read_text(encoding='utf-8'))
    report=dict(status='FAILED',scope='Real evidence predicate; only B16 capacity metadata mocked. Does not run capacity or dispatch training.',tests={})
    try:
        proof=deepcopy(actual)
        proof['capacity']=dict(status='PASSED',batch=16,imgsz=640,AMP=True,loss=1.,optimizer_steps=0)
        require_preflight(proof);report['tests']['complete_normal_evidence']=True
        proof['cuda']['fuse']['half_fuse']=warning
        require_preflight(proof);report['tests']['complete_cutoff_warning_accepted']=True
        def blocked(name,modify):
            incomplete=deepcopy(proof);modify(incomplete)
            try:require_preflight(incomplete)
            except (RuntimeError,KeyError,TypeError):report['tests'][name]=True
            else:raise AssertionError('Missing mandatory gate accepted: '+name)
        blocked('missing_CUDA_loss',lambda r:r['cuda'].pop('loss'))
        blocked('missing_CUDA_AMP_loss',lambda r:r['cuda'].pop('AMP_loss'))
        blocked('missing_DN',lambda r:r['cuda']['AMP_loss'].update(dynamic_DN=[]))
        blocked('nonfinite_gradient',lambda r:r['cuda']['AMP_loss']['steps'][0]['new_grad_norms'].update(injected=float('nan')))
        blocked('no_B16_capacity',lambda r:r['capacity'].update(status='NOT_RUN'))
        blocked('wrong_capacity_batch',lambda r:r['capacity'].update(batch=8))
        blocked('missing_validation_half',lambda r:r['cuda'].pop('training_precision'))
        blocked('missing_nonzero_bbox',lambda r:r['cuda']['training_precision'].update(nonzero_bbox_parameters=0))
        blocked('empty_warning_string',lambda r:r['cuda']['fuse'].update(half_fuse=dict(status='PASS_WITH_BASELINE_CUTOFF_TIE',acceptance='ACCEPTED_WITH_WARNING')))
        report['status']='PASSED'
    finally:atomic_json(output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checks',type=Path,required=True);p.add_argument('--cutoff',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.checks,a.cutoff,a.output)
