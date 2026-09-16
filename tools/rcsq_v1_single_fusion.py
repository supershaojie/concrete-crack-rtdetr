"""Single-module fusion: original tolerances, full-ID replay, measured half cutoff evidence."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from init_rcsq_v1 import require,write_json
from c19_lif_v1_probe import capture
from c19_lif_v1_diagnostic import (PRE_KEYS,schema,validate,selection_report,align_to_ids,
                                 compare_records,rng_state,restore_rng)


def single_fusion(a,b,image,device,precision,folder,parent_factory,source):
    """No changed-set acceptance except proven original CUDA true-half cutoff ties.

    RCS-Q runs after selection, so original baseline must reproduce BOTH complete
    natural candidate sequences and the exact cutoff relation. Both full candidate
    sets are replayed through both new and original models. No intersection test,
    arbitrary score band, increased allclose tolerance, or caller status is used.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    atol,rtol=(2e-5,2e-4) if precision=='fp32' else (3e-3,3e-2)
    state=rng_state()
    report=dict(status='FAILED',device=device,precision=precision,tolerances=dict(atol=atol,rtol=rtol),
                nonzero_rcsq=True,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,natural_status=None)
    def run(model,ids=None):
        restore_rng(state)
        with torch.no_grad(),(torch.autocast('cuda',dtype=torch.float16) if precision=='amp' else nullcontext()):
            records=capture(model,image,fixed_ids=ids)[1]
        validate(records,'single_rcsq_fusion')
        return records
    def trace(r):
        return {k:r[k].tolist() if torch.is_tensor(r[k]) else r[k] for k in
                ('actual_gather_verified','topk_calls','candidate_indices','native_candidate_indices')}
    def compare(left,right,keys=None,collect=False):
        return compare_records(left,right,atol,rtol,device=device,precision=precision,keys=keys,collect=collect)
    try:
        left,right=run(a),run(b)
        require(left['mode']==right['mode']=='C2','Single module must have original baseline topology')
        require(torch.count_nonzero(a.model[-1].rcsq.out_proj.weight)>0,'Zero-only fusion does not exercise RCS-Q')
        require(torch.equal(a.model[-1].rcsq.out_proj.weight,b.model[-1].rcsq.out_proj.weight),'Fuse changed RCS-Q state')
        require(sum(isinstance(m,torch.nn.BatchNorm2d) for m in b.modules())<
                sum(isinstance(m,torch.nn.BatchNorm2d) for m in a.modules()),'Ordinary fusion was skipped')
        report['traces']=dict(natural_a=trace(left),natural_b=trace(right))
        report['pre_selection']=compare(left,right,PRE_KEYS)
        report['selection']=selection=selection_report(left,right)
        keys=schema(left);post=[k for k in keys if k not in PRE_KEYS+['candidate_indices']]
        report['natural_outputs']=compare(left,right,post,collect=True)
        for tag,ids in (('A',left['candidate_indices']),('B',right['candidate_indices'])):
            ra,rb=run(a,ids),run(b,ids)
            report['replay_'+tag]=compare(ra,rb)
            report['traces']['replay_'+tag+'_a']=trace(ra);report['traces']['replay_'+tag+'_b']=trace(rb)
        if selection['kind']!='SET_DRIFT':
            report['aligned_outputs']=compare(left,align_to_ids(left,right),post)
            report['status']='PASSED' if selection['kind']=='IDENTICAL' else 'PASS_WITH_CANDIDATE_PERMUTATION'
            report['natural_status']=selection['kind']
        else:
            report['natural_status']='NATURAL_SELECTION_DRIFT'
            require(device=='cuda' and precision=='half' and selection['score_dtype_a']=='torch.float16',
                    'Cutoff exception restricted to CUDA true-half; FP32/AMP drift needs review')
            require(all(v['cutoff_eligible'] for v in selection['images'] if v['a_only']),
                    'Changed candidate outside original measured one-ULP cutoff policy')
            parent=parent_factory().eval().float()
            common=parent.state_dict()
            require(len(common)==533 and all(torch.equal(v.cpu(),source.state_dict()[k].cpu()) for k,v in common.items()),
                    'Original baseline common states are not exact')
            require(all(torch.equal(v.to(a.state_dict()[k]),a.state_dict()[k]) for k,v in common.items()),'Half common-state cast mismatch')
            parent.to(device);parent_fused=deepcopy(parent).fuse(verbose=False)
            parent.half();parent_fused.half()
            pa,pb=run(parent),run(parent_fused)
            parent_selection=selection_report(pa,pb)
            # Whole produced-score/ID/cutoff statistics must match, not merely similar drift.
            require(parent_selection==selection,'Original baseline does not reproduce identical natural cutoff evidence')
            proof=report['original_parent']=dict(common_states=533,common_exact=True,half_cast_exact=True,
                selection=parent_selection,pre_selection=compare(pa,pb,PRE_KEYS),
                natural_outputs=compare(pa,pb,post,collect=True),natural_a=trace(pa),natural_b=trace(pb))
            for tag,ids in (('A',left['candidate_indices']),('B',right['candidate_indices'])):
                ra,rb=run(parent,ids),run(parent_fused,ids)
                proof['replay_'+tag]=compare(ra,rb)
                proof['replay_'+tag+'_a']=trace(ra);proof['replay_'+tag+'_b']=trace(rb)
            report['status']='PASS_WITH_BASELINE_CUTOFF_TIE'
        report['operator_status']='OPERATOR_CHECK_PASSED'
    except BaseException as error:
        report['failure']=getattr(error,'detail',dict(error=repr(error)))
        raise
    finally:
        restore_rng(state);write_json(folder/'fuse_diagnostic.json',report)
    return report
