"""Finite fusion-only reproduction. No dataset evaluation, training, or lock changes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from init_c19_lif_v1 import controlled_models,build_training_model,require,runtime,sha256,build
from check_c19_lif_v1 import degenerations,fuse_checks
from c19_lif_v1_diagnostic import atomic_json,fusion_protocol,PASS,restore_rng
from ultralytics import RTDETR
from ultralytics.utils.patches import torch_load


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED',scope='finite_fusion_only',formal_training='NOT_RUN',full_val_test='NOT_RUN',
                capacity=dict(status='NOT_RUN'),runtime=runtime(),devices={},
                original_AutoDL_0_2='NOT_PROVEN: old archive contains no original failure tensors')
    def persist():atomic_json(args.output/'checks.json',report)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
        report['source']=dict(path=str(args.source.resolve()),sha256=sha256(args.source))
        report['initialized']=dict(path=str(args.initialized.resolve()),sha256=sha256(args.initialized),read_only=True)
        _,expected,mapping=controlled_models(args.source)
        weights=RTDETR(str(args.initialized)).model
        require(all(torch.equal(v,weights.state_dict()[k]) for k,v in expected.state_dict().items()),'Old init differs from controlled public source')
        torch.manual_seed(42);target,adapt=build_training_model(weights.yaml,weights,dict(nc=1,channels=3))
        report.update(mapping=mapping,trainer_rebuild=adapt);persist()
        if args.fixture:
            fixture=torch_load(args.fixture,map_location='cpu')
            device=fixture['device'];precision=fixture['precision']
            model=deepcopy(target).eval().to(device)
            if precision=='half':model.half()
            model.load_state_dict(fixture['unfused'],strict=True)
            fused=deepcopy(target).eval().fuse(verbose=False).to(device)
            if precision=='half':fused.half()
            fused.load_state_dict(fixture['fused'],strict=True)
            def parent():
                p=build('C2',nc=1).eval();p.load_state_dict({k:model.state_dict()[k].float().cpu() for k in p.state_dict()},strict=True);return p
            restore_rng(fixture['rng'])
            result=fusion_protocol(model,fused,fixture['image'].to(device),args.output/'fixture_replay',device,precision,parent)
            report['fixture_replay']=result;report['status']='PASSED' if result['status'] in PASS else 'REQUIRES_REVIEW'
        else:
            for device in args.devices:
                if device=='cuda' and not torch.cuda.is_available():
                    report['devices'][device]=dict(status='NOT_RUN',reason='CUDA unavailable');persist();continue
                section=report['devices'][device]={};persist()
                try:
                    # Preserve old RNG prefix, including all six parent fixtures.
                    # No changed seed, tensor generation, or model initialization.
                    torch.manual_seed(321)
                    section['degenerations']=degenerations(target,device)
                    section['fuse']={};persist()
                    fuse_checks(target,device,args.output,section['fuse'],persist)
                    section['status']='PASSED'
                except BaseException as error:
                    section.update(status='FAILED',failure=getattr(error,'detail',dict(error=repr(error))))
                persist()
            report['status']='PASSED' if all(s['status']=='PASSED' for s in report['devices'].values()) else 'FAILED'
    except BaseException as error:
        report['failure']=getattr(error,'detail',dict(error=repr(error)))
        raise
    finally:persist()
    print(json.dumps(dict(status=report['status'],report=str(args.output/'checks.json')),indent=2))
    return 0 if report['status']=='PASSED' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initialized','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--devices',nargs='+',choices=['cpu','cuda'],default=['cpu','cuda'])
    parser.add_argument('--fixture',type=Path,help='Replay a saved diagnostic fixture; never a trained checkpoint')
    raise SystemExit(run(parser.parse_args()))
