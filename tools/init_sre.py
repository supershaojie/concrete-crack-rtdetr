"""Build and audit fair nc80 SRE initialization and actual native nc1 Trainer loading."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sre_common import *
import torch
import init_c19_lif_v1 as parent_tools
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules import SRE, SRERepC3, RepC3, Conv, LIFDown, RTDETRDecoder, RTDETRDecoderCBR
from ultralytics.utils.patches import torch_load


def is_added(key):
    return key.startswith('model.19.sre.')


def build(variant, nc=80, baseline=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/VARIANTS[variant][int(baseline)]),nc=nc,verbose=False)


def verify_model(model, variant, zero=False):
    source_contract()
    config = YAML.load(MODEL_DIR/VARIANTS[variant][0])
    baseline = YAML.load(MODEL_DIR/VARIANTS[variant][1])
    expected = deepcopy(baseline)
    expected['head'][19-len(baseline['backbone'])][2:] = ['SRERepC3',[256,0.5,32,8,5]]
    require(all(config[k]==expected[k]==model.yaml[k] for k in ('backbone','head','scales')), 'SRE-only wiring changed')
    require(len(model.model)==27 and type(model.model[19]) is SRERepC3, 'Expected single wrapper at original node19')
    require(len(model.model[19].m)==3 and sum(isinstance(m,SRE) for m in model.modules())==1, 'Repeat/wrapper count changed')
    require(model.model[19].f==-1 and model.model[20].f==-1 and model.model[26].f==[19,22,25], 'Graph connections changed')
    require(sum(p.numel() for n,p in model.named_parameters() if is_added(n))==16704, 'New parameter delta changed')
    head = model.model[26]
    require(head.hidden_dim==256 and head.num_queries==300 and len(head.decoder.layers)==3 and head.decoder.eval_idx==2,
            'Decoder contract changed')
    if variant=='cbr_lif_sre_v1':
        require(type(model.model[20]) is LIFDown and type(head) is RTDETRDecoderCBR, 'Original CBR/LIF missing')
        require(head.cbr.rho==head.cbr.normal_fraction==.10, 'CBR fraction changed')
    else:
        require(type(model.model[20]) is Conv and type(head) is RTDETRDecoder, 'C2 ablation changed downsample/decoder')
    if head.nc==1 and not model.is_fused():
        require(sum(p.numel() for p in model.parameters())==dict(cbr_lif_sre_v1=20166469,sre_v1=20099476)[variant],
                'nc1 parameter count differs')
    if zero:
        branch = model.model[19].sre
        require(torch.count_nonzero(branch.W_o.weight)==0, 'SRE output must initialize zero')
        require(torch.count_nonzero(branch.W_d.weight)>0 and torch.count_nonzero(branch.W_q.weight)>0,
                'SRE upstream Xavier must be nonzero')
        require(torch.equal(branch.GN.weight,torch.ones_like(branch.GN.weight)) and torch.count_nonzero(branch.GN.bias)==0,
                'SRE GN initialization changed')


def controlled_models(source, variant):
    base, pair, parent_report = parent_tools.controlled_models(source)
    parent = pair if variant=='cbr_lif_sre_v1' else base
    target = build(variant)
    fresh_parent = build(variant,baseline=True)
    before, after = fresh_parent.state_dict(), target.state_dict()
    require(all(k in after and torch.equal(v,after[k]) for k,v in before.items()), 'SRE constructor disturbed parent RNG')
    extra = set(after)-set(before)
    require(extra=={k for k in after if is_added(k)} and extra<=dict(target.named_parameters()).keys(), 'Unexpected new states')
    require(not set(before)-set(after), 'Missing parent states')
    target.load_state_dict({**after,**parent.state_dict()},strict=True)
    verify_model(target,variant,zero=True)
    rows = [dict(source=k,target=k,shape=list(v.shape),equal=torch.equal(v,target.state_dict()[k]))
            for k,v in parent.state_dict().items()]
    require(all(r['equal'] for r in rows), 'Controlled common state mismatch')
    other = build('sre_v1' if variant=='cbr_lif_sre_v1' else 'cbr_lif_sre_v1')
    require(all(torch.equal(target.state_dict()[k],other.state_dict()[k]) for k in extra), 'Two SRE initializations differ')
    report = dict(status='PASSED',variant=variant,source=str(Path(source).resolve()),source_sha256=sha256(source),
                  base_commit=BASE_COMMIT,source_nc=80,target_nc=80,COMMON=rows,NEW_TRAINABLE=sorted(extra),
                  NEW_BUFFER=[],MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],ALLOWED_CLASS_ADAPTATION=[],
                  new_parameters=16704,parent_initialization_audit=parent_report,two_variants_sre_equal=True,
                  constructor_common_equal=True,storage='FP32; source half-quantized values exactly promoted')
    return parent,target,report


def native_rebuild(cfg, weights, data):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = data
    return RTDETRTrainer.get_model(trainer,cfg=deepcopy(cfg),weights=weights,verbose=False)


def build_training_model(cfg, weights, data, variant, zero=True):
    require(weights is not None, 'Controlled weights required')
    # Parent construction/loading has no effect on target classification adaptation RNG.
    with torch.random.fork_rng(devices=[]):
        parent = native_rebuild(str(MODEL_DIR/VARIANTS[variant][1]),weights,data)
    target = native_rebuild(cfg,weights,data)
    verify_model(target,variant,zero=zero)
    before,after = weights.state_dict(),target.state_dict()
    allowed = {'model.26.denoising_class_embed.weight','model.26.enc_score_head.weight','model.26.enc_score_head.bias'}
    allowed |= {f'model.26.dec_score_head.{i}.{s}' for i in range(3) for s in ('weight','bias')}
    changed = {k for k in before if before[k].shape!=after[k].shape}
    require(set(before)==set(after) and changed==(allowed if weights.model[-1].nc!=data['nc'] else set()),
            'Unexpected native Trainer class adaptation')
    require(all(torch.equal(v,after[k]) for k,v in before.items() if k not in changed), 'Native Trainer lost controlled state')
    require(all(torch.equal(v,after[k]) for k,v in parent.state_dict().items()), 'Native nc1 parent classification/state differs')
    return target,dict(status='PASSED',native_get_model=True,parent_nc1_all_equal=True,nc=data['nc'],
                       COMMON=[k for k in before if not is_added(k) and k not in changed],
                       NEW_TRAINABLE=[k for k in before if is_added(k)],NEW_BUFFER=[],MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],
                       ALLOWED_CLASS_ADAPTATION=[dict(name=k,source_shape=list(before[k].shape),target_shape=list(after[k].shape),
                            parent_nc1_exact=True) for k in sorted(changed)])


def actual_train_api(initialized, variant, expected):
    observed = {}
    class ProbeComplete(Exception): pass
    class AuditTrainer(RTDETRTrainer):
        def __init__(self,overrides,_callbacks):
            self.args=SimpleNamespace(**{k:v for k,v in overrides.items() if k!='session'})
            self.data=dict(nc=1,channels=3)
            torch.random.default_generator.manual_seed(42)
        def train(self):
            require(all(torch.equal(v,self.model.state_dict()[k]) for k,v in expected.state_dict().items()),
                    'Actual RTDETR.train native reconstruction differs')
            observed.update(status='PASSED',native_inherited_get_model=True,optimizer_steps=0,
                            matched_state_tensors=len(expected.state_dict()))
            raise ProbeComplete()
    args,_=recipe(variant,initialized,ROOT/'docs/sre/parent_data.yaml')
    with torch.random.fork_rng(devices=[]), patch('ultralytics.engine.model.checks.check_pip_update_available'):
        try: RTDETR(str(initialized)).train(trainer=AuditTrainer,**args)
        except ProbeComplete: pass
    require(observed,'Actual Trainer not reached')
    return observed


def initialize(source,variant,output_dir=None):
    p=paths(variant,output_dir)
    require(not p['init'].exists() and not p['initialization'].exists(),'Existing init/report preserved; choose a new --output-dir')
    _,target,report=controlled_models(source,variant)
    target.eval(); target.args={**DEFAULT_CFG_DICT,'model':str(MODEL_DIR/VARIANTS[variant][0]),'task':'detect'}
    target.task,target.pt_path='detect',str(p['init'].resolve())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        nc1,nc1_report=build_training_model(target.yaml,target,dict(nc=1,channels=3),variant)
    checkpoint=dict(epoch=-1,best_fitness=None,model=deepcopy(target).float(),ema=None,updates=None,optimizer=None,scaler=None,
                    train_args=target.args,train_metrics=None,train_results=None,date=datetime.now(timezone.utc).isoformat(),
                    sre_provenance=report)
    p['folder'].mkdir(parents=True,exist_ok=True)
    with p['init'].open('xb') as stream: torch.save(checkpoint,stream)
    restored=RTDETR(str(p['init'])).model
    require(all(torch.equal(v,restored.state_dict()[k]) for k,v in target.state_dict().items()),'Immediate reload differs')
    report.update(output=str(p['init'].resolve()),output_sha256=sha256(p['init']),reload_exact=True,
                  native_nc1=nc1_report,actual_train_api=actual_train_api(p['init'],variant,nc1),runtime=runtime(),
                  identity_at_initialization=code_identity())
    write_json(p['initialization'],report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,default='cbr_lif_sre_v1')
    parser.add_argument('--source',type=Path,default=MAIN/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(); torch.set_num_threads(4)
    if not args.source.is_file():
        pending=dict(status='PENDING',reason='Unified source unavailable',source=str(args.source),
                     variant=args.variant,formal_training='NOT_STARTED',final_test='NOT_RUN')
        write_json(paths(args.variant,args.output_dir)['folder']/'initialization_pending.json',pending)
        print(json.dumps(pending,indent=2));raise SystemExit(2)
    report=initialize(args.source,args.variant,args.output_dir)
    print(json.dumps(dict(status=report['status'],variant=args.variant,output=report['output'],sha256=report['output_sha256']),indent=2))
