"""Original C17 + original LIF v1, strictly initialized from the untrained C2 public source."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import torch

from init_lif_down import ROOT, MODEL_DIR, SOURCE_SHA256, C2_COMMIT, require, sha256, write_json
from init_lif_down import runtime as lif_runtime
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import AIFI, CSCEFv51, Conv, LIFDown
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from c17_lif_v1_topology import inspect as topology, state_mapping

BASE_COMMIT = '0e95bbade3558b0d2b77c5531483c60810391d88'
C17_COMMIT = '0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139'
VARIANT = 'c17_lif_v1'
CONFIGS = dict(c2='rtdetr-resnet18-lite.yaml', lif='rtdetr-resnet18-lite-lif-down.yaml',
               c17='rtdetr-resnet18-lite-cscef-v51.yaml',
               c17_lif_v1='rtdetr-resnet18-lite-cscef-v51-lif-down.yaml')
VARIANTS = {VARIANT: (CONFIGS[VARIANT], CONFIGS['c2'], 'c17_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug')}
MODULE_HASHES = {'lif_down.py':'26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7',
                'cscef_v5.py':'a8ed29a6c6205ee51ec9fc0ea2e0bfed37a9c39a69dbf3dc761ee6debe037474',
                'cscef_v51.py':'b1879215de5eceba6d88a856fc0ef9f4d7287a755e97c543ab9a4bb50677102c'}
LIF_NAMES = {'B_proj','P','U_mix','U_dw','O_proj'}
CS_NAMES = {'lateral_projection','semantic_projection','mix_projection','depthwise_conv','output_projection'}


def is_added(key):
    parts = key.split('.')
    return len(parts) > 3 and parts[2] in LIF_NAMES | CS_NAMES


def runtime():
    info = lif_runtime()
    info['cscef_module'] = inspect.getfile(CSCEFv51)
    require(Path(info['cscef_module']).resolve() == ROOT/'ultralytics-main/ultralytics/nn/modules/cscef_v51.py', 'Wrong CSCEF source')
    info['original_modules'] = {}
    for name, expected in MODULE_HASHES.items():
        p = ROOT/'ultralytics-main/ultralytics/nn/modules'/name
        digest = hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        require(digest == expected, 'Original module modified: '+name)
        info['original_modules'][name] = dict(path=str(p), sha256=sha256(p), lf_sha256=digest)
    files=['ultralytics-main/ultralytics/nn/tasks.py','ultralytics-main/ultralytics/nn/modules/__init__.py',
           'ultralytics-main/ultralytics/models/rtdetr/train.py','ultralytics-main/ultralytics/engine/trainer.py',
           'ultralytics-main/ultralytics/cfg/models/rt-detr/'+CONFIGS[VARIANT],
           'tools/init_c17_lif_v1.py','tools/c17_lif_v1_topology.py','tools/check_c17_lif_v1.py',
           'tools/train_c17_lif_v1.py','tools/c17_lif_v1_results.py']
    info['code_lf_sha256']={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in files}
    return info


def build(variant=VARIANT, nc=80):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/CONFIGS[variant]),nc=nc,verbose=False)


def verify_model(model, variant=VARIANT, zero=False):
    require(variant == VARIANT and type(model) is RTDETRDetectionModel, 'Wrong experiment/model')
    expected = YAML.load(MODEL_DIR/CONFIGS['c17'])
    t = topology(expected)
    expected['head'][t['p3_to_p4']['downsample']-len(expected['backbone'])][2]='LIFDown'
    require(all(model.yaml[k] == expected[k] for k in ('backbone','head','scales')), 'Only original C17 first downsample may change')
    actual = topology(model.yaml)
    cs, lif = model.model[t['cscef']], model.model[t['p3_to_p4']['downsample']]
    require(type(cs) is CSCEFv51 and type(lif) is LIFDown, 'Wrong original module types')
    require(sum(type(m) is LIFDown for m in model.modules())==1 and sum(type(m) is CSCEFv51 for m in model.modules())==1,'Extra module')
    require(type(model.model[t['p4_to_p5']['downsample']]) is Conv, 'Second downsample changed')
    require(sum(type(m) is AIFI for m in model.modules())==1,'Original AIFI missing')
    require(sum(p.numel() for p in cs.parameters())==26912,'CSCEF parameter delta')
    require(sum(p.numel() for n,p in lif.named_parameters() if n.split('.')[0] in LIF_NAMES)==21104,'LIF parameter delta')
    require(len(cs.state_dict())==7 and len(list(cs.parameters()))==5,'CSCEF state contract')
    require(model.model[-1].hidden_dim==256 and model.model[-1].num_queries==300 and len(model.model[-1].decoder.layers)==3,'Decoder changed')
    if model.model[-1].nc==1: require(sum(p.numel() for p in model.parameters())==20130788,'nc1 parameter count')
    if zero:
        require(torch.count_nonzero(cs.output_projection.weight)==torch.count_nonzero(lif.O_proj.weight)==0,'Initial output must be zero')
    return actual


def compare_states(a, b, mapping):
    rows=[]
    for source,target in mapping.items():
        x,y=a[source].cpu(),b[target].cpu()
        require(x.shape==y.shape, f'Shape mismatch {source}->{target}')
        rows.append(dict(source=source,target=target,shape=list(x.shape),equal=torch.equal(x,y),max_abs=float((x.double()-y.double()).abs().max())))
    require(all(r['equal'] for r in rows), 'Mapped tensor values differ')
    return rows


def controlled_models(source, variant=VARIANT):
    require(variant==VARIANT,'Wrong experiment')
    source=Path(source)
    require(sha256(source)==SOURCE_SHA256,'Wrong public init hash')
    ckpt=torch_load(source,map_location='cpu')
    require(ckpt.get('epoch')==-1 and all(ckpt.get(k) is None for k in ('ema','optimizer','scaler','updates','train_metrics','train_results','best_fitness')),'Trained source forbidden')
    original=deepcopy(ckpt['model']).float()
    require(original.model[-1].nc==80,'Source must be nc80')
    models={name:build(name) for name in CONFIGS}
    public=original.state_dict()
    require(set(public)==set(models['c2'].state_dict()) and len(public)==533,'C2 state contract')
    reports={}
    for name, model in models.items():
        m=state_mapping(models['c2'].yaml,model.yaml,public)
        state=model.state_dict()
        require(all(public[k].shape==state[v].shape for k,v in m.items()),'Public shape mismatch')
        model.load_state_dict({**state,**{v:public[k] for k,v in m.items()}},strict=True)
        reports[name]=compare_states(public,model.state_dict(),m)
    target=models[VARIANT];verify_model(target,zero=True)
    common={r['target'] for r in reports[VARIANT]}
    params=dict(target.named_parameters());extra=set(target.state_dict())-common
    new_trainable=sorted(extra & set(params));new_buffer=sorted(extra-set(params))
    require(len(extra)==12 and len(new_trainable)==10 and len(new_buffer)==2,'Unexpected new states')
    require(all(is_added(k) for k in new_trainable),'Unexpected innovation state')
    parents={}
    for parent in ('c17','lif'):
        keys=[k for k in models[parent].state_dict() if is_added(k) or k.endswith(('scharr_x','scharr_y'))]
        m=state_mapping(models[parent].yaml,target.yaml,keys)
        parents[parent]=compare_states(models[parent].state_dict(),target.state_dict(),m)
    report=dict(source=str(source.resolve()),source_sha256=SOURCE_SHA256,source_nc=80,target_nc=80,seed=42,
                c2_commit=C2_COMMIT,lif_commit=BASE_COMMIT,c17_commit=C17_COMMIT,
                COMMON=reports[VARIANT],NEW_TRAINABLE=new_trainable,NEW_BUFFER=new_buffer,
                ALLOWED_CLASS_ADAPTATION=[],MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],
                parent_initial_values=parents,storage='FP32; public half source promoted exactly',runtime=runtime())
    return models,report


def native_rebuild(weights, nc=1):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data=dict(nc=nc,channels=3)
    return trainer.get_model(cfg=deepcopy(weights.yaml),weights=weights,verbose=False)


def rebuild_audit(weights, target):
    verify_model(target,zero=True)
    before,after=weights.state_dict(),target.state_dict();head=topology(target.yaml)['decoder']
    expected={f'model.{head}.{s}' for s in ['denoising_class_embed.weight','enc_score_head.weight','enc_score_head.bias',*[f'dec_score_head.{i}.{s}' for i in range(3) for s in ['weight','bias']]]}
    require(set(before)==set(after),'Rebuild state keys changed')
    skips={k for k in before if before[k].shape!=after[k].shape}
    require(skips==(expected if weights.model[-1].nc!=target.model[-1].nc else set()),'Unexpected class adaptation')
    equal=compare_states(before,after,{k:k for k in before if k not in skips})
    return dict(loaded_exact=len(equal),ALLOWED_CLASS_ADAPTATION=[dict(key=k,source_shape=list(before[k].shape),target_shape=list(after[k].shape),rule='unchanged native nc1 constructor at training RNG; exact C2 comparison below') for k in sorted(skips)],
                MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],new_states_exact=[k for k in after if is_added(k) or k.endswith(('scharr_x','scharr_y'))])


def build_training_model(cfg, weights, data, variant=VARIANT):
    require(variant==VARIANT and data['nc']==1 and data['channels']==3,'Wrong training data contract')
    # The shadow C2 uses the same incoming Trainer RNG. No extra draw reaches real construction.
    with torch.random.fork_rng(devices=[]):
        base80=build('c2');m=state_mapping(base80.yaml,weights.yaml,base80.state_dict())
        base80.load_state_dict({k:weights.state_dict()[v] for k,v in m.items()},strict=True)
        baseline=native_rebuild(base80)
    trainer=RTDETRTrainer.__new__(RTDETRTrainer);trainer.data=data
    target=trainer.get_model(cfg=cfg,weights=weights,verbose=False)
    audit=rebuild_audit(weights,target)
    audit['COMMON']=compare_states(baseline.state_dict(),target.state_dict(),state_mapping(baseline.yaml,target.yaml,baseline.state_dict()))
    audit['native_method']='unchanged RTDETRTrainer.get_model, shadow C2 in fork_rng; no post-rebuild patch'
    return target,audit


def initialize(source, output, variant=VARIANT):
    output=Path(output);require(not output.exists(),'Existing init preserved: '+str(output))
    models,report=controlled_models(source,variant);target=models[VARIANT].eval()
    target.args={**DEFAULT_CFG_DICT,'model':str(MODEL_DIR/CONFIGS[VARIANT]),'task':'detect'}
    target.task='detect';target.pt_path=str(output.resolve())
    checkpoint=dict(epoch=-1,best_fitness=None,model=deepcopy(target).float(),ema=None,updates=None,optimizer=None,scaler=None,
                    train_args=target.args,train_metrics=None,train_results=None,date=datetime.now(timezone.utc).isoformat(),
                    version=ultralytics.__version__,license='AGPL-3.0',c17_lif_v1_provenance=report)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:torch.save(checkpoint,f)
    loaded=RTDETR(str(output)).model
    compare_states(target.state_dict(),loaded.state_dict(),{k:k for k in target.state_dict()})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        _,audit=build_training_model(loaded.yaml,loaded,dict(nc=1,channels=3))
    report.update(output=str(output.resolve()),output_sha256=sha256(output),reload_exact=True,trainer_rebuild=audit,status='passed')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    report=initialize(a.source,a.output);write_json(a.report,report)
    for row in report['COMMON']: print(row['source']+' -> '+row['target'])
    print('PASS: 533 COMMON, 10 NEW_TRAINABLE, 2 NEW_BUFFER; native nc1 rebuild audited')
