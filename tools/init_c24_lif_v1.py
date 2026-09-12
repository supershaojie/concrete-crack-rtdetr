"""Public C2 state plus original isolated seed-42 parent initial values."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import sys
from c24_lif_v1_common import *
sys.path.insert(0,str(ROOT/'ultralytics-main'))
import torch
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import SCCAAIFI,LIFDown,Conv,RTDETRDecoder
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from c24_lif_v1_topology import locate,variant_config

MODEL_DIR=ROOT/'ultralytics-main/ultralytics/cfg/models/rt-detr'
CONFIGS=dict(c2='rtdetr-resnet18-lite.yaml',c24='rtdetr-resnet18-lite-scca.yaml',
             lif='rtdetr-resnet18-lite-lif-down.yaml',combo=MODEL)
COUNTS=dict(c2=20082772,c24=20148312,lif=20103876,combo=20169416)

def added(key):
    if '.scca_' in key:return 'c24'
    if len(key.split('.'))>3 and key.split('.')[2] in {'B_proj','P','U_mix','U_dw','O_proj'}:return 'lif'
    return None

def tensor_hash(v):
    return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def build(kind='combo',nc=80):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return RTDETRDetectionModel(str(MODEL_DIR/CONFIGS[kind]),nc=nc,verbose=False)

def runtime():
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT/'ultralytics-main'),'Wrong ultralytics source')
    return dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,ultralytics=ultralytics.__file__,
        scca=str(ROOT/'ultralytics-main/ultralytics/nn/modules/scca_aifi.py'),
        lif=str(ROOT/'ultralytics-main/ultralytics/nn/modules/lif_down.py'),commit=git('rev-parse','HEAD'))

def verify_model(model,zero=False):
    base=YAML.load(MODEL_DIR/CONFIGS['c2']);expected=variant_config(base);top=locate(model.yaml)
    require(type(model) is RTDETRDetectionModel,'Native model required')
    require(all(model.yaml[k]==expected[k] for k in ['backbone','head','scales']),'Only two replacements allowed')
    require(type(model.model[top['aifi']]) is SCCAAIFI,'Missing original SCCA')
    lif=model.model[top['p3_to_p4']['downsample']]
    require(type(lif) is LIFDown and hasattr(lif,'bn'),'Original LIF with BN required')
    require(type(model.model[top['p4_to_p5']['downsample']]) is Conv,'Second downsample changed')
    head=model.model[top['decoder']]
    require(type(head) is RTDETRDecoder and head.f==top['decoder_inputs'] and head.num_queries==300 and
            head.decoder.num_layers==3 and len(head.input_proj)==3,'Decoder contract changed')
    require(sum(type(m) is SCCAAIFI for m in model.modules())==sum(type(m) is LIFDown for m in model.modules())==1,'Extra module')
    require(sum(p.numel() for n,p in model.named_parameters() if added(n))==86644,'New parameter count')
    require(len(model.state_dict())==543,'Unexpected state schema')
    require(not any(added(n) for n,_ in model.named_buffers()),'New buffers forbidden')
    if head.nc==1:require(sum(p.numel() for p in model.parameters())==COUNTS['combo'],'nc1 parameter count')
    if zero:
        s=model.model[top['aifi']]
        require(torch.count_nonzero(s.scca_o.weight)==torch.count_nonzero(lif.O_proj.weight)==0,'Output initialization changed')
    return top

def controlled_models(source):
    require(Path(source).is_file() and sha256(source)==SOURCE_SHA256,'Public source SHA256 mismatch')
    ckpt=torch_load(source,map_location='cpu')
    require(ckpt.get('epoch')==-1 and all(ckpt.get(k) is None for k in
        ('ema','optimizer','scaler','updates','train_metrics','train_results','best_fitness')),'Trained/resumable source forbidden')
    public=ckpt['model'].float().state_dict();models={k:build(k) for k in CONFIGS}
    require(ckpt['model'].model[-1].nc==80,'Public source must be nc80')
    require(public.keys()==models['c2'].state_dict().keys(),'Public key mismatch')
    require(all(public[k].shape==v.shape for k,v in models['c2'].state_dict().items()),'Public shape mismatch')
    reference={k:v.clone() for k,v in models['c2'].state_dict().items()}
    for kind,m in models.items():
        state=m.state_dict()
        require(all(torch.equal(v,state[k]) for k,v in reference.items()),'Constructor advanced public RNG: '+kind)
        require(set(state)-set(reference)=={k for k in state if added(k)},'Unexpected parent state')
        m.load_state_dict({**state,**public},strict=True)
    target=models['combo'];state=target.state_dict();parent_rows=[]
    for k in state:
        kind=added(k)
        if kind:
            parent=models[kind].state_dict()[k]
            parent_rows.append(dict(key=k,parent=kind,constructor_equal=torch.equal(state[k],parent),sha256=tensor_hash(parent)))
            state[k]=parent.clone()
    target.load_state_dict(state,strict=True);verify_model(target,zero=True)
    params=dict(target.named_parameters())
    rows=[dict(key=k,category='NEW_TRAINABLE' if added(k) else 'COMMON',source=added(k) or 'public_c2',
               shape=list(v.shape),sha256=tensor_hash(v),equal=torch.equal(v,models[added(k)].state_dict()[k] if added(k) else public[k]))
          for k,v in target.state_dict().items()]
    require(all(r['equal'] for r in rows),'Mapped value mismatch')
    report=dict(source=str(Path(source).resolve()),source_sha256=SOURCE_SHA256,source_nc=80,source_epoch=-1,seed=42,
        fingerprint=fingerprint(),rows=rows,parent_initial_values=parent_rows,COMMON=[r['key'] for r in rows if r['category']=='COMMON'],
        NEW_TRAINABLE=[k for k in params if added(k)],NEW_BUFFER=[],ALLOWED_CLASS_ADAPTATION=[],MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[])
    return models,report

def rebuild_audit(source,target):
    verify_model(target,zero=True);before=source.state_dict();after=target.state_dict();head=locate(target.yaml)['decoder']
    allowed={f'model.{head}.denoising_class_embed.weight',f'model.{head}.enc_score_head.weight',f'model.{head}.enc_score_head.bias'}
    allowed|={f'model.{head}.dec_score_head.{i}.{suffix}' for i in range(3) for suffix in ('weight','bias')}
    require(before.keys()==after.keys(),'Native Trainer state schema changed')
    skipped={k for k in before if before[k].shape!=after[k].shape}
    require(skipped==allowed,'Expected exactly nine classification adaptations')
    require(all(torch.equal(v.cpu(),after[k].cpu()) for k,v in before.items() if k not in skipped),'Trainer did not load exact values/buffers')
    return dict(loaded_exact=len(after)-len(skipped),COMMON=[k for k in after if k not in skipped and not added(k)],
        NEW_TRAINABLE=[k for k in after if added(k)],NEW_BUFFER=[],ALLOWED_CLASS_ADAPTATION=sorted(skipped),
        MISSING=[],UNEXPECTED=[],SHAPE_MISMATCH=[],rows=[dict(key=k,category='ALLOWED_CLASS_ADAPTATION' if k in skipped else
        'NEW_TRAINABLE' if added(k) else 'COMMON',before_shape=list(before[k].shape),after_shape=list(v.shape),sha256=tensor_hash(v)) for k,v in after.items()])

def native_rebuild(source,kind='combo'):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer);trainer.data=dict(nc=1,channels=3)
    return RTDETRTrainer.get_model(trainer,cfg=str(MODEL_DIR/CONFIGS[kind]),weights=source,verbose=False)

def audited_rebuild(source):
    # Both constructions see the native caller RNG. Only the actual combo advances it once.
    with torch.random.fork_rng(devices=[]):baseline=native_rebuild(source,'c2')
    target=native_rebuild(source);report=rebuild_audit(source,target)
    require(all(torch.equal(v,target.state_dict()[k]) for k,v in baseline.state_dict().items()),'Native C2 class initialization differs')
    report['native_c2_nc1_exact']=True
    return target,report

def initialize(source,output):
    output=Path(output);models,report=controlled_models(source);target=models['combo'].eval()
    target.args={**DEFAULT_CFG_DICT,'model':str(MODEL_DIR/MODEL),'task':'detect'};target.task='detect';target.pt_path=str(output.resolve())
    ckpt=dict(epoch=-1,best_fitness=None,model=deepcopy(target).float(),ema=None,updates=None,optimizer=None,scaler=None,
        train_args=target.args,train_metrics=None,train_results=None,date=datetime.now(timezone.utc).isoformat(),
        version=ultralytics.__version__,c24_lif_v1_provenance=report)
    if output.exists():
        previous=torch_load(output,map_location='cpu')
        require(previous.get('c24_lif_v1_provenance')==report,'Existing init provenance differs; preserved')
        require(previous.get('epoch')==-1 and all(previous.get(k) is None for k in ['optimizer','ema','updates','scaler']),'Existing init has training state')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_name(output.name+'.'+__import__('uuid').uuid4().hex+'.partial')
        with temp.open('xb') as f:torch.save(ckpt,f)
        os.link(temp,output);temp.unlink()
    reloaded=RTDETR(str(output)).model
    require(target.state_dict().keys()==reloaded.state_dict().keys() and all(torch.equal(v,reloaded.state_dict()[k]) for k,v in target.state_dict().items()),'Saved init changed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);_,audit=audited_rebuild(reloaded)
    return dict(report,output=str(output.resolve()),output_sha256=sha256(output),reload_exact=True,native_trainer=audit,runtime=runtime())

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);write_json(a.report,initialize(a.source,a.output))
