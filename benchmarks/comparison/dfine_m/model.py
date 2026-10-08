"""Unmodified official D-FINE-M under a private namespace; strict COCO tuning."""
from __future__ import annotations
import copy
import importlib
import inspect
from pathlib import Path
import re
import sys
import types
import torch
from support import (HERE,LOCK,atomic_bytes,checked_source,checked_weight,digest,canonical,
                     load_yaml,read_json,sha256,write_json)
PREFIX='_crack_dfine_official'

def package(name,path):
    if name in sys.modules:
        if list(sys.modules[name].__path__)!=[str(path)]: raise RuntimeError('Official namespace already belongs to another source')
        return sys.modules[name]
    m=types.ModuleType(name); m.__path__=[str(path)]; m.__package__=name
    sys.modules[name]=m; return m

def official_modules(source):
    source,_=checked_source(source); base=source/'src'
    root=package(PREFIX,base)
    # Skip eager package registries for unrelated datasets/backbones/solver/profiler.
    # Execute original workspace, native architecture, criterion, matcher and EMA files unchanged.
    for rel in ('core','nn','nn/backbone','zoo','misc','optim','data'):
        package(PREFIX+'.'+rel.replace('/','.'),base/rel)
    workspace=importlib.import_module(PREFIX+'.core.workspace')
    core=sys.modules[PREFIX+'.core']; core.register=workspace.register; core.create=workspace.create
    core.GLOBAL_CONFIG=workspace.GLOBAL_CONFIG
    sys.modules[PREFIX+'.data'].DataLoader=torch.utils.data.DataLoader
    importlib.import_module(PREFIX+'.nn.backbone.hgnetv2')
    native=importlib.import_module(PREFIX+'.zoo.dfine')
    ema=importlib.import_module(PREFIX+'.optim.ema').ModelEMA
    for name,mod in list(sys.modules.items()):
        if name.startswith(PREFIX+'.') and getattr(mod,'__file__',None):
            if not Path(mod.__file__).resolve().is_relative_to(base): raise RuntimeError('Wrong official module import: '+name)
    return native,ema,workspace

def merge(left,right):
    for k,v in right.items():
        if isinstance(v,dict) and isinstance(left.get(k),dict): merge(left[k],v)
        else: left[k]=copy.deepcopy(v)
    return left

def expand_config(source,path=None,inventory=None,stack=()):
    source=Path(source); path=Path(path or source/read_json(LOCK)['config']).resolve()
    if path in stack: raise ValueError('Recursive official config include')
    obj=load_yaml(path); result={}
    if inventory is not None: inventory[path.relative_to(source).as_posix()]=sha256(path)
    for rel in obj.pop('__include__',[]):
        merge(result,expand_config(source,path.parent/rel,inventory,(*stack,path)))
    return merge(result,obj)

def structure(cfg,source):
    expanded=expand_config(source); adapted=copy.deepcopy(expanded)
    adapted.update(num_classes=1,remap_mscoco_category=False,eval_spatial_size=[640,640])
    adapted['HGNetv2']['pretrained']=False  # Full COCO checkpoint covers backbone; no ImageNet fetch.
    adapted['DFINEPostProcessor']['num_top_queries']=300
    # These objects are not built by this runner. Save actual replacement objects separately.
    for key in ('train_dataloader','val_dataloader','lr_scheduler','lr_warmup_scheduler','evaluator'):
        adapted.pop(key,None)
    adapted.update(epochs=cfg['epochs'],use_amp=cfg['amp'],use_ema=cfg['ema'])
    return expanded,adapted

def build(cfg,source,device='cpu',audit=None):
    native,EMA,workspace=official_modules(source)
    original,adapted=structure(cfg,source)
    utils=importlib.import_module(PREFIX+'.core.yaml_utils')
    global_cfg=utils.merge_config(adapted,workspace.GLOBAL_CONFIG,inplace=False)
    model=workspace.create(adapted['model'],global_cfg).to(device)
    criterion=workspace.create(adapted['criterion'],global_cfg).to(device)
    post=workspace.create(adapted['postprocessor'],global_cfg).to(device)
    if (type(model.backbone).__name__!='HGNetv2' or adapted['HGNetv2']['name']!='B2'
        or adapted['HGNetv2']['return_idx']!=[1,2,3]
        or list(model.encoder.in_channels)!=[384,768,1536]
        or model.encoder.hidden_dim!=256 or model.decoder.num_layers!=4
        or model.decoder.num_queries!=300 or post.num_top_queries!=300
        or criterion.num_classes!=1 or post.num_classes!=1 or post.remap_mscoco_category):
        raise ValueError('Actual official M architecture/class/postprocessor differs')
    model.float(); criterion.float(); post.float()
    if audit is not None:
        audit=Path(audit); inventory={}; expand_config(source,inventory=inventory)
        import yaml
        atomic_bytes(audit/'official_expanded.yaml',yaml.safe_dump(original,sort_keys=False).encode())
        atomic_bytes(audit/'model_expanded.yaml',yaml.safe_dump(adapted,sort_keys=False).encode())
        for name in inventory:
            atomic_bytes(audit/'official_includes'/name,(Path(source)/name).read_bytes())
        write_json(audit/'model_objects.json',{'official_include_sha256':inventory,
            'constructor_config':adapted,'model':repr(model),'criterion':repr(criterion),'postprocessor':repr(post),
            'parameters':sum(p.numel() for p in model.parameters()),
            'visualization_candidate_layers':[n for n,m in model.named_modules() if isinstance(m,torch.nn.Conv2d) and n.startswith(('backbone.stages.','encoder.'))],
            'module_sources':{n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
                if n.startswith(PREFIX+'.') and getattr(m,'__file__',None)},
            'normalization':'RGB float32 [0,1]; official M has no external mean/std transform',
            'continuous_training':'No official solver stage1 reload/EMA restart/120 stop policy; native FDR/GO-LSD unchanged'})
    return model,criterion,post,EMA

CLASS_KEY=re.compile(r'^decoder\.(?:denoising_class_embed\.weight|enc_score_head\.(?:weight|bias)|dec_score_head\.\d+\.(?:weight|bias))$')

def load_coco(model,path,audit=None):
    path=checked_weight(path); checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    key='ema.module' if 'ema' in checkpoint else 'model'
    source=checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    if key!=read_json(LOCK)['weights']['selected_initialization_key']: raise ValueError('Official asset checkpoint key changed')
    fresh=model.state_dict(); loaded={}; skipped={}
    if set(source)!=set(fresh): raise ValueError('COCO checkpoint has missing/extra native model keys')
    for k,v in fresh.items():
        if source[k].shape!=v.shape:
            if not CLASS_KEY.fullmatch(k): raise ValueError('Non-class shape mismatch: '+k)
            skipped[k]={'source_shape':list(source[k].shape),'target_shape':list(v.shape),'reason':'class-count-only reinitialization'}
        else: loaded[k]=source[k]
    if len(skipped)!=11: raise ValueError('Unexpected class reconstruction keys for official M')
    model.load_state_dict({**fresh,**loaded},strict=True)
    report={'mode':'fresh_COCO_fine_tune','source_checkpoint_key':key,'checkpoint_keys':list(checkpoint),
        'weight_path':str(path),'weight_sha256':sha256(path),'source':read_json(LOCK)['weights'],
        'loaded':{k:list(v.shape) for k,v in loaded.items()},'skipped':skipped,'new_initialized':list(skipped),
        'initialization_order':['construct official M with pretrained=False','load compatible full COCO tensors','construct new EMA/optimizer/scheduler/scaler'],
        'strict_load':'Merged exact fresh class-only keys + all COCO compatible keys, strict=True',
        'resume_state_inherited':False,'imagenet_backbone_download':False}
    print('COCO tuning: source='+key+' loaded='+str(len(loaded))+' class-only new='+str(len(skipped)),flush=True)
    if audit is not None: write_json(audit,report)
    return report

def selected_checkpoint(path,identity=None):
    path=Path(path); ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if ckpt.get('schema')!='dfine_m_selected_v1' or ckpt.get('selected_key')!='selected_state_dict':
        raise ValueError('Expected a D-FINE selected checkpoint, not COCO/last/another model')
    if identity is not None and ckpt.get('identity')!=identity: raise ValueError('Selected checkpoint run identity differs')
    required='ema' if ckpt['identity']['ema'] else 'model'
    if ckpt.get('selected_source')!=required: raise ValueError('Selected EMA/model source differs from explicit recipe')
    state=ckpt['selected_state_dict']
    if any(v.is_floating_point() and v.dtype!=torch.float32 for v in state.values()): raise ValueError('Selected checkpoint is not FP32')
    return ckpt

def load_for_visualization(config,checkpoint,device='cpu'):
    """Return (gradient-capable native model, metadata); no deploy/fusion/no_grad wrapper."""
    from configuration import frozen_config,paths
    if isinstance(config,dict): cfg=config
    else:
        config=Path(config)
        cfg=frozen_config(config.parent) if (config.parent/'config_identity.json').exists() else load_yaml(config)
    ckpt=selected_checkpoint(checkpoint)
    if digest(canonical(cfg))!=ckpt['identity']['config_sha256']: raise ValueError('Visualization configuration differs from selected checkpoint')
    model,_,_,_=build(cfg,paths(cfg)['source'],device)
    model.load_state_dict(ckpt['selected_state_dict'],strict=True); model.float().eval()
    # Fresh native construction retains trainable floating weights and fixed integer/constants.
    # State-dict restoration does not inherit EMA's requires_grad=False flags.
    info={'checkpoint':str(Path(checkpoint).absolute()),'sha256':sha256(checkpoint),
        'selected_key':'selected_state_dict','selected_source':ckpt['selected_source'],'epoch':ckpt['epoch'],
        'forward_format':'dict: pred_logits [B,300,1] (raw logits), pred_boxes [B,300,4] normalized cxcywh',
        'candidate_layers':[n for n,m in model.named_modules() if (n.startswith('backbone.stages.') or n.startswith('encoder.')) and isinstance(m,torch.nn.Conv2d)],
        'input':'RGB float32 [0,1] letterbox640; inverse uses round-derived gain_x/gain_y,left/top',
        'gradient_capable':True,'deploy_or_fusion':False}
    return model,info
