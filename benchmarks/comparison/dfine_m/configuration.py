"""Strict defaults -> YAML -> CLI and immutable run configuration."""
from __future__ import annotations
import math
import os
from pathlib import Path
import re
import shlex
import sys
import uuid
import yaml
from support import (HERE,ROOT,SOURCE,ASSET,LOCK,adapter_hash,atomic_bytes,canonical,digest,
                     environment,git,load_yaml,read_json,sha256,write_json)
RUN_ROOT=ROOT/'outputs/dfine-m-configurable'
FIXED=('model','initialization','num_classes','early_stopping','imgsz','cache_images','multi_scale',
       'optimizer','lr_schedule','ema_stage_restart','eval_conf','eval_max_det','eval_nms','copy_paste','bgr')
PATHS=('data_yaml','data_root','public_coco','reuse_manifest','source','weights','run_root')
INTS={'epochs':(1,10000),'batch_size':(1,256),'eval_batch_size':(1,256),'train_workers':(0,128),
      'eval_workers':(0,128),'seed':(0,2**32-1),'gradient_accumulation_steps':(1,128),
      'ema_warmup_updates':(0,10000000),'close_mosaic':(0,10000),'print_freq':(1,100000),
      'preflight_steps':(1,20),'amp_growth_interval':(1,1000000)}
FLOATS={**{k:(0,1) for k in ('lr0','backbone_lr','weight_decay','warmup_start_factor','lrf','ema_decay',
    'hsv_h','hsv_s','hsv_v','translate','scale','flipud','fliplr','mosaic','mixup','cutmix')},
    'degrees':(0,180),'shear':(0,180),'perspective':(0,.001),'warmup_epochs':(0,10000),
    'gradient_clip_max_norm':(0,1000),'amp_init_scale':(1e-8,2**32),
    'amp_growth_factor':(1.000001,100),'amp_backoff_factor':(0.000001,.999999)}

def safe_run_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value):
        raise ValueError('run-id: 1-80 letters/digits/underscore/hyphen')
    return value

def yaml_mapping(raw):
    class Strict(yaml.SafeLoader): pass
    def mapping(loader,node,deep=False):
        d={}
        for key,val in node.value:
            k=loader.construct_object(key,deep=deep)
            if not isinstance(k,str) or k in d: raise ValueError('Invalid/duplicate YAML key: '+str(k))
            d[k]=loader.construct_object(val,deep=deep)
        return d
    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    obj=yaml.load(raw.decode('utf-8-sig'),Loader=Strict)
    if not isinstance(obj,dict): raise ValueError('Config must be a mapping')
    return obj

def cli_overrides(items):
    d={}
    for item in items:
        key,sep,value=item.partition('=')
        if not sep or not value or key in d or key.strip()!=key: raise ValueError('Invalid/duplicate --set '+item)
        d[key]=yaml.safe_load(value)
    return d

def resolve_config(user=None,overrides=None):
    defaults=load_yaml(HERE/'default_config.yaml'); user=user or {}; overrides=overrides or {}
    unknown=(set(user)|set(overrides))-set(defaults)
    if unknown: raise ValueError('Unknown configuration fields: '+', '.join(sorted(unknown)))
    for values in ({**defaults,**user},{**defaults,**user,**overrides}):
        for k in FIXED:
            if type(values[k]) is not type(defaults[k]) or values[k]!=defaults[k]:
                raise ValueError(k+' is fixed for this implemented model/public protocol')
        for k,(low,high) in INTS.items():
            if type(values[k]) is not int or not low<=values[k]<=high: raise ValueError(k+' outside integer range')
        for k,(low,high) in FLOATS.items():
            if isinstance(values[k],str):
                try: values[k]=float(values[k])
                except ValueError: pass
            v=values[k]
            if type(v) not in (int,float) or not math.isfinite(v) or not low<=v<=high:
                raise ValueError(k+' outside finite numeric range')
        for k in ('amp','deterministic','ema','drop_last','enforce_counts'):
            if type(values[k]) is not bool: raise ValueError(k+' must be boolean')
        if not isinstance(values['betas'],list) or len(values['betas'])!=2 or any(
            type(x) not in (int,float) or not 0<x<1 for x in values['betas']): raise ValueError('Invalid AdamW betas')
        for k in PATHS:
            if values[k] is not None and (not isinstance(values[k],str) or not values[k]): raise ValueError(k+' must be path or null')
        if not values['data_yaml']: raise ValueError('data_yaml is required')
        if not isinstance(values['device'],str) or not re.fullmatch(r'cpu|cuda:[0-9]+',values['device']): raise ValueError('device: cpu or cuda:index')
        if min(values['lr0'],values['backbone_lr'],values['lrf'],values['warmup_start_factor'])<=0: raise ValueError('LR/factors must be positive')
        if values['close_mosaic']>values['epochs'] or values['warmup_epochs']>=values['epochs']: raise ValueError('close_mosaic<=epochs, warmup_epochs<epochs required')
    return values

def candidate(config=None,items=(),clone=None):
    if config and clone: raise ValueError('config and clone-config-from are mutually exclusive')
    raw=Path(config).read_bytes() if config else b'{}\n'; user=yaml_mapping(raw)
    if clone:
        user=frozen_config(Path(clone),check_runtime=False)
        raw=yaml.safe_dump(user,sort_keys=False).encode()
    overrides=cli_overrides(items)
    return resolve_config(user,overrides),raw,overrides

def paths(cfg):
    result={k:str(Path(cfg[k] or default).absolute()) for k,default in
        [('source',SOURCE),('weights',ASSET),('run_root',RUN_ROOT)]}
    return result

def freeze(run,run_id,cfg,raw,overrides,smoke=False):
    if smoke and not run_id.startswith('SMOKE_ONLY_'): raise ValueError('Smoke run ID must start SMOKE_ONLY_')
    if not smoke and (not cfg['enforce_counts'] or run_id.startswith('SMOKE_')): raise ValueError('Formal run requires frozen real data counts')
    if not smoke and git('status','--porcelain','--untracked-files=normal'): raise ValueError('Commit changes before preparing a formal run')
    run=Path(run); run.mkdir(parents=True,exist_ok=False)
    atomic_bytes(run/'user_config.yaml',raw)
    atomic_bytes(run/'resolved_config.yaml',yaml.safe_dump(cfg,sort_keys=False).encode())
    write_json(run/'cli_overrides.json',overrides)
    write_json(run/'runtime_paths.json',{**paths(cfg),'python':os.path.abspath(sys.executable),'sys_prefix':os.path.abspath(sys.prefix)})
    write_json(run/'run_id.json',{'run_id':safe_run_id(run_id),'run_uuid':uuid.uuid4().hex,'scope':'SMOKE_ONLY' if smoke else 'FORMAL'})
    write_json(run/'launch_command.json',{'argv':[sys.executable,*sys.argv],'bash':shlex.join([sys.executable,*sys.argv]),
        'shell_invocation':os.environ.get('DFINE_LAUNCH_COMMAND'),'cwd':str(Path.cwd())})
    env=environment(); write_json(run/'environment.json',env)
    files=('user_config.yaml','resolved_config.yaml','cli_overrides.json','runtime_paths.json','run_id.json','launch_command.json','environment.json')
    write_json(run/'config_identity.json',{'config_sha256':digest(canonical(cfg)),'model_code_sha':git('rev-parse','HEAD'),
        'adapter_sha256':adapter_hash(),'files':{n:sha256(run/n) for n in files},'scope':'SMOKE_ONLY' if smoke else 'FORMAL',
        'upstream':read_json(LOCK),'environment_sha256':digest(canonical(env))})
    return cfg

def frozen_config(run,check_runtime=True):
    run=Path(run); rec=read_json(run/'config_identity.json')
    for n,expected in rec['files'].items():
        if sha256(run/n)!=expected: raise ValueError('Frozen artifact changed: '+n)
    cfg=load_yaml(run/'resolved_config.yaml')
    if digest(canonical(cfg))!=rec['config_sha256']: raise ValueError('Frozen config changed')
    if check_runtime:
        if git('rev-parse','HEAD')!=rec['model_code_sha'] or adapter_hash()!=rec['adapter_sha256']:
            raise ValueError('Run code identity differs; use the frozen commit')
        if rec['scope']=='FORMAL' and git('status','--porcelain','--untracked-files=normal'):
            raise ValueError('Formal worktree must remain clean')
        frozen=read_json(run/'runtime_paths.json')
        if os.path.abspath(sys.executable)!=frozen['python'] or os.path.abspath(sys.prefix)!=frozen['sys_prefix']:
            raise ValueError('Use the recorded interpreter: '+frozen['python'])
        if environment()!=read_json(run/'environment.json'): raise ValueError('Frozen environment changed')
    return cfg
