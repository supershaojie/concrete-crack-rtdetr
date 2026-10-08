"""Defaults < user YAML < CLI; frozen runs cannot be edited or retuned."""
from __future__ import annotations
import math
import re
import sys
import uuid
from pathlib import Path
from support import HERE, RUN_ROOT, atomic_bytes, canonical, code_identity, digest, git, load_yaml, read_json, sha256, write_json

INTS={'epochs':(1,10000),'patience':(1,10000),'batch_size':(1,128),'eval_batch_size':(1,128),
      'train_workers':(0,128),'eval_workers':(0,128),'imgsz':(32,2048),'seed':(0,4294967295),
      'close_mosaic':(0,10000),'trainable_backbone_layers':(0,5),'amp_growth_interval':(1,1000000000),
      'model_transform_min_size':(32,2048),'model_transform_max_size':(32,2048),
      'box_detections_per_img':(1,300),'rpn_pre_nms_top_n_train':(1,100000),
      'rpn_post_nms_top_n_train':(1,100000),'rpn_pre_nms_top_n_test':(1,100000),'rpn_post_nms_top_n_test':(1,100000)}
NUMS={'lr0':(0,10),'lrf':(0,1),'momentum':(0,1),'weight_decay':(0,1),'warmup_epochs':(0,10000),
      'warmup_start_factor':(0,1),'amp_init_scale':(0,1e12),'box_score_thresh':(.001,1),
      'box_nms_thresh':(0,1),'rpn_nms_thresh':(0,1),'hsv_h':(0,1),'hsv_s':(0,1),'hsv_v':(0,1),
      'degrees':(0,180),'translate':(0,1),'scale':(0,1),'shear':(0,89),'perspective':(0,.01),
      'flipud':(0,1),'fliplr':(0,1),'mosaic':(0,1),'mixup':(0,1),'cutmix':(0,1)}
BOOLS=('early_stopping','amp','deterministic','nesterov')
FIXED={'model':'fasterrcnn_resnet50_fpn','cache_images':False,'multi_scale':False,'optimizer':'SGD',
       'gradient_accumulation_steps':1,'ema':False,'lr_schedule':'warmup_then_cosine',
       'gradient_clip':None,'copy_paste':0.0,'bgr':0.0}
PATHS=('data_yaml','data_root','public_coco','reuse_manifest','weights_file','weights_cache')

def safe_run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value):
        raise ValueError('run-id requires 1-80 ASCII letters/digits/_/-')
    return value

def yaml_mapping(raw):
    import yaml
    class Strict(yaml.SafeLoader): pass
    def mapping(loader,node,deep=False):
        result={}
        for k,v in node.value:
            key=loader.construct_object(k,deep=deep)
            if not isinstance(key,str) or key in result: raise ValueError('Non-string/duplicate config key: '+str(key))
            result[key]=loader.construct_object(v,deep=deep)
        return result
    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    value=yaml.load(raw.decode('utf-8-sig'),Loader=Strict)
    if not isinstance(value,dict): raise ValueError('Expected YAML mapping')
    return value

def cli_overrides(items):
    import yaml
    result={}
    for item in items:
        k,sep,v=item.partition('=')
        if not sep or not v.strip() or k.strip()!=k or k in result: raise ValueError('Invalid/duplicate --set key=value')
        result[k]=yaml.safe_load(v)
        if isinstance(result[k],str) and re.fullmatch(r'[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+',v): result[k]=float(v)
    return result

def validate_config(cfg):
    defaults=load_yaml(HERE/'default_config.yaml')
    if set(cfg)!=set(defaults): raise ValueError('Unknown/missing fields: '+str(set(cfg)^set(defaults)))
    for k,v in FIXED.items():
        if cfg[k]!=v or (isinstance(v,bool) and type(cfg[k]) is not bool) or (type(v) is int and type(cfg[k]) is not int):
            raise ValueError('Unsupported option '+k+'; implemented value is '+repr(v))
    for k,(lo,hi) in INTS.items():
        if type(cfg[k]) is not int or not lo<=cfg[k]<=hi: raise ValueError(k+' invalid integer range')
    for k,(lo,hi) in NUMS.items():
        if type(cfg[k]) not in (int,float) or not math.isfinite(cfg[k]) or not lo<=cfg[k]<=hi: raise ValueError(k+' invalid finite numeric range')
    for k in BOOLS:
        if type(cfg[k]) is not bool: raise ValueError(k+' requires boolean')
    for k in PATHS:
        if cfg[k] is not None and (not isinstance(cfg[k],str) or not cfg[k]): raise ValueError(k+' requires nonempty path/null')
    if not cfg['data_yaml']: raise ValueError('data_yaml is required')
    if cfg['initialization'] not in ('coco_detection_pretrained','random'): raise ValueError('Unknown initialization')
    if cfg['initialization']=='random' and (cfg['weights_file'] or cfg['weights_sha256'] or cfg['trainable_backbone_layers']!=5):
        raise ValueError('Random runs use both weights=None and all backbone layers; no COCO file')
    if cfg['weights_sha256'] is not None and not re.fullmatch('[0-9a-f]{64}',cfg['weights_sha256']): raise ValueError('Invalid weights SHA256')
    if len({cfg[k] for k in ('imgsz','model_transform_min_size','model_transform_max_size')})!=1 or cfg['imgsz']%32:
        raise ValueError('Data/model transform sizes must be equal and divisible by 32')
    if cfg['warmup_epochs']>=cfg['epochs'] or cfg['close_mosaic']>cfg['epochs']: raise ValueError('Invalid epoch boundaries')
    if min(cfg['lr0'],cfg['amp_init_scale'],cfg['warmup_start_factor'])<=0: raise ValueError('LR/AMP/warmup start must be positive')
    if cfg['nesterov'] and cfg['momentum']<=0: raise ValueError('Nesterov needs momentum > 0')
    if cfg['device']!='cpu' and not re.fullmatch(r'cuda:[0-9]+',str(cfg['device'])): raise ValueError('device must be cpu or cuda:N')
    if cfg['amp'] and cfg['device']=='cpu': raise ValueError('AMP requires CUDA; explicitly use amp=false for CPU')
    if cfg['rpn_post_nms_top_n_train']>cfg['rpn_pre_nms_top_n_train'] or cfg['rpn_post_nms_top_n_test']>cfg['rpn_pre_nms_top_n_test']:
        raise ValueError('RPN post-NMS count cannot exceed pre-NMS count')
    return cfg

def resolve_config(user=None, overrides=None):
    defaults=load_yaml(HERE/'default_config.yaml'); user=user or {}; overrides=overrides or {}
    if (set(user)|set(overrides))-set(defaults): raise ValueError('Unknown configuration fields: '+str((set(user)|set(overrides))-set(defaults)))
    validate_config({**defaults,**user})
    return validate_config({**defaults,**user,**overrides})

def candidate(config=None, items=(), clone=None):
    if config and clone: raise ValueError('--config and --clone-config-from are mutually exclusive')
    raw=Path(config).read_bytes() if config else b'{}\n'
    origin={'mode':'yaml' if config else 'committed_defaults','path':str(Path(config).absolute()) if config else None}
    if clone:
        import yaml
        source=Path(clone) if Path(clone).is_absolute() else RUN_ROOT/safe_run_id(str(clone))
        raw=yaml.safe_dump(frozen_config(source,check_code=False),sort_keys=False).encode()
        origin={'mode':'clone_recipe_only','source':str(source.absolute()),'training_state_inherited':False}
    overrides=cli_overrides(items)
    return resolve_config(yaml_mapping(raw),overrides),raw,overrides,origin

def freeze_config(run,cfg,raw,overrides,origin,scope='FORMAL',require_clean=True):
    import yaml
    run=Path(run)
    if require_clean and git('status','--porcelain','--untracked-files=normal'): raise ValueError('Commit code before formal preflight')
    run.mkdir(parents=True,exist_ok=False)
    atomic_bytes(run/'user_config.yaml',raw)
    atomic_bytes(run/'resolved_config.yaml',yaml.safe_dump(cfg,sort_keys=False).encode())
    write_json(run/'cli_overrides.json',overrides); write_json(run/'config_input.json',origin)
    write_json(run/'run_id.json',{'run_id':run.name,'run_uuid':uuid.uuid4().hex,'experiment':'fasterrcnn-r50-fpn-configurable','scope':scope})
    write_json(run/'launch_command.json',{'argv':[sys.executable,*sys.argv],'cwd':str(Path.cwd()),'shell':__import__('os').environ.get('FRCNN_LAUNCH_COMMAND')})
    files=('user_config.yaml','resolved_config.yaml','cli_overrides.json','config_input.json','run_id.json','launch_command.json')
    write_json(run/'config_identity.json',{'config_sha256':digest(canonical(cfg)), 'model_code_sha':git('rev-parse','HEAD'),
        'code_sha256':code_identity(),'files':{n:sha256(run/n) for n in files}})
    return cfg

def frozen_config(run,check_code=True):
    run=Path(run); ident=read_json(run/'config_identity.json')
    for n,h in ident['files'].items():
        if sha256(run/n)!=h: raise ValueError('Frozen run file changed: '+n)
    cfg=load_yaml(run/'resolved_config.yaml'); validate_config(cfg)
    if digest(canonical(cfg))!=ident['config_sha256']: raise ValueError('Frozen configuration changed')
    if check_code and (git('rev-parse','HEAD')!=ident['model_code_sha'] or code_identity()!=ident['code_sha256']):
        raise ValueError('Run code identity differs')
    return cfg
