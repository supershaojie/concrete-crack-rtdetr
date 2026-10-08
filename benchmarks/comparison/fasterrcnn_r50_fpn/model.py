"""Official standard COCO construction followed by a two-class predictor."""
from __future__ import annotations
import math
import os
from pathlib import Path
import random
import re
import time
import urllib.request
from unittest.mock import patch
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from support import LOCK, RUNTIME, local_lock, read_json, recipe, sha256, write_json

LOSS_KEYS={'loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg'}

def seed_all(seed=42,deterministic=True):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=deterministic
    torch.use_deterministic_algorithms(deterministic,warn_only=True)

def build_kwargs(cfg):
    return {'initialization':cfg['initialization'],'num_classes_after_transfer':2,
        'trainable_backbone_layers':cfg['trainable_backbone_layers'],
        'min_size':cfg['model_transform_min_size'],'max_size':cfg['model_transform_max_size'],
        **{k:cfg[k] for k in ('box_score_thresh','box_nms_thresh','box_detections_per_img','rpn_nms_thresh',
            'rpn_pre_nms_top_n_train','rpn_post_nms_top_n_train','rpn_pre_nms_top_n_test','rpn_post_nms_top_n_test')}}

def prepare_weights(cfg):
    if cfg['initialization']=='random': return {'mode':'random','weights':None,'weights_backbone':None}
    url=Weights.COCO_V1.url
    filename=Path(urllib.request.urlparse(url).path).name if hasattr(urllib.request,'urlparse') else url.rsplit('/',1)[1]
    prefix=re.search(r'-([0-9a-f]+)\.',filename).group(1)
    expected=cfg['weights_sha256'] or read_json(LOCK)['coco_v1']['sha256']
    cache=Path(cfg['weights_cache'] or RUNTIME/'weights').absolute()
    destination=Path(cfg['weights_file']).absolute() if cfg['weights_file'] else cache/filename
    def validate():
        actual=sha256(destination)
        if not actual.startswith(prefix) or actual!=expected:
            raise ValueError('Official COCO weight SHA mismatch: '+str(destination))
        return actual
    if cfg['weights_file'] and not destination.is_file(): raise FileNotFoundError('weights_file missing: '+str(destination))
    cache.mkdir(parents=True,exist_ok=True)
    with local_lock(cache/'download.lock'):
        if not destination.exists():
            temporary=destination.with_suffix('.download.partial')
            for attempt in range(1,4):
                print(f'COCO download {attempt}/3: {url} -> {destination}',flush=True)
                try:
                    with urllib.request.urlopen(url,timeout=45) as response, temporary.open('wb') as out:
                        size=int(response.headers.get('Content-Length',0)); completed=0; last=-1
                        while True:
                            block=response.read(1024*1024)
                            if not block: break
                            out.write(block); completed+=len(block)
                            bucket=completed//(8*1024*1024)
                            if bucket!=last:
                                print(f'COCO bytes {completed}/{size or "unknown"}',flush=True); last=bucket
                        out.flush(); os.fsync(out.fileno())
                    if size and completed!=size: raise IOError('Incomplete COCO download')
                    actual=sha256(temporary)
                    if not actual.startswith(prefix) or actual!=expected: raise ValueError('Downloaded COCO hash mismatch')
                    os.replace(temporary,destination); break
                except Exception:
                    if attempt==3: raise
                    time.sleep(min(attempt*2,5))
        actual=validate()
    record={'mode':'coco_detection_pretrained','enum':'FasterRCNN_ResNet50_FPN_Weights.COCO_V1',
        'url':url,'file':str(destination),'filename':filename,'sha256':actual,
        'verification':'official filename prefix and complete measured locked SHA256; optional supplied full SHA',
        'actual_load':'official weights builder path, validated cached state supplied to enum.get_state_dict'}
    write_json(cache/'coco_receipt.json',record)
    return record

def assert_backbone_size(module,inputs):
    size=module.comparison_input_size
    if tuple(inputs[0].shape[-2:])!=(size,size): raise ValueError('Backbone received wrong input shape: '+str(inputs[0].shape))
    module.comparison_observed_size=list(inputs[0].shape[-2:])

def structure(model):
    from torchvision.ops.misc import FrozenBatchNorm2d
    bn=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)]
    frozen_bn=[(n,m) for n,m in model.named_modules() if isinstance(m,FrozenBatchNorm2d)]
    return {'model_class':type(model).__module__+'.'+type(model).__name__,
        'resnet_blocks':[len(model.backbone.body[k]) for k in ('layer1','layer2','layer3','layer4')],
        'box_head':type(model.roi_heads.box_head).__name__,'box_predictor':str(model.roi_heads.box_predictor),
        'parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_backbone_keys':[n for n,p in model.backbone.named_parameters() if p.requires_grad],
        'frozen_backbone_keys':[n for n,p in model.backbone.named_parameters() if not p.requires_grad],
        'batchnorm_count':len(bn),'frozen_batchnorm_count':len(frozen_bn),
        'normalization_modules':{n:type(m).__name__ for n,m in bn+frozen_bn},
        'normalization_statistics':'FrozenBatchNorm buffers remain fixed during COCO fine-tuning' if frozen_bn else 'BatchNorm native running statistics',
        'anchors':{'sizes':model.rpn.anchor_generator.sizes,'aspect_ratios':model.rpn.anchor_generator.aspect_ratios},
        'rpn':{'nms_thresh':model.rpn.nms_thresh,'score_thresh':model.rpn.score_thresh,
               'pre_nms_top_n':model.rpn._pre_nms_top_n,'post_nms_top_n':model.rpn._post_nms_top_n},
        'roi':{'score_thresh':model.roi_heads.score_thresh,'nms_thresh':model.roi_heads.nms_thresh,'detections_per_img':model.roi_heads.detections_per_img},
        'roi_align':str(model.roi_heads.box_roi_pool),
        'transform':{'min_size':model.transform.min_size,'max_size':model.transform.max_size,
                     'mean':model.transform.image_mean,'std':model.transform.image_std,'size_divisible':model.transform.size_divisible},
        'losses':sorted(LOSS_KEYS),'fusion':'none'}

def build_model(cfg=None):
    cfg=cfg or recipe(); spec=build_kwargs(cfg); init=prepare_weights(cfg)
    kwargs={k:v for k,v in spec.items() if k not in ('initialization','num_classes_after_transfer')}
    loaded=[]; excluded=[]; source_elements=0
    if cfg['initialization']=='coco_detection_pretrained':
        source=torch.load(init['file'],map_location='cpu',weights_only=True)
        with patch.object(Weights,'get_state_dict',return_value=source):
            model=fasterrcnn_resnet50_fpn(weights=Weights.COCO_V1,weights_backbone=None,**kwargs)
        before=model.state_dict()
        # The official FPN/RPN loaders migrate legacy checkpoint names (version<2).
        # Check that only these exact official renames occurred and all loaded values are equal.
        migrated={}; remapped={}; consumed=[]
        for key,value in source.items():
            destination=key
            match=re.fullmatch(r'backbone.fpn.(inner_blocks|layer_blocks).([0-9]+).(weight|bias)',key)
            if match and key not in before:
                destination=f'backbone.fpn.{match[1]}.{match[2]}.0.{match[3]}'
            if key in ('rpn.head.conv.weight','rpn.head.conv.bias') and key not in before:
                destination='rpn.head.conv.0.0.'+key.rsplit('.',1)[1]
            if key.endswith('num_batches_tracked') and destination not in before:
                consumed.append(key); continue
            if destination in migrated: raise ValueError('Duplicate official weight migration key')
            migrated[destination]=value
            if destination!=key: remapped[key]=destination
        if set(migrated)!=set(before) or not all(torch.equal(before[k],migrated[k]) for k in before):
            raise ValueError('Unexpected COCO construction/load mismatch beyond official FPN/RPN migration')
        excluded=sorted(k for k in source if k.startswith('roi_heads.box_predictor.'))
        loaded=sorted(k for k in before if not k.startswith('roi_heads.box_predictor.'))
        source_elements=sum(migrated[k].numel() for k in loaded)
        model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
        del source,before,migrated
    else:
        model=fasterrcnn_resnet50_fpn(weights=None,weights_backbone=None,num_classes=2,**kwargs)
    model.backbone.comparison_input_size=cfg['imgsz']
    model.backbone.register_forward_pre_hook(assert_backbone_size)
    observed=structure(model)
    if observed['resnet_blocks']!=[3,4,6,3] or observed['box_head']!='TwoMLPHead': raise ValueError('Wrong standard architecture')
    if cfg['initialization']=='coco_detection_pretrained' and (observed['batchnorm_count'] or not observed['frozen_batchnorm_count']):
        raise ValueError('COCO builder must retain FrozenBatchNorm')
    new=sorted(k for k in model.state_dict() if k.startswith('roi_heads.box_predictor.'))
    return model,{**init,'constructor':spec,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'official_legacy_key_renames':remapped if loaded else {},'consumed_FrozenBN_bookkeeping':consumed if loaded else [],
        'loaded_keys':loaded,'loaded_key_count':len(loaded),'loaded_elements':source_elements,
        'excluded_source_keys':excluded,'excluded_key_count':len(excluded),
        'new_initialized_keys':new if loaded else sorted(model.state_dict()),
        'new_initialized_key_count':len(new) if loaded else len(model.state_dict()),'structure':observed}

def finite_losses(losses):
    if set(losses)!=LOSS_KEYS or not all(torch.isfinite(v).all() for v in losses.values()): raise FloatingPointError('Four finite native losses required')
    total=sum(losses.values())
    if not torch.isfinite(total).all(): raise FloatingPointError('Nonfinite total loss')
    return total

def finite_gradients(model):
    count=0
    for name,p in model.named_parameters():
        if p.grad is not None:
            count+=1
            if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite unscaled gradient: '+name)
    if not count: raise FloatingPointError('No gradients')
    return count
