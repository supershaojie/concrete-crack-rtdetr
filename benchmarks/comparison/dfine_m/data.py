"""Read-only shared labels/images, COCO derivation, explicit native label0 targets."""
from __future__ import annotations
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import augmentation_reference as augment
from augment_b19 import counters,describe_transform,snapshot,training_transforms
from bbox_reference import Instances
from support import atomic_bytes,canonical,digest,load_yaml,read_json,sha256,write_json
from light_data import (preflight,reuse_preflight,verify_inputs,image_size,annotation_rows,reuse_gt)

def matching_reuse_inventory(cfg):
    yaml_path=Path(cfg['data_yaml']).resolve(); data_cfg=load_yaml(yaml_path)
    root=Path(cfg['data_root'] or data_cfg.get('path',yaml_path.parent))
    root=(root if root.is_absolute() else yaml_path.parent/root).resolve()
    projects={yaml_path.parent.parent}
    for name in ('Crack_RTDETR-bench-yolo26m-configurable','Crack_RTDETR-bench-yolov5m-coco-native-ft-v1',
                 'Crack_RTDETR-bench-yolov8m-configurable'):
        projects.add(yaml_path.parent.parent.parent/name)
    expected='3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483'
    for project in sorted(projects):
        for model in ('yolo26m-configurable','yolov5m-coco-native-ft-v1','yolov8m-configurable','yolov8m-coco-b19-pilot'):
            parent=project/'outputs'/model
            if not parent.is_dir(): continue
            for candidate in sorted(parent.iterdir(),reverse=True)[:200]:
                for inventory in (candidate,candidate/'data'):
                    path=inventory/'manifest.json'
                    if not path.is_file() or not (inventory/'input_checksums.json').is_file(): continue
                    try:
                        meta=read_json(path)
                        if (meta.get('status')!='LIGHT_CHECKED' or Path(meta['data_root']).resolve()!=root
                            or cfg['enforce_counts'] and meta['dataset_identity_sha256']!=expected): continue
                        verify_inputs(inventory)
                        # Actual YAML path/order comparison still happens in reuse_preflight before any writes.
                        from dataset import resolve_splits
                        selected,_=resolve_splits(data_cfg,root,yaml_path.parent.parent)
                        for split in ('train','val','test'):
                            if [str(p.resolve()) for p in selected[split]]!=[str((root/r['image']).resolve())
                                for r in meta['records'] if r['split']==split]: raise ValueError('cache split/order differs')
                        print('Auto-selected verified read-only inventory: '+str(inventory),flush=True)
                        return str(inventory)
                    except (OSError,ValueError,KeyError) as exc:
                        print('Excluded stale/nonmatching inventory '+str(inventory)+': '+str(exc),flush=True)
    return None

def prepare_inputs(cfg,run):
    output=Path(run)/'data'
    output.mkdir(parents=True,exist_ok=True)
    reuse=cfg['reuse_manifest'] or matching_reuse_inventory(cfg)
    if reuse:
        manifest=reuse_preflight(reuse,cfg['data_yaml'],output,cfg['data_root'],cfg['enforce_counts'])
    else:
        manifest=preflight(cfg['data_yaml'],output,cfg['data_root'],cfg['public_coco'],cfg['enforce_counts'])
    reused_sizes={}
    if reuse:
        cache=Path(reuse)/'cache/train'
        for path in sorted(cache.glob('*.labels.json')):
            cached=read_json(path)
            for r in cached.get('entries',[]):
                if r['image'] not in reused_sizes: reused_sizes[r['image']]=r
    write_json(output/'reuse_receipt.json',{'inventory':reuse,'selection':'explicit' if cfg['reuse_manifest'] else 'bounded_verified_auto' if reuse else 'new_light_inventory','read_only':True})
    root=Path(manifest['data_root'])
    for i,r in enumerate(manifest['records'],1):
        if 'width' not in r:
            old=reused_sizes.get(r['image'])
            if old is not None and old['boxes']==r['boxes']:
                r['width'],r['height']=old['width'],old['height']
            else: r['width'],r['height']=image_size(root/r['image'])
        if i%500==0 or i==len(manifest['records']): print(f'COCO/header preparation: {i}/{len(manifest["records"])}',flush=True)
    train=[r for r in manifest['records'] if r['split']=='train']
    train_gt={'info':{'split':'train','dataset_identity_sha256':manifest['dataset_identity_sha256']},
        'images':[{'id':r['image_id'],'file_name':r['image'],'width':r['width'],'height':r['height']} for r in train],
        'annotations':annotation_rows(train),'categories':[{'id':1,'name':'crack'}]}
    write_json(output/'gt/train.json',train_gt); write_json(output/'manifest.json',manifest)
    names=list(read_json(output/'input_checksums.json'))+['gt/train.json']
    write_json(output/'input_checksums.json',{n:sha256(output/n) for n in names})
    write_json(output/'target_contract.json',{'original_yolo_class':0,'derived_coco_category':1,'native_training_label':0,
        'predicted_native_label':0,'exported_public_category':1,
        'mapping':'Read YOLO0/derived COCO1 identity, explicitly create long labels filled with 0; postprocessor native0 maps once to public1.',
        'boxes':'normalized cxcywh float32, [N,4]; empty [0,4], labels [0]',
        'shared_data_writes':False,'image_headers_only':True,'stats':manifest['splits']})
    historical={'val':'d83b3da6128c79bf2c2449ca5cebf447deb20cdd0bae06edee1a0f238659082c',
        'test':'0ad14fa4ec33d5d708a1e71b0f35215e7d1730162815f2c70350ebea48eca469'}
    write_json(output/'public_gt_identity.json',{s:{'observed_sha256':sha256(output/f'gt/{s}.json'),
        'historical_sha256':value,'same_bytes':sha256(output/f'gt/{s}.json')==value} for s,value in historical.items()})
    seals=read_json(output/'input_checksums.json')
    for name in ('target_contract.json','public_gt_identity.json','reuse_receipt.json'): seals[name]=sha256(output/name)
    write_json(output/'input_checksums.json',seals)
    return manifest

def validate_target(target,imgsz):
    boxes,labels=target['boxes'],target['labels']
    if (boxes.ndim!=2 or boxes.shape[1]!=4 or labels.shape!=(len(boxes),) or labels.dtype!=torch.int64
        or labels.numel() and (labels!=0).any()): raise ValueError('Invalid native class0 target')
    if not torch.isfinite(boxes).all() or (boxes.numel() and ((boxes<0).any() or (boxes>1).any() or (boxes[:,2:]<=0).any())):
        raise ValueError('Invalid normalized native cxcywh boxes')
    if tuple(target['size'].tolist())!=(imgsz,imgsz): raise ValueError('Target/network size differs')

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,manifest,cfg,epoch=0):
        self.records=[r for r in manifest['records'] if r['split']=='train']; self.root=Path(manifest['data_root'])
        self.cfg=cfg; self.imgsz=cfg['imgsz']; self.split='train'; self.use_segments=self.use_keypoints=False
        # Train-only indices; no image contents cached, partner transforms have bounded depth.
        self.buffer=range(len(self.records)); self.augmentation_counts=counters()
        self.closed=cfg['close_mosaic']>0 and epoch>=cfg['epochs']-cfg['close_mosaic']
        hyp=SimpleNamespace(**cfg)
        if self.closed: hyp.mosaic=hyp.mixup=0.
        chain=training_transforms(self,self.imgsz,hyp,0. if self.closed else cfg['cutmix'])
        chain.append(augment.Format(bbox_format='xywh',normalize=True,batch_idx=False,bgr=0.))
        self.transforms=chain
        self.transform_report.update(actual_transform_tree=describe_transform(chain),closed=self.closed,
            fixed_collate='torch.stack only; no scaling',worker_lifecycle='new nonpersistent workers each epoch; exhausted iterator and no old prefetch',
            native_target='YOLO0/COCO1 -> long native0; normalized cxcywh')
    def __len__(self): return len(self.records)
    def get_image_and_label(self,index):
        r=self.records[index]; image=cv2.imread(str(self.root/r['image']),cv2.IMREAD_COLOR)
        if image is None or image.shape[:2]!=(r['height'],r['width']): raise ValueError('Decode/dimensions differ: '+r['image'])
        image=cv2.resize(image,(self.imgsz,self.imgsz),interpolation=cv2.INTER_LINEAR)
        rows=np.asarray(r['boxes'],dtype=np.float32).reshape(-1,5)
        if len(rows) and np.any(rows[:,0]!=0): raise ValueError('Only YOLO class0 is allowed')
        return {'img':image,'im_file':str(self.root/r['image']),'ori_shape':(r['height'],r['width']),
            'resized_shape':(self.imgsz,self.imgsz),'cls':rows[:,:1],
            'instances':Instances(rows[:,1:].copy(),segments=np.empty((0,1000,2),np.float32),bbox_format='xywh',normalized=True)}
    def __getitem__(self,index):
        r=self.records[index]; out=self.transforms(self.get_image_and_label(index))
        if len(out['cls']) and (out['cls']!=0).any(): raise ValueError('Augmentation changed native label0')
        tensor=out['img'].float().div_(255.)
        target={'boxes':out['bboxes'].float().reshape(-1,4),'labels':torch.zeros(len(out['bboxes']),dtype=torch.int64),
            'orig_size':torch.tensor([r['width'],r['height']]),'size':torch.tensor([self.imgsz,self.imgsz]),
            'image_id':torch.tensor([r['image_id']])}
        validate_target(target,self.imgsz)
        if tensor.shape!=(3,self.imgsz,self.imgsz) or not torch.isfinite(tensor).all() or tensor.min()<0 or tensor.max()>1:
            raise ValueError('Invalid fixed RGB/[0,1] training input')
        return tensor,target

def fixed_collate(batch):
    images,targets=zip(*batch)
    return torch.stack(images),list(targets)

def worker_seed(worker_id):
    import random
    seed=torch.initial_seed()%2**32; np.random.seed(seed); random.seed(seed)

def train_loader(manifest,cfg,epoch,generator):
    dataset=TrainDataset(manifest,cfg,epoch)
    loader=torch.utils.data.DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,
        num_workers=cfg['train_workers'],drop_last=cfg['drop_last'],collate_fn=fixed_collate,
        generator=generator,worker_init_fn=worker_seed,persistent_workers=False,pin_memory=cfg['device'].startswith('cuda'))
    if len(loader)==0: raise ValueError('Train loader has no actual batches')
    return loader

def letterbox(image,imgsz=640):
    h,w=image.shape[:2]; scale=min(imgsz/h,imgsz/w)
    resized_w,resized_h=round(w*scale),round(h*scale)
    dw,dh=(imgsz-resized_w)/2,(imgsz-resized_h)/2
    left,right=round(dw-.1),round(dw+.1); top,bottom=round(dh-.1),round(dh+.1)
    if (resized_w,resized_h)!=(w,h): image=cv2.resize(image,(resized_w,resized_h),interpolation=cv2.INTER_LINEAR)
    image=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))).float()/255.
    return tensor,{'gain_x':resized_w/w,'gain_y':resized_h/h,'left':left,'top':top,
        'right':right,'bottom':bottom,'resized_wh':[resized_w,resized_h],'original_wh':[w,h],
        'auto':False,'scaleup':True}

def inverse_boxes(boxes,meta):
    boxes=boxes.clone().float()
    boxes[:,[0,2]]=(boxes[:,[0,2]]-meta['left'])/meta['gain_x']
    boxes[:,[1,3]]=(boxes[:,[1,3]]-meta['top'])/meta['gain_y']
    return boxes  # No clipping/NMS/deletion; evaluator only applies the common confidence/max_det rule.

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self,gt,root): self.images=sorted(gt['images'],key=lambda im:im['id']); self.root=Path(root)
    def __len__(self): return len(self.images)
    def __getitem__(self,index):
        im=self.images[index]; image=cv2.imread(str(self.root/im['file_name']),cv2.IMREAD_COLOR)
        if image is None or image.shape[:2]!=(im['height'],im['width']): raise ValueError('Eval image dimensions differ')
        tensor,meta=letterbox(image)
        return tensor,im,meta

def eval_collate(batch):
    tensor,images,meta=zip(*batch); return torch.stack(tensor),list(images),list(meta)
