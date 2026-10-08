"""Standalone image/box augmentation and TorchVision image/target lists."""
from __future__ import annotations
from collections import deque
from copy import deepcopy
import os
from pathlib import Path
import random
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import augmentation_reference as augment
from instances import Instances
from data import image_size, read_labels
from support import canonical, digest, local_lock, read_json, write_json

class MotherHSV(augment.RandomHSV):
    def __call__(self,labels):
        img=labels['img']
        if img.shape[-1]!=3: raise ValueError('Three-channel BGR required for HSV')
        if self.hgain or self.sgain or self.vgain:
            r=np.random.uniform(-1,1,3)*[self.hgain,self.sgain,self.vgain]
            x=np.arange(256,dtype=r.dtype)
            hue=((x+r[0]*180)%180).astype(img.dtype)
            sat=np.clip(x*(r[1]+1),0,255).astype(img.dtype)
            val=np.clip(x*(r[2]+1),0,255).astype(img.dtype); sat[0]=0
            h,s,v=cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2HSV))
            hsv=cv2.merge((cv2.LUT(h,hue),cv2.LUT(s,sat),cv2.LUT(v,val)))
            cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR,dst=img)
        labels['hsv_executed']=True
        return labels

class NoAlbumentations:
    def __init__(self,*args,**kwargs): self.transform=None
    def __call__(self,labels): return labels

def decode(path):
    image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Image decode failed: '+str(path))
    return image

class TrainDataset(Dataset):
    def __init__(self,run,manifest,cfg):
        from augment_b19 import counters
        self.run=Path(run); self.manifest=manifest; self.cfg=deepcopy(cfg)
        self.records=[r for r in manifest['records'] if r['split']=='train']
        self.shared_root=Path(manifest['data_root']); self.imgsz=cfg['imgsz']; self.split='train'
        self.use_segments=False; self.use_keypoints=False; self.augmentation_counts=counters()
        self.buffer=deque(maxlen=min(len(self.records),cfg['batch_size']*8,1000))
        self.entries=self.label_size_cache(); self.set_epoch(0)

    def label_size_cache(self):
        identity=digest(canonical({'records':self.records,'format':'frcnn_train_header_labels_v2'}))
        file=self.run/'cache'/('train-'+identity+'.json')
        with local_lock(file.with_suffix('.lock')):
            if file.exists():
                cached=read_json(file)
                if cached['identity']!=identity or len(cached['entries'])!=len(self.records): raise ValueError('Wrong/incomplete train header cache')
                return cached['entries']
            entries=[]
            for r in self.records:
                boxes,label_sha=read_labels(self.shared_root/r['label'])
                if boxes!=r['boxes'] or label_sha!=r['label_sha256']: raise ValueError('Train label changed: '+r['label'])
                w,h=(r['width'],r['height']) if 'width' in r else image_size(self.shared_root/r['image'])
                entries.append({'image':r['image'],'width':w,'height':h,'boxes':boxes})
                if len(entries)%500==0 or len(entries)==len(self.records): print(f'Train header cache {len(entries)}/{len(self.records)}',flush=True)
            write_json(file,{'identity':identity,'entries':entries})
        return entries

    def __len__(self): return len(self.records)

    def get_image_and_label(self,index):
        r=self.records[index]; e=self.entries[index]
        image=decode(self.shared_root/r['image'])
        if image.shape[:2]!=(e['height'],e['width']): raise ValueError('Decoded dimensions changed: '+r['image'])
        image=cv2.resize(image,(self.imgsz,self.imgsz),interpolation=cv2.INTER_LINEAR)
        if index not in self.buffer: self.buffer.append(index)
        boxes=np.asarray(e['boxes'],dtype=np.float32).reshape(-1,5)
        return {'img':image,'im_file':str(self.shared_root/r['image']), 'ori_shape':(e['height'],e['width']),
            'resized_shape':(self.imgsz,self.imgsz),'cls':boxes[:,:1],
            'instances':Instances(boxes[:,1:],segments=np.zeros((0,1000,2),dtype=np.float32),bbox_format='xywh',normalized=True)}

    def set_epoch(self,epoch):
        from augment_b19 import training_transforms, describe_transform
        self.epoch=epoch
        hyp=SimpleNamespace(**self.cfg)
        probability=self.cfg['cutmix']
        if self.cfg['close_mosaic'] and epoch>=self.cfg['epochs']-self.cfg['close_mosaic']:
            hyp.mosaic=hyp.mixup=0.; probability=0.
        self.transforms=training_transforms(self,self.imgsz,hyp,probability)
        self.transforms.append(augment.Format(bbox_format='xyxy',normalize=False,batch_idx=False,bgr=0.0))
        self.active_probabilities={'mosaic':hyp.mosaic,'mixup':hyp.mixup,'cutmix':probability}
        self.transform_report['actual_transform_tree']=describe_transform(self.transforms)

    def __getitem__(self,index):
        from augment_b19 import snapshot
        before=snapshot(self)
        sample=self.transforms(self.get_image_and_label(index))
        after=snapshot(self)
        image=sample['img'].float().div_(255)
        boxes=sample['bboxes'].to(dtype=torch.float32).reshape(-1,4); classes=sample['cls'].reshape(-1)
        if image.shape!=(3,self.imgsz,self.imgsz) or (classes!=0).any() or len(classes)!=len(boxes): raise ValueError('Invalid augmented image/classes')
        if not torch.isfinite(boxes).all() or ((boxes[:,2:]-boxes[:,:2])<=0).any(): raise ValueError('Invalid augmented boxes')
        target={'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64),
            'image_id':torch.tensor([self.records[index]['image_id']],dtype=torch.int64),
            'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),'iscrowd':torch.zeros(len(boxes),dtype=torch.int64)}
        # Shared counts are authoritative. Per-sample deltas may include concurrent workers.
        meta={'epoch':self.epoch,'worker_pid':os.getpid(),'hsv':bool(sample.get('hsv_executed')),
            'empty_target':not len(boxes), **{k:after[k]['applied']-before[k]['applied'] for k in ('mosaic','mixup','cutmix')}}
        return image,target,meta

def letterbox(image,size=640):
    h,w=image.shape[:2]; ratio=min(size/h,size/w)
    rw,rh=round(w*ratio),round(h*ratio)
    dw,dh=(size-rw)/2,(size-rh)/2
    left,right,top,bottom=round(dw-.1),round(dw+.1),round(dh-.1),round(dh+.1)
    result=cv2.copyMakeBorder(cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR),top,bottom,left,right,
        cv2.BORDER_CONSTANT,value=(114,114,114))
    if result.shape!=(size,size,3): raise ValueError('Letterbox shape differs')
    return result,{'original_width':w,'original_height':h,'resized_width':rw,'resized_height':rh,
        'gain_x':rw/w,'gain_y':rh/h,'left':left,'top':top,'right':right,'bottom':bottom,
        'auto':False,'scaleup':True,'output_size':size}

def inverse_boxes(boxes,geometry):
    boxes=boxes.detach().cpu().clone().float()
    boxes[:,[0,2]]=(boxes[:,[0,2]]-geometry['left'])/geometry['gain_x']
    boxes[:,[1,3]]=(boxes[:,[1,3]]-geometry['top'])/geometry['gain_y']
    return boxes

class EvalDataset(Dataset):
    def __init__(self,root,gt,size):
        self.root=Path(root); self.images=sorted(gt['images'],key=lambda im:im['id']); self.size=size
    def __len__(self): return len(self.images)
    def __getitem__(self,index):
        im=self.images[index]; image=decode(self.root/im['file_name'])
        if image.shape[:2]!=(im['height'],im['width']): raise ValueError('Evaluation image dimensions changed')
        image,geometry=letterbox(image,self.size)
        tensor=torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))).float()/255
        return tensor,im,geometry

def collate(batch): return tuple(map(list,zip(*batch)))

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32; random.seed(seed); np.random.seed(seed)

def train_loader(dataset,cfg,generator):
    return torch.utils.data.DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,
        num_workers=cfg['train_workers'],persistent_workers=False,pin_memory=cfg['device'].startswith('cuda'),
        drop_last=False,collate_fn=collate,worker_init_fn=seed_worker,generator=generator)

def augmentation_record(cfg):
    from support import HERE,read_json
    return {'source':read_json(HERE/'augmentation.lock.json'),
        'parameters':{k:cfg[k] for k in ('hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr','mosaic','mixup','cutmix','copy_paste','bgr')},
        'close_zero_based_epoch':cfg['epochs']-cfg['close_mosaic'],
        'worker_strategy':'nonpersistent workers; consume entire epoch, discard iterator; new transform copied next epoch',
        'image_cache':'no disk image cache; bounded train index buffer only',
        'normalization':'RGB CHW float32 /255; official model image_mean/image_std exactly once'}
