"""Isolated upstream process. Invoked by run.py, never import from the mother trainer."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

from support import HERE, PROJECT, environment, read_json, write_json, initialization_type, initialization_record
from assets import verify

def activate(assets):
    verified = verify(assets)
    upstream = Path(verified['upstream'])
    # Do this before ANY upstream import (including optional logger import-time login).
    for name in ('wandb','clearml','comet_ml','albumentations'):
        sys.modules[name] = None
    os.environ.update(YOLOv5_AUTOINSTALL='false', WANDB_MODE='disabled',
                      COMET_MODE='DISABLED', MPLBACKEND='Agg',
                      YOLOV5_CONFIG_DIR=str(Path(assets).resolve()/'config'),
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path = [str(upstream), str(HERE)] + [p for p in sys.path
                if p and Path(p).resolve() not in (PROJECT, PROJECT/'ultralytics-main', HERE, upstream)]
    os.chdir(upstream)
    import torch
    import torchvision
    import models.yolo
    import utils.general
    if not Path(models.yolo.__file__).resolve().is_relative_to(upstream):
        raise RuntimeError('Wrong models import')
    if not Path(utils.general.__file__).resolve().is_relative_to(upstream):
        raise RuntimeError('Wrong utils import')
    if 'ultralytics' in sys.modules:
        raise RuntimeError('Unexpected new Ultralytics API in original YOLOv5 process')
    # Exercise compiled NMS on CPU; does not allocate a GPU.
    torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([.5]), .7)
    return verified

def options(run, assets, resume=False):
    import train
    old = sys.argv
    try:
        sys.argv = ['train.py']
        opt = train.parse_opt()
    finally:
        sys.argv = old
    recipe = read_json(HERE/'recipe.json')
    for key in ('epochs','patience','batch_size','workers','device','seed','optimizer','cos_lr',
                'cache','multi_scale','rect','freeze','image_weights','quad','imgsz'):
        setattr(opt,key,recipe[key])
    scratch = initialization_type() == 'random'
    opt.weights = str(run/'native/weights/last.pt') if resume else '' if scratch else str(assets/'yolov5m.pt')
    opt.cfg = str(assets/'upstream/models/yolov5m.yaml') if scratch and not resume else ''
    opt.hyp = str(HERE/'hyp.yaml')
    opt.data = str(run/'data/data.yaml')
    opt.project, opt.name, opt.save_dir = str(run), 'native', str(run/'native')
    opt.resume, opt.noplots = resume, True
    opt.noval, opt.nosave, opt.noautoanchor, opt.exist_ok = False, False, False, False
    return opt

def check_model(model, nc):
    from models.yolo import Detect
    head = model.model[-1]
    if type(head) is not Detect or (head.nc,head.na,head.nl) != (nc,3,3):
        raise ValueError('Not the original anchor-based 3-level Detect')
    if (model.yaml['depth_multiple'],model.yaml['width_multiple']) != (.67,.75):
        raise ValueError('Not the M scale')
    return {'head':'Detect', 'nc':head.nc, 'na':head.na, 'nl':head.nl,
            'parameters_unfused':sum(p.numel() for p in model.parameters()), 'scale':'m',
            'strides':model.stride.tolist(),
            'depth_multiple':.67, 'width_multiple':.75}

def build_initial_model(verified):
    from models.yolo import Model
    from utils.general import init_seeds
    init_seeds(42, deterministic=True)
    if initialization_type() == 'random':
        # This branch never deserializes a checkpoint or imports a model state.
        model = Model(str(Path(verified['upstream'])/'models/yolov5m.yaml'), ch=3, nc=1)
        return model, {**initialization_record(verified['upstream']),
                       'loaded_tensors':0, 'total_tensors':len(model.state_dict()),
                       'construction':'official Model(YAML, ch=3, nc=1); no state import'}
    return load_crack_model(verified['weights'], training=True)

def load_crack_model(weights, training=False, expected_identity=None):
    import torch
    from models.yolo import Model
    from utils.general import intersect_dicts
    if training and initialization_type() == 'random':
        raise ValueError('Scratch initialization may not load external model state')
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    if expected_identity is not None and ckpt.get('comparison_identity') != expected_identity:
        raise ValueError('Checkpoint belongs to another initialization/run')
    if training:
        check_model(ckpt['model'],80)
        model = Model(ckpt['model'].yaml, ch=3, nc=1)
        state = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict())
        model.load_state_dict(state,strict=False)
        missing = sorted(set(model.state_dict())-set(state))
        record = {'loaded_tensors':len(state),'total_tensors':len(model.state_dict()),'missing_keys':missing}
    else:
        model = (ckpt.get('ema') or ckpt['model']).float()
        record = {'checkpoint_epoch':ckpt['epoch']+1, 'EMA':ckpt.get('ema') is not None}
    check_model(model,1)
    return model, record

def preprocess(im, size=640):
    import cv2
    from utils.augmentations import letterbox
    h,w = im.shape[:2]
    r = size/max(h,w)
    nw,nh = max(1,int(w*r)),max(1,int(h*r))
    if (nw,nh)!=(w,h):
        im = cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR if r>1 else cv2.INTER_AREA)
    im, ratio, pad = letterbox(im,size,auto=False,scaleup=False,stride=32)
    # Exact integer padding and actual x/y resize gains; no rounding of output predictions.
    transform = (nw/w*ratio[0], nh/h*ratio[1], round(pad[0]-.1), round(pad[1]-.1))
    return im,transform

def invert(boxes, transform):
    gx,gy,left,top = transform
    boxes[:,[0,2]] = (boxes[:,[0,2]]-left)/gx
    boxes[:,[1,3]] = (boxes[:,[1,3]]-top)/gy
    return boxes

def export(run, split, checkpoint):
    import cv2
    import numpy as np
    import torch
    from utils.general import init_seeds, non_max_suppression
    from utils.torch_utils import select_device
    from support import sha256
    gt = read_json(run/'evaluation'/(split+'_gt.json'))
    manifest = read_json(run/'data/manifest.json')
    config = read_json(run/'frozen.json')
    out = run/'evaluation'/(split+'_predictions.jsonl')
    if out.exists() or out.with_suffix('.partial').exists():
        raise FileExistsError('Prediction output/partial exists; inspect before retry')
    model, ckpt_record = load_crack_model(checkpoint, expected_identity=config['checkpoint_identity'])
    init_seeds(42, deterministic=True)
    device = select_device('0',batch_size=16)
    model = model.to(device).float().eval()  # deliberately unfused; no AutoShape or hidden threshold
    identity = dict(config['prediction_identity'])
    identity.update(checkpoint_sha256=sha256(checkpoint), split=split, checkpoint=str(checkpoint),
                    checkpoint_record=ckpt_record, runtime=environment(),
                    official_pretraining=config['initialization']['pretraining_source'],
                    initialization=config['initialization'])
    partial = out.with_suffix('.partial')
    with partial.open('x',encoding='utf-8') as stream, torch.inference_mode():
        stream.write(json.dumps({'type':'metadata','schema_version':1,'identity':identity})+'\n')
        images = sorted(gt['images'],key=lambda x:x['id'])
        for start in range(0,len(images),16):
            batch, transforms = [], []
            ims = images[start:start+16]
            for row in ims:
                im = cv2.imread(str(Path(manifest['root'])/row['file_name']))
                if im is None or im.shape[:2] != (row['height'],row['width']):
                    raise ValueError('Decode/dimensions differ: '+row['file_name'])
                processed, transform = preprocess(im)
                batch.append(np.ascontiguousarray(processed[:,:,::-1].transpose(2,0,1)))
                transforms.append(transform)
            tensor = torch.from_numpy(np.stack(batch)).to(device).float()/255
            raw = model(tensor,augment=False)
            detections = non_max_suppression(raw,conf_thres=.001,iou_thres=.7,
                                             agnostic=False,multi_label=False,max_det=300)
            if len(detections)!=len(ims):
                raise RuntimeError('Incomplete NMS batch')
            for row,det,transform in zip(ims,detections,transforms):
                det = det.float().cpu()
                invert(det[:,:4],transform)  # no clipping; public schema requires raw float coordinates
                predictions=[]
                for x1,y1,x2,y2,score,cls in det.tolist():
                    if cls!=0 or not np.isfinite([x1,y1,x2,y2,score]).all() or x2<=x1 or y2<=y1:
                        raise ValueError('Invalid original-pixel prediction')
                    predictions.append({'category_id':1,'bbox':[x1,y1,x2,y2],'score':score})
                record={'split':split,'image_id':row['id'],'relative_path':row['file_name'],
                        'width':row['width'],'height':row['height'],'box_format':'xyxy',
                        'coordinate_space':'original_image_pixels','predictions':predictions}
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(f'{split}: {min(start+16,len(images))}/{len(images)}',flush=True)
    partial.rename(out)
    write_json(run/'evaluation'/(split+'_export.json'),{'status':'completed','images':len(images),
               'predictions_sha256':sha256(out),'checkpoint_sha256':identity['checkpoint_sha256']})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','train','export','resources','smoke'])
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'])
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    a.run,a.assets=a.run.resolve(),a.assets.resolve()
    if a.action in ('check','smoke','resources'):
        os.environ['CUDA_VISIBLE_DEVICES']=''
    verified=activate(a.assets)
    import torch
    torch.set_num_threads(4)
    if a.action=='check':
        import train
        from models.yolo import Model
        from utils.general import init_seeds
        init_seeds(42,deterministic=True)
        m,record=build_initial_model(verified)
        write_json(a.run/'model_check.json',{**check_model(m,1),**record,'environment':environment(),
                   'train_module':train.__file__, 'upstream':verified['upstream']})
        write_json(a.run/'resolved_options.json',{k:str(v) if isinstance(v,Path) else v
                                                for k,v in vars(options(a.run,a.assets)).items()})
    elif a.action=='train':
        import train
        from utils.callbacks import Callbacks
        from utils.torch_utils import select_device
        opt=options(a.run,a.assets,a.resume)
        if not a.resume and (a.run/'native').exists():
            raise FileExistsError('Native training directory already exists')
        if a.resume:
            checkpoint=torch.load(opt.weights,map_location='cpu',weights_only=False)
            from bench_yolov5_runtime import validate_checkpoint
            validate_checkpoint(checkpoint, a.run/'native', resume=True)
            stored=checkpoint['opt']
            for key in ('epochs','batch_size','imgsz','optimizer','seed','patience','cos_lr','data','save_dir'):
                if stored[key]!=getattr(opt,key):
                    raise ValueError('Resume option differs: '+key)
        train.train(opt.hyp,opt,select_device(opt.device,batch_size=opt.batch_size),Callbacks())
        if not (a.run/'native/training_complete.json').is_file():
            raise RuntimeError('No training completion record')
    elif a.action=='export':
        if a.split is None:
            p.error('--split is required')
        export(a.run,a.split,a.run/'native/weights/best.pt')
    elif a.action=='resources':
        from copy import deepcopy
        import thop
        m,_=load_crack_model(a.run/'native/weights/best.pt',
                            expected_identity=read_json(a.run/'frozen.json')['checkpoint_identity'])
        results={}
        for fused in (False,True):
            model=deepcopy(m).eval()
            if fused:
                model.fuse()
            with torch.inference_mode():
                macs,_=thop.profile(model,inputs=(torch.zeros(1,3,640,640),),verbose=False)
            results['fused' if fused else 'unfused']={'parameters':sum(x.numel() for x in model.parameters()),
                   'GMACs':macs/1e9,'GFLOPs_2_per_MAC':2*macs/1e9}
        write_json(a.run/'resources.json',{'nc':1,'input':[1,3,640,640],'counter':'thop; FLOPs=2*MACs',
                   'latency':'NOT_MEASURED',**results})
    else:
        from validation import smoke
        smoke(a.assets,a.run)

if __name__=='__main__':
    main()
