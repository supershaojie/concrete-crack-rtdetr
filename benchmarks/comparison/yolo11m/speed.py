"""Optional later timing on an otherwise idle GPU; never part of training admission."""
from __future__ import annotations
import argparse
from pathlib import Path
import statistics
from support import configure, read_json, sha256, write_json, validate_checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--iterations',type=int,default=100)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if min(a.warmup,a.iterations)<1: raise ValueError('Positive iteration counts required')
    configure(a.run/'speed_runtime')
    import torch
    from ultralytics import YOLO
    from adapters import model_identity
    ckpt=a.run/'train/weights/best.pt'
    state=read_json(a.run/'train_status.json')
    if state['status']!='completed' or sha256(ckpt)!=state['best_sha256']:
        raise ValueError('Completed selected best required')
    raw=torch.load(ckpt,map_location='cpu',weights_only=False)
    validate_checkpoint(raw,read_json(a.run/'identity.json'))
    model=YOLO(str(ckpt),task='detect').model.float()
    architecture=model_identity(model,1)
    model=model.fuse(verbose=False).cuda(0).eval()
    x=torch.zeros(1,3,640,640,device='cuda:0')
    times=[]
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for _ in range(a.warmup): model(x)
        torch.cuda.synchronize()
        for _ in range(a.iterations):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();model(x);end.record();end.synchronize()
            times.append(start.elapsed_time(end))
    write_json(a.output,{'scope':'model forward only; excludes image IO/preprocessing/NMS',
        'hardware_exclusivity':'must be established separately by operator; no GPU-idle admission lock',
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA':torch.version.cuda,
        'batch':1,'imgsz':640,'precision':'FP32','fused':True,'warmup':a.warmup,'iterations':a.iterations,
        'milliseconds_mean':statistics.mean(times),'milliseconds_median':statistics.median(times),
        'samples_ms':times,'architecture':architecture,'checkpoint_sha256':sha256(ckpt)})
if __name__=='__main__':main()
