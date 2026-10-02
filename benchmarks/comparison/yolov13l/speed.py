"""Optional later network-only timing on user-arranged exclusive hardware; never a training gate."""
import argparse
from pathlib import Path
import statistics
from support import configure, read_json, sha256, validate_checkpoint, run_identity, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--warmup',type=int,default=10)
    args=p.parse_args()
    if args.output.exists() or min(args.batch,args.iterations,args.warmup)<1:
        raise ValueError('Output already exists or invalid sample counts')
    run=args.run.resolve();source=configure(run/'speed_runtime')
    from data import verify_inputs
    manifest=verify_inputs(run,('val',))
    identity=read_json(run/'identity.json')
    if run_identity(manifest,source,run_id=read_json(run/'run_id.json')['run_id'])!=identity:
        raise ValueError('Run identity changed')
    status=read_json(run/'train_status.json');checkpoint=run/'train/weights/best.pt'
    if status['status']!='completed' or sha256(checkpoint)!=status['best_sha256']:
        raise ValueError('Expected unchanged completed best')
    import torch
    from ultralytics import YOLO
    from adapters import model_identity
    from backend import native_fp32,ArithmeticAudit
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);validate_checkpoint(ckpt,identity)
    del ckpt
    model=YOLO(str(checkpoint),task='detect').model
    model_identity(model,1)
    model=model.float().cuda().eval().fuse(verbose=False)
    x=torch.zeros(args.batch,3,640,640,device='cuda')
    times=[]
    with torch.no_grad(),native_fp32():
        with ArithmeticAudit(require_fp32=True) as audit: model(x)
        for _ in range(args.warmup): model(x)
        torch.cuda.synchronize()
        for _ in range(args.iterations):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();model(x);end.record();end.synchronize();times.append(start.elapsed_time(end))
    write_json(args.output,{'scope':'network_only_no_preprocess_NMS_IO','precision':'FP32','fused':True,'imgsz':640,
        'batch':args.batch,'batch_latency_ms_median':statistics.median(times),'samples_ms':times,
        'FPS_from_median':1000*args.batch/statistics.median(times),'checkpoint_sha256':sha256(checkpoint),
        'gpu':torch.cuda.get_device_name(0),'source':source,'dtype_audit':audit.report(),
        'hardware_exclusivity':'Not enforced or proved by this command; use paper timing only when independently exclusive'})


if __name__=='__main__': main()
