"""Explicit, isolated capacity probe; synthetic tensors only, never a formal run."""
import argparse
from pathlib import Path
from support import configure, ROOT, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1)
    args=p.parse_args()
    if args.output.exists() or args.batch<1: raise ValueError('Output exists or invalid batch')
    configure(args.output.parent/'capacity_runtime')
    import torch
    from ultralytics.cfg import get_cfg
    from support import recipe
    from run import load_initial_model
    torch.set_num_threads(2)
    model,record=load_initial_model()
    model=model.cuda().train();model.args=get_cfg(overrides=recipe())
    torch.cuda.reset_peak_memory_stats()
    batch={'img':torch.rand(args.batch,3,640,640,device='cuda'),
           'batch_idx':torch.arange(args.batch,device='cuda'),
           'cls':torch.zeros(args.batch,1,device='cuda'),
           'bboxes':torch.tensor([[.5,.5,.3,.3]],device='cuda').repeat(args.batch,1)}
    with torch.autocast('cuda',dtype=torch.float16):
        loss,items=model(batch)
    loss.sum().backward()
    grads=[v.grad for v in model.parameters() if v.grad is not None]
    if not torch.isfinite(loss).all() or not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Nonfinite actual-input forward/backward')
    write_json(args.output,{'status':'FINITE_SYNTHETIC_FORWARD_BACKWARD','batch':args.batch,'imgsz':640,'amp':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gradient_tensors':len(grads),
        'parameters_unfused':record['parameters_unfused'],'loss_items_smoke_only':items.tolist(),
        'optimizer_steps':0,'formal_training':False,'parallel_batch16_capacity':'NOT_VERIFIED'})


if __name__=='__main__': main()
