"""Subprocess-only CUDA operators and actual configured training capacity."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import traceback
from support import environment, load_yaml, write_json

def probe(run,cfg,output,local_compat=False):
    import torch
    import torchvision
    from data_adapter import TrainDataset,train_loader
    from model import build_model,finite_gradients,finite_losses,seed_all
    result={'status':'running','python':os.path.abspath(sys.executable),'isolation':'dedicated process; formal model/optimizer/BN/RNG rebuilt after exit',
        'requested_batch':cfg['batch_size'],'requested_size':cfg['imgsz'],'device':cfg['device'],'amp':cfg['amp'],
        'formal_capacity_claim':not local_compat}
    try:
        env=environment(strict=not local_compat); result['environment']=env
        seed_all(cfg['seed'],cfg['deterministic'])
        device=torch.device(cfg['device'])
        if device.type=='cuda':
            if not torch.cuda.is_available(): raise RuntimeError('Requested CUDA unavailable')
            result['memory_before']={'free_total':list(torch.cuda.mem_get_info(device)),'allocated':torch.cuda.memory_allocated(device)}
        boxes=torch.tensor([[0.,0.,10.,10.],[.5,.5,9.5,9.5]],device=device)
        scores=torch.tensor([.9,.8],device=device)
        keep=torchvision.ops.nms(boxes,scores,.7)
        feature=torch.randn(1,4,16,16,device=device,requires_grad=True)
        roi=torchvision.ops.roi_align(feature,[boxes[:1]],(7,7)); roi.sum().backward()
        if keep.tolist()!=[0] or feature.grad is None: raise ValueError('NMS/RoIAlign operator probe failed')
        result['native_ops']={'nms_device':str(keep.device),'roi_align_shape':list(roi.shape),'roi_align_backward':True,
            'extension':env['torchvision_extension_path']}
        manifest=__import__('support').read_json(Path(run)/'manifest.json')
        ds=TrainDataset(run,manifest,cfg); loader=train_loader(ds,cfg,torch.Generator().manual_seed(cfg['seed']))
        iterator=iter(loader)
        try: images,targets,traces=next(iterator)
        finally:
            if hasattr(iterator,'_shutdown_workers'): iterator._shutdown_workers()
            del iterator
        if len(images)!=cfg['batch_size']: raise ValueError('Capacity batch is smaller than requested; no batch1 substitution allowed')
        model,init=build_model(cfg); model.to(device).train()
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=cfg['lr0'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'],nesterov=cfg['nesterov'])
        scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'],init_scale=cfg['amp_init_scale'],growth_interval=cfg['amp_growth_interval'])
        optimizer.zero_grad(set_to_none=True)
        images=[im.to(device) for im in images]; targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=cfg['amp']): loss=finite_losses(model(images,targets))
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); gradients=finite_gradients(model)
        scaler.step(optimizer); scaler.update()
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            result['memory_after']={'free_total':list(torch.cuda.mem_get_info(device)),
                'max_allocated':torch.cuda.max_memory_allocated(device),'max_reserved':torch.cuda.max_memory_reserved(device)}
        from augment_b19 import snapshot
        with torch.no_grad():
            model.eval(); transformed,_=model.transform([images[0]],None)
            maps=model.backbone(transformed.tensors)
        layers={'actual_input':list(transformed.tensors.shape), 'outputs':{k:list(v.shape) for k,v in maps.items()},
            'named_modules':[n for n,_ in model.named_modules() if n.startswith(('backbone.body.layer','backbone.fpn'))]}
        write_json(Path(run)/'visualization_layers.json',layers)
        result.update(status='PASSED_LOCAL_COMPATIBILITY_SMOKE' if local_compat else 'PASSED_ACTUAL_CONFIGURED_CAPACITY',
            actual_batch=len(images),actual_input=model.backbone.comparison_observed_size,loss=float(loss.detach()),
            gradient_tensors=gradients,scaler_enabled=scaler.is_enabled(),initialization=init,
            transform_tree=ds.transform_report,augmentation_counts=snapshot(ds),finite_sample_traces=traces,
            exit_code=0,capacity_limit='one actual step at current memory state; does not promise concurrent full-run memory')
        write_json(output,result); return result
    except BaseException as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc(),exit_code=1)
        write_json(output,result); raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--local-compat',action='store_true')
    args=parser.parse_args(); probe(args.run,load_yaml(args.config),args.output,args.local_compat)
if __name__=='__main__': main()
