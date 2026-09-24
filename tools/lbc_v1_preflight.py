"""Two real augmented train mini-batches; bounded isolated native optimizer updates."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
import torch
from PIL import Image, ImageDraw
from lbc_v1_training import LBCTrainer, native_copy, epoch_start, batch_end
from init_c19_lif_v1 import require, write_json
from c19_lif_v1_diagnostic import atomic_json as write_json
from ultralytics.models.rtdetr.lbc import regions
from ultralytics.utils.torch_utils import autocast


def visualize(batch, stats, destination, limit=8):
    """Detached post-augmentation GT, buffers and actual selected top-k cells."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    hf,wf=stats['feature_hw'];h,w=batch['img'].shape[-2:];sx,sy=w/wf,h/hf
    _,counts,geometry=regions(batch,(hf,wf))
    output=[]
    for i,tensor in enumerate(batch['img'][:limit]):
        canvas=Image.fromarray((tensor.detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8'))
        draw=ImageDraw.Draw(canvas)
        for row in geometry:
            if row['image']!=i:continue
            draw.rectangle(row['exclusion'].tolist(),outline='#f0bd43',width=1)
            draw.rectangle(row['box'].tolist(),outline='#00ff80',width=1)
        for row in stats.get('selected_points',[]):
            if row['image']!=i:continue
            for side,color in [('positive','#00ffff'),('negative','#ff3366')]:
                for index in row[side]:
                    x,y=(index%wf+.5)*sx,(index//wf+.5)*sy
                    draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        draw.rectangle((0,0,min(w,460),20),fill='black')
        draw.text((3,3),'GT green / exclusion yellow / top-k pos cyan, neg red',fill='white')
        path=destination/f'image_{i:02d}.png';canvas.save(path);output.append(str(path))
    return dict(counts=counts, images=output, limitation='Canvas-only regions may include padding and unlabeled cracks')


def bounded_check(args, folder, server=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    settings=deepcopy(args)
    settings.update(project=str(folder),name='isolated_trainer',save_dir=str(folder/'isolated_trainer'),
                    exist_ok=False,plots=False)
    if not server:
        settings.update(device='cpu',batch=4,imgsz=160,workers=0,amp=False)
    started=time.monotonic()
    report=dict(status='FAILED',scope='isolated diagnostic; never used as formal initialization',
                formal_training='NOT_RUN',server_capacity=server,batch=settings['batch'],imgsz=settings['imgsz'],
                max_microbatches=16,max_seconds=900,epoch_condition=20,source_train_batches=2,reused_batches=True)
    if server:
        require(torch.cuda.is_available(),'Server CUDA required')
        torch.cuda.reset_peak_memory_stats()
    trainer=None
    try:
        trainer=LBCTrainer(overrides=settings)
        trainer._setup_train()
        require(trainer.amp is bool(settings['amp']), 'Native AMP check changed the requested precision')
        trainer.epoch=20;trainer.start_epoch=20
        trainer._lbc_ni=20*len(trainer.train_loader)-1
        trainer._lbc_last_opt_step=trainer._lbc_ni
        trainer.model.train();trainer.model.lbc_capture_points=True
        epoch_start(trainer)
        trainer.scheduler.last_epoch=19;trainer.scheduler.step()
        # Native e=20 is beyond warmup, and nbs/batch determines accumulation.
        trainer.optimizer.zero_grad(set_to_none=True)
        head_before={k:v.detach().clone() for k,v in trainer.model.lbc_head.state_dict().items()}
        backbone=trainer.model.model[5].blocks[0].branch2a.conv.weight
        backbone_before=backbone.detach().clone()
        iterator=iter(trainer.train_loader)
        raw_batches=[next(iterator),next(iterator)]
        coverage=[];rows=[]
        for i in range(16):
            if time.monotonic()-started>=900:break
            with autocast(trainer.amp):
                batch=trainer.preprocess_batch(deepcopy(raw_batches[i%2]))
                loss,trainer.loss_items=trainer.model(batch)
                trainer.loss=loss.sum()  # native single-device wrapper; no extra batch factor
            require(torch.isfinite(trainer.loss).all(), 'Nonfinite preflight loss')
            trainer.scaler.scale(trainer.loss).backward()
            if i<2:
                coverage.append(visualize(batch,trainer.model.lbc_last,folder/f'train_batch_{i}',limit=8))
            row=dict(index=i,loss=float(trainer.loss.detach()),**trainer.model.lbc_last)
            row.pop('selected_points',None);rows.append(row)
            if trainer._lbc_ni-trainer._lbc_last_opt_step>=trainer.accumulate:
                trainer.optimizer_step()
            report.update(microbatches=i+1,effective_updates=trainer.lbc_effective_updates,
                          overflow_skips=trainer.lbc_overflow_skips,losses=rows)
            write_json(folder/'bounded_progress.json',report)
            if trainer.lbc_effective_updates>=2:break
        report.update(microbatches=len(rows),amp=trainer.amp,actual_shape=list(batch['img'].shape),
            losses=rows,coverage=coverage,effective_updates=trainer.lbc_effective_updates,
            overflow_skips=trainer.lbc_overflow_skips,optimizer_groups=trainer.lbc_optimizer_groups,
            head_updated=any(not torch.equal(v,trainer.model.lbc_head.state_dict()[k]) for k,v in head_before.items()),
            backbone_updated=not torch.equal(backbone_before,backbone),clip=getattr(trainer,'lbc_clip',None),
            trainer_rebuild=trainer.lbc_model_audit)
        pairs=sum(c['counts']['pairs'] for c in coverage);gt=sum(c['counts']['gt'] for c in coverage)
        report['pair_coverage']=pairs/max(gt,1)
        report['mechanism_coverage']='INSUFFICIENT' if pairs==0 or report['pair_coverage']<.1 else 'OBSERVED_ON_TWO_BATCHES'
        require(1<=trainer.lbc_effective_updates<=2 and report['head_updated'] and report['backbone_updated'],
                'Capacity check incomplete: effective update or head/backbone update missing')
        require(report['mechanism_coverage']!='INSUFFICIENT','Mechanism coverage insufficient; inspect visual evidence')
        # Check a real, updated native scaler state, including the growth tracker.
        state=trainer.scaler.state_dict()
        restored_scaler=torch.cuda.amp.GradScaler(enabled=trainer.amp)
        restored_scaler.load_state_dict(deepcopy(state))
        require(restored_scaler.state_dict()==state,'Native scaler roundtrip mismatch')
        report['scaler_roundtrip']=dict(exact=True,enabled=trainer.amp,state=state)
        require(all('lbc_head.'+k in trainer.ema.ema.state_dict() for k in head_before),'EMA missing head')
        report['ema_covers_head']=True
        # Deploy path equality is covered by CPU lifecycle. Updated EMA fusion uses an isolated FP32 copy.
        from lbc_v1_precision import check_fusion
        report['fusion']=check_fusion(native_copy(trainer.ema.ema.cpu().float()).to(batch['img'].device),
                                     batch['img'][:2].float(),folder/'fusion')
        report['status']='PASSED' if server else 'PASSED_LOCAL_SMALL_ONLY'
        return report
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        report['seconds']=time.monotonic()-started
        if server:
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        write_json(folder/'bounded.json',report)
