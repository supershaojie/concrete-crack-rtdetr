"""Original trainer, explicit GIC configuration, real epoch and bounded CUDA smoke."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

from gic_v1_common import ROOT, OUT, RUN, INIT, MODEL, require, now, sha256, read_json, write_json
import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.gic_loss import CONFIG, eta_at_epoch
from ultralytics.utils.patches import torch_load


def amp_assets(main):
    """Mother's native AMP probe uses existing local assets; never download weights."""
    from ultralytics.utils import ASSETS
    records=[]
    for target,candidates in ((ROOT/'yolo26n.pt',[main/'yolo26n.pt',main/'weights/yolo26n.pt']),
                              (ASSETS/'bus.jpg',[main/'ultralytics-main/ultralytics/assets/bus.jpg',main/'bus.jpg'])):
        if not target.is_file():
            source=next((p for p in candidates if p.is_file()),None)
            require(source is not None,f'Missing native AMP check asset: {target}; supply the existing mother asset')
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as a,target.open('xb') as b: shutil.copyfileobj(a,b)
        records.append(dict(path=str(target),sha256=sha256(target)))
    write_json(OUT/'amp_assets.json',records)


def configure_model(model, epoch=0):
    model.gic_config=deepcopy(CONFIG)
    model.gic_epoch=int(epoch)
    model.gic_collect_diagnostics=False
    if hasattr(model,'criterion'): del model.criterion
    return model


class GICTrainer(RTDETRTrainer):
    """Native optimizer, warmup, EMA, scheduler, DN and best-fitness selection."""
    def __init__(self,*args,gic_binding=None,**kwargs):
        self.gic_binding=gic_binding
        super().__init__(*args,**kwargs)
        self.add_callback('on_train_batch_start',self._before_batch)
        self.add_callback('on_train_batch_end',self._after_batch)
        self.add_callback('on_train_epoch_end',self._record_epoch)

    @staticmethod
    def _before_batch(trainer):
        trainer._oom_retries=3  # Mother's documented opt-out; never halve B16 after OOM.
        trainer.model.gic_epoch=int(trainer.epoch)
        trainer.model.gic_collect_diagnostics=trainer.gic_batch_index%100==0

    @staticmethod
    def _after_batch(trainer):
        require(bool(torch.isfinite(trainer.loss)),'Nonfinite training loss; preserving failure without recovery/replay')
        if trainer.model.gic_collect_diagnostics:
            row=dict(epoch=trainer.epoch,batch=trainer.gic_batch_index,
                     diagnostic=trainer.model.criterion.diagnostics,time=now())
            with (OUT/'gic_diagnostics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
        trainer.gic_batch_index+=1

    @staticmethod
    def _record_epoch(trainer):
        write_json(OUT/'progress.json',dict(epoch_completed=trainer.epoch+1,eta_eff=eta_at_epoch(trainer.epoch),
                   loss_items=trainer.tloss.detach().cpu().tolist(),time=now()))

    def get_model(self,cfg=None,weights=None,verbose=True):
        from init_c19_lif_v1 import build_training_model,verify_model
        require(weights is not None,'Verified public initialization or same-run resume required')
        if not self.resume:
            model,audit=build_training_model(cfg,weights,self.data)
        else:
            require(getattr(weights,'gic_identity',None)==self.gic_binding,'Resume checkpoint identity differs')
            require(getattr(weights,'gic_config',None)==CONFIG,'Resume loss definition differs')
            model=super().get_model(cfg,weights,verbose)
            verify_model(model)
            require(set(model.state_dict())==set(weights.state_dict()),'Resume state inventory changed')
            require(all(torch.equal(v,weights.state_dict()[k].float()) for k,v in model.state_dict().items()),'Resume weights changed')
            audit=dict(resume=True,loaded_exact=len(model.state_dict()))
        configure_model(model)
        model.gic_identity=self.gic_binding
        write_json(OUT/'trainer_loading.json',audit)
        return model

    def _model_train(self):
        super()._model_train()
        epoch=int(getattr(self,'epoch',self.start_epoch))
        self.gic_batch_index=0
        self.model.gic_epoch=epoch
        if getattr(self,'ema',None):
            self.ema.ema.gic_epoch=epoch
            self.ema.ema.gic_collect_diagnostics=False

    def _setup_train(self):
        super()._setup_train()
        require(self.args.amp and self.amp,'Native AMP probe disabled AMP; recipe must not change')
        require(type(self.optimizer) is torch.optim.AdamW,'Optimizer changed')
        expected=read_json(OUT/'prepare.json')['args']
        changes={k:[v,getattr(self.args,k,None)] for k,v in expected.items()
                 if type(getattr(self.args,k,None)) is not type(v) or getattr(self.args,k,None)!=v}
        require(set(changes)<={'model','resume'},f'Effective trainer recipe differs: {changes}')
        write_json(OUT/'training_setup.json',dict(args=vars(self.args),allowed_changes=changes,
                   start_epoch=self.start_epoch,binding=self.gic_binding,amp=bool(self.amp),
                   optimizer=type(self.optimizer).__name__,parameters=sum(p.numel() for p in self.model.parameters())))

    def _load_checkpoint_state(self,ckpt):
        require(ckpt.get('optimizer') is not None and ckpt.get('scaler') is not None and ckpt.get('ema') is not None,
                'Cannot resume a stripped or incomplete checkpoint')
        require(getattr(ckpt['ema'],'gic_identity',None)==self.gic_binding,'Checkpoint binding differs')
        require(getattr(ckpt['ema'],'gic_config',None)==CONFIG,'Checkpoint loss config differs')
        super()._load_checkpoint_state(ckpt)

    def _handle_nan_recovery(self,epoch):
        require(bool(torch.isfinite(self.tloss).all()) and all(torch.isfinite(p).all() for p in self.model.parameters()),
                'Nonfinite loss/parameters; no automatic training replay')
        return False

    def setup_model(self):
        # Only needed for a Windows checkout whose username contains an apostrophe.
        if isinstance(self.model,(str,Path)) and "'" in str(self.model):
            self.model=str(Path(self.model).resolve().relative_to(ROOT))
        return super().setup_model()

    def save_model(self):
        self.ema.ema.gic_config=deepcopy(CONFIG)
        self.ema.ema.gic_epoch=int(self.epoch)
        self.ema.ema.gic_identity=self.gic_binding
        super().save_model()
        write_json(OUT/'checkpoint_state.json',dict(epoch=self.epoch,best_fitness=self.best_fitness,
                   fitness=self.fitness,binding=self.gic_binding,best=str(self.best),last=str(self.last)))

    def final_eval(self):
        # Final independent FP32 inference/export is explicitly deferred to finish.
        # Native training-epoch val, early stopping and best selection are unchanged.
        from ultralytics.utils.torch_utils import strip_optimizer
        complete=self.epoch+1>=self.epochs
        early=self.epoch+1-self.stopper.best_epoch>=self.stopper.patience
        require(complete or early,'Training interrupted before budget/early stopping; resume required')
        require(self.best.is_file(),'Missing selected best; last must not substitute for best')
        selected=torch_load(self.best,map_location='cpu')
        record=dict(status='COMPLETE',binding=self.gic_binding,completed_epoch=self.epoch+1,
                    best_epoch=selected['epoch'],completion='budget_completed' if complete else 'early_stopped',
                    selection='native training val fitness/early stopping',time=now())
        last=strip_optimizer(self.last)
        strip_optimizer(self.best,updates={'train_results':last.get('train_results')})
        record['weights']={name:dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size)
                           for name,p in [('best',self.best),('last',self.last)]}
        write_json(OUT/'training_completed.json',record)
        print('Training completed; independent FP32 val/test/export pending tools/gic_v1.sh finish',flush=True)


def real_smoke(main,data_root,folder,modes=(False,True)):
    """One fixed real B16/640 batch; one FP32 and one AMP step in separate fresh models.

    This is a wiring/capacity test, not an augmented training or benefit estimate.
    It deliberately does not enter Trainer.train() or a validation loop.
    """
    from init_c19_lif_v1 import controlled_models,build_training_model
    from c19_lif_v1_data import real_batch
    from ultralytics.utils.torch_utils import init_seeds
    import numpy as np
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=Path(main)/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt'
    report=dict(status='RUNNING',steps=[],batch=16,imgsz=640,max_steps=2,
                python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                amp_numeric_scale=1.0,
                scope='fixed 16 real train images, original stretch RGB /255, no augmentation; unit-scale numerical gradients; formal native scaler unchanged')
    if not torch.cuda.is_available() or not source.is_file() or not (Path(data_root)/'images/train').is_dir():
        report.update(status='NOT_RUN',reason='CUDA/public initialization/real train data unavailable')
        write_json(folder/'smoke.json',report);return report
    try:
        torch.set_num_threads(4)
        batch,records=real_batch(Path(data_root),size=640,count=16)
        report['images']=records;report['gpu']=torch.cuda.get_device_name(0)
        require(len(modes)<=2 and all(type(m) is bool for m in modes),'At most two temporary smoke steps')
        for amp in modes:
            report['attempting']='AMP' if amp else 'FP32'
            write_json(folder/'smoke.json',report)
            init_seeds(42,deterministic=True)
            _,weights,initialization=controlled_models(source)
            model,loading=build_training_model(str(MODEL),weights,dict(nc=1,channels=3))
            del weights
            model.nc=1;configure_model(model,19);model.gic_collect_diagnostics=True
            model.train().cuda(0)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
            # A unit scale isolates AMP numerical wiring from expected initial
            # dynamic-scaler overflows. Formal GICTrainer uses the native scaler.
            scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=1.0)
            gpu={k:v.cuda(0) for k,v in batch.items()}
            torch.cuda.reset_peak_memory_stats(0)
            with torch.autocast(device_type='cuda',enabled=amp): loss,items=model(gpu)
            require(bool(torch.isfinite(loss)),'Nonfinite smoke loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite smoke gradients')
            diagnostic=model.criterion.diagnostics
            require(diagnostic and diagnostic['eta_eff']==.5,'GIC not wired at real epoch 19')
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            probe=model.model[-1].dec_score_head[-1].weight
            before=probe.detach().clone()
            scaler.step(optimizer);scaler.update()
            require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite smoke updated parameters')
            require(bool((probe.detach()!=before).any()),'No effective classification parameter update')
            report['steps'].append(dict(mode='AMP' if amp else 'FP32',loss=float(loss.detach()),
                   diagnostic=diagnostic,backward=True,parameter_update=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(0)))
            write_json(folder/'initialization.json',initialization);write_json(folder/'nc1_loading.json',loading)
            del model,optimizer,scaler,loss,items,gpu,probe,before
            torch.cuda.empty_cache()
        report['status']='PASS'
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='OOM',reason=str(error),remaining_modes='NOT_RUN; no automatic recipe change or retry')
    except BaseException as error:
        report.update(status='FAIL',reason=repr(error));raise
    finally:
        write_json(folder/'smoke.json',report)
    return report
