"""Experiment-scoped native RTDETR training with complete epoch-boundary state."""
from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
import os

import torch
from init_lcd_v1 import ROOT, RESEARCH, require, verify_model, tensor_digest, NEW_KEYS
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.patches import torch_load
from c19_lif_v1_diagnostic import rng_state, restore_rng, atomic_json


def atomic_checkpoint(path, value):
    path=Path(path); temp=path.with_suffix('.lcd.tmp')
    torch.save(value,temp); os.replace(temp,path)


class LCDTrainer(RTDETRTrainer):
    """Loss, optimizer groups, warmup, AMP, global clip and EMA are inherited."""

    def __init__(self,*args,**kwargs):
        self._lcd_last_opt_step=-1
        self.lcd_effective_updates=0; self.lcd_overflows=0; self.lcd_consecutive_overflows=0
        self.lcd_pending=None; self.lcd_epoch_batches=0
        super().__init__(*args,**kwargs)
        self.add_callback('on_train_epoch_start',self._lcd_epoch_start)
        self.add_callback('on_train_batch_start',self._lcd_batch_start)
        self.add_callback('on_train_batch_end',self._lcd_batch_end)
        self.add_callback('on_train_epoch_end',self._lcd_epoch_end)
        self.add_callback('on_pretrain_routine_end',self._lcd_restore_scheduler)

    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose)
        verify_model(model,zero=not bool(self.resume))
        if weights is not None:
            require(tensor_digest(model)==tensor_digest(weights),'Native LCD Trainer rebuild lost state')
        return model

    @staticmethod
    def _lcd_restore_scheduler(t):
        if t.lcd_pending:
            t.scheduler.load_state_dict(t.lcd_pending['scheduler'])

    @staticmethod
    def _lcd_epoch_start(t):
        t.lcd_epoch_batches=0; t.lcd_epoch_updates=t.lcd_effective_updates
        if t.lcd_pending:
            state=t.lcd_pending
            for n,p in t.model.named_parameters():
                p.grad=state['pending_scaled_gradients'][n].to(p.device) if n in state['pending_scaled_gradients'] else None
            restore_rng(state['rng']);t.lcd_pending=None

    @staticmethod
    def _lcd_batch_start(t):
        t._oom_retries=3  # native first-epoch OOM retry must not reduce formal B16
        t.model.model[5].lcd.collect_diagnostics=t.lcd_epoch_batches<2

    @staticmethod
    def _lcd_batch_end(t):
        require(torch.isfinite(t.loss),'Nonfinite loss; stopped without changing recipe')
        t.lcd_epoch_batches+=1

    @staticmethod
    def _lcd_epoch_end(t):
        require(t.lcd_effective_updates>t.lcd_epoch_updates,'No effective update in this epoch')
        t.model.model[5].lcd.collect_diagnostics=False
        from init_lcd_v1 import write_json
        row=dict(epoch=t.epoch,effective_updates=t.lcd_effective_updates,scaled_overflows=t.lcd_overflows,
                 **t.model.model[5].lcd.diagnostics,gradients=getattr(t,'lcd_gradients',{}))
        folder=ROOT/'outputs/lcd_v1/diagnostics';folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/f'epoch_{t.epoch:03d}.json',row)

    def optimizer_step(self):
        old=max([int(v['step'].item()) for v in self.optimizer.state.values() if 'step' in v] or [0])
        scale=self.scaler.get_scale()
        self.lcd_gradients={n: (float(p.grad.detach().float().norm()/scale) if p.grad is not None else None)
                            for n,p in self.model.named_parameters() if n in NEW_KEYS}
        self.lcd_gradients={n:(v if v is None or math.isfinite(v) else 'scaled_overflow') for n,v in self.lcd_gradients.items()}
        super().optimizer_step()  # original unscale -> clip(all,10) -> scaler.step/update -> EMA
        new=max([int(v['step'].item()) for v in self.optimizer.state.values() if 'step' in v] or [0])
        if new>old:
            self.lcd_effective_updates+=1;self.lcd_consecutive_overflows=0
        else:
            self.lcd_overflows+=1;self.lcd_consecutive_overflows+=1
            require(self.lcd_consecutive_overflows<64,'64 consecutive AMP skips; no recipe change attempted')

    def save_model(self):
        # Keep parent best-selection/CSV/metadata semantics, augment only this experiment.
        super().save_model()
        ckpt=torch_load(self.last,map_location='cpu')
        model=deepcopy(self.model).cpu().float();model.model[5].lcd.collect_diagnostics=False
        ema=deepcopy(self.ema.ema).cpu().float();ema.model[5].lcd.collect_diagnostics=False
        ckpt.update(model=model,ema=ema,optimizer=deepcopy(self.optimizer.state_dict()),
                    lcd_state=dict(research=RESEARCH,identity=self.lcd_identity,
                        scheduler=self.scheduler.state_dict(),accumulate=self.accumulate,
                        last_opt_step=self._lcd_last_opt_step,
                        pending_scaled_gradients={n:p.grad.detach().cpu().clone() for n,p in self.model.named_parameters() if p.grad is not None},
                        rng=rng_state(),stopper=deepcopy(vars(self.stopper)),
                        effective_updates=self.lcd_effective_updates,overflows=self.lcd_overflows,
                        consecutive_overflows=self.lcd_consecutive_overflows,
                        epoch_boundary=True,arbitrary_batch_exact_resume=False))
        atomic_checkpoint(self.last,ckpt)
        if self.best_fitness==self.fitness:atomic_checkpoint(self.best,ckpt)
        if self.save_period>0 and self.epoch%self.save_period==0:
            atomic_checkpoint(self.wdir/f'epoch{self.epoch}.pt',ckpt)

    def resume_training(self,ckpt):
        if ckpt is None or not self.resume:return
        state=ckpt.get('lcd_state')
        require(state and state['research']==RESEARCH and state['epoch_boundary'],'LCD full epoch state missing')
        # lcd_identity is supplied before train(), and excludes mutable logs/reports.
        require(state['identity']==self.lcd_identity,'Resume source/init/recipe/data identity changed')
        super().resume_training(ckpt)
        self.model.load_state_dict(ckpt['model'].float().state_dict(),strict=True)
        self._lcd_last_opt_step=state['last_opt_step'];self.accumulate=state['accumulate']
        self.lcd_effective_updates=state['effective_updates'];self.lcd_overflows=state['overflows']
        self.lcd_consecutive_overflows=state['consecutive_overflows']
        vars(self.stopper).update(state['stopper']);self.lcd_pending=state
        verify_model(self.model)

    def final_eval(self):
        # Preserve complete best/last state; parent final_eval strips optimizer and epoch.
        if self.best.exists():
            self.validator.args.plots=self.args.plots;self.validator.args.compile=False
            self.metrics=self.validator(model=self.best);self.metrics.pop('fitness',None)
            self.run_callbacks('on_fit_epoch_end')
