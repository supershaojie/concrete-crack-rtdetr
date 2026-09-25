"""Native Trainer with bounded scalar observation; no change to formal optimizer/loss/AMP."""
from __future__ import annotations

from copy import copy, deepcopy
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from cqs_v1_common import *
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils.torch_utils import ModelEMA


def head_of(model):
    if not isinstance(model, RTDETRDetectionModel):
        model = model.model  # AutoBackend
    return verify_model(model)


class CQSValidator(RTDETRValidator):
    """Native per-epoch postprocess/fitness, with evidence that CQS actually ran."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.cqs_head = head_of(model)
        self.cqs_before = self.cqs_head.cqs_calls

    def finalize_metrics(self):
        require(self.cqs_head.cqs_calls > self.cqs_before, 'Validator did not execute CQS')
        self.cqs_evidence = dict(head=type(self.cqs_head).__name__, config=self.cqs_head.cqs_config,
                                 calls=self.cqs_head.cqs_calls-self.cqs_before, native_epoch_policy=True)
        return super().finalize_metrics()


def selection_summary(row):
    extra = row['extra_count'].detach().float().cpu()
    factors = row['discount_factors'].detach().float().cpu().flatten()
    return dict(pool_size=row['pool_size'], core_queries=240, complement_queries=60, images=extra.numel(),
                extra_mean=float(extra.mean()), extra_min=float(extra.min()), extra_max=float(extra.max()),
                native_overlap_mean=float(row['overlap'].mean()), discount_min=float(factors.min()),
                discount_mean=float(factors.mean()), discount_max=float(factors.max()),
                discount_histogram=torch.histc(factors,bins=10,min=.5,max=1.).int().tolist(),
                duplicate_indices=int(row['duplicates'].sum()), nonfinite=row['nonfinite'],
                representative_indices=row['representative_indices'].cpu().tolist())


class BoundedStop(Exception):
    pass


class CQSTrainer(RTDETRTrainer):
    def __init__(self, *args, cqs_folder=None, diagnostic=False, formal=False, dispatch=None, interface_scale=None, **kwargs):
        self.cqs_folder = Path(cqs_folder or OUT)
        self.cqs_folder.mkdir(parents=True,exist_ok=True)
        self.diagnostic, self.formal, self.dispatch = diagnostic, formal, dispatch
        require(not formal or interface_scale is None,'Formal scaler may not be overridden')
        self.interface_scale = interface_scale
        self.micro_batches = 0
        self.effective_updates = 0
        self.update_rows = []
        self.epoch_events = []
        self.cqs_started = time.monotonic()
        super().__init__(*args, **kwargs)
        self.add_callback('on_train_start', self.cqs_setup)
        self.add_callback('on_train_batch_start', self.cqs_batch_start)
        self.add_callback('on_train_batch_end', self.cqs_batch_end)
        self.add_callback('on_fit_epoch_end', self.cqs_epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model,audit = training_rebuild(cfg,weights,self.data,initial=not bool(self.args.resume))
        write_json(self.cqs_folder/'trainer_rebuild.json',audit)
        return model

    def get_validator(self):
        self.loss_names = 'giou_loss','cls_loss','l1_loss'
        return CQSValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args))

    def cqs_setup(self, trainer):
        head = verify_model(self.model)
        verify_model(self.ema.ema)
        require(all(p.requires_grad for p in self.model.parameters()), 'A model parameter is frozen')
        ids = [id(p) for g in self.optimizer.param_groups for p in g['params']]
        require(all(ids.count(id(p))==1 for p in self.model.parameters()), 'Optimizer missing/duplicate parameters')
        require(type(self.optimizer) is torch.optim.AdamW, 'Expected native AdamW')
        if self.formal or self.diagnostic:
            require(self.amp, 'Real native check_amp did not enable AMP')
        if self.diagnostic or (self.interface_scale is not None and not self.resume):
            require(self.diagnostic or self.interface_scale==128,'Only isolated diagnostic scale128 is supported')
            self.scaler = torch.cuda.amp.GradScaler(enabled=True,init_scale=128)
        if self.diagnostic:
            if not hasattr(self.model,'criterion'):
                self.model.criterion = self.model.init_criterion()
            def record_losses(module, inputs, output):
                self.last_all_losses = {k:v.detach() for k,v in output.items()}
            self.model.criterion.register_forward_hook(record_losses)
        self._step_observer = self.optimizer.register_step_post_hook(self._on_effective_step)
        self._sample_params = {k:v for k,v in self.model.named_parameters() if k in {
            'model.26.enc_score_head.bias','model.26.enc_bbox_head.layers.2.bias',
            'model.26.cbr.offset_out.bias','model.20.O_proj.weight'}}
        actual = vars(self.args)
        if self.formal:
            planned = YAML.load(OUT/'train_args.yaml')
            allowed_resume = {'model','resume'} if self.resume else set()
            diff = {k:[v,actual.get(k)] for k,v in planned.items()
                    if k not in allowed_resume and (type(actual.get(k)) is not type(v) or actual.get(k)!=v)}
            require(not diff, f'Actual formal recipe differs: {diff}')
        self.setup_evidence = dict(head=type(head).__name__,config=head.cqs_config,
            parameters=sum(p.numel() for p in self.model.parameters()),AMP=self.amp,batch=self.batch_size,
            imgsz=self.args.imgsz,optimizer=type(self.optimizer).__name__,nbs=self.args.nbs,
            accumulate=self.accumulate,scaler=self.scaler.state_dict(),
            diagnostic_only_scale=128 if self.diagnostic or self.interface_scale else None,
            optimizer_groups=[dict(lr=g['lr'],weight_decay=g['weight_decay'],kind=g.get('param_group'),
                                   count=len(g['params'])) for g in self.optimizer.param_groups],
            actual_args=actual,criterion_loss_gain=self.model.init_criterion().loss_gain,
            matcher_cost_gain=self.model.init_criterion().matcher.cost_gain)
        write_json(self.cqs_folder/'training_setup.json',self.setup_evidence)

    def _on_effective_step(self, optimizer, args, kwargs):
        self.effective_updates += 1

    def cqs_batch_start(self, trainer):
        # Native auto-OOM fallback must not silently change B16/640.
        self._oom_retries = 3
        if self.diagnostic and time.monotonic()-self.cqs_started >= 880:
            raise BoundedStop('time boundary')
        self.model.model[-1]._cqs_sink = ([],False) if self.micro_batches%100==0 or self.diagnostic else None

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.formal:
            write_json(self.cqs_folder/'state.json',dict(dispatch=self.dispatch,status='RUNNING',
                phase='TRAINING_RUNNING',pid=os.getpid(),epoch=self.epoch+1,micro_batches=self.micro_batches,
                updated=utc(),shape=list(batch['img'].shape)))
        self.last_shape = list(batch['img'].shape)
        return batch

    def optimizer_step(self):
        before = {k:v.detach().clone() for k,v in self._sample_params.items()}
        scale = self.scaler.get_scale()
        count = self.effective_updates
        gradients = {}
        if self.diagnostic or self.micro_batches < 2:
            groups = dict(LIF='model.20.',CBR='model.26.cbr.',encoder_class='model.26.enc_score_head.',
                          encoder_bbox='model.26.enc_bbox_head.',backbone='model.0.')
            for name,prefix in groups.items():
                grads = [p.grad.detach().float()/scale for k,p in self.model.named_parameters() if k.startswith(prefix) and p.grad is not None]
                gradients[name] = dict(finite=all(bool(torch.isfinite(g).all()) for g in grads),
                                       norm=sum(float(g.norm()) for g in grads),with_grad=len(grads))
        super().optimizer_step()  # native unscale, clipping=10, scaler.step/update, zero_grad, EMA
        changed = {k:bool(torch.any(v.detach()!=before[k])) for k,v in self._sample_params.items()}
        row = dict(micro_batch=self.micro_batches,loss=float(self.loss.detach()),loss_items=self.loss_items,
                   accumulate=self.accumulate,scale_before=scale,scale_after=self.scaler.get_scale(),
                   effective=self.effective_updates>count,parameter_changed=changed,gradients=gradients)
        if self.diagnostic or self.micro_batches < 2:
            if self.diagnostic:
                row['all_losses'] = self.last_all_losses
            self.update_rows.append(json_safe(row))

    def _load_checkpoint_state(self, ckpt):
        super()._load_checkpoint_state(ckpt)
        require(self.scaler.state_dict()==ckpt['scaler'],'Native resume scaler differs')
        compare_states(ckpt['ema'].float(),self.ema.ema)
        actual=self.optimizer.state_dict()
        saved=ckpt['optimizer']
        require(actual['param_groups']==saved['param_groups'],'Native resume optimizer groups differ')
        for key,values in saved['state'].items():
            for name,value in values.items():
                restored=actual['state'][key][name]
                require(torch.equal(value.to(restored),restored) if isinstance(value,torch.Tensor) else value==restored,
                        f'Native resume optimizer state differs: {key}.{name}')
        write_json(self.cqs_folder/'resume_state.json',dict(optimizer_exact_after_dtype_restore=True,scaler_exact=True,
                   ema_exact=True,ema_updates=self.ema.updates,source_epoch=ckpt['epoch'],head=type(self.model.model[-1]).__name__))

    def cqs_batch_end(self, trainer):
        head = self.model.model[-1]
        sink = head._cqs_sink
        head._cqs_sink = None
        if sink:
            require(len(sink[0]) == 1,'Expected one CQS selection per train micro-batch')
            row = dict(epoch=self.epoch+1,step=self.epoch*len(self.train_loader)+self.micro_batches%len(self.train_loader),
                       dispatch=self.dispatch,**selection_summary(sink[0][0]))
            require(row['duplicate_indices']==0 and row['nonfinite']==0,'Invalid selection diagnostic')
            self.epoch_events.append(row)
            with (self.cqs_folder/'selection_events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.micro_batches += 1
        if self.diagnostic:
            write_json(self.cqs_folder/'capacity_progress.json',dict(micro_batches=self.micro_batches,
                effective_updates=self.effective_updates,updates=self.update_rows,shape=self.last_shape))
            if self.effective_updates>=2 or self.micro_batches>=16:
                raise BoundedStop('two updates or 16 micro-batches')

    def cqs_epoch_end(self, trainer):
        if not self.epoch_events:
            return
        events = self.epoch_events
        total = sum(r['images'] for r in events)
        summary = dict(epoch=self.epoch+1,dispatch=self.dispatch,sampled_batches=len(events),sampled_images=total,
                       pool_size=sorted({r['pool_size'] for r in events}),core_queries=240,complement_queries=60,
                       extra_mean=sum(r['extra_mean']*r['images'] for r in events)/total,
                       native_overlap_mean=sum(r['native_overlap_mean']*r['images'] for r in events)/total,
                       discount_mean=sum(r['discount_mean']*r['images'] for r in events)/total,
                       discount_min=min(r['discount_min'] for r in events),discount_max=max(r['discount_max'] for r in events),
                       discount_histogram=np.asarray([r['discount_histogram'] for r in events]).sum(0).tolist(),
                       nonfinite=sum(r['nonfinite'] for r in events),duplicates=sum(r['duplicate_indices'] for r in events),
                       epoch_validator=getattr(self.validator,'cqs_evidence',None))
        with (self.cqs_folder/'selection_epochs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(summary,allow_nan=False)+'\n')
        self.epoch_events = []


class LimitedLoader:
    def __init__(self, loader):
        self.loader, self.dataset = loader, loader.dataset
    def __len__(self):
        return 1
    def __iter__(self):
        yield next(iter(self.loader))


def capacity_worker(folder):
    folder = Path(folder)
    args = YAML.load(OUT/'train_args.yaml')
    args.update(project=str(folder),name='isolated',save_dir=str(folder/'isolated'),workers=8,plots=False,save=False)
    report = dict(status='FAIL',scope='server B16/640 bounded native Trainer',max_micro_batches=16,max_seconds=900,
                  diagnostic_only_scale=128,formal_scaler='unchanged native default',formal_training='NOT_RUN')
    t = None
    try:
        require(torch.cuda.is_available(),'CUDA required')
        torch.cuda.reset_peak_memory_stats()
        t = CQSTrainer(overrides=args,cqs_folder=folder,diagnostic=True)
        try:
            t.train()
        except BoundedStop as end:
            report['boundary'] = str(end)
        require(t.effective_updates>=1,'Zero effective optimizer updates')
        require(t.last_shape == [16,3,640,640] and t.micro_batches<=16,'Capacity shape/bound violated')
        successful = [r for r in t.update_rows if r['effective']]
        require(any(any(r['parameter_changed'].values()) for r in successful),'No real parameter changes')
        require(all(isinstance(r['loss'],(int,float)) and math.isfinite(r['loss']) and all(g['finite'] for g in r['gradients'].values()) for r in successful),
                'Nonfinite successful update')
        require(all(any(r['gradients'][key]['norm']>0 for r in successful) for key in ('LIF','CBR','encoder_class','encoder_bbox')),
                'Missing related gradients')
        # One real validation batch with native training-time half/EMA behavior.
        t.validator.dataloader = LimitedLoader(t.get_dataloader(t.data['val'],batch_size=16,rank=-1,mode='val'))
        t.validator.args.plots = False
        t.validate()
        report['epoch_validator'] = t.validator.cqs_evidence
        # Isolated checkpoint/predict; never fed back into formal initialization.
        snapshot = folder/'diagnostic_predict.pt'
        m = deepcopy(t.ema.ema).float().eval();m.args = vars(t.args)
        torch.save(dict(model=m,ema=None,train_args=vars(t.args)),snapshot)
        loaded = RTDETR(str(snapshot));h=verify_model(loaded.model);before=h.cqs_calls
        loaded.predict(source=np.zeros((640,640,3),dtype=np.uint8),imgsz=640,device='0',verbose=False,save=False)
        require(h.cqs_calls>before,'Predict did not execute CQS')
        report.update(status='PASS',predict=dict(head=type(h).__name__,calls=h.cqs_calls-before),
                      setup=t.setup_evidence,shape=t.last_shape)
    except BaseException as error:
        import traceback
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if t:
            report.update(micro_batches=t.micro_batches,effective_updates=t.effective_updates,updates=t.update_rows,
                          overflow_skips=sum(not r['effective'] for r in t.update_rows),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(folder/'capacity.json',report)


def fusion_check(model, device='cpu'):
    a = deepcopy(model).to(device).eval().float()
    b = deepcopy(a).fuse(verbose=False)
    compare_states(a.model[20],b.model[20])
    require(hasattr(b.model[20],'bn'),'LIF BN was fused')
    x = torch.rand(2,3,160,160,device=device)
    with torch.no_grad(),a.model[-1].capture_selection(full=True) as ar,b.model[-1].capture_selection(full=True) as br:
        ya,yb = a(x)[0],b(x)[0]
    ids_a,ids_b = ar[0]['indices'],br[0]['indices']
    overlap = (ids_a[:,:,None]==ids_b[:,None,:]).any(-1).sum(-1)
    require(torch.isfinite(ya).all() and torch.isfinite(yb).all(),'Nonfinite fused final output')
    alignment = dict(status='NOT_APPLICABLE',reason='different query sets')
    if bool((overlap==300).all()):
        order=(ids_a[:,:,None]==ids_b[:,None,:]).long().argmax(-1)
        aligned=yb.gather(1,order[...,None].expand(-1,-1,yb.shape[-1]))
        alignment=dict(status='MEASURED',max_abs=float((ya-aligned).abs().max()),
                       allclose=bool(torch.allclose(ya,aligned,atol=2e-5,rtol=2e-4)))
    return dict(status='OBSERVED',lif_states_exact=True,config=CQS_CONFIG,
                indices_exact=bool(torch.equal(ids_a,ids_b)),overlap_count=overlap.cpu().tolist(),
                final_output_max_abs=float((ya-yb).abs().max()),same_ids_alignment=alignment,
                note='Natural beta0.5 selections recorded; no parent cutoff exemption or equivalence claim')


def independent_load(checkpoint, data, folder, device):
    model = RTDETR(str(checkpoint))
    head = verify_model(model.model)
    before = head.cqs_calls
    model.val(data=str(data),device='0' if device=='cuda' else 'cpu',imgsz=160,batch=2,workers=0,
              half=False,plots=False,project=str(folder),name='independent_val',validator=CQSValidator,verbose=False)
    require(head.cqs_calls>before,'Independent val did not use CQS')
    calls_val=head.cqs_calls-before;before=head.cqs_calls
    model.predict(np.zeros((160,160,3),dtype=np.uint8),imgsz=160,device='0' if device=='cuda' else 'cpu',
                  save=False,verbose=False)
    require(head.cqs_calls>before,'Independent predict did not use CQS')
    # Full decode input on equal features gives a nontrivial geometric complement.
    h=verify_model(RTDETR(str(checkpoint)).model).to(device).eval()
    with torch.no_grad(),h.capture_selection(full=True) as rows:
        h._get_decoder_input(torch.ones(2,8400,256,device=device),[[80,80],[40,40],[20,20]])
    introduced=int(rows[0]['extra_count'].sum())
    require(introduced>0,'Lifecycle regression: CQS did not change native query set')
    report=dict(status='PASS',head=type(head).__name__,config=head.cqs_config,val_calls=calls_val,
                predict_calls=head.cqs_calls-before,nondegenerate_extra_count=introduced,
                checkpoint_sha256=sha256(checkpoint))
    write_json(Path(folder)/'independent_load.json',report)


def local_lifecycle(folder,device,source):
    """Two-image real Trainer interface test. Explicitly not a formal capacity check."""
    import cv2
    folder=Path(folder)/'lifecycle';folder.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(42)
    for split in ('train','val'):
        (folder/'data/images'/split).mkdir(parents=True)
        (folder/'data/labels'/split).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(folder/f'data/images/{split}/{i}.jpg'),rng.integers(0,256,(160,160,3),dtype=np.uint8))
            (folder/f'data/labels/{split}/{i}.txt').write_text('0 0.5 0.5 0.2 0.3\n',encoding='utf-8')
    data=folder/'data.yaml';YAML.save(data,dict(path=str(folder/'data'),train='images/train',val='images/val',names={0:'crack'}))
    init=folder.parent/'controlled_init.pt'
    require(init.is_file(),'Lifecycle requires controlled init')
    args=YAML.load(ROOT/'docs/cqs_v1/parent_args.yaml')
    args.update(model=str(init),data=str(data),project=str(folder),name='native',save_dir=str(folder/'native'),
                batch=2,imgsz=160,workers=0,epochs=2,device='0' if device=='cuda' else 'cpu',amp=device=='cuda',
                plots=False,mosaic=0.,mixup=0.)
    # Stop just after native first-epoch save, keeping optimizer/scaler/EMA for real resume.
    class FirstEpochSaved(Exception):
        pass
    wrapper=RTDETR(str(init))
    def saved(t):
        raise FirstEpochSaved('intentional interface-test interruption after native checkpoint save')
    wrapper.add_callback('on_model_save',saved)
    try:
        wrapper.train(trainer=partial(CQSTrainer,cqs_folder=folder/'first',interface_scale=128 if device=='cuda' else None),**args)
    except FirstEpochSaved:
        pass
    trainer=wrapper.trainer
    require(trainer.last.is_file() and trainer.effective_updates>=1,'No native save/update')
    checkpoint=torch_load(trainer.last,map_location='cpu')
    require(checkpoint['optimizer'] and checkpoint['scaler'] is not None and checkpoint['ema'] is not None,'Missing native resume state')
    original_last=folder/'first_epoch_last.pt'
    import shutil
    shutil.copyfile(trainer.last,original_last)
    report=dict(scope='B2/160 synthetic images; native interface test only',diagnostic_only_scale=128 if device=='cuda' else None,first_updates=trainer.effective_updates,
                first_update_evidence=trainer.update_rows,
                first_epoch_validator=trainer.validator.cqs_evidence,fusion=fusion_check(checkpoint['ema'].float(),device))
    resume=RTDETR(str(trainer.last))
    resume.train(resume=True,trainer=partial(CQSTrainer,cqs_folder=folder/'resume',interface_scale=128 if device=='cuda' else None))
    t=resume.trainer
    require(t.start_epoch==1 and t.effective_updates>=1,'Native resume failed to continue at epoch 2')
    require(t.ema.updates>checkpoint['updates'],'EMA did not resume')
    report['resume']=dict(start_epoch=t.start_epoch,actual_epoch=t.epoch+1,effective_updates=t.effective_updates,
                          ema_updates=t.ema.updates,prior_ema_updates=checkpoint['updates'],validator=t.validator.cqs_evidence,
                          updates=t.update_rows,restore=read_json(folder/'resume/resume_state.json'))
    command=[sys.executable,str(ROOT/'tools/cqs_v1.py'),'_load-check',str(t.best),str(data),str(folder),device]
    subprocess.run(command,env=os.environ.copy(),cwd=ROOT,check=True,timeout=180)
    report['independent_process']=read_json(folder/'independent_load.json')
    return report
