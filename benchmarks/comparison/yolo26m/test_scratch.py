"""Synthetic CUDA smoke with interruption, same-run resume and production FP32 export."""
from __future__ import annotations
import argparse
from copy import copy
import gc
import json
import os
from pathlib import Path
import random
from unittest.mock import patch
import uuid
from support import (ROOT, SOURCE, configure, model_yaml, model_config, read_json, recipe,
                     run_identity, validate_checkpoint, write_json, sha256)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    run=a.run.resolve()
    run.mkdir(parents=True,exist_ok=False)
    source=configure(run/'runtime')
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from ultralytics.data.build import build_dataloader
    from ultralytics.utils.torch_utils import ModelEMA
    from adapters import ComparisonTrainer, strict_amp_probe, reset_workers
    from run import load_initial_model
    from data import preflight
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA smoke requires a GPU, no silent recipe fallback'
    forbidden=AssertionError('Forbidden external model state import')
    with patch('torch.load',side_effect=forbidden), patch.object(torch.nn.Module,'load_state_dict',side_effect=forbidden):
        model,record=load_initial_model()
        again,_=load_initial_model()
    assert record['end2end'] and record['reg_max']==1 and record['pretrained_tensors_loaded']==0
    assert all(torch.equal(v,again.state_dict()[k]) for k,v in model.state_dict().items())
    assert isinstance(model.model[-1].dfl,torch.nn.Identity)
    assert model.model[-1].cv2 is not None and model.model[-1].one2one_cv2 is not None
    report={'scope':'SMOKE_ONLY_SYNTHETIC','formal_training':False,
            'formal_batch16_imgsz640_parallel_capacity':'NOT_TESTED','initialization':record,
            'state_imports_forbidden_during_construction':'passed','native_reg_max_1_Identity':'passed'}
    del again
    model=model.cuda().train()
    ema=ModelEMA(model)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.937)
    before={k:v.clone() for k,v in model.state_dict().items()}
    ema_before={k:v.clone() for k,v in ema.ema.state_dict().items()}
    py,npstate=random.getstate(),np.random.get_state()
    cpu,cuda=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
    assert strict_amp_probe(model)
    assert model.training and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,ema.ema.state_dict()[k]) for k,v in ema_before.items())
    assert not optimizer.state and ema.updates==0
    assert py==random.getstate() and np.array_equal(npstate[1],np.random.get_state()[1])
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    report['amp_isolation']='FP32/FP16 comparison passed; model/BN/EMA/optimizer/Python/NumPy/CPU/CUDA RNG unchanged'
    del model,ema,optimizer,before,ema_before
    gc.collect();torch.cuda.empty_cache()

    shared=run/'synthetic'
    for split in ('train','val','test'):
        (shared/'images'/split).mkdir(parents=True)
        (shared/'labels'/split).mkdir(parents=True)
        for i in range(2):
            Image.new('RGB',(96,80),(70+i*20,100,150)).save(shared/'images'/split/(str(i)+'.jpg'))
            (shared/'labels'/split/(str(i)+'.txt')).write_text('0 .5 .5 .4 .4\n',encoding='utf-8')
    data=run/'source.yaml'
    data.write_text(yaml.safe_dump({'path':shared.as_posix(),'names':{0:'crack'},
        **{s:'images/'+s for s in ('train','val','test')}}),encoding='utf-8')
    manifest=preflight(data,run,shared,enforce_counts=False)
    run_id='SMOKE_'+uuid.uuid4().hex
    ident=run_identity(manifest,source,require_clean=False,run_id=run_id)
    write_json(run/'run_id.json',{'run_id':run_id})
    write_json(run/'identity.json',ident)

    class SmokeTrainer(ComparisonTrainer):
        def get_dataloader(self,dataset_path,batch_size=2,rank=0,mode='train'):
            ds=self.build_dataset(dataset_path,mode,2)
            loader=build_dataloader(ds,2,0,shuffle=mode=='train',rank=-1)
            loader.reset=lambda:reset_workers(loader)
            return loader

    overrides=recipe()
    overrides.update(model=str(model_yaml()),data=str(run/'data.yaml'),project=str(run),name='train',
                     exist_ok=True,epochs=3,close_mosaic=1,batch=2,imgsz=64,workers=0,plots=False)
    t=SmokeTrainer(overrides=overrides,run=run,manifest=manifest,identity=ident)
    # Reject supplied weights BEFORE calling any native transfer operation.
    with patch('ultralytics.models.yolo.detect.DetectionTrainer.get_model',side_effect=AssertionError('must reject first')):
        try:
            t.get_model(model_config(),weights=object())
        except ValueError:
            pass
        else:
            raise AssertionError('New scratch accepted weights')
    def interrupt_after_complete_epoch(trainer):
        if trainer.epoch==1:
            raise KeyboardInterrupt('Intentional smoke interruption after complete checkpoint')
    t.add_callback('on_train_epoch_start',interrupt_after_complete_epoch)
    with patch('torch.load',side_effect=forbidden), patch.object(torch.nn.Module,'load_state_dict',side_effect=forbidden):
        try:
            t.train()
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('Smoke interruption not exercised')
    ckpt=torch.load(run/'train/weights/last.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(ckpt,ident,resume=True)
    for bad in ({**ckpt,'comparison_identity':None},
                {**ckpt,'comparison_identity':{**ident,'initialization_type':'coco'}},
                {**ckpt,'comparison_identity':{**ident,'run_id':'other-scratch-run'}}):
        try:
            validate_checkpoint(bad,ident,resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Foreign checkpoint accepted')
    for field in ('optimizer','scaler','scheduler','criterion_state','training_model_state','rng_state'):
        bad={**ckpt,field:None}
        try:validate_checkpoint(bad,ident,resume=True)
        except ValueError:pass
        else:raise AssertionError('Missing resume state accepted: '+field)
    for changed in ({'model':'other-model'},{'scope':'FORMAL'},{'dataset_identity_sha256':'wrong'},
                    {'recipe_sha256':'wrong'},{'upstream_commit':'wrong'}):
        bad={**ckpt,'comparison_identity':{**ident,**changed}}
        try:validate_checkpoint(bad,ident,resume=True)
        except ValueError:pass
        else:raise AssertionError('Foreign identity accepted')
    original_record=(run/'initialization.json').read_bytes()
    del t,ckpt,bad
    gc.collect();torch.cuda.empty_cache()
    overrides.update(model=str(run/'train/weights/last.pt'),resume=str(run/'train/weights/last.pt'))
    t=SmokeTrainer(overrides=overrides,run=run,manifest=manifest,identity=ident)
    def assert_restored(trainer):
        c=torch.load(run/'train/weights/last.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v.cpu(),c['training_model_state'][k]) for k,v in trainer.model.state_dict().items())
        assert trainer.scaler.state_dict()==c['scaler']
        assert trainer.scheduler.state_dict()==c['scheduler']
        assert vars(trainer.stopper)==c['stopper_state']
        assert trainer.comparison_last_opt_step==c['last_opt_step']
        assert trainer.ema.updates==c['updates']
        assert all(torch.equal(v.cpu(),c['ema'].state_dict()[k]) for k,v in trainer.ema.ema.state_dict().items())
        current=trainer.optimizer.state_dict()
        assert current['param_groups']==c['optimizer']['param_groups']
        assert all(torch.equal(v.cpu(),c['optimizer']['state'][k][n].cpu()) for k,row in current['state'].items() for n,v in row.items() if torch.is_tensor(v))
    t.add_callback('on_train_start',assert_restored)
    t.train()
    assert t.epoch+1==3 and original_record==(run/'initialization.json').read_bytes()
    last=torch.load(t.last,map_location='cpu',weights_only=False)
    assert last['comparison_identity']==ident
    assert last['criterion_state']['updates']==3
    assert last['criterion_state']['o2m']==.1
    restored=read_json(next(run.glob('resume_state_*.json')))
    assert restored['criterion']['updates']==1 and abs(restored['criterion']['o2m']-.45)<1e-12
    assert last['augmentation_state']['phase']=='closed'
    completed={'status':'completed','completed_epoch':3,'best_epoch':t.comparison_best_epoch,
               'best_sha256':sha256(t.best),'last_sha256':sha256(t.last),'scope':'SMOKE_ONLY'}
    write_json(run/'train_status.json',completed)
    report['native_lifecycle']={'batch':2,'imgsz':64,'epochs':3,'device':'cuda:0','AMP':True,
        'interrupted_after_epoch':1,'resumed_to_epoch':3,'foreign_checkpoint_rejection':'passed',
        'zero_state_imports_during_first_epoch':'passed','initialization_record_preserved':True,'loss_schedule_resume':restored['criterion'],
        'final_loss_schedule':last['criterion_state'],'full_training_model_scaler_scheduler_stopper_ema_counter_restored':True}
    del t,last
    gc.collect();torch.cuda.empty_cache()
    from export import export_split,evaluate_split
    # Smoke can run on an uncommitted patch; all code/data/config/run comparisons still execute.
    with patch('export.run_identity',side_effect=lambda m,s,**kw:run_identity(m,s,require_clean=False,**kw)):
        for split in ('val','test'):
            result=export_split(run,split,manifest,source)
            assert result['images']==2
            evaluate_split(run,split)
    report['export_evaluate']='Actual scratch best; production FP32 640 exporter and common evaluator, two synthetic images per split'
    write_json(run/'scratch_smoke.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
