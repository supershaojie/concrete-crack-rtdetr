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
    assert record['parameters_unfused']==25856899 and record['pretrained_tensors_loaded']==0
    assert all(torch.equal(v,again.state_dict()[k]) for k,v in model.state_dict().items())
    assert not model.model[-1].dfl.conv.weight.requires_grad
    assert torch.equal(model.model[-1].dfl.conv.weight.flatten(),torch.arange(16,dtype=torch.float32))
    report={'scope':'SMOKE_ONLY_SYNTHETIC','formal_training':False,
            'formal_batch16_imgsz640_parallel_capacity':'NOT_TESTED','initialization':record,
            'state_imports_forbidden_during_construction':'passed','native_fixed_DFL':'passed'}
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
                     exist_ok=True,epochs=2,batch=2,imgsz=64,workers=0,plots=False)
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
    original_record=(run/'initialization.json').read_bytes()
    del t,ckpt,bad
    gc.collect();torch.cuda.empty_cache()
    overrides.update(model=str(run/'train/weights/last.pt'),resume=str(run/'train/weights/last.pt'))
    t=SmokeTrainer(overrides=overrides,run=run,manifest=manifest,identity=ident)
    t.train()
    assert t.epoch+1==2 and original_record==(run/'initialization.json').read_bytes()
    last=torch.load(t.last,map_location='cpu',weights_only=False)
    assert last['comparison_identity']==ident
    completed={'status':'completed','completed_epoch':2,'best_epoch':t.comparison_best_epoch,
               'best_sha256':sha256(t.best),'last_sha256':sha256(t.last),'scope':'SMOKE_ONLY'}
    write_json(run/'train_status.json',completed)
    report['native_lifecycle']={'batch':2,'imgsz':64,'epochs':2,'device':'cuda:0','AMP':True,
        'interrupted_after_epoch':1,'resumed_to_epoch':2,'foreign_checkpoint_rejection':'passed',
        'zero_state_imports_during_first_epoch':'passed','initialization_record_preserved':True}
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
