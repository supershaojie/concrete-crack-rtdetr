"""Bounded synthetic CUDA lifecycle test. NEVER a formal-data training entry point."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
import random
from pathlib import Path
import sys
from unittest.mock import patch
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parent))

from support import HERE, PROJECT, read_json, write_json, initialization_record, git
from worker import activate, build_initial_model, check_model, options, load_crack_model, export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=PROJECT/'outputs/yolov5m-scratch/assets')
    a=parser.parse_args()
    run,assets=a.run.resolve(),a.assets.resolve()
    run.mkdir(parents=True,exist_ok=False)
    verified=activate(assets)
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from utils.callbacks import Callbacks
    from utils.torch_utils import ModelEMA
    from bench_yolov5_runtime import strict_amp_probe, validate_checkpoint
    import train
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'This smoke requires CUDA; no silent CPU/AMP fallback'
    report={'scope':'SMOKE_ONLY_SYNTHETIC', 'formal_training':False,
            'formal_batch16_imgsz640_parallel_capacity':'NOT_TESTED'}
    forbidden=AssertionError('External model-state loading during scratch initialization')
    with patch('torch.load',side_effect=forbidden), patch.object(torch.nn.Module,'load_state_dict',side_effect=forbidden):
        model,record=build_initial_model(verified)
        again,_=build_initial_model(verified)
    assert all(torch.equal(v,again.state_dict()[k]) for k,v in model.state_dict().items())
    assert record['pretrained_tensors_loaded']==0
    report['initialization']={**record,**check_model(model,1), 'state_imports_forbidden':'passed',
                              'same_seed_fresh_rebuild':'passed'}
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

    # Every object used above is discarded. Native train re-seeds and constructs again.
    for split in ('train','val','test'):
        (run/'images'/split).mkdir(parents=True)
        (run/'labels'/split).mkdir(parents=True)
        for i in range(2):
            Image.new('RGB',(96,80),(70+i*20,100,150)).save(run/'images'/split/(str(i)+'.jpg'))
            (run/'labels'/split/(str(i)+'.txt')).write_text('0 .5 .5 .4 .4\n',encoding='utf-8')
    data=run/'source.yaml'
    data.write_text(yaml.safe_dump({'path':run.as_posix(),'names':{0:'crack'},
         **{s:'images/'+s for s in ('train','val','test')}}),encoding='utf-8')
    from data import inspect,save_check,make_gt
    manifest=inspect(data,run,run,enforce_counts=False)
    save_check(manifest,run/'data')
    sys.path.insert(0,str(HERE.parent/'evaluation'))
    from evaluate import POLICY_SHA,evaluate,read_public
    init=initialization_record(verified['upstream'])
    ident={'run_id':uuid.uuid4().hex,'initialization':init,'scope':'SMOKE_ONLY'}
    write_json(run/'frozen.json',{'checkpoint_identity':ident,'initialization':init,
       'prediction_identity':{'model':'original_yolov5m_random_SMOKE','model_code_sha':git('rev-parse','HEAD'),
       'dataset_identity_sha256':manifest['dataset_identity_sha256'],
       'evaluation_config_sha256':POLICY_SHA,'postprocessing':'production FP32 on synthetic images'}})
    def smoke_options(resume=False):
        opt=options(run,assets,resume)
        opt.epochs,opt.batch_size,opt.imgsz,opt.workers,opt.device=2,2,64,0,'0'
        return opt
    cb=Callbacks()
    calls=[]
    def interrupt_after_complete_epoch():
        calls.append(True)
        if len(calls)==2:
            raise KeyboardInterrupt('Intentional smoke interruption after complete checkpoint')
    cb.register_action('on_train_epoch_start',callback=interrupt_after_complete_epoch)
    with patch('torch.load',side_effect=forbidden), patch.object(torch.nn.Module,'load_state_dict',side_effect=forbidden):
        try:
            train.train(str(HERE/'hyp.yaml'),smoke_options(),torch.device('cuda:0'),cb)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('Smoke interrupt not exercised')
    assert not (run/'native/training_complete.json').exists()
    checkpoint=torch.load(run/'native/weights/last.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(checkpoint,run/'native',resume=True)
    for bad in ({**checkpoint,'comparison_identity':None},
                {**checkpoint,'comparison_identity':{**ident,'run_id':'another-scratch-run'}},
                {**checkpoint,'comparison_identity':{**ident,'initialization':{'initialization_type':'coco'}}}):
        try:
            validate_checkpoint(bad,run/'native',resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Foreign checkpoint accepted')
    original_record=(run/'native/initialization.json').read_bytes()
    del checkpoint,bad
    gc.collect();torch.cuda.empty_cache()
    train.train(str(HERE/'hyp.yaml'),smoke_options(True),torch.device('cuda:0'),Callbacks())
    assert original_record==(run/'native/initialization.json').read_bytes()
    complete=read_json(run/'native/training_complete.json')
    assert complete['completed_epochs']==2
    ckpt=torch.load(run/'native/weights/last.pt',map_location='cpu',weights_only=False)
    try:
        validate_checkpoint(ckpt,run/'native',resume=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Completed checkpoint accepted for resume')
    report['native_lifecycle']={'batch':2,'imgsz':64,'epochs':2,'device':'cuda:0','AMP':True,
        'interrupted_after_epoch':1,'resumed_to_epoch':2,'foreign_and_completed_checkpoint_rejection':'passed',
        'zero_state_imports_during_first_epoch':'passed','initialization_record_preserved':True,
        'completion':complete}
    (run/'evaluation').mkdir()
    for split in ('val','test'):
        gt=make_gt(manifest,split)
        write_json(run/'evaluation'/(split+'_gt.json'),gt)
        export(run,split,run/'native/weights/best.pt')
        pred_identity,rows=read_public(run/'evaluation'/(split+'_predictions.jsonl'))
        metrics=evaluate(gt,rows,pred_identity)
        assert metrics['images']==2
    report['export_evaluate']='Actual scratch best; production FP32 640 exporter and common evaluator, two synthetic images per split'
    write_json(run/'scratch_smoke.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
