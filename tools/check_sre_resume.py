"""Disposable synthetic CUDA audit of native save_model/get_model/resume_training.

Runs no data loader, val/test split or formal epoch. B2/160 native AMP updates
exercise learned SRE state; this is not the real-data B16/640 capacity gate.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from check_sre import build, CONFIGS, detection_updates
from preflight_sre import native_resume_audit
from sre_common import ROOT, code_identity, runtime, require, write_json
import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA


def run(variants=None):
    report=dict(status='RUNNING',scope='Synthetic B2/160; actual native save/get_model/resume methods; no real-data capacity claim',
        identity=code_identity(),runtime=runtime(),variants={},formal_training='NOT_STARTED',final_test='NOT_RUN',
        real_B16_640_capacity='PENDING: separate preflight_sre.py gate')
    if not torch.cuda.is_available():
        report.update(status='PENDING',reason='CUDA unavailable; native GradScaler-state resume requires CUDA')
        return report
    with tempfile.TemporaryDirectory(prefix='sre-native-resume-') as temporary:
        for variant in variants or CONFIGS:
            folder=Path(temporary)/variant;folder.mkdir()
            model=build(CONFIGS[variant][1]).to('cuda')
            ema=ModelEMA(model)
            optimizer,scaler,updates=detection_updates(model,'cuda',amp=True)
            require(updates['effective_updates']>=2,'Need two actual learned updates')
            ema.update(model)
            require(ema.updates==1 and torch.count_nonzero(ema.ema.model[19].sre.W_o.weight)>0,'Native EMA learned update missing')
            args=YAML.load(ROOT/'docs/sre/parent_args.yaml')
            args.update(model=str(folder/'disposable_smoke.pt'),data='SYNTHETIC_NO_DATASET',batch=2,imgsz=160,
                        project=str(folder),name='disposable_resume_audit',save_dir=str(folder),workers=0)
            model.args=args;ema.ema.args=args
            trainer=RTDETRTrainer.__new__(RTDETRTrainer)
            trainer.model=model;trainer.optimizer=optimizer;trainer.scaler=scaler;trainer.ema=ema
            trainer.data=dict(nc=1,channels=3);trainer.device=torch.device('cuda:0')
            trainer.args=SimpleNamespace(**args);trainer.wdir=folder/'weights'
            trainer.last=trainer.wdir/'last.pt';trainer.best=trainer.wdir/'best.pt'
            trainer.csv=folder/'results.csv';trainer.save_period=-1;trainer.metrics={}
            audit=native_resume_audit(trainer,folder)
            audit.update(no_formal_epoch_run=True,no_val_or_test=True,checkpoint_discarded=True,
                input_scope='synthetic B2/160',EMA_updates_before_save=ema.updates)
            report['variants'][variant]=dict(status='PASSED',updates=updates,native_resume=audit)
            del trainer,model,optimizer,scaler,ema
            gc.collect();torch.cuda.empty_cache()
    report['status']='PASSED'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['both']+list(CONFIGS),default='both')
    args=parser.parse_args();torch.set_num_threads(4)
    try:report=run(None if args.variant=='both' else [args.variant])
    except BaseException as error:
        write_json(args.output,dict(status='FAILED',error=repr(error),formal_training='NOT_STARTED',final_test='NOT_RUN'))
        raise
    write_json(args.output,report)
    print(json.dumps(dict(status=report['status'],output=str(args.output)),indent=2))
    return 0 if report['status']=='PASSED' else 2


if __name__=='__main__':raise SystemExit(main())
