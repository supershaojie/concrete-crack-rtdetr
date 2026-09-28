"""New interpreter: native CPU resume, finite GPU warmup and ONE real B1 FP32 val batch."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
from pathlib import Path
from types import SimpleNamespace
import traceback

from qcc_v1_common import ROOT, MAIN, MODEL, OUT, data_config, runtime, write_json, sha256, recipe, initialize, SOURCE
import torch
from ultralytics.models.rtdetr.qcc_model import QCCTrainer, QCCDetectionModel
from ultralytics.models.rtdetr.qcc_loss import ramp, QCCDetectionLoss
from ultralytics.models.rtdetr.qcc_val import QCCValidator, EVAL
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load


def run(checkpoint, output, device):
    output.mkdir(parents=True,exist_ok=True)
    report=dict(status='FAIL',runtime=runtime(),checkpoint_sha256=sha256(checkpoint),
                scope='new-process native CPU optimizer/EMA resume (disabled scaler), GPU finite FP32 warmup and one real B1/640 val; NOT formal B16 or full val/test')
    try:
        ckpt=torch_load(checkpoint,map_location='cpu')
        t=QCCTrainer.__new__(QCCTrainer); t.data=dict(nc=1,channels=3)
        t.model=t.get_model(deepcopy(ckpt['ema'].yaml),ckpt['ema'].float(),False); t.model.nc=1
        t.optimizer=RTDETRTrainer.build_optimizer(t,t.model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
        t.scaler=torch.amp.GradScaler('cuda',enabled=False); t.ema=ModelEMA(t.model)
        t.resume=True; t.epochs=200; t.args=SimpleNamespace(model=str(checkpoint),close_mosaic=10)
        t.resume_training(ckpt)
        assert t.start_epoch==20 and t.ema.updates==ckpt['updates']
        assert t.scaler.state_dict()==ckpt['scaler']
        actual=t.optimizer.state_dict()
        assert actual['param_groups']==ckpt['optimizer']['param_groups']
        for k,state in ckpt['optimizer']['state'].items():
            for name,v in state.items():
                a=actual['state'][k][name]
                assert torch.equal(a.cpu(),v.to(a.dtype)) if torch.is_tensor(v) else a==v
        assert all(torch.equal(v,ckpt['ema'].state_dict()[k].float()) for k,v in t.ema.ema.state_dict().items())
        t.epoch=t.start_epoch; QCCTrainer._qcc_epoch_start(t)
        assert t.model.qcc_epoch==20 and ramp(t.model.qcc_epoch)==1
        report['resume']=dict(status='PASS',start_epoch=t.start_epoch,optimizer_exact=True,ema_exact=True,scaler='CPU disabled exact; CUDA PENDING')
        del t,ckpt,actual
        gc.collect()
        warmup=[]; original=AutoBackend.warmup
        def observed(backend,imgsz=(1,3,640,640)):
            handle=backend.model.register_forward_pre_hook(lambda m,a:warmup.append(dict(shape=list(a[0].shape),
                finite=bool(torch.isfinite(a[0]).all()),zero=bool(torch.count_nonzero(a[0])==0),dtype=str(a[0].dtype))))
            try: return original(backend,imgsz)
            finally: handle.remove()
        AutoBackend.warmup=observed
        validator=QCCValidator(args=dict(EVAL,batch=1,model=str(checkpoint),data=str(data_config()),split='val',device=device,plots=False),save_dir=output/'real_val')
        validator.one_batch=True; validator.export_path=output/'one_val_queries_gt.jsonl.gz'
        validator.export_identity=dict(diagnostic=True,checkpoint_sha256=sha256(checkpoint),split='val')
        try: metrics=validator(model=str(checkpoint))
        finally: AutoBackend.warmup=original
        assert len(validator.qcc_seen)==1
        if device!='cpu': assert warmup and all(r['finite'] and r['zero'] for r in warmup)
        import gzip,json
        with gzip.open(validator.export_path,'rt',encoding='utf-8') as f: rows=[json.loads(x) for x in f]
        assert len(rows)==1 and len(rows[0]['query_indices'])==300 and len(rows[0]['logits'])==300
        assert len(rows[0]['gt_classes'])==validator.qcc_gt_count
        report.update(status='PASS',warmup=warmup,real_val_images=1,metrics=metrics,export_sha256=sha256(validator.export_path),actual_settings=validator.actual_settings)
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        write_json(output/'lifecycle.json',report)


def setup_only(init, output):
    from qcc_v1 import strict_amp_resources
    from qcc_v1_preflight import close_workers
    import time
    report=dict(status='FAIL',scope='local actual Trainer setup B2/160 workers0, original AMP; no optimizer update, NOT formal B16 capacity',runtime=runtime())
    t=None
    try:
        report['resources']=strict_amp_resources()
        args,_=recipe(data_config())
        folder=output/f'setup_{time.time_ns()}'
        args.update(model=str(init),project=str(folder),name='run',save_dir=str(folder/'run'),batch=2,imgsz=160,workers=0,plots=False)
        t=QCCTrainer(overrides=args); t._setup_train()
        assert type(t.model) is QCCDetectionModel and type(t.model.init_criterion()) is QCCDetectionLoss
        t.epoch=6; QCCTrainer._qcc_epoch_start(t)
        t.model.train()
        # Lazily create actual criterion through real loss, without another model forward.
        assert t.model.qcc_epoch==6
        report.update(status='PASS',args=vars(t.args),rebuild=t.qcc_rebuild,amp=t.qcc_amp_evidence,scaler=t.scaler.state_dict(),accumulate=t.accumulate)
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if t: close_workers(t)
        write_json(output/'setup.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--setup-only',action='store_true')
    args=p.parse_args(); torch.set_num_threads(4)
    if args.setup_only: setup_only(args.checkpoint,args.output)
    else: run(args.checkpoint,args.output,args.device)
