"""Small local native AMP/FP16 validator check; not the server B16/640 gate."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch
from pds_v1_common import *
from pds_v1_trainer import PDSTrainer
from pds_v1_deploy import strip_model
from check_pds_v1 import batch
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_amp
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA, autocast


def run(checkpoint,folder):
    torch.set_num_threads(4)
    folder.mkdir(parents=True,exist_ok=True)
    report=dict(status="FAILED",scope="local B2/160 only; server B16/640 PENDING",diagnostic_only_scale=128)
    try:
        model=torch_load(checkpoint,map_location="cpu")["model"].cuda().float().train()
        require(check_amp(model),"Real native AMP check failed")
        t=PDSTrainer.__new__(PDSTrainer)
        t.pds_effective_steps=0
        t.pds_overflow_skips=0
        t.pds_step_diagnostics=[]
        t.model=model
        t.save_dir=folder
        t.optimizer=t.build_optimizer(model,name="AdamW",lr=.0005,momentum=.937,decay=.0001)
        t.scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=128)
        t.ema=ModelEMA(model)
        t.epoch=20
        model.pds_epoch=20
        b=batch("cuda:0")
        observed={}
        def dtype_hook(m,args,out):
            observed.update(input_dtypes=[str(x.dtype) for x in args],output_dtypes=[str(x.dtype) for x in out])
        hook=model.pds_head.register_forward_hook(dtype_hook)
        initial=model.pds_head.box.weight.detach().clone()
        with autocast(True):
            loss,items=model(b)
        hook.remove()
        require(observed["output_dtypes"]==["torch.float32","torch.float32"],"PDS head left FP32 island")
        t.scaler.scale(loss).backward()
        t.optimizer_step()
        require(t.pds_effective_steps==1,"Local AMP had no actual optimizer update")
        require(not torch.equal(initial,model.pds_head.box.weight),"AMP head unchanged")
        sample={k:v.detach().cpu().clone() for k,v in b.items()}
        sample["img"]=(sample["img"]*255).round().to(torch.uint8)
        class OneBatch:
            dataset=[0,1]
            def __len__(self):return 1
            def __iter__(self):yield deepcopy(sample)
        class FiniteValidator(RTDETRValidator):
            def init_metrics(self,m):
                require(self.training and self.args.half,"Native validator did not select FP16")
            def update_metrics(self,preds,data):
                require(all(torch.isfinite(v).all() for p in preds for v in p.values()),"Nonfinite half predictions")
            def gather_stats(self):pass
            def get_stats(self):return {}
            def finalize_metrics(self):pass
            def print_results(self):pass
        def sentinel(module,args):
            raise AssertionError("PDS called in native half validation")
        hook=t.ema.ema.pds_head.register_forward_pre_hook(sentinel)
        fake=SimpleNamespace(device=torch.device("cuda"),data=dict(nc=1,names={0:"crack"}),
             amp=True,ema=t.ema,model=model,args=SimpleNamespace(compile=False),
             loss_items=torch.zeros(3,device="cuda"),stopper=SimpleNamespace(possible_stop=False),
             epoch=20,epochs=200,world_size=1,
             label_loss_items=lambda loss,prefix:{prefix+"/"+str(i):float(v) for i,v in enumerate(loss)})
        val=FiniteValidator(dataloader=OneBatch(),save_dir=folder/"half_validator",
              args=dict(imgsz=160,plots=False,save_json=False,save_txt=False,workers=0,task="detect"))
        try:
            val(fake)
        finally:
            hook.remove()
        require(torch.isfinite(val.loss).all(),"Nonfinite half main validation loss")
        stripped,comparison=strip_model(t.ema.ema)
        hook=t.ema.ema.pds_head.register_forward_pre_hook(sentinel)
        try:
            backend=AutoBackend(model=deepcopy(t.ema.ema),device=torch.device("cuda"),fp16=True,verbose=False)
            backend.warmup(imgsz=(1,3,160,160))
            with torch.no_grad():
                pred=backend(b["img"][:1])[0]
            require(pred.shape==(1,300,5) and torch.isfinite(pred).all(),"AutoBackend finite prediction failed")
            require(hasattr(backend.model.model[20],"bn"),"Original LIF fusion protection lost")
        finally:
            hook.remove()
        report.update(status="PASS",native_check_amp=True,pds_precision=observed,
                      effective_updates=1,optimizer_diagnostics=t.pds_step_diagnostics,
                      native_FP16_validator_pds_calls=0,AutoBackend_fused_half_pds_calls=0,
                      LIF_BN_retained=True,deploy_from_half_roundtrip_exact=comparison["raw_output_exact"])
    except BaseException as error:
        import traceback
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(folder/"amp.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.checkpoint,a.output),indent=2))
