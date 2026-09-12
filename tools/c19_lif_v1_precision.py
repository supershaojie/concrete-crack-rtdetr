"""Finite native epoch-validator/AutoBackend paths on one disposable updated copy."""
from copy import deepcopy
from types import SimpleNamespace
import torch

from init_c19_lif_v1 import require,ROOT,sha256
from c19_lif_v1_diagnostic import fusion_protocol,atomic_json,rng_state,restore_rng
from c19_lif_v1_cutoff import fusion_accepted
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.patches import torch_load


def native_precision_smoke(checkpoint,initial,batch,folder):
    folder.mkdir(parents=True,exist_ok=False)
    state=rng_state();report=dict(status='FAILED',formal_training='NOT_RUN',full_val_test='NOT_RUN',scope='one disposable three-step AMP smoke copy; two train images')
    try:
        model=torch_load(checkpoint,map_location='cpu')['model'].eval().cuda().float();model.nc=1
        before=initial.state_dict();after=model.state_dict()
        bbox=[k for k in after if '.dec_bbox_head.' in k and not torch.equal(after[k].cpu(),before[k].cpu()) and after[k].abs().max()>0]
        require(bbox,'Updated bbox regression not exercised')
        originally_zero=[k for k in bbox if not torch.count_nonzero(before[k])]
        require(originally_zero,'Final zero-initialized bbox projection was not updated')
        report['nonzero_bbox_parameters']=len(originally_zero);report['updated_zero_initialized_bbox']=originally_zero
        report['bbox_update_max']={k:float((after[k].cpu()-before[k].cpu()).abs().max()) for k in bbox}
        source_files=['engine/trainer.py','engine/validator.py','models/rtdetr/val.py','nn/autobackend.py']
        report['source_sha256']={p:sha256(ROOT/'ultralytics-main/ultralytics'/p) for p in source_files}
        # Real BaseValidator.__call__: native half selection, forward, native loss,
        # native RTDETR postprocess and final model.float(). Only metric aggregation
        # is replaced, since this is a two-image engineering check, not evaluation.
        sample={k:v.detach().cpu().clone() for k,v in batch.items()}
        sample['img']=(sample['img']*255).round().to(torch.uint8)
        class OneBatch:
            dataset=list(range(len(sample['img'])))
            def __len__(self):return 1
            def __iter__(self):yield deepcopy(sample)
        observed={}
        class FiniteValidator(RTDETRValidator):
            def init_metrics(self,m):
                observed.update(model_dtype=str(next(m.parameters()).dtype),half=self.args.half,
                                bn_count=sum(isinstance(v,torch.nn.BatchNorm2d) for v in m.modules()))
                require(m is model and self.training and self.args.half and next(m.parameters()).dtype==torch.float16,'Native EMA validation precision changed')
            def update_metrics(self,preds,data):
                require(data['img'].dtype==torch.float16 and len(preds)==len(sample['img']),'Native validation shape/dtype wrong')
                require(all(torch.isfinite(v).all() for p in preds for v in p.values()),'Nonfinite native half validation')
                observed['images']=len(preds);observed['input_dtype']=str(data['img'].dtype)
            def gather_stats(self):pass
            def get_stats(self):return {}
            def finalize_metrics(self):pass
            def print_results(self):pass
        validator=FiniteValidator(dataloader=OneBatch(),save_dir=folder/'epoch_validation',args=dict(imgsz=int(batch['img'].shape[-1]),plots=False,save_json=False,save_txt=False,workers=0,task='detect'))
        trainer=SimpleNamespace(device=torch.device('cuda'),data=dict(nc=1,names={0:'crack'}),amp=True,ema=SimpleNamespace(ema=model),model=model,
            args=SimpleNamespace(compile=False),loss_items=torch.zeros(3,device='cuda'),stopper=SimpleNamespace(possible_stop=False),epoch=0,epochs=200,world_size=1,
            label_loss_items=lambda loss,prefix:{prefix+'/'+str(i):float(v) for i,v in enumerate(loss)})
        bn_before=sum(isinstance(v,torch.nn.BatchNorm2d) for v in model.modules())
        losses=validator(trainer)
        require(torch.isfinite(validator.loss).all() and next(model.parameters()).dtype==torch.float32,'Native validation loss/FP32 restoration failed')
        require(observed['bn_count']==bn_before and hasattr(model.model[20],'bn'),'Epoch validation unexpectedly fused model')
        report['native_epoch_validation']=dict(status='PASSED',**observed,loss=losses,fused=False,restored_dtype='torch.float32',metrics='NOT_EVALUATED')
        # The half round-trip is native validator behavior on this disposable EMA.
        x=batch['img'][:1].cuda()
        backend=AutoBackend(model=deepcopy(model),device=torch.device('cuda'),fp16=True,verbose=False)
        with torch.no_grad():prediction=backend(x)[0]
        require(prediction.shape==(1,300,5) and torch.isfinite(prediction).all(),'AutoBackend fuse->half finite/shape failed')
        require(next(backend.model.parameters()).dtype==torch.float16 and hasattr(backend.model.model[20],'bn'),'AutoBackend precision/LIF BN changed')
        report['autobackend_fuse_half']=dict(status='PASSED',shape=list(prediction.shape),model_dtype='torch.float16',output_dtype=str(prediction.dtype),fused=True,lif_bn_retained=True)
        report['updated_FP32_fusion']=fusion_protocol(model,deepcopy(model).fuse(verbose=False),x,folder/'updated_FP32_fusion','cuda','fp32')
        require(fusion_accepted(report['updated_FP32_fusion'],'cuda','fp32'),'Updated nonzero bbox fusion failed')
        report['status']='PASSED'
    finally:
        restore_rng(state);atomic_json(folder/'precision_smoke.json',report)
    return report
