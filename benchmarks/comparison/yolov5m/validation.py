"""Bounded synthetic CPU checks including actual InfiniteDataLoader worker state."""
from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import numpy as np
import torch
from PIL import Image
from support import HERE, load_yaml, write_json
from utils.dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
import utils.dataloaders as dataloaders
from bench_yolov5_runtime import close_augmentation

class TraceDataset(LoadImagesAndLabels):
    def _comparison_pre_transform(self,index):
        self.pre_calls += 1
        return super()._comparison_pre_transform(index)

    def load_mosaic(self,index):
        self.mosaic_calls += 1
        return super().load_mosaic(index)

    def __getitem__(self,index):
        self.pre_calls=self.mosaic_calls=0
        calls={'hsv':0,'geometry':0}
        hsv,geo=dataloaders.augment_hsv,dataloaders.random_perspective
        def track_hsv(*args,**kwargs):
            calls['hsv']+=1
            return hsv(*args,**kwargs)
        def track_geo(*args,**kwargs):
            calls['geometry']+=1
            return geo(*args,**kwargs)
        dataloaders.augment_hsv,dataloaders.random_perspective=track_hsv,track_geo
        try:
            result=super().__getitem__(index)
        finally:
            dataloaders.augment_hsv,dataloaders.random_perspective=hsv,geo
        return torch.tensor([self.pre_calls,self.mosaic_calls,calls['hsv'],calls['geometry']]),result[0]

def probe_collate(batch):
    return torch.stack([b[0] for b in batch]),torch.stack([b[1] for b in batch])

def smoke(assets, output):
    from worker import build_initial_model,preprocess,invert
    from assets import verify
    from utils.general import init_seeds
    from utils.loss import ComputeLoss
    init_seeds(42,deterministic=True)
    output.mkdir(parents=True,exist_ok=False)
    report={'data':'synthetic only','device':'cpu','formal_640_training':'NOT_RUN'}
    # Original single-class M; true forward, official loss, and one backward at 64 only.
    model,loaded=build_initial_model(verify(assets))
    hyp=load_yaml(HERE/'hyp.yaml')
    model.hyp={**hyp,'cls':hyp['cls']/80,'obj':hyp['obj']*(64/640)**2,'label_smoothing':0.0}
    model.train()
    pred=model(torch.rand(2,3,64,64))
    loss,items=ComputeLoss(model)(pred,torch.tensor([[0.,0.,.5,.5,.25,.25],[1.,0.,.4,.4,.2,.3]]))
    assert torch.isfinite(loss).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    from utils.torch_utils import smart_optimizer
    optimizer=smart_optimizer(model,'SGD',hyp['lr0'],hyp['momentum'],hyp['weight_decay'])
    before=model.model[0].conv.weight.detach().clone()
    nonzero=sum(bool(torch.count_nonzero(p.grad)) for p in model.parameters() if p.grad is not None)
    assert nonzero>0 and all(p.requires_grad for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
    optimizer.step()
    assert not torch.equal(before,model.model[0].conv.weight)
    report['model']={**loaded,'input':[2,3,64,64],'loss':float(loss),'loss_items':items.tolist(),
                     'backward':'finite nonzero gradients','SGD_update':'actual first Conv weight changed',
                     'optimizer_type':type(optimizer).__module__+'.'+type(optimizer).__name__,
                     'nonzero_gradient_tensors':nonzero,'all_parameters_trainable':True}
    from test_cutmix import verify_reference, geometry_cases
    report['cutmix_reference_equivalence']=verify_reference()
    report['cutmix_geometry']=geometry_cases()
    # Non-square/odd-dimension input catches separate x/y gain and padding rounding.
    im=np.zeros((79,133,3),dtype=np.uint8)
    processed,transform=preprocess(im,640)
    boxes=torch.tensor([[2.125,3.25,120.875,70.5]],dtype=torch.float32)
    original=boxes.clone()
    gx,gy,left,top=transform
    boxes[:,[0,2]]=boxes[:,[0,2]]*gx+left
    boxes[:,[1,3]]=boxes[:,[1,3]]*gy+top
    invert(boxes,transform)
    assert torch.allclose(boxes,original,atol=2e-5)
    report['letterbox']={'shape':list(processed.shape),'inverse_max_abs_error':float((boxes-original).abs().max())}
    with tempfile.TemporaryDirectory(dir=output) as td:
        root=Path(td)
        (root/'images').mkdir();(root/'labels').mkdir()
        for i in range(64):
            pixels=np.full((80,112,3),(30+i*10,80,150),dtype=np.uint8)
            Image.fromarray(pixels).save(root/'images'/f'{i}.png')
            (root/'labels'/f'{i}.txt').write_text('0 .5 .5 .3 .3\n')
        dataset=TraceDataset(str(root/'images'),img_size=64,batch_size=2,augment=True,
                             hyp={**hyp,'mosaic':1.0,'mixup':1.0,'cutmix':0.0},cache_images=False)
        # MixUp must also operate on the non-Mosaic branch.
        dataset.hyp['mosaic']=0.0
        marker,_=dataset[0]
        assert marker.tolist()==[2,0,1,2],marker
        dataset.hyp['mosaic']=1.0
        generator=torch.Generator().manual_seed(42)
        loader=InfiniteDataLoader(dataset,batch_size=2,num_workers=2,collate_fn=probe_collate,
                                  worker_init_fn=seed_worker,generator=generator,shuffle=True)
        assert not close_augmentation(loader,189)
        before=torch.cat([markers for markers,_ in loader])
        assert (before[:,0]==2).all() and (before[:,1]==2).all()
        assert close_augmentation(loader,190)
        after=torch.cat([markers for markers,_ in loader])
        assert (after==torch.tensor([1,0,1,1])).all(),after
        assert not close_augmentation(loader,191)
        loader.iterator._shutdown_workers()
        report['epoch_boundary']={'workers':2,'epoch_index_189':before.tolist(),'epoch_index_190':after.tolist(),
                'marker_columns':['pre_transforms','mosaics','HSV','geometry'],
                'non_mosaic_mixup':marker.tolist(),'retained_geometry_HSV':'passed'}
        # Force each mixing probability to test the real chain and worker copies.
        dataset=TraceDataset(str(root/'images'),img_size=64,batch_size=2,augment=True,
                             hyp={**hyp,'mosaic':1.0,'mixup':1.0,'cutmix':1.0},cache_images=False)
        raw,original_hw,resized_hw=dataset.load_image(0)
        assert raw.shape[:2]==(64,64) and original_hw==(80,112) and resized_hw==(64,64)
        from b19_augment import snapshot
        loader=InfiniteDataLoader(dataset,batch_size=2,num_workers=2,collate_fn=probe_collate,
                                 worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
        active=torch.cat([m for m,_ in loader])
        assert (active==torch.tensor([3,3,1,3])).all(),active
        assert not close_augmentation(loader,189)
        assert close_augmentation(loader,190)
        shut=snapshot(dataset)
        closed=torch.cat([m for m,_ in loader])
        end=snapshot(dataset)
        assert (closed==torch.tensor([1,0,1,1])).all(),closed
        assert shut['mixup_applied']>0 and shut['cutmix_triggered']>0
        assert shut['cutmix_applied']>0,shut
        for key in ('mosaic_applied','mixup_triggered','cutmix_triggered'):
            assert end[key]==shut[key],(key,shut,end)
        loader.iterator._shutdown_workers()
        # A resumed loader starting in the final ten epochs must close immediately.
        resumed=InfiniteDataLoader(dataset,batch_size=2,num_workers=2,collate_fn=probe_collate,
                                  worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
        dataset._comparison_closed=False
        assert close_augmentation(resumed,190)
        assert (torch.cat([m for m,_ in resumed])==torch.tensor([1,0,1,1])).all()
        resumed.iterator._shutdown_workers()
        report['cutmix_chain_workers']={'workers':2,'forced_probabilities':'Mosaic/MixUp/CutMix=1 only for this smoke',
           'before_close':shut,'after_close':end,'actual_applied':shut['cutmix_applied'],
           'closed_worker_state':'passed','resume_at_190':'passed','train_stretch':'80x112 -> 64x64; no letterbox padding'}
    # Lossless public JSONL contract + empty predictions, reuse the existing evaluator fixtures.
    import sys
    sys.path.insert(0,str(HERE.parent/'evaluation'))
    from evaluate import evaluate,read_public
    from test_preparation import PreparationTests
    fixtures=PreparationTests()
    gt,ident,rows=fixtures.fixtures()
    rows[0]['predictions']=[{'category_id':1,'score':.9,'bbox':[10.,10.,30.,30.]}]
    import json
    cache=output/'synthetic_predictions.jsonl'
    cache.write_text(json.dumps({'type':'metadata','schema_version':1,'identity':ident})+'\n'
                     +'\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
    ident2,records=read_public(cache)
    metrics=evaluate(gt,records,ident2)
    assert metrics['empty_prediction_images']==1 and abs(metrics['AP50']-.995)<1e-8
    report['public_evaluator']={k:metrics[k] for k in ('precision','recall','AP50','AP75','mAP50_95')}
    write_json(output/'smoke.json',report)
    print(report)
