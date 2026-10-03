"""Bounded native preprocessing probes; imported only after worker.activate()."""
from pathlib import Path
import tempfile
import random
import numpy as np
import torch
import cv2
from PIL import Image
from config import resolve, native_hyp
from utils.dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
import utils.dataloaders as dataloaders
from bench_yolov5_runtime import close_augmentation

class TraceDataset(LoadImagesAndLabels):
    def load_mosaic(self, index):
        self.mosaic_calls += 1
        return super().load_mosaic(index)

    def __getitem__(self, index):
        self.mosaic_calls = 0
        calls = {'hsv':0, 'geometry':0}
        hsv, geo = dataloaders.augment_hsv, dataloaders.random_perspective
        def track_hsv(*args, **kwargs):
            calls['hsv'] += 1
            return hsv(*args, **kwargs)
        def track_geo(*args, **kwargs):
            calls['geometry'] += 1
            return geo(*args, **kwargs)
        dataloaders.augment_hsv, dataloaders.random_perspective = track_hsv, track_geo
        try:
            result = super().__getitem__(index)
        finally:
            dataloaders.augment_hsv, dataloaders.random_perspective = hsv, geo
        return torch.tensor([self.mosaic_calls, calls['hsv'], calls['geometry']]), result[0]

def probe_collate(batch):
    return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch])

def preprocessing(output):
    from worker import preprocess, invert
    from utils.augmentations import augment_hsv, random_perspective
    report = {}
    original = torch.tensor([[2.125,3.25,120.875,70.5]])
    image, transform = preprocess(np.zeros((79,133,3), dtype=np.uint8))
    boxes = original.clone()
    gx,gy,left,top = transform
    boxes[:,[0,2]] = boxes[:,[0,2]]*gx+left
    boxes[:,[1,3]] = boxes[:,[1,3]]*gy+top
    invert(boxes,transform)
    assert torch.allclose(boxes,original,atol=2e-5)
    report['export_inverse'] = {'max_abs_error':float((boxes-original).abs().max()), 'processed_shape':list(image.shape)}
    color = cv2.cvtColor(np.array([[[30,200,180]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)
    np.random.seed(42)
    gains = np.random.uniform(-1,1,3)*[.5,.2,.1]+1
    np.random.seed(42)
    actual = color.copy()
    augment_hsv(actual,.5,.2,.1)
    hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    x = np.arange(256,dtype=gains.dtype)
    expected = cv2.cvtColor(cv2.merge((cv2.LUT(h,((x*gains[0])%180).astype('uint8')),
        cv2.LUT(s,np.clip(x*gains[1],0,255).astype('uint8')),
        cv2.LUT(v,np.clip(x*gains[2],0,255).astype('uint8')))),cv2.COLOR_HSV2BGR)
    assert np.array_equal(actual,expected)
    report['HSV'] = 'actual native multiplicative LUT verified'
    image = np.zeros((64,64,3),dtype=np.uint8)
    targets = np.array([[0,10,10,20,20],[0,45,45,55,55]],dtype=np.float32)
    cropped, labels = random_perspective(image,targets,degrees=0,translate=0,scale=0,
                                        shear=0,perspective=0,border=(-16,-16))
    assert cropped.shape == (32,32,3)
    assert np.allclose(labels,[[0,0,0,4,4]])  # second box fails native retained-area filter
    report['box_crop_filter'] = {'output':labels.tolist(),'native_area_filter_dropped':1}
    with tempfile.TemporaryDirectory(dir=output) as td:
        root = Path(td)
        (root/'images').mkdir();(root/'labels').mkdir()
        for i in range(16):
            Image.new('RGB',(112,80),(40+i*3,80,150)).save(root/'images'/f'{i:02d}.png')
            (root/'labels'/f'{i:02d}.txt').write_text('0 .5 .5 .3 .3\n',encoding='utf-8')
        raw = native_hyp(resolve({'mosaic':0,'degrees':0,'translate':0,'scale':0,
                                 'fliplr':0,'hsv_h':0,'hsv_s':0,'hsv_v':0}))
        ds = LoadImagesAndLabels(str(root/'images'),img_size=64,batch_size=2,augment=True,hyp=raw)
        im, original_hw, resized_hw = ds.load_image(0)
        assert (original_hw,resized_hw) == ((80,112),(45,64))
        tensor, labels, _, _ = ds[0]
        assert tuple(tensor.shape)==(3,64,64)
        assert np.allclose(labels[0,2:].numpy(),[.5,.5,.3,.3*45/64],atol=1e-6)
        report['aspect_resize_label_alignment'] = {'original_hw':list(original_hw),
            'resized_hw':list(resized_hw),'final_hw':list(tensor.shape[1:]),'label_xywhn':labels[0,2:].tolist()}
        trace = TraceDataset(str(root/'images'),img_size=64,batch_size=2,augment=True,
                             hyp={**raw,'mosaic':1.,'mixup':1.})
        loader = InfiniteDataLoader(trace,batch_size=2,num_workers=2,collate_fn=probe_collate,
            worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
        assert not close_augmentation(loader,3,epochs=7,close=3)
        active = torch.cat([m for m,_ in loader])
        assert (active==torch.tensor([2,1,2])).all(),active
        assert close_augmentation(loader,4,epochs=7,close=3)
        closed = torch.cat([m for m,_ in loader])
        assert (closed==torch.tensor([0,1,1])).all(),closed
        assert not close_augmentation(loader,5,epochs=7,close=3)
        assert not close_augmentation(loader,6,epochs=7,close=0)
        loader.iterator._shutdown_workers()
        trace._comparison_closed=False
        resumed = InfiniteDataLoader(trace,batch_size=2,num_workers=2,collate_fn=probe_collate,
            worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
        assert close_augmentation(resumed,4,epochs=7,close=3)
        assert (torch.cat([m for m,_ in resumed])==torch.tensor([0,1,1])).all()
        resumed.iterator._shutdown_workers()
        report['workers_close_mosaic'] = {'workers':2,'epochs':7,'close_mosaic':3,'boundary_index':4,
            'active_marker':[2,1,2],'closed_marker':[0,1,1],
            'marker_columns':['Mosaic','HSV','RandomPerspective'],'resume_at_boundary':'passed','close_0':'passed'}
    return report
