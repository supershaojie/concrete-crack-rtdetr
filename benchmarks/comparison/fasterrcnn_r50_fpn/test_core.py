"""Focused regression tests for configuration, geometry, augmentation and lifecycle guards."""
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from configuration import resolve_config,cli_overrides,yaml_mapping,freeze_config,frozen_config
from data import preflight
from data_adapter import TrainDataset,letterbox,inverse_boxes,train_loader
from engine import StepSchedule,validate_checkpoint
from export import evaluator_api,prediction_identity,result_record
from model import build_kwargs
from packaging_run import assert_allowed_file,build_archive
from support import ROOT,canonical,digest,write_json

def synthetic_data(root):
    from PIL import Image
    root=Path(root); records=[]
    for split,n in (('train',6),('val',2),('test',2)):
        (root/'images'/split).mkdir(parents=True); (root/'labels'/split).mkdir(parents=True)
        for i in range(n):
            array=np.zeros((47+i,83+i,3),np.uint8); array[:]=[25+i*15,130,200]
            Image.fromarray(array).save(root/'images'/split/(str(i)+'.png'))
            (root/'labels'/split/(str(i)+'.txt')).write_text('' if i==0 else '0 0.2 0.2 0.12 0.15\n',encoding='utf-8')
    cfg={'path':str(root),'names':{0:'crack'},'nc':1,**{s:'images/'+s for s in ('train','val','test')}}
    import yaml
    path=root/'dataset.yaml'; path.write_text(yaml.safe_dump(cfg),encoding='utf-8'); return path

class Core(unittest.TestCase):
    def test_precedence_types_unknown_and_ranges(self):
        cfg=resolve_config({'lr0':.03},{'lr0':.01}); self.assertEqual(cfg['lr0'],.01)
        self.assertFalse(resolve_config()['early_stopping'])
        self.assertEqual(cli_overrides(['lr0=1e-3'])['lr0'],.001)
        for override in ({'typo':1},{'mosaic':1.1},{'batch_size':True},{'optimizer':'AdamW'},
            {'gradient_accumulation_steps':2},{'ema':True},{'imgsz':800},{'box_score_thresh':0},
            {'initialization':'random','weights_file':'other.pt'},{'amp':'true'},{'warmup_epochs':200}):
            with self.subTest(override=override),self.assertRaises(ValueError): resolve_config(override)
        with self.assertRaises(ValueError): yaml_mapping(b'lr0: 0.1\nlr0: 0.2\n')
        with self.assertRaises(ValueError): cli_overrides(['lr0=.1','lr0=.2'])

    def test_schedule_endpoints_and_resume(self):
        cfg=resolve_config(); schedule=StepSchedule(cfg,378)
        self.assertEqual(schedule.total,75600); self.assertEqual(schedule.warmup,1890)
        for step,value in ((0,.00002),(1889,.02),(1890,.02),(75599,.0002)):
            self.assertAlmostEqual(schedule.lr(step),value,places=12)
        schedule.steps_done=1890; other=StepSchedule(cfg,378); other.load_state_dict(schedule.state_dict())
        self.assertEqual(other.lr(),schedule.lr()); other.steps_done=75600
        with self.assertRaises(ValueError): other.lr()
        no_warmup=StepSchedule(resolve_config({'warmup_epochs':0}),378)
        self.assertAlmostEqual(no_warmup.lr(0),.02); self.assertAlmostEqual(no_warmup.lr(75599),.0002)
        with self.assertRaises(ValueError): StepSchedule(cfg,377).load_state_dict(schedule.state_dict())

    def test_non_square_odd_padding_roundtrip(self):
        for h,w in ((47,83),(101,200),(200,101),(333,1000)):
            image=np.zeros((h,w,3),np.uint8); actual,g=letterbox(image,640)
            b=torch.tensor([[0.,0.,w,h],[1.,2.,w-3,h-4]])
            transformed=b.clone(); transformed[:,[0,2]]=b[:,[0,2]]*g['gain_x']+g['left']; transformed[:,[1,3]]=b[:,[1,3]]*g['gain_y']+g['top']
            self.assertEqual(actual.shape,(640,640,3)); self.assertTrue(torch.allclose(inverse_boxes(transformed,g),b,atol=1e-4))
            self.assertEqual(g['left']+g['right']+g['resized_width'],640)

    def test_checkpoint_identity_rejections(self):
        cfg=resolve_config(); identity={'run_uuid':'mine'}
        checkpoint={'format':'frcnn_configurable_checkpoint_v1','comparison_identity':identity,'model_build':build_kwargs(cfg),
            'best_epoch':2,'completed_epoch':3,'best_score':.55,'early_stopping':{'enabled':False,'patience':50,'best_epoch':2,'bad_epochs':1}}
        validate_checkpoint(checkpoint,identity,cfg)
        for mutated in ({**checkpoint,'format':'frcnn_scratch_checkpoint_v1'},{**checkpoint,'comparison_identity':{'run_uuid':'other'}},
            {**checkpoint,'best_epoch':4},{**checkpoint,'best_score':float('nan')}):
            with self.assertRaises(ValueError): validate_checkpoint(mutated,identity,cfg)
        with self.assertRaises(ValueError): validate_checkpoint(checkpoint,identity,cfg,resume=True)
        with self.assertRaises(ValueError): validate_checkpoint({},identity,cfg)

    def test_real_augmentation_empty_targets_and_worker_close(self):
        from augment_b19 import snapshot
        torch.manual_seed(42); np.random.seed(42); __import__('random').seed(42)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            base=Path(tmp); data=synthetic_data(base/'data'); run=base/'run'; run.mkdir()
            manifest=preflight(data,run,enforce_counts=False)
            cfg=resolve_config({'train_workers':2,'mosaic':1.,'mixup':1.,'cutmix':1.,'imgsz':64,
                'model_transform_min_size':64,'model_transform_max_size':64})
            ds=TrainDataset(run,manifest,cfg)
            for i in range(40):
                image,target,meta=ds[i%len(ds)]
                self.assertEqual(image.shape,(3,64,64)); self.assertTrue(image.dtype==torch.float32)
                self.assertTrue((target['labels']==1).all()); self.assertEqual(target['boxes'].shape[1],4)
            counts=snapshot(ds)
            self.assertGreater(counts['mosaic']['applied'],0); self.assertGreater(counts['mixup']['applied'],0)
            self.assertGreater(counts['cutmix']['triggered'],0); self.assertGreater(counts['cutmix']['applied'],0)
            ds.set_epoch(194); self.assertEqual(ds.active_probabilities,{'mosaic':1.,'mixup':1.,'cutmix':1.})
            # Iterate both sides using nonpersistent spawned workers to verify the close state is copied.
            for epoch in (194,195):
                ds.set_epoch(epoch); before=snapshot(ds)
                for images,targets,traces in train_loader(ds,cfg,torch.Generator().manual_seed(42)):
                    self.assertTrue(all(t['epoch']==epoch for t in traces))
                    if epoch==195: self.assertTrue(all(not(t['mosaic'] or t['mixup'] or t['cutmix']) for t in traces))
                after=snapshot(ds)
                if epoch==195:
                    for key in ('mosaic','mixup','cutmix'): self.assertEqual(before[key]['applied'],after[key]['applied'])
            # No geometric filters/mixers: legal empty source remains exactly empty.
            empty_cfg={**cfg,**{k:0. for k in ('mosaic','mixup','cutmix','degrees','translate','scale','shear','perspective')}}
            empty=TrainDataset(run,manifest,empty_cfg)[0][1]
            self.assertEqual(tuple(empty['boxes'].shape),(0,4)); self.assertEqual(tuple(empty['labels'].shape),(0,))
            self.assertTrue(empty['labels'].dtype==torch.int64)
            self.assertFalse(any((base/'data').rglob('*.cache'))); self.assertFalse(any((base/'data').rglob('*.npy')))

    def test_gt_byte_preserving_reuse_and_public_schema(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            base=Path(tmp); data=synthetic_data(base/'data'); run=base/'a'; run.mkdir()
            manifest=preflight(data,run,enforce_counts=False)
            second=base/'b'; second.mkdir(); preflight(data,second,public_coco=run/'gt',enforce_counts=False,reuse_manifest=run/'manifest.json')
            self.assertEqual((run/'gt/val.json').read_bytes(),(second/'gt/val.json').read_bytes())
            gt=json.loads((run/'gt/val.json').read_text()); cfg=resolve_config()
            identity=prediction_identity({'model':'Faster R-CNN','model_code_sha':'a'*40,'dataset_identity_sha256':manifest['dataset_identity_sha256']},'b'*64,'val',run/'gt/val.json',cfg)
            rows=[]
            for image in sorted(gt['images'],key=lambda x:x['id']):
                out={'boxes':torch.empty(0,4),'scores':torch.empty(0),'labels':torch.empty(0,dtype=torch.int64)}
                _,g=letterbox(np.zeros((image['height'],image['width'],3),np.uint8))
                rows.append(result_record(out,image,g,'val',cfg))
            metric=evaluator_api().evaluate(gt,rows,identity)
            self.assertEqual(metric['images'],2); self.assertEqual(metric['empty_prediction_images'],2)
            with self.assertRaises(ValueError): evaluator_api().evaluate(gt,rows[:1],identity)
            out={'boxes':torch.tensor([[1.,1.,5.,5.]]),'scores':torch.tensor([.5]),'labels':torch.tensor([0])}
            with self.assertRaises(ValueError): result_record(out,gt['images'][0],g,'val',cfg)

    def test_frozen_run_edits_and_clones(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            run=Path(tmp)/'frozen'; cfg=resolve_config()
            freeze_config(run,cfg,b'{}\n',{}, {},scope='SMOKE_ONLY',require_clean=False)
            self.assertEqual(frozen_config(run,check_code=False),cfg)
            (run/'resolved_config.yaml').write_text('model: tampered\n')
            with self.assertRaises(ValueError): frozen_config(run,check_code=False)

    def test_light_package_exclusion_and_real_cap_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            base=Path(tmp); txt=base/'core.json'; txt.write_text('{"real":true}\n')
            weight=base/'best.pt'; weight.write_bytes(b'weights')
            with self.assertRaises(ValueError): assert_allowed_file(weight,base)
            package=base/'ok.tar.gz'; report=build_archive(package,{'core.json':txt},{'weights_included':False})
            self.assertLess(report['bytes'],100000000)
            import tarfile
            with tarfile.open(package) as archive: self.assertNotIn('best.pt',archive.getnames())
            with self.assertRaises(ValueError): build_archive(base/'oversize.tar.gz',{'core.json':txt},{},limit=64)
            self.assertFalse((base/'oversize.tar.gz').exists()); self.assertTrue(list(base.glob('*.partial')))
            with self.assertRaises(FileExistsError): build_archive(package,{'core.json':txt},{})

if __name__=='__main__':
    (ROOT/'outputs').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
