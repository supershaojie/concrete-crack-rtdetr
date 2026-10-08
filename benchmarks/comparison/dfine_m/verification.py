"""CPU regression checks for high-risk detector/data/config contracts."""
import ast
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from configuration import resolve_config,yaml_mapping,cli_overrides
from data import letterbox,inverse_boxes,validate_target,TrainDataset,fixed_collate,prepare_inputs,train_loader
from model import official_modules
from optimizer import build_optimizer,UpdateSchedule
from support import HERE,SOURCE,digest,read_json,evaluator_api,sha256
from augment_b19 import snapshot
from smoke import synthetic_data

class ConfigTests(unittest.TestCase):
    def test_precedence_and_rejections(self):
        cfg=resolve_config({'lr0':.0002},cli_overrides(['lr0=0.0003','betas=[0.8, 0.99]']))
        self.assertEqual(cfg['lr0'],.0003); self.assertEqual(cfg['betas'],[.8,.99])
        for invalid in ({'unknown':1},{'multi_scale':True},{'optimizer':'MuSGD'},{'epochs':4},{'eval_max_det':100},{'seed':True},{'device':0}):
            with self.assertRaises(ValueError): resolve_config(invalid)
        with self.assertRaises(ValueError): yaml_mapping(b'lr0: 1\nlr0: 2\n')
        with self.assertRaises(ValueError): cli_overrides(['lr0=1','lr0=2'])
    def test_schedule_endpoints_resume_and_group_coverage(self):
        model=torch.nn.Module(); model.backbone=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4))
        model.encoder=torch.nn.Linear(4,4); model.decoder=torch.nn.Linear(4,1)
        cfg=resolve_config(); opt,groups=build_optimizer(model,cfg)
        names=[n for g in groups for n in g['names']]
        self.assertEqual(set(names),set(dict(model.named_parameters()))); self.assertEqual(len(names),len(set(names)))
        self.assertTrue(all(g['weight_decay']==0 for g in groups if g['role'].endswith('zero_decay')))
        schedule=UpdateSchedule(opt,20,5,.001,.01)
        self.assertAlmostEqual(schedule.factor(0),.001); self.assertAlmostEqual(schedule.factor(4),1.)
        self.assertAlmostEqual(schedule.factor(5),1.); self.assertAlmostEqual(schedule.factor(19),.01)
        for i in range(8): schedule.apply(); schedule.advance()
        state=schedule.state_dict(); schedule2=UpdateSchedule(opt,20,5,.001,.01)
        schedule2.targets=state['targets']; schedule2.load_state_dict(state)
        self.assertEqual(schedule2.apply(),schedule.apply())

class CoordinateTests(unittest.TestCase):
    def test_round_padding_and_official_postprocessor_inverse(self):
        native,_,_=official_modules(SOURCE); post=native.DFINEPostProcessor(num_classes=1,num_top_queries=300,remap_mscoco_category=False)
        for width,height in ((321,100),(101,79),(128,96),(99,100),(641,337)):
            image=np.zeros((height,width,3),np.uint8); tensor,meta=letterbox(image)
            self.assertEqual(tuple(tensor.shape),(3,640,640)); self.assertFalse(meta['auto']); self.assertTrue(meta['scaleup'])
            original=torch.tensor([[width*.2,height*.15,width*.7,height*.8]])
            inp=original.clone(); inp[:,[0,2]]=inp[:,[0,2]]*meta['gain_x']+meta['left']; inp[:,[1,3]]=inp[:,[1,3]]*meta['gain_y']+meta['top']
            from torchvision.ops import box_convert
            box=box_convert(inp,'xyxy','cxcywh')/640
            output={'pred_logits':torch.full((1,300,1),torch.logit(torch.tensor(.6))), 'pred_boxes':box[:,None].expand(1,300,4).clone()}
            prediction=post(output,torch.tensor([[640.,640.]]))[0]
            recovered=inverse_boxes(prediction['boxes'],meta)
            torch.testing.assert_close(recovered,original.expand(300,4),rtol=0,atol=.0001)
            torch.testing.assert_close(prediction['scores'],torch.full((300,),.6)); self.assertTrue((prediction['labels']==0).all())
        _,meta=letterbox(np.zeros((100,321,3),np.uint8)); self.assertEqual(meta['bottom']-meta['top'],1)
    def test_targets_and_empty(self):
        validate_target({'labels':torch.empty(0,dtype=torch.int64),'boxes':torch.empty(0,4),'size':torch.tensor([640,640])},640)
        with self.assertRaises(ValueError): validate_target({'labels':torch.tensor([1]),'boxes':torch.tensor([[.5,.5,.2,.2]]),'size':torch.tensor([640,640])},640)

class AugmentationTests(unittest.TestCase):
    def test_reference_class_bodies_and_no_trainer_import(self):
        lock=read_json(HERE/'augmentation.lock.json'); raw=(HERE/'augmentation_reference.py').read_text()
        classes={n.name:ast.get_source_segment(raw,n) for n in ast.parse(raw).body if isinstance(n,ast.ClassDef)}
        for name,expected in lock['reference_pipeline']['verbatim_class_sha256'].items(): self.assertEqual(digest(classes[name].encode()),expected,name)
        import sys
        self.assertNotIn('ultralytics',sys.modules)
    def test_real_mixers_and_closed_new_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=synthetic_data(root/'data'); run=root/'run'; run.mkdir()
            cfg=resolve_config({'data_yaml':str(data),'enforce_counts':False,'epochs':6,'warmup_epochs':0,
                'close_mosaic':1,'mosaic':1.,'mixup':1.,'cutmix':1.,'train_workers':0})
            manifest=prepare_inputs(cfg,run); dataset=TrainDataset(manifest,cfg,0)
            for i in range(4):
                image,target=dataset[i]; validate_target(target,640); self.assertEqual(tuple(image.shape),(3,640,640))
            counts=snapshot(dataset); self.assertGreater(counts['mosaic']['applied'],0)
            self.assertGreater(counts['mixup']['applied'],0); self.assertGreater(counts['cutmix']['triggered'],0)
            closed=TrainDataset(manifest,cfg,5)
            for i in range(4): closed[i]
            self.assertTrue(closed.closed); self.assertTrue(all(x['triggered']==0 for x in snapshot(closed).values()))
            image,target=closed[3]; self.assertEqual(tuple(target['boxes'].shape),(0,4))
            self.assertEqual(tuple(fixed_collate([closed[0],closed[3]])[0].shape),(2,3,640,640))

    def test_real_workers_rebuilt_at_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=synthetic_data(root/'data'); run=root/'run'; run.mkdir()
            cfg=resolve_config({'data_yaml':str(data),'enforce_counts':False,'epochs':6,'warmup_epochs':0,
                'close_mosaic':1,'mosaic':1.,'mixup':1.,'cutmix':1.,'train_workers':2,'batch_size':2,'device':'cpu'})
            manifest=prepare_inputs(cfg,run); generator=torch.Generator().manual_seed(42)
            for epoch in (4,5):
                loader=train_loader(manifest,cfg,epoch,generator)
                count=0
                for images,targets in loader:
                    self.assertEqual(tuple(images.shape),(2,3,640,640)); count+=len(images)
                self.assertEqual(count,4)
                actual=snapshot(loader.dataset)
                if epoch==4: self.assertGreater(actual['mosaic']['applied'],0)
                else: self.assertTrue(all(v['triggered']==0 for v in actual.values()))
                self.assertFalse(loader.persistent_workers)

    def test_auto_reuse_preserves_gt_and_rejects_changed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=synthetic_data(root/'data')
            cfg=resolve_config({'data_yaml':str(source),'enforce_counts':False})
            previous=root/'outputs/yolo26m-configurable/old'; previous.mkdir(parents=True)
            first=prepare_inputs(cfg,previous)
            expected={s:sha256(previous/f'data/gt/{s}.json') for s in ('val','test')}
            current=root/'new'; current.mkdir(); second=prepare_inputs(cfg,current)
            self.assertEqual(first['dataset_identity_sha256'],second['dataset_identity_sha256'])
            self.assertIsNotNone(read_json(current/'data/reuse_receipt.json')['inventory'])
            for split in ('val','test'): self.assertEqual(expected[split],sha256(current/f'data/gt/{split}.json'))
            (root/'data/labels/train/0.txt').write_bytes(b'0 0.5 0.5 0.2 0.2\n')
            bad=root/'bad'; bad.mkdir(); explicit={**cfg,'reuse_manifest':str(previous/'data')}
            with self.assertRaises(ValueError): prepare_inputs(explicit,bad)

class EvaluationTests(unittest.TestCase):
    def test_public_policy_order_and_empty_prediction_record(self):
        api=evaluator_api()
        gt={'info':{'split':'val','dataset_identity_sha256':'synthetic'},'images':[{'id':1,'width':10,'height':10},{'id':2,'width':10,'height':10}],
            'annotations':[{'image_id':1,'category_id':1,'bbox':[1,1,4,4]}]}
        identity={'model':'dfine_m','model_code_sha':'a'*40,'checkpoint_sha256':'b'*64,'dataset_identity_sha256':'synthetic',
            'evaluation_config_sha256':api.POLICY_SHA,'postprocessing':'native one sigmoid, input640 inverse once, no NMS'}
        rows=[{'split':'val','image_id':i,'width':10,'height':10,'box_format':'xyxy','coordinate_space':'original_image_pixels',
            'predictions':[{'category_id':1,'bbox':[1,1,5,5],'score':.9}] if i==1 else []} for i in (1,2)]
        result=api.evaluate(gt,rows,identity); self.assertEqual(result['images'],2); self.assertGreater(result['AP50'],.9)
        with self.assertRaises(ValueError): api.evaluate(gt,list(reversed(rows)),identity)
        with self.assertRaises(ValueError): api.evaluate(gt,rows[:1],identity)
    def test_package_weight_rejected(self):
        from packaging_run import safe_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'bad.pth'; p.write_bytes(b'not included')
            with self.assertRaises(ValueError): safe_file(p,root)

if __name__=='__main__': unittest.main(verbosity=2)
