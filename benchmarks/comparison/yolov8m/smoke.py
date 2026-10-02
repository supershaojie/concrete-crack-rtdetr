"""Bounded CPU tests: synthetic data, two workers, and one 64x64 M-model loss/backward."""
from __future__ import annotations

import argparse
import ast
from copy import copy
import inspect
import json
import os
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # this test never allocates a GPU
from support import (HERE, ROOT, SOURCE, canonical, configure, local_lock, read_json, recipe,
                     sha256, write_json)
# Spawned workers re-execute this module; the parent's isolated runtime is inherited.
configure(Path(os.environ.get('YOLOV8M_RUNTIME', ROOT / 'outputs/yolov8m_validation/smoke_runtime')), SOURCE)
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.data import augment
from ultralytics.data.build import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.results import Results
from adapters import (ComparisonTrainer, CompleteValidator, IsolatedDataset, MotherHSV, SquarePredictor, reset_workers)
from data import preflight, verify_inputs
from export import evaluator_api, result_record
from run import load_initial_model

COUNTS = {'mosaic': 0, 'mixup': 0, 'hsv': 0}


class CountMosaic(augment.Mosaic):
    def _mix_transform(self, labels):
        COUNTS['mosaic'] += 1
        return super()._mix_transform(labels)


class CountMixUp(augment.MixUp):
    def _mix_transform(self, labels):
        COUNTS['mixup'] += 1
        return super()._mix_transform(labels)


class CountHSV(MotherHSV):
    def __call__(self, labels):
        COUNTS['hsv'] += 1
        return super().__call__(labels)


class ObservedDataset(IsolatedDataset):
    def __getitem__(self, index):
        COUNTS.update(mosaic=0, mixup=0, hsv=0)
        result = super().__getitem__(index)
        result['observed_operations'] = tuple(COUNTS[k] for k in ('mosaic', 'mixup', 'hsv'))
        result['worker_pid'] = os.getpid()
        return result


def fixture(base):
    data = base / 'shared'
    for split in ('train', 'val', 'test'):
        (data / 'images' / split).mkdir(parents=True)
        (data / 'labels' / split).mkdir(parents=True)
        for i in range(8 if split == 'train' else 2):
            pixels = np.full((32+i, 48+i, 3), 30+i*20, dtype=np.uint8)
            Image.fromarray(pixels).save(data / 'images' / split / (str(i)+'.png'))
            (data / 'labels' / split / (str(i)+'.txt')).write_text('0 0.5 0.5 0.5 0.5\n' if i % 2 == 0 else '')
    (data / 'labels/train.cache').write_bytes(b'other-framework-cache-do-not-touch')
    (data / 'images/train/0.npy').write_bytes(b'other-image-cache-do-not-touch')
    config = base / 'source.yaml'
    config.write_text('path: ' + data.as_posix() + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [crack]\n')
    run = base / 'run'
    run.mkdir()
    manifest = preflight(config, run, enforce_counts=False)
    return data, run, manifest


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        output = ROOT / 'outputs/yolov8m_validation'
        output.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=output)
        self.base = Path(self.tmp.name)
        self.data, self.run, self.manifest = fixture(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, cls=IsolatedDataset, hyp=None):
        return cls(img_path=str(self.run/'train.txt'), imgsz=64, batch_size=2, augment=True,
                   hyp=hyp or get_cfg(overrides=recipe()), rect=False, cache=False, data={'names': {0:'crack'}},
                   manifest=self.manifest, split='train', cache_root=self.run/'cache')

    def test_cache_isolation_negatives_and_no_source_mutation(self):
        before = {p.relative_to(self.data).as_posix():sha256(p) for p in self.data.rglob('*') if p.is_file()}
        ds = self.dataset()
        sample = ds[0]
        self.assertEqual(tuple(sample['img'].shape), (3,64,64))
        self.assertEqual(len(ds), 8)
        self.assertEqual(sum(len(r['cls']) == 0 for r in ds.labels), 4)
        self.assertTrue(ds.cache_path.is_relative_to(self.run/'cache'))
        ds2 = self.dataset()
        self.assertEqual(ds2.cache_path, ds.cache_path)
        after = {p.relative_to(self.data).as_posix():sha256(p) for p in self.data.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        verify_inputs(self.run)

    def test_labels_changed_after_preflight_fail(self):
        (self.data/'labels/train/0.txt').write_text('1 0.5 0.5 0.5 0.5\n')
        with self.assertRaises(ValueError):
            verify_inputs(self.run)
        with self.assertRaises(ValueError):
            self.dataset()

    def test_corrupt_jpeg_never_repaired(self):
        from data import image_size
        path = self.base/'broken.jpg'
        Image.new('RGB',(20,20)).save(path)
        path.write_bytes(path.read_bytes()[:-2])
        before = sha256(path)
        with self.assertRaisesRegex(ValueError, 'NOT be repaired'):
            image_size(path)
        self.assertEqual(before,sha256(path))

    def test_additive_hsv_matches_mother_pixel_formula(self):
        original = np.random.default_rng(13).integers(0,256,(16,16,3),dtype=np.uint8)
        gain = np.array([.6,-.2,.4])
        with patch('numpy.random.uniform', return_value=gain.copy()):
            actual = MotherHSV(.015,.5,.35)({'img': original.copy()})['img']
        r=gain*np.array([.015,.5,.35]); x=np.arange(256,dtype=r.dtype)
        h,s,v=cv2.split(cv2.cvtColor(original,cv2.COLOR_BGR2HSV))
        luts=[((x+r[0]*180)%180).astype(np.uint8),np.clip(x*(r[1]+1),0,255).astype(np.uint8),
              np.clip(x*(r[2]+1),0,255).astype(np.uint8)]
        luts[1][0]=0
        expected=cv2.cvtColor(cv2.merge([cv2.LUT(a,b) for a,b in zip([h,s,v],luts)]),cv2.COLOR_HSV2BGR)
        np.testing.assert_array_equal(actual,expected)

    def test_epoch_191_actual_worker_batches(self):
        hyp=get_cfg(overrides={**recipe(),'mosaic':1.0,'mixup':1.0})
        with patch.object(augment,'Mosaic',CountMosaic), patch.object(augment,'MixUp',CountMixUp), patch.object(augment,'RandomHSV',CountHSV):
            ds=self.dataset(ObservedDataset,hyp)
            loader=build_dataloader(ds,2,2,shuffle=False,rank=-1)
            loader.reset=lambda: reset_workers(loader)
            try:
                self.assertEqual(loader.num_workers,2)
                before=next(iter(loader))
                self.assertTrue(all(m>0 and u>0 and h>0 for m,u,h in before['observed_operations']))
                fake=ComparisonTrainer.__new__(ComparisonTrainer)
                fake.train_loader=loader; fake.args=hyp; fake.epochs=200
                # Execute the actual upstream epoch condition, without running 191 training epochs.
                tree=ast.parse(textwrap.dedent(inspect.getsource(BaseTrainer._do_train)))
                trigger=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(
                    isinstance(k,ast.Attribute) and k.attr=='_close_dataloader_mosaic' for k in ast.walk(n))
                    and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='epoch')
                code=compile(ast.fix_missing_locations(ast.Module(body=[trigger],type_ignores=[])),'upstream_epoch_trigger','exec')
                exec(code,{'self':fake,'epoch':189})
                still=next(iter(loader))
                self.assertTrue(all(m>0 and u>0 for m,u,h in still['observed_operations']))
                exec(code,{'self':fake,'epoch':190})
                after=[next(iter(loader)) for _ in range(3)]
                self.assertTrue(all(m==0 and u==0 and h>0 for b in after for m,u,h in b['observed_operations']))
                self.assertTrue(set(before['worker_pid']).isdisjoint({p for b in after for p in b['worker_pid']}))
            finally:
                if hasattr(loader.iterator,'_shutdown_workers'):
                    loader.iterator._shutdown_workers()

    def test_square_letterbox_inverse_and_public_empty_records(self):
        predictor=SquarePredictor(overrides={'conf':.001,'iou':.7,'max_det':300,'agnostic_nms':False,'rect':False})
        predictor.imgsz=[640,640]
        predictor.model=SimpleNamespace(stride=32,names={0:'crack'})
        original=np.zeros((100,200,3),dtype=np.uint8)
        self.assertEqual(predictor.pre_transform([original])[0].shape,(640,640,3))
        predictor.batch=(['fixture.png'],None,None)
        # Pixel box [20,10,80,60] -> letterbox xywh [160,272,192,160].
        native=torch.tensor([[[160.],[272.],[192.],[160.],[.9]]])
        result=predictor.postprocess(native,torch.zeros(1,3,640,640),[original])[0]
        im={'id':1,'file_name':'fixture.png','width':200,'height':100}
        row=result_record(result,im,'val')
        np.testing.assert_allclose(row['predictions'][0]['bbox'],[20,10,80,60],atol=1e-5)
        self.assertEqual(row['predictions'][0]['category_id'],1)
        empty_im={**im,'id':2,'file_name':'negative.png'}
        empty=result_record(Results(original,path='negative.png',names={0:'crack'},boxes=torch.empty(0,6)),empty_im,'val')
        api=evaluator_api()
        identity={'model':'synthetic','model_code_sha':'a'*40,'checkpoint_sha256':'b'*64,
                  'dataset_identity_sha256':'c'*64,'evaluation_config_sha256':api.POLICY_SHA,'postprocessing':'synthetic known box'}
        gt={'info':{'split':'val','dataset_identity_sha256':'c'*64},'images':[im,empty_im],
            'annotations':[{'image_id':1,'category_id':1,'bbox':[20,10,60,50]}]}
        metrics=api.evaluate(gt,[row,empty],identity)
        self.assertEqual(metrics['images'],2)
        self.assertEqual(metrics['empty_prediction_images'],1)
        self.assertAlmostEqual(metrics['AP50'],.995)
        with self.assertRaises(ValueError):
            api.evaluate(gt,[row],identity)

    def test_best_uses_full_precision_map_only(self):
        t=ComparisonTrainer.__new__(ComparisonTrainer)
        t.comparison_run=self.run; t.best_fitness=None; t.epoch=0
        class FakeValidator:
            metrics=SimpleNamespace(box=SimpleNamespace(map=.500001))
            def __call__(self,trainer): return {'fitness':.65,'metrics/mAP50-95(B)':.5}
        t.validator=FakeValidator()
        _,fitness=t.validate()
        self.assertEqual(fitness,.500001)
        t.epoch=1; t.validator.metrics.box.map=.500002
        t.validate()
        self.assertEqual(t.comparison_best_epoch,2)
        self.assertEqual(t.best_fitness,.500002)

    def test_same_run_lock(self):
        with local_lock(self.run/'.lock'):
            with self.assertRaises(RuntimeError):
                with local_lock(self.run/'.lock'):
                    pass

    def test_native_val_batch16_and_nms_without_time_truncation(self):
        trainer=ComparisonTrainer.__new__(ComparisonTrainer)
        trainer.args=SimpleNamespace(workers=8)
        trainer.build_dataset=lambda *a: SimpleNamespace()
        with patch('adapters.build_dataloader',return_value=SimpleNamespace()) as build:
            trainer.get_dataloader('fixture',batch_size=32,mode='val')
            self.assertEqual(build.call_args.args[1:3],(16,0))
        validator=CompleteValidator(args=get_cfg(overrides=recipe()))
        validator.lb=[]
        with patch('ultralytics.utils.ops.non_max_suppression',return_value=[]) as nms:
            validator.postprocess(torch.zeros(1,5,1))
            self.assertEqual(nms.call_args.kwargs['max_time_img'],float('inf'))
            self.assertFalse(nms.call_args.kwargs['agnostic'])

    def test_resume_restores_patience_state_and_refreshes_closed_workers(self):
        from ultralytics.utils.torch_utils import EarlyStopping
        trainer=ComparisonTrainer.__new__(ComparisonTrainer)
        trainer.resume=True; trainer.start_epoch=191; trainer.epochs=200; trainer.best_fitness=.5
        trainer.args=SimpleNamespace(close_mosaic=10,patience=50)
        trainer.stopper=EarlyStopping(50)
        resets=[]
        trainer.train_loader=SimpleNamespace(reset=lambda: resets.append(True))
        with patch.object(BaseTrainer,'resume_training'):
            trainer.resume_training({'comparison_best_epoch':151})
        self.assertEqual(resets,[True])
        self.assertFalse(trainer.stopper(200,.49))
        self.assertTrue(trainer.stopper(201,.49))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tests-only',action='store_true',help='Run integration checks without repeating the M forward/backward')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Smoke report exists')
    torch.set_num_threads(2)
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests))
    if not tests.wasSuccessful():
        raise SystemExit(1)
    if args.tests_only:
        write_json(args.output,{'status':'PASSED_CPU_INTEGRATION_TESTS','tests':tests.testsRun,
                               'model_forward_repeated':False,'formal_training_started':False})
        return
    model,initialization=load_initial_model()
    model.args=get_cfg(overrides=recipe())
    model.train()
    torch.manual_seed(42)
    batch={'img':torch.rand(2,3,64,64),'batch_idx':torch.tensor([0,1]),
           'cls':torch.zeros(2,1),'bboxes':torch.tensor([[.5,.5,.3,.3],[.4,.4,.2,.2]])}
    loss,items=model(batch)
    if not torch.isfinite(loss).all(): raise ValueError('Nonfinite CPU loss')
    loss.sum().backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads): raise ValueError('Invalid CPU gradients')
    write_json(args.output,{'status':'PASSED_CPU_SYNTHETIC_ONLY','tests':tests.testsRun,
        'model':initialization,'loss':float(loss.sum()),'loss_items':items.tolist(),
        'gradient_tensors':len(grads),'batch':2,'imgsz':64,'device':'cpu',
        'worker_test':{'workers':2,'epoch189_zero_based':'Mosaic+MixUp applied',
                       'epoch190_zero_based':'new worker PIDs; no Mosaic/MixUp, HSV still executes'},
        'formal_training_started':False,'full_split_inference':False})


if __name__=='__main__': main()
