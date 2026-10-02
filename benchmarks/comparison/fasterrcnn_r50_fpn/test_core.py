"""Bounded contract tests: geometry, real augmentation workers, LR, state and public metrics."""
from __future__ import annotations

from copy import deepcopy
import gzip
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from support import ROOT, canonical, digest, read_json, recipe, sha256, write_json
from data import preflight, verify_inputs
from data_adapter import (EvalDataset, MotherHSV, TrainDataset, collate, forward_boxes,
                          inverse_boxes, letterbox, train_loader)
from engine import StepSchedule, validate_checkpoint
from export import evaluate_split, evaluator_api, prediction_identity, result_record
from model import build_kwargs, build_model, seed_all


def fixture(base, count=8):
    root, run = base/'shared', base/'run'
    run.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        (root/'images'/split).mkdir(parents=True)
        (root/'labels'/split).mkdir(parents=True)
        for i in range(count if split == 'train' else 2):
            pixels = np.full((80+i, 119+i, 3), (30+i*20, 60, 120), dtype=np.uint8)
            Image.fromarray(pixels).save(root/'images'/split/(str(i)+'.png'))
            (root/'labels'/split/(str(i)+'.txt')).write_text('0 .5 .5 .5 .5\n' if i % 2 == 0 else '')
    (root/'labels/train.cache').write_bytes(b'other-model-label-cache')
    (root/'images/train/0.npy').write_bytes(b'other-model-image-cache')
    data = base/'source.yaml'
    data.write_text('path: '+root.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [crack]\n')
    manifest = preflight(data, run, enforce_counts=False)
    return root, run, manifest


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        (ROOT/'outputs/fasterrcnn-validation/tests').mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'outputs/fasterrcnn-validation/tests')
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_scratch_structure_initialization_and_no_external_loads(self):
        seed_all()
        model, receipt = build_model()
        self.assertEqual(receipt['pretrained_tensors_loaded'], 0)
        self.assertEqual(receipt['intercepted_calls'], [])
        self.assertEqual(receipt['structure']['resnet_blocks'], [3, 4, 6, 3])
        self.assertEqual(receipt['structure']['box_head'], 'TwoMLPHead')
        self.assertEqual(receipt['structure']['frozen_batchnorm_count'], 0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(model.roi_heads.score_thresh, .001)
        self.assertEqual(model.roi_heads.detections_per_img, 300)
        self.assertEqual(model.rpn._post_nms_top_n, {'training': 2000, 'testing': 1000})
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)
        first = model.backbone.body.conv1.weight.detach().clone()
        bn = model.backbone.body.bn1
        self.assertTrue(torch.equal(bn.weight, torch.ones_like(bn.weight)))
        self.assertTrue(torch.equal(bn.bias, torch.zeros_like(bn.bias)))
        del model
        seed_all()
        second, _ = build_model()
        self.assertTrue(torch.equal(first, second.backbone.body.conv1.weight))
        with self.assertRaises(ValueError):
            build_model({**recipe(), 'weights_backbone': 'IMAGENET1K_V1'})

    def test_letterbox_full_transform_inverse_odd_non_square(self):
        seed_all()
        model, _ = build_model()
        model.eval()
        for height, width in ((101, 203), (301, 83), (100, 200)):
            pixels, geom = letterbox(np.zeros((height, width, 3), dtype=np.uint8))
            original = torch.tensor([[3.25, 4.5, width-2.75, height-7.25]])
            boxed = forward_boxes(original, geom)
            image = torch.from_numpy(pixels.transpose(2, 0, 1)).float()/255
            inputs, targets = model.transform([image], [{'boxes': boxed, 'labels': torch.tensor([1])}])
            self.assertEqual(tuple(inputs.tensors.shape), (1, 3, 640, 640))
            # TorchVision postprocess -> data inverse once -> true original pixels.
            outputs = model.transform.postprocess([{'boxes': targets[0]['boxes'], 'labels': torch.tensor([1]),
                                                   'scores': torch.tensor([.9])}], inputs.image_sizes, [(640, 640)])
            restored = inverse_boxes(outputs[0]['boxes'], geom)
            torch.testing.assert_close(restored, original, atol=3e-5, rtol=1e-6)
            im = {'id': 1, 'file_name': 'known.png', 'width': width, 'height': height}
            row = result_record(outputs[0], im, geom, 'val')
            self.assertEqual(row['predictions'][0]['category_id'], 1)
            np.testing.assert_allclose(row['predictions'][0]['bbox'], original[0].numpy(), atol=3e-5)
            expected_zero = (0-model.transform.image_mean[0])/model.transform.image_std[0]
            self.assertAlmostEqual(float(inputs.tensors[0, 0, geom['top']+2, geom['left']+2]), expected_zero, places=5)

    def test_cache_readonly_mapping_rgb_and_empty_targets(self):
        root, run, manifest = fixture(self.base)
        before = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}
        cfg = recipe()
        cfg.update({k: 0.0 for k in ('mosaic', 'mixup', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees',
                    'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr')})
        ds = TrainDataset(run, manifest, cfg)
        image, target, _ = ds[0]
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(tuple(image.shape), (3, 640, 640))
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([30, 60, 120])/255)
        torch.testing.assert_close(target['boxes'], torch.tensor([[160., 160., 480., 480.]]))
        self.assertEqual(target['labels'].dtype, torch.int64)
        self.assertEqual(target['labels'].tolist(), [1])
        self.assertEqual(ds[1][1]['boxes'].shape, (0, 4))
        self.assertEqual(ds[1][1]['labels'].shape, (0,))
        self.assertTrue(ds.cache_path.is_relative_to(run/'cache'))
        self.assertEqual(ds.cache_path, TrainDataset(run, manifest, cfg).cache_path)
        after = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        verify_inputs(run)
        (root/'labels/train/0.txt').write_text('1 .5 .5 .5 .5\n')
        with self.assertRaises(ValueError):
            verify_inputs(run)

    def test_real_worker_mosaic_mixup_and_epoch_191_close(self):
        root, run, manifest = fixture(self.base)
        cfg = {**recipe(), 'input_height': 64, 'input_width': 64, 'batch_size': 2,
               'train_workers': 2, 'mosaic': 1.0, 'mixup': 1.0}
        ds = TrainDataset(run, manifest, cfg)
        loader = train_loader(ds, cfg, torch.Generator().manual_seed(42))
        ds.set_epoch(189)
        before = [t for _, _, traces in loader for t in traces]
        self.assertTrue(all(t['mosaic'] > 0 and t['mixup'] > 0 and t['hsv'] for t in before))
        ds.set_epoch(190)
        after = [t for _, _, traces in loader for t in traces]
        self.assertTrue(all(t['mosaic'] == 0 and t['mixup'] == 0 and t['hsv'] and t['epoch'] == 190 for t in after))
        self.assertTrue(set(t['worker_pid'] for t in before).isdisjoint(t['worker_pid'] for t in after))
        self.assertFalse(loader.persistent_workers)

    def test_hsv_additive_mother_formula(self):
        pixels = np.random.default_rng(10).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        random_gain = np.array([.6, -.2, .4])
        with patch('numpy.random.uniform', return_value=random_gain.copy()):
            actual = MotherHSV(.015, .5, .35)({'img': pixels.copy()})['img']
        r = random_gain*np.array([.015, .5, .35])
        x = np.arange(256, dtype=r.dtype)
        luts = [((x+r[0]*180)%180).astype(np.uint8), np.clip(x*(r[1]+1), 0, 255).astype(np.uint8),
                np.clip(x*(r[2]+1), 0, 255).astype(np.uint8)]
        luts[1][0] = 0
        expected = cv2.cvtColor(cv2.merge([cv2.LUT(a, b) for a, b in zip(
            cv2.split(cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)), luts)]), cv2.COLOR_HSV2BGR)
        np.testing.assert_array_equal(actual, expected)

    def test_public_metrics_preserve_empty_images_and_reject_invalid_records(self):
        api = evaluator_api()
        _, geom = letterbox(np.zeros((100, 200, 3), dtype=np.uint8))
        im = {'id': 1, 'file_name': 'known.png', 'width': 200, 'height': 100}
        output = {'boxes': forward_boxes(torch.tensor([[20., 10., 80., 60.]]), geom),
                  'scores': torch.tensor([.9]), 'labels': torch.tensor([1])}
        row = result_record(output, im, geom, 'val')
        empty_im = {**im, 'id': 2}
        empty = result_record({'boxes': torch.empty(0, 4), 'scores': torch.empty(0),
                               'labels': torch.empty(0, dtype=torch.int64)}, empty_im, geom, 'val')
        ident = {'model': 'SYNTHETIC_TEST', 'model_code_sha': 'a'*40, 'checkpoint_sha256': 'b'*64,
            'dataset_identity_sha256': 'c'*64, 'evaluation_config_sha256': api.POLICY_SHA, 'postprocessing': 'known boxes'}
        gt = {'info': {'split': 'val', 'dataset_identity_sha256': 'c'*64}, 'images': [im, empty_im],
              'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [20, 10, 60, 50]}]}
        metrics = api.evaluate(gt, [row, empty], ident)
        self.assertEqual(metrics['empty_prediction_images'], 1)
        self.assertAlmostEqual(metrics['AP50'], .995)
        with self.assertRaises(ValueError):
            api.evaluate(gt, [row], ident)
        with self.assertRaises(ValueError):
            result_record({**output, 'labels': torch.tensor([0])}, im, geom, 'val')

        # A foreign prediction cache with the same GT and internally valid checksum is rejected.
        run = self.base/'cached-run'
        gt_path = run/'gt/val.json'
        write_json(gt_path, gt)
        stored = {**ident, 'run_uuid': 'own-synthetic-unit-run', 'recipe': recipe()}
        write_json(run/'identity.json', stored)
        write_json(run/'train_status.json', {'status': 'completed', 'best_sha256': 'b'*64})
        path = run/'predictions/val.jsonl.gz'
        path.parent.mkdir()
        expected = prediction_identity(stored, 'b'*64, 'val', gt_path, recipe())
        def cache(meta):
            with gzip.open(path, 'wb') as stream:
                stream.write(canonical({'type': 'metadata', 'schema_version': 1, 'identity': meta}))
                stream.write(canonical(row)); stream.write(canonical(empty))
            write_json(run/'predictions/val_complete.json', {'sha256': sha256(path), 'checkpoint_sha256': 'b'*64})
        cache(expected)
        self.assertAlmostEqual(evaluate_split(run, 'val')['AP50'], .995)
        cache({**expected, 'run_uuid': 'foreign-synthetic-unit-run'})
        with self.assertRaisesRegex(ValueError, 'another run'):
            evaluate_split(run, 'val')

    def test_step_lr_boundaries_and_resume_no_restart(self):
        cfg = recipe()
        schedule = StepSchedule(cfg, 378)
        self.assertAlmostEqual(schedule.lr(0), .00002)
        self.assertAlmostEqual(schedule.lr(1890-1), .02)
        self.assertAlmostEqual(schedule.lr(1890), .02)
        self.assertAlmostEqual(schedule.lr(75600-1), .0002)
        schedule.steps_done = 1980
        resumed = StepSchedule(cfg, 378)
        resumed.load_state_dict(schedule.state_dict())
        self.assertEqual(schedule.lr(), resumed.lr())
        with self.assertRaises(ValueError):
            StepSchedule({**cfg, 'lr0': .01}, 378).load_state_dict(schedule.state_dict())

    def test_checkpoint_rejects_other_model_run_init_and_smoke_scope(self):
        cfg = recipe()
        identity = {'model': 'Faster R-CNN (ResNet-50-FPN)', 'run_uuid': 'own', 'scope': 'FORMAL',
                    'initialization_type': 'random'}
        ckpt = {'format': 'frcnn_scratch_checkpoint_v1', 'comparison_identity': identity,
                'model_build': build_kwargs(cfg), 'completed_epoch': 3, 'best_epoch': 2, 'best_score': .25,
                'early_stopping': {'patience': 50, 'best_epoch': 2, 'bad_epochs': 1}}
        validate_checkpoint(ckpt, identity, cfg)
        for key, value in [('model', 'YOLO'), ('run_uuid', 'foreign'), ('initialization_type', 'coco'),
                           ('scope', 'SMOKE_ONLY_SYNTHETIC')]:
            with self.assertRaises(ValueError):
                validate_checkpoint({**ckpt, 'comparison_identity': {**identity, key: value}}, identity, cfg)
        with self.assertRaises(ValueError):
            validate_checkpoint(ckpt, identity, cfg, resume=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
