"""Detection CutMix geometry, bounded training-chain activation and resize inversion."""
from copy import deepcopy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from smoke import IntegrationTests, augment, IsolatedDataset, SquarePredictor
from augment_b19 import DetectionCutMix, counters, snapshot
from support import recipe, native_recipe
from ultralytics.cfg import get_cfg
from ultralytics.utils.instance import Instances


class B19Tests(IntegrationTests):
    # Inherit fixtures without rerunning the old integration methods in this class.
    def labels(self, boxes, pixel):
        return {'img':np.full((100,100,3),pixel,dtype=np.uint8),
                'cls':np.zeros((len(boxes),1),dtype=np.float32),
                'instances':Instances(np.asarray(boxes,dtype=np.float32).reshape(-1,4)/100,
                    segments=np.zeros((0,1000,2),dtype=np.float32), bbox_format='xyxy', normalized=True)}

    def mixer(self):
        ds=SimpleNamespace(split='train',use_segments=False,use_keypoints=False,augmentation_counts=counters())
        return DetectionCutMix(ds,p=1.0)

    def test_detection_cutmix_hard_labels_clipping_and_retention(self):
        mixer=self.mixer()
        labels=self.labels([[2,2,12,12]],17)
        labels['mix_labels']=[self.labels([[35,50,50,70],[79,79,99,99],[0,0,3,3]],222)]
        with patch.object(mixer,'_rand_bbox',return_value=(40,40,80,80)):
            result=mixer._mix_transform(labels)
        np.testing.assert_array_equal(result['instances'].bboxes,[[2,2,12,12],[40,50,50,70]])
        np.testing.assert_array_equal(result['cls'],[[0],[0]])
        self.assertFalse(result['instances'].normalized)
        self.assertTrue(np.all(result['img'][40:80,40:80]==222))
        self.assertTrue(np.all(result['img'][:40]==17))
        self.assertEqual(snapshot(mixer.dataset)['cutmix']['applied'],1)

    def test_cutmix_skips_main_collision_and_insufficient_donor_overlap(self):
        for main,donor in (([[45,45,55,55]],[[50,50,60,60]]), ([[2,2,12,12]],[[79,79,99,99]])):
            mixer=self.mixer(); labels=self.labels(main,17); before=labels['img'].copy()
            labels['mix_labels']=[self.labels(donor,222)]
            with patch.object(mixer,'_rand_bbox',return_value=(40,40,80,80)):
                result=mixer._mix_transform(labels)
            np.testing.assert_array_equal(result['img'],before)
            self.assertEqual(len(result['cls']),len(main))
            self.assertEqual(snapshot(mixer.dataset)['cutmix']['skipped_geometry'],1)

    def test_real_b19_chain_probabilities_and_bounded_train_partners(self):
        cfg=get_cfg(overrides=native_recipe())
        self.assertEqual(cfg.cutmix,0.)  # Native CutMix disabled; reference adapter owns it.
        ds=self.dataset(hyp=cfg)
        report=ds.transform_report
        self.assertEqual(report['probabilities'],dict(mosaic=1.,mixup=.234,cutmix=.065,copy_paste=0.))
        self.assertEqual(report['geometry'],{k:recipe()[k] for k in ('degrees','translate','scale','shear','perspective')})
        self.assertEqual(report['hsv'],{k:recipe()[k] for k in ('hsv_h','hsv_s','hsv_v')})
        cutmix=ds.transforms.transforms[2]
        self.assertIsInstance(cutmix,DetectionCutMix)
        self.assertIs(cutmix.pre_transform,ds.transforms.transforms[0])
        self.assertNotIn(cutmix,cutmix.pre_transform.transforms)
        with patch.object(ds,'get_image_and_label',wraps=ds.get_image_and_label) as partners:
            random.seed(42); np.random.seed(42)
            for i in range(256):
                result=ds[i%len(ds)]
                self.assertTrue(torch.isfinite(result['bboxes']).all())
                self.assertTrue(((result['bboxes']>=0)&(result['bboxes']<=1)).all())
            self.assertTrue(all(0<=call.args[0]<len(ds) for call in partners.call_args_list))
        values=snapshot(ds)
        self.assertGreater(values['mixup']['applied'],0)
        self.assertGreater(values['cutmix']['triggered'],0)
        self.assertGreater(values['cutmix']['applied'],0)
        self.assertEqual(values['cutmix']['triggered'],values['cutmix']['applied']+values['cutmix']['skipped_geometry'])
        print('B19_ACTUAL_CHAIN_COUNTS',values,flush=True)

    def test_odd_rounding_uses_actual_resize_gains_and_integer_pad(self):
        predictor=SquarePredictor(overrides={'conf':.001,'iou':.7,'max_det':300,'agnostic_nms':False,'rect':False})
        predictor.imgsz=[640,640]; predictor.model=SimpleNamespace(stride=32,names={0:'crack'},end2end=True)
        original=np.zeros((333,1000,3),dtype=np.uint8)
        predictor.pre_transform([original]); geo=predictor.inverse_geometry[0]
        self.assertEqual(geo['resized'],(640,213)); self.assertEqual(geo['top'],213)
        box=np.array([100.,30.,600.,290.]); transformed=box*np.array([geo['gain_x'],geo['gain_y']]*2)+np.array([geo['left'],geo['top']]*2)
        x1,y1,x2,y2=transformed
        native=torch.tensor([[[x1,y1,x2,y2,.9,0.]]],dtype=torch.float32)
        predictor.batch=(['odd.png'],None,None)
        result=predictor.postprocess(native,torch.zeros(1,3,640,640),[original])[0]
        np.testing.assert_allclose(result.boxes.xyxy[0].numpy(),box,atol=4e-5)

    def test_padding_only_predictions_remain_false_positives(self):
        predictor=SquarePredictor(overrides={'conf':.001,'iou':.7,'max_det':300,'agnostic_nms':False,'rect':False})
        predictor.imgsz=[640,640]; predictor.model=SimpleNamespace(stride=32,names={0:'crack'},end2end=True)
        original=np.zeros((100,200,3),dtype=np.uint8)
        predictor.pre_transform([original]); predictor.batch=(['padding.png'],None,None)
        # A real positive-area NMS box entirely inside top padding must survive.
        native=torch.tensor([[[110.,30.,210.,70.,.9,0.]]])
        result=predictor.postprocess(native,torch.zeros(1,3,640,640),[original])[0]
        self.assertEqual(len(result.boxes),1)
        box=result.boxes.xyxy[0].numpy()
        self.assertLess(box[3],0); self.assertGreater(box[3],box[1])


if __name__=='__main__':
    # Only new methods here; existing integration suite has its own entry point.
    suite=unittest.TestSuite(B19Tests(name) for name in B19Tests.__dict__ if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
