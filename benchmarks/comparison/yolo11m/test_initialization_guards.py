"""Reject contradictory initialization inputs before native model construction."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from support import ROOT,configure,checked_weight,validate_checkpoint,initialization_type,recipe,native_recipe,read_json,LOCK
configure(ROOT/'outputs/yolo11m_configurable_validation/guard-runtime')
from adapters import ComparisonTrainer
from ultralytics.models.yolo.detect import DetectionTrainer


class GuardTests(unittest.TestCase):
    def test_coco_still_requires_verified_weights(self):
        trainer=ComparisonTrainer.__new__(ComparisonTrainer)
        trainer.comparison_identity={'initialization_type':'coco_detection_pretrained'}
        trainer.args=SimpleNamespace(resume=False,pretrained=True)
        with patch.object(DetectionTrainer,'get_model',side_effect=AssertionError('must reject before construction')):
            with self.assertRaises(ValueError):
                trainer.get_model(weights=None)
        with tempfile.TemporaryDirectory() as td:
            fake=Path(td)/'fake.pt';fake.write_bytes(b'not the pinned checkpoint')
            with self.assertRaises(ValueError):
                checked_weight(fake)

    def test_pilot_recipe_cannot_claim_random_initialization(self):
        with patch('support.recipe',return_value={**recipe(),'initialization_type':'random','pretrained':False}):
            with self.assertRaises(ValueError):
                initialization_type()

    def test_resume_requires_same_initialization_run_and_unfinished_checkpoint(self):
        identity={'initialization_type':'coco_detection_pretrained','run_id':'pilot-a',
                  'source_sha256':read_json(LOCK)['weights']['sha256']}
        state={k:{} for k in ('model','optimizer','ema','scaler','scheduler','rng','augmentation_closed','optimizer_steps')}
        checkpoint={'epoch':4,'optimizer':{},'comparison_identity':identity,'pilot_recipe':recipe(),
                    'comparison_training_state':state,'comparison_initialization':{'source_sha256':identity['source_sha256']}}
        validate_checkpoint(checkpoint,identity,resume=True)
        for changed in ({**identity,'initialization_type':'random'}, {**identity,'run_id':'pilot-b'},
                        {**identity,'run_id':'SMOKE_other'}, {**identity,'model':'yolov5m'}):
            with self.assertRaises(ValueError):
                validate_checkpoint({**checkpoint,'comparison_identity':changed},identity,resume=True)
        for update in ({'epoch':199},{'comparison_training_state':{**state,'optimizer':None}},
                       {'comparison_training_state':{}}, {'pilot_recipe':{**recipe(),'cutmix':0.0}}):
            with self.assertRaises(ValueError):
                validate_checkpoint({**checkpoint,**update},identity,resume=True)

    def test_formal_resume_rejects_changed_native_hyperparameters(self):
        identity={'scope':'PILOT_FORMAL','source_sha256':read_json(LOCK)['weights']['sha256']}
        state={k:{} for k in ('model','optimizer','ema','scaler','scheduler','rng','augmentation_closed','optimizer_steps')}
        ckpt={'epoch':4,'comparison_identity':identity,'pilot_recipe':recipe(),'train_args':native_recipe(),
              'comparison_training_state':state,'comparison_initialization':{'source_sha256':identity['source_sha256']}}
        validate_checkpoint(ckpt,identity,resume=True)
        for field,value in (('batch',32),('lr0',.001),('amp',False),('optimizer','MuSGD'),('exist_ok',True)):
            with self.assertRaises(ValueError):
                validate_checkpoint({**ckpt,'train_args':{**native_recipe(),field:value}},identity,resume=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
