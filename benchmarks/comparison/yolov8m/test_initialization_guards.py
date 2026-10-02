"""Reject contradictory initialization inputs before native model construction."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from support import ROOT,configure,checked_weight,validate_checkpoint,initialization_type
configure(ROOT/'outputs/yolov8m-scratch/guard-runtime')
from adapters import ComparisonTrainer
from ultralytics.models.yolo.detect import DetectionTrainer


class GuardTests(unittest.TestCase):
    def test_coco_still_requires_verified_weights(self):
        trainer=ComparisonTrainer.__new__(ComparisonTrainer)
        trainer.comparison_identity={'initialization_type':'coco'}
        trainer.args=SimpleNamespace(resume=False,pretrained=True)
        with patch.object(DetectionTrainer,'get_model',side_effect=AssertionError('must reject before construction')):
            with self.assertRaises(ValueError):
                trainer.get_model(weights=None)
        with tempfile.TemporaryDirectory() as td:
            fake=Path(td)/'fake.pt';fake.write_bytes(b'not the pinned checkpoint')
            with self.assertRaises(ValueError):
                checked_weight(fake)

    def test_scratch_recipe_cannot_claim_pretrained(self):
        with patch('support.recipe',return_value={'pretrained':True}):
            with self.assertRaises(ValueError):
                initialization_type()

    def test_resume_requires_same_initialization_run_and_unfinished_checkpoint(self):
        identity={'initialization_type':'random','run_id':'scratch-a'}
        checkpoint={'epoch':4,'optimizer':{},'comparison_identity':identity}
        validate_checkpoint(checkpoint,identity,resume=True)
        for changed in ({'initialization_type':'coco','run_id':'scratch-a'},
                        {'initialization_type':'random','run_id':'scratch-b'}):
            with self.assertRaises(ValueError):
                validate_checkpoint({**checkpoint,'comparison_identity':changed},identity,resume=True)
        for update in ({'epoch':199},{'optimizer':None}):
            with self.assertRaises(ValueError):
                validate_checkpoint({**checkpoint,**update},identity,resume=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
