"""Reject contradictory initialization inputs before native model construction."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from support import ROOT,configure,checked_weight,validate_checkpoint,initialization_type,canonical,digest
configure(ROOT/'outputs/yolo11m-scratch/guard-runtime')
from adapters import ComparisonTrainer
from ultralytics.models.yolo.detect import DetectionTrainer


class GuardTests(unittest.TestCase):
    def test_coco_is_rejected_by_scratch_branch(self):
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
        args={'epochs':200,'patience':50,'amp':True,'pretrained':False}
        checkpoint={'epoch':4,'optimizer':{},'comparison_identity':identity,
                    'checkpoint_schema':'yolo11m_scratch_fp32_resume_v1','train_args':args,
                    'comparison_expanded_args_sha256':digest(canonical(args)),
                    'model':object(),'ema':object(),'updates':4,'scaler':{'scale':65536},
                    'scheduler':{},'comparison_best_epoch':5,'best_fitness':.1,
                    'comparison_initialization_sha256':'a'*64,
                    'comparison_stopper':{'best_epoch':5,'best_fitness':.1,'patience':50}}
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
