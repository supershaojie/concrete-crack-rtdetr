"""Initialization, identity, environment-path and launcher contract tests."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from support import ROOT, initialization_type, validate_checkpoint, recipe
from bootstrap import executable_path


class GuardTests(unittest.TestCase):
    def test_scratch_recipe_cannot_claim_pretrained(self):
        with patch('support.recipe',return_value={'pretrained':True}):
            with self.assertRaises(ValueError): initialization_type()

    def test_identity_and_required_resume_state(self):
        identity={'model':'official_yolov13l_random_to_crack','initialization_type':'random','run_id':'run-a'}
        ckpt={'comparison_identity':identity,'epoch':2,'optimizer':{},'ema':{},'updates':1,'scaler':{},
              'comparison_best_epoch':2,'comparison_initialization_sha256':'a'*64,
              'comparison_stopper':{'best_epoch':2,'best_fitness':.5},'best_fitness':.5,
              'comparison_model_state':{},'comparison_rng':{},'comparison_scheduler':{},
              'train_results':{},'train_args':{'epochs':200,'patience':50}}
        validate_checkpoint(ckpt,identity,resume=True)
        for field,value in [('run_id','different'),('model','official_yolov8m_random_to_crack'),('initialization_type','coco')]:
            with self.assertRaises(ValueError):
                validate_checkpoint({**ckpt,'comparison_identity':{**identity,field:value}},identity,True)
        for field in ('optimizer','ema','scaler','comparison_model_state','comparison_rng','comparison_scheduler'):
            with self.assertRaises(ValueError): validate_checkpoint({**ckpt,field:None},identity,True)
        for changes in ({'epoch':199},{'comparison_best_epoch':4},{'comparison_stopper':{'best_epoch':1,'best_fitness':.5}},
                        {'epoch':51}):
            with self.assertRaises(ValueError): validate_checkpoint({**ckpt,**changes},identity,True)

    def test_interpreter_path_is_not_resolved_through_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)/'base';base.write_text('python')
            link=Path(td)/'venv'/'bin'/'python';link.parent.mkdir(parents=True)
            try: link.symlink_to(base)
            except OSError: self.skipTest('Host does not permit symlinks')
            self.assertEqual(executable_path(link),link.absolute())
            self.assertNotEqual(executable_path(link),base.resolve())

    def test_recipe_matches_verified_reference_except_model(self):
        from support import load_yaml
        prior=load_yaml(ROOT/'benchmarks/comparison/yolov8m/recipe.yaml')
        actual=recipe()
        prior.pop('model');actual.pop('model')
        self.assertEqual(prior,actual)


if __name__=='__main__': unittest.main(verbosity=2)
