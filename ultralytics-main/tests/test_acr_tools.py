"""Recipe invariance, evaluation alignment, and non-overwriting process supervision."""
import sys
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from init_acr import verify_protected, ROOT, DEFAULT_OUTPUT
from train_acr import build_locked_args, DEFAULT_NAME, typed_equal, actual_args_check, claim_launch
from ultralytics.utils import YAML


class ToolsTests(unittest.TestCase):
    def test_full_recipe_locked(self):
        original=YAML.load(ROOT/'ultralytics-main/tests/fixtures/c2_original_args.yaml')
        target,rows=build_locked_args(original,DEFAULT_OUTPUT,DEFAULT_NAME)
        self.assertEqual(len(rows),109)
        self.assertEqual({r['field'] for r in rows if not r['equal']},{'model','name','save_dir'})
        self.assertFalse(typed_equal(True,1));actual_args_check(target,target.copy())
        altered=original.copy();altered['hsv_s']=.51
        with self.assertRaises(RuntimeError):build_locked_args(altered,DEFAULT_OUTPUT,DEFAULT_NAME)
        altered=original.copy();altered.pop('seed')
        with self.assertRaises(RuntimeError):build_locked_args(altered,DEFAULT_OUTPUT,DEFAULT_NAME)
        verify_protected()

    def test_sorted_mask_issue_and_duplicate_rules(self):
        from acr_results import postprocess_audited,duplicate_diagnostics
        x=torch.tensor([[[.5,.5,.2,.2,.01],[.5,.5,.2,.2,.9],[.5,.5,.2,.2,.8]]])
        corrected,n=postprocess_audited(x,640,.25)
        historical,_=postprocess_audited(x,640,.25,'historical')
        self.assertEqual(n,1)
        torch.testing.assert_close(corrected[0]['conf'],torch.tensor([.9,.8]))
        torch.testing.assert_close(historical[0]['conf'],torch.tensor([.8,.01]))
        self.assertTrue(torch.equal(x[0,:,4],torch.tensor([.01,.9,.8])))
        gt={'bboxes':torch.tensor([[256.,256.,384.,384.]]),'cls':torch.tensor([0.])}
        self.assertEqual(duplicate_diagnostics(gt,corrected[0])['duplicate_pairs'],1)
        gt['cls']=torch.tensor([1.])
        self.assertEqual(duplicate_diagnostics(gt,corrected[0])['duplicate_pairs'],0)

    def test_launch_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            plan={'target_args':{'save_dir':str(Path(tmp)/'new')},'runtime':{'git_commit':'test'},'report_dir':tmp}
            with patch('train_acr.recheck'),patch('train_acr.assert_no_active_launch'),patch('train_acr.combination_review',return_value={}):
                claim_launch(plan)
                with self.assertRaises(FileExistsError):claim_launch(plan)

    def test_package_exclusions_size_inventory_and_no_overwrite(self):
        from acr_results import package
        from init_acr import sha256
        import tarfile
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            root=Path(tmp);run=root/'run';launch=root/'launch';val=root/'val';test=root/'test'
            for p in (run/'weights',launch,val,test):p.mkdir(parents=True)
            (run/'weights/best.pt').write_bytes(b'unit fixture, not a model')
            for n in ('args.yaml','results.csv','results.png'):(run/n).write_bytes(b'unit fixture')
            for n in ('initialization.json','audit.json','launch_plan.json','train_args.yaml','actual_train_args.yaml',
                      'preflight.json','tmux.json','exit_code.json','process_exit_code.json','resources.json',
                      'console.log','bootstrap.log','allocation.jsonl'):(launch/n).write_text('{}',encoding='utf8')
            report={'status':'completed','split':'val','checkpoint_sha256':sha256(run/'weights/best.pt')}
            (val/'metrics_summary.json').write_text(json.dumps(report),encoding='utf8')
            (val/'key_predictions_gt.jsonl.gz').write_bytes(b'unit fixture')
            archive=root/'package.tar.gz'
            package(run,launch,val,test,archive)
            with tarfile.open(archive) as tar:
                names=tar.getnames()
                self.assertIn('MANIFEST.json',names)
                self.assertIn('evaluation/test/NOT_RUN.txt',names)
                self.assertIn('training/results.png',names)
                self.assertFalse(any(n.endswith(('.pt','.jpg','.jpeg')) for n in names))
            self.assertLessEqual(archive.stat().st_size,20*1024*1024)
            inventory=json.loads(archive.with_name(archive.name+'.inventory.json').read_text())
            self.assertEqual(inventory['sha256'],sha256(archive))
            with self.assertRaises(RuntimeError):package(run,launch,val,test,archive)
            # Direct training has its own honest setup record and never needs the old failed audit.
            (launch/'launch_plan.json').write_text(json.dumps({'launch_mode':'direct'}),encoding='utf8')
            for n in ('audit.json','preflight.json'):(launch/n).unlink()
            for n in ('direct_launch.json','training_setup.json','statistics.json'):
                (launch/n).write_text(json.dumps({'full_preflight':'not_run'}),encoding='utf8')
            direct_archive=root/'direct-package.tar.gz'
            package(run,launch,val,test,direct_archive)
            with tarfile.open(direct_archive) as tar:
                names=tar.getnames()
                self.assertIn('launch/FULL_PREFLIGHT_NOT_RUN.txt',names)
                self.assertIn('launch/training_setup.json',names)
                self.assertNotIn('launch/audit.json',names)
                self.assertNotIn('launch/preflight.json',names)
                self.assertFalse(any(n.endswith(('.pt','.jpg','.jpeg')) for n in names))

    def test_supervisor_retains_child_failure(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            result=subprocess.run([sys.executable,str(ROOT/'tools/acr_tmux_worker.py'),tmp,
                                   sys.executable,'-c','raise SystemExit(7)'],capture_output=True)
            self.assertEqual(result.returncode,7)
            record=json.loads((Path(tmp)/'process_exit_code.json').read_text())
            self.assertEqual(record['exit_code'],7)


if __name__=='__main__':unittest.main()
