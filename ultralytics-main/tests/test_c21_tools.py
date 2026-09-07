from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from init_rtdetr_r18_lite_sala_controlled import DEFAULT_OUTPUT, verify_protected
from train_rtdetr_r18_lite_c21 import build_locked_args, actual_args_check, DEFAULT_NAME
from ultralytics.utils import YAML


class C21ToolTests(unittest.TestCase):
    def test_full_recipe_and_reject_mutations(self):
        recipe = YAML.load(ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml")
        target, rows = build_locked_args(recipe, DEFAULT_OUTPUT, DEFAULT_NAME)
        self.assertEqual(len(rows), 109)
        self.assertEqual({r["field"] for r in rows if not r["equal"]}, {"model", "name", "save_dir"})
        self.assertEqual(target["project"], recipe["project"])
        actual_args_check(target, deepcopy(target))
        for key, value in (("amp", 1), ("epochs", 100), ("project", "runs/new"), ("hsv_s", .6)):
            bad = deepcopy(recipe); bad[key] = value
            with self.assertRaises(RuntimeError): build_locked_args(bad, DEFAULT_OUTPUT, DEFAULT_NAME)
        bad = deepcopy(recipe); bad.pop("erasing")
        with self.assertRaises(RuntimeError): build_locked_args(bad, DEFAULT_OUTPUT, DEFAULT_NAME)

    def test_protected_c2_and_supervisor_exit(self):
        self.assertEqual(verify_protected()["status"], "passed")
        from c21_tmux_worker import supervise
        import json
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(supervise(folder, [sys.executable, "-c", "raise SystemExit(7)"]), 7)
            report = json.loads((Path(folder) / "process_exit_code.json").read_text())
            self.assertEqual(report["exit_code"], 7)

    def test_shape_diagnostics_and_package(self):
        import json
        import tarfile
        from unittest.mock import patch
        import torch
        from c21_results import shape_diagnostics, package
        from init_rtdetr_r18_lite_sala_controlled import sha256
        boxes = torch.tensor([[0., 0., 4., 40.], [100.,100.,140.,140.]])
        groups = {}
        shape_diagnostics({'bboxes':boxes, 'cls':torch.zeros(2)},
                          {'bboxes':boxes, 'conf':torch.tensor([.5,.1]), 'cls':torch.zeros(2)}, groups)
        self.assertEqual(groups['short_lt8']['coverage_iou75_count'], 1)
        self.assertEqual(groups['short_ge32']['coverage_iou50_count'], 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run=root/'run'; launch=root/'launch'
            val=root/'val'; test=root/'test'
            (run/'weights').mkdir(parents=True); launch.mkdir(); val.mkdir(); test.mkdir()
            (run/'weights/best.pt').write_bytes(b'fixture not a model')
            digest = sha256(run/'weights/best.pt')
            for name in ('args.yaml','results.csv'): (run/name).write_text('fixture')
            for name in ('initialization.json','audit.json','launch_plan.json','train_args.yaml','actual_train_args.yaml',
                         'preflight.json','tmux.json','exit_code.json','process_exit_code.json',
                         'console.log','bootstrap.log','allocation.jsonl'):
                (launch/name).write_text('{}')
            for split, path in (('val',val),('test',test)):
                (path/'metrics_summary.json').write_text(json.dumps({'status':'completed','split':split,'checkpoint_sha256':digest}))
                (path/'key_predictions_gt.jsonl.gz').write_bytes(b'fixture')
            with patch('c21_results.runtime_info', return_value={'fixture':True}), patch('c21_results.subprocess.check_output',return_value=b'fixture diff'):
                package(run,launch,val,test,root/'small.tar.gz')
                with tarfile.open(root/'small.tar.gz') as archive:
                    names=archive.getnames()
                    self.assertIn('training/results.csv',names)
                    self.assertIn('launch/allocation.jsonl',names)
                    self.assertFalse(any(n.endswith(('.pt','.png','.cache')) for n in names))
                with self.assertRaises(RuntimeError): package(run,launch,val,test,root/'small.tar.gz')
                data=json.loads((test/'metrics_summary.json').read_text()); data['checkpoint_sha256']='wrong'
                (test/'metrics_summary.json').write_text(json.dumps(data))
                with self.assertRaises(RuntimeError): package(run,launch,val,test,root/'wrong.tar.gz')

    def test_launch_lock_and_existing_results(self):
        import json
        from unittest.mock import patch
        from train_rtdetr_r18_lite_c21 import claim_launch, verify_claim, recheck
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run'
            plan={'target_args':{'save_dir':str(output)}, 'runtime':{'git_commit':'fixture'}}
            with patch('train_rtdetr_r18_lite_c21.recheck'), patch('train_rtdetr_r18_lite_c21.assert_no_active_launch'):
                token=claim_launch(plan)
                verify_claim(plan,token)
                with self.assertRaises(FileExistsError): claim_launch(plan)
                with self.assertRaises(RuntimeError): verify_claim(plan,'wrong')


if __name__ == '__main__':
    unittest.main()
