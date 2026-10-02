"""Portable mocked-tmux and legacy process-selection tests; no real process is signalled."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('legacy_jobs',ROOT/'scripts/legacy_yolo_jobs.py')
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        base=patch.object(legacy,'BASE',ROOT/'outputs/legacy-fixture')
        base.start()
        self.addCleanup(base.stop)

    def test_target_selection_requires_model_path_action_and_run_root(self):
        for model in legacy.MODELS:
            root=legacy.BASE/('Crack_RTDETR-bench-'+model)
            row={'cwd':str(root),'argv':['python',str(root/'benchmarks/comparison'/model/'run.py'),
                                        'train','--run',str(root/'outputs'/model/'actual_run')]}
            self.assertIsNotNone(legacy.training_process(row,model))
            for action in ('export','check','summary'):
                other={**row,'argv':list(row['argv'])}
                other['argv'][2]=action
                self.assertIsNone(legacy.training_process(other,model))
            other={**row,'argv':list(row['argv'])}
            other['argv'][-1]=str(legacy.BASE/'another_experiment/output')
            self.assertIsNone(legacy.training_process(other,model))
            other={**row,'argv':['python','train.py']}
            self.assertIsNone(legacy.training_process(other,model))
            self.assertIsNone(legacy.training_process(row,'yolov8m' if model=='yolov5m' else 'yolov5m'))

    def test_completed_checkpoint_preserved_without_unpickling(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            weights=root/'run/native/weights'
            weights.mkdir(parents=True)
            for name in ('best.pt','last.pt'):
                torch.save({'epoch':7,'model':None,'optimizer':{}},weights/name)
            before={p.name:p.read_bytes() for p in weights.iterdir()}
            with patch('torch.load',side_effect=AssertionError('unpickling forbidden')):
                receipt=legacy.preserve_checkpoints(root/'run','yolov5m',root/'audit')
            self.assertEqual(receipt['last.pt']['completed_epochs'],8)
            self.assertEqual(before,{p.name:p.read_bytes() for p in weights.iterdir()})
            (weights/'last.pt').write_bytes(b'broken incomplete checkpoint')
            with patch.object(legacy.time,'sleep'):
                with self.assertRaises(RuntimeError):
                    legacy.preserve_checkpoints(root/'run','yolov5m',root/'corrupt_audit')

    def test_tmux_uses_returned_window_and_keeps_existing_session(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():
            self.skipTest('bash unavailable')
        for model in legacy.MODELS:
            script=ROOT/'scripts'/('autodl_'+model+'.sh')
            if not script.exists():
                continue
            with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as td:
                root=Path(td); bindir=root/'bin';bindir.mkdir()
                mock=bindir/'tmux'
                mock.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TMUX_CALLS"\n'
                    'case "$1" in\nhas-session) exit "$FAKE_PRESENT";;\n'
                    'new-session) printf "@42\\n";;\nset-option|wait-for) exit 0;;\n*) exit 99;;\nesac\n',
                    encoding='utf-8')
                mock.chmod(0o755)
                env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                         FAKE_TMUX_CALLS=(root/'calls').as_posix(),FAKE_PRESENT='1',
                         YOLOV5_PYTHON=Path(sys.executable).as_posix())
                args=['handoff_test'] if model=='yolov5m' else ['start','--run-id','handoff_test']
                result=subprocess.run([bash,script.relative_to(ROOT).as_posix(),*args],env=env,cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                calls=(root/'calls').read_text().splitlines()
                self.assertIn('set-option -w -t @42 remain-on-exit on',calls)
                self.assertTrue(any('new-session -d -P -F #{window_id}' in line for line in calls))
                self.assertFalse(any('kill' in line for line in calls))
                (root/'calls').write_text('')
                env['FAKE_PRESENT']='0'
                result=subprocess.run([bash,script.relative_to(ROOT).as_posix(),*args],env=env,cwd=ROOT,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual((root/'calls').read_text().strip(),'has-session -t =comparison-'+model+'-scratch')


if __name__=='__main__':
    (ROOT/'outputs').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
