"""Small launcher/output failure tests; no model imports or training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from support import HERE, ROOT, read_json
from stage import supervise


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'outputs/fasterrcnn-validation'
        root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root)
        self.path=Path(self.temp.name)
        self.tee=shutil.which('tee') or 'C:/Program Files/Git/usr/bin/tee.exe'
        self.shell_env=dict(os.environ)
        if os.name == 'nt':
            self.shell_env['PATH']='C:/Program Files/Git/usr/bin'+os.pathsep+os.environ['PATH']

    def tearDown(self):
        self.temp.cleanup()

    def test_success_and_real_failure_codes(self):
        for code in (0,7):
            name='exit'+str(code)
            rc=supervise(self.path,name,[sys.executable,'-c',f'print("stage-{code}"); raise SystemExit({code})'],tee=self.tee)
            self.assertEqual(rc,code)
            self.assertEqual(read_json(self.path/(name+'.exit.json')),{'command_exit':code,'tee_exit':0,'signal':None})

    def test_tee_failure_is_failure(self):
        # Python cannot execute the nonexistent log filename: a deterministic failing writer.
        rc=supervise(self.path,'writer_failure',[sys.executable,'-c','pass'],tee=sys.executable)
        result=read_json(self.path/'writer_failure.exit.json')
        self.assertEqual(result['tee_exit'],2)
        self.assertEqual(rc,2)

    def test_existing_output_protected(self):
        log=self.path/'exists.log'
        log.write_text('retained')
        with self.assertRaises(FileExistsError):
            supervise(self.path,'exists',[sys.executable,'-c','raise SystemExit(42)'],tee=self.tee)
        self.assertEqual(log.read_text(),'retained')

    @unittest.skipIf(os.name=='nt','POSIX process-group signal check requires Linux; pending on Windows')
    def test_sigterm_reaches_own_child_and_records_both_codes(self):
        proc=subprocess.Popen([sys.executable,str(HERE/'stage.py'),'--directory',str(self.path),
             '--name','signal','--',sys.executable,'-c','import time; print("ready",flush=True); time.sleep(30)'])
        try:
            deadline=time.monotonic()+10
            while not (self.path/'signal.log').exists() and time.monotonic()<deadline:
                time.sleep(.05)
            proc.send_signal(signal.SIGTERM)
            self.assertEqual(proc.wait(timeout=10),143)
            record=read_json(self.path/'signal.exit.json')
            self.assertEqual(record['signal'],15)
            self.assertIsInstance(record['command_exit'],int)
            self.assertIsInstance(record['tee_exit'],int)
        finally:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=10)

    def test_shell_rejects_own_existing_session_only(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file(): self.skipTest('bash unavailable')
        bindir=self.path/'bin'; bindir.mkdir()
        mock=bindir/'tmux'
        mock.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TMUX_CALLS"\n'
                        '[[ "$1" == has-session ]] && exit 0\nexit 99\n',encoding='utf-8')
        mock.chmod(0o755)
        env=dict(self.shell_env,PATH=str(bindir)+os.pathsep+self.shell_env['PATH'],FAKE_TMUX_CALLS=(self.path/'calls').as_posix())
        result=subprocess.run([bash,'scripts/autodl_fasterrcnn_r50_fpn.sh','start','--run-id','session_guard_test'],
                              cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,73,result.stderr)
        self.assertEqual((self.path/'calls').read_text().strip(),'has-session -t =comparison-fasterrcnn-r50-fpn-scratch')

    def test_shell_existing_run_protection_before_bootstrap(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file(): self.skipTest('bash unavailable')
        root=ROOT/'outputs/fasterrcnn-r50-fpn-scratch'; root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='guard_',dir=root) as name:
            retained=Path(name)/'retained.txt'; retained.write_text('old run')
            result=subprocess.run([bash,'scripts/autodl_fasterrcnn_r50_fpn.sh','run','--run-id',Path(name).name],
                                  cwd=ROOT,env=self.shell_env,capture_output=True,text=True)
            self.assertEqual(result.returncode,73,result.stderr)
            self.assertEqual(retained.read_text(),'old run')

    def test_tmux_window_is_retained_before_releasing_launch_gate(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file(): self.skipTest('bash unavailable')
        bindir=self.path/'bin'; bindir.mkdir()
        mock=bindir/'tmux'
        mock.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TMUX_CALLS"\n'
                        'case "$1" in\n has-session) exit 1;;\n new-session) printf "@27\\n";;\n'
                        ' set-option|wait-for) exit 0;;\n *) exit 99;;\n esac\n',encoding='utf-8')
        mock.chmod(0o755)
        env=dict(self.shell_env,PATH=str(bindir)+os.pathsep+self.shell_env['PATH'],
                 FAKE_TMUX_CALLS=(self.path/'calls').as_posix())
        result=subprocess.run([bash,'scripts/autodl_fasterrcnn_r50_fpn.sh','start','--run-id','gate_test'],
                              cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        lines=(self.path/'calls').read_text().splitlines()
        self.assertIn('new-session -d -P -F #{window_id} -s comparison-fasterrcnn-r50-fpn-scratch',lines[1])
        self.assertEqual(lines[2],'set-option -w -t @27 remain-on-exit on')
        self.assertTrue(lines[3].startswith('wait-for -S fasterrcnn-r50-fpn-scratch-launch-gate_test-'))


if __name__=='__main__':
    unittest.main(verbosity=2)
