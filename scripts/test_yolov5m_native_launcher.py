"""Mock tmux and interpreter checks; never launches or signals a training process."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class LauncherTests(unittest.TestCase):
    def test_commands_sessions_and_same_run_protection(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():self.skipTest('Bash unavailable')
        (ROOT/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as td:
            p=Path(td);(p/'bin').mkdir();(p/'runs').mkdir()
            fake=p/'bin/tmux'
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TMUX_CALLS"\n'
                'case "$1" in\nhas-session) exit "$FAKE_PRESENT";;\nnew-session) printf "@42\\n";;\n'
                'set-option|wait-for) exit 0;;\n*) exit 99;;\nesac\n',encoding='utf-8')
            fake.chmod(0o755)
            config=p/'one.yaml';config.write_text('lr0: .003\n')
            env=dict(os.environ,PATH=str(p/'bin')+os.pathsep+os.environ['PATH'],
                FAKE_TMUX_CALLS=(p/'calls').as_posix(),FAKE_PRESENT='1',
                YOLOV5_PYTHON=Path(sys.executable).as_posix(),YOLOV5_OUTPUT_ROOT=(p/'runs').as_posix())
            def call(*args):
                return subprocess.run([bash,'scripts/autodl_yolov5m_coco_native_ft_v1.sh',*args],
                    cwd=ROOT,env=env,capture_output=True,text=True)
            result=call('start','--run-id','one','--config',config.as_posix())
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            lines=(p/'calls').read_text()
            self.assertIn('comparison-yolov5m-one',lines)
            self.assertIn('set-option -w -t @42 remain-on-exit on',lines)
            self.assertIn('--config',lines)
            self.assertNotIn('kill',lines)
            result=call('start','--run-id','two','--config',config.as_posix())
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('comparison-yolov5m-two',(p/'calls').read_text())
            env['FAKE_PRESENT']='0'
            self.assertEqual(call('start','--run-id','one','--config',config.as_posix()).returncode,2)
            env['FAKE_PRESENT']='1'
            (p/'runs/one').mkdir();(p/'runs/one/frozen.json').write_text('{}')
            self.assertEqual(call('start','--run-id','one','--config',config.as_posix()).returncode,2)
            config.unlink()
            self.assertEqual(call('resume','--run-id','one').returncode,0)
            self.assertEqual(call('finalize','--run-id','one').returncode,0)
            self.assertEqual(call('resume','--run-id','one','--config','missing.yaml').returncode,2)
            self.assertEqual(call('start','--run-id','../bad','--config','missing.yaml').returncode,2)
            self.assertEqual(call('start','--run-id','new','--set','lr0=.1').returncode,2)
            (p/'runs/one.active.lock').write_text('123')
            self.assertEqual(call('resume','--run-id','one').returncode,2)

if __name__=='__main__':unittest.main(verbosity=2)
