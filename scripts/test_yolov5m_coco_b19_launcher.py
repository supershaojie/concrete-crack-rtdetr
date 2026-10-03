"""Mocked tmux tests for the pilot launcher; never starts/signals training."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_session_guard_and_returned_window(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():
            self.skipTest('Bash unavailable')
        script='scripts/autodl_yolov5m_coco_b19_pilot.sh'
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as td:
            p=Path(td);(p/'bin').mkdir()
            mock=p/'bin/tmux'
            mock.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TMUX_CALLS"\n'
              'case "$1" in\nhas-session) exit "$FAKE_PRESENT";;\nnew-session) printf "@42\\n";;\n'
              'set-option|wait-for) exit 0;;\n*) exit 99;;\nesac\n',encoding='utf-8')
            mock.chmod(0o755)
            env=dict(os.environ,PATH=str(p/'bin')+os.pathsep+os.environ['PATH'],
                FAKE_TMUX_CALLS=(p/'calls').as_posix(),FAKE_PRESENT='1',
                YOLOV5_PYTHON=Path(sys.executable).as_posix())
            call=lambda:subprocess.run([bash,script,'pilot_launcher_test'],cwd=ROOT,env=env,capture_output=True,text=True)
            result=call()
            self.assertEqual(result.returncode,0,result.stderr)
            lines=(p/'calls').read_text().splitlines()
            self.assertIn('set-option -w -t @42 remain-on-exit on',lines)
            self.assertTrue(any('new-session -d -P -F #{window_id} -s comparison-yolov5m-coco-b19-pilot' in s for s in lines))
            self.assertTrue(any('autodl_yolov5m_coco_b19_pilot.sh' in s for s in lines))
            self.assertFalse(any('kill' in s for s in lines))
            (p/'calls').write_text('')
            env['FAKE_PRESENT']='0'
            result=call()
            self.assertEqual(result.returncode,2)
            self.assertEqual((p/'calls').read_text().strip(),'has-session -t =comparison-yolov5m-coco-b19-pilot')


if __name__=='__main__':
    unittest.main(verbosity=2)
