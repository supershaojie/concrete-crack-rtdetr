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
        root=ROOT/'outputs/yolo11m_configurable_validation'
        root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root)
        self.path=Path(self.temp.name)
        self.tee=shutil.which('tee') or 'C:/Program Files/Git/usr/bin/tee.exe'
        self.bash_env=dict(os.environ)
        if os.name=='nt':
            self.bash_env['PATH']='C:/Program Files/Git/usr/bin'+os.pathsep+self.bash_env['PATH']

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



if __name__=='__main__':
    unittest.main(verbosity=2)
