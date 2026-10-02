"""Focused checks for pipeline failure propagation, read-only GT and replay."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from support import read_json
from data import inspect, make_gt
from run import stage

class ControlTests(unittest.TestCase):
    def test_real_exit_codes_and_existing_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            stage(root,'ok',[sys.executable,'-c',"print('ok')"])
            with self.assertRaises(subprocess.CalledProcessError):
                stage(root,'fail',[sys.executable,'-c',"print('failure evidence'); raise SystemExit(7)"])
            s=read_json(root/'status.json')['stages']
            self.assertEqual((s['ok']['process_exit_code'],s['ok']['tee_exit_code']),(0,0))
            self.assertEqual((s['fail']['process_exit_code'],s['fail']['tee_exit_code']),(7,0))
            previous=(root/'fail.log').read_bytes()
            with self.assertRaises(FileExistsError):
                stage(root,'fail',[sys.executable,'-c',"raise SystemExit(0)"])
            self.assertEqual(previous,(root/'fail.log').read_bytes())
            # A failed tee/log creation must never start the dependent command.
            (root/'broken.log').mkdir()
            with self.assertRaises(OSError):
                stage(root,'broken',[sys.executable,'-c',"raise SystemExit(0)"])
            self.assertEqual(read_json(root/'status.json')['stages']['broken']['tee_exit_code'],1)

    def test_interrupt_writes_status(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch('run.subprocess.Popen',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    stage(root,'signal',[sys.executable,'-c','pass'])
            self.assertEqual(read_json(root/'status.json')['stages']['signal']['status'],'interrupted')

    def test_light_gt_and_cache_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ('train','val','test'):
                (root/'images'/split).mkdir(parents=True)
                (root/'labels'/split).mkdir(parents=True)
                Image.new('RGB',(133,79)).save(root/'images'/split/'a.jpg')
                (root/'labels'/split/'a.txt').write_text('0 .5 .5 .2 .4\n' if split!='test' else '')
            yaml=root/'data.yaml'
            yaml.write_text('path: /old/windows/root\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [crack]\n')
            # Proves inspection does not open image content.
            with patch('PIL.Image.open',side_effect=AssertionError('no image opens in light check')):
                manifest=inspect(yaml,root,root,enforce_counts=False)
            with patch('PIL.Image.Image.load',side_effect=AssertionError('no pixel decode for JPEG headers')):
                gt=make_gt(manifest,'val')
            self.assertEqual(gt['images'][0]['id'],2)
            self.assertEqual(gt['annotations'][0]['category_id'],1)
            self.assertEqual(gt['annotations'][0]['bbox'],[53.2,23.7,26.6,31.6])
            (root/'coco').mkdir()
            cached=root/'coco/val.json';cached.write_text(json.dumps(gt))
            (root/'dataset_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(make_gt(manifest,'val',cached)['annotations'],gt['annotations'])
            gt['images'][0]['file_name']='D:/old/a.jpg';cached.write_text(json.dumps(gt))
            with self.assertRaises(ValueError):make_gt(manifest,'val',cached)
            self.assertEqual(make_gt(manifest,'test')['annotations'],[])

if __name__=='__main__':
    unittest.main(verbosity=2)
