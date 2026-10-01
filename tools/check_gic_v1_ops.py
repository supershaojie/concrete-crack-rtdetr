"""Focused recipe, evaluation evidence, offline packaging and lifecycle regression checks."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from gic_v1_common import ROOT,BASE,read_json,write_json,sha256,compare_recipe,load_yaml,recipe,source_identity
import torch
from ultralytics.utils import ops
import gic_v1_eval as evaluation
import gic_v1_pack as packing
import gic_v1 as lifecycle
from gic_v1_analysis import detection_pairs


class OperationsTests(unittest.TestCase):
    def test_mother_sources_and_recipe(self):
        for path in ('ultralytics-main/ultralytics/nn/modules/cbr.py','ultralytics-main/ultralytics/nn/modules/lif_down.py',
                     'ultralytics-main/ultralytics/nn/modules/head.py','ultralytics-main/ultralytics/models/rtdetr/train.py',
                     'ultralytics-main/ultralytics/utils/loss.py',
                     'ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml'):
            original=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT).replace(b'\r\n',b'\n')
            self.assertEqual(original,(ROOT/path).read_bytes().replace(b'\r\n',b'\n'),path)
        archived=load_yaml(ROOT/'docs/c19_lif_v1/resolved_formal_config.yaml')
        actual=load_yaml(ROOT/'docs/gic_v1/mother_actual_args.yaml')
        audit=compare_recipe(actual,archived)
        self.assertEqual(audit['fields_checked'],109);self.assertIsNotNone(audit['alias'])
        altered=dict(actual,batch=8)
        with self.assertRaises(RuntimeError):compare_recipe(altered,archived)
        target,diff=recipe('/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml',ROOT/'docs/gic_v1/mother_actual_args.yaml')
        self.assertEqual(target['batch'],16);self.assertEqual(target['warmup_epochs'],5)
        self.assertEqual(target['save_dir'],str(lifecycle.RUN))

    def test_sorted_conf_protocol_matches_mother_and_all_queries(self):
        source=subprocess.check_output(['git','show',BASE+':tools/c19_lif_v1_results.py'],cwd=ROOT).decode()
        tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess')
        ns={'torch':torch,'ops':ops};exec(compile(ast.Module(body=[fn],type_ignores=[]),'mother_postprocess','exec'),ns)
        torch.manual_seed(42);raw=torch.rand(2,300,5);raw[:,:,4]=torch.linspace(0,.003,300).roll(72)
        expected,_=ns['postprocess'](raw,640,.001);actual=evaluation.postprocess(raw,640,.001)
        for a,b in zip(actual,expected):
            for key in ('bboxes','conf','cls'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,check_dtype=False)
            self.assertEqual(len(a['all_scores']),300)
            torch.testing.assert_close(a['conf'],a['all_scores'][a['query_ids']])

    def test_native_tp_pairing_duplicates_empty(self):
        gt=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])
        boxes=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.],[20.,20.,30.,30.],[40.,40.,50.,50.]])
        pairs,_=detection_pairs(gt,boxes);self.assertEqual(len(pairs),2)
        self.assertEqual(len(detection_pairs(gt[:0],boxes)[0]),0)
        self.assertEqual(len(detection_pairs(gt,boxes[:0])[0]),0)

    def test_success_missing_export_never_reinfers(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';run=root/'run';folder=out/'evaluations/val/existing'
            (run/'weights').mkdir(parents=True);(run/'weights/best.pt').write_bytes(b'fixture')
            current={'sha256':'fixture'};h=sha256(run/'weights/best.pt')
            identity={'binding':current,'best_sha256':h,'protocol':evaluation.PROTOCOL,'settings':evaluation.EVAL,
                      'evaluator':evaluation.evaluator_identity()}
            write_json(out/'training_completed.json',{'binding':current,'weights':{'best':{'sha256':h}}})
            write_json(folder/'metrics.json',{'status':'COMPLETE','exit_code':0,'identity':identity,'artifacts':{}})
            with patch.multiple(evaluation,OUT=out,RUN=run),patch.object(evaluation,'binding',return_value=current),patch.object(evaluation,'RTDETR',side_effect=AssertionError('unexpected inference')):
                with self.assertRaisesRegex(RuntimeError,'reinference is forbidden'):evaluation.evaluate('val')

    def test_partial_pack_offline_with_best_readback_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';run=root/'run'
            (run/'weights').mkdir(parents=True);(run/'weights/best.pt').write_bytes(b'best fixture, not a model')
            with patch.multiple(packing,OUT=out,RUN=run):
                report=packing.package();self.assertEqual(report['status'],'PARTIAL')
                manifest=packing.verify_archive(report['path'])
                self.assertIn('training/weights/best.pt',{r['path'] for r in manifest})
                self.assertTrue(packing.package()['reused'])

    def test_status_readonly_and_interrupted_not_running(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            write_json(out/'training.json',{'status':'RUNNING','process':{'pid':12345}})
            before={p:p.read_bytes() for p in out.rglob('*') if p.is_file()}
            with patch.object(lifecycle,'OUT',out),patch.object(lifecycle,'process_identity',return_value=None),patch.object(lifecycle,'binding',side_effect=AssertionError('status touched data')):
                r=lifecycle.status();self.assertEqual(r['training']['status'],'INTERRUPTED')
            self.assertEqual(before,{p:p.read_bytes() for p in out.rglob('*') if p.is_file()})

    def test_tmux_script_retains_own_pane_and_pipeline_status(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            def fake_run(argv,**kwargs):
                return types.SimpleNamespace(returncode=1 if argv[:2]==['tmux','has-session'] else 0)
            with patch.object(lifecycle,'OUT',out),patch.object(lifecycle,'ready',return_value={'fixture':True}),patch.object(lifecycle.subprocess,'run',side_effect=fake_run):
                result=lifecycle.dispatch('finish')
            script=Path(result['folder'])/'worker.sh';s=script.read_text()
            self.assertIn('remain-on-exit on',s);self.assertIn('-p -t "$TMUX_PANE"',s)
            self.assertNotIn('kill-session',s);self.assertNotIn('set-option -g',s)
            self.assertIn('codes=("${PIPESTATUS[@]}")',s);self.assertIn('pack_codes=("${PIPESTATUS[@]}")',s)
            self.assertIn('exit "$result"',s)
            bash=Path('C:/Program Files/Git/bin/bash.exe')
            if not bash.is_file():bash=Path('/bin/bash')
            subprocess.run([str(bash),'-n'],input=s,text=True,check=True)

    def test_preflight_reuse_requires_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);artifact=out/'test.json';artifact.write_text('{}')
            record=dict(status='PASS',exit_code=0,binding={'id':1},artifacts={str(artifact):sha256(artifact)})
            write_json(out/'preflight.json',record)
            with patch.object(lifecycle,'OUT',out):
                self.assertTrue(lifecycle.successful_preflight({'id':1}))
                self.assertFalse(lifecycle.successful_preflight({'id':2}))
                artifact.write_text('changed');self.assertFalse(lifecycle.successful_preflight({'id':1}))

    def test_pipeline_failure_never_reports_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);folder=out/'dispatch'
            write_json(out/'training.json',dict(status='COMPLETE',folder=str(folder)))
            with patch.object(lifecycle,'OUT',out):
                self.assertEqual(lifecycle.state('training')['status'],'EXIT_PENDING')
                write_json(folder/'exit.json',dict(python=0,tee=1))
                self.assertEqual(lifecycle.state('training')['status'],'FAILED')
                write_json(folder/'exit.json',dict(python=1,tee=0))
                self.assertEqual(lifecycle.state('training')['status'],'FAILED')
                write_json(folder/'exit.json',dict(python=0,tee=0))
                self.assertEqual(lifecycle.state('training')['status'],'COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    torch.set_num_threads(4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OperationsTests))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                failures=[(str(t),v) for t,v in result.errors+result.failures],scope='fixtures; actual Linux tmux and official eval NOT_RUN locally')
    if args.report:write_json(args.report,report)
    sys.exit(0 if result.wasSuccessful() else 1)
