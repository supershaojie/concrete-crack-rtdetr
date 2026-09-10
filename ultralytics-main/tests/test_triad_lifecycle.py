"""Failure-sensitive lifecycle and evidence tests, without launching training."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import tempfile
import json
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from triad_compat import ROOT,VARIANTS,YAML,write_json,SOURCE_SHA256
import train_triad_compat as train
import triad_compat_results as results


class LifecycleTests(unittest.TestCase):
    def test_all_109_recipe_fields_and_types(self):
        p=ROOT/'docs/triad_compat/c2_args.yaml'
        for variant in VARIANTS:
            args,rows=train.recipe(p,variant,ROOT/'outputs'/variant/'init.pt')
            self.assertEqual(len(rows),109)
            self.assertEqual({r['field'] for r in rows if r['changed']},{'model','name','save_dir'})
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            bad=Path(temp)/'args.yaml';cfg=YAML.load(p);cfg['batch']=16.0;YAML.save(bad,cfg)
            with self.assertRaises(RuntimeError):train.recipe(bad,'triad_v1',bad.with_suffix('.pt'))

    def test_variants_do_not_share_mutable_paths(self):
        all_paths=[train.paths(v) for v in VARIANTS]
        for key in ('run','launch','init','session','lock'):
            self.assertEqual(len(set(p[key] for p in all_paths)),7)

    def test_status_requires_token_exit_completion_and_artifacts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            r=Path(temp);p=dict(launch=r/'launch',run=r/'run',session='test-triad');p['launch'].mkdir()
            self.assertEqual(train.run_state(p),'NOT_STARTED')
            write_json(p['launch']/'launch_state.json',dict(status='DISPATCHED',token='a'))
            write_json(p['launch']/'process.json',dict(pid=100,process_token='101',token='a'))
            with patch.object(train,'process_token',return_value='101'):self.assertEqual(train.run_state(p),'RUNNING')
            with patch.object(train,'process_token',return_value='102'):self.assertEqual(train.run_state(p),'FAILED')
            write_json(p['launch']/'exit_code.json',dict(exit_code=0,token='a'))
            (p['launch']/'process_exit_code.txt').write_text('0')
            self.assertEqual(train.run_state(p),'FAILED')
            for name in ('weights/best.pt','weights/last.pt','results.csv','args.yaml','results.png'):
                f=p['run']/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'fixture')
            write_json(p['launch']/'worker_complete.json',dict(token='a',artifacts_complete=True))
            self.assertEqual(train.run_state(p),'SUCCESS')
            write_json(p['launch']/'exit_code.json',dict(exit_code=0,token='different'))
            self.assertEqual(train.run_state(p),'FAILED')

    def test_sorted_conf_policy(self):
        x=torch.tensor([[[.5,.5,.2,.2,.0001],[.3,.3,.1,.1,.9],[.4,.4,.1,.1,.8]]])
        selected,full,affected=results.native_predictions(x,640,.001)
        self.assertEqual(len(selected[0]['conf']),2)
        self.assertEqual(len(full[0]['conf']),3)
        self.assertEqual(affected,1)
        self.assertEqual(results.EVAL['seed'],42)

    def test_test_rejects_unvalidated_or_different_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            r=Path(temp);w=r/'fake.pt';w.write_bytes(b'not a checkpoint');data=r/'data.yaml';data.write_text('nc: 1')
            with patch.object(results,'runtime',return_value={'commit':'abc'}):
                with self.assertRaises(RuntimeError):results.evaluate('triad_v1',w,data,'test',r/'test')
                prior=r/'val.json';write_json(prior,dict(status='completed',split='val',checkpoint_sha256='wrong'))
                with self.assertRaises(RuntimeError):results.evaluate('triad_v1',w,data,'test',r/'test',val_report=prior)

    def test_pack_does_not_fill_missing_evaluations(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            r=Path(temp);p=dict(launch=r/'launch',run=r/'run',session='fixture')
            with patch.object(results,'paths',return_value=p),patch.object(results,'evaluate',side_effect=AssertionError('pack invoked evaluation')):
                with self.assertRaises(RuntimeError):results.package('triad_v1',r/'pack.tar.gz')
            self.assertFalse((r/'pack.tar.gz').exists())

    def test_oom_retry_is_disabled_without_changing_batch(self):
        from types import SimpleNamespace
        trainer=SimpleNamespace(args=SimpleNamespace(batch=16),_oom_retries=0)
        train.disable_oom_retry(trainer)
        self.assertEqual(trainer.args.batch,16);self.assertEqual(trainer._oom_retries,3)

    def test_start_dispatches_only_after_preflight_and_protects_repeat(self):
        from types import SimpleNamespace
        from triad_compat import build
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            r=Path(temp);data=r/'data.yaml';YAML.save(data,YAML.load(ROOT/'docs/triad_compat/c2_data.yaml'))
            p=dict(variant='triad_v1',name='fixture',run=r/'run',launch=r/'launch',init=r/'init.pt',source=r/'source.pt',
                   c2_args=ROOT/'docs/triad_compat/c2_args.yaml',session='triad-fixture',lock=r/'lock')
            args=dict(save_dir=str(p['run']),data=str(data));events=[]
            def initialize(source,output,variant):
                events.append('initialize');output.write_bytes(b'fixture');return dict(status='passed',source_sha256=SOURCE_SHA256)
            def smoke(*a,**kw):events.append('smoke');return dict(status='passed')
            def dispatch(command,**kw):
                if 'new-session' in command:events.append('dispatch')
                return SimpleNamespace(returncode=1 if 'has-session' in command else 0)
            with ExitStack() as stack:
                for module,name,value in ((train,'paths',lambda v:p),(train,'runtime',lambda:dict(commit='a'*40,dirty=False)),
                    (train,'verify_environment',lambda i:None),(train,'recipe',lambda *a:(args,[])),
                    (train,'check_det_dataset',lambda *a,**k:dict(nc=1,train=str(r),val=str(r),test=str(r))),
                    (train,'amp_resources',lambda:[]),(train,'initialize',initialize),(train,'record_source',lambda *a:None),
                    (train,'RTDETR',lambda *a:SimpleNamespace(model=build('triad_v1')))):
                    stack.enter_context(patch.object(module,name,side_effect=value))
                stack.enter_context(patch.object(train.shutil,'which',return_value='/usr/bin/tmux'))
                stack.enter_context(patch.object(train.subprocess,'run',side_effect=dispatch))
                stack.enter_context(patch('check_triad_compat.smoke',side_effect=smoke))
                stack.enter_context(patch('check_triad_compat.precision',return_value=dict(status='passed')))
                stack.enter_context(patch('check_triad_compat_gradients.run_checks',return_value=dict(status='passed')))
                train.start_direct('triad_v1')
                with self.assertRaises(RuntimeError):train.start_direct('triad_v1')
            self.assertEqual(events,['initialize','smoke','dispatch'])
            script=(p['launch']/'worker.sh').read_text()
            self.assertIn('process_exit_code.txt',script)
            self.assertIn('train_triad_compat.py',script)
            self.assertIn('--token',script)
            self.assertEqual(json.loads((p['launch']/'plan.json').read_text())['preflight'],'passed')

    def test_wrong_worker_token_cannot_overwrite_existing_exit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            launch=Path(temp);write_json(launch/'plan.json',dict(token='owned'))
            write_json(launch/'exit_code.json',dict(exit_code=0,token='owned'))
            before=(launch/'exit_code.json').read_bytes()
            with patch.object(train,'paths',return_value=dict(launch=launch)):
                with self.assertRaises(RuntimeError):train.worker('triad_v1','wrong')
            self.assertEqual((launch/'exit_code.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
