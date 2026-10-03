"""Small configuration, real optimizer/dataset and launcher isolation checks."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import (ASSET, ROOT, SOURCE, canonical, configure, digest, load_yaml, native_recipe,
                     read_json, recipe, sha256, validate_checkpoint, write_json)
from configuration import DEFAULT_CONFIG, freeze_config, frozen_config, resolve_config, safe_run_id, yaml_mapping
configure(ROOT / 'outputs/yolov8m-configurable-validation/unit-runtime', SOURCE)
import numpy as np
from PIL import Image
import torch
from ultralytics.cfg import get_cfg
from adapters import ComparisonTrainer
from data import preflight, reuse_preflight, verify_inputs
from run import summary


class ConfigurableTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / 'outputs/yolov8m-configurable-validation'
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def freeze(self, name, raw):
        run = self.base / name
        config = freeze_config(run, raw, {'python':str(Path(sys.executable).resolve())}, name, require_clean=False)
        return run, config

    def test_default_equals_every_pilot_value(self):
        self.assertEqual(resolve_config(load_yaml(DEFAULT_CONFIG)), recipe())
        self.assertEqual(resolve_config({}), recipe())
        get_cfg(overrides=native_recipe(resolve_config({})))

    def test_strict_schema_types_ranges_and_combinations(self):
        for update in ({'erasing':0}, {'auto_augment':None}, {'beta2':.9}, {'optimizer':'auto'},
                       {'optimizer':'MuSGD'}, {'batch':8}, {'device':'0'}, {'amp':False},
                       {'optimizer':'AdamW','warmup_momentum':.5},
                       {'copy_paste':.1}, {'bgr':.1}, {'epochs':True}, {'workers':1.0},
                       {'cos_lr':'false'}, {'lr0':float('nan')}, {'cutmix':1.1},
                       {'momentum':0}, {'epochs':3}, {'close_mosaic':201},
                       {'warmup_epochs':201}, {'box':0,'cls':0,'dfl':0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                resolve_config(update)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            yaml_mapping(b'lr0: 0.01\nlr0: 0.02\n')
        with self.assertRaises(ValueError):
            yaml_mapping(b'- not-a-mapping')
        for name in ('../run', 'a.b', '-run', 'a/b', 'a b', 'x'*81):
            with self.assertRaises(ValueError):
                safe_run_id(name)
        self.assertEqual(safe_run_id('v8_cfg-02'), 'v8_cfg-02')

    def test_frozen_candidates_ignore_source_moves_and_other_candidates(self):
        candidate = self.base/'a.yaml'
        candidate.write_bytes(b'lr0: 0.012\ncutmix: 0.2\n')
        a, ca = self.freeze('candidate_a', candidate.read_bytes())
        candidate.write_bytes(b'lr0: 0.003\ncutmix: 0.6\n')
        b, cb = self.freeze('candidate_b', candidate.read_bytes())
        candidate.rename(self.base/'moved.yaml')
        self.assertEqual(frozen_config(a,check_code=False),ca)
        self.assertEqual(frozen_config(b,check_code=False),cb)
        self.assertNotEqual(read_json(a/'config_identity.json')['config_sha256'],
                            read_json(b/'config_identity.json')['config_sha256'])
        self.assertEqual((a/'user_config.yaml').read_bytes(),b'lr0: 0.012\ncutmix: 0.2\n')
        with self.assertRaises(FileExistsError):
            freeze_config(a,b'lr0: 0.1\n',{},'candidate_a',require_clean=False)
        (a/'resolved_config.yaml').write_bytes((b/'resolved_config.yaml').read_bytes())
        with self.assertRaisesRegex(ValueError,'Frozen run artifact'):
            frozen_config(a,check_code=False)

    def make_data(self):
        root = self.base/'data'
        for split in ('train','val','test'):
            (root/'images'/split).mkdir(parents=True)
            (root/'labels'/split).mkdir(parents=True)
            for index in range(4 if split=='train' else 2):
                Image.fromarray(np.random.default_rng(index).integers(0,256,(71,93,3),dtype=np.uint8)).save(
                    root/'images'/split/(str(index)+'.jpg'))
                (root/'labels'/split/(str(index)+'.txt')).write_text('0 .5 .5 .3 .3\n' if index%2==0 else '')
        import yaml
        data = self.base/'data.yaml'
        data.write_text(yaml.safe_dump({'path':root.as_posix(),'names':{0:'crack'},
                                       **{s:'images/'+s for s in ('train','val','test')}}),encoding='utf-8')
        old = self.base/'inputs'; old.mkdir()
        manifest = preflight(data,old,root,enforce_counts=False)
        return root,data,old,manifest

    def test_two_frozen_configs_reach_real_optimizer_and_native_dataset(self):
        root,data,old,manifest = self.make_data()
        a,ca = self.freeze('real_a', b'lr0: 0.012\ncutmix: 0.2\ntranslate: 0.11\nnbs: 40\n')
        b,cb = self.freeze('real_b', b'optimizer: AdamW\nlr0: 0.003\ncutmix: 0.6\ntranslate: 0.23\nnbs: 40\n')
        for run in (a,b):
            reuse_preflight(old,data,run,root,enforce_counts=False)
            verify_inputs(run)
        for run,expected in ((a,ca),(b,cb)):
            config = frozen_config(run,check_code=False)
            trainer = ComparisonTrainer.__new__(ComparisonTrainer)
            trainer.comparison_run=run; trainer.comparison_manifest=manifest
            trainer.comparison_config=config; trainer.comparison_reuse_cache=None
            trainer.args=get_cfg(overrides={**native_recipe(config),'imgsz':64})
            trainer.data={'names':{0:'crack'}}
            ds=trainer.build_dataset(str(run/'train.txt'),'train',16)
            self.assertEqual(ds.transform_report['probabilities']['cutmix'],expected['cutmix'])
            self.assertEqual(ds.transform_report['geometry']['translate'],expected['translate'])
            model=torch.nn.Sequential(torch.nn.Conv2d(3,4,3),torch.nn.BatchNorm2d(4),torch.nn.Conv2d(4,1,1))
            decay=config['weight_decay']*16*max(round(config['nbs']/16),1)/config['nbs']
            optimizer=trainer.build_optimizer(model,name=config['optimizer'],lr=config['lr0'],
                                                momentum=config['momentum'],decay=decay)
            self.assertEqual(type(optimizer).__name__,config['optimizer'])
            self.assertEqual([g['lr'] for g in optimizer.param_groups],[config['lr0']]*3)
            self.assertEqual([g['weight_decay'] for g in optimizer.param_groups],[0,decay,0])
            if config['optimizer']=='AdamW':
                self.assertEqual(optimizer.param_groups[0]['betas'],(config['momentum'],.999))
            ds.close_mosaic(deepcopy(trainer.args))
            self.assertEqual(ds.transform_report['probabilities'],dict(mosaic=0,mixup=0,cutmix=0,copy_paste=0))
        # Native Adam is also accepted without adding non-native beta/epsilon overrides.
        cfg=resolve_config({'optimizer':'Adam','momentum':.8})
        optimizer=trainer.build_optimizer(model,name=cfg['optimizer'],lr=cfg['lr0'],momentum=cfg['momentum'])
        self.assertIs(type(optimizer),torch.optim.Adam)

    def test_dynamic_epoch_checkpoint_guards_and_no_raw_rescaling(self):
        cfg=resolve_config({'epochs':12,'close_mosaic':3,'warmup_epochs':1,'optimizer':'AdamW',
                            'nbs':40,'lr0':.002,'weight_decay':.0007})
        identity={'scope':'CONFIGURABLE_FORMAL','source_sha256':'a'*64}
        state={k:{} for k in ('model','optimizer','ema','scaler','scheduler','rng','augmentation_closed','optimizer_steps')}
        ckpt={'epoch':10,'comparison_identity':identity,'pilot_recipe':cfg,'train_args':native_recipe(cfg),
              'comparison_training_state':state,'comparison_initialization':{'source_sha256':'a'*64}}
        validate_checkpoint(ckpt,identity,resume=True,config=cfg)
        self.assertEqual(ckpt['train_args']['weight_decay'],.0007)
        for update in ({'epoch':11},{'pilot_recipe':recipe()},
                       {'train_args':{**native_recipe(cfg),'weight_decay':.00056}},
                       {'train_args':{**native_recipe(cfg),'optimizer':'SGD'}}):
            with self.assertRaises(ValueError):
                validate_checkpoint({**ckpt,**update},identity,resume=True,config=cfg)
        # patience=0 retains native disabled-early-stopping behavior.
        from ultralytics.utils.torch_utils import EarlyStopping
        from ultralytics.engine.trainer import BaseTrainer
        trainer=ComparisonTrainer.__new__(ComparisonTrainer)
        trainer.resume=True; trainer.comparison_config=cfg; trainer.comparison_identity=identity
        trainer.args=SimpleNamespace(close_mosaic=0,patience=0)
        trainer.epochs=12; trainer.start_epoch=3; trainer.best_fitness=.4
        trainer.stopper=EarlyStopping(0)
        with patch.object(BaseTrainer,'resume_training'), patch('adapters.validate_checkpoint'), patch.object(
                ComparisonTrainer,'restore_training_state'):
            trainer.resume_training({'comparison_best_epoch':1})
        self.assertFalse(trainer.stopper.possible_stop)

    def test_dynamic_close_trigger_rebuilds_actual_data_workers(self):
        import ast
        import inspect
        import textwrap
        from smoke import ObservedDataset, CountHSV, augment
        from adapters import reset_workers
        from ultralytics.data.build import build_dataloader
        from ultralytics.engine.trainer import BaseTrainer
        root,data,run,manifest=self.make_data()
        cfg=resolve_config({'epochs':37,'close_mosaic':5,'mosaic':1.0,'mixup':1.0,'cutmix':1.0})
        hyp=get_cfg(overrides=native_recipe(cfg))
        with patch.object(augment,'RandomHSV',CountHSV):
            ds=ObservedDataset(img_path=str(run/'train.txt'),imgsz=64,batch_size=2,augment=True,
                hyp=hyp,rect=False,cache=False,data={'names':{0:'crack'}},manifest=manifest,
                split='train',cache_root=run/'cache',cutmix_probability=cfg['cutmix'])
            loader=build_dataloader(ds,2,2,shuffle=False,rank=-1)
            loader.reset=lambda:reset_workers(loader)
            try:
                before=next(iter(loader))
                self.assertTrue(all(m>0 and u>0 and c>0 and h>0 for m,u,c,h in before['observed_operations']))
                fake=ComparisonTrainer.__new__(ComparisonTrainer)
                fake.train_loader=loader; fake.args=hyp; fake.epochs=cfg['epochs']; fake.comparison_run=run
                tree=ast.parse(textwrap.dedent(inspect.getsource(BaseTrainer._do_train)))
                trigger=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(
                    isinstance(k,ast.Attribute) and k.attr=='_close_dataloader_mosaic' for k in ast.walk(n))
                    and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='epoch')
                code=compile(ast.fix_missing_locations(ast.Module(body=[trigger],type_ignores=[])),'native_trigger','exec')
                exec(code,{'self':fake,'epoch':31})
                self.assertFalse(ds.augmentation_closed)
                exec(code,{'self':fake,'epoch':32})
                after=[next(iter(loader)) for _ in range(3)]
                self.assertTrue(all(m==u==c==0 and h>0 for batch in after for m,u,c,h in batch['observed_operations']))
                self.assertTrue(set(before['worker_pid']).isdisjoint({p for b in after for p in b['worker_pid']}))
                self.assertEqual(read_json(run/'augmentation_closed.json')['zero_based_epoch'],32)
                write_json(ROOT/'outputs/yolov8m-configurable-validation/dynamic_workers.json',
                    {'scope':'SMOKE_ONLY','workers':2,'epochs':37,'close_mosaic':5,'zero_based_trigger':32,
                     'new_worker_pids':True,'mosaic_mixup_cutmix_after_close':0,'HSV_remains_active':True})
            finally:
                if hasattr(loader.iterator,'_shutdown_workers'):
                    loader.iterator._shutdown_workers()

    def test_tuning_without_test_is_normal_and_test_failure_is_separate(self):
        run,_=self.freeze('summary_run',b'{}')
        write_json(run/'train_status.json',{'status':'completed','completed_epoch':137,'best_epoch':87,
                                           'stop_reason':'patience','exit_code':0})
        metrics=run/'metrics/val.json'; write_json(metrics,{'display_percent':{'mAP50-95':37.4}})
        write_json(run/'evaluate_val_status.json',{'status':'completed','metrics':str(metrics)})
        row=summary(run)
        self.assertEqual(row['status'],'tuning_completed')
        self.assertEqual(row['final_test_status'],'not_requested')
        write_json(run/'evaluate_test_status.json',{'status':'failed','error':'synthetic export failure'})
        row=summary(run)
        self.assertEqual(row['training_status'],'completed')
        self.assertEqual(row['tuning_status'],'completed')
        self.assertEqual(row['final_test_status'],'failed')
        write_json(run/'train_status.json',{'status':'interrupted','exit_code':130})
        write_json(run/'training_progress.json',{'completed_epoch':17,'best_epoch':14})
        row=summary(run)
        self.assertEqual(row['completed_epoch'],17)
        self.assertEqual(row['training_status'],'interrupted')

    def test_public_metric_cache_reuse_and_mismatch_rejection(self):
        import gzip
        from export import evaluate_split,evaluator_api
        run,_=self.freeze('cache_run',b'{}')
        identity={'model':'synthetic','model_code_sha':'a'*40,'checkpoint_sha256':'b'*64,
                  'dataset_identity_sha256':'c'*64,'evaluation_config_sha256':evaluator_api().POLICY_SHA,
                  'postprocessing':'synthetic known empty record','config_sha256':'d'*64}
        gt={'info':{'split':'val','dataset_identity_sha256':'c'*64},
            'images':[{'id':1,'file_name':'empty.jpg','width':100,'height':80}],
            'categories':[{'id':1,'name':'crack'}],'annotations':[]}
        write_json(run/'gt/val.json',gt)
        prediction=run/'predictions/val.jsonl.gz'; prediction.parent.mkdir()
        with gzip.open(prediction,'wt',encoding='utf-8') as stream:
            stream.write(canonical({'type':'metadata','schema_version':1,'identity':identity}).decode())
            stream.write(canonical({'split':'val','image_id':1,'image':'empty.jpg','width':100,'height':80,
                'box_format':'xyxy','coordinate_space':'original_image_pixels','predictions':[]}).decode())
        write_json(run/'predictions/val_complete.json',{'sha256':sha256(prediction)})
        with patch('export.export_identity',return_value=(identity,{},None)):
            first=evaluate_split(run,'val',{}, {})
            with patch.object(evaluator_api(),'evaluate',side_effect=AssertionError('must reuse public val')):
                second=evaluate_split(run,'val',{}, {})
            self.assertEqual(first,second)
        with patch('export.export_identity',return_value=({**identity,'config_sha256':'e'*64},{},None)):
            with self.assertRaisesRegex(ValueError,'Prediction cache'):
                evaluate_split(run,'val',{}, {})
        metrics=run/'metrics/val.json'
        metrics.write_text(metrics.read_text().replace('"images":1','"images":2'))
        with patch('export.export_identity',return_value=(identity,{},None)):
            with self.assertRaisesRegex(ValueError,'checksum'):
                evaluate_split(run,'val',{}, {})

    def test_launcher_existing_run_and_active_session_are_protected(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file():
            self.skipTest('Bash unavailable')
        script=ROOT/'scripts/autodl_yolov8m_configurable.sh'
        bash_env=dict(os.environ)
        if os.name=='nt':
            bash_env['PATH']='C:/Program Files/Git/usr/bin'+os.pathsep+bash_env['PATH']
        output=ROOT/'outputs/yolov8m-configurable'; output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='guard_',dir=output) as td:
            result=subprocess.run([bash,str(script),'start','--run-id',Path(td).name,'--config','/unused.yaml'],
                                  env=bash_env,capture_output=True,text=True)
            self.assertEqual(result.returncode,73,result.stderr)
        bindir=self.base/'bin'; bindir.mkdir()
        tmux=bindir/'tmux'
        tmux.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TMUX_CALLS"\n'
                        '[[ "$1" == has-session ]] && exit 0\n'
                        '[[ "$1" == list-panes ]] && { echo 0; exit 0; }\nexit 99\n',encoding='utf-8')
        tmux.chmod(0o755)
        env=dict(bash_env,PATH=str(bindir)+os.pathsep+bash_env['PATH'],FAKE_TMUX_CALLS=(self.base/'calls').as_posix())
        result=subprocess.run([bash,str(script),'start','--run-id','session_only','--config','/unused.yaml',
                               '--python',Path(bash).as_posix()],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,73,result.stderr)
        calls=(self.base/'calls').read_text()
        self.assertIn('=comparison-yolov8m-session_only',calls)
        self.assertNotIn('kill',calls)
        self.assertNotIn('new-session',calls)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
