"""Risk-based configuration, immutability, native optimizer, pack and launcher tests."""
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
from support import (ROOT, HERE, SOURCE, configure, native_recipe, recipe, read_json, sha256, write_json)
from configuration import (candidate, cli_overrides, freeze_config, frozen_config, frozen_paths,
                           interpreter_path, resolve_config, safe_run_id, schema, yaml_mapping)

configure(ROOT/'outputs/yolo26m-configurable-validation/unit-runtime',SOURCE)
import torch
from ultralytics.cfg import get_cfg
from adapters import ComparisonTrainer
from smoke import fixture, IsolatedDataset
from packaging_run import archive_config, pack
from run import current_state, summary


class ConfigurableTests(unittest.TestCase):
    def setUp(self):
        out = ROOT/'outputs/yolo26m-configurable-validation'
        out.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=out)
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def freeze(self,name='candidate',raw=b'{}\n',overrides=None):
        run = self.base/name
        paths = {'python':str(Path(sys.executable).absolute()),'sys_prefix':str(Path(sys.prefix).absolute())}
        cfg = freeze_config(run,raw,paths,name,require_clean=False,overrides=overrides)
        return run,cfg

    def test_defaults_are_exact_planned_x13_recipe_and_native_cfg(self):
        cfg = resolve_config({})
        self.assertEqual(cfg,recipe())
        self.assertEqual(cfg['hsv_h'],.0377); self.assertEqual(cfg['mixup'],.234)
        self.assertEqual(cfg['cutmix'],.065); self.assertEqual(cfg['close_mosaic'],5)
        self.assertEqual(get_cfg(overrides=native_recipe(cfg)).optimizer,'MuSGD')
        self.assertEqual(schema(),read_json(HERE/'config_schema.json'))

    def test_priority_scalar_types_full_yaml_and_duplicate_rejection(self):
        path = self.base/'config.yaml'; path.write_bytes(b'lr0: 0.02\namp: false\nfreeze: null\n')
        cfg,_,overrides,_ = candidate(path,['lr0=0.005','amp=true','workers=0'])
        self.assertEqual(cfg['lr0'],.005); self.assertIs(cfg['amp'],True)
        self.assertIsNone(cfg['freeze']); self.assertIs(type(overrides['workers']),int)
        self.assertEqual(cli_overrides(['lr0=1e-3'])['lr0'],.001)
        self.assertEqual(yaml_mapping(b'lr0: 1e-3\n')['lr0'],.001)
        self.assertEqual(resolve_config({'epochs':3},{'warmup_epochs':0,'close_mosaic':0})['epochs'],3)
        self.assertEqual(resolve_config(cfg),cfg)
        for raw in (b'lr0: .1\nlr0: .2\n',b'1: 2\n',b'- list\n'):
            with self.subTest(raw=raw),self.assertRaises(ValueError): yaml_mapping(raw)
        for items in (['lr0=.1','lr0=.2'],['batch'],['batch='],[' lr0=.1']):
            with self.subTest(items=items),self.assertRaises(ValueError): cli_overrides(items)

    def test_invalid_types_ranges_inactive_fields_and_fixed_identities(self):
        for cfg in ({'optimizer':'auto'},{'optimizer':[]},{'batch':-1},{'batch':.5},{'epochs':True},
                    {'workers':1.0},{'cos_lr':'false'},{'amp':1},{'cache':'ram'},{'rect':True},
                    {'multi_scale':True},{'copy_paste':.1},{'bgr':.1},{'freeze':0},{'classes':[0]},
                    {'model':'yolov8m.pt'},{'initialization_type':'random'},{'resume':True},
                    {'pretrained':False},{'task':'segment'},{'imgsz':320},{'conf':.01},
                    {'erasing':.1},{'auto_augment':'randaugment'},{'beta2':.9},{'muon':.8},{'sgd':.2},{'muon_w':.5},{'sgd_w':.5},{'end2end':False},{'reg_max':16},
                    {'lr0':float('nan')},{'lr0':float('inf')},{'lr0':0},{'momentum':1},
                    {'cutmix':1.01},{'perspective':.01},{'close_mosaic':201},
                    {'warmup_epochs':201},{'box':0,'cls':0,'dfl':0},
                    {'optimizer':'AdamW','warmup_momentum':.5}):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError): resolve_config(cfg)
        for name in ('../run','a/b','a b','-run','x'*81):
            with self.assertRaises(ValueError): safe_run_id(name)
        self.assertEqual(resolve_config({},cli_overrides(['batch=8','amp=false']))['batch'],8)

    def test_candidate_file_edits_do_not_change_frozen_run_and_tampering_fails(self):
        path = self.base/'candidate.yaml'; path.write_bytes(b'lr0: 0.012\n')
        run,cfg = self.freeze(raw=path.read_bytes(),overrides={'mixup':.18})
        path.write_bytes(b'lr0: 0.003\n'); path.unlink()
        self.assertEqual(frozen_config(run,check_code=False),cfg)
        with self.assertRaises(FileExistsError): self.freeze()
        (run/'cli_overrides.json').write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError,'Frozen run artifact'): frozen_config(run,check_code=False)

    def test_clone_inherits_only_recipe_new_uuid_and_coco_initialization(self):
        original,_ = self.freeze('original',overrides={'lr0':.012})
        # A foreign training-state filename must never be consumed by the cloning interface.
        (original/'last.pt').write_bytes(b'do not restore this checkpoint')
        cfg,raw,overrides,origin = candidate(items=['lr0=.005'],clone=original)
        cloned = self.base/'cloned'
        freeze_config(cloned,raw,read_json(original/'runtime_paths.json'),'cloned',require_clean=False,
                      overrides=overrides,origin=origin)
        self.assertEqual(cfg['lr0'],.005); self.assertIs(cfg['resume'],False); self.assertIs(cfg['pretrained'],True)
        self.assertEqual(cfg['initialization_type'],'coco_detection_pretrained')
        self.assertFalse(origin['training_state_inherited'])
        self.assertNotEqual(read_json(original/'run_id.json')['run_uuid'],read_json(cloned/'run_id.json')['run_uuid'])
        self.assertFalse((cloned/'last.pt').exists())

    def test_native_optimizers_and_actual_transform_parameters_follow_overrides(self):
        data,run,manifest = fixture(self.base)
        for name in ('MuSGD','SGD','Adam','AdamW'):
            cfg = resolve_config({},cli_overrides([f'optimizer={name}','lr0=0.005','momentum=0.9',
                'batch=8','mixup=0.18','cutmix=0.2','hsv_h=0.04','box=6.0']))
            trainer = ComparisonTrainer.__new__(ComparisonTrainer)
            trainer.args = get_cfg(overrides=native_recipe(cfg))
            model = torch.nn.Sequential(torch.nn.Conv2d(3,4,3),torch.nn.BatchNorm2d(4))
            optimizer = trainer.build_optimizer(model,name=trainer.args.optimizer,lr=trainer.args.lr0,
                                                 momentum=trainer.args.momentum,decay=trainer.args.weight_decay)
            from optimizer import expected_class
            self.assertIs(type(optimizer),expected_class(name))
            self.assertTrue(all(g['lr']==.005 for g in optimizer.param_groups if g['params']))
            if name in ('SGD','MuSGD'): self.assertTrue(optimizer.param_groups[0]['nesterov'])
            else: self.assertEqual(optimizer.param_groups[0]['betas'],(.9,.999))
            ds = IsolatedDataset(img_path=str(run/'train.txt'),imgsz=64,batch_size=8,augment=True,
                hyp=trainer.args,rect=False,cache=False,data={'names':{0:'crack'}},manifest=manifest,
                split='train',cache_root=run/'cache',cutmix_probability=cfg['cutmix'])
            self.assertEqual(ds.transforms.transforms[1].p,.18)
            self.assertEqual(ds.transforms.transforms[2].p,.2)
            self.assertEqual(ds.transforms.transforms[4].hgain,.04)
            self.assertEqual(trainer.args.box,6.)
            self.assertIn('actual_transform_tree',ds.transform_report)

    def test_validation_batch_tracks_physical_batch_and_workers_zero(self):
        trainer = ComparisonTrainer.__new__(ComparisonTrainer)
        trainer.args = SimpleNamespace(batch=8,workers=8)
        trainer.build_dataset = lambda *args: SimpleNamespace()
        with patch('adapters.build_dataloader',return_value=SimpleNamespace()) as build:
            trainer.get_dataloader('fixture',batch_size=32,mode='val')
            self.assertEqual(build.call_args.args[1:3],(8,0))

    def test_venv_path_preserves_symlink_and_checks_prefix(self):
        run,_ = self.freeze()
        with patch('configuration.frozen_config'):
            self.assertEqual(frozen_paths(run)['python'],str(Path(sys.executable).absolute()))
        self.assertEqual(interpreter_path('/tmp/env/bin/python'),os.path.normcase(os.path.abspath('/tmp/env/bin/python')))
        with patch('configuration.frozen_config'),patch('configuration.sys.prefix','/other/env'):
            with self.assertRaisesRegex(ValueError,'sys.prefix'): frozen_paths(run)

    def test_status_dead_process_preflight_failure_and_not_requested_test(self):
        run,_ = self.freeze()
        write_json(run/'preflight_status.json',{'status':'running','pid':os.getpid(),'process_created':0})
        self.assertEqual(current_state(run/'preflight_status.json')['status'],'failed')
        result = summary(run,save=False,quiet=True)
        self.assertEqual(result['status'],'preflight_failed'); self.assertEqual(result['final_test_status'],'not_requested')
        self.assertFalse((run/'summary.json').exists())

    def test_pack_is_read_only_review_excludes_weights_and_archive_states(self):
        run,_ = self.freeze()
        weights = run/'train/weights'; weights.mkdir(parents=True)
        (weights/'best.pt').write_bytes(b'fake excluded weight')
        (run/'cache').mkdir(); (run/'cache'/'dataset.txt').write_text('must not be included')
        before = {p.relative_to(run).as_posix():sha256(p) for p in run.rglob('*') if p.is_file()}
        args = SimpleNamespace(run=run,out=self.base/'review.tar.gz',include_weights=False)
        with patch('packaging_run.frozen_config',side_effect=lambda path:frozen_config(path,check_code=False)):
            result = pack(args)
        import tarfile
        with tarfile.open(args.out) as archive:
            names=archive.getnames()
        self.assertFalse(any(p.endswith('.pt') or 'cache/dataset' in p for p in names))
        self.assertFalse(result['includes_weights']); self.assertIn('SNAPSHOT_INCOMPLETE',result['scope'])
        self.assertEqual(before,{p.relative_to(run).as_posix():sha256(p) for p in run.rglob('*') if p.is_file()})
        args.include_weights=True; args.out=self.base/'weights.tar.gz'
        with patch('packaging_run.frozen_config',side_effect=lambda path:frozen_config(path,check_code=False)):
            with self.assertRaisesRegex(ValueError,'completed'): pack(args)
        planned = SimpleNamespace(run=self.base/'not_started',run_id='not_started',config=None,set=[],clone_config_from=None,out=self.base/'archive.json')
        record=archive_config(planned)
        self.assertEqual(record['status'],'NOT_RUN'); self.assertIs(record['training_state_inherited'],False)

    def test_finalize_rejects_unfinished_and_best_tampering(self):
        from run import legal_completion
        run,cfg = self.freeze()
        write_json(run/'train_status.json',{'status':'failed','exit_code':1})
        with self.assertRaisesRegex(ValueError,'legal'): legal_completion(run,cfg)
        weights = run/'train/weights'; weights.mkdir(parents=True)
        for name in ('best','last'): (weights/(name+'.pt')).write_bytes(b'known')
        seals={'epoch':200,'best_epoch':200,**{k+'_sha256':sha256(weights/(k+'.pt')) for k in ('best','last')}}
        write_json(run/'checkpoint_integrity.json',seals)
        write_json(run/'train_status.json',{'status':'completed','exit_code':0,'stop_reason':'epoch_limit',
            'completed_epoch':200,'best_epoch':200,**{k+'_sha256':seals[k+'_sha256'] for k in ('best','last')}})
        legal_completion(run,cfg)
        (weights/'best.pt').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'checksum'): legal_completion(run,cfg)

    def test_launcher_isolates_session_and_rejects_resume_parameter_changes(self):
        bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
        if not Path(bash).is_file(): self.skipTest('Bash unavailable')
        bindir=self.base/'bin'; bindir.mkdir()
        tmux=bindir/'tmux'
        tmux.write_bytes(b'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_TMUX_CALLS"\n[[ "$1" == has-session ]] && exit 0\n[[ "$1" == list-panes ]] && { printf "0\\n"; exit 0; }\nexit 99\n')
        tmux.chmod(0o755)
        extra='C:/Program Files/Git/usr/bin'+os.pathsep if os.name=='nt' else ''
        env=dict(os.environ,PATH=str(bindir)+os.pathsep+extra+os.environ['PATH'],FAKE_TMUX_CALLS=(self.base/'calls').as_posix())
        script=str(ROOT/'scripts/autodl_yolo26m_configurable.sh')
        result=subprocess.run([bash,script,'start','--run-id','guard_test','--python',sys.executable,'--detach'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,73,result.stderr)
        calls=(self.base/'calls').read_text()
        self.assertIn('=comparison-yolo26m-guard_test',calls)
        self.assertNotIn('yolov8m',calls)
        result=subprocess.run([bash,script,'resume','--run-id','guard_test','--set','lr0=.005','--python',sys.executable],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,64,result.stderr)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
