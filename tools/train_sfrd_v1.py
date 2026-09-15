"""Plan or explicitly train SFR-D with the unchanged authoritative 109-field C2 recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from init_sfrd_v1 import (ROOT, MODEL_DIR, VARIANTS, SOURCE_SHA256, require, sha256, write_json,
                          build_training_model, new_keys, runtime, tensor_hash)
from train_c19_lif_v1 import disable_oom_retry, optimizer_groups, verify_server_environment
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from c19_lif_v1_data import dataset_inventory


def source_hashes():
    paths = list((ROOT/'tools').glob('*sfrd_v1*.py'))
    paths += [ROOT/'ultralytics-main/ultralytics/nn'/p for p in (
        'tasks.py', 'modules/sfr_d.py', 'modules/lif_down.py', 'modules/cbr.py', 'modules/transformer.py')]
    paths += [MODEL_DIR/v[0] for v in VARIANTS.values()]
    paths += [ROOT/'tools'/p for p in ('c19_lif_v1_diagnostic.py','c19_lif_v1_probe.py','c19_lif_v1_cutoff.py')]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(paths)}


def recipe(c2_args, initialized, variant, data, project):
    source = YAML.load(c2_args); expected = YAML.load(ROOT/'docs/c19_lif_v1/c2_args.yaml')
    require(set(source) == set(expected) and all(type(source[k]) is type(expected[k]) and source[k] == expected[k] for k in source),
            'Actual C2 args differ from authoritative 109-field archive')
    name = variant+'_rtdetr_r18_lite_e200_b16_onlineaug'
    target = {**source, 'model':str(Path(initialized).resolve()), 'data':str(Path(data).resolve()),
              'project':str(Path(project).resolve()), 'name':name, 'save_dir':str(Path(project).resolve()/name)}
    rows = [dict(field=k, original=source[k], resolved=target[k], changed=source[k] != target[k]) for k in sorted(source)]
    require({r['field'] for r in rows if r['changed']} <= {'model','data','project','name','save_dir'}, 'Recipe changed')
    return target, rows


def verify_data(path):
    config = YAML.load(path); expected = YAML.load(ROOT/'docs/c19_lif_v1/c2_data.yaml')
    require(set(config) == set(expected), 'Unexpected data config fields')
    require({k:v for k,v in config.items() if k != 'path'} == {k:v for k,v in expected.items() if k != 'path'}, 'Class/splits changed')
    data_root = Path(config['path']).resolve()
    inventory = dataset_inventory(data_root)
    # Match the committed successful original combination fingerprints, not counts alone.
    historical = json.loads((ROOT/'docs/c19_lif_v1/checks.json').read_text(encoding='utf-8'))['dataset']
    require(inventory == historical, 'Data split/label fingerprints differ from original CBR+LIF')
    return data_root, inventory


class SFRDTrainer(RTDETRTrainer):
    """Native training, with audited strict state mapping before optimizer creation."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        variant = weights.sfrd_variant
        model, report = build_training_model(cfg, weights, self.data, variant, initial=not self.args.resume)
        self.sfrd_initial_report = report
        return model


def require_preflight(report, args, variant, inventory):
    require(report['status'] == 'PASSED' and report['variant'] == variant, 'Required preflight missing/failed/pending')
    require(report['source_sha256'] == SOURCE_SHA256 and report['initialized_sha256'] == sha256(args['model']), 'Preflight initial state differs')
    require(report['source_hashes'] == source_hashes(), 'Code changed since preflight')
    require(report['dataset'] == inventory, 'Dataset changed since preflight')
    cap = report['capacity']
    require(cap['status'] == 'PASSED' and cap['batch'] == 16 and cap['imgsz'] == 640 and cap['AMP'] is True and
            cap['optimizer_steps'] == 0, 'CUDA 640 B16 AMP capacity has not passed')
    require(all(report['cuda'][key]['status'] == 'PASSED' for key in ('loss','AMP_loss')), 'CUDA loss checks missing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan','train'])
    parser.add_argument('--variant', choices=list(VARIANTS), default='cbr_lif_sfrd_v1')
    for key in ('initialized','c2-args','data','project','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args(); torch.set_num_threads(4)
    resolved, diff = recipe(args.c2_args, args.initialized, args.variant, args.data, args.project)
    _, inventory = verify_data(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    YAML.save(args.output/'resolved_formal_config.yaml', resolved)
    write_json(args.output/'recipe_diff.json', diff)
    write_json(args.output/'dataset_inventory.json', inventory)
    if args.mode == 'plan':
        print('Plan written; no training started.')
        return
    require(args.preflight is not None, 'Pass completed --preflight report')
    report = json.loads(args.preflight.read_text(encoding='utf-8'))
    require_preflight(report, resolved, args.variant, inventory)
    verify_server_environment(runtime())
    require((ROOT/'.git').is_file(), 'Independent linked worktree required')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(), 'Uncommitted source changes')
    checkpoint = torch_load(args.initialized, map_location='cpu')
    require(checkpoint.get('epoch') == -1 and all(checkpoint.get(k) is None for k in ('ema','optimizer','scaler','updates')), 'Clean initial checkpoint required')
    require(checkpoint['sfrd_v1_provenance']['source_sha256'] == SOURCE_SHA256 and
            checkpoint['model'].sfrd_variant == args.variant, 'Wrong initialization provenance/variant')
    run = Path(resolved['save_dir']); require(not run.exists(), 'Existing training results protected')
    lock = run.with_name(run.name+'.sfrd.lock'); lock.parent.mkdir(parents=True,exist_ok=True); lock.mkdir(exist_ok=False)
    write_json(lock/'owner.json', dict(runtime=runtime(), worktree=str(ROOT), variant=args.variant))
    model = RTDETR(str(args.initialized))
    model.add_callback('on_train_batch_start', disable_oom_retry)
    def setup(trainer):
        actual = vars(trainer.args)
        require(all(actual[k] == v and type(actual[k]) is type(v) for k,v in resolved.items()), 'Effective formal args changed')
        require(bool(trainer.amp) and trainer.batch_size == 16, 'AMP/batch changed')
        expected_state = trainer.sfrd_initial_report['state_sha256']
        require({k:tensor_hash(v) for k,v in trainer.model.state_dict().items()} == expected_state, 'Training setup altered initial state')
        ids = [id(p) for g in trainer.optimizer.param_groups for p in g['params']]
        require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in trainer.model.parameters()}, 'Optimizer missing/duplicate params')
        require(type(trainer.optimizer) is torch.optim.AdamW, 'Optimizer changed')
        write_json(args.output/'trainer_initialization.json', trainer.sfrd_initial_report)
        write_json(args.output/'optimizer.json', optimizer_groups(trainer.model,trainer.optimizer))
        YAML.save(args.output/'actual_args.yaml', actual)
    model.add_callback('on_train_start', setup)
    write_json(args.output/'launch.json', dict(runtime=runtime(), args=resolved, preflight_sha256=sha256(args.preflight), initialized_sha256=sha256(args.initialized)))
    try:
        model.train(trainer=SFRDTrainer, **resolved)
        write_json(args.output/'exit.json', dict(status='COMPLETED', epochs_completed=model.trainer.epoch+1,
            stop_reason='maximum_epochs' if model.trainer.epoch+1 == 200 else 'early_stop_or_native_stop'))
    except BaseException as error:
        write_json(args.output/'exit.json', dict(status='FAILED', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
