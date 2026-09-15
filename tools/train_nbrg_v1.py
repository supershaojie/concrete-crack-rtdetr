"""C2-locked NBR-G experiment plan/start; start requires bound CPU/CUDA/B16 evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch
from init_nbrg_v1 import (ROOT, MODEL_DIR, VARIANTS, SOURCE_SHA256, require, sha256, write_json,
                          build_training_model, verify_model, runtime, gate_key)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from c19_lif_v1_data import dataset_inventory


def source_fingerprint():
    import hashlib
    paths = sorted(list((ROOT / 'tools').glob('*.py')) + list((ROOT / 'ultralytics-main/ultralytics').rglob('*.py')) +
                   list(MODEL_DIR.glob('*.yaml')))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def recipe(c2_args, variant, initial, data, project):
    source = YAML.load(c2_args)
    expected = YAML.load(ROOT / 'docs/c19_lif_v1/c2_args.yaml')
    require(set(source) == set(expected) and all(type(v) is type(expected[k]) and v == expected[k]
            for k, v in source.items()), 'Authoritative C2 args changed; review all fields')
    name = variant + '_rtdetr_r18_lite_e200_b16_onlineaug'
    target = {**source, 'model': str(Path(initial).resolve()), 'data': str(Path(data).resolve()),
              'project': str(Path(project).resolve()), 'name': name, 'save_dir': str((Path(project) / name).resolve())}
    changes = {k for k in source if source[k] != target[k]}
    require(changes <= {'model', 'data', 'project', 'name', 'save_dir'}, 'Training recipe drift')
    return target, [dict(field=k, c2=source[k], target=target[k], changed=k in changes,
                        reason='experiment identity/environment path mapping' if k in changes else 'inherited') for k in sorted(source)]


def verify_data(data):
    config, expected = YAML.load(data), YAML.load(ROOT / 'docs/c19_lif_v1/c2_data.yaml')
    require({k: v for k, v in config.items() if k != 'path'} == {k: v for k, v in expected.items() if k != 'path'},
            'Classes/split locations differ from C2')
    inventory = dataset_inventory(Path(config['path']))
    known = json.loads((ROOT / 'docs/nbrg_v1/authoritative_dataset_inventory.json').read_text(encoding='utf-8'))
    require(inventory == known, 'Dataset path/label fingerprint differs from original CBR+LIF; no automatic remapping')
    return inventory


def trainer_class(variant, source, metadata):
    class NBRGTrainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model, audit = build_training_model(cfg, weights, self.data, variant, source)
            if metadata:
                write_json(Path(metadata) / 'nc1_loading.json', audit)
            self.nbrg_initial_audit = audit
            return model

        def build_optimizer(self, model, *args, **kwargs):
            verify_model(model, variant, zero=True)
            optimizer = super().build_optimizer(model, *args, **kwargs)
            ids = [id(p) for group in optimizer.param_groups for p in group['params']]
            require(all(ids.count(id(p)) == 1 for p in model.parameters() if p.requires_grad), 'Optimizer coverage')
            if metadata:
                write_json(Path(metadata) / 'optimizer_audit.json', dict(status='PASSED', before_first_update=True,
                    gates=[dict(name=k, trainable=p.requires_grad, included=ids.count(id(p)))
                           for k, p in model.named_parameters() if gate_key(k)]))
            return optimizer
    return NBRGTrainer


def validate_preflights(paths, variant, source, initial, data):
    seen = set()
    for path in paths:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        require(report.get('status') == 'PASSED', 'Incomplete/failed preflight report: ' + str(path))
        require(report['source_sha256'] == sha256(source) == SOURCE_SHA256, 'Source differs from preflight')
        require(report['source_fingerprint'] == source_fingerprint(), 'Code differs from preflight')
        require(report['initial_sha256'][variant] == sha256(initial), 'Initialization differs from preflight')
        row = report['variants'][variant]
        require(row['status'] == 'PASSED', 'Preflight not passed: ' + str(path))
        if report['mode'] in ('cuda', 'capacity'):
            require(report['dataset_inventory'] == verify_data(data), 'Data differs from preflight')
            current = runtime()
            require(all(report['runtime'][k] == current[k] for k in ('torch', 'cuda', 'gpu')), 'CUDA preflight environment differs')
        seen.add(report['mode'])
    require({'cpu', 'cuda', 'capacity'} <= seen, 'Require CPU, CUDA and real B16/640 AMP reports')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'start'))
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--initial-dir', type=Path, required=True)
    parser.add_argument('--c2-args', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True, help='Formal training parent, separate from preflight/test/metadata')
    parser.add_argument('--metadata', type=Path, required=True, help='New metadata directory; existing evidence is preserved')
    parser.add_argument('--preflight', type=Path, nargs='+', default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    initial = args.initial_dir / (args.variant + '.pt')
    require(sha256(args.source) == SOURCE_SHA256, 'Wrong source checkpoint')
    target, diff = recipe(args.c2_args, args.variant, initial, args.data, args.project)
    require(initial.is_file(), 'Generate all three initializations first')
    require(not args.metadata.exists(), 'Existing metadata preserved; use a new directory')
    args.metadata.mkdir(parents=True, exist_ok=False)
    YAML.save(args.metadata / 'train_args.yaml', target)
    write_json(args.metadata / 'args_diff.json', diff)
    write_json(args.metadata / 'plan.json', dict(action=args.action, variant=args.variant, yaml=str(MODEL_DIR / VARIANTS[args.variant][0]),
        args=target, runtime=runtime(), source_fingerprint=source_fingerprint(), source_sha256=sha256(args.source),
        initial_sha256=sha256(initial), c2_args_sha256=sha256(args.c2_args), formal_training='NOT_STARTED', final_test='NOT_RUN'))
    if args.action == 'plan':
        print('Plan written; no training started:', args.metadata)
        return
    require(torch.cuda.is_available(), 'Formal training requires CUDA')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip(),
            'Commit/review source before formal launch')
    require(not Path(target['save_dir']).exists(), 'Existing formal run preserved')
    validate_preflights(args.preflight, args.variant, args.source, initial, args.data)
    write_json(args.metadata / 'dataset_inventory.json', verify_data(args.data))
    # Same native AMP check resources as the original combination launcher.
    from train_c19_lif_v1 import ensure_amp_resources
    ensure_amp_resources(args.source.resolve().parent.parent, args.metadata / 'amp_resources.json')
    reservation = Path(target['save_dir'] + '.lock')
    reservation.parent.mkdir(parents=True, exist_ok=True)
    reservation.mkdir(exist_ok=False)
    write_json(reservation / 'owner.json', dict(metadata=str(args.metadata.resolve()), variant=args.variant, runtime=runtime()))
    api = RTDETR(str(initial))
    state = dict(status='STARTING', formal_training='NOT_COMPLETE', final_test='NOT_RUN')
    write_json(args.metadata / 'training_state.json', state)

    def setup(trainer):
        verify_model(trainer.model, args.variant, zero=True)
        require(bool(trainer.amp), 'Native AMP disabled AMP; stop without changing the recipe')
        actual = vars(trainer.args)
        changed = {k: [v, actual.get(k)] for k, v in target.items() if type(v) is not type(actual.get(k)) or v != actual[k]}
        write_json(args.metadata / 'actual_args_diff.json', changed)
        require(not changed, 'Final Trainer args differ from C2 recipe')
        YAML.save(args.metadata / 'actual_train_args.yaml', actual)
        state.update(status='RUNNING')
        write_json(args.metadata / 'training_state.json', state)

    api.add_callback('on_train_start', setup)
    api.add_callback('on_train_batch_start', lambda trainer: setattr(trainer, '_oom_retries', 3))
    try:
        api.train(trainer=trainer_class(args.variant, args.source, args.metadata), **target)
        completed = int(api.trainer.epoch) + 1
        require((Path(target['save_dir']) / 'results.csv').is_file(), 'Missing results.csv')
        require(completed == 200 or (bool(api.trainer.stopper.possible_stop) and completed < 200),
                'Unexpected early termination; do not label as completed')
        state.update(status='COMPLETED', formal_training='COMPLETE', actual_epochs=completed,
            stop_reason='patience=50 early stopping' if bool(api.trainer.stopper.possible_stop) and completed < 200 else 'epoch limit' if completed == 200 else 'trainer stopped; inspect logs',
            best=str(api.trainer.best), last=str(api.trainer.last))
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=repr(error))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.metadata / 'training_state.json', state)


if __name__ == '__main__':
    main()
