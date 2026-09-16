"""LSRT lifecycle: isolated plan, gated start, learned-state resume; never starts implicitly."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from init_lsrt_v1 import (ROOT, MODEL_DIR, VARIANTS, SOURCE_SHA256, BASE_COMMIT, LSRT_KEYS,
                          require, sha256, write_json, runtime, fingerprint, verify_model,
                          build_training_model)
import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from train_c19_lif_v1 import ensure_amp_resources, disable_oom_retry

MAIN = Path(os.environ.get('LSRT_V1_MAIN', '/root/autodl-tmp/projects/Crack_RTDETR'))
ARCHIVED_ARGS = ROOT / 'docs/c19_lif_v1/c2_args.yaml'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + ROOT.as_posix(), *args],
                                   cwd=ROOT, text=True).strip()


def recipe(c2_args, variant, initialized, data, project):
    """All 109 authoritative fields survive; only model/output/environment paths differ."""
    original, expected = YAML.load(c2_args), YAML.load(ARCHIVED_ARGS)
    require(len(original) == 109 and set(original) == set(expected), 'C2 field inventory changed')
    require(all(type(original[k]) is type(expected[k]) and original[k] == expected[k] for k in original),
            'C2 recipe differs from verified complete archive')
    target = dict(original)
    target.update(model=str(Path(initialized).resolve()), data=str(Path(data).resolve()),
                  project=str(Path(project).resolve()), name=VARIANTS[variant][2])
    target['save_dir'] = str(Path(target['project']) / target['name'])
    rows = [dict(field=k, C2=original[k], target=target[k], changed=original[k] != target[k],
                 reason=('model/output identity or explicitly verified environment path' if original[k] != target[k] else 'inherited'))
            for k in sorted(original)]
    require({r['field'] for r in rows if r['changed']} <= {'model', 'data', 'project', 'name', 'save_dir'},
            'Unapproved recipe change')
    return target, rows


def dataset_inventory(data_yaml):
    """Hash actual bytes, not mtimes; never run inference or decode test images."""
    path = Path(data_yaml).resolve()
    supplied, expected = YAML.load(path), YAML.load(ROOT/'docs/c19_lif_v1/c2_data.yaml')
    supplied_semantics = {k: v for k, v in supplied.items() if k != 'path'}
    expected_semantics = {k: v for k, v in expected.items() if k != 'path'}
    require(supplied_semantics == expected_semantics, 'Original split/names/nc semantics changed')
    resolved = check_det_dataset(str(path), autodownload=False)
    require(resolved['nc'] == 1 and resolved['names'] == {0: 'crack'}, 'Expected original crack class')
    dataset = Path(resolved['path']).resolve()
    result = dict(schema=1, data_yaml_sha256=sha256(path), dataset_root=str(dataset),
                  config=supplied_semantics, splits={})
    # JSON representation is the public schema (integer class IDs become string keys).
    result['config'] = json.loads(json.dumps(result['config']))
    seen = set()
    for split in ('train', 'val', 'test'):
        location = dataset/'images'/split
        require(Path(resolved[split]).resolve() == location.resolve(), 'Resolved split differs')
        files = sorted(p for p in location.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        require(files, 'Missing split ' + split)
        paths, labels, images, boxes = [], [], [], 0
        for image in files:
            require(image.resolve() not in seen, 'Duplicate image path across splits')
            seen.add(image.resolve())
            rel = image.relative_to(dataset).as_posix()
            label = dataset/'labels'/split/image.relative_to(location).with_suffix('.txt')
            require(label.is_file(), 'Missing label: ' + str(label))
            for line in label.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = [float(v) for v in line.split()]
                require(len(row) == 5 and row[0] == 0 and all(math.isfinite(v) for v in row), 'Invalid crack label')
                require(all(0 <= v <= 1 for v in row[1:]) and row[3] > 0 and row[4] > 0, 'Invalid normalized box')
                boxes += 1
            paths.append(rel)
            images.append(rel + ':' + sha256(image))
            labels.append(label.relative_to(dataset).as_posix() + ':' + sha256(label))
        def digest(rows):
            return hashlib.sha256('\n'.join(rows).encode()).hexdigest()
        result['splits'][split] = dict(images=len(files), boxes=boxes,
                                      split_paths_sha256=digest(paths), label_inventory_sha256=digest(labels),
                                      image_inventory_sha256=digest(images))
    return result


def verify_baseline_inventory(current, baseline_path):
    baseline = read_json(baseline_path)
    source = baseline.get('splits', baseline.get('dataset', baseline))
    evidence = {}
    for split in ('train', 'val', 'test'):
        require(split in source, 'Original inventory lacks ' + split)
        required = ('images', 'boxes', 'split_paths_sha256', 'label_inventory_sha256')
        require(all(source[split].get(k) == current['splits'][split][k] for k in required),
                'Data differs from original experiment: ' + split)
        if 'image_inventory_sha256' in source[split]:
            require(source[split]['image_inventory_sha256'] == current['splits'][split]['image_inventory_sha256'],
                    'Original image bytes changed: ' + split)
        evidence[split] = dict(paths_labels_counts_equal=True,
                              historical_image_hash_available='image_inventory_sha256' in source[split])
    return dict(status='PASSED', file=str(Path(baseline_path).resolve()), sha256=sha256(baseline_path),
                splits=evidence, note='Legacy baseline inventories lack image-content hashes; current image bytes are frozen for preflight/start.')


def create_plan(options):
    folder = options.output.resolve()
    require(not folder.exists(), 'Existing plan directory preserved: ' + str(folder))
    args, rows = recipe(options.c2_args, options.variant, options.initialized, options.data, options.project)
    require(not Path(args['save_dir']).exists(), 'Existing formal run preserved')
    require(sha256(options.source) == SOURCE_SHA256, 'Fixed source hash mismatch')
    require(options.initialized.is_file(), 'Initialize from fixed source before plan')
    folder.mkdir(parents=True, exist_ok=False)
    # Missing real data is recorded as pending; plan itself never reserves the formal run.
    inventory, baseline, pending = None, None, []
    if options.data.is_file():
        inventory = dataset_inventory(options.data)
        if options.baseline_inventory:
            baseline = verify_baseline_inventory(inventory, options.baseline_inventory)
        else:
            pending.append('Original experiment dataset inventory must be supplied to --baseline-inventory')
    else:
        pending.append('Real dataset YAML/images/labels unavailable')
    plan = dict(schema=1, status='PENDING' if pending else 'READY_FOR_PREFLIGHT', variant=options.variant,
                created=utc(), cwd=str(ROOT), source=str(options.source.resolve()),
                initialized=str(options.initialized.resolve()), fingerprints=fingerprint(options.variant, options.source, options.initialized),
                c2_args=str(options.c2_args.resolve()), c2_args_sha256=sha256(options.c2_args), args=args,
                data_inventory=inventory, baseline_inventory=baseline, pending=pending,
                formal_optimizer_steps=0, training='NOT_STARTED', final_test='NOT_RUN')
    YAML.save(folder/'train_args.yaml', args)
    write_json(folder/'parameter_diff.json', rows)
    shutil.copyfile(options.c2_args, folder/'authoritative_c2_args.yaml')
    if options.data.is_file():
        shutil.copyfile(options.data, folder/'data_config.yaml')
    if options.baseline_inventory:
        shutil.copyfile(options.baseline_inventory, folder/'baseline_dataset_inventory.json')
    write_json(folder/'plan.json', plan)
    print(json.dumps(dict(plan=str(folder/'plan.json'), status=plan['status'], fields=len(rows),
                          changed=[r['field'] for r in rows if r['changed']], pending=pending), indent=2))
    return plan


def require_preflight(report, plan):
    require(report.get('status') == 'PASSED', 'Preflight missing, failed or pending; never edit its status')
    require(report.get('variant') == plan['variant'], 'Preflight variant differs')
    require(report.get('fingerprints') == plan['fingerprints'], 'Preflight source/config/source-weight/init differs')
    require(report.get('data_inventory') == plan['data_inventory'] and plan['data_inventory'] is not None,
            'Real image/label inventory differs from preflight')
    required_checks = ('initialization','native_trainer','topology','actual_train_API','AdamW_coverage','complexity',
                       'cpu_module','cpu_initial_equivalence','cpu_FP32_loss','cpu_reload_fusion',
                       'cuda_module','cuda_initial_equivalence','cuda_FP32_loss','cuda_reload_fusion','cuda_AMP_loss')
    checks = report.get('checks', {})
    require(all(checks.get(name, {}).get('status') == 'PASSED' for name in required_checks),
            'One or more specific mandatory preflight checks missing/failed')
    require(report.get('formal_optimizer_steps') == 0 and not report.get('pending'),
            'Preflight must retain zero formal updates with no pending items')
    for name in ('cpu_FP32_loss','cuda_FP32_loss','cuda_AMP_loss'):
        check = checks[name]
        require(check.get('optimizer_steps') == 0 and len(check.get('steps', [])) == 2,
                'Zero/activated LSRT native loss and backward missing: ' + name)
        require(all(type(row.get('loss')) in (int,float) and math.isfinite(row['loss'])
                    and row.get('gradients') and row.get('dn_num_split') for row in check['steps']),
                'Finite native loss/gradient/DN evidence missing: ' + name)
    require(checks['cuda_module'].get('checks', {}).get('native_amp', {}).get('status') == 'PASSED',
            'CUDA native AMP module check missing')
    capacity = report.get('capacity', {})
    require(capacity.get('status') == 'PASSED' and capacity.get('batch') == 16 and capacity.get('imgsz') == 640
            and capacity.get('AMP') is True and capacity.get('optimizer_steps') == 0,
            'Real B16/640 native AMP capacity forward/backward missing')
    require(type(capacity.get('loss')) in (int, float) and math.isfinite(capacity['loss']), 'Capacity finite loss missing')
    require(capacity.get('native_training_augmentation') is True and len(capacity.get('batches', [])) == 2,
            'Native augmented order and high-GT training capacity batches missing')
    for row in capacity['batches']:
        require(len(row.get('images', [])) == 16 and len(row.get('GT_per_image', [])) == 16
                and row.get('dn_num_split') and row.get('peak_allocated_bytes', 0) > 0
                and type(row.get('loss')) in (int,float) and math.isfinite(row['loss']),
                'Incomplete real capacity GT/DN/memory/loss evidence')


def verify_plan(plan_path, preflight_path):
    plan = read_json(plan_path)
    require(plan['schema'] == 1 and plan['variant'] in VARIANTS, 'Invalid plan')
    require(Path(plan['cwd']).resolve() == ROOT, 'Plan belongs to another worktree')
    require(plan['fingerprints'] == fingerprint(plan['variant'], plan['source'], plan['initialized']), 'Plan inputs/source changed')
    require(plan['fingerprints']['source_sha256'] == SOURCE_SHA256, 'Fixed source mismatch')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Commit reviewed source before formal start')
    require(sha256(plan['c2_args']) == plan['c2_args_sha256'], 'Authoritative recipe changed')
    args, rows = recipe(plan['c2_args'], plan['variant'], plan['initialized'], plan['args']['data'], plan['args']['project'])
    require(args == plan['args'], 'Plan recipe altered')
    require(YAML.load(Path(plan_path).parent/'train_args.yaml') == args, 'Plan YAML differs')
    require(read_json(Path(plan_path).parent/'parameter_diff.json') == rows, 'Parameter diff altered')
    inventory = dataset_inventory(args['data'])
    require(inventory == plan['data_inventory'], 'Dataset images/labels/config changed since plan')
    require((plan.get('baseline_inventory') or {}).get('status') == 'PASSED', 'Original data identity not verified')
    frozen_baseline = Path(plan_path).parent/'baseline_dataset_inventory.json'
    require(sha256(frozen_baseline) == plan['baseline_inventory']['sha256'], 'Baseline inventory evidence changed')
    verify_baseline_inventory(inventory, frozen_baseline)
    report = read_json(preflight_path)
    require_preflight(report, plan)
    return plan, report


def verify_environment():
    info = runtime()
    require(torch.cuda.is_available(), 'CUDA is required for formal training')
    require(platform.python_version() == '3.10.13' and info['torch'] == '2.1.2+cu121',
            'Expected existing successful Python3.10.13/torch2.1.2+cu121; report mismatch without upgrading')
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'rtdetr', 'Activate existing rtdetr environment')
    return info


def optimizer_audit(trainer):
    model = trainer.model
    names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    ids = [id(p) for group in trainer.optimizer.param_groups for p in group['params']]
    count = Counter(ids)
    require(type(trainer.optimizer) is torch.optim.AdamW, 'Original AdamW optimizer changed')
    require(set(ids) == set(names) and all(v == 1 for v in count.values()), 'Trainable parameters omitted/duplicated in optimizer')
    require(LSRT_KEYS <= {names[i] for i in ids}, 'LSRT parameters registered too late')
    return dict(optimizer='AdamW', trainable_tensors=len(names), all_exactly_once=True,
                LSRT_occurrences={n: count[id(p)] for n,p in model.named_parameters() if n in LSRT_KEYS},
                groups=[dict(index=i, lr=g['lr'], weight_decay=g.get('weight_decay'),
                             parameters=[names[id(p)] for p in g['params']]) for i,g in enumerate(trainer.optimizer.param_groups)])


def training_state(trainer, optimizer_steps, finished=False):
    completed = int(trainer.epoch) + 1
    if not finished:
        status = 'RUNNING'
    elif completed >= int(trainer.epochs):
        status = 'COMPLETED_200_EPOCHS'
    elif trainer.stop and trainer.stopper.patience and completed - trainer.stopper.best_epoch >= trainer.stopper.patience:
        status = 'EARLY_STOPPED_PATIENCE'
    else:
        status = 'STOPPED_OTHER'
    return dict(status=status, completed_epochs=completed, expected_epochs=trainer.epochs,
                optimizer_steps_this_invocation=optimizer_steps, updated=utc())


def run_training(options, resume=False):
    plan_path, preflight_path = options.plan.resolve(), options.preflight.resolve()
    plan, preflight = verify_plan(plan_path, preflight_path)
    info = verify_environment()
    variant, args = plan['variant'], dict(plan['args'])
    run = Path(args['save_dir'])
    output = options.output.resolve()
    require(not output.exists(), 'Existing launch audit directory preserved')
    require(ROOT != MAIN.resolve(), 'Use independent LSRT worktree')
    checkpoint = options.checkpoint.resolve() if resume else Path(plan['initialized'])
    if resume:
        require(run.is_dir() and checkpoint == (run/'weights/last.pt').resolve(), 'Resume only this plan actual last.pt')
        saved = torch_load(checkpoint, map_location='cpu')
        require(saved.get('epoch', -1) >= 0 and saved.get('optimizer') is not None, 'Checkpoint is stripped/completed; no resumable training state')
        trained = saved.get('ema') if saved.get('ema') is not None else saved.get('model')
        require(trained is not None, 'Missing learned checkpoint model')
        for p in trained.parameters():
            p.requires_grad_(True)  # Loader/native Trainer also restores training; never zero learned weights.
        verify_model(trained, variant, zero=False)
        checkpoint_args = saved['train_args']
        require(all(checkpoint_args.get(k) == v for k,v in args.items() if k not in {'model','resume'}),
                'Resume checkpoint recipe differs from pinned experiment')
        # The previous launch is mandatory so a foreign trained model cannot be resumed under this plan.
        previous = read_json(options.previous_launch/'launch.json')
        require(previous['plan_sha256'] == sha256(plan_path) and previous['run'] == str(run)
                and previous['fingerprints'] == plan['fingerprints'], 'Resume provenance/source differs')
        args.update(model=str(checkpoint), resume=str(checkpoint))
    else:
        require(not run.exists(), 'Existing formal run preserved; no automatic name2/name3')
        raw = torch_load(checkpoint, map_location='cpu')
        require(raw.get('epoch') == -1 and raw.get('optimizer') is None and raw.get('updates') is None,
                'New start requires zero-update initialized checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    # Shared atomic lock. A stopped process leaves evidence; operators inspect it before explicit recovery.
    lock = run.with_name(run.name + '.lsrt_v1.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        require(lock.is_dir(), 'Missing original run reservation')
        previous_owner = read_json(lock/'owner.json')
        require(previous_owner.get('plan_sha256') == sha256(plan_path), 'Run reservation belongs to another plan')
        if previous_owner.get('pid'):
            try:
                os.kill(int(previous_owner['pid']), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Previous worker PID is live/reused; reservation preserved for manual review')
        claim = lock/'resume_claim'
        claim.mkdir(exist_ok=False)
    else:
        lock.mkdir(exist_ok=False)
    write_json(lock/'owner.json', dict(pid=os.getpid(), plan_sha256=sha256(plan_path), variant=variant, started=utc()))
    write_json(output/'launch.json', dict(mode='resume' if resume else 'start', variant=variant, run=str(run),
               plan_sha256=sha256(plan_path), preflight_sha256=sha256(preflight_path),
               fingerprints=plan['fingerprints'], checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), runtime=info))
    shutil.copyfile(preflight_path, output/'preflight.json')
    shutil.copyfile(plan_path, output/'plan.json')
    (output/'pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    (output/'source_from_base.patch').write_bytes(subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(), 'diff', '--binary', BASE_COMMIT, 'HEAD'], cwd=ROOT))
    ensure_amp_resources(MAIN, output/'amp_resources.json')
    steps, ended = [0], [False]
    trainer_ref = [None]

    class AuditedTrainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model, audit = build_training_model(cfg, weights, self.data, variant)
            write_json(output/'nc1_loading.json', audit)
            return model

        def optimizer_step(self):
            super().optimizer_step()
            steps[0] += 1

    def setup(trainer):
        trainer_ref[0] = trainer
        graph = verify_model(trainer.model, variant, zero=not resume)
        require(trainer.model.model[-1].nc == 1, 'True Trainer model must be nc=1')
        require(bool(trainer.amp), 'Native AMP check disabled AMP; refusing changed recipe')
        actual = vars(trainer.args)
        differences = {k: [v, actual.get(k)] for k,v in args.items()
                       if type(v) is not type(actual.get(k)) or v != actual.get(k)}
        require(not differences, 'Actual Trainer recipe differs: ' + str(differences))
        require(Path(trainer.save_dir).resolve() == run.resolve(), 'Trainer silently changed run directory')
        YAML.save(output/'actual_train_args.yaml', actual)
        write_json(output/'training_setup.json', dict(graph=graph, optimizer=optimizer_audit(trainer),
                   amp=True, recipe_differences=differences, formal_optimizer_steps_before_start=steps[0], resume=resume))

    def epoch_end(trainer):
        write_json(output/'training_state.json', training_state(trainer, steps[0]))

    def train_end(trainer):
        ended[0] = True
        write_json(output/'training_state.json', training_state(trainer, steps[0], finished=True))

    code = 1
    try:
        model = RTDETR(str(checkpoint))
        model.add_callback('on_train_start', setup)
        model.add_callback('on_train_batch_start', disable_oom_retry)
        model.add_callback('on_train_epoch_end', epoch_end)
        model.add_callback('on_train_end', train_end)
        write_json(output/'training_state.json', dict(status='STARTING', optimizer_steps_this_invocation=0))
        model.train(trainer=AuditedTrainer, **args)
        require(ended[0], 'Trainer returned without on_train_end; completion not established')
        code = 0
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        status = 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else ('OOM' if isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower() else 'EXCEPTION')
        write_json(output/'training_state.json', dict(status=status, error=repr(error),
                   optimizer_steps_this_invocation=steps[0], updated=utc()))
        raise
    finally:
        write_json(output/'exit.json', dict(exit_code=code, ended_callback=ended[0], finished=utc()))
        if resume:
            claim.rmdir()  # Only our empty per-invocation claim; preserve all run/reservation evidence.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    p = commands.add_parser('plan', help='Freeze recipe/data/fingerprints, without creating a formal training run')
    p.add_argument('--variant', choices=VARIANTS, default='cbr_lif_lsrt_v1')
    for name in ('source','initialized','data','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--c2-args', type=Path, default=ARCHIVED_ARGS)
    p.add_argument('--project', type=Path, default=ROOT/'outputs/lsrt_v1/train')
    p.add_argument('--baseline-inventory', type=Path, default=ROOT/'docs/c19_lif_v1/checks.json')
    for mode in ('start', 'resume'):
        p = commands.add_parser(mode, help='Explicit formal training invocation; must run only after user authorization')
        for name in ('plan','preflight','output'):
            p.add_argument('--'+name, type=Path, required=True)
        if mode == 'resume':
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--previous-launch', type=Path, required=True)
    p = commands.add_parser('resources', help='Prepare native AMP check bus.jpg and yolo26n.pt using original helper')
    p.add_argument('--main', type=Path, default=MAIN)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('inventory', help='Freeze images/labels/config without inference')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('status')
    p.add_argument('--launch', type=Path, required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    if options.mode == 'plan':
        create_plan(options)
    elif options.mode in {'start','resume'}:
        run_training(options, resume=options.mode == 'resume')
    elif options.mode == 'resources':
        require(not options.output.exists(), 'Existing resource audit preserved')
        options.output.parent.mkdir(parents=True, exist_ok=True)
        ensure_amp_resources(options.main, options.output)
    elif options.mode == 'inventory':
        require(not options.output.exists(), 'Existing inventory preserved')
        write_json(options.output, dataset_inventory(options.data))
    else:
        path = options.launch/'training_state.json'
        print(json.dumps(read_json(path) if path.is_file() else dict(status='NOT_STARTED'), indent=2))


if __name__ == '__main__':
    main()
