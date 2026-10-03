"""YOLOv8m comparison: preflight, train, export, CPU evaluate, summary, check and measure."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid

from support import (ASSET, HERE, ROOT, SOURCE, canonical, checked_weight, configure, digest,
                     local_lock, read_json, recipe, run_identity, sha256, status, write_json,
                     initialization_type, initialization_record, native_recipe, validate_checkpoint,
                     runtime_environment, verify_environment)
from configuration import (DEFAULT_DATA, DEFAULT_DATA_ROOT, DEFAULT_SOURCE, DEFAULT_WEIGHT,
                           RUN_ROOT, freeze_config, frozen_config, frozen_paths, safe_run_id,
                           resolve_config, yaml_mapping)


def load_pretrained(source=SOURCE, weight=ASSET, config=None):
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import intersect_dicts
    from adapters import model_identity
    weight = checked_weight(weight)
    original = YOLO(str(weight), task='detect').model.float()
    coco = model_identity(original, 80)
    # Exactly the official DetectionTrainer.get_model adaptation to nc=1.
    model = DetectionModel(deepcopy(original.yaml), nc=1, verbose=False)
    model.load(original)
    target = model_identity(model, 1)
    transferred = intersect_dicts(original.state_dict(), model.state_dict())
    if not all(torch.equal(model.state_dict()[k], v) for k, v in transferred.items()):
        raise ValueError('Pretrained parameter transfer differs')
    record = {**initialization_record(source, config), 'coco': coco, 'crack': target, 'checkpoint_sha256': sha256(weight),
              'transferred_tensors': len(transferred), 'destination_tensors': len(model.state_dict()),
              'transferred_elements': sum(v.numel() for v in transferred.values()),
              'transferred_keys': sorted(transferred),
              'missing_or_reshaped_keys': sorted(set(model.state_dict())-set(transferred)),
              'pretrained_tensors_loaded':len(transferred), 'matching_values_verified_before_first_update':True,
              'skip_reason':'six class-output tensors change nc80 -> nc1; all 469 matching tensors loaded'}
    if len(transferred) != 469 or target['parameters_unfused'] != 25856899:
        raise ValueError('Official pretrained coverage/architecture changed')
    return model, record

def load_initial_model(source=SOURCE, weight=ASSET, config=None):
    initialization_type(config)
    return load_pretrained(source, weight, config)


def check(args):
    identity = configure(args.output.parent / 'check_runtime', args.source)
    import torch
    torch.set_num_threads(2)
    from ultralytics.cfg import get_cfg
    from bootstrap import environment_probe
    resolved = resolve_config(yaml_mapping(args.config.read_bytes())) if args.config else recipe()
    cfg = get_cfg(overrides=native_recipe(resolved))
    model, record = load_initial_model(args.source, args.weights, resolved)
    write_json(args.output, {'status': 'VERIFIED_IMPORT_CONFIG_AND_INITIALIZATION_NO_TRAINING',
               'source': identity, 'environment': environment_probe(), 'expanded_args': vars(cfg),
               'initialization': record, 'adapter_config':{'cutmix':resolved['cutmix'],
               'implementation':'augment_b19.DetectionCutMix', 'source':read_json(HERE/'augmentation.lock.json')},
               'frozen_recipe':resolved})
    print(f'CHECK PASSED: nc=1 M, {record["transferred_tensors"]}/{record["destination_tensors"]} tensors transferred', flush=True)


def train(args, manifest, source):
    import torch
    config = frozen_config(args.run)
    paths = frozen_paths(args.run)
    environment = verify_environment(args.run)
    from ultralytics.cfg import get_cfg
    from adapters import ComparisonTrainer
    run = args.run
    identity = run_identity(manifest, source, run_id=read_json(run/'run_id.json')['run_id'],
                            config=config, environment=environment, run_uuid=read_json(run/'run_id.json')['run_uuid'])
    previous = read_json(run / 'train_status.json') if (run / 'train_status.json').exists() else {}
    if previous.get('status') == 'completed':
        raise ValueError('Training is already completed; use export/evaluate/summary')
    overrides = native_recipe(config)
    overrides.update(model=str(checked_weight(paths['weights'])),
                     pretrained=True, resume=False, data=str(run / 'data.yaml'),
                     project=str(run), name='train', exist_ok=False)
    if args.resume:
        if not (run / 'identity.json').is_file() or read_json(run / 'identity.json') != identity:
            raise ValueError('Explicit resume requires identical code, adapter, dataset and frozen recipe')
        last = run / 'train/weights/last.pt'
        # Only this run's own last checkpoint; no arbitrary pickle or automatic latest-run selection.
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        validate_checkpoint(ckpt, identity, resume=True, config=config)
        best = torch.load(run / 'train/weights/best.pt', map_location='cpu', weights_only=False)
        if best.get('comparison_identity') != identity or best['epoch']+1 != ckpt['comparison_best_epoch']:
            raise ValueError('Best/last save was interrupted between files; checkpoint pair needs inspection')
        if overrides['patience'] and ckpt['epoch']+1-ckpt['comparison_best_epoch'] >= overrides['patience']:
            raise ValueError('Checkpoint already reached patience; resuming would change the stopping rule')
        write_json(run / 'resume_request.json', {'checkpoint': str(last), 'sha256': sha256(last),
                   'saved_epoch': ckpt['epoch']+1, 'identity': identity})
        # A signal can arrive after CSV append but before checkpoint save. Preserve
        # that log and restore the checkpoint's completed rows before appending.
        import pandas as pd
        from support import atomic_bytes
        csv_path = run / 'train/results.csv'
        if csv_path.exists():
            atomic_bytes(csv_path.with_name('results.pre_resume_' + uuid.uuid4().hex[:8] + '.csv'), csv_path.read_bytes())
        atomic_bytes(csv_path, pd.DataFrame(ckpt['train_results']).to_csv(index=False).encode())
        overrides['resume'] = str(last)
        overrides['model'] = str(last)
        del ckpt, best
    elif (run / 'train').exists() or (run / 'identity.json').exists():
        raise FileExistsError('Existing training output protected; use explicit --resume after inspecting failure')
    if os.environ.get('RANK', '-1') != '-1' or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This experiment is a single independent process, not DDP')
    get_cfg(overrides=overrides)  # fail on unsupported fields before trainer setup
    status(run, 'train', 'running', command=sys.argv, resume=args.resume)
    write_json(run / 'identity.json', identity)
    write_json(run / 'source_identity.json', source)
    write_json(run / 'frozen_recipe.json', config)
    (run / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    trainer = ComparisonTrainer(overrides=overrides, run=run, manifest=manifest, identity=identity,
                                config=config, reuse_cache=paths.get('label_cache'))
    write_json(run / 'expanded_train_args.json', vars(trainer.args))
    write_json(run / 'adapter_config.json', {'cutmix':config['cutmix'],
               'initialization_type':initialization_type(config), 'augmentation_source':read_json(HERE/'augmentation.lock.json')})

    def actual_setup(t):
        if t.batch_size != 16 or t.train_loader.batch_size != 16 or t.test_loader.batch_size != 16 or not t.amp:
            raise ValueError('Actual batch/validation batch/AMP differs from frozen recipe')
        expected_optimizer = getattr(torch.optim, config['optimizer'])
        if type(t.optimizer) is not expected_optimizer:
            raise ValueError('Actual optimizer differs from frozen configuration')
        nominal_accumulate = max(round(config['nbs'] / config['batch']), 1)
        effective_decay = config['weight_decay'] * config['batch'] * nominal_accumulate / config['nbs']
        if [g['weight_decay'] for g in t.optimizer.param_groups] != [0.0, effective_decay, 0.0]:
            raise ValueError('Actual native effective weight decay differs from run configuration')
        if any(g['initial_lr'] != config['lr0'] for g in t.optimizer.param_groups):
            raise ValueError('Actual initial group learning rate differs from frozen lr0')
        for group in t.optimizer.param_groups:
            if config['optimizer'] == 'SGD':
                if group['momentum'] != config['momentum'] or group['nesterov'] is not True:
                    raise ValueError('Actual SGD momentum/nesterov differs from native run configuration')
            elif group['betas'] != (config['momentum'], .999) or group['eps'] != 1e-8:
                raise ValueError('Actual Adam/AdamW beta/epsilon differs from native run configuration')
        if [n for n,p in t.model.named_parameters() if not p.requires_grad] != ['model.22.dfl.conv.weight']:
            raise ValueError('Only the native fixed DFL projection may be frozen')
        setup = {
            'optimizer': type(t.optimizer).__name__, 'batch': t.batch_size,
            'train_loader_batch': t.train_loader.batch_size, 'val_loader_batch': t.test_loader.batch_size,
            'train_workers': t.train_loader.num_workers, 'val_workers': t.test_loader.num_workers,
            'amp': t.amp, 'amp_probe': 'isolated actual model, CUDA, 64x64 FP32/FP16; allclose rtol=0.1 atol=0.5; all RNG streams restored',
            'nominal_nbs': t.args.nbs, 'accumulate_after_warmup': t.accumulate,
            'warmup_iterations': max(round(t.args.warmup_epochs*len(t.train_loader)), 100) if t.args.warmup_epochs > 0 else -1,
            'accumulation_warmup': 'round(linear interpolation 1 -> nbs/batch), minimum 1',
            'scheduler': {'type':'cosine' if config['cos_lr'] else 'linear',
                          'lrf':config['lrf'], 'epochs':config['epochs'], 'step':'native epoch start'},
            'parameter_groups': [{k: g.get(k) for k in ('lr', 'initial_lr', 'momentum', 'betas', 'eps', 'weight_decay', 'nesterov')}
                                  for g in t.optimizer.param_groups],
            'weight_decay_scaling': {'raw':config['weight_decay'], 'batch':config['batch'],
                                     'nominal_accumulate':nominal_accumulate, 'nbs':config['nbs'],
                                     'effective':effective_decay, 'rule':'raw * batch * nominal_accumulate / nbs'},
            'optimizer_field_semantics': 'momentum is SGD momentum or Adam/AdamW beta1; native beta2=0.999 and eps=1e-8',
            'native_validation': {'precision': 'actual input dtype recorded per validation in native_validation.json', 'conf': .001, 'iou': .7, 'max_det': 300,
                                   'rect': False, 'batch': 16, 'selection': 'full precision native mAP50-95'},
            'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                            'device': str(t.device), 'gpu': torch.cuda.get_device_name(t.device)},
            'augmentation':t.train_loader.dataset.transform_report,
            'label_cache':{s:getattr(loader.dataset,'label_cache_provenance',None)
                           for s,loader in (('train',t.train_loader),('val',t.test_loader))},
            'augmentation_close_zero_based_epoch':config['epochs']-config['close_mosaic'] if config['close_mosaic'] else None,
            'trainable_parameter_elements':sum(p.numel() for p in t.model.parameters() if p.requires_grad),
            'fixed_parameters':[n for n,p in t.model.named_parameters() if not p.requires_grad],
            'loss':{'raw':{k:config[k] for k in ('box','cls','dfl')},
                    'effective':{k:getattr(t.args,k) for k in ('box','cls','dfl')},
                    'rule':'native v8 detection gains; no adapter rescaling'},
            'config_sha256':identity['config_sha256'], 'resume':args.resume}
        if args.resume:
            write_json(run / ('actual_training_setup_resume_' + uuid.uuid4().hex[:8] + '.json'), setup)
        else:
            write_json(run / 'actual_training_setup.json', setup)

    def epoch_done(t):
        from augment_b19 import snapshot
        row = {'epoch': t.epoch+1, 'accumulate': t.accumulate, 'amp': t.amp, 'batch': t.batch_size,
               'learning_rates': t.lr, 'best_epoch': t.comparison_best_epoch, 'fitness': t.fitness,
               'augmentation':t.train_loader.dataset.transform_report['probabilities'],
               'augmentation_closed':t.train_loader.dataset.augmentation_closed,
               'augmentation_counts_cumulative':snapshot(t.train_loader.dataset),
               'losses': t.tloss.detach().cpu().tolist()}
        with (run / 'epoch_trace.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(canonical(row).decode())

    trainer.add_callback('on_pretrain_routine_end', actual_setup)
    trainer.add_callback('on_fit_epoch_end', epoch_done)
    trainer.train()
    completed = trainer.epoch + 1
    if completed != config['epochs'] and not (config['patience'] and
            completed-trainer.comparison_best_epoch >= config['patience']):
        raise ValueError('Native trainer returned before the configured epoch/patience stopping rule')
    result = {'completed_epoch': completed, 'best_epoch': trainer.comparison_best_epoch,
              'configured_epochs':config['epochs'],
              'stop_reason': 'epoch_limit' if completed == config['epochs'] else 'patience',
              'best_sha256': sha256(trainer.best), 'last_sha256': sha256(trainer.last),
              'best_checkpoint': str(trainer.best), 'last_checkpoint': str(trainer.last),
              'best_training_val_mAP50_95': trainer.best_fitness, 'exit_code': 0}
    status(run, 'train', 'completed', **result)
    return result


def summary(run):
    rows = {}
    states = {}
    errors = {}
    for split in ('val', 'test'):
        state_path = run / ('evaluate_' + split + '_status.json')
        state = read_json(state_path) if state_path.exists() else {}
        export_state_path = run / ('export_' + split + '_status.json')
        exported = read_json(export_state_path) if export_state_path.exists() else {}
        states[split] = (exported['status'] if exported.get('status') in ('failed','interrupted') else
                         state.get('status', 'exported_pending_evaluation' if exported.get('status') == 'completed'
                                   else exported.get('status','not_requested')))
        errors[split] = state.get('error') or exported.get('error')
        metrics = Path(state['metrics']) if states[split] == 'completed' else None
        rows[split] = read_json(metrics)['display_percent'] if metrics and metrics.is_file() else None
    training = read_json(run / 'train_status.json') if (run / 'train_status.json').exists() else {}
    progress = read_json(run / 'training_progress.json') if (run / 'training_progress.json').exists() else {}
    tuning = ('completed' if training.get('status') == 'completed' and rows['val'] is not None else
              'failed_or_interrupted' if training.get('status') in ('failed','interrupted') or
              states['val'] in ('failed','interrupted') else 'incomplete')
    state = 'finalized' if tuning == 'completed' and rows['test'] is not None else (
        'tuning_completed' if tuning == 'completed' else tuning)
    result = {'status': state, 'output': str(run), 'completed_epoch': training.get('completed_epoch', progress.get('completed_epoch')),
              'best_epoch': training.get('best_epoch', progress.get('best_epoch')), 'stop_reason': training.get('stop_reason'),
              'training_status':training.get('status','not_started'), 'tuning_status':tuning,
              'public_val_status':states['val'], 'final_test_status':states['test'],
              'training_exit_code':training.get('exit_code'),
              'model_code_sha':read_json(run/'config_identity.json')['model_code_sha'] if (run/'config_identity.json').exists() else None,
              'config_sha256':read_json(run/'config_identity.json')['config_sha256'] if (run/'config_identity.json').exists() else None,
              'best_sha256':training.get('best_sha256'),
              'errors':{'training':training.get('error'), 'public_val':errors['val'], 'final_test':errors['test']},
              'unified_metrics_percent': rows, 'native_training_selection_separate': True}
    write_json(run / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def measure(args):
    configure(args.output.parent / 'measure_runtime', args.source)
    import torch
    torch.set_num_threads(2)
    from thop import profile
    model, record = load_initial_model(args.source, args.weights)
    model = model.float().cpu().eval()
    values = {}
    for name, m in [('unfused', model), ('fused', deepcopy(model).fuse(verbose=False))]:
        with torch.no_grad():
            macs, _ = profile(m, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
        values[name] = {'parameters': sum(p.numel() for p in m.parameters()),
                        'MACs': macs, 'GFLOPs_2x_MACs': 2*macs/1e9}
    write_json(args.output, {'nc': 1, 'imgsz': [640, 640], 'device': 'cpu', 'precision': 'FP32',
               'method': 'ultralytics-thop; one multiply-add = 2 FLOPs; unsupported ops depend on THOP',
               'initialization':record,
               'speed': 'not measured; parallel GPU timing is not a paper speed benchmark', **values})


def prepare(args):
    run_id = safe_run_id(args.run_id)
    if run_id.startswith('SMOKE_'):
        raise ValueError('SMOKE_ run IDs are reserved for the internal synthetic verification script')
    run = (args.run or RUN_ROOT / run_id).resolve()
    if run.name != run_id:
        raise ValueError('Run directory basename must equal run-id')
    from data import discover_reuse_run
    reuse = args.reuse_run or discover_reuse_run(args.source, args.data_root)
    paths = {'python':str(Path(sys.executable).resolve()), 'source':str(args.source.resolve()),
             'weights':str(args.weights.resolve()), 'data':str(args.data.resolve()),
             'data_root':str(args.data_root.resolve()),
             'public_coco':str(args.public_coco.resolve()) if args.public_coco else None,
             'reuse_run':str(reuse.resolve()) if reuse else None,
             'label_cache':str((reuse/'cache').resolve()) if reuse else None}
    for field in ('python','weights','data'):
        if not Path(paths[field]).is_file():
            raise FileNotFoundError(field + ': ' + paths[field])
    for field in ('source','data_root'):
        if not Path(paths[field]).is_dir():
            raise FileNotFoundError(field + ': ' + paths[field])
    config = freeze_config(run, args.config.read_bytes(), paths, run_id)
    print(json.dumps({'run':str(run), 'run_id':run_id, 'config_sha256':digest(canonical(config)),
                      'source_yaml':'copied into run; no longer read', 'reused_inputs':paths['reuse_run']}), flush=True)
    return run


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'preflight', 'train', 'export', 'evaluate', 'summary', 'guard'):
        sub = commands.add_parser(command)
        sub.add_argument('--run', type=Path, required=command not in ('prepare','preflight'))
        if command in ('export', 'evaluate'):
            sub.add_argument('--split', choices=('val', 'test'), required=True)
        if command == 'train':
            sub.add_argument('--resume', action='store_true')
        if command in ('prepare','preflight'):
            sub.add_argument('--config', type=Path, required=command == 'prepare')
            sub.add_argument('--run-id', required=command == 'prepare')
            sub.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
            sub.add_argument('--weights', type=Path, default=DEFAULT_WEIGHT)
            sub.add_argument('--data', type=Path, default=DEFAULT_DATA)
            sub.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
            sub.add_argument('--public-coco', type=Path)
            sub.add_argument('--reuse-run', type=Path)
    for command in ('check', 'measure'):
        sub = commands.add_parser(command)
        sub.add_argument('--source', type=Path, default=SOURCE)
        sub.add_argument('--weights', type=Path, default=ASSET)
        if command == 'check':
            sub.add_argument('--config', type=Path)
        sub.add_argument('--output', type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command in ('check', 'measure'):
        if args.output.exists():
            raise FileExistsError('Output exists: ' + str(args.output))
        return globals()[args.command](args)
    if args.command == 'prepare' or (args.command == 'preflight' and args.config):
        if not args.config or not args.run_id:
            raise ValueError('A new run requires --config and --run-id')
        args.run = prepare(args)
        if args.command == 'prepare':
            return
    if args.run is None:
        raise ValueError('Use --run for an existing frozen run, or --config and --run-id for a new run')
    args.run = args.run.resolve()
    if not args.run.is_dir():
        raise FileNotFoundError('Run does not exist: ' + str(args.run))
    if args.command == 'summary':
        return summary(args.run)
    if args.command == 'guard':
        frozen_paths(args.run)
        with local_lock(args.run / '.pipeline.lock'), local_lock(args.run / '.run.lock'):
            return
    step = args.command + ('_' + args.split if hasattr(args, 'split') else '')
    interrupted = {'code': 130}
    def on_signal(number, frame):
        interrupted['code'] = 128 + number
        raise KeyboardInterrupt('signal ' + str(number))
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with ExitStack() as locks:
        if os.environ.get('YOLOV8M_PIPELINE_LOCKED') != str(args.run):
            locks.enter_context(local_lock(args.run / '.pipeline.lock'))
        locks.enter_context(local_lock(args.run / '.run.lock'))
        try:
            from data import preflight, verify_inputs
            config = frozen_config(args.run)
            paths = frozen_paths(args.run)
            os.environ['YOLOV8M_FROZEN_RUN'] = str(args.run)
            source = configure(args.run / 'runtime', Path(paths['source']))
            if args.command == 'preflight':
                status(args.run, step, 'running', command=sys.argv)
                environment = runtime_environment()
                if (args.run/'environment.json').exists():
                    verify_environment(args.run)
                else:
                    write_json(args.run/'environment.json', environment)
                from data import reuse_preflight
                result = (reuse_preflight(paths['reuse_run'], paths['data'], args.run, paths['data_root'])
                          if paths['reuse_run'] else preflight(paths['data'], args.run, paths['data_root'], paths['public_coco']))
                import torch
                torch.set_num_threads(2)
                model, initialization = load_initial_model(Path(paths['source']), Path(paths['weights']), config)
                write_json(args.run/'asset_reuse_check.json', {'source':source, 'environment':environment,
                    'initialization':initialization, 'installed_or_downloaded':False})
                write_json(args.run/'source_identity.json', source)
                del model
                status(args.run, step, 'completed', exit_code=0, counts=result['splits'])
            else:
                verify_environment(args.run)
                manifest = verify_inputs(args.run, ('train', 'val') if args.command == 'train' else (args.split,))
                if args.command == 'train':
                    return train(args, manifest, source)
                from export import evaluate_split, export_split
                status(args.run, step, 'running', command=sys.argv)
                result = (export_split(args.run, args.split, manifest, source) if args.command == 'export'
                          else evaluate_split(args.run, args.split, manifest, source))
                status(args.run, step, 'completed', exit_code=0,
                       **{k: result[k] for k in ('metrics', 'images', 'path') if k in result})
        except BaseException as exc:
            state = read_json(args.run / (step+'_status.json')) if (args.run / (step+'_status.json')).exists() else {}
            code = interrupted['code'] if isinstance(exc, KeyboardInterrupt) else 1
            error = {'error': str(exc), 'exit_code': code, 'command': sys.argv, 'traceback': traceback.format_exc()}
            # A rejected duplicate invocation does not erase prior successful evidence.
            if state.get('status') == 'completed' and step != 'train':
                write_json(args.run / (step+'_prior_completed_'+uuid.uuid4().hex[:8]+'.json'), state)
            if state.get('status') != 'completed' or step != 'train':
                status(args.run, step, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', **error)
            write_json(args.run / (step+'_last_error.json'), error)
            traceback.print_exc()
            raise SystemExit(code)


if __name__ == '__main__':
    main()
