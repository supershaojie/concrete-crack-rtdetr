"""YOLOv13l comparison: preflight, train, export, CPU evaluate, summary, check and measure."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid

from support import (HERE, ROOT, SOURCE, canonical, configure, digest,
                     local_lock, read_json, recipe, run_identity, sha256, status, write_json,
                     initialization_type, initialization_record, model_config, model_yaml, validate_checkpoint)


def load_initial_model(source=SOURCE):
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from adapters import model_identity, native_initialization
    init_seeds(42, deterministic=True)
    # Explicit scale on the pinned official dictionary; no YOLO(.pt), load() or state transfer.
    from support import no_external_initialization
    with no_external_initialization():
        model = DetectionModel(model_config(source), nc=1, verbose=False)
    record = {**model_identity(model,1), **initialization_record(source),
              'native_initialization_checks': native_initialization(model),
              'transferred_tensors':0, 'transferred_elements':0, 'transferred_keys':[],
              'destination_tensors':len(model.state_dict()),
              'construction':'DetectionModel(official YAML dict with scale=l, nc=1), weights=None'}
    return model, record


def check(args):
    identity = configure(args.output.parent / 'check_runtime', args.source)
    import torch
    torch.set_num_threads(2)
    from ultralytics.cfg import get_cfg
    from bootstrap import environment_probe
    cfg = get_cfg(overrides=recipe())
    model, record = load_initial_model(args.source)
    from backend import optional_flash_parity
    amp_record=None
    if torch.cuda.is_available():
        from adapters import strict_amp_probe
        strict_amp_probe(model.cuda())
        from adapters import AMP_PROBE_RECORD
        amp_record=AMP_PROBE_RECORD
    parity=optional_flash_parity()
    write_json(args.output, {'status': 'VERIFIED_IMPORT_CONFIG_AND_INITIALIZATION_NO_TRAINING',
               'amp_probe':amp_record,'flash_parity':parity,
               'source': identity, 'environment': environment_probe(), 'expanded_args': vars(cfg),
               'initialization': record, 'unsupported_inactive_fields': {'cutmix': 'absent in v8.3.63'}})
    print(f'CHECK PASSED: nc=1 L, {record["transferred_tensors"]}/{record["destination_tensors"]} tensors transferred', flush=True)


def train(args, manifest, source):
    import torch
    from bootstrap import environment_probe
    environment = environment_probe()
    from bootstrap import verify_frozen_environment
    verify_frozen_environment(environment)
    from ultralytics.cfg import get_cfg
    from adapters import ComparisonTrainer
    run = args.run
    identity = run_identity(manifest, source, run_id=read_json(run/'run_id.json')['run_id'])
    if identity['scope'] != 'formal':
        raise ValueError('Formal CLI rejects smoke run identities')
    previous = read_json(run / 'train_status.json') if (run / 'train_status.json').exists() else {}
    if previous.get('status') == 'completed':
        raise ValueError('Training is already completed; use export/evaluate/summary')
    overrides = recipe()
    overrides.update(model=str(model_yaml(args.source)),
                     pretrained=False, resume=False, data=str(run / 'data.yaml'),
                     project=str(run), name='train', exist_ok=True)
    if args.resume:
        if not (run / 'identity.json').is_file() or read_json(run / 'identity.json') != identity:
            raise ValueError('Explicit resume requires identical code, adapter, dataset and frozen recipe')
        last = run / 'train/weights/last.pt'
        pair=read_json(run/'checkpoint_pair.json')
        if sha256(last)!=pair['last_sha256'] or sha256(run/'train/weights/best.pt')!=pair['best_sha256']:
            raise ValueError('Checkpoint pair differs or was interrupted during save')
        # Only this run's own last checkpoint; no arbitrary pickle or automatic latest-run selection.
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        validate_checkpoint(ckpt, identity, resume=True)
        if ckpt['comparison_initialization_sha256'] != sha256(run/'initialization.json'):
            raise ValueError('Original initialization record changed')
        best = torch.load(run / 'train/weights/best.pt', map_location='cpu', weights_only=False)
        if best.get('comparison_identity') != identity or best['epoch']+1 != ckpt['comparison_best_epoch'] or best['best_fitness'] != ckpt['best_fitness']:
            raise ValueError('Best/last save was interrupted between files; checkpoint pair needs inspection')
        if ckpt['epoch']+1-ckpt['comparison_best_epoch'] >= overrides['patience']:
            raise ValueError('Checkpoint already reached patience; resuming would change the stopping rule')
        write_json(run / ('resume_request_'+uuid.uuid4().hex+'.json'), {'checkpoint': str(last), 'sha256': sha256(last),
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
    archive = run/('resume_audit_'+uuid.uuid4().hex) if args.resume else run
    write_json(archive / 'identity.json', identity)
    write_json(archive / 'source_identity.json', source)
    write_json(archive / 'environment.json', environment)
    write_json(archive / 'frozen_recipe.json', recipe())
    (archive / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    trainer = ComparisonTrainer(overrides=overrides, run=run, manifest=manifest, identity=identity)
    write_json(archive / 'expanded_train_args.json', vars(trainer.args))
    from support import atomic_bytes
    bootstrap_dir = ROOT/'.runtime/yolov13l-scratch'
    for name in ('bootstrap.json','environment_probe.json','base_environment_probe.json','environment_identity.json','pip_freeze.txt'):
        path=bootstrap_dir/name
        if path.exists(): atomic_bytes(archive/'bootstrap'/name, path.read_bytes())

    def actual_setup(t):
        if t.batch_size != 16 or t.train_loader.batch_size != 16 or t.test_loader.batch_size != 16 or not t.amp:
            raise ValueError('Actual batch/validation batch/AMP differs from frozen recipe')
        if not isinstance(t.optimizer, torch.optim.SGD):
            raise ValueError('Actual optimizer is not SGD')
        write_json(archive / 'actual_training_setup.json', {
            'optimizer': type(t.optimizer).__name__, 'batch': t.batch_size,
            'train_loader_batch': t.train_loader.batch_size, 'val_loader_batch': t.test_loader.batch_size,
            'train_workers': t.train_loader.num_workers, 'val_workers': t.test_loader.num_workers,
            'amp': t.amp, 'amp_probe': t.amp_probe_record,
            'nominal_nbs': t.args.nbs, 'accumulate_after_warmup': t.accumulate,
            'warmup_iterations': max(round(t.args.warmup_epochs*len(t.train_loader)), 100),
            'accumulation_warmup': 'round(linear interpolation 1 -> nbs/batch), minimum 1',
            'scheduler': 'cosine one_cycle(1,0.01,200); scheduler steps at each epoch start',
            'parameter_groups': [{k: g.get(k) for k in ('lr', 'initial_lr', 'momentum', 'weight_decay', 'nesterov')}
                                  for g in t.optimizer.param_groups],
            'weight_decay_scaling': '0.0005 * batch16 * accumulate4 / nbs64 = 0.0005',
            'attention_backend': 'native explicit matmul + stable softmax',
            'native_validation': {'precision': 'FP16 on CUDA', 'conf': .001, 'iou': .7, 'max_det': 300,
                                   'rect': False, 'batch': 16, 'selection': 'full precision native mAP50-95'},
            'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                            'device': str(t.device), 'gpu': torch.cuda.get_device_name(t.device)}})

    def epoch_done(t):
        row = {'epoch': t.epoch+1, 'accumulate': t.accumulate, 'amp': t.amp, 'batch': t.batch_size,
               'learning_rates': t.lr, 'best_epoch': t.comparison_best_epoch, 'fitness': t.fitness,
               'mosaic': t.train_loader.dataset.transforms.transforms[0].transforms[0].p,
               'mixup': t.train_loader.dataset.transforms.transforms[1].p,
               'losses': t.tloss.detach().cpu().tolist()}
        with (run / 'epoch_trace.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(canonical(row).decode())

    trainer.add_callback('on_pretrain_routine_end', actual_setup)
    trainer.add_callback('on_fit_epoch_end', epoch_done)
    trainer.train()
    completed = trainer.epoch + 1
    result = {'completed_epoch': completed, 'best_epoch': trainer.comparison_best_epoch,
              'stop_reason': 'epoch_limit' if completed == 200 else 'patience',
              'best_sha256': sha256(trainer.best), 'last_sha256': sha256(trainer.last),
              'best_checkpoint': str(trainer.best), 'last_checkpoint': str(trainer.last),
              'best_training_val_mAP50_95': trainer.best_fitness, 'exit_code': 0}
    status(run, 'train', 'completed', **result)
    return result


def summary(run):
    rows = {}
    for split in ('val', 'test'):
        state_path = run / ('evaluate_' + split + '_status.json')
        state = read_json(state_path) if state_path.exists() else {}
        metrics = Path(state['metrics']) if state.get('status') == 'completed' else None
        rows[split] = read_json(metrics)['display_percent'] if metrics and metrics.is_file() else None
    training = read_json(run / 'train_status.json') if (run / 'train_status.json').exists() else {}
    progress = read_json(run / 'training_progress.json') if (run / 'training_progress.json').exists() else {}
    failed = [read_json(p) for p in run.glob('*_status.json') if read_json(p).get('status') in ('failed', 'interrupted')]
    state = 'failed_or_interrupted' if failed else ('completed' if training.get('status') == 'completed' and all(rows.values()) else 'incomplete')
    result = {'status': state, 'output': str(run), 'completed_epoch': training.get('completed_epoch', progress.get('completed_epoch')),
              'best_epoch': training.get('best_epoch', progress.get('best_epoch')), 'stop_reason': training.get('stop_reason'),
              'unified_metrics_percent': rows, 'native_training_selection_separate': True,
              'initialization_type':'random','pretraining_source':None,'historical_RTDETR_initialization':'ImageNet backbone'}
    write_json(run / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def measure(args):
    configure(args.output.parent / 'measure_runtime', args.source)
    import torch
    torch.set_num_threads(2)
    from backend import ArithmeticAudit, native_fp32
    model,record=load_initial_model(args.source)
    model=model.float().cpu().eval()
    values={}
    for name,m in [('unfused',model),('fused',deepcopy(model).fuse(verbose=False))]:
        with torch.no_grad(),native_fp32(),ArithmeticAudit(require_fp32=True) as audit:
            m(torch.zeros(1,3,640,640))
        values[name]={'parameters':sum(p.numel() for p in m.parameters()),**audit.report()}
    write_json(args.output,{'nc':1,'imgsz':[640,640],'device':'cpu','precision':'FP32',
               'method':'Dispatcher counts Conv/Linear/mm/bmm including attention and HyperACE; MACs x2',
               'full_FLOPs':False,'initialization':record,
               'speed':'not measured; use speed command later on exclusive hardware',**values})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for command in ('preflight', 'train', 'export', 'evaluate', 'summary'):
        sub = commands.add_parser(command)
        sub.add_argument('--run', type=Path, required=True)
        if command in ('train', 'export'):
            sub.add_argument('--source', type=Path, default=SOURCE)
        if command in ('export', 'evaluate'):
            sub.add_argument('--split', choices=('val', 'test'), required=True)
        if command == 'train':
            sub.add_argument('--resume', action='store_true')
        if command == 'preflight':
            sub.add_argument('--data', type=Path, required=True)
            sub.add_argument('--data-root', type=Path)
            sub.add_argument('--public-coco', type=Path)
    for command in ('check', 'measure'):
        sub = commands.add_parser(command)
        sub.add_argument('--source', type=Path, default=SOURCE)
        sub.add_argument('--output', type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command in ('check', 'measure'):
        if args.output.exists():
            raise FileExistsError('Output exists: ' + str(args.output))
        return globals()[args.command](args)
    args.run = args.run.resolve()
    if args.command == 'preflight':
        args.run.mkdir(parents=True, exist_ok=False)
    elif not args.run.is_dir():
        raise FileNotFoundError('Run does not exist: ' + str(args.run))
    if args.command == 'summary':
        return summary(args.run)
    step = args.command + ('_' + args.split if hasattr(args, 'split') else '')
    interrupted = {'code': 130}
    def on_signal(number, frame):
        interrupted['code'] = 128 + number
        raise KeyboardInterrupt('signal ' + str(number))
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with local_lock(args.run / '.run.lock'):
        try:
            from data import preflight, verify_inputs
            if args.command == 'preflight':
                status(args.run, step, 'running', command=sys.argv)
                result = preflight(args.data, args.run, args.data_root, args.public_coco)
                write_json(args.run/'run_id.json', {'run_id':uuid.uuid4().hex})
                status(args.run, step, 'completed', exit_code=0, counts=result['splits'])
            else:
                manifest = verify_inputs(args.run, ('train', 'val') if args.command == 'train' else (args.split,))
                if args.command == 'train':
                    return train(args, manifest, configure(args.run / 'runtime', args.source))
                from export import evaluate_split, export_split
                source = configure(args.run / 'runtime', args.source) if args.command == 'export' else None
                status(args.run, step, 'running', command=sys.argv)
                result = (export_split(args.run, args.split, manifest, source) if args.command == 'export'
                          else evaluate_split(args.run, args.split))
                status(args.run, step, 'completed', exit_code=0,
                       **{k: result[k] for k in ('metrics', 'images', 'path') if k in result})
        except BaseException as exc:
            state = read_json(args.run / (step+'_status.json')) if (args.run / (step+'_status.json')).exists() else {}
            code = interrupted['code'] if isinstance(exc, KeyboardInterrupt) else 1
            error = {'error': str(exc), 'exit_code': code, 'command': sys.argv, 'traceback': traceback.format_exc()}
            # A rejected duplicate invocation does not erase prior successful evidence.
            if state.get('status') != 'completed':
                status(args.run, step, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', **error)
            write_json(args.run / (step+'_last_error.json'), error)
            traceback.print_exc()
            raise SystemExit(code)


if __name__ == '__main__':
    main()
