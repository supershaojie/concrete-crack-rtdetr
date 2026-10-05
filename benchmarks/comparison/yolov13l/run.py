"""YOLOv13-L comparison: preflight, train, export, CPU evaluate, summary, check and measure."""
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
                     runtime_environment, verify_environment, verify_checkpoint_files)
from configuration import (DEFAULT_DATA, DEFAULT_DATA_ROOT, DEFAULT_SOURCE, DEFAULT_WEIGHT,
                           RUN_ROOT, freeze_config, frozen_config, frozen_paths, safe_run_id,
                           resolve_config, yaml_mapping, candidate)


def load_pretrained(source=SOURCE, weight=ASSET, config=None):
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from adapters import model_identity, transfer_report
    weight = checked_weight(weight)
    original = YOLO(str(weight), task='detect').model.float()
    coco = model_identity(original, 80)
    # Exactly the official DetectionTrainer.get_model adaptation to nc=1.
    model = DetectionModel(deepcopy(original.yaml), nc=1, verbose=False)
    model.load(original)
    target = model_identity(model, 1)
    record = {**initialization_record(source, config), 'coco': coco, 'crack': target, 'checkpoint_sha256': sha256(weight),
              **transfer_report(original, model)}
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
    resolved, _, _, _ = candidate(args.config, args.set, args.clone_config_from)
    cfg = get_cfg(overrides=native_recipe(resolved))
    model, record = load_initial_model(args.source, args.weights, resolved)
    write_json(args.output, {'status': 'VERIFIED_IMPORT_CONFIG_AND_INITIALIZATION_NO_TRAINING',
               'source': identity, 'environment': environment_probe(), 'expanded_args': vars(cfg),
               'initialization': record, 'adapter_config':{'cutmix':resolved['cutmix'],
               'implementation':'augment_b19.DetectionCutMix', 'source':read_json(HERE/'augmentation.lock.json')},
               'frozen_recipe':resolved})
    print(f'CHECK PASSED: YOLOv13-L nc=1, {record["transferred_tensors"]}/{record["destination_tensors"]} tensors transferred', flush=True)


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
        verify_checkpoint_files(run)
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
        trace_path = run/'epoch_trace.jsonl'
        if trace_path.exists():
            atomic_bytes(trace_path.with_name('epoch_trace.pre_resume_'+uuid.uuid4().hex[:8]+'.jsonl'),trace_path.read_bytes())
        atomic_bytes(trace_path,b''.join(canonical(row) for row in ckpt['comparison_epoch_trace']))
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
        if (t.batch_size != config['batch'] or t.train_loader.batch_size != config['batch']
                or t.test_loader.batch_size != config['batch'] or t.test_loader.num_workers != 0
                or bool(t.amp) != config['amp']):
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
        if [n for n,p in t.model.named_parameters() if not p.requires_grad] != [f'model.{len(t.model.model)-1}.dfl.conv.weight']:
            raise ValueError('Only the native fixed DFL projection may be frozen')
        if not args.resume:
            from adapters import tensor_digest
            initialization = read_json(run/'initialization.json')
            state = t.model.state_dict()
            if any(tensor_digest(state[k]) != expected for k,expected in initialization['matching_tensor_sha256'].items()):
                raise ValueError('Transferred COCO tensors were reset before first optimizer update')
            write_json(run/'initialization_before_optimizer.json', {
                'status':'VERIFIED_ALL_MATCHING_VALUES', 'tensors':initialization['transferred_tensors'],
                'protected_tensors':initialization['protected_tensor_count'],
                'initialization_sha256':sha256(run/'initialization.json'), 'optimizer_steps':0})
        setup = {
            'optimizer': type(t.optimizer).__name__, 'batch': t.batch_size,
            'train_loader_batch': t.train_loader.batch_size, 'val_loader_batch': t.test_loader.batch_size,
            'train_workers': t.train_loader.num_workers, 'val_workers': t.test_loader.num_workers,
            'amp': t.amp, 'amp_probe': 'isolated actual model, CUDA, 64x64 FP32/FP16; allclose rtol=0.1 atol=0.5; all RNG streams restored',
            'amp_arithmetic_probe':t.amp_probe_record if t.amp else {'status':'NOT_REQUESTED'},
            'attention_backend':'native', 'flash_parity':'NOT_VERIFIED',
            'tf32':{'matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn':torch.backends.cudnn.allow_tf32},
            'scaler_initial_scale_after_setup':t.scaler.get_scale(),
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
                                   'rect': False, 'batch': config['batch'], 'workers':0, 'selection': 'full precision native mAP50-95',
                                   'native_fitness_definition':'pinned author weights [0,0,0,0,1] for P/R/AP50/AP75/mAP50-95'},
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
                    'rule':'pinned author detection gains; no adapter rescaling'},
            'config_sha256':identity['config_sha256'], 'resume':args.resume}
        if args.resume:
            write_json(run / ('actual_training_setup_resume_' + uuid.uuid4().hex[:8] + '.json'), setup)
        else:
            write_json(run / 'actual_training_setup.json', setup)

    def epoch_done(t):
        row = t.comparison_epoch_trace[-1]
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


def current_state(path):
    if not Path(path).is_file():
        return {}
    state = read_json(path)
    if state.get('status') == 'running' and state.get('pid'):
        import psutil
        try:
            process = psutil.Process(state['pid'])
            alive = (process.create_time() == state.get('process_created')
                     and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            alive = False
        if not alive:
            state = {**state, 'status':'failed', 'error':'Stage process exited without recording completion',
                     'exit_code':state.get('exit_code'), 'inferred_after_process_exit':True}
    return state


def summary(run, save=True, quiet=False):
    rows = {}
    states = {}
    errors = {}
    for split in ('val', 'test'):
        state_path = run / ('evaluate_' + split + '_status.json')
        state = current_state(state_path)
        export_state_path = run / ('export_' + split + '_status.json')
        exported = current_state(export_state_path)
        states[split] = (exported['status'] if exported.get('status') in ('failed','interrupted') else
                         state.get('status', 'exported_pending_evaluation' if exported.get('status') == 'completed'
                                   else exported.get('status','not_requested')))
        errors[split] = state.get('error') or exported.get('error')
        metrics = Path(state['metrics']) if states[split] == 'completed' else None
        rows[split] = read_json(metrics)['display_percent'] if metrics and metrics.is_file() else None
    training = current_state(run/'train_status.json')
    preflight_state = current_state(run/'preflight_status.json')
    progress = read_json(run / 'training_progress.json') if (run / 'training_progress.json').exists() else {}
    tuning = ('completed' if training.get('status') == 'completed' and rows['val'] is not None else
              'failed_or_interrupted' if training.get('status') in ('failed','interrupted') or
              states['val'] in ('failed','interrupted') else 'incomplete')
    state = 'finalized' if tuning == 'completed' and rows['test'] is not None else (
        'tuning_completed' if tuning == 'completed' else tuning)
    if training.get('status') == 'running':
        state = 'training'
    elif preflight_state.get('status') == 'running':
        state = 'preflight'
    elif preflight_state.get('status') in ('failed','interrupted') and not training:
        state = 'preflight_failed'
    elif not training and not preflight_state:
        state = 'not_started'
    result = {'status': state, 'output': str(run), 'completed_epoch': training.get('completed_epoch', progress.get('completed_epoch')),
              'scope':read_json(run/'config_identity.json').get('scope','CONFIGURABLE_FORMAL') if (run/'config_identity.json').exists() else None,
              'best_epoch': training.get('best_epoch', progress.get('best_epoch')), 'stop_reason': training.get('stop_reason'),
              'preflight_status':preflight_state.get('status','not_started'),
              'training_status':training.get('status','not_started'), 'tuning_status':tuning,
              'public_val_status':states['val'], 'final_test_status':states['test'],
              'training_exit_code':training.get('exit_code'),
              'model_code_sha':read_json(run/'config_identity.json')['model_code_sha'] if (run/'config_identity.json').exists() else None,
              'config_sha256':read_json(run/'config_identity.json')['config_sha256'] if (run/'config_identity.json').exists() else None,
              'best_sha256':training.get('best_sha256'),
              'last_sha256':training.get('last_sha256'),
              'best_checkpoint':training.get('best_checkpoint'), 'last_checkpoint':training.get('last_checkpoint'),
              'dataset_identity_sha256':read_json(run/'manifest.json')['dataset_identity_sha256'] if (run/'manifest.json').exists() else None,
              'errors':{'training':training.get('error'), 'public_val':errors['val'], 'final_test':errors['test']},
              'unified_metrics_percent': rows, 'native_training_selection_separate': True}
    if save:
        write_json(run / 'summary.json', result)
    if not quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def measure(args):
    configure(args.output.parent / 'measure_runtime', args.source)
    import torch
    torch.set_num_threads(2)
    from backend import ArithmeticAudit, native_fp32
    model, record = load_initial_model(args.source, args.weights)
    model = model.float().cpu().eval()
    values = {}
    for name, m in [('unfused', model), ('fused', deepcopy(model).fuse(verbose=False))]:
        with torch.no_grad(), native_fp32(), ArithmeticAudit(require_fp32=True) as audit:
            m(torch.zeros(1,3,640,640))
        counted = audit.report()
        values[name] = {'parameters': sum(p.numel() for p in m.parameters()),
                        **counted}
    write_json(args.output, {'nc': 1, 'imgsz': [640, 640], 'device': 'cpu', 'precision': 'FP32',
               'method': 'TorchDispatch Conv/mm/bmm count including AAttn and HyperACE; MACs x 2; exclusions explicitly recorded, not complete FLOPs',
               'initialization':record,
               'speed': 'not measured; parallel GPU timing is not a paper speed benchmark', **values})


def prepare(args):
    run_id = safe_run_id(args.run_id)
    if run_id.startswith('SMOKE_'):
        raise ValueError('SMOKE_ IDs are reserved for explicit synthetic verification')
    run = (args.run or RUN_ROOT/run_id).resolve()
    if run.name != run_id:
        raise ValueError('Run basename must equal run-id')
    if run.exists():
        raise FileExistsError('Protected existing run: ' + str(run))
    resolved, raw, overrides, origin = candidate(args.config, args.set, args.clone_config_from)
    if Path(sys.prefix).absolute() == Path(sys.base_prefix).absolute():
        raise ValueError('Use the isolated venv interpreter, not the mother Python')
    paths = {'python':str(Path(sys.executable).absolute()), 'sys_prefix':str(Path(sys.prefix).absolute()),
             'source':str(args.source.resolve()), 'weights':str(args.weights.resolve()),
             'data':str(args.data.resolve()), 'data_root':str(args.data_root.resolve()),
             'public_coco':str(args.public_coco.resolve()) if args.public_coco else None,
             'reuse_run':str(args.reuse_run.resolve()) if args.reuse_run else None,
             'label_cache':None}
    for field in ('python','weights','data'):
        if not Path(paths[field]).is_file():
            raise FileNotFoundError(field + ': ' + paths[field])
    for field in ('source','data_root'):
        if not Path(paths[field]).is_dir():
            raise FileNotFoundError(field + ': ' + paths[field])
    checked_weight(paths['weights'])
    # Import only after strict candidate validation and dedicated settings setup.
    source = configure(ROOT/'.runtime/yolov13l-configurable/prepare', args.source)
    freeze_config(run, raw, paths, run_id, overrides=overrides, origin=origin)
    write_json(run/'source_identity.json', source)
    write_json(run/'environment.json', runtime_environment())
    status(run, 'prepared', 'completed', exit_code=0)
    print(json.dumps({'run':str(run), 'run_id':run_id, 'config_sha256':digest(canonical(resolved)),
        'initialization':'official COCO asset; resume=false; cloned recipes never inherit training state'}), flush=True)
    return run


def legal_completion(run, config):
    training = read_json(Path(run)/'train_status.json')
    completed, best = training.get('completed_epoch',0), training.get('best_epoch',0)
    valid = (training.get('stop_reason') == 'epoch_limit' and completed == config['epochs']) or (
        training.get('stop_reason') == 'patience' and config['patience'] > 0 and
        completed-best >= config['patience'] and completed < config['epochs'])
    if (training.get('status') != 'completed' or training.get('exit_code') != 0 or not valid
            or not 1 <= best <= completed <= config['epochs']):
        raise ValueError('finalize requires evidence of legal epoch-limit/patience completion')
    seals = verify_checkpoint_files(run)
    if any(seals[k+'_sha256'] != training[k+'_sha256'] for k in ('best','last')):
        raise ValueError('Completed checkpoint identity differs')
    return training


def guard(run, operation):
    config = frozen_config(run)
    paths = frozen_paths(run)
    # Resume checkpoints contain author model objects: pin imports/backend before
    # deserializing any verified local checkpoint, including this launcher guard.
    configure(run/'runtime',Path(paths['source']))
    verify_environment(run)
    with local_lock(run/'.pipeline.lock'), local_lock(run/'.run.lock'):
        if operation == 'finalize':
            legal_completion(run, config)
        elif operation == 'resume':
            training = read_json(run/'train_status.json') if (run/'train_status.json').exists() else {}
            if training.get('status') == 'completed':
                raise ValueError('Completed training cannot resume; use finalize')
            verify_checkpoint_files(run)
            import torch
            ckpt = torch.load(run/'train/weights/last.pt',map_location='cpu',weights_only=False)
            validate_checkpoint(ckpt,read_json(run/'identity.json'),resume=True,config=config)
            if config['patience'] and ckpt['epoch']+1-ckpt['comparison_best_epoch'] >= config['patience']:
                raise ValueError('Checkpoint already reached patience')


def add_candidate_options(sub):
    group = sub.add_mutually_exclusive_group()
    group.add_argument('--config',type=Path)
    group.add_argument('--clone-config-from',type=Path)
    sub.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    for command in ('config','check-config'):
        sub = commands.add_parser(command,help='Generate/validate a candidate without model creation or training')
        add_candidate_options(sub)
        sub.add_argument('--out',type=Path,required=command=='config')
    for command in ('prepare','preflight','train','export','evaluate','summary','status','guard','pack','archive-config'):
        sub = commands.add_parser(command)
        sub.add_argument('--run',type=Path)
        sub.add_argument('--run-id')
        if command in ('export','evaluate'):
            sub.add_argument('--split',choices=('val','test'),required=True)
        if command == 'train':
            sub.add_argument('--resume',action='store_true')
        if command == 'guard':
            sub.add_argument('--operation',choices=('resume','finalize'),required=True)
        if command in ('prepare','archive-config'):
            add_candidate_options(sub)
        if command == 'prepare':
            sub.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
            sub.add_argument('--weights',type=Path,default=DEFAULT_WEIGHT)
            sub.add_argument('--data',type=Path,default=DEFAULT_DATA)
            sub.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT)
            sub.add_argument('--public-coco',type=Path)
            sub.add_argument('--reuse-run',type=Path)
        if command in ('pack','archive-config'):
            sub.add_argument('--out',type=Path)
        if command == 'pack':
            sub.add_argument('--include-weights',action='store_true')
    for command in ('check','measure'):
        sub = commands.add_parser(command)
        sub.add_argument('--source',type=Path,default=SOURCE)
        sub.add_argument('--weights',type=Path,default=ASSET)
        sub.add_argument('--output',type=Path,required=True)
        if command == 'check':
            add_candidate_options(sub)
    return p


def main():
    args = parser().parse_args()
    if args.command in ('config','check-config'):
        from configuration import schema
        resolved, _, overrides, origin = candidate(args.config,args.set,args.clone_config_from)
        result = {'status':'CONFIG_VALID_NO_MODEL_OR_TRAINING','resolved_config':resolved,
                  'config_sha256':digest(canonical(resolved)), 'cli_overrides':overrides, 'input':origin}
        if args.out:
            import yaml
            args.out.parent.mkdir(parents=True,exist_ok=True)
            with args.out.open('xb') as stream:
                stream.write(yaml.safe_dump(resolved,sort_keys=False).encode('utf-8'))
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return
    if args.command in ('check','measure'):
        if args.output.exists():
            raise FileExistsError('Output exists: ' + str(args.output))
        return globals()[args.command](args)
    if args.run_id:
        safe_run_id(args.run_id)
    if args.run and args.run_id and args.run.name != args.run_id:
        raise ValueError('--run basename and --run-id differ')
    args.run = (args.run or (RUN_ROOT/args.run_id if args.run_id else None))
    if args.run is None:
        raise ValueError('Specify --run or --run-id')
    args.run = args.run.resolve()
    if args.command == 'archive-config':
        from packaging_run import archive_config
        return archive_config(args)
    if args.command == 'prepare':
        if not args.run_id:
            raise ValueError('prepare requires --run-id')
        return prepare(args)
    if args.command == 'status' and not args.run.exists():
        print(json.dumps({'status':'not_started','run':str(args.run),'final_test_status':'not_requested'}))
        return
    if not args.run.is_dir():
        raise FileNotFoundError('Run does not exist: ' + str(args.run))
    if args.command in ('summary','status'):
        return summary(args.run,save=args.command=='summary')
    if args.command == 'pack':
        from packaging_run import pack
        return pack(args)
    if args.command == 'guard':
        return guard(args.run,args.operation)
    step = args.command + ('_' + args.split if hasattr(args,'split') else '')
    interrupted = {'code':130}
    def on_signal(number,frame):
        interrupted['code'] = 128+number
        raise KeyboardInterrupt('signal ' + str(number))
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    with ExitStack() as locks:
        if os.environ.get('YOLOV13L_PIPELINE_LOCKED') != str(args.run):
            locks.enter_context(local_lock(args.run/'.pipeline.lock'))
        locks.enter_context(local_lock(args.run/'.run.lock'))
        try:
            from data import preflight, verify_inputs
            config = frozen_config(args.run)
            paths = frozen_paths(args.run)
            os.environ['YOLOV13L_FROZEN_RUN'] = str(args.run)
            source = configure(args.run/'runtime',Path(paths['source']))
            verify_environment(args.run)
            if args.command == 'preflight':
                status(args.run,step,'running',command=[sys.executable,*sys.argv])
                from data import reuse_preflight
                result = verify_inputs(args.run) if (args.run/'input_checksums.json').exists() else (
                          reuse_preflight(paths['reuse_run'],paths['data'],args.run,paths['data_root'])
                          if paths['reuse_run'] else preflight(paths['data'],args.run,paths['data_root'],paths['public_coco']))
                import torch
                torch.set_num_threads(2)
                from verification import preflight_model_probe
                model, initialization = load_initial_model(Path(paths['source']),Path(paths['weights']),config)
                write_json(args.run/'preflight_initialization.json',initialization)
                probe = preflight_model_probe(model,config,args.run,result)
                write_json(args.run/'preflight_model_probe.json',probe)
                write_json(args.run/'asset_reuse_check.json',{'source':source,
                    'initialization':initialization,'installed_or_downloaded':False})
                del model
                status(args.run,step,'completed',exit_code=0,counts=result['splits'])
            else:
                manifest = verify_inputs(args.run,('train','val') if args.command=='train' else (args.split,))
                if args.command == 'train':
                    return train(args,manifest,source)
                from export import evaluate_split, export_split
                status(args.run,step,'running',command=[sys.executable,*sys.argv])
                result = (export_split(args.run,args.split,manifest,source) if args.command=='export'
                          else evaluate_split(args.run,args.split,manifest,source))
                status(args.run,step,'completed',exit_code=0,
                       **{k:result[k] for k in ('metrics','images','path') if k in result})
        except BaseException as exc:
            state_path = args.run/(step+'_status.json')
            state = read_json(state_path) if state_path.exists() else {}
            code = interrupted['code'] if isinstance(exc,KeyboardInterrupt) else 1
            error = {'error':str(exc),'exit_code':code,'command':[sys.executable,*sys.argv],
                     'traceback':traceback.format_exc()}
            if state.get('status') == 'completed' and step != 'train':
                write_json(args.run/(step+'_prior_completed_'+uuid.uuid4().hex[:8]+'.json'),state)
            if state.get('status') != 'completed' or step != 'train':
                status(args.run,step,'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',**error)
            write_json(args.run/(step+'_last_error.json'),error)
            traceback.print_exc()
            raise SystemExit(code)

if __name__ == '__main__':
    main()
