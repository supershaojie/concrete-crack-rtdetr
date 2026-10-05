"""Read-only identity-checked packages, and explicit Git-trackable light archives."""
from __future__ import annotations
import io
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone
from configuration import candidate, frozen_config, RUN_ROOT, safe_run_id
from support import (HERE, ROOT, LOCK, adapter_hash, canonical, checked_source, digest, git, read_json,
                     sha256, verify_checkpoint_files)


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))
    return path


def verified_snapshot(run):
    from run import summary, legal_completion
    from data import verify_inputs
    config = frozen_config(run)
    paths = read_json(run/'runtime_paths.json')
    if 'source' in paths:
        checked_source(Path(paths['source']))
    snapshot = summary(run,save=False,quiet=True)
    if snapshot['training_status'] == 'completed':
        legal_completion(run,config)
        manifest = verify_inputs(run)
        from export import export_identity, evaluator_api
        for split in ('val','test'):
            status = snapshot['public_val_status' if split=='val' else 'final_test_status']
            if status != 'completed':
                continue
            expected, _, _ = export_identity(run,split,manifest,read_json(run/'source_identity.json'))
            pred = run/'predictions'/f'{split}.jsonl.gz'
            done = read_json(run/'predictions'/f'{split}_complete.json')
            identity, _ = evaluator_api().read_public(pred)
            metrics = read_json(run/'metrics'/f'{split}.json')
            seal = read_json(run/'metrics'/f'{split}_complete.json')
            if (identity != expected or done['sha256'] != sha256(pred)
                    or metrics['identity'] != expected or metrics['predictions_sha256'] != done['sha256']
                    or metrics['gt_sha256'] != sha256(run/'gt'/f'{split}.json')
                    or seal['sha256'] != sha256(run/'metrics'/f'{split}.json')):
                raise ValueError('Public result identity/checksum changed: ' + split)
    snapshot['package_scope'] = ('RESULTS_AFTER_LEGAL_TRAINING' if snapshot['tuning_status']=='completed'
                                 else 'SNAPSHOT_INCOMPLETE_NOT_FORMAL_RESULTS')
    if snapshot.get('scope') == 'SMOKE_ONLY':
        snapshot['package_scope'] = 'SMOKE_ONLY_SYNTHETIC_VERIFICATION'
    return config, snapshot


def pack(args):
    run = args.run
    _, snapshot = verified_snapshot(run)
    if args.include_weights and snapshot['training_status'] != 'completed':
        raise ValueError('--include-weights requires legally completed training; use a read-only configuration snapshot while active')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = args.out or ROOT/'outputs/yolo11m-packages'/f'{run.name}_{stamp}_{"weights" if args.include_weights else "review"}.tar.gz'
    destination = destination.absolute()
    if destination.is_relative_to(run):
        raise ValueError('Pack output must be outside the original run (the run is read-only)')
    # Whitelist evidence, never recursively copy dataset/cache/venv/prediction-side caches.
    allowed = {'.json','.yaml','.txt','.csv','.jsonl','.log','.gz'}
    files = {}
    for path in sorted(run.rglob('*')):
        relative = path.relative_to(run)
        if not path.is_file() or path.is_symlink() or any(p in ('cache','runtime','native_predict') for p in relative.parts):
            continue
        if path.suffix in allowed and not path.name.startswith('.') and (path.suffix != '.gz' or relative.parts[0]=='predictions'):
            files['run/'+relative.as_posix()] = path
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            files['implementation/'+path.relative_to(HERE).as_posix()] = path
    for path in (HERE.parent/'common.py',HERE.parent/'dataset.py',HERE.parent/'evaluation/evaluate.py',
                 HERE.parent/'configs/protocol.yaml',ROOT/'scripts/autodl_yolo11m_configurable.sh'):
        files['shared/'+path.relative_to(ROOT).as_posix()] = path
    weights = []
    if args.include_weights:
        verify_checkpoint_files(run)
        for name in ('best','last'):
            path = run/'train/weights'/f'{name}.pt'
            files['run/train/weights/'+path.name] = path
    for name in ('best','last'):
        path = run/'train/weights'/f'{name}.pt'
        weights.append({'kind':name,'original_path':str(path),'present':path.is_file(),
                        'sha256':sha256(path) if path.is_file() and snapshot['training_status']!='running' else None,
                        'included':args.include_weights})
    inventory = {name:{'original_path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path)} for name,path in files.items()}
    metadata = {'scope':snapshot['package_scope'], 'includes_weights':args.include_weights,
        'description':'best/last weights with evidence' if args.include_weights else 'review evidence; .pt excluded; not a weight backup',
        'run':str(run), 'weights':weights, 'summary':snapshot, 'files':inventory,
        'model_code_sha':read_json(run/'config_identity.json')['model_code_sha']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with destination.open('xb') as output, tarfile.open(fileobj=output,mode='w:gz') as archive:
            raw = canonical(metadata)
            info = tarfile.TarInfo('package_manifest.json'); info.size=len(raw)
            archive.addfile(info,io.BytesIO(raw))
            for name,path in files.items():
                # Reject files changed while taking an active snapshot; retain the partial package for inspection.
                if sha256(path) != inventory[name]['sha256']:
                    raise ValueError('Evidence changed during snapshot; retry after this epoch/stage: ' + name)
                archive.add(path,arcname=name,recursive=False)
                if sha256(path) != inventory[name]['sha256']:
                    raise ValueError('Evidence changed while packing: ' + name)
    except BaseException:
        # Do not remove a prior destination if exclusive creation failed.
        raise
    result = {'path':str(destination),'sha256':sha256(destination),'scope':metadata['scope'],
              'includes_weights':args.include_weights,'weights':weights,'files':len(files)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def archive_config(args):
    run = args.run
    run_id = safe_run_id(args.run_id or run.name)
    if run.is_dir():
        if args.config or args.set or args.clone_config_from:
            raise ValueError('An existing run archive must use frozen evidence; overrides are forbidden')
        config, snapshot = verified_snapshot(run)
        record = {'status':snapshot['package_scope'],'resolved_config':config,'summary':snapshot,
            'config_identity':read_json(run/'config_identity.json'),
            'cli_overrides':read_json(run/'cli_overrides.json'), 'runtime_paths':read_json(run/'runtime_paths.json')}
        for name in ('native_args','expanded_train_args','actual_training_setup','initialization',
                     'preflight_initialization','source_identity','environment','identity','launch_command'):
            if (run/(name+'.json')).is_file():
                record[name] = read_json(run/(name+'.json'))
    else:
        config, _, overrides, origin = candidate(args.config,args.set,args.clone_config_from)
        record = {'status':'NOT_RUN', 'resolved_config':config, 'cli_overrides':overrides, 'input':origin,
            'config_sha256':digest(canonical(config)), 'model_code_sha':git('rev-parse','HEAD'),
            'adapter_sha256':adapter_hash(), 'planned_initialization':read_json(LOCK),
            'dataset_identity_status':'NOT_RUN; server data still requires preflight',
            'reference_data':read_json(ROOT/'docs/comparison/evidence/yolov8m_delivery_validation.json')['data'],
            'training_state_inherited':False}
    destination = args.out or ROOT/'docs/comparison/archives/yolo11m'/f'{run_id}_{record["status"]}.json'
    record.update(run_id=run_id,archive_schema_version=1,
                  archive_created_utc=datetime.now(timezone.utc).isoformat(),
                  git_sync_status='FILE_GENERATED_ONLY; Git commit/push is a separate explicit step')
    exclusive_json(destination,record)
    print(json.dumps({'path':str(destination.absolute()),'sha256':sha256(destination),
                      'status':record['status'],'git_committed_or_pushed':False},ensure_ascii=False))
    return record
