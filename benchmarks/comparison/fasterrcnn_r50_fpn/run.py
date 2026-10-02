"""Isolated Faster R-CNN: preflight, train, export, CPU evaluate, summary, check, measure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid

from support import (HERE, ENV, RUNTIME, canonical, code_identity, environment, local_lock, read_json,
                     recipe, run_identity, status, write_json)


def summary(run):
    training = read_json(run/'train_status.json') if (run/'train_status.json').exists() else {}
    progress = read_json(run/'training_progress.json') if (run/'training_progress.json').exists() else {}
    rows = {}
    for split in ('val', 'test'):
        state_path = run/('evaluate_'+split+'_status.json')
        state = read_json(state_path) if state_path.exists() else {}
        result_path = Path(state['metrics']) if state.get('status') == 'completed' else None
        rows[split] = read_json(result_path)['display_percent'] if result_path and result_path.is_file() else None
    failures = [read_json(p) for p in run.glob('*_status.json') if read_json(p).get('status') in ('failed', 'interrupted')]
    result = {'status': 'failed_or_interrupted' if failures else (
        'completed' if training.get('status') == 'completed' and all(rows.values()) else 'incomplete'),
        'run': str(run), 'model': 'Faster R-CNN (ResNet-50-FPN)', 'initialization_type': 'random',
        'reference_RTDETR_initialization': 'ImageNet backbone; initialization conditions differ',
        'completed_epoch': training.get('completed_epoch', progress.get('epoch')),
        'best_epoch': training.get('best_epoch', progress.get('best_epoch')),
        'stop_reason': training.get('stop_reason', training.get('status')),
        'failure_reasons': [{k: row.get(k) for k in ('stage', 'error', 'exit_code')} for row in failures],
        'metrics_percent': rows, 'metric_policy': 'corrected_sorted_conf_mask_v1',
        'same_best_sha256': training.get('best_sha256'), 'missing_metrics_are_not_zero': True}
    write_json(run/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def strict_identity(run, manifest):
    from bootstrap import assert_interpreter
    assert_interpreter(Path(sys.executable), ENV)
    recorded = Path((RUNTIME/'python_path.txt').read_text().strip())
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(str(recorded)):
        raise ValueError('Current interpreter differs from bootstrap receipt')
    env = environment(strict=True)
    cfg = recipe()
    info = read_json(run/'run_id.json')
    if info['scope'] != 'FORMAL':
        raise ValueError('Formal CLI refuses smoke/local-data-check runs')
    probe = read_json(run/'preflight_model.json')
    if probe['environment'] != env:
        raise ValueError('Interpreter/package/binary/CUDA environment changed since isolated preflight')
    if probe['status'] != 'PASSED_TARGET_VERSION_ISOLATED_PROBE' or read_json(run/'preflight_identity.json') != {
            'code_sha256': code_identity(), 'python': os.path.abspath(sys.executable)}:
        raise ValueError('Missing/stale target-version isolated preflight')
    ident = run_identity(manifest, info['run_uuid'], cfg, env)
    return ident, cfg, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for command in ('preflight', 'train', 'export', 'evaluate', 'summary'):
        sub = commands.add_parser(command)
        sub.add_argument('--run', type=Path, required=True)
        if command == 'preflight':
            sub.add_argument('--data', type=Path, required=True)
            sub.add_argument('--data-root', type=Path)
            sub.add_argument('--public-coco', type=Path)
            sub.add_argument('--local-data-only', action='store_true', help='Light local data check; cannot be trained')
        if command == 'train':
            sub.add_argument('--resume', action='store_true')
        if command in ('export', 'evaluate'):
            sub.add_argument('--split', choices=('val', 'test'), required=True)
    sub = commands.add_parser('check')
    sub.add_argument('--output', type=Path, required=True)
    sub.add_argument('--local-compat', action='store_true', help='Explicitly non-target local smoke, never authorizes training')
    sub = commands.add_parser('measure')
    sub.add_argument('--output', type=Path, required=True)
    sub.add_argument('--local-compat', action='store_true')
    sub = commands.add_parser('speed')
    sub.add_argument('--run', type=Path, required=True)
    sub.add_argument('--output', type=Path, required=True)
    sub.add_argument('--confirm-exclusive', action='store_true')
    a = p.parse_args()
    if a.command in ('check', 'measure', 'speed'):
        if a.output.exists():
            raise FileExistsError('Existing measurement/probe output protected')
        if a.command == 'check':
            from probe import check
            return check(a.output, a.local_compat)
        from measure import measure, speed
        return measure(a.output, a.local_compat) if a.command == 'measure' else speed(a)
    a.run = a.run.resolve()
    if a.command == 'preflight':
        a.run.mkdir(parents=True, exist_ok=False)
    elif not a.run.is_dir():
        raise FileNotFoundError('Missing explicit run directory')
    if a.command == 'summary':
        return summary(a.run)
    stage = a.command + ('_'+a.split if hasattr(a, 'split') else '')
    exit_code = 130
    def interrupt(number, frame):
        nonlocal exit_code
        exit_code = 128+number
        raise KeyboardInterrupt('signal '+str(number))
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    with local_lock(a.run/'.run.lock'):
        try:
            from data import preflight, verify_inputs
            if a.command == 'preflight':
                status(a.run, stage, 'running')
                manifest = preflight(a.data, a.run, a.data_root, a.public_coco)
                write_json(a.run/'run_id.json', {'run_uuid': uuid.uuid4().hex,
                    'scope': 'LOCAL_DATA_CHECK_ONLY' if a.local_data_only else 'FORMAL'})
                if not a.local_data_only:
                    subprocess.run([sys.executable, str(HERE/'run.py'), 'check', '--output',
                                    str(a.run/'preflight_model.json')], check=True)
                    write_json(a.run/'preflight_identity.json', {'code_sha256': code_identity(),
                                                               'python': os.path.abspath(sys.executable)})
                status(a.run, stage, 'completed', exit_code=0, counts=manifest['splits'],
                       model_probe='NOT_RUN_LOCAL_DATA_CHECK' if a.local_data_only else 'PASSED_TARGET_VERSION')
                return
            if a.command == 'evaluate':
                # Recalculation consumes immutable GT/prediction caches only: no source image IO, GPU or detector.
                from export import evaluate_split
                status(a.run, stage, 'running')
                result = evaluate_split(a.run, a.split)
            else:
                manifest = verify_inputs(a.run, ('train', 'val') if a.command == 'train' else (a.split,))
                ident, cfg, env = strict_identity(a.run, manifest)
                if a.command == 'train':
                    from engine import train
                    return train(a.run, manifest, ident, cfg, env, resume=a.resume)
                if read_json(a.run/'identity.json') != ident:
                    raise ValueError('Export code/model/run identity changed')
                from export import export_split
                status(a.run, stage, 'running')
                result = export_split(a.run, a.split, manifest, ident, cfg)
            status(a.run, stage, 'completed', exit_code=0,
                   **{k: result[k] for k in ('metrics', 'path', 'images') if k in result})
        except BaseException as exc:
            code = exit_code if isinstance(exc, KeyboardInterrupt) else 1
            state_path = a.run/(stage+'_status.json')
            old = read_json(state_path) if state_path.exists() else {}
            error = {'error': str(exc), 'traceback': traceback.format_exc(), 'exit_code': code, 'argv': sys.argv}
            if old.get('status') != 'completed':
                status(a.run, stage, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', **error)
            write_json(a.run/(stage+'_error_'+uuid.uuid4().hex+'.json'), error)
            traceback.print_exc()
            raise SystemExit(code)


if __name__ == '__main__':
    main()
