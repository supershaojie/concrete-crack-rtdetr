"""Read-only comparison preparation: evidence, dataset audit, COCO conversion; no training/inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from common import canonical, digest, load_yaml, new_output, sha256, write_json
from assets import build_assets
from augmentation import build_table, probe_transforms
from dataset import audit_dataset, finalize_fingerprints, SPLITS
from convert_annotations import convert

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=PROJECT, help='Read-only project/data path resolution base')
    p.add_argument('--data', type=Path, help='Actual data YAML, or archived frozen copy with explicit --data-root')
    p.add_argument('--data-root', type=Path, help='Explicit root migration mapping; does not rewrite the YAML')
    p.add_argument('--family-map', type=Path)
    p.add_argument('--dataset-manifest', type=Path, help='Reuse an explicitly supplied completed audit, without rescanning; recorded as prior evidence')
    p.add_argument('--assets-config', type=Path, default=HERE / 'configs/asset_locations.yaml')
    p.add_argument('--asset-index', type=Path, help='Reuse an earlier index and its raw small-file copies; no weight rescan')
    p.add_argument('--profile', choices=['windows', 'server'], default='windows' if os.name == 'nt' else 'server')
    p.add_argument('--run', action='append', default=[], metavar='MODEL=PATH', help='Override model roots; repeat for multiple roots of same model')
    p.add_argument('--output', type=Path, required=True, help='New, previously nonexistent output directory')
    p.add_argument('--extract-caches', action='store_true', help='Copy only compressed high-precision prediction caches to ignored outputs; never weights')
    p.add_argument('--probe-transforms', action='store_true', help='Optional CPU construction of transform objects, no detector or images')
    p.add_argument('--missing-labels', choices=['unknown','negative'], default='unknown')
    args = p.parse_args()
    if args.dataset_manifest and (args.data or args.data_root or args.family_map):
        p.error('--dataset-manifest cannot be combined with new --data/--data-root/--family-map inputs')
    if args.asset_index and (args.run or args.extract_caches):
        p.error('--asset-index cannot be combined with --run or --extract-caches')
    return args


def prepare(args, output):
    registry = load_yaml(HERE / 'configs/model_registry.yaml')
    locations = load_yaml(args.assets_config)[args.profile]
    overridden = set()
    for value in args.run:
        model, separator, path = value.partition('=')
        if not separator or model not in registry['historical_experiments'] or not path:
            raise ValueError('--run must be c2|cbr|lif|mother=PATH')
        if model not in overridden:
            locations[model] = []; overridden.add(model)
        locations[model].append(path)
    if args.asset_index:
        index = json.loads(args.asset_index.read_text(encoding='utf-8'))
        contents = {k: {r['source']: Path(r['archived_copy']).read_bytes() for r in v['files'] if 'archived_copy' in r}
                    for k,v in index.items()}
        for model in index.values():
            model['reused_index'] = {'source': str(args.asset_index), 'sha256': sha256(args.asset_index), 'assets_rescanned': False}
        write_json(output / 'asset_index.json', index)
    else:
        index, contents = build_assets(locations, output, registry['historical_experiments'], args.extract_caches)
    mother = index['mother']
    actual = mother['actual_training_args']
    raw = contents['mother']
    historical = next((json.loads(v) for k,v in raw.items() if k.endswith('dataset_inventory.json')), None)
    freeze = next((v.decode('utf-8-sig') for k,v in raw.items() if k.endswith('pip_freeze.txt')), None)
    augmentation = build_table(actual, PROJECT, HERE / 'configs/mother_augmentation.yaml', freeze)
    augmentation['args_source'] = mother['args_source']
    if args.probe_transforms:
        augmentation['runtime_probe'] = probe_transforms(PROJECT, actual)
    write_json(output / 'augmentation_table.json', augmentation)
    dataset = None
    data_error = None
    data_yaml = args.data
    if data_yaml is None and actual and Path(actual['data']).is_file():
        data_yaml = Path(actual['data'])
    if data_yaml is None:
        data_yaml = next((Path(r['archived_copy']) for r in mother['files'] if r['member'].endswith('data_config.yaml') and 'archived_copy' in r), None)
    if args.dataset_manifest:
        dataset = json.loads(args.dataset_manifest.read_text(encoding='utf-8'))
        dataset['reused_audit'] = {'source': str(args.dataset_manifest.resolve()), 'sha256': sha256(args.dataset_manifest),
                                  'status': 'PRIOR_OBSERVED_AUDIT_REUSED; current files not rescanned'}
        # Record hashes cover all fields, including source-family identity, after the mapping join.
        for split in SPLITS:
            part = [r for r in dataset['records'] if r['split'] == split]
            dataset['splits'][split]['manifest_sha256'] = digest(b''.join(canonical(r) for r in part))
            (output / (split + '.txt')).write_bytes(('\n'.join(r['image'] for r in part) + '\n').encode())
        finalize_fingerprints(dataset)
        write_json(output / 'dataset_manifest.json', dataset)
    elif data_yaml:
        config = load_yaml(data_yaml)
        root = args.data_root or Path(config.get('path', data_yaml.parent))
        if not root.is_absolute():
            root = args.project / root
        try:
            dataset = audit_dataset(data_yaml, root, args.project, output, args.family_map, historical, args.missing_labels)
        except FileNotFoundError as e:
            data_error = str(e)
    else:
        data_error = 'Actual training data YAML unavailable; provide --data and --data-root'
    if dataset is None:
        dataset = {'status': 'UNAVAILABLE', 'reason': data_error, 'families': {'status': 'UNKNOWN'}, 'splits': {}}
        write_json(output / 'dataset_manifest.json', dataset)
    conversion = convert(dataset, output / 'coco') if dataset['status'] == 'AUDITED' else {'status': 'NOT_RUN_DATA_UNAVAILABLE_OR_INVALID'}
    code_sha = subprocess.check_output(['git','-c','safe.directory=' + PROJECT.as_posix(), 'rev-parse','HEAD'], cwd=PROJECT, text=True).strip()
    code_status = subprocess.check_output(['git','-c','safe.directory=' + PROJECT.as_posix(), 'status','--porcelain',
                                           '--','benchmarks/comparison','scripts/autodl_prepare_comparison.sh'], cwd=PROJECT, text=True).strip()
    preparation_sources = {p.relative_to(PROJECT).as_posix(): digest(p.read_bytes().replace(b'\r\n',b'\n'))
                           for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.json')}
    preparation_sources['scripts/autodl_prepare_comparison.sh'] = digest((PROJECT/'scripts/autodl_prepare_comparison.sh').read_bytes().replace(b'\r\n',b'\n'))
    family_state = dataset['families']['status']
    comparisons = [v['equal'] for split in dataset['splits'].values() for v in split.get('historical_comparison', {}).values()]
    identity_status = ('MATCHES_ARCHIVED_PATHS_AND_LABELS' if all(comparisons) else 'DIFFERS_FROM_ARCHIVED_DATA') if comparisons else 'HISTORICAL_IDENTITY_UNVERIFIED'
    paper = 'BLOCKED_UNSEEN_ORIGINAL_GENERALIZATION' if family_state == 'CONFIRMED_OVERLAP' else 'UNKNOWN_REQUIRES_REVIEW'
    summary = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'process_exit_code': 0,
               'software_status': 'PREPARATION_COMPLETED', 'data_status': dataset['status'], 'paper_status': paper,
               'historical_data_identity_status': identity_status,
               'family_status': family_state, 'code_sha': code_sha, 'project': str(args.project.resolve()),
               'preparation_git_status': code_status, 'preparation_sources_lf_sha256': preparation_sources,
               'python': sys.executable, 'dataset_identity_sha256': dataset.get('dataset_identity_sha256'),
               'actual_training_data_argument': actual.get('data') if actual else None,
               'resolved_data_yaml': str(data_yaml) if data_yaml else None,
               'splits': dataset['splits'], 'conversion': conversion,
               'missing_assets': {k:v['missing'] for k,v in index.items()},
               'runtime_transform_probe': augmentation['runtime_probe']['status'],
               'server_state': 'NOT_CHECKED_FROM_WINDOWS' if os.name == 'nt' else 'THIS_INVOCATION_ONLY',
               'scope': 'No training, full model inference, dataset repair, re-split, environment installation or weight copy'}
    write_json(output / 'summary.json', summary)
    lines = ['# Comparison preparation', '', 'Process exit code: 0', 'Software: PREPARATION_COMPLETED',
             'Data: ' + dataset['status'], 'Paper: ' + paper, '',
             'Family overlap: ' + family_state, 'Details: dataset_manifest.json, augmentation_table.json, asset_index.json, summary.json.', '',
             'An exit code of zero means the preparation process completed. It does not certify data independence or paper conclusions.',
             'If overlap is confirmed: retain these results as comparisons within the old fixed split. For unseen-original generalization, split original families first, augment only training, and rerun C2/CBR/LIF/mother and all comparison models under that new protocol. Do not pool old and new metrics.']
    (output / 'SUMMARY.md').write_bytes(('\n'.join(lines)+'\n').encode())
    (output / 'COMPLETE.txt').write_text('Preparation complete. process_exit_code=0\npaper_status=' + paper + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    try:
        output = new_output(args.output)
    except OSError as e:
        print('Cannot create fresh output: ' + str(e), file=sys.stderr); return 2
    try:
        prepare(args, output)
        return 0
    except Exception as e:
        write_json(output / 'summary.json', {'process_exit_code': 2, 'software_status': 'FAILED', 'error': repr(e), 'paper_status': 'NOT_CERTIFIED'})
        (output / 'COMPLETE.txt').write_text('Preparation FAILED, process_exit_code=2\n' + repr(e), encoding='utf-8')
        print('Preparation failed: ' + repr(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
