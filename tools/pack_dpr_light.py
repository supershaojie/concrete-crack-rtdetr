"""Create and read-back verify a <20 MiB evidence archive; never invokes training/evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 20 * 1024 * 1024
VARIANTS = ('cbr_lif_dpr_v1', 'dpr_v1')
POLICY = 'corrected_sorted_conf_mask_v1'
SOURCE_SHA256 = 'fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e'
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300, augment=False, rect=False, seed=42)


def assess_evidence(entries, variant, head):
    """Inspect report contents; retained failed attempts never count as completed evidence."""
    missing, selected, rejected = [], {}, {}
    objects = {}
    for name, raw in entries.items():
        if name.startswith('metadata/') and name.endswith('.json'):
            try:
                value = json.loads(raw)
                if isinstance(value, dict):
                    objects[name] = value
            except (ValueError, UnicodeDecodeError):
                rejected[name] = ['invalid JSON']
    run_name = variant + '_rtdetr_r18_lite_e200_b16_onlineaug'

    def choose(label, candidates, check):
        failures = []
        for name, value in sorted(candidates, reverse=True):
            try:
                errors = check(value)
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                errors = ['invalid report schema: ' + str(error)]
            if not errors:
                selected[label] = name
                return value
            rejected[name] = errors
            failures.append(name + ': ' + '; '.join(errors))
        missing.append(label + ': ' + (' | '.join(failures) if failures else 'no eligible report'))
        return None

    def predicate(*conditions):
        return [message for passed, message in conditions if not passed]

    def finite(value, low=0., high=float('inf')):
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high

    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)

    selection = choose('frozen best selection',
        ((name, value) for name, value in objects.items() if Path(name).name == 'best_selection.json'),
        lambda value: predicate((value.get('variant') == variant, 'wrong variant'),
            (value.get('head') == head, 'wrong source HEAD'), (value.get('policy') == POLICY, 'wrong evaluation policy'),
            (value.get('settings') == EVAL, 'wrong evaluation settings'),
            (valid_sha(value.get('checkpoint_sha256')), 'missing best SHA256'),
            (isinstance(value.get('dataset_inventory'), dict) and all(split in value['dataset_inventory'] for split in ('train','val','test')), 'missing dataset identity')))
    frozen = selection or {}
    init = choose('initialization audit',
        ((name, value) for name, value in objects.items() if 'init' in Path(name).name and 'variant' in value),
        lambda value: predicate((value.get('status') == 'PASSED', 'initialization did not pass'),
            (value.get('variant') == variant, 'wrong variant'), (value.get('source_sha256') == SOURCE_SHA256, 'wrong public source'),
            (value.get('reload_exact') is True, 'reload audit absent'),
            (value.get('native_trainer_rebuild', {}).get('status') == 'PASSED', 'native rebuild did not pass'),
            (value.get('train_entry_rebuild', {}).get('status') == 'PASSED', 'train entry rebuild did not pass')))
    choose('preflight report',
        ((name, value) for name, value in objects.items() if Path(name).name == 'preflight.json'),
        lambda value: predicate((value.get('status') == 'PASSED', 'preflight did not pass'),
            (value.get('variant') == variant, 'wrong variant'),
            (value.get('report_kind') == 'full_preflight_engineering', 'wrong preflight kind'),
            (value.get('contract') == 'dpr_acceptance_v1', 'wrong preflight contract'),
            (value.get('code_identity', {}).get('head') == head, 'wrong preflight source HEAD'),
            (bool(init) and value.get('initialization_sha256') == init.get('output_sha256'), 'different initialization'),
            (bool(selection) and value.get('dataset_identity') == frozen.get('dataset_inventory'), 'different frozen dataset')))
    def training_check(value):
        recipe = value.get('recipe', {})
        epochs, done = recipe.get('epochs'), value.get('completed_epochs', 0)
        complete = value.get('status') == 'COMPLETED' or (type(epochs) is int and type(done) is int and done >= epochs)
        return predicate((complete, 'training did not complete'), (value.get('variant') == variant, 'wrong variant'),
                         (value.get('head') == head, 'wrong training source HEAD'),
                         (recipe.get('name') == run_name, 'wrong training run'))
    choose('training completion state',
        ((name, value) for name, value in objects.items() if Path(name).name.startswith('launch_')), training_check)
    try:
        args = yaml.safe_load(entries['training/args.yaml'])
        if not isinstance(args, dict) or args.get('name') != run_name:
            raise ValueError('wrong run identity')
        selected['training args'] = 'training/args.yaml'
    except (KeyError, ValueError, yaml.YAMLError) as error:
        missing.append('training args: missing/invalid or wrong variant (' + str(error) + ')')
    try:
        rows = list(csv.DictReader(io.StringIO(entries['training/results.csv'].decode('utf-8-sig'))))
        if not rows or not any(key.strip() == 'epoch' for key in rows[0]):
            raise ValueError('no epoch result rows')
        selected['training results'] = 'training/results.csv'
    except (KeyError, ValueError, UnicodeDecodeError, csv.Error) as error:
        missing.append('training results: missing/invalid (' + str(error) + ')')

    successful_splits = {}
    for split in ('val', 'test'):
        def evaluation_check(value):
            errors = predicate((value.get('status') == 'completed', 'evaluation did not complete'),
                (value.get('variant') == variant, 'wrong variant'), (value.get('policy') == POLICY, 'wrong policy'),
                (value.get('evidence_scope') == 'full_split' and value.get('export_complete') is True, 'incomplete split/export'),
                (bool(selection) and value.get('checkpoint_sha256') == frozen.get('checkpoint_sha256'), 'different frozen best SHA'),
                (value.get('runtime', {}).get('commit') == head, 'wrong evaluation source HEAD'),
                (bool(selection) and value.get('dataset_inventory') == frozen.get('dataset_inventory'), 'different frozen dataset'),
                (valid_sha(value.get('data_sha256')), 'missing data configuration SHA256'),
                (all(type(value.get('settings', {}).get(key)) is type(expected) and value['settings'][key] == expected
                     for key, expected in EVAL.items()), 'wrong effective evaluation settings'))
            for key in ('precision', 'recall', 'mAP50', 'AP75', 'mAP50_95'):
                if not finite(value.get(key), high=1.):
                    errors.append('missing/nonfinite ' + key)
            ap = value.get('ap_by_class')
            if not (isinstance(ap, list) and len(ap) == 1 and isinstance(ap[0], list) and len(ap[0]) == 10
                    and all(finite(number, high=1.) for number in ap[0])):
                errors.append('missing/invalid ten IoU AP values')
            elif all(finite(value.get(key), high=1.) for key in ('mAP50', 'AP75', 'mAP50_95')):
                if not (math.isclose(ap[0][0], value['mAP50'], abs_tol=1e-12) and
                        math.isclose(ap[0][5], value['AP75'], abs_tol=1e-12) and
                        math.isclose(sum(ap[0])/10, value['mAP50_95'], abs_tol=1e-12)):
                    errors.append('AP summary differs from ten IoU values')
            inventory = frozen.get('dataset_inventory', {}).get(split, {})
            for field, inventory_key in (('images', 'images'), ('ground_truth', 'boxes')):
                if not (type(value.get(field)) is int and value[field] > 0 and value[field] == inventory.get(inventory_key)):
                    errors.append('missing/wrong full-split ' + field)
            speed = value.get('speed_ms_per_image')
            if not isinstance(speed, dict) or not speed or not all(finite(number) for number in speed.values()):
                errors.append('missing/nonfinite speed metrics')
            if split == 'test' and 'val' in successful_splits and value.get('data_sha256') != successful_splits['val'].get('data_sha256'):
                errors.append('val/test data config differs')
            return errors
        evidence = choose(split + ' metrics',
            ((name, value) for name, value in objects.items() if Path(name).name == 'metrics.json' and value.get('split') == split),
            evaluation_check)
        if evidence:
            successful_splits[split] = evidence
            prefix = selected[split + ' metrics'].rsplit('/', 1)[0] + '/'
            for label, suffix in (('PR curve', 'PR_curve.png'), ('confusion matrix', '/confusion_matrix.png')):
                matching = [name for name, raw in entries.items() if name.startswith(prefix) and name.endswith(suffix)
                            and raw.startswith(b'\x89PNG\r\n\x1a\n')]
                if matching:
                    selected[split + ' ' + label] = matching[0]
                else:
                    missing.append(split + ' ' + label + ': missing PNG in selected successful evaluation')
        else:
            missing.extend((split + ' PR curve: no successful matching evaluation',
                            split + ' confusion matrix: no successful matching evaluation'))
    return dict(missing=missing, selected=selected, rejected_reports=rejected)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def package(variant, output=None, run_dir=None, metadata=None):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(output or ROOT / 'outputs/dpr' / variant / ('dpr_light_' + stamp + '.tar.gz')).resolve()
    if output.exists() or Path(str(output) + '.sha256').exists():
        raise FileExistsError('Existing package preserved: ' + str(output))
    metadata = Path(metadata or ROOT / 'outputs/dpr' / variant).resolve()
    run = Path(run_dir) if run_dir else Path('/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series') / (variant + '_rtdetr_r18_lite_e200_b16_onlineaug')
    entries, omitted, missing = {}, [], []

    def add(path, name):
        if path.is_symlink():
            omitted.append(dict(path=str(path), reason='symlink'))
            return
        size = path.stat().st_size
        if path.name.endswith('.log'):
            with path.open('rb') as stream:
                stream.seek(max(0, size - 65536))
                entries[name + '.tail.txt'] = stream.read()
            omitted.append(dict(path=str(path), reason='log tail only', bytes=size, sha256=digest(path)))
        elif size > 4 * 1024 * 1024:
            # Preserve complete source/evidence separately; never truncate a JSON report.
            omitted.append(dict(path=str(path), reason='large artifact remains on host', bytes=size, sha256=digest(path)))
            missing.append(name + ': oversized; see omitted')
        else:
            entries[name] = path.read_bytes()

    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in tracked:
        if name and (name.startswith('ultralytics-main/ultralytics/') or
                     name.startswith('docs/dpr/') or
                     (name.startswith('tools/') and ('dpr' in name or Path(name).name in
                      {'init_lif_down.py', 'init_c19_lif_v1.py', 'lif_down_topology.py', 'c19_lif_v1_data.py'}))):
            add(ROOT / name, 'source/' + name)
    for folder, prefix in ((metadata, 'metadata'), (run, 'training')):
        if not folder.is_dir():
            missing.append(prefix + ': directory missing')
            continue
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path == output or path.name.startswith('dpr_light_'):
                continue
            name = prefix + '/' + path.relative_to(folder).as_posix()
            if path.suffix.lower() in {'.pt', '.pth', '.zip', '.npz', '.npy'} or path.name == 'predictions_gt.jsonl.gz':
                omitted.append(dict(path=str(path.resolve()), reason='weights/large predictions/intermediate excluded',
                                    bytes=path.stat().st_size, sha256=digest(path)))
            elif path.suffix.lower() in {'.json', '.yaml', '.csv', '.txt', '.log', '.exit_code'} or path.name.endswith('.json.gz') or (
                path.suffix.lower() == '.png' and ('curve' in path.name or 'confusion_matrix' in path.name or path.name == 'results.png')):
                add(path, name)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    evidence = assess_evidence(entries, variant, head)
    missing.extend(evidence['missing'])
    info = dict(schema='dpr_light_v1', variant=variant, created=datetime.now(timezone.utc).isoformat(),
                head=head, tracked_worktree_status=subprocess.check_output(
                    ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines(),
                evidence_complete=not missing, missing=missing, omitted=omitted,
                selected_evidence=evidence['selected'], rejected_reports=evidence['rejected_reports'],
                evaluation_policy='corrected_sorted_conf_mask_v1', limit_bytes_exclusive=LIMIT,
                note='Archive integrity is independent of experiment acceptance; missing evidence is not PASSED.')
    entries['PACKAGE_INFO.json'] = (json.dumps(info, ensure_ascii=False, indent=2) + '\n').encode()
    manifest = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in entries.items()}
    entries['MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, data in sorted(entries.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    content = buffer.getvalue()
    if len(content) >= LIMIT:
        raise RuntimeError('Package exceeds <20 MiB contract; no archive written')
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        assert len(archive.getnames()) == len(entries) == len(set(archive.getnames()))
        for name, row in manifest.items():
            value = archive.extractfile(name).read()
            assert len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(content)
    result = dict(path=str(output), bytes=len(content), sha256=digest(output), archive_integrity='PASSED',
                  evidence_complete=not missing, missing=missing, omitted_count=len(omitted))
    with Path(str(output) + '.sha256').open('x', encoding='utf-8') as stream:
        stream.write(result['sha256'] + '  ' + output.name + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default=VARIANTS[0])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    package(args.variant, args.output, args.run_dir, args.metadata)
