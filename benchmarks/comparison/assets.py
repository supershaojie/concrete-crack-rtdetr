"""Index selected run evidence, including tar members, without unpickling/copying weights."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

from common import digest, write_json

SMALL_NAMES = {'args.yaml', 'actual_train_args.yaml', 'train_args.yaml', 'authoritative_c2_args.yaml',
               'data_config.yaml', 'metrics.json', 'metrics_summary.json', 'results.csv', 'source_record.json',
               'dataset_inventory.json', 'pip_freeze.txt', 'package.json', 'initialization.json',
               'git.json', 'selection_record.json', 'environment.json', 'checkpoint_model_yaml.json'}


def selected(name):
    if name.endswith(('source/git.json', 'source/checkpoint_model_yaml.json')):
        return True
    if '/source/' in '/' + name or name.startswith('source/'):
        return False
    return Path(name).name in SMALL_NAMES | {'best.pt', 'last.pt', 'predictions_gt.jsonl.gz', 'predictions.json'}


def scan_source(source, output, model_id, extract_caches=False):
    source = Path(source)
    if not source.exists():
        return [], {'path': str(source), 'status': 'MISSING'}, {}
    indexed, contents = [], {}

    def consume(name, stream, size):
        h, chunks = hashlib.sha256(), []
        small = Path(name).name in SMALL_NAMES and size <= 4 * 1024 * 1024
        cache = None
        if extract_caches and name.endswith('predictions_gt.jsonl.gz'):
            cache = output / 'prediction_caches' / model_id / (digest(name.encode())[:12] + '.jsonl.gz')
            cache.parent.mkdir(parents=True, exist_ok=True)
        writer = cache.open('xb') if cache else None
        try:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
                if small:
                    chunks.append(block)
                if writer:
                    writer.write(block)
        finally:
            if writer:
                writer.close()
        ref = str(source / name) if source.is_dir() else str(source) + '::' + name
        row = {'source': ref, 'member': name, 'bytes': size, 'sha256': h.hexdigest(), 'evidence': 'observed_file_bytes'}
        if cache:
            row['local_cache'] = str(cache)
        if small:
            raw = b''.join(chunks)
            contents[ref] = raw
            dest = output / 'raw_evidence' / model_id / (digest(ref.encode())[:10] + '_' + Path(name).name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            row['archived_copy'] = str(dest)
        if name.endswith('predictions_gt.jsonl.gz'):
            row['precision'] = 'full_float_export_declared_by_source; validate metrics hash before reuse'
        elif name.endswith('predictions.json'):
            row['precision'] = 'rounded_native_export; cannot certify exact historical AP'
        indexed.append(row)

    if source.is_dir():
        for p in sorted(source.rglob('*')):
            if p.is_file() and selected(p.relative_to(source).as_posix()):
                with p.open('rb') as stream:
                    consume(p.relative_to(source).as_posix(), stream, p.stat().st_size)
    else:
        # A single sequential compressed-archive pass. No extractall or weight deserialization.
        with tarfile.open(source, 'r|*') as archive:
            for member in archive:
                if member.isfile() and selected(member.name):
                    with archive.extractfile(member) as stream:
                        consume(member.name, stream, member.size)
    return indexed, {'path': str(source), 'status': 'OBSERVED'}, contents


def build_assets(locations, output, registry, extract_caches=False):
    import yaml
    index, all_contents = {}, {}
    for model_id, spec in registry.items():
        files, sources, contents = [], [], {}
        for location in locations.get(model_id, []):
            found, state, raw = scan_source(location, output, model_id, extract_caches)
            files.extend(found); sources.append(state); contents.update(raw)
        args_files = [r for r in files if Path(r['member']).name in ('args.yaml', 'actual_train_args.yaml')
                      and not any(s in r['member'] for s in ('test_run/', 'references/', 'authoritative'))]
        args_files.sort(key=lambda r: (0 if '/training/args.yaml' in '/' + r['member'] or '/train_run/args.yaml' in '/' + r['member'] else
                                       1 if r['member'] == 'args.yaml' else 2, r['source']))
        actual_args = yaml.safe_load(contents[args_files[0]['source']]) if args_files else None
        metrics = []
        for row in files:
            if Path(row['member']).name not in ('metrics.json', 'metrics_summary.json'):
                continue
            d = json.loads(contents[row['source']])
            metrics.append({'source': row['source'], 'sha256': row['sha256'], 'evidence': 'observed_archived_result_not_new_inference',
                            'values': d})
        csv_best = []
        for row in files:
            if Path(row['member']).name != 'results.csv' or 'references/' in row['member']:
                continue
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(io.StringIO(contents[row['source']].decode('utf-8-sig')))]
            usable = [r for r in rows if 'metrics/mAP50-95(B)' in r]
            if usable:
                best = max(usable, key=lambda r: float(r['metrics/mAP50-95(B)']))
                csv_best.append({'source': row['source'], 'epochs_logged': len(rows), 'csv_max_map_epoch': int(best['epoch']),
                                 'csv_max_map': float(best['metrics/mAP50-95(B)']),
                                 'status': 'CSV_ROUNDED_CANDIDATE; checkpoint epoch not deserialized',
                                 'rule': 'training val mAP50-95; save on equality to best_fitness; CSV rounding may hide ties'})
        bests = [r for r in files if r['member'].endswith('weights/best.pt')]
        lasts = [r for r in files if r['member'].endswith('weights/last.pt')]
        conflicts = []
        for candidate in args_files[1:]:
            other = yaml.safe_load(contents[candidate['source']])
            differences = {k: {'selected': actual_args.get(k), 'other': other.get(k)} for k in set(actual_args) | set(other)
                           if actual_args.get(k) != other.get(k) or type(actual_args.get(k)) is not type(other.get(k))}
            if differences:
                conflicts.append({'source': candidate['source'], 'fields': differences})
        for metric in metrics:
            d = metric['values']; expected = d.get('checkpoint_sha256', d.get('best_sha256'))
            metric['best_hash_matches_observed_weight'] = expected in {r['sha256'] for r in bests} if expected and bests else None
            pred_hash = d.get('predictions_gt_sha256')
            metric['prediction_hash_matches_observed_cache'] = pred_hash in {r['sha256'] for r in files if r['member'].endswith('predictions_gt.jsonl.gz')} if pred_hash else None
        missing = []
        for name, present in [('training_args', args_files), ('best_weight', bests), ('last_weight', lasts), ('training_results_csv', csv_best), ('independent_metrics', metrics)]:
            if not present:
                missing.append(name)
        for split in ('val', 'test'):
            if not any(m['values'].get('split') == split for m in metrics):
                missing.append('independent_' + split + '_metrics')
        observed_commits = {m['values']['runtime']['commit'] for m in metrics
                            if isinstance(m['values'].get('runtime'), dict) and m['values']['runtime'].get('commit')}
        observed_commits.update(m['values']['source_commit'] for m in metrics if m['values'].get('source_commit'))
        code_records, selections = [], []
        for row in files:
            if row['member'].endswith(('git.json', 'selection_record.json', 'checkpoint_model_yaml.json')):
                record = {'source': row['source'], 'values': json.loads(contents[row['source']])}
                if row['member'].endswith('git.json'):
                    code_records.append(record)
                    if record['values'].get('commit'):
                        observed_commits.add(record['values']['commit'])
                else:
                    selections.append(record)
        observed_commits = sorted(observed_commits)
        initializations = []
        for row in files:
            if row['member'].endswith('initialization.json'):
                init = json.loads(contents[row['source']])
                initializations.append({'evidence': 'observed_archived_record', 'file': row['source'],
                                        'values': {k:v for k,v in init.items() if k in ('source','source_sha256','output','output_sha256','status','variant','seed')}})
        index[model_id] = {'model': spec, 'sources': sources, 'files': files, 'args_source': args_files[0] if args_files else None,
                           'actual_training_args': actual_args, 'args_conflicts': conflicts, 'metrics': metrics,
                           'observed_runtime_commits': observed_commits,
                           'code_records': code_records, 'checkpoint_selection_records': selections,
                           'code_identity_status': 'observed_archived_runtime' if observed_commits else 'historical_reference_only',
                           'initialization_sources': initializations,
                           'actual_training_model_argument': actual_args.get('model') if actual_args else None,
                           'actual_training_data_argument': actual_args.get('data') if actual_args else None,
                           'best_weights': bests, 'last_weights': lasts, 'training_selection': csv_best, 'missing': missing,
                           'weight_backup_note': 'Index and hashes only; no weight bytes copied into preparation outputs or Git'}
        all_contents[model_id] = contents
        print('Indexed assets: ' + model_id + ' (' + str(len(files)) + ' files)', flush=True)
    write_json(output / 'asset_index.json', index)
    return index, all_contents
