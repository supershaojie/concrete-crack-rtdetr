"""Audit the actual split inputs once; never repair, filter, or write source data."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import io
import itertools
import math
from pathlib import Path

from common import canonical, digest, load_yaml, sha256, write_json

SPLITS = ('train', 'val', 'test')
EXTENSIONS = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm', '.heic'}


def finalize_fingerprints(manifest):
    """Identity includes image bytes as well as labels; legacy path fingerprints stay separate."""
    for split in SPLITS:
        part = [r for r in manifest['records'] if r['split'] == split]
        manifest['splits'][split]['manifest_sha256'] = digest(b''.join(canonical(r) for r in part))
        manifest['splits'][split]['image_inventory_sha256'] = digest('\n'.join(
            r['image'] + ':' + str(r.get('image_sha256')) for r in part).encode())
    manifest['dataset_identity_algorithm'] = 'comparison_dataset_v1_paths_labels_image_bytes'
    manifest['dataset_identity_sha256'] = digest(canonical({s: {k: v for k,v in manifest['splits'][s].items()
        if k in ('split_paths_sha256','label_inventory_sha256','image_inventory_sha256')} for s in SPLITS}))


def parse_labels(raw, location):
    boxes, errors = [], []
    for line_no, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = [float(v) for v in line.split()]
            if len(fields) != 5 or not all(math.isfinite(v) for v in fields):
                raise ValueError('expected five finite numbers')
            c, x, y, w, h = fields
            if c != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('class must be 0; normalized x,y in [0,1], w,h in (0,1]')
            boxes.append([0, x, y, w, h])
        except ValueError as e:
            errors.append({'path': location, 'line': line_no, 'error': str(e), 'text': line})
    return boxes, errors


def label_for(image):
    parts = list(image.parts)
    if 'images' not in parts:
        raise ValueError('Cannot map label without an images path component: ' + str(image))
    index = len(parts) - 1 - parts[::-1].index('images')
    parts[index] = 'labels'
    return Path(*parts).with_suffix('.txt')


def resolve_splits(config, root, project):
    """Directory/list semantics match base.py; ./ in a list means its parent, bare paths use project."""
    original = str(config.get('path', '')).replace('\\', '/').rstrip('/')

    def mapped(value, relative_base):
        value = str(value).replace('\\', '/')
        if original and value.startswith(original + '/'):
            return (root / value[len(original) + 1:]).resolve()
        p = Path(value)
        return (p if p.is_absolute() else relative_base / p).resolve()

    result, evidence = {}, {}
    for split in SPLITS:
        entries = config.get(split)
        if not entries:
            raise FileNotFoundError('Missing split entry: ' + split)
        entries = entries if isinstance(entries, list) else [entries]
        images, resolution = [], []
        for entry in entries:
            p = mapped(entry, root)
            if p.is_dir():
                found = [q.resolve() for q in p.rglob('*') if q.is_file() and q.suffix.lower() in EXTENSIONS]
                resolution.append({'entry': str(entry), 'resolved': str(p), 'kind': 'directory', 'images': len(found)})
            elif p.is_file():
                found = []
                for line in p.read_text(encoding='utf-8-sig').splitlines():
                    if not line.strip():
                        continue
                    line = line.strip()
                    q = mapped(line[2:] if line.startswith('./') else line,
                               p.parent if line.startswith('./') else project)
                    if q.suffix.lower() not in EXTENSIONS:
                        raise ValueError('Unsupported image-list entry (not silently ignored): ' + line)
                    found.append(q)
                resolution.append({'entry': str(entry), 'resolved': str(p), 'kind': 'image_list',
                                   'sha256': sha256(p), 'images': len(found)})
            else:
                raise FileNotFoundError('Split input unavailable: ' + str(p))
            images.extend(found)
        if not images:
            raise ValueError('Empty image split: ' + split)
        result[split] = sorted(images, key=lambda q: q.relative_to(root).as_posix())
        evidence[split] = resolution
    return result, evidence


def family_audit(records, mapping):
    if not mapping or not mapping.is_file():
        return {'status': 'UNKNOWN', 'reason': 'Original-family mapping unavailable; byte uniqueness does not prove source independence'}
    with mapping.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_name = defaultdict(list)
    for row in rows:
        name = row.get('output_name') or row.get('image_name') or Path(row.get('output_image', '')).name
        family = row.get('source_id') or row.get('source_image')
        if name and family:
            by_name[name].append((family, row))
    families, missing, conflicts, split_mismatches = defaultdict(list), [], [], []
    for record in records:
        candidates = by_name[Path(record['image']).name]
        identities = {v[0] for v in candidates}
        if len(identities) != 1:
            (conflicts if identities else missing).append(record['image'])
            continue
        family, row = candidates[0]
        record['source_family'] = family
        families[family].append({'split': record['split'], 'image': record['image'],
                                 'source_image': row.get('source_image')})
        if row.get('split') != record['split']:
            split_mismatches.append({'image': record['image'], 'mapping_split': row.get('split'),
                                     'actual_split': record['split'], 'family': family})
    pairs = {}
    for a, b in itertools.combinations(SPLITS, 2):
        shared = {k: v for k, v in families.items() if {a, b} <= {r['split'] for r in v}}
        pairs[a + '_' + b] = {'families': len(shared),
                              'images': {s: sum(r['split'] == s for v in shared.values() for r in v) for s in (a, b)},
                              'examples': [{'family': k, 'members': v} for k, v in sorted(shared.items())[:5]]}
    return {'status': 'CONFIRMED_OVERLAP' if any(v['families'] for v in pairs.values()) else ('UNKNOWN' if missing or conflicts else 'NO_OVERLAP_IN_MAPPING'),
            'mapping': str(mapping), 'mapping_sha256': sha256(mapping), 'mapping_rows': len(rows),
            'method': 'explicit source_id/source_image mapping joined by unique image basename; actual split from resolved data YAML',
            'mapped_images': sum(map(len, families.values())), 'families': len(families),
            'missing': missing, 'conflicts': conflicts, 'split_mismatch_count': len(split_mismatches),
            'split_mismatch_examples': split_mismatches[:10], 'pairs': pairs}


def audit_dataset(data_yaml, root, project, output, family_map=None, historical=None, missing_labels='unknown'):
    from PIL import Image
    config = load_yaml(data_yaml)
    names = config.get('names')
    if names not in ({0: 'crack'}, {'0': 'crack'}, ['crack']):
        raise ValueError('Expected one foreground class crack, YOLO id 0')
    root, project = Path(root).resolve(), Path(project).resolve()
    split_files, resolution = resolve_splits(config, root, project)
    records, errors, warnings, decoded, hashes = [], [], [], {}, defaultdict(list)
    for split in SPLITS:
        seen = set()
        for ordinal, image in enumerate(split_files[split], 1):
            rel = image.relative_to(root).as_posix()
            if rel in seen:
                errors.append({'path': rel, 'error': 'Duplicate image entry within split'})
            seen.add(rel)
            record = {'split': split, 'image': rel, 'image_id': len(records) + 1}
            try:
                if image not in decoded:
                    # One disk read of each image, shared by hashing and full decode; bounded to one file in RAM.
                    raw = image.read_bytes()
                    with Image.open(io.BytesIO(raw)) as im:
                        orientation = im.getexif().get(274, 1)
                        im.load()
                        decoded[image] = {'image_sha256': digest(raw), 'width': im.width, 'height': im.height,
                                          'image_bytes': len(raw), 'exif_orientation': orientation}
                record.update(decoded[image])
                if record['exif_orientation'] != 1:
                    errors.append({'path': rel, 'error': 'EXIF orientation requires explicit loader alignment'})
                hashes[record['image_sha256']].append({'split': split, 'image': rel})
            except (OSError, ValueError) as e:
                errors.append({'path': rel, 'error': 'Image decode/read failed: ' + str(e)})
            try:
                label = label_for(image)
                record['label'] = label.relative_to(root).as_posix()
                if not label.is_file():
                    record.update(label_status='missing', boxes=[], label_sha256=None, class_counts={}, box_count=0)
                    issue = {'path': record['label'], 'error': 'Missing label', 'policy': missing_labels}
                    (warnings if missing_labels == 'negative' else errors).append(issue)
                else:
                    raw = label.read_bytes()
                    boxes, issues = parse_labels(raw, record['label'])
                    record.update(label_sha256=digest(raw), label_lf_sha256=digest(raw.replace(b'\r\n', b'\n')),
                                  label_status='invalid' if issues else ('valid' if boxes else 'empty_legal_negative'),
                                  boxes=boxes, box_count=len(boxes), class_counts=dict(Counter(str(b[0]) for b in boxes)))
                    errors.extend(issues)
                    if len({tuple(b) for b in boxes}) != len(boxes):
                        errors.append({'path': record['label'], 'error': 'Duplicate boxes; native loader would deduplicate, manual review required'})
                    outside = sum(x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6 for _, x,y,w,h in boxes)
                    if outside:
                        warnings.append({'path': record['label'], 'warning': 'Normalized center/size valid but corners outside image; preserved without clipping', 'boxes': outside})
            except (OSError, ValueError, UnicodeError) as e:
                errors.append({'path': rel, 'error': 'Label read/parse failed: ' + str(e)})
                record.setdefault('boxes', [])
                record['label_status'] = 'invalid'
            records.append(record)
            if ordinal % 500 == 0:
                print('Auditing ' + split + ': ' + str(ordinal) + '/' + str(len(split_files[split])), flush=True)
        print('Audited ' + split + ': ' + str(len(split_files[split])) + ' images', flush=True)
    summary = {}
    for split in SPLITS:
        part = [r for r in records if r['split'] == split]
        paths = [r['image'] for r in part]
        labels = [r.get('label', '') + ':' + str(r.get('label_sha256')) for r in part]
        summary[split] = {'images': len(part), 'boxes': sum(len(r['boxes']) for r in part),
                          'label_status_counts': dict(Counter(r['label_status'] for r in part)),
                          'split_paths_sha256': digest('\n'.join(paths).encode()),
                          'label_inventory_sha256': digest('\n'.join(labels).encode()),
                          'manifest_sha256': digest(b''.join(canonical(r) for r in part))}
        if historical and split in historical:
            summary[split]['historical_comparison'] = {k: {'observed': summary[split][k], 'archived': historical[split][k],
                                                          'equal': summary[split][k] == historical[split][k]}
                                                        for k in ('images', 'boxes', 'split_paths_sha256', 'label_inventory_sha256')}
        (output / (split + '.txt')).write_bytes(('\n'.join(paths) + '\n').encode())
    family = family_audit(records, Path(family_map) if family_map else root / 'split_manifest.csv')
    # Only directory inputs declare a complete image directory; lists can intentionally use subsets.
    expected_labels = {r.get('label') for r in records}
    orphan_labels = set()
    for entries in resolution.values():
        for entry in entries:
            if entry['kind'] != 'directory':
                continue
            label_dir = label_for(Path(entry['resolved']) / '_probe.jpg').parent
            if label_dir.is_dir():
                orphan_labels.update(p.relative_to(root).as_posix() for p in label_dir.rglob('*.txt')
                                     if p.relative_to(root).as_posix() not in expected_labels)
    if orphan_labels:
        warnings.append({'warning': 'Orphan labels in directory inputs (no corresponding selected image)',
                         'count': len(orphan_labels), 'paths': sorted(orphan_labels)})
    for split in SPLITS:
        summary[split]['manifest_sha256'] = digest(b''.join(canonical(r) for r in records if r['split'] == split))
    duplicates = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in duplicates if len({r['split'] for r in v}) > 1]
    result = {'schema_version': 1, 'evidence': 'observed_current', 'status': 'INVALID' if errors else 'AUDITED',
              'data_yaml': str(data_yaml), 'data_yaml_sha256': sha256(data_yaml), 'data_config': config,
              'data_config_canonical_sha256': digest(canonical(config)), 'data_root_mapping': {'configured': config.get('path'), 'observed': str(root)},
              'resolution': resolution, 'splits': summary, 'errors': errors, 'warnings': warnings,
              'missing_label_policy': missing_labels, 'orphan_labels': sorted(orphan_labels),
              'exact_duplicates': {'groups': len(duplicates), 'cross_split_groups': len(cross), 'examples': cross[:10]},
              'families': family, 'records': records}
    finalize_fingerprints(result)
    write_json(output / 'dataset_manifest.json', result)
    return result
