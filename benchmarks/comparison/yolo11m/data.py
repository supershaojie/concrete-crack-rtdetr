"""Light checks: sorted actual splits, labels, stats and only necessary image headers."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from support import canonical, digest, load_yaml, read_json, sha256, write_json
from dataset import SPLITS, label_for, parse_labels, resolve_splits

EXPECTED = {'train': (6048, 45573), 'val': (1728, 12840), 'test': (864, 6663)}


def image_size(path):
    """No decode, repair or image-content hash. Reject EXIF orientation ambiguity."""
    from PIL import Image
    with Image.open(path) as im:
        width, height = im.size
        if min(width, height) < 10 or im.getexif().get(274, 1) != 1:
            raise ValueError('Small image or unsupported EXIF orientation: ' + str(path))
        if im.format in {'JPEG', 'MPO'}:
            with Path(path).open('rb') as stream:
                stream.seek(-2, 2)
                if stream.read() != b'\xff\xd9':
                    raise ValueError('JPEG end marker missing; shared image will NOT be repaired: ' + str(path))
    return width, height


def read_labels(path):
    raw = Path(path).read_bytes()
    boxes, errors = parse_labels(raw, str(path))
    if errors:
        raise ValueError(str(errors))
    if len({tuple(b) for b in boxes}) != len(boxes):
        raise ValueError('Duplicate labels would be removed by native loader: ' + str(path))
    return boxes, digest(raw)


def annotation_rows(records):
    result = []
    for r in records:
        for cls, x, y, w, h in r['boxes']:
            assert cls == 0
            box = [(x-w/2)*r['width'], (y-h/2)*r['height'], w*r['width'], h*r['height']]
            result.append({'id': len(result)+1, 'image_id': r['image_id'], 'category_id': 1,
                           'bbox': box, 'area': box[2]*box[3], 'iscrowd': 0})
    return result


def reuse_gt(candidate, records, split):
    """Only reuse a complete, label-equivalent, dimension/path-correct public COCO file."""
    gt = read_json(candidate)
    if gt['info']['split'] != split or gt['categories'] != [{'id': 1, 'name': 'crack'}]:
        raise ValueError('Public GT split/category mismatch: ' + str(candidate))
    images = {im['file_name']: im for im in gt['images']}
    ids = [im['id'] for im in gt['images']]
    if len(images) != len(gt['images']) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate image in public GT')
    if set(images) != {r['image'] for r in records}:
        raise ValueError('Public GT image set differs from actual YAML')
    by_id = defaultdict(list)
    for ann in gt['annotations']:
        if ann['image_id'] not in ids or ann['category_id'] != 1 or ann.get('iscrowd', 0):
            raise ValueError('Invalid public GT annotation')
        by_id[ann['image_id']].append(ann['bbox'])
    for r in records:
        im = images[r['image']]
        if (im['width'], im['height']) != (r['width'], r['height']):
            raise ValueError('Public GT image dimensions differ: ' + r['image'])
        expected = sorted(a['bbox'] for a in annotation_rows([r]))
        actual = sorted(by_id[im['id']])
        if len(actual) != len(expected) or any(len(a) != 4 or any(
                not math.isfinite(x) or abs(x-y) > 1e-7 for x, y in zip(a, b))
                for a, b in zip(actual, expected)):
            raise ValueError('Public GT labels differ: ' + r['image'])
        r['image_id'] = im['id']  # preserve the public IDs, never infer them from basename
    return gt


def preflight(data_yaml, run, data_root=None, public_coco=None, enforce_counts=True):
    import yaml
    data_yaml, run = Path(data_yaml).resolve(), Path(run).resolve()
    cfg = load_yaml(data_yaml)
    if cfg.get('names') not in (['crack'], {0: 'crack'}, {'0': 'crack'}) or cfg.get('nc', 1) != 1:
        raise ValueError('Expected the single YOLO class 0 = crack')
    root = Path(data_root) if data_root else Path(cfg.get('path', data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()
    files, resolution = resolve_splits(cfg, root, data_yaml.parent.parent)
    records, summaries, errors, seen = [], {}, [], set()
    for split in SPLITS:
        for path in files[split]:
            rel = path.relative_to(root).as_posix()
            if rel in seen:
                errors.append({'image': rel, 'error': 'Duplicate image path within/across actual splits'})
            seen.add(rel)
            r = {'split': split, 'image': rel, 'image_id': len(records)+1}
            try:
                stat = path.stat()
                label = label_for(path)
                boxes, label_hash = read_labels(label)
                r.update(image_bytes=stat.st_size, image_mtime_ns=stat.st_mtime_ns,
                         label=label.relative_to(root).as_posix(), label_sha256=label_hash, boxes=boxes)
                # Only val/test headers now; train shapes are built on first dataset use.
                if split != 'train':
                    r['width'], r['height'] = image_size(path)
            except (OSError, ValueError, UnicodeError) as exc:
                errors.append({'image': rel, 'error': str(exc)})
            records.append(r)
        part = [r for r in records if r['split'] == split]
        summaries[split] = {'images': len(part), 'labels': sum('label_sha256' in r for r in part),
                            'boxes': sum(len(r.get('boxes', [])) for r in part),
                            'empty_labels': sum('boxes' in r and not r['boxes'] for r in part),
                            'split_paths_sha256': digest('\n'.join(r['image'] for r in part).encode()),
                            'label_inventory_sha256': digest('\n'.join(
                                r.get('label', '') + ':' + str(r.get('label_sha256')) for r in part).encode())}
        expected_labels = {r.get('label') for r in part}
        orphans = set()
        for item in resolution[split]:
            if item['kind'] == 'directory':
                label_dir = label_for(Path(item['resolved']) / '_probe.jpg').parent
                if label_dir.is_dir():
                    orphans.update(p.relative_to(root).as_posix() for p in label_dir.rglob('*.txt')
                                   if p.relative_to(root).as_posix() not in expected_labels)
        summaries[split]['orphan_labels'] = sorted(orphans)
        if orphans:
            errors.append({'split': split, 'error': 'Orphan labels in actual directory split', 'labels': sorted(orphans)})
        if enforce_counts and (summaries[split]['images'], summaries[split]['boxes']) != EXPECTED[split]:
            errors.append({'split': split, 'error': 'Counts differ; inspect actual YAML/paths/labels, no data changed',
                           'expected': EXPECTED[split], 'observed': summaries[split]})
        print(split + ': ' + str(summaries[split]), flush=True)
    manifest = {'status': 'INVALID' if errors else 'LIGHT_CHECKED', 'schema_version': 1,
                'data_yaml': str(data_yaml), 'data_yaml_sha256': sha256(data_yaml), 'data_root': str(root),
                'resolution': resolution, 'splits': summaries, 'errors': errors, 'records': records,
                'identity_algorithm': 'light_v1_relative_paths_label_bytes_image_sizes_NO_image_content_hash',
                'limits': 'No full image hash/decode/family audit; existing public historical evidence remains separate.'}
    manifest['dataset_identity_sha256'] = digest(canonical([
        {k: r.get(k) for k in ('split', 'image', 'image_bytes', 'label', 'label_sha256')} for r in records]))
    if enforce_counts:
        from support import ROOT
        previous = read_json(ROOT/'docs/comparison/evidence/yolov8m_delivery_validation.json')['data']
        for split in SPLITS:
            for key in ('split_paths_sha256','label_inventory_sha256'):
                if summaries[split][key] != previous['splits'][split][key]:
                    errors.append({'split':split, 'field':key, 'expected':previous['splits'][split][key],
                                   'actual':summaries[split][key], 'error':'Frozen manifest differs'})
        if manifest['dataset_identity_sha256'] != previous['dataset_identity_sha256']:
            errors.append({'error':'Frozen light data identity differs',
                'expected':previous['dataset_identity_sha256'], 'actual':manifest['dataset_identity_sha256']})
        manifest['status'] = 'INVALID' if errors else 'LIGHT_CHECKED'
    if errors:
        write_json(run / 'manifest.json', manifest)
        raise ValueError('Light preflight failed; see manifest.json errors')
    for split in SPLITS:
        part = [r for r in records if r['split'] == split]
        (run / (split + '.txt')).write_text('\n'.join(str(root / r['image']) for r in part)+'\n', encoding='utf-8')
        if split == 'train':
            continue  # Training consumes YOLO labels directly, no train COCO conversion.
        candidate = Path(public_coco) / (split + '.json') if public_coco else None
        if candidate and candidate.is_file():
            gt = reuse_gt(candidate, part, split)
            origin = {'reused_from': str(candidate.resolve()), 'sha256': sha256(candidate),
                      'previous_dataset_identity': gt['info'].get('dataset_identity_sha256')}
        else:
            gt = {'images': [{'id': r['image_id'], 'file_name': r['image'], 'width': r['width'],
                             'height': r['height'], 'split': split} for r in part],
                  'annotations': annotation_rows(part), 'categories': [{'id': 1, 'name': 'crack'}]}
            origin = {'generated_from': 'actual YOLO labels + header dimensions only'}
        gt['info'] = {'split': split, 'dataset_identity_sha256': manifest['dataset_identity_sha256'],
                      'identity_algorithm': manifest['identity_algorithm'], 'origin': origin,
                      'coordinate_rule': 'float64 original pixel xywh; no clipping or rounding'}
        write_json(run / 'gt' / (split+'.json'), gt)
    write_json(run / 'manifest.json', manifest)
    derived = {'path': str(root), 'nc': 1, 'names': {0: 'crack'},
               **{s: str(run / (s+'.txt')) for s in SPLITS}}
    (run / 'data.yaml').write_text(yaml.safe_dump(derived, allow_unicode=True, sort_keys=False), encoding='utf-8')
    write_json(run / 'input_checksums.json', {name: sha256(run / name) for name in
        ('manifest.json', 'data.yaml', 'train.txt', 'val.txt', 'test.txt', 'gt/val.json', 'gt/test.json')})
    return manifest


def verify_inputs(run, splits=SPLITS):
    run = Path(run)
    for name, expected in read_json(run / 'input_checksums.json').items():
        if sha256(run / name) != expected:
            raise ValueError('Run input artifact changed: ' + name)
    manifest = read_json(run / 'manifest.json')
    if manifest['status'] != 'LIGHT_CHECKED':
        raise ValueError('Expected successful light preflight')
    root = Path(manifest['data_root'])
    for r in manifest['records']:
        if r['split'] not in splits:
            continue
        stat = (root / r['image']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (r['image_bytes'], r['image_mtime_ns']):
            raise ValueError('Image stat changed since preflight: ' + r['image'])
        if sha256(root / r['label']) != r['label_sha256']:
            raise ValueError('Label changed since preflight: ' + r['label'])
    return manifest
