"""Light list/label checks; never decode or hash image contents during preparation."""
from __future__ import annotations
import argparse
from pathlib import Path
from support import PROJECT, canonical, digest, load_yaml, read_json, sha256, write_json
from dataset import SPLITS, label_for, parse_labels, resolve_splits

EXPECTED = {'train': (6048, 45573), 'val': (1728, 12840), 'test': (864, 6663)}

def inspect(data_yaml, root, project, enforce_counts=True):
    root, project = Path(root).resolve(), Path(project).resolve()
    cfg = load_yaml(data_yaml)
    if cfg.get('names') not in ({0:'crack'}, {'0':'crack'}, ['crack']) or cfg.get('nc', 1) != 1:
        raise ValueError('Expected exactly crack, YOLO class 0')
    files, resolution = resolve_splits(cfg, root, project)
    rows, parts, seen = [], {}, set()
    for split in SPLITS:
        paths, label_inventory, boxes_count, empty = [], [], 0, 0
        for image in files[split]:
            relative = image.relative_to(root).as_posix()
            if relative in seen:
                raise ValueError('Duplicate image across/in split: ' + relative)
            seen.add(relative)
            label = label_for(image)
            raw = label.read_bytes()  # Missing labels are investigated, never silently interpreted as negative.
            boxes, errors = parse_labels(raw, str(label))
            if errors:
                raise ValueError(str(errors[:5]))
            if len({tuple(b) for b in boxes}) != len(boxes):
                raise ValueError('Duplicate labels would be silently removed by upstream: ' + str(label))
            if any(min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6 for _,x,y,w,h in boxes):
                raise ValueError('Out-of-bounds labels need investigation: ' + str(label))
            stat = image.stat()
            if not stat.st_size:
                raise ValueError('Zero-byte image: ' + str(image))
            rows.append({'image_id': len(rows)+1, 'split': split, 'image': relative,
                         'image_bytes': stat.st_size, 'label': label.relative_to(root).as_posix(),
                         'label_sha256': digest(raw), 'boxes': boxes})
            paths.append(relative)
            label_inventory.append(label.relative_to(root).as_posix()+':'+digest(raw))
            boxes_count += len(boxes)
            empty += not boxes
        part = {'images':len(paths), 'labels':len(paths), 'boxes':boxes_count, 'empty_labels':empty,
                'split_paths_sha256':digest('\n'.join(paths).encode()),
                'label_inventory_sha256':digest('\n'.join(sorted(label_inventory)).encode())}
        # Limit orphan-label enumeration to directories belonging to this split.
        expected_labels = {label_for(x).resolve() for x in files[split]}
        actual_labels = {p.resolve() for d in {p.parent for p in expected_labels} for p in d.glob('*.txt')}
        if actual_labels != expected_labels:
            raise ValueError('Orphan/unmatched labels: ' + str(sorted(map(str, actual_labels-expected_labels))[:5]))
        if enforce_counts and (len(paths), boxes_count) != EXPECTED[split]:
            raise ValueError(f'{split}: actual {(len(paths),boxes_count)} differs from {EXPECTED[split]}; source data unchanged')
        parts[split] = part
    identity = digest(canonical([{k:r[k] for k in ('split','image','image_bytes','label_sha256')} for r in rows]))
    if enforce_counts:
        previous = read_json(PROJECT/'docs/comparison/evidence/yolov5m_validation.json')['light_data']
        differences = {s:{k:{'expected':previous['splits'][s][k], 'actual':parts[s][k]}
                          for k in ('split_paths_sha256','label_inventory_sha256')
                          if previous['splits'][s][k] != parts[s][k]} for s in SPLITS}
        if any(differences.values()) or identity != previous['identity_sha256']:
            raise ValueError('Frozen data differs: '+str({'splits':differences,
                'expected_identity':previous['identity_sha256'], 'actual_identity':identity}))
    return {'status':'LIGHT_CHECKED', 'root':str(root), 'data_yaml':str(Path(data_yaml).resolve()),
            'data_yaml_sha256':sha256(data_yaml), 'resolution':resolution, 'splits':parts, 'records':rows,
            'dataset_identity_algorithm':'light_v1_sorted_paths_image_sizes_label_bytes_NOT_image_content_hash',
            'dataset_identity_sha256':identity, 'image_decode':'NOT_RUN', 'image_content_hash':'NOT_RUN'}

def save_check(manifest, output):
    import yaml
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', manifest)
    cfg = {'path':manifest['root'], 'nc':1, 'names':{0:'crack'}}
    for split in SPLITS:
        target = output/(split+'.txt')
        target.write_text(''.join((Path(manifest['root'])/r['image']).as_posix()+'\n'
                                 for r in manifest['records'] if r['split']==split), encoding='utf-8')
        cfg[split] = target.as_posix()
    (output/'data.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

def make_gt(manifest, split, cached=None):
    """Reuse independently validated COCO if supplied; otherwise read only image headers."""
    from PIL import Image
    rows = [r for r in manifest['records'] if r['split']==split]
    old = read_json(cached) if cached else None
    dims = {}
    if old:
        if old['info']['split'] != split or len(old['images']) != len(rows):
            raise ValueError('Cached GT split/coverage mismatch')
        dims = {im['file_name']:im for im in old['images']}
        if set(dims) != {r['image'] for r in rows}:
            raise ValueError('Cached GT paths require explicit mapping to the actual root')
        evidence_path=Path(cached).parent.parent/'dataset_manifest.json'
        if not evidence_path.is_file():
            raise ValueError('Cached GT requires its small source manifest for size/label validation')
        evidence=read_json(evidence_path)
        inventory={r['image']:r for r in evidence['records'] if r['split']==split}
        for r in rows:
            prior=inventory.get(r['image'],{})
            if any(prior.get(k)!=r[k] for k in ('image_bytes','label_sha256','image_id')):
                raise ValueError('Cached GT manifest does not match actual file size/labels/ID')
    images, annotations = [], []
    for r in rows:
        if old:
            im = dims[r['image']]
            if im['id'] != r['image_id']:
                raise ValueError('Cached stable image_id differs')
            w, h = im['width'], im['height']
        else:
            with Image.open(Path(manifest['root'])/r['image']) as im:
                if im.getexif().get(274,1) != 1:
                    raise ValueError('EXIF orientation requires explicit alignment')
                w, h = im.size  # header only; no load()/verify()/image SHA
        images.append({'id':r['image_id'], 'file_name':r['image'], 'width':w, 'height':h, 'split':split})
        for _,x,y,bw,bh in r['boxes']:
            box = [(x-bw/2)*w, (y-bh/2)*h, bw*w, bh*h]
            annotations.append({'id':len(annotations)+1, 'image_id':r['image_id'], 'category_id':1,
                                'bbox':box, 'area':box[2]*box[3], 'iscrowd':0})
    if old:
        normalize = lambda a: sorted((x['image_id'],x['category_id'],*x['bbox'],x.get('iscrowd',0)) for x in a)
        if normalize(old['annotations']) != normalize(annotations):
            raise ValueError('Cached GT annotations differ from current labels')
    return {'info':{'split':split, 'dataset_identity_sha256':manifest['dataset_identity_sha256'],
                    'identity_algorithm':manifest['dataset_identity_algorithm'],
                    'dimensions_source':str(cached) if old else 'PIL header; EXIF=1; no pixel decode',
                    'cached_gt_sha256':sha256(cached) if old else None},
            'images':images, 'annotations':annotations, 'categories':[{'id':1,'name':'crack'}]}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = inspect(a.data, a.data_root, a.source_project)
    save_check(m, a.output)
    print(m['splits'])
