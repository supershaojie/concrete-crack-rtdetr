"""Convert an audited manifest to COCO without rescanning or altering the source dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import new_output, write_json
from dataset import SPLITS


def convert(manifest, output):
    if manifest.get('status') != 'AUDITED' or manifest.get('errors'):
        raise ValueError('Conversion blocked by dataset errors; inspect dataset_manifest.json (nothing repaired or discarded)')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    annotation_id = 1
    for split in SPLITS:
        images, annotations = [], []
        for r in manifest['records']:
            if r['split'] != split:
                continue
            images.append({'id': r['image_id'], 'file_name': r['image'], 'width': r['width'], 'height': r['height'], 'split': split})
            for cls, x, y, w, h in r['boxes']:
                assert cls == 0
                px, py, pw, ph = (x-w/2)*r['width'], (y-h/2)*r['height'], w*r['width'], h*r['height']
                annotations.append({'id': annotation_id, 'image_id': r['image_id'], 'category_id': 1,
                                    'bbox': [px, py, pw, ph], 'area': pw*ph, 'iscrowd': 0})
                annotation_id += 1
        counts[split] = {'input_images': manifest['splits'][split]['images'], 'output_images': len(images),
                         'input_boxes': manifest['splits'][split]['boxes'], 'output_boxes': len(annotations)}
        assert counts[split]['input_images'] == len(images) and counts[split]['input_boxes'] == len(annotations)
        write_json(output / (split + '.json'), {'info': {'split': split, 'dataset_identity_sha256': manifest['dataset_identity_sha256'],
                                                       'coordinate_rule': 'YOLO normalized cxcywh to original-pixel xywh; float64, no clipping/rounding'},
                                               'images': images, 'annotations': annotations,
                                               'categories': [{'id': 1, 'name': 'crack'}]})
    write_json(output / 'conversion_counts.json', counts)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(convert(json.loads(args.manifest.read_text(encoding='utf-8')), args.output), indent=2))


if __name__ == '__main__':
    main()
