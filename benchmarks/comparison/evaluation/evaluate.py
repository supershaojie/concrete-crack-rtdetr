"""CPU evaluation of exported predictions + independent COCO GT; never loads a detector."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import canonical, digest, sha256, write_json
from native_metrics import Matcher, ap_per_class, box_iou
import numpy as np
import torch

POLICY = {'name': 'corrected_sorted_conf_mask_v1', 'conf': 0.001, 'max_det': 300,
          'coordinates': 'original_image_pixels_xyxy', 'nms': False, 'tie_order': 'preserve_export_order',
          'native_source_sha': 'a0459d6a652cb702699087c88fa39a3e4c4087ec'}
POLICY_SHA = digest(canonical(POLICY))


def open_text(path):
    return gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')


def validate_identity(identity, gt):
    for key in ('model', 'model_code_sha', 'checkpoint_sha256', 'dataset_identity_sha256', 'evaluation_config_sha256', 'postprocessing'):
        if not identity.get(key):
            raise ValueError('Missing prediction identity: ' + key)
    for key, length in [('model_code_sha', 40), ('checkpoint_sha256', 64)]:
        if not re.fullmatch('[0-9a-f]{' + str(length) + '}', identity[key]):
            raise ValueError('Invalid identity digest: ' + key)
    if identity['dataset_identity_sha256'] != gt['info']['dataset_identity_sha256']:
        raise ValueError('Prediction/GT dataset identity differs')
    if identity['evaluation_config_sha256'] != POLICY_SHA:
        raise ValueError('Prediction evaluation policy differs')


def valid_boxes(boxes):
    return all(len(b) == 4 and all(math.isfinite(v) for v in b) and b[2] > b[0] and b[3] > b[1] for b in boxes)


def evaluate(gt, records, identity):
    validate_identity(identity, gt)
    torch.set_num_threads(1)
    images = {r['id']: r for r in gt['images']}
    if len(images) != len(gt['images']):
        raise ValueError('Duplicate GT image IDs')
    truths = defaultdict(list)
    for r in gt['annotations']:
        if r['image_id'] not in images or r['category_id'] != 1 or r.get('iscrowd', 0):
            raise ValueError('Unknown image/category or unsupported crowd GT')
        x, y, w, h = r['bbox']; truths[r['image_id']].append([x, y, x+w, y+h])
    matcher, seen, tp_all, scores_all, n_gt, n_pred, empty = Matcher(), set(), [], [], 0, 0, 0
    expected_order = sorted(images)
    for row in records:
        image_id = row['image_id']
        if image_id not in images or image_id in seen:
            raise ValueError('Unknown/duplicate prediction image ID: ' + str(image_id))
        if image_id != expected_order[len(seen)]:
            raise ValueError('Prediction records must follow ascending image_id; global score ties depend on input order')
        im = images[image_id]
        if (row['split'], row['width'], row['height']) != (gt['info']['split'], im['width'], im['height']):
            raise ValueError('Split/dimensions differ for image ' + str(image_id))
        if row.get('box_format') != 'xyxy' or row.get('coordinate_space') != 'original_image_pixels':
            raise ValueError('Expected original-image pixel xyxy coordinates')
        predictions = row['predictions']
        if any(p['category_id'] != 1 or not math.isfinite(p['score']) or not 0 <= p['score'] <= 1 for p in predictions):
            raise ValueError('Invalid prediction class or confidence')
        boxes = [p['bbox'] for p in predictions]
        if not valid_boxes(boxes) or not valid_boxes(truths[image_id]):
            raise ValueError('Invalid finite positive-area xyxy box')
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        scores = torch.tensor([p['score'] for p in predictions], dtype=torch.float32)
        order = scores.argsort(descending=True, stable=True)
        b, scores = b[order], scores[order]
        mask = scores > POLICY['conf']  # compute mask AFTER sorting, never on unsorted confidence
        b, scores = b[mask][:POLICY['max_det']], scores[mask][:POLICY['max_det']]
        true = torch.tensor(truths[image_id], dtype=torch.float32).reshape(-1, 4)
        correct = matcher.match_predictions(torch.ones(len(b)), torch.ones(len(true)), box_iou(true, b))
        tp_all.append(correct.numpy()); scores_all.append(scores.numpy())
        n_gt += len(true); n_pred += len(b); empty += int(len(b) == 0); seen.add(image_id)
    if seen != set(images):
        raise ValueError('Missing explicit prediction records for ' + str(len(set(images)-seen)) + ' images')
    if not images:
        raise ValueError('Empty GT split')
    if n_gt:
        values = ap_per_class(np.concatenate(tp_all), np.concatenate(scores_all), np.ones(n_pred), np.ones(n_gt), plot=False)
        p, r, f1, ap = values[2:6]
        metrics = {'precision': float(p.mean()), 'recall': float(r.mean()), 'f1': float(f1.mean()),
                   'AP50': float(ap[:, 0].mean()), 'AP75': float(ap[:, 5].mean()), 'mAP50_95': float(ap.mean()), 'ap_by_class': ap.tolist()}
    else:
        metrics = {'precision': None, 'recall': None, 'AP50': None, 'AP75': None, 'mAP50_95': None, 'reason': 'No positive GT; AP undefined'}
    return {'status': 'COMPLETED_CPU_EXPORTED_PREDICTIONS', 'identity': identity, 'policy': POLICY,
            'policy_sha256': POLICY_SHA, 'images': len(images), 'ground_truth': n_gt, 'metric_predictions': n_pred,
            'empty_prediction_images': empty, 'ap_iou_thresholds': [round(.5+.05*i, 2) for i in range(10)],
            'runtime': {'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu'}, **metrics}


def read_public(path):
    with open_text(path) as stream:
        header = json.loads(next(stream))
        if header.get('type') != 'metadata' or header.get('schema_version') != 1:
            raise ValueError('First prediction line must contain schema_version=1 metadata')
    def rows():
        with open_text(path) as stream:
            next(stream)
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    return header['identity'], rows()


def read_legacy(cache, report, gt):
    """Lossless adapter for existing full-float cache only; rounded native JSON is deliberately unsupported."""
    d = json.loads(Path(report).read_text(encoding='utf-8'))
    if d.get('status') != 'completed' or d.get('policy') != POLICY['name'] or not d.get('export_complete') or d.get('split') != gt['info']['split']:
        raise ValueError('Legacy cache needs completed matching full-split metrics identity')
    if sha256(cache) != d.get('predictions_gt_sha256'):
        raise ValueError('Legacy cache hash differs from archived metrics')
    images = {r['file_name']: r for r in gt['images']}
    if digest('\n'.join(sorted(images)).encode()) != d['split_paths_sha256']:
        raise ValueError('Legacy split paths differ from independent GT')
    identity = {'model': 'archived_rtdetr', 'model_code_sha': d['runtime']['commit'], 'checkpoint_sha256': d['checkpoint_sha256'],
                'dataset_identity_sha256': gt['info']['dataset_identity_sha256'], 'evaluation_config_sha256': POLICY_SHA,
                'postprocessing': 'RT-DETR native 300 queries, stretch inversion, no NMS; high precision archive adapter',
                'legacy_cache_sha256': d['predictions_gt_sha256'], 'legacy_metrics_source': str(report)}
    def rows():
        with open_text(cache) as stream:
            for line in stream:
                row = json.loads(line); im = images[row['image']]
                if row['original_size_hw'] != [im['height'], im['width']] or row.get('box_format') != 'xyxy' or row.get('coordinate_space') != 'original_image_pixels':
                    raise ValueError('Legacy coordinate metadata differs')
                if any(p['class_id'] != 0 for p in row['predictions']):
                    raise ValueError('Unexpected legacy class')
                yield {'image_id': im['id'], 'split': d['split'], 'width': im['width'], 'height': im['height'],
                       'box_format': 'xyxy', 'coordinate_space': 'original_image_pixels',
                       'predictions': [{'category_id': 1, 'score': p['score'], 'bbox': p['bbox']} for p in row['predictions']]}
    return identity, rows(), d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--predictions', type=Path)
    group.add_argument('--legacy-cache', type=Path)
    parser.add_argument('--legacy-metrics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite evaluation output')
    gt = json.loads(args.gt.read_text(encoding='utf-8'))
    archived = None
    if args.legacy_cache:
        if not args.legacy_metrics:
            parser.error('--legacy-cache requires --legacy-metrics')
        identity, records, archived = read_legacy(args.legacy_cache, args.legacy_metrics, gt)
    else:
        identity, records = read_public(args.predictions)
    result = evaluate(gt, records, identity)
    result['gt_sha256'] = sha256(args.gt)
    if archived:
        result['archived_metric_comparison'] = {key: {'computed': result[key], 'archived': archived[old],
                                                     'delta': result[key]-archived[old]} for key, old in
                                               [('precision','precision'),('recall','recall'),('AP50','mAP50'),('AP75','AP75'),('mAP50_95','mAP50_95')]}
        result['comparison_note'] = 'Independent YOLO-derived GT and original-pixel FP32 arithmetic; inspect actual differences, never hardcode a reproduction pass'
    write_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('identity','ap_by_class')}, indent=2))


if __name__ == '__main__':
    main()
