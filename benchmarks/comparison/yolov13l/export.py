"""Bounded FP32 prediction batches and lossless public schema records."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys
import uuid

from support import COMMON, canonical, read_json, run_identity, sha256, write_json, validate_checkpoint

SETTINGS = {'imgsz': 640, 'batch': 16, 'workers': 0, 'conf': 0.001, 'iou': 0.7,
            'max_det': 300, 'half': False, 'augment': False, 'rect': False, 'seed': 42,
            'agnostic_nms': False, 'classes': None, 'device': 0, 'save': False,
            'save_txt': False, 'save_conf': False, 'verbose': False}
POSTPROCESSING = ('official YOLOv13 class-aware NMS conf>0.001 iou=0.7 max_det=300 max_nms=30000 no-time-truncation; '
    '640 square letterbox auto=False; native scale_boxes clips and restores original pixels once; '
    'Results.xyxy exported directly without another inverse transform; FP32, no TTA')


def evaluator_api():
    sys.path.insert(0, str(COMMON / 'evaluation'))
    import evaluate
    return evaluate


def result_record(result, im, split):
    if tuple(result.orig_shape) != (im['height'], im['width']):
        raise ValueError('Decoded original dimensions differ for ' + im['file_name'])
    rows = []
    for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                result.boxes.cls.cpu().tolist()):
        if cls != 0 or not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4 or not all(
                math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Invalid prediction at ' + im['file_name'] + ': ' + str((box, score, cls)))
        rows.append({'category_id': 1, 'bbox': box, 'score': score})
    return {'split': split, 'image_id': im['id'], 'image': im['file_name'], 'width': im['width'],
            'height': im['height'], 'box_format': 'xyxy', 'coordinate_space': 'original_image_pixels',
            'predictions': rows}


def export_split(run, split, manifest, source_identity):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from adapters import SquarePredictor, model_identity
    from backend import ArithmeticAudit, native_fp32
    from bootstrap import verify_frozen_environment
    verify_frozen_environment()
    run = Path(run)
    training = read_json(run / 'train_status.json')
    if training['status'] != 'completed':
        raise ValueError('Select the finished training best before exporting val/test')
    expected_identity = read_json(run/'identity.json')
    if run_identity(manifest, source_identity, run_id=read_json(run/'run_id.json')['run_id']) != expected_identity:
        raise ValueError('Code/source/recipe/data identity changed since training')
    checkpoint = run / 'train/weights/best.pt'
    if sha256(checkpoint) != training['best_sha256']:
        raise ValueError('Selected best checkpoint changed')
    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(ckpt, expected_identity)
    if ckpt['comparison_initialization_sha256'] != sha256(run/'initialization.json') or ckpt['epoch']+1 != training['best_epoch']:
        raise ValueError('Best epoch or initialization receipt differs')
    del ckpt
    gt_path = run / 'gt' / (split+'.json')
    gt = read_json(gt_path)
    identity = {**read_json(run / 'identity.json'), 'checkpoint_sha256': training['best_sha256'],
                'evaluation_config_sha256': evaluator_api().POLICY_SHA, 'postprocessing': POSTPROCESSING,
                'inference': SETTINGS, 'source': source_identity, 'split': split, 'gt_sha256': sha256(gt_path)}
    destination = run / 'predictions' / (split+'.jsonl.gz')
    done_path = run / 'predictions' / (split+'_complete.json')
    if destination.exists():
        done = read_json(done_path)
        old, _ = evaluator_api().read_public(destination)
        if old != identity or done['sha256'] != sha256(destination):
            raise ValueError('Existing prediction identity/content differs; never overwrite')
        print('Reusing completed prediction cache: ' + str(destination), flush=True)
        return done
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
    model = YOLO(str(checkpoint), task='detect')
    model_identity(model.model, 1)
    init_seeds(42, deterministic=True)
    images = sorted(gt['images'], key=lambda im: im['id'])
    root = Path(manifest['data_root'])
    seen = 0
    with native_fp32(), ArithmeticAudit(require_fp32=True) as dtype_audit, gzip.open(temporary, 'wt', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical({'type': 'metadata', 'schema_version': 1, 'identity': identity}).decode())
        for offset in range(0, len(images), 16):
            batch = images[offset:offset+16]
            # A full list passed to predict would eagerly load all images. Bound it to 16.
            results = list(model.predict(source=[str(root / im['file_name']) for im in batch],
                          predictor=SquarePredictor, stream=True, **SETTINGS,
                          project=str(run / 'native_predict'), name=split, exist_ok=True))
            if len(results) != len(batch) or model.predictor.model.fp16:
                raise ValueError('Missing result or unexpected FP16 predictor')
            for result, im in zip(results, batch):
                if Path(result.path).resolve() != (root / im['file_name']).resolve():
                    raise ValueError('Prediction path/order differs: ' + str(result.path))
                stream.write(canonical(result_record(result, im, split)).decode())
                seen += 1
            print(f'Export {split}: {seen}/{len(images)}', flush=True)
    if seen != len(images):
        raise ValueError('Incomplete split export')
    temporary.rename(destination)
    write_json(run/'predictions'/(split+'_fp32_arithmetic.json'), dtype_audit.report())
    done = {'status': 'completed', 'images': seen, 'sha256': sha256(destination),
            'path': str(destination), 'checkpoint_sha256': training['best_sha256'], 'settings': SETTINGS, 'backend': 'native', 'internal_fp32_audit': str(run/'predictions'/(split+'_fp32_arithmetic.json'))}
    write_json(done_path, done)
    return done


def evaluate_split(run, split):
    run = Path(run)
    prediction = run / 'predictions' / (split+'.jsonl.gz')
    gt_path = run / 'gt' / (split+'.json')
    done = read_json(run / 'predictions' / (split+'_complete.json'))
    if done['sha256'] != sha256(prediction):
        raise ValueError('Prediction cache checksum differs')
    api = evaluator_api()
    identity, records = api.read_public(prediction)
    result = api.evaluate(read_json(gt_path), records, identity)  # only the public AP implementation
    result.update(gt_sha256=sha256(gt_path), predictions_sha256=done['sha256'], units='raw fractions in [0,1]')
    fields = {'P': 'precision', 'R': 'recall', 'AP50': 'AP50', 'AP75': 'AP75', 'mAP50-95': 'mAP50_95'}
    result['raw_0_1'] = {name: result[key] for name, key in fields.items()}
    result['display_percent'] = {name: value*100 if value is not None else None for name, value in result['raw_0_1'].items()}
    output = run / 'metrics' / (split+'.json')
    # CPU recalculation is safe; retain different prior results instead of overwriting evidence.
    if output.exists() and read_json(output) != result:
        output = output.with_name(split + '_cpu_' + uuid.uuid4().hex[:8] + '.json')
    write_json(output, result)
    print(json.dumps({'split': split, 'raw_0_1': result['raw_0_1'], 'percent': result['display_percent']}), flush=True)
    return {'metrics': str(output), **result}
