"""Bounded FP32 prediction batches and lossless public schema records."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys
import uuid

from support import COMMON, canonical, digest, read_json, run_identity, sha256, write_json, validate_checkpoint

SETTINGS = {'imgsz': 640, 'batch': 16, 'workers': 0, 'conf': 0.001, 'iou': 0.7,
            'max_det': 300, 'half': False, 'augment': False, 'rect': False, 'seed': 42,
            'agnostic_nms': False, 'classes': None, 'device': 0, 'save': False,
            'save_txt': False, 'save_conf': False, 'verbose': False}
POSTPROCESSING = ('official YOLOv8 class-aware NMS conf>0.001 iou=0.7 max_det=300; '
    '640 square letterbox auto=False; inverse uses actual rounded resize gains x/y and integer left/top pad; no coordinate clipping or box filtering after NMS (padding-only false positives retained); '
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


def export_identity(run, split, manifest, source_identity):
    from configuration import frozen_config
    run = Path(run)
    configurable = (run / 'config_identity.json').exists()
    config = frozen_config(run) if configurable else None
    training = read_json(run / 'train_status.json')
    if training['status'] != 'completed':
        raise ValueError('Select the finished training best before exporting val/test')
    expected_identity = read_json(run / 'identity.json')
    if configurable and expected_identity.get('scope') != 'SMOKE_ONLY':
        completed = training.get('completed_epoch', 0)
        best_epoch = training.get('best_epoch', 0)
        valid_stop = (training.get('stop_reason') == 'epoch_limit' and completed == config['epochs']) or (
            training.get('stop_reason') == 'patience' and config['patience'] > 0
            and completed-best_epoch >= config['patience'] and completed < config['epochs'])
        if not valid_stop or not 1 <= best_epoch <= completed <= config['epochs'] or training.get('exit_code') != 0:
            raise ValueError('Training has no verified epoch-limit/patience completion')
    environment = read_json(run / 'environment.json') if configurable else None
    if run_identity(manifest, source_identity, run_id=read_json(run/'run_id.json')['run_id'],
                    config=config, environment=environment, run_uuid=read_json(run/'run_id.json').get('run_uuid')) != expected_identity:
        raise ValueError('Code/source/config/data/environment identity changed since training')
    checkpoint = run / 'train/weights/best.pt'
    if sha256(checkpoint) != training['best_sha256']:
        raise ValueError('Selected best checkpoint changed')
    gt_path = run / 'gt' / (split + '.json')
    return ({**expected_identity, 'checkpoint_sha256': training['best_sha256'],
             'evaluation_config_sha256': evaluator_api().POLICY_SHA, 'postprocessing': POSTPROCESSING,
             'inference': SETTINGS, 'source': source_identity, 'split': split, 'gt_sha256': sha256(gt_path)},
            config, checkpoint)


def export_split(run, split, manifest, source_identity):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from adapters import SquarePredictor, model_identity
    run = Path(run)
    identity, config, checkpoint = export_identity(run, split, manifest, source_identity)
    training = read_json(run / 'train_status.json')
    expected_identity = read_json(run/'identity.json')
    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(ckpt, expected_identity, config=config)
    del ckpt
    gt_path = run / 'gt' / (split+'.json')
    gt = read_json(gt_path)
    destination = run / 'predictions' / (split+'.jsonl.gz')
    done_path = run / 'predictions' / (split+'_complete.json')
    if done_path.exists() and not destination.exists():
        raise ValueError('Completed prediction cache file is missing; evidence retained')
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
    # Native compatibility key is half EMA; recover the stored selected EMA in FP32.
    selected = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.model.float().load_state_dict(selected['comparison_training_state']['ema'], strict=True)
    del selected
    model_identity(model.model, 1)
    init_seeds(42, deterministic=True)
    images = sorted(gt['images'], key=lambda im: im['id'])
    root = Path(manifest['data_root'])
    seen = 0
    with gzip.open(temporary, 'wt', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical({'type': 'metadata', 'schema_version': 1, 'identity': identity}).decode())
        for offset in range(0, len(images), 16):
            batch = images[offset:offset+16]
            # A full list passed to predict would eagerly load all images. Bound it to 16.
            results = list(model.predict(source=[str(root / im['file_name']) for im in batch],
                          predictor=SquarePredictor, stream=True, **SETTINGS,
                          project=str(run / 'native_predict'), name=split, exist_ok=True))
            if (len(results) != len(batch) or model.predictor.model.fp16
                    or next(model.predictor.model.model.parameters()).dtype != torch.float32):
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
    done = {'status': 'completed', 'images': seen, 'sha256': sha256(destination),
            'path': str(destination), 'checkpoint_sha256': training['best_sha256'], 'settings': SETTINGS}
    write_json(done_path, done)
    return done


def evaluate_split(run, split, manifest=None, source_identity=None):
    run = Path(run)
    prediction = run / 'predictions' / (split+'.jsonl.gz')
    gt_path = run / 'gt' / (split+'.json')
    done = read_json(run / 'predictions' / (split+'_complete.json'))
    if done['sha256'] != sha256(prediction):
        raise ValueError('Prediction cache checksum differs')
    api = evaluator_api()
    identity, records = api.read_public(prediction)
    if (run / 'config_identity.json').exists():
        if manifest is None or source_identity is None:
            raise ValueError('Configurable evaluation requires verified run inputs/source')
        expected, _, _ = export_identity(run, split, manifest, source_identity)
        if identity != expected:
            raise ValueError('Prediction cache config/checkpoint/GT/protocol identity differs')
    output = run / 'metrics' / (split + '.json')
    complete = run / 'metrics' / (split + '_complete.json')
    if output.is_file():
        cached = read_json(output)
        if (run / 'config_identity.json').exists() and (not complete.is_file()
                or read_json(complete).get('sha256') != sha256(output)):
            raise ValueError('Public metric cache completion/checksum differs; evidence retained')
        if (cached.get('identity') != identity or cached.get('gt_sha256') != sha256(gt_path)
                or cached.get('predictions_sha256') != done['sha256'] or cached.get('policy_sha256') != api.POLICY_SHA):
            raise ValueError('Existing metric cache identity differs; evidence retained')
        print('Reusing completed public metric cache: ' + str(output), flush=True)
        return {'metrics':str(output), **cached}
    result = api.evaluate(read_json(gt_path), records, identity)  # only the public AP implementation
    result.update(gt_sha256=sha256(gt_path), predictions_sha256=done['sha256'], units='raw fractions in [0,1]')
    fields = {'P': 'precision', 'R': 'recall', 'AP50': 'AP50', 'AP75': 'AP75', 'mAP50-95': 'mAP50_95'}
    result['raw_0_1'] = {name: result[key] for name, key in fields.items()}
    result['display_percent'] = {name: value*100 if value is not None else None for name, value in result['raw_0_1'].items()}
    # CPU recalculation is safe; retain different prior results instead of overwriting evidence.
    if output.exists() and read_json(output) != result:
        output = output.with_name(split + '_cpu_' + uuid.uuid4().hex[:8] + '.json')
    write_json(output, result)
    write_json(complete, {'sha256':sha256(output), 'predictions_sha256':done['sha256'],
                          'gt_sha256':sha256(gt_path), 'identity_sha256':digest(canonical(identity))})
    print(json.dumps({'split': split, 'raw_0_1': result['raw_0_1'], 'percent': result['display_percent']}), flush=True)
    return {'metrics': str(output), **result}
