"""Native ROI postprocessing, one data-layer inverse, shared CPU metrics."""
from __future__ import annotations

import gzip
import hashlib
import math
import os
from pathlib import Path
import sys
import uuid

from support import COMMON, canonical, read_json, sha256, write_json

POSTPROCESSING = ('Native RPN NMS 0.7 train 2000/test 1000 proposals; native ROI class-aware NMS '
    'score>0.001 IoU0.7 max300 (inside model); no additional NMS; square letterbox RGB [0,1]; '
    'model normalization and {size} transform; one inverse using actual integer resized dimensions '
    'and padding; float32 original-image pixel xyxy, no extra clipping/rounding, FP32, no TTA')


def evaluator_api():
    sys.path.insert(0, str(COMMON/'evaluation'))
    import evaluate
    return evaluate


def prediction_identity(identity, checkpoint_sha, split, gt_path, cfg):
    return {**identity, 'checkpoint_sha256': checkpoint_sha, 'split': split, 'gt_sha256': sha256(gt_path),
            'evaluation_config_sha256': evaluator_api().POLICY_SHA,
            'postprocessing': POSTPROCESSING.format(size=cfg['input_height']),
            'inference': {'batch': cfg['eval_batch_size'], 'workers': cfg['eval_workers'],
                'input_size': [cfg['input_height'], cfg['input_width']], 'precision': 'FP32', 'tta': False,
                'foreground_to_public': {'1': 1}, 'background': 0,
                'box_score_thresh': cfg['box_score_thresh'], 'box_nms_thresh': cfg['box_nms_thresh'],
                'box_detections_per_img': cfg['box_detections_per_img']}}


def result_record(output, im, geometry, split):
    import torch
    from data_adapter import inverse_boxes
    if output['boxes'].dtype != torch.float32 or output['scores'].dtype != torch.float32:
        raise ValueError('Prediction output must be FP32')
    boxes = inverse_boxes(output['boxes'], geometry).tolist()
    scores, labels = output['scores'].cpu().tolist(), output['labels'].cpu().tolist()
    if not len(boxes) == len(scores) == len(labels) or len(boxes) > 300:
        raise ValueError('Malformed native detection output')
    predictions = []
    for box, score, label in zip(boxes, scores, labels):
        if label != 1 or not math.isfinite(score) or not .001 < score <= 1 or not all(math.isfinite(v) for v in box):
            raise ValueError('Invalid foreground/score/box output at ' + im['file_name'])
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Invalid box area at ' + im['file_name'])
        predictions.append({'category_id': 1, 'bbox': box, 'score': score})
    return {'split': split, 'image_id': im['id'], 'image': im['file_name'],
        'width': im['width'], 'height': im['height'], 'box_format': 'xyxy',
        'coordinate_space': 'original_image_pixels', 'geometry': geometry, 'predictions': predictions}


def predict_records(model, manifest, gt, cfg, device):
    import torch
    from data_adapter import EvalDataset, collate
    from tqdm import tqdm
    ds = EvalDataset(manifest['data_root'], gt, cfg['input_height'])
    # Use an independent generator: validation does not advance training loader RNG.
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['eval_batch_size'],
        num_workers=cfg['eval_workers'], collate_fn=collate, shuffle=False,
        generator=torch.Generator().manual_seed(cfg['seed']), persistent_workers=False)
    model.float().eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
        for images, metas, geometries in tqdm(loader, desc='FP32 '+gt['info']['split'], dynamic_ncols=True):
            outputs = model([im.to(device) for im in images])
            if len(outputs) != len(metas):
                raise ValueError('Missing model outputs')
            for output, meta, geometry in zip(outputs, metas, geometries):
                yield result_record(output, meta, geometry, gt['info']['split'])


def metric_units(result):
    names = {'P': 'precision', 'R': 'recall', 'AP50': 'AP50', 'AP75': 'AP75', 'mAP50-95': 'mAP50_95'}
    return {**result, 'raw_0_1': {n: result[k] for n, k in names.items()},
        'display_percent': {n: result[k]*100 if result[k] is not None else None for n, k in names.items()},
        'units': {'raw_0_1': 'fraction', 'display_percent': 'percent'},
        'PR_working_point': 'public implementation smoothed max-F1 at AP50'}


def validate_epoch(model, run, manifest, cfg, identity, epoch, device):
    gt_path = Path(run)/'gt/val.json'
    gt = read_json(gt_path)
    # Epoch-only identity; no checkpoint has been saved yet. These records are not final caches.
    fingerprint = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        fingerprint.update(canonical({'name': name, 'shape': list(array.shape), 'dtype': str(array.dtype)}))
        fingerprint.update(array.tobytes())
    epoch_identity = prediction_identity(identity, fingerprint.hexdigest(), 'val', gt_path, cfg)
    epoch_identity['checkpoint_role'] = 'UNSAVED_CURRENT_EPOCH_' + str(epoch)
    epoch_identity['checkpoint_digest_kind'] = 'actual in-memory state tensors; final exports use checkpoint FILE SHA256'
    result = metric_units(evaluator_api().evaluate(gt, predict_records(model, manifest, gt, cfg, device), epoch_identity))
    score = result['mAP50_95']
    if score is None or not math.isfinite(score):
        raise ValueError('Validation mAP is undefined/nonfinite; cannot select a checkpoint')
    write_json(Path(run)/'epoch_metrics'/f'{epoch:04d}.json', result)
    return result


def export_split(run, split, manifest, identity, cfg):
    import torch
    from engine import load_checkpoint, validate_checkpoint
    from model import build_model, seed_all
    run = Path(run)
    state = read_json(run/'train_status.json')
    if state['status'] != 'completed':
        raise ValueError('val/test final export requires completed training and fixed best')
    checkpoint = run/'checkpoints/best.pt'
    best_sha = sha256(checkpoint)
    if state['best_sha256'] != best_sha:
        raise ValueError('Selected best changed')
    ckpt = load_checkpoint(checkpoint)
    validate_checkpoint(ckpt, identity, cfg)
    if ckpt['completed_epoch'] != state['best_epoch']:
        raise ValueError('Best epoch mismatch')
    gt_path = run/'gt'/(split+'.json')
    export_id = prediction_identity(identity, best_sha, split, gt_path, cfg)
    destination = run/'predictions'/(split+'.jsonl.gz')
    receipt = run/'predictions'/(split+'_complete.json')
    if destination.exists():
        done = read_json(receipt)
        old, rows = evaluator_api().read_public(destination)
        if old != export_id or done['sha256'] != sha256(destination):
            raise ValueError('Protected existing prediction cache has different identity/content')
        return done
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    seed_all(cfg['seed'], cfg['deterministic'])
    model, _ = build_model(cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    model.to(cfg['device']).float().eval()
    gt = read_json(gt_path)
    seen = 0
    with gzip.open(temporary, 'wt', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical({'type': 'metadata', 'schema_version': 1, 'identity': export_id}).decode())
        for row in predict_records(model, manifest, gt, cfg, cfg['device']):
            stream.write(canonical(row).decode())
            seen += 1
    if seen != len(gt['images']):
        raise ValueError('Incomplete prediction image set')
    os.replace(temporary, destination)
    done = {'status': 'completed', 'images': seen, 'sha256': sha256(destination),
            'checkpoint_sha256': best_sha, 'best_epoch': state['best_epoch'], 'path': str(destination),
            'precision': 'FP32', 'scope': identity['scope']}
    write_json(receipt, done)
    return done


def evaluate_split(run, split):
    run = Path(run)
    destination = run/'predictions'/(split+'.jsonl.gz')
    done = read_json(run/'predictions'/(split+'_complete.json'))
    if done['sha256'] != sha256(destination):
        raise ValueError('Prediction cache checksum changed')
    identity, records = evaluator_api().read_public(destination)
    gt_path = run/'gt'/(split+'.json')
    if identity['gt_sha256'] != sha256(gt_path):
        raise ValueError('Ground truth changed')
    stored = read_json(run/'identity.json')
    training = read_json(run/'train_status.json')
    if training['status'] != 'completed':
        raise ValueError('Final CPU evaluation requires a completed selected best')
    expected = prediction_identity(stored, training['best_sha256'], split, gt_path, stored['recipe'])
    if identity != expected or done['checkpoint_sha256'] != training['best_sha256']:
        raise ValueError('Prediction cache belongs to another run/model/config/selected best')
    result = metric_units(evaluator_api().evaluate(read_json(gt_path), records, identity))
    result.update(predictions_sha256=done['sha256'], gt_sha256=sha256(gt_path))
    output = run/'metrics'/(split+'.json')
    if output.exists() and read_json(output) != result:
        output = output.with_name(split+'_cpu_'+uuid.uuid4().hex[:8]+'.json')
    write_json(output, result)
    print(split + ': ' + str(result['display_percent']), flush=True)
    return {'metrics': str(output), **result}
