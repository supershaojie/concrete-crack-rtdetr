"""Explicit later val/test of the training-selected best/EMA; never runs during initialization/preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from init_rcsq_v1 import require, runtime, sha256, verify_model, write_json
from train_rcsq_v1 import load_plan
from c19_lif_v1_results import EVAL, POLICY, postprocess
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator


def evaluate(plan_path, split, output, device='0', val_report=None):
    p = load_plan(plan_path)
    output = Path(output).resolve()
    require(split in ('val', 'test') and not output.exists(), 'Evaluation split invalid or output already exists')
    launch = Path(p['output']) / 'launch'
    result_path = launch / 'latest_training_result.json'
    if not result_path.is_file():
        result_path = launch / 'training_result.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    require(result['status'] in ('COMPLETED_200_EPOCHS', 'PATIENCE_EARLY_STOP'), 'Formal training completion/early-stop evidence missing')
    weights = Path(p['args']['save_dir']) / 'weights/best.pt'
    digest = sha256(weights)
    require(digest == result['best_sha256'], 'Selected best checkpoint differs from recorded training result')
    if split == 'test':
        require(val_report is not None, 'Validate the selected best checkpoint before its independent test')
        prior = json.loads(Path(val_report).read_text(encoding='utf-8'))
        require(prior.get('status') == 'COMPLETED' and prior.get('split') == 'val' and prior.get('checkpoint_sha256') == digest and
                prior.get('data_sha256') == p['data_sha256'] and prior.get('policy') == POLICY and
                prior.get('code_fingerprint') == p['code']['sha256'], 'Val/test checkpoint, protocol, data, or source changed')
        require(all(type(prior['settings'][k]) is type(v) and prior['settings'][k] == v for k, v in EVAL.items()), 'Val/test settings differ')
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(EVAL, split=split, data=p['args']['data'], device=device, plots=True, save_json=False, save_txt=False,
                    project=str(output), name='plots', exist_ok=False)
    report = dict(status='FAILED', split=split, variant=p['variant'], checkpoint=str(weights), checkpoint_sha256=digest,
                  checkpoint_source='native loader prefers saved EMA', data_sha256=p['data_sha256'], policy=POLICY,
                  code_fingerprint=p['code']['sha256'], runtime=runtime(), settings=settings, dataset=p['data_inventory'][split],
                  selection='training val best; same frozen checkpoint for independent val/test',
                  precision_recall_policy='each model own maximum-F1 working point', evidence_scope='full_split')
    counts = dict(images=0, ground_truth=0)

    class ProtocolValidator(RTDETRValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            require(not self.training and not self.args.half, 'Independent FP32 evaluation required')
            require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()), 'Effective evaluation recipe changed')
            report['actual_settings'] = vars(self.args).copy()

        def postprocess(self, preds):
            # Original experiment's corrected sorted-confidence policy, without NMS or altered boxes.
            return postprocess(preds, self.args.imgsz, self.args.conf)[0]

        def update_metrics(self, preds, batch):
            counts['images'] += len(preds)
            counts['ground_truth'] += len(batch['cls'])
            return super().update_metrics(preds, batch)

    try:
        model = RTDETR(str(weights))
        verify_model(model.model, p['variant'], zero=False)
        require(model.model.model[-1].nc == 1, 'Evaluation expects nc=1')
        report['parameters_unfused'] = sum(v.numel() for v in model.model.parameters())
        report['out_proj_norm'] = float(model.model.model[-1].rcsq.out_proj.weight.float().norm())
        metrics = model.val(validator=ProtocolValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), 'Unexpected/nonfinite AP array')
        require(counts['images'] == p['data_inventory'][split]['images'] and
                counts['ground_truth'] == p['data_inventory'][split]['boxes'], 'Incomplete evaluation split')
        require(sha256(weights) == digest, 'Checkpoint changed during evaluation')
        report.update(status='COMPLETED', precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP50=float(metrics.box.map50), AP75=float(ap[:, 5].mean()), AP50_95=float(metrics.box.map),
                      ap_by_class=ap.tolist(), speed_ms_per_image=metrics.speed, **counts)
    except BaseException as error:
        report.update(error=repr(error), **counts)
        raise
    finally:
        write_json(output / 'metrics.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--val-report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.plan, args.split, args.output, args.device, args.val_report)
