"""Explicit future val/test of the training-selected best/EMA; never called by plan/init."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from init_bscrep_v1 import ROOT, VARIANTS, require, sha256, runtime, verify_model, write_json
from bscrep_v1_common import DEFAULT_VARIANT, paths, verified_data, git, code_hashes
from c19_lif_v1_results import EVAL, POLICY, postprocess
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator


def evaluate(variant, split, output, main=None, val_report=None):
    p = paths(variant, main)
    weights, output = p['run']/'weights/best.pt', Path(output).resolve()
    require(not p['run'].with_name(p['run'].name + '.bscrep-active').exists(), 'Training is active; freeze its selected best checkpoint first')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Evaluation source must match the delivered commit')
    require(weights.is_file(), 'Training-selected best.pt is required; no last.pt substitution')
    require(not output.exists(), 'Existing evaluation protected')
    _, inventory = verified_data(p['data'])
    digest, data_digest, info, sources = sha256(weights), sha256(p['data']), runtime(), code_hashes()
    if split == 'test':
        require(val_report and Path(val_report).is_file(), 'First validate this fixed best checkpoint')
        prior = json.loads(Path(val_report).read_text())
        require(prior['status'] == 'completed' and prior['split'] == 'val' and
                prior['checkpoint_sha256'] == digest and prior['data_sha256'] == data_digest and
                prior['runtime']['commit'] == info['commit'] and prior['settings'] == EVAL and
                prior['policy'] == POLICY and prior['variant'] == variant and prior.get('source_hashes') == sources,
                'Val/test checkpoint/config/policy mismatch')
    model = RTDETR(str(weights))  # Native loader selects EMA when present, as in parent protocol.
    verify_model(model.model, variant)
    require(model.model.model[-1].nc == 1, 'Expected task nc=1')
    report = dict(status='failed', variant=variant, split=split, runtime=info, settings=EVAL,
                  checkpoint=str(weights), checkpoint_sha256=digest, data_sha256=data_digest,
                  policy=POLICY, dataset_inventory=inventory, source_hashes=sources, selection='training val best; native EMA loader',
                  precision_recall_policy='each model own maximum-F1 working point')
    output.mkdir(parents=True, exist_ok=False)
    class ParentPolicyValidator(RTDETRValidator):
        def postprocess(self, predictions):
            return postprocess(predictions, self.args.imgsz, self.args.conf)[0]
        def init_metrics(self, model):
            super().init_metrics(model)
            require(not self.training and not self.args.half, 'Parent independent FP32 evaluation required')
            require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v
                        for k, v in EVAL.items()), 'Evaluation threshold/recipe changed')
    try:
        metrics = model.val(validator=ParentPolicyValidator, **EVAL, data=str(p['data']), split=split,
                            device='0', plots=True, save_json=False, save_txt=False,
                            project=str(output), name='plots', exist_ok=False)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), 'Nonfinite/unexpected AP')
        require(sha256(weights) == digest and sha256(p['data']) == data_digest, 'Evaluation input changed')
        require(code_hashes() == sources, 'Evaluation source changed while running')
        report.update(status='completed', precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      AP75=float(ap[:, 5].mean()), ap_by_class=ap.tolist(), speed_ms_per_image=metrics.speed)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(output/'metrics.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('val', 'test'))
    parser.add_argument('--variant', choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument('--main', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--val-report', type=Path)
    args = parser.parse_args()
    evaluate(args.variant, args.split, args.output, args.main, args.val_report)
