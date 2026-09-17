"""Finite warmup regression and explicit partial THOP accounting; no training/test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ultralytics-main'))
import torch
from init_lif_down import require, write_json, runtime, sha256


def add_ops(module, count):
    """THOP versions and standalone diagnostic fixtures use Tensor or numeric counters."""
    if isinstance(module.total_ops, torch.Tensor):
        module.total_ops.add_(count)
    else:
        module.total_ops += count


def csr_conv_macs(module, inputs, output):
    """Count ALL four CSR convolutions once; this composite THOP hook owns its children.

    offset_pw uses functional FP32 convolution and is not detected by a leaf hook.
    Sampling and elementwise work are deliberately reported separately, never as full GFLOPs.
    """
    b, c, h, w = inputs[0].shape
    d = module.in_proj.out_channels
    add_ops(module, b * h * w * (c*d + d*9 + d*12 + d*c))


def warmup_check():
    from ultralytics.nn.autobackend import AutoBackend
    seen = []
    old = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    dummy = SimpleNamespace(pt=True, jit=False, onnx=False, engine=False, saved_model=False,
                            pb=False, triton=True, nn_module=False, device=torch.device('cpu'),
                            fp16=False, forward=lambda x: seen.append(x.clone()))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        AutoBackend.warmup(dummy, (1, 3, 32, 64))
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warning)
    require(len(seen) == 1 and torch.isfinite(seen[0]).all() and torch.count_nonzero(seen[0]) == 0,
            'AutoBackend warmup input is not finite zero')
    for counter in (0, torch.zeros(1, dtype=torch.float64)):
        fixture = SimpleNamespace(total_ops=counter)
        add_ops(fixture, 17)
        require(float(fixture.total_ops) == 17, 'THOP numeric/Tensor counter regression')
    return dict(status='PASSED', scope='real AutoBackend.warmup input generation, capture-only backend',
                shape=list(seen[0].shape), deterministic=True, finite=True, zero=True,
                thop_counter_types=['int', 'Tensor'])


def complexity():
    import thop
    from ultralytics.nn.modules.csr_p3 import CSR
    from ultralytics.nn.tasks import RTDETRDetectionModel
    directory = ROOT / 'ultralytics-main/ultralytics/cfg/models/rt-detr'
    rows = []
    names = ('rtdetr-resnet18-lite-cbr-lif-down', 'rtdetr-resnet18-lite-cbr-lif-csr-p3-v1',
             'rtdetr-resnet18-lite', 'rtdetr-resnet18-lite-csr-p3-v1')
    x = torch.zeros(1, 3, 640, 640)
    for name in names:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = RTDETRDetectionModel(str(directory / (name+'.yaml')), nc=1, verbose=False).eval()
        for fused in (False, True):
            candidate = deepcopy(model)
            if fused:
                candidate.fuse(verbose=False)
            parameters = sum(p.numel() for p in candidate.parameters())
            macs, _ = thop.profile(candidate, inputs=(x,), custom_ops={CSR: csr_conv_macs}, verbose=False)
            rows.append(dict(model=name, nc=1, input=[1, 3, 640, 640], fused=fused,
                             parameters=parameters, partial_thop_macs=macs, partial_2mac_gops=2*macs/1e9))
            del candidate
            gc.collect()
        del model
    # Analytic CSR supplement; the implementation skips two identically zero center differences.
    pixels, channels, points = 80*80, 32, 12
    return dict(status='PASSED', tool='thop '+getattr(thop,'__version__','unknown'), rows=rows,
                scope='Partial THOP registered-module MACs; CSR convolutions counted exactly once by composite hook. '
                      'Not total network GFLOPs: functional attention, CBR/LIF arithmetic, grids, sampling, '
                      'softmax, activation, and memory traffic are not fully counted.',
                csr_analytic_supplement=dict(convolution_macs=pixels*(256*32+32*9+32*12+32*256),
                    sample_calls_per_forward=24, bilinear_sample_values=2*points*channels*pixels,
                    interpolation_scalar_ops_estimate=7*2*points*channels*pixels,
                    weighted_difference_scalar_ops_estimate=3*points*channels*pixels,
                    assumptions='Four multiplies + three adds per bilinear output, then subtract/multiply/accumulate; '
                                'excludes coordinate generation, clamps, loads, tanh/cumsum/softmax and final residual add. '
                                'Analytic arithmetic only, not measured speed or complete GFLOPs.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--complexity', action='store_true')
    parser.add_argument('--native-amp', action='store_true', help='Execute actual native AMP resource check on local CUDA')
    parser.add_argument('--amp-check-weights', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing runtime report: '+str(args.output))
    torch.set_num_threads(4)
    report = dict(runtime=runtime(), warmup=warmup_check())
    if args.complexity:
        report['complexity'] = complexity()
    if args.native_amp:
        from init_csr_p3 import build
        from preflight_csr_p3 import enforced_amp_check
        from ultralytics.utils import ASSETS
        require(torch.cuda.is_available(), 'CUDA required for native AMP check')
        require(args.amp_check_weights is not None, '--amp-check-weights required')
        model = build(nc=1).cuda()
        enforced_amp_check(model, args.amp_check_weights)
        report['native_amp'] = dict(status='PASSED', device=torch.cuda.get_device_name(),
            checkpoint_sha256=sha256(args.amp_check_weights), asset_sha256=sha256(ASSETS/'bus.jpg'),
            scope='Actual unchanged check_amp; resource failure/skip is rejected; not B16/640 capacity')
    write_json(args.output, report)


if __name__ == '__main__':
    main()
