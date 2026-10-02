"""Honest partial operation accounting, and a separate opt-in exclusive speed entry."""
from __future__ import annotations

import os
import subprocess
import time

from support import environment, read_json, recipe, sha256, write_json


def measure(output, local_compat=False):
    import torch
    from model import build_model, seed_all
    env = environment(strict=not local_compat)
    torch.set_num_threads(2)
    seed_all()
    model, init = build_model()
    model.cpu().float().eval()
    observed = {}
    def rpn_hook(module, inputs, outputs):
        observed['rpn_post_nms_proposals'] = [len(b) for b in outputs[0]]
    def roi_hook(module, inputs):
        observed['roi_align_proposals'] = [len(b) for b in inputs[1]]
    model.rpn.register_forward_hook(rpn_hook)
    model.roi_heads.box_roi_pool.register_forward_pre_hook(roi_hook)
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True, with_flops=True) as profiler:
        predictions = model([torch.zeros(3, 640, 640)])
    events = profiler.key_averages()
    counted = {e.key: {'calls': e.count, 'FLOPs': e.flops} for e in events if e.flops}
    zeros = {e.key: e.count for e in events if not e.flops}
    flops = sum(e.flops for e in events)
    observed['final_detections'] = [len(p['boxes']) for p in predictions]
    parameters = sum(p.numel() for p in model.parameters())
    result = {'status': 'PARTIAL_OPERATION_ACCOUNTING_NOT_A_TOTAL_GFLOPS_CLAIM',
        'environment': env, 'foreground_classes': 1, 'num_classes_including_background': 2,
        'input': {'batch': 1, 'RGB_CHW': [3, 640, 640], 'pixels': 'zeros', 'seed': 42, 'device': 'cpu', 'precision': 'FP32'},
        'training_model_parameters': parameters, 'deployment_model_parameters': parameters,
        'deployment_model': 'same unfused architecture and state_dict; no ONNX/TensorRT conversion',
        'fused': False, 'observed': observed, 'GFLOPs_total': None,
        'FLOPs_counted': flops, 'MAC_equivalents_counted': flops/2, 'GFLOPs_counted_2x_MACs': flops/1e9,
        'counted_operations': counted, 'unaccounted_or_zero_flop_operations': zeros,
        'method': 'torch.profiler with_flops=True; multiply-add=2 operations. Conv/matrix and supported arithmetic '
            'only; native NMS, RoIAlign, pooling, sorting, interpolation and other zero-flop entries are NOT silently counted. '
            'Dynamic proposals are measured on this input. The partial value must not enter a paper as total GFLOPs.',
        'speed': 'not measured; use separate speed --confirm-exclusive after training'}
    write_json(output, result)
    print('Parameters='+str(parameters)+'; counted GFLOPs='+str(flops/1e9)+' (partial, total unavailable)', flush=True)
    return result


def speed(args):
    if not args.confirm_exclusive:
        raise ValueError('Speed benchmark needs --confirm-exclusive; this is never a training prerequisite')
    import numpy as np
    import torch
    from data import verify_inputs
    from data_adapter import EvalDataset
    from engine import load_checkpoint, validate_checkpoint
    from model import build_model, seed_all
    from run import strict_identity
    run = args.run.resolve()
    manifest = verify_inputs(run, ('val',))
    identity, cfg, env = strict_identity(run, manifest)
    state = read_json(run/'train_status.json')
    checkpoint = run/'checkpoints/best.pt'
    if state['status'] != 'completed' or sha256(checkpoint) != state['best_sha256']:
        raise ValueError('Measure only the fixed completed best')
    other = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name',
                                     '--format=csv,noheader,nounits'], text=True)
    if any(line.strip() and line.split(',')[0].strip() != str(os.getpid()) for line in other.splitlines()):
        raise RuntimeError('Other GPU compute processes detected; exclusive speed result unavailable')
    seed_all()
    model, _ = build_model(cfg)
    ckpt = load_checkpoint(checkpoint)
    validate_checkpoint(ckpt, identity, cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    model.cuda().float().eval()
    gt = read_json(run/'gt/val.json')
    ds = EvalDataset(manifest['data_root'], gt)
    timings = []
    with torch.inference_mode():
        for i in range(120):
            # Preprocessing and transfers are outside timed model latency, explicitly reported.
            image = ds[i % min(len(ds), 32)][0].cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model([image])
            torch.cuda.synchronize()
            if i >= 20:
                timings.append((time.perf_counter()-start)*1000)
    result = {'status': 'EXCLUSIVE_REQUESTED_AND_PROCESS_LIST_CHECKED', 'scope': 'FP32 model latency, batch1, no TTA',
        'checkpoint_sha256': state['best_sha256'], 'environment': env, 'gpu': torch.cuda.get_device_name(0),
        'warmup_iterations': 20, 'measured_iterations': 100, 'raw_milliseconds': timings,
        'median_ms': float(np.median(timings)), 'mean_ms': float(np.mean(timings)),
        'p95_ms': float(np.percentile(timings, 95)), 'mean_fps': 1000/float(np.mean(timings)),
        'includes': 'backbone/FPN/RPN/NMS/RoIAlign/ROI postprocessing',
        'excludes': 'disk IO, data letterbox, H2D transfer and data inverse',
        'note': 'User must maintain exclusive GPU access during the whole measurement; no process is stopped by this tool'}
    write_json(args.output, result)
    return result
