"""Small synthetic GPU train -> deliberate interrupt -> explicit resume -> best export."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import random
import subprocess
import sys
import uuid

import numpy as np
import torch

from support import HERE, code_identity, environment, read_json, recipe, run_identity, sha256, status, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--local-compat', action='store_true')
    p.add_argument('--probe-receipt', type=Path, help='Reuse a just-completed probe; still verifies isolated child state')
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    from engine import checked_pair, train, validate_checkpoint
    from export import evaluate_split, export_split
    from model import build_model, seed_all
    from test_core import fixture
    env = environment(strict=not a.local_compat)
    seed_all()
    parent, _ = build_model()
    # Check that an independent process cannot consume parent's model/BN/optimizer/scaler/RNG.
    optimizer = torch.optim.SGD(parent.parameters(), lr=.02)
    scaler = torch.cuda.amp.GradScaler(init_scale=128)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state, cuda_state = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    optim_state, scale_state = optimizer.state_dict(), scaler.state_dict()
    probe_path = a.output/'isolated_probe.json'
    argv = [sys.executable, str(HERE/'run.py'), 'check', '--output', str(probe_path)]
    if a.local_compat:
        argv.append('--local-compat')
    if a.probe_receipt:
        # The original probe already ran as its own CLI process. A no-GPU child checks the same process boundary.
        probe = read_json(a.probe_receipt)
        if not probe['status'].startswith('PASSED_') or probe['environment'] != env:
            raise ValueError('Probe receipt does not match this environment')
        subprocess.run([sys.executable, '-c', 'import random; random.seed(999); import torch; torch.manual_seed(999)'], check=True)
        write_json(probe_path, {**probe, 'reused_from': str(a.probe_receipt), 'sha256': sha256(a.probe_receipt)})
    else:
        subprocess.run(argv, check=True)
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in before.items())
    assert optimizer.state_dict() == optim_state and scaler.state_dict() == scale_state
    assert py_state == random.getstate() and np.array_equal(np_state[1], np.random.get_state()[1])
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(cuda_state, torch.cuda.get_rng_state_all()))
    del parent, before, optimizer, scaler
    gc.collect(); torch.cuda.empty_cache()
    root, run, manifest = fixture(a.output, count=4)
    cfg = {**recipe(), 'input_height': 64, 'input_width': 64, 'model_transform_min_size': 64,
           'model_transform_max_size': 64, 'batch_size': 2, 'eval_batch_size': 2, 'train_workers': 0,
           'epochs': 3, 'warmup_epochs': 1, 'close_mosaic': 1}
    ident = run_identity(manifest, 'SMOKE_'+uuid.uuid4().hex, cfg, env,
                         scope='SMOKE_ONLY_SYNTHETIC', require_clean=False)
    write_json(run/'run_id.json', {'run_uuid': ident['run_uuid'], 'scope': ident['scope']})
    source_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}
    try:
        train(run, manifest, ident, cfg, env, interrupt_after_epoch=1)
    except KeyboardInterrupt:
        status(run, 'train', 'interrupted', scope=ident['scope'], exit_code=130, reason='intentional synthetic epoch-boundary interruption')
    else:
        raise AssertionError('Synthetic interruption was not exercised')
    ckpt, index = checked_pair(run, ident, cfg)
    assert index['completed_epoch'] == 1 and ckpt['schedule']['steps_done'] == 2
    for key, value in [('model', 'YOLO'), ('run_uuid', 'other-run'), ('initialization_type', 'coco'), ('scope', 'FORMAL')]:
        try:
            validate_checkpoint({**ckpt, 'comparison_identity': {**ident, key: value}}, ident, cfg, resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Foreign checkpoint was accepted')
    initialization_before = (run/'initialization.json').read_bytes()
    del ckpt
    gc.collect(); torch.cuda.empty_cache()
    train(run, manifest, ident, cfg, env, resume=True)
    assert (run/'initialization.json').read_bytes() == initialization_before
    last, final_index = checked_pair(run, ident, cfg)
    assert last['schedule']['steps_done'] == 6 and final_index['completed_epoch'] == 3
    assert last['history'][2]['observed_augmentation_calls']['mosaic'] == 0
    assert last['history'][2]['observed_augmentation_calls']['mixup'] == 0
    del last
    gc.collect(); torch.cuda.empty_cache()
    exported = {}
    for split in ('val', 'test'):
        done = export_split(run, split, manifest, ident, cfg)
        result = evaluate_split(run, split)
        exported[split] = {'images': done['images'], 'best_sha256': done['checkpoint_sha256'],
                           'metric_status': result['status'], 'raw_0_1': result['raw_0_1']}
    assert exported['val']['best_sha256'] == exported['test']['best_sha256']
    assert source_hashes == {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}
    report = {'status': 'PASSED_SYNTHETIC_GPU_LIFECYCLE', 'scope': ident['scope'], 'environment': env,
        'code_sha256': code_identity(), 'run': str(run.resolve()), 'batch': 2, 'size': 64, 'epochs': 3,
        'completed_optimizer_steps': 6, 'interrupt_after_epoch': 1, 'resumed_epochs': [2, 3],
        'parent_model_BN_optimizer_scaler_Python_NumPy_CPU_CUDA_RNG_unchanged': True,
        'initialization_record_preserved': True, 'foreign_checkpoint_rejections': 4,
        'source_images_labels_and_neighbor_caches_unchanged': True, 'exported_actual_smoke_best': exported,
        'formal_training_started': False, 'full_split_inference_started': False,
        'formal_batch16_640_parallel_capacity': 'NOT_TESTED',
        'target_versions_verified': not a.local_compat}
    write_json(a.output/'smoke.json', report)
    print('SYNTHETIC GPU LIFECYCLE PASSED: '+str(a.output/'smoke.json'), flush=True)


if __name__ == '__main__':
    main()
