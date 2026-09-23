"""Bounded pre-selection mother/ROR control; never changes a training plan or gate."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import torch
from init_c19_lif_v1 import ROOT, SOURCE_SHA256, controlled_models, build_training_model, require, runtime, sha256
from c19_lif_v1_diagnostic import (LIF_KEYS, PRE_KEYS, atomic_json, compare_records, rng_state, restore_rng)
from c19_lif_v1_probe import capture
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import init_seeds
from ror_v1_training import install

BASE = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
ATOL, RTOL = 2e-5, 2e-4


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def backend():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(), cudnn_enabled=torch.backends.cudnn.enabled,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_version=torch.backends.cudnn.version(), deterministic=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                threads=torch.get_num_threads(), autocast_cuda=torch.is_autocast_enabled(),
                environment={k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
                                                           'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')})


@contextmanager
def isolated_backend(disable_tf32):
    before = backend()
    try:
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(before['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = before['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = before['cudnn_tf32']
        require(backend() == before, 'Diagnostic backend restoration failed')


def checked_fuse(model):
    """Check storage/state before the original in-place fuse touches only a deepcopy."""
    prior_backend = backend()
    require(all(not m.training for m in model.modules()), 'Fusion source must be entirely eval')
    require(all(v.dtype == torch.float32 for v in model.state_dict().values() if v.is_floating_point()), 'Expected FP32 model state')
    before = state_hash(model.state_dict())
    bn_state = {n: b for n, b in model.named_buffers() if n.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    fused = deepcopy(model)
    require(state_hash(fused.state_dict()) == before, 'Deepcopy weights/BN differ')
    original_pointers = {v.data_ptr() for v in model.state_dict().values() if v.numel()}
    require(not original_pointers.intersection(v.data_ptr() for v in fused.state_dict().values() if v.numel()), 'Fusion aliases source storage')
    fused.fuse(verbose=False)
    # RepConv creates fresh Conv2d children whose default .training is True.
    # Conv2d has no train/eval branch, but normalize all flags before comparing.
    created_train_flags = {n: type(m).__name__ for n, m in fused.named_modules() if m.training}
    require(all(t == 'Conv2d' for t in created_train_flags.values()), 'Fusion created a mode-sensitive training module')
    fused.eval()
    require(state_hash(model.state_dict()) == before, 'In-place fusion mutated comparison source')
    lif_a = next(m for m in model.modules() if type(m).__name__ == 'LIFDown')
    lif_b = next(m for m in fused.modules() if type(m).__name__ == 'LIFDown')
    require(hasattr(lif_b, 'bn') and state_hash(lif_a.state_dict()) == state_hash(lif_b.state_dict()), 'LIF fusion protection changed')
    counts = [sum(isinstance(m, torch.nn.BatchNorm2d) for m in x.modules()) for x in (model, fused)]
    require(counts[1] < counts[0], 'Ordinary BN not fused')
    require(all(not m.training for m in fused.modules()), 'Fused model not eval')
    require(backend() == prior_backend, 'Fusion changed backend configuration')
    return fused, dict(source_sha256=before, bn_sha256=state_hash(bn_state), copy_state_exact=True,
                       disjoint_storage=True, source_unchanged=True, all_eval=True, lif_bn_and_state_exact=True,
                       normalized_new_conv_flags=created_train_flags,
                       bn_counts=counts, fused_sha256=state_hash(fused.state_dict()), backend=backend())


def error_distribution(a, b):
    finite = torch.isfinite(a) & torch.isfinite(b)
    aa, bb = a[finite].double(), b[finite].double()
    delta = (aa - bb).abs()
    if not delta.numel(): return dict(finite_count=0)
    budget = delta / (ATOL + RTOL * bb.abs())
    return dict(finite_count=delta.numel(), nonfinite_count=int((~finite).sum()), mean_abs=float(delta.mean()),
                rms=float(delta.square().mean().sqrt()), abs_quantiles=dict(zip(('p50', 'p90', 'p99', 'p999', 'max'),
                    torch.quantile(delta, torch.tensor([.5, .9, .99, .999, 1.], dtype=torch.double)).tolist())),
                max_tolerance_ratio=float(budget.max()))


def mother_class():
    # Only tasks/loss/ROR differ in the production package. Execute the exact mother
    # tasks source; all shared module implementations must still equal that commit.
    from check_ror_v1 import parent_module
    changed = subprocess.check_output(['git', 'diff', BASE, '--name-only', '--', 'ultralytics-main'], cwd=ROOT, text=True).splitlines()
    allowed = {'ultralytics-main/ultralytics/nn/tasks.py', 'ultralytics-main/ultralytics/models/utils/loss.py',
               'ultralytics-main/ultralytics/models/utils/ror.py'}
    require(set(changed) <= allowed, 'Shared mother implementation differs: ' + repr(set(changed) - allowed))
    module = parent_module('ultralytics-main/ultralytics/nn/tasks.py', 'ultralytics.nn._ror_fusion_mother_tasks')
    raw = subprocess.check_output(['git', 'show', BASE + ':ultralytics-main/ultralytics/nn/tasks.py'], cwd=ROOT)
    return module.RTDETRDetectionModel, dict(commit=BASE, tasks_sha256=hashlib.sha256(raw).hexdigest(),
        shared_production_sources_match=True, changed_production_files=changed,
        scope='Exact mother tasks class and unchanged shared operators; training criterion is not called in eval capture')


def load_fixture(path, device):
    if str(path) == 'auto':
        candidates = []
        for report_path in (ROOT / 'outputs/ror_v1/checks').glob('fusion_*/fuse_diagnostic.json'):
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if (report.get('precision') == 'fp32' and report.get('device') == torch.device(device).type
                    and (report.get('failure') or {}).get('key') in LIF_KEYS + PRE_KEYS):
                if (report_path.parent / 'fixture.pt').is_file(): candidates.append(report_path)
        require(bool(candidates), 'No saved failure fixture; pass --fixture PATH or explicitly use --source for an initialization-only control')
        path = max(candidates, key=lambda p: p.stat().st_mtime_ns).parent / 'fixture.pt'
    path = Path(path).resolve()
    prior = json.loads((path.parent / 'fuse_diagnostic.json').read_text(encoding='utf-8'))
    require(prior.get('precision') == 'fp32' and prior.get('failure'), 'Expected an existing FP32 failure fixture')
    digest = sha256(path)
    require(digest == prior.get('fixture', {}).get('sha256'), 'Failure fixture hash does not match its original report')
    payload = torch_load(path, map_location='cpu')
    require(payload['precision'] == 'fp32' and payload['nc'] == 1, 'Fixture precision/class differs')
    image = payload['image']
    require(image.dtype == torch.float32 and list(image.shape) == [2, 3, 160, 160], 'Bounded ROR fixture must be B2/160 FP32')
    model = install(RTDETRDetectionModel(deepcopy(payload['yaml']), nc=1, verbose=False), dict(check='fusion-control'))
    model.load_state_dict(payload['unfused'], strict=True)
    # These non-state metadata values match the integration test at its fusion call.
    model.model[-1].num_denoising = 0
    model.criterion.set_epoch(20)
    return model, image, dict(kind='exact_failed_fixture', path=str(path), sha256=digest,
        original_failure=prior['failure'], original_runtime=prior['runtime'], original_precision_path=prior['precision_path'],
        saved_fused_sha256=state_hash(payload['fused'])), payload['rng']


def compare_mode(model, parent_type, image, state, device, report, persist):
    mother = parent_type(deepcopy(model.yaml), nc=1, verbose=False)
    mother.load_state_dict(model.state_dict(), strict=True)
    mother.model[-1].num_denoising = model.model[-1].num_denoising
    mother = mother.to(device).float().eval()
    require(state_hash(mother.state_dict()) == state_hash(model.state_dict()), 'Mother/ROR initial weights or BN differ')
    report.update(backend=backend(), input_sha256=state_hash({'image': image}), input_dtype=str(image.dtype),
                  input_shape=list(image.shape), autocast=False, same_rng=True, same_initial_state=True, models={})
    keys = LIF_KEYS + PRE_KEYS
    records = {}
    for name, candidate in (('mother', mother), ('ror', model)):
        entry = report['models'][name] = {}
        fused, entry['state_audit'] = checked_fuse(candidate)
        pair = []
        for item in (candidate, fused):
            restore_rng(state)
            with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
                pair.append(capture(item, image)[1])
        require(backend() == report['backend'], 'Backend changed between comparison forwards')
        require(state_hash(candidate.state_dict()) == entry['state_audit']['source_sha256'], 'Eval changed state')
        entry['pre_selection'] = compare_records(*pair, ATOL, RTOL, device + '.fp32.fuse_pre_selection',
                                                 torch.device(device).type, 'fp32', keys, collect=True)
        for key in keys:
            if pair[0][key].is_floating_point():
                entry['pre_selection'][key]['distribution'] = error_distribution(pair[0][key], pair[1][key])
        entry['failed_keys'] = [k for k, row in entry['pre_selection'].items() if row.get('status', '').startswith('FAILED')]
        entry['status'] = 'FAIL' if entry['failed_keys'] else 'PASS'
        records[name] = [{k: rec[k] for k in keys} for rec in pair]
        persist()
        del pair, fused
    report['ror_vs_mother_exact'] = {label: {k: torch.equal(records['mother'][i][k], records['ror'][i][k]) for k in keys}
                                     for i, label in enumerate(('unfused', 'fused'))}
    report['all_cross_exact'] = all(v for row in report['ror_vs_mother_exact'].values() for v in row.values())
    report['status'] = 'PASS' if all(v['status'] == 'PASS' for v in report['models'].values()) and report['all_cross_exact'] else 'REVIEW_REQUIRED'
    persist()


def diagnose(args):
    output = args.output or ROOT / 'outputs/ror_v1' / ('fusion_control_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', formal_training='NOT_RUN', gate_updated=False, scope='FP32 pre-selection only; not full candidate/replay fusion acceptance',
                  tolerances=dict(atol=ATOL, rtol=RTOL, changed=False), modes={}, output=str(output))
    def persist(): atomic_json(output / 'diagnostic.json', report)
    try:
        torch.set_num_threads(4)
        init_seeds(42, deterministic=True)  # same original preflight seed/backend initialization
        report.update(runtime=runtime(), backend_before=backend())
        report['diagnostic_source_sha256'] = {name: sha256(ROOT/'tools'/name) for name in
            ('ror_v1_fusion_diagnostic.py', 'check_ror_v1.py', 'c19_lif_v1_diagnostic.py', 'c19_lif_v1_probe.py')}
        device = args.device
        if torch.device(device).type == 'cuda':
            report['gpu_capability'] = list(torch.cuda.get_device_capability(device))
        if shutil.which('tmux'):
            result = subprocess.run(['tmux', 'has-session', '-t', '=ror-v1-training'], capture_output=True, text=True)
            report['tmux'] = dict(created_by_diagnostic=False, session_exists=result.returncode == 0, stderr=result.stderr.strip())
        else: report['tmux'] = dict(created_by_diagnostic=False, session_exists=None, reason='tmux unavailable')
        parent_type, report['mother_source'] = mother_class()
        if args.source:
            require(sha256(args.source) == SOURCE_SHA256, 'Wrong unified source')
            _, initial, _ = controlled_models(args.source)
            model, _ = build_training_model(initial.yaml, initial, dict(nc=1, channels=3))
            model = install(model, dict(check='fusion-control'))
            image = torch.rand(2, 3, 160, 160, device=device)
            report['fixture'] = dict(kind='controlled_initialization_only', source_sha256=sha256(args.source),
                                     limitation='Does not reproduce post-optimizer/BN state of the reported server failure')
            state = rng_state()
            del initial
        else: model, image, report['fixture'], state = load_fixture(args.fixture, device)
        model = model.to(device).float().eval(); image = image.to(device)
        persist()
        for name, disable in (('native', False), ('tf32_disabled_diagnostic_only', True)):
            entry = report['modes'][name] = {}
            try:
                with isolated_backend(disable): compare_mode(model, parent_type, image, state, device, entry, persist)
            except Exception as error:
                entry.update(status='ERROR', error=repr(error), traceback=traceback.format_exc())
            finally:
                report['backend_after'] = backend()
                report['backend_restored'] = backend() == report['backend_before']
                persist()
        native = report['modes']['native']; isolated = report['modes']['tf32_disabled_diagnostic_only']
        saved = report['fixture'].get('saved_fused_sha256')
        if saved and native.get('models', {}).get('ror', {}).get('state_audit'):
            report['saved_fused_state_reproduced'] = saved == native['models']['ror']['state_audit']['fused_sha256']
            failure = report['fixture']['original_failure']
            measured = native['models']['ror'].get('pre_selection', {}).get(failure.get('key'), {})
            report['original_failure_reproduced'] = all(measured.get(k) == failure.get(k) for k in
                ('status', 'shape', 'dtype_a', 'dtype_b', 'allclose_failed_count', 'max_abs_error'))
        if native.get('status') == 'PASS': report['finding'] = 'NATIVE_PRE_SELECTION_PASS; reported server failure not reproduced here'
        elif (native.get('all_cross_exact') and native['models']['mother']['status'] == 'FAIL' and isolated.get('status') == 'PASS'):
            report['finding'] = 'MOTHER_AND_ROR_SAME_NATIVE_ERROR_REMOVED_BY_TF32_ISOLATION; platform/backend effect, not ROR-specific'
        elif native.get('all_cross_exact'): report['finding'] = 'SHARED_MOTHER_ROR_ERROR; TF32 isolation did not establish a cause'
        else: report['finding'] = 'UNRESOLVED; inspect source/state/cross-comparison or diagnostic errors'
        del model, image
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # A distinct bounded test, using the original initialization and original
        # backend, must not inherit the fusion fixture or TF32 isolation settings.
        if args.check_update:
            require(args.update_source and args.update_source.is_file(), '--check-update requires --update-source')
            require(sha256(args.update_source) == SOURCE_SHA256, 'Wrong update source')
            from check_ror_v1 import integration
            init_seeds(42, deterministic=True)
            report['cuda_direct_update'] = {}
            try: integration(args.update_source, device, report['cuda_direct_update'], persist,
                             stop_after_update=True, evidence_dir=output/'update')
            except Exception as error:
                report['cuda_direct_update'].update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            finally: persist()
        else: report['cuda_direct_update'] = dict(status='NOT_RUN', reason='Separate from fusion; previous REVIEW_REQUIRED is not cleared')
        report['status'] = 'PASS' if native.get('status') == 'PASS' and report['backend_restored'] else 'REVIEW_REQUIRED'
        if report['cuda_direct_update']['status'] in ('FAIL', 'REVIEW_REQUIRED'): report['status'] = 'REVIEW_REQUIRED'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
        raise
    finally: persist()
    print(json.dumps({k: report[k] for k in ('status', 'finding', 'output', 'formal_training', 'gate_updated')}, ensure_ascii=False))
    print('CUDA direct update: ' + report['cuda_direct_update']['status'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--fixture', type=Path, default=Path('auto'), help='Existing failure fixture, auto selects newest matching-device FP32 failure')
    source.add_argument('--source', type=Path, help='Explicit alternative: controlled initialization, not exact failing state')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, help='New output directory; never overwrite earlier evidence')
    parser.add_argument('--check-update', action='store_true', help='Also rerun the independent bounded zero-weight optimizer check')
    parser.add_argument('--update-source', type=Path)
    args = parser.parse_args()
    report = diagnose(args)
    raise SystemExit(0 if report['status'] == 'PASS' else 2)


if __name__ == '__main__': main()
