"""Finite negative contracts and native CUDA GradScaler tests; no dataset evaluation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from c24_lif_v1_common import ROOT, require, read_json, write_json
from c24_lif_v1_amp import calibration, Evidence, model_evidence
from c24_lif_v1_acceptance import numerical_status, fusion_status, stage_status, aggregate, blocking_summary


def lifecycle_checks(output):
    import train_c24_lif_v1 as lifecycle
    import c24_lif_v1_pack as packaging
    from c24_lif_v1_pack import verify_archive, LIMIT
    results = {}
    for fail,check_only in ((True,False),(False,False),(False,True)):
        with TemporaryDirectory(dir=output, prefix='lifecycle_') as temp:
            folder = Path(temp)
            paths = dict(run=folder/'formal_run', launch=folder/'launch', download=folder/'download')
            calls = dict(preflight=0, dispatch=0, release=[])
            def preflight():
                calls['preflight'] += 1
                if fail:
                    write_json(paths['launch']/'preflight/preflight.json', dict(scope='server',
                        terminal_exception_stage='native_initialization_B16_640', stages=[
                        dict(name='fusion_cuda_amp', status='REQUIRES_REVIEW'),
                        dict(name='fusion_cuda_half', status='REQUIRES_REVIEW'),
                        dict(name='native_initialization_B16_640', status='BLOCKED', error='fixture nonfinite')]))
                    raise RuntimeError('Preflight fixture blocked')
                return dict(fixture=True,preflight_status='PASSED')
            original_run = lifecycle.subprocess.run
            def dispatch(*args, **kwargs):
                if args[0][0] != 'tmux':
                    return original_run(*args, **kwargs)
                calls['dispatch'] += 1
                return SimpleNamespace(returncode=0)
            with patch.object(lifecycle, 'os', SimpleNamespace(name='posix', getpid=os.getpid)), \
                 patch.object(lifecycle, 'verify_delivery', lambda: None), \
                 patch.object(lifecycle, 'paths', lambda: paths), patch.object(packaging, 'paths', lambda: paths), \
                 patch.object(lifecycle, 'tmux_active', lambda: False), patch.object(lifecycle, 'live', lambda owner: False), \
                 patch.object(lifecycle, 'acquire', lambda kind: dict(token='fixture', pid=os.getpid())), \
                 patch.object(lifecycle, 'release', lambda kind, info: calls['release'].append(kind)), \
                 patch.object(lifecycle, 'preflight_once', preflight), patch.object(lifecycle.shutil, 'which', lambda name: name), \
                 patch.object(lifecycle.subprocess, 'run', dispatch):
                # Only tmux is intercepted. Packaging still reads actual Git identity.
                try:
                    lifecycle.start_direct(check_only=check_only)
                    require(not fail, 'Failure swallowed by start-direct')
                except RuntimeError as error:
                    require(fail and 'fixture blocked' in str(error), 'Original preflight exception replaced')
            require(calls['preflight'] == 1 and calls['dispatch'] == (0 if fail or check_only else 1) and
                    calls['release'].count('preflight') == 1, 'Duplicate preflight/dispatch or missing lock release')
            row = dict(status='PASSED', **calls, actual_training='NOT_RUN_MOCKED_DISPATCH')
            if fail:
                meta = read_json(paths['launch']/'failure_package.json')
                require(meta is not None, 'Automatic failure LIGHT missing: ' + str(read_json(paths['launch']/'failure_package_error.json')))
                archive = Path(meta['path'])
                manifest = verify_archive(archive)
                require(archive.stat().st_size <= LIMIT and 'metadata/preflight/blocking_summary.json' in manifest, 'Failure LIGHT limit/evidence missing')
                import tarfile
                with tarfile.open(archive) as t:
                    blockers = json.load(t.extractfile('metadata/preflight/blocking_summary.json'))
                require(blockers['first_unaccepted_stage'] == 'fusion_cuda_amp' and blockers['capacity_native_init'] == 'BLOCKED' and
                        blockers['terminal_exception_stage'] == 'native_initialization_B16_640', 'Failure LIGHT omitted independent blockers')
                row.update(archive_bytes=archive.stat().st_size, all_blockers=True, no_worker=True)
            results['failed' if fail else 'diagnosis_only' if check_only else 'accepted'] = row
    return results


def model_contracts(source, output):
    from init_c24_lif_v1 import controlled_models, native_rebuild
    from c24_lif_v1_loss import diagnostic_loss
    from c24_lif_v1_numerics import capture
    from c24_lif_v1_amp import rtdetr_forward
    from check_c24_lif_v1 import diagnostic_settings, synthetic_batch
    output = Path(output)
    with diagnostic_settings():
        parents, _ = controlled_models(source)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = native_rebuild(parents['combo'], 'combo')
        del parents
        with patch('c24_lif_v1_numerics.activate', side_effect=AssertionError('native initialization called activate')):
            result = diagnostic_loss(model, 'cpu', output, label='zero_activate_forbidden')
        require(result['status'] == 'PASSED' and result['source_model_unchanged'], 'True zero initialization/state isolation failed')
        attempt = read_json(output/'zero_activate_forbidden/calibration/attempt_00.json')
        require(any(v == 0 for v in attempt['branch_gradients']['new_tensors'].values()), 'Zero upstream gradient contract unexercised')
        require(attempt['native_loss_terms_finite'] and len(attempt['native_loss_terms']) > 3, 'Auxiliary/DN loss terms missing')
        model.eval();head=model.model[-1]
        hooks = {id(m): (len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()}
        prior = head.__dict__.get('_get_decoder_input');native_topk=torch.topk
        with patch.object(head.enc_output, 'forward', side_effect=RuntimeError('injected encoder error')):
            try:
                capture(model, torch.rand(1,3,160,192))
                raise AssertionError('Capture exception not exercised')
            except RuntimeError as error:
                require('injected encoder error' in str(error), 'Unexpected capture failure')
        require(head.__dict__.get('_get_decoder_input') is prior and torch.topk is native_topk and
                hooks == {id(m):(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()}, 'Capture hook/top-k wrapper leaked')
        model.train();model.nc=1;model.criterion=model.init_criterion()
        count=len(model.criterion._forward_hooks)
        with patch.object(model.criterion,'forward',side_effect=RuntimeError('injected loss error')):
            try:
                rtdetr_forward(model,synthetic_batch(),lambda **values:None)
                raise AssertionError('Loss exception not exercised')
            except RuntimeError as error:
                require('injected loss error' in str(error),'Unexpected loss failure')
        require(len(model.criterion._forward_hooks)==count,'Loss observation hook leaked')
    report=dict(status='PASSED',zero_activate_forbidden=True,source_unchanged=True,zero_upstream_gradients_allowed=True,
                auxiliary_DN_loss_terms_recorded=True,capture_hooks_and_topk_restored=True,loss_hook_restored=True)
    write_json(output/'model_contracts.json',report)
    return report


def server_gate_contracts(output):
    from c24_lif_v1_acceptance import require_preflight, required_stages
    from c24_lif_v1_numerics import TOLERANCES
    original=read_json(ROOT/'docs/c24_lif_v1/preflight_fix/local_preflight.json')
    fixture=deepcopy(original)
    fixture.update(scope='server',status='PASSED',server_B16_640=dict(status='PASSED'))
    rows={r['name']:r for r in fixture['stages']}
    fp32=rows['fusion_cuda_fp32']['result']
    # Deliberately synthetic evidence tests only the reader/launch contract.
    # No claim that these converted precision or B16 records were measured.
    def precision(value,mode):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='mode':value[k]=mode
                elif k=='atol':value[k]=TOLERANCES[mode][0]
                elif k=='rtol':value[k]=TOLERANCES[mode][1]
                else:precision(v,mode)
        elif isinstance(value,list):
            for v in value:precision(v,mode)
    for mode in ('amp','half'):
        result=deepcopy(fp32);precision(result,mode)
        rows['fusion_cuda_'+mode].update(status='PASSED',result=result)
    image_identity=['fixture/image_%02d.jpg'%i for i in range(16)]
    native=deepcopy(rows['native_initialization_small_cuda']['result'])
    stress=deepcopy(rows['native_loss_cuda']['result']['checks']['nonzero_branch_stress_cold_amp'])
    for label,result in [('native_initialization_B16_640',native),('nonzero_branch_stress_B16_640',stress)]:
        result['input']=[16,3,640,640]
        result['calibration']['input'].update(shape=[16,3,640,640],images=image_identity)
        rows[label]=dict(name=label,status='PASSED',result=result)
    rows['real_capacity_batch']=dict(name='real_capacity_batch',status='PASSED',result=dict(status='PASSED',fixture=True))
    fixture['stages']=[rows[n] for n in required_stages(True)]
    require(require_preflight(fixture),'Complete synthetic server contract rejected')
    cases={}
    def capacity(r):return next(s['result'] for s in r['stages'] if s['name']=='native_initialization_B16_640')
    mutations=[('missing_stage',lambda r:r['stages'].pop()),
        ('zero_actual_updates',lambda r:capacity(r)['calibration'].update(actual_optimizer_steps=0)),
        ('FP32_mislabelled_as_AMP',lambda r:capacity(r)['calibration'].update(amp=False)),
        ('manual_cold_scale_128',lambda r:capacity(r)['calibration'].update(initial_scale=128.)),
        ('different_batch_identity',lambda r:capacity(r)['calibration']['input'].update(augmented_tensor_sha256='wrong')),
        ('lost_scaler_reload',lambda r:capacity(r)['calibration'].pop('save_load_model_optimizer_scaler')),
        ('activate_native',lambda r:next(iter(capacity(r)['initial_model']['output_projections'].values())).update(zero=False)),
        ('cold_state_leak',lambda r:capacity(r)['calibration']['initial_model']['parameters'].update(sha256='wrong'))]
    for label,mutate in mutations:
        bad=deepcopy(fixture);mutate(bad)
        try:
            require_preflight(bad)
            raise AssertionError('Accepted '+label)
        except RuntimeError:
            cases[label]='BLOCKED_AS_EXPECTED'
    try:
        require_preflight(original)
        raise AssertionError('Actual local review evidence accepted as server capacity')
    except RuntimeError:
        cases['actual_local_not_server']='BLOCKED_AS_EXPECTED'
    report=dict(status='PASSED',scope='SYNTHETIC_READER_CONTRACT_ONLY_NOT_SERVER_CAPACITY',cases=cases)
    write_json(Path(output)/'server_gate_contracts.json',report)
    return report


def checks(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available(), 'Native CUDA tests require CUDA')
    torch.set_num_threads(4)
    results = {}
    def fixture():
        model = torch.nn.Linear(1, 1, bias=False).cuda()
        with torch.no_grad():
            model.weight.fill_(.25)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        return model, opt, dict(img=torch.ones(2, 1, device='cuda'))
    def forward(model, batch, observe):
        pred = model(batch['img'])
        observe(predictions_finite=bool(torch.isfinite(pred).all()))
        return pred.float().square().mean() * 4
    def run(name, change=None, fn=forward, **kwargs):
        model, opt, batch = fixture()
        if change:
            change(model, opt, batch)
        prior = opt.__dict__.get('step')
        report = calibration(model, opt, batch, fn, output / name, label=name,
            optimizer_factory=lambda m: torch.optim.AdamW(m.parameters(), lr=.001), **kwargs)
        require(opt.__dict__.get('step') is prior, 'Optimizer step wrapper leaked')
        require(all(p.grad is None for p in model.parameters()), 'Gradients leaked from diagnostic')
        results[name] = report
        return report
    r = run('native_overflow_then_update')
    require(r['status'] == 'PASSED' and r['initial_scale'] == 65536 and r['actual_optimizer_steps'] == 2 and
            any(a['status'] == 'OVERFLOW_SKIPPED' for a in r['attempts']) and
            r['save_load_model_optimizer_scaler'] == 'EXACT', 'Native overflow/backoff/update contract failed')
    for value in ('nan', 'inf'):
        def nonfinite(model, batch, observe):
            observe(predictions_finite=True)
            return model(batch['img']).float().sum() * float(value)
        r = run('forward_' + value, fn=nonfinite)
        require(r['status'] == 'BLOCKED' and r['actual_optimizer_steps'] == 0 and
                read_json(output / ('forward_' + value) / 'attempt_00.json')['loss_finite'] is False,
                'Nonfinite forward accepted or evidence lost')
    def infinite_gradient(m, o, b):
        m.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
    r = run('persistent_overflow', change=infinite_gradient, budget=3)
    require(r['status'] == 'BLOCKED' and r['actual_optimizer_steps'] == 0 and len(r['attempts']) == 3 and
            all(a['status'] == 'OVERFLOW_SKIPPED' for a in r['attempts']), 'Persistent overflow not capped')
    for name, change in [('missing_parameter', lambda m, o, b: o.param_groups[0]['params'].clear()),
                         ('duplicate_parameter', lambda m, o, b: o.param_groups[0]['params'].append(m.weight))]:
        r = run(name, change=change)
        require(r['status'] == 'BLOCKED' and r['actual_optimizer_steps'] == 0, 'Bad optimizer coverage accepted')
    for target in ('parameter', 'optimizer'):
        def pollution(m, o, b):
            old = o.step
            def corrupt(*args, **kwargs):
                result = old(*args, **kwargs)
                with torch.no_grad():
                    if target == 'parameter':
                        m.weight.fill_(float('nan'))
                    else:
                        o.state[m.weight]['exp_avg'].fill_(float('inf'))
                return result
            o.step = corrupt
        r = run('post_update_' + target + '_pollution', change=pollution, amp=False, budget=2)
        require(r['status'] == 'BLOCKED' and r['actual_optimizer_steps'] == 1, 'Polluted update accepted')
    def prediction_nan(m, b, observe):
        observe(predictions_finite=False)
        return m(b['img']).float().sum()
    require(run('prediction_nonfinite', fn=prediction_nan)['status'] == 'BLOCKED', 'Prediction nonfinite accepted')

    from check_c24_lif_v1 import diagnostic_settings
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all(),
              torch.get_default_dtype(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled(), torch.is_grad_enabled())
    try:
        with diagnostic_settings():
            random.random();np.random.rand();torch.rand(4);torch.rand(4, device='cuda')
            torch.set_default_dtype(torch.float64)
            raise RuntimeError('injected exception')
    except RuntimeError:
        pass
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all(),
             torch.get_default_dtype(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(), torch.is_grad_enabled())
    require(before[0] == after[0] and all(np.array_equal(a, b) for a, b in zip(before[1], after[1])) and
            torch.equal(before[2], after[2]) and all(torch.equal(a, b) for a, b in zip(before[3], after[3])) and
            before[4:] == after[4:], 'RNG/backend/dtype/grad context leaked')
    results['exception_context_restore'] = dict(status='PASSED')

    from c24_lif_v1_numerics import negative_checks, SCHEMA
    results['numerical_negatives'] = negative_checks()
    # Load real full prior local reports, update only their evidence schema. No
    # values/tolerances/IDs are replaced. Exercise the new launch verifier too.
    def schema(value):
        if isinstance(value, dict):
            return {k: SCHEMA if k == 'schema' else schema(v) for k, v in value.items()}
        if isinstance(value, list):
            return [schema(v) for v in value]
        return value
    original = schema(read_json(ROOT / 'docs/c24_lif_v1/checks/fusion_cuda_fp32.json')['result'])
    require(fusion_status(original) == 'PASSED', 'Genuine FP32 same-ID evidence rejected')
    case = original['cases'][0]
    for label, mutate in [('continuous_error', lambda c: c['continuous'][0].update(status='BLOCKED', over_tolerance=1)),
                          ('missing_candidate', lambda c: c.pop('candidate_ids_a')),
                          ('illegal_candidate', lambda c: c['candidate_ids_a'][0].__setitem__(0, 99999)),
                          ('changed_tolerance', lambda c: c['continuous'][0].update(atol=1)),
                          ('unknown_status', lambda c: c.update(status='OK'))]:
        changed = deepcopy(case);mutate(changed)
        require(numerical_status(changed) == 'BLOCKED', 'Accepted ' + label)
        results[label] = dict(status='BLOCKED_AS_EXPECTED')
    for mode in ('amp', 'half'):
        drift = schema(read_json(ROOT / ('docs/c24_lif_v1/checks/fusion_cuda_' + mode + '.json'))['result'])
        require(fusion_status(drift) == stage_status('fusion_cuda_' + mode, drift) == 'REQUIRES_REVIEW', 'Drift acceptance weakened')
        results['unexplained_' + mode + '_drift'] = dict(status='REQUIRES_REVIEW_AS_EXPECTED')
    # Synthetic exact parent proof exercises the EXISTING half-only exception,
    # never promoted to evidence for an actual model or server result.
    warning = deepcopy(drift)
    for c in warning['cases']:
        if c['natural_relation'] == 'SET_DRIFT':
            c['comparison_fingerprints'] = dict(fixture='exact synthetic proof only')
            parent = deepcopy(c)
            parent.update(comparability='VERIFIED', parent_control=None)
            c['parent_control'] = parent
    expected_warning = 'ACCEPTED_WITH_BASELINE_CUTOFF_WARNING'
    require(fusion_status(warning) == stage_status('fusion_cuda_half', warning) == aggregate([expected_warning, 'PASSED']) == expected_warning,
            'Legal warning semantics disagree between gates')
    warning['cases'][0]['mode'] = 'amp'
    require(fusion_status(warning) not in ('PASSED', expected_warning), 'Half exception incorrectly extended to AMP')
    results['shared_warning_semantics'] = dict(status='PASSED', evidence='SYNTHETIC_TEST_ONLY')
    combined = blocking_summary(dict(scope='server', terminal_exception_stage='native_initialization_B16_640', stages=[
        dict(name='fusion_cuda_amp', status='REQUIRES_REVIEW'), dict(name='fusion_cuda_half', status='REQUIRES_REVIEW'),
        dict(name='native_initialization_B16_640', status='BLOCKED', result=dict(error='injected'))]))
    require(combined['first_unaccepted_stage'] == 'fusion_cuda_amp' and combined['capacity_native_init'] == 'BLOCKED' and
            combined['training_dispatched'] is False and len(combined['unresolved_stages']) >= 3, 'Earlier independent blocker lost')
    write_json(output / 'blocking_summary.json', combined)
    results['all_blockers_preserved'] = dict(status='PASSED')
    # Sharding must retain every bad gradient, including the final name.
    data = [dict(name='parameter_' + str(i), inf=1, nan=0, finite_abs_max=2.) for i in range(2000)]
    writer = Evidence(output / 'shard_test');writer.write('many_gradients', data)
    writer.write('single_large_item', [dict(gradients=data)])
    require(all(p.stat().st_size <= 64000 for p in (output / 'shard_test').glob('*.json')), 'Evidence record limit exceeded')
    def resolve_value(value, lookup):
        if isinstance(value, dict) and 'structured_shard' in value:
            return resolve_value(lookup(value['structured_shard']['path']), lookup)
        if isinstance(value, dict) and 'structured_shards' in value:
            return [row for part in value['structured_shards'] for row in resolve_value(lookup(part['path']), lookup)]
        if isinstance(value, dict):
            return {k: resolve_value(v, lookup) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_value(v, lookup) for v in value]
        return value
    lookup=lambda name:read_json(output/'shard_test'/name)
    require(resolve_value(lookup('many_gradients.json'),lookup)==data and
            resolve_value(lookup('single_large_item.json'),lookup)==[dict(gradients=data)], 'Gradient shards lost names')
    from c24_lif_v1_pack import shard_json_entries, encode
    entries={'metadata/single.json':encode([dict(gradients=data)])}
    shard_json_entries(entries)
    require(all(len(v)<=64000 for v in entries.values()) and
            resolve_value(json.loads(entries['metadata/single.json']),lambda name:json.loads(entries[name]))==[dict(gradients=data)],
            'LIGHT single-item structured shards lost evidence')
    results['all_gradient_names_bounded'] = dict(status='PASSED')
    results['single_preflight_and_failure_light'] = lifecycle_checks(output)
    results['server_gate_contracts'] = server_gate_contracts(output)
    write_json(output / 'regression.json', results)
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/c24_lif_v1_preflight_fix/contracts')
    p.add_argument('--source', type=Path)
    args = p.parse_args()
    print(json.dumps({k: v.get('status', 'PASSED') for k, v in checks(args.output).items()}, indent=2))
    if args.source:
        print(json.dumps(model_contracts(args.source,args.output),indent=2))
