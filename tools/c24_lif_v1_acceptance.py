"""One fail-closed verifier shared by numerical checks and all launch gates."""
from c24_lif_v1_common import require

ACCEPTED = ('PASSED', 'ACCEPTED_WITH_BASELINE_CUTOFF_WARNING')


def numerical_status(case):
    from c24_lif_v1_numerics import SCHEMA, CONTINUOUS, SELECTED, TOLERANCES, cutoff_acceptance
    try:
        require(case.get('schema') == SCHEMA and case.get('mode') in TOLERANCES, 'Numerical schema/mode missing')
        require(case.get('status') in (*ACCEPTED, 'BLOCKED', 'REQUIRES_REVIEW'), 'Unknown numerical status')
        require(case.get('status') != 'BLOCKED', 'Explicit numerical failure')
        require(case.get('operator_status') == 'PASSED', 'Numerical operator failure')
        relation = case.get('natural_relation')
        require(relation in ('IDENTICAL', 'PERMUTATION', 'SET_DRIFT'), 'Illegal candidate relation')
        def rows(field, keys):
            values = case.get(field, [])
            require([v.get('key') for v in values] == list(keys), 'Missing/duplicate numerical key')
            for value in values:
                require(value.get('status') == 'PASSED' and (value.get('finite') is True or value.get('exact') is True), 'Invalid numerical evidence')
                if value.get('finite') is True:
                    require((value.get('atol'), value.get('rtol')) == TOLERANCES[case['mode']] and value.get('over_tolerance') == 0,
                            'Changed tolerance/over-tolerance result')
        rows('continuous', CONTINUOUS)
        enc = next(r for r in case['continuous'] if r['key'] == 'encoder_logits')
        size = enc['shape_a'][1]
        a, b = case.get('candidate_ids_a'), case.get('candidate_ids_b')
        require(isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == enc['shape_a'][0] > 0, 'Missing candidate batches')
        require(all(len(ids) == len(set(ids)) == 300 and all(type(i) is int and 0 <= i < size for i in ids)
                    for batches in (a, b) for ids in batches), 'Illegal candidate IDs')
        actual = 'IDENTICAL' if a == b else 'PERMUTATION' if all(set(x) == set(y) for x, y in zip(a, b)) else 'SET_DRIFT'
        require(actual == relation, 'Mislabelled candidate relation')
        if relation != 'SET_DRIFT':
            rows('id_aligned', SELECTED)
            return 'PASSED'
        for side, ids in [('A', a), ('B', b)]:
            replay = case.get('replay', {}).get(side, {})
            require(replay.get('candidate_ids_a') == replay.get('candidate_ids_b') == ids and
                    replay.get('natural_relation') == 'IDENTICAL' and numerical_status(replay) == 'PASSED', 'Missing/invalid two-sided replay')
        parent = case.get('parent_control')
        if parent and parent.get('comparability') == 'VERIFIED':
            # A claimed parent warning must carry the same full numerical evidence.
            require(parent.get('natural_relation') == 'SET_DRIFT' and parent.get('parent_control') is None, 'Invalid parent proof')
            parent_copy = dict(parent, comparability='DIAGNOSTIC_ONLY')
            require(numerical_status(parent_copy) == 'REQUIRES_REVIEW', 'Invalid parent numerical evidence')
        return cutoff_acceptance(case, parent)
    except (RuntimeError, KeyError, TypeError, ValueError, IndexError):
        return 'BLOCKED'


def fusion_status(result):
    if not (result.get('lif_bn_state_exact') and result.get('repeat_fuse') == 'EXACT' and
            result.get('physical_negatives') == 'BLOCKED_AS_EXPECTED'):
        return 'BLOCKED'
    if [c.get('input') for c in result.get('cases', [])] != [[1, 3, 640, 640], [1, 3, 160, 192]]:
        return 'BLOCKED'
    statuses = [numerical_status(c) for c in result['cases'] + [result.get('save_load', {})]]
    return aggregate(statuses)


def aggregate(statuses):
    if not statuses or any(s not in (*ACCEPTED, 'REQUIRES_REVIEW') for s in statuses):
        return 'BLOCKED'
    if 'REQUIRES_REVIEW' in statuses:
        return 'REQUIRES_REVIEW'
    return 'ACCEPTED_WITH_BASELINE_CUTOFF_WARNING' if 'ACCEPTED_WITH_BASELINE_CUTOFF_WARNING' in statuses else 'PASSED'


def stage_status(name, result):
    if name.startswith('fusion_'):
        return fusion_status(result)
    if isinstance(result, dict):
        status = result.get('status', 'PASSED')
        return status if status in (*ACCEPTED, 'BLOCKED', 'REQUIRES_REVIEW', 'NOT_RUN') else 'BLOCKED'
    return 'PASSED' if result else 'BLOCKED'


def required_stages(server):
    names = ['negative_gate', 'lif_original_unit', 'initialization', 'structure', 'history']
    for device in ('cpu', 'cuda'):
        names += ['degeneration_' + device]
        names += ['fusion_' + device + '_' + m for m in (('fp32',) if device == 'cpu' else ('fp32', 'amp', 'half'))]
        if device == 'cuda':
            names += ['original_parent_cutoff_cuda_amp', 'original_parent_cutoff_cuda_half']
        names += ['native_initialization_small_' + device, 'native_loss_' + device, 'ingress_' + device]
    if server:
        names += ['real_capacity_batch', 'native_initialization_B16_640', 'nonzero_branch_stress_B16_640']
    return names


def blocking_summary(report):
    rows = report.get('stages', [])
    unresolved = [dict(name=r['name'], status=r['status'], error=r.get('error') or r.get('result', {}).get('error')
                       if isinstance(r.get('result', {}), dict) else r.get('error'))
                  for r in rows if r['status'] not in ACCEPTED]
    present = {r['name'] for r in rows}
    unresolved += [dict(name=n, status='NOT_RUN', error='Required stage missing')
                   for n in required_stages(report.get('scope') == 'server') if n not in present]
    states = {r['name']: r['status'] for r in rows}
    return dict(first_unaccepted_stage=unresolved[0]['name'] if unresolved else None,
                terminal_exception_stage=report.get('terminal_exception_stage'), unresolved_stages=unresolved,
                capacity_native_init=states.get('native_initialization_B16_640', 'NOT_RUN'),
                capacity_nonzero_stress=states.get('nonzero_branch_stress_B16_640', 'NOT_RUN'),
                amp_calibration=states.get('native_initialization_B16_640', 'NOT_RUN'),
                training_dispatched=False, scope=report.get('scope'))


def require_preflight(report):
    from c24_lif_v1_numerics import SCHEMA
    require(report.get('schema') == SCHEMA and report.get('scope') == 'server', 'Preflight schema/server scope missing')
    rows = report.get('stages', [])
    require([s.get('name') for s in rows] == required_stages(True), 'Incomplete/duplicate preflight stage schema')
    statuses = []
    def loss_result(result, native=False, capacity=False, warmed=False, amp=False):
        require(result.get('status') == 'PASSED' and result.get('source_model_unchanged') is True, 'Loss diagnostic unaccepted/polluted source')
        require(result.get('coverage') == 336 and len(result.get('new_tensors', [])) == 10, 'Incomplete optimizer coverage')
        require(result.get('initialization') == ('native_zero_initialization' if native else 'nonzero_branch_stress') and
                result.get('amp_start') == ('FP32_WARMED' if warmed else 'COLD'), 'Initialization/AMP identity mismatch')
        projections = result.get('initial_model', {}).get('output_projections', {})
        require(len(projections) == 2 and all(v.get('zero') is native for v in projections.values()), 'Wrong projection initialization')
        c = result.get('calibration', {})
        require(c.get('amp') is amp, 'Wrong FP32/AMP diagnostic identity')
        require(c.get('input', {}).get('shape') == result.get('input'), 'Input identity changed inside calibration')
        if not warmed:
            require(c.get('initial_model') == result.get('initial_model'), 'Cold initialization state changed before calibration')
        require(c.get('status') == 'PASSED' and c.get('consecutive_updates', 0) >= 2 and c.get('actual_optimizer_steps', 0) >= 2 and
                c.get('save_load_model_optimizer_scaler') == 'EXACT', 'Missing actual optimizer/scaler acceptance')
        require(c.get('coverage', {}).get('valid') is True and 2 <= c.get('budget', 0) <= 12, 'Invalid bounded AMP contract')
        attempts = c.get('attempts', [])
        require(2 <= len(attempts) <= c['budget'] and all(v.get('status') in ('UPDATED', 'OVERFLOW_SKIPPED') and
                v.get('loss_finite') is True and v.get('actual_step_delta') == (1 if v['status'] == 'UPDATED' else 0) for v in attempts),
                'Malformed/empty update evidence')
        require(all(v['status'] == 'UPDATED' for v in attempts[-2:]) and
                sum(v['actual_step_delta'] for v in attempts) == c['actual_optimizer_steps'], 'Unproven consecutive actual steps')
        if amp:
            require(c.get('initial_scale') == (128. if warmed else 65536.), 'Cold AMP must use native default scaler')
        if capacity:
            require(result.get('input') == [16, 3, 640, 640] and c.get('amp') is True and c.get('peak_allocated_bytes', 0) > 0 and
                    c.get('peak_reserved_bytes', 0) > 0, 'B16/640 CUDA capacity missing')
        if warmed:
            require(result.get('fp32_warmup', {}).get('status') == 'PASSED', 'Missing explicitly recorded warmup')
        if 'updated_nonzero_bbox_fusion' in result:
            require(numerical_status(result['updated_nonzero_bbox_fusion']) in ACCEPTED, 'Updated bbox fusion unaccepted')
    for stage in rows:
        result = stage.get('result')
        require(result, 'Empty stage ' + stage['name'])
        status = stage_status(stage['name'], result)
        require(status == stage.get('status') and status in ACCEPTED, 'Unaccepted stage ' + stage['name'])
        statuses.append(status)
        name = stage['name']
        if name.startswith('native_initialization_'):
            loss_result(result, native=True, capacity=name.endswith('B16_640'), amp=not name.endswith('cpu'))
        elif name == 'nonzero_branch_stress_B16_640':
            loss_result(result, capacity=True, amp=True)
        elif name.startswith('native_loss_'):
            expected = ['nonzero_branch_stress_fp32']
            if name.endswith('cuda'):
                expected += ['nonzero_branch_stress_cold_amp', 'nonzero_branch_stress_fp32_warmed_amp']
            require(list(result.get('checks', {})) == expected, 'Missing cold/warmed stress distinction')
            for label, check in result['checks'].items():
                if name.endswith('cuda') and label == 'nonzero_branch_stress_fp32':
                    require('updated_nonzero_bbox_fusion' in check, 'Updated bbox regression missing')
                loss_result(check, warmed=label.endswith('fp32_warmed_amp'), amp=label.endswith('amp'))
    require(report.get('status') == aggregate(statuses) and report['status'] in ACCEPTED, 'Preflight aggregate not accepted')
    require(not blocking_summary(report)['unresolved_stages'], 'Outstanding preflight blockers')
    require(report.get('server_B16_640', {}).get('status') == 'PASSED', 'No B16/640 result')
    capacities = [r['result']['calibration']['input'] for r in rows if r['name'] in
                  ('native_initialization_B16_640', 'nonzero_branch_stress_B16_640')]
    require(len(capacities) == 2 and capacities[0] == capacities[1] and len(capacities[0].get('images', [])) == 16,
            'Native/stress capacity did not use the same identified real augmented B16 batch')
    return True
