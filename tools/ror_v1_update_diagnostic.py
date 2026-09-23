"""ROR zero-weight CUDA update diagnosis: six reset trials, no training gate changes."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
import torch
from init_c19_lif_v1 import ROOT, SOURCE_SHA256, controlled_models, build_training_model, require, runtime, sha256
from check_ror_v1 import BASE, parent_module, targets_of
from c19_lif_v1_diagnostic import atomic_json, rng_state, restore_rng
from ror_v1_fusion_diagnostic import backend, mother_class
from ror_v1_training import install
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import init_seeds

REPEATS = 3
SAMPLES = 3


def digest_tree(value):
    """Content identity including nonpersistent buffers, RNG arrays and optimizer state."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str(('tensor', str(x.dtype), tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str(('array', str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for k in sorted(x, key=str): visit(k); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x: visit(item)
        else: h.update(repr((type(x).__name__, x)).encode())
    visit(value)
    return h.hexdigest()


def cpu_copy(state):
    return {n: v.detach().cpu().clone() for n, v in state.items()}


def group_spec(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [{**{k: v for k, v in group.items() if k != 'params'},
             'params': [names[id(p)] for p in group['params']]} for group in optimizer.param_groups]


def graph_signature(loss, names):
    """Ordered connected autograd graph; disconnected isfinite checks are excluded."""
    root = loss.grad_fn
    require(root is not None, 'Loss has no gradient graph')
    nodes = [root]; indices = {root: 0}; rows = []; counts = Counter()
    for node in nodes:
        kind = type(node).__name__; counts[kind] += 1
        edges = []
        for child, slot in node.next_functions:
            if child is not None and child not in indices:
                indices[child] = len(nodes); nodes.append(child)
            edges.append((None if child is None else indices[child], slot))
        variable = getattr(node, 'variable', None)
        leaf = None if variable is None else names.get(id(variable), ('unnamed', list(variable.shape)))
        rows.append((kind, leaf, edges))
    return dict(sha256=digest_tree(rows), nodes=len(rows), operations=dict(sorted(counts.items())))


def read_prior(path):
    path = Path(path)
    info = dict(path=str(path.resolve()), sha256=sha256(path), thresholds='Frozen original per-state bounds; never recalibrated')
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith('/cuda_step_variation.json')]
            require(len(members) == 1, 'Prior archive must contain exactly one step-variation report')
            raw = archive.read(members[0]); info.update(member=members[0], member_sha256=hashlib.sha256(raw).hexdigest())
            rows = json.loads(raw)
    else: rows = json.loads(path.read_text(encoding='utf-8'))
    require(bool(rows) and all(v['bound'] > 0 for v in rows.values()), 'Invalid prior bounds')
    info['failed_parameters'] = [n for n, v in rows.items() if v['ror_vs_parent_max_abs'] > v['bound']]
    return rows, info


def restore_trial(model, optimizer, snapshot):
    model.load_state_dict(snapshot['state'], strict=True)
    buffers = dict(model.named_buffers())
    require(set(buffers) == set(snapshot['buffers']), 'Registered buffers changed')
    with torch.no_grad():
        for name, value in snapshot['buffers'].items(): buffers[name].copy_(value)
    model.train()
    if hasattr(model.criterion, 'ror_weight'):
        model.criterion.set_epoch(0); model.criterion.reset_statistics()
    model.zero_grad(set_to_none=True)
    optimizer.load_state_dict(deepcopy(snapshot['optimizer']))
    restored = dict(state=digest_tree(model.state_dict()), buffers=digest_tree(dict(model.named_buffers())),
                    optimizer=digest_tree(optimizer.state_dict()), groups=digest_tree(group_spec(model, optimizer)))
    restore_rng(snapshot['rng'])
    restored['rng'] = digest_tree(rng_state())
    require(restored == snapshot['identity'], 'Incomplete parameter/buffer/RNG/optimizer restoration')
    require(all(p.grad is None for p in model.parameters()), 'Stale gradients')
    return restored


def first_adam(initial, gradient, group):
    return initial * (1 - group['lr'] * group['weight_decay']) - group['lr'] * gradient / (gradient.abs() + group['eps'])


def independent_trial(label, model, optimizer, snapshot, batch, result, persist):
    result.update(status='RUNNING', phase='forward', reset=restore_trial(model, optimizer, snapshot))
    require(not optimizer.state, 'This diagnosis is only the first AdamW step')
    predictions = []; matches = []; terms = []
    original_predict = model.predict
    def predict(*args, **kwargs):
        require(not kwargs.get('return_cbr_details', False), 'Zero weight requested CBR diagnostic path')
        value = original_predict(*args, **kwargs)
        predictions.append(digest_tree(value))
        return value
    def matched(module, inputs, output): matches.append(digest_tree(output))
    def criterion(module, inputs, output):
        terms.append({k: float(v.detach()) for k, v in output.items()})
        require('loss_ror' not in output, 'Zero weight added loss_ror')
    handles = [model.criterion.matcher.register_forward_hook(matched), model.criterion.register_forward_hook(criterion)]
    try:
        with patch.object(model, 'predict', side_effect=predict), \
                patch.object(model.model[-1], 'forward_with_diagnostics', side_effect=AssertionError('Unexpected CBR diagnostic forward')), \
                patch('ultralytics.models.utils.ror.matched_loss', side_effect=AssertionError('Unexpected zero-weight ROR graph')):
            loss, shown = model.loss(batch)
        require(torch.isfinite(loss), 'Nonfinite loss')
        result.update(phase='backward', prediction_hashes=predictions, matcher_hashes=matches, loss_terms=terms,
                      loss=float(loss.detach()), shown=shown.tolist(), graph=graph_signature(loss, {id(p): n for n, p in model.named_parameters()}),
                      pre_backward_rng=digest_tree(rng_state()), forward_buffers=digest_tree(dict(model.named_buffers())))
        persist()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            loss.backward()
        result['backward_warnings'] = sorted({str(w.message) for w in caught})
    finally:
        for handle in handles: handle.remove()
    raw = cpu_copy({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    result['phase'] = 'clip_and_step'
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    coefficient = (10. / (norm + 1e-6)).clamp(max=1.)
    require(torch.isfinite(norm), 'Nonfinite gradient norm')
    clipped = cpu_copy({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    optimizer.step()
    after = cpu_copy(dict(model.named_parameters()))
    groups = {n: group for group in group_spec(model, optimizer) for n in group['params']}
    formula = {}
    for name, g in clipped.items():
        expected = first_adam(snapshot['state'][name], g, groups[name])
        formula[name] = dict(max_abs=float((after[name]-expected).abs().max()),
                             passed=bool(torch.allclose(after[name], expected, atol=2e-7, rtol=2e-6)))
    result.update(status='PASS' if all(r['passed'] for r in formula.values()) else 'FAIL', phase='complete',
                  status_scope='Trial execution and its own AdamW formula only; not pairwise independent-update consistency',
                  total_gradient_norm=float(norm), clip_coefficient=float(coefficient), formula=formula,
                  formula_tolerance=dict(atol=2e-7, rtol=2e-6), gradient_none=[n for n, _ in model.named_parameters() if n not in raw],
                  optimizer_after=digest_tree(optimizer.state_dict()), label=label)
    persist()
    return dict(raw=raw, clipped=clipped, after=after)


def identical_gradient_control(models, optimizers, snapshot, raw, result, persist):
    """No backward calls: same raw gradients -> native clip -> independent AdamW."""
    result.update(status='RUNNING', scope='Optimizer wiring only; cannot pass independent-backward equivalence', trials={})
    values = {}; grads = {}; states = {}
    for label, model in models.items():
        optimizer = optimizers[label]
        result['trials'][label] = dict(reset=restore_trial(model, optimizer, snapshot))
        for name, p in model.named_parameters():
            p.grad = raw[name].to(p.device).clone() if name in raw else None
        assigned = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        require(digest_tree(assigned) == digest_tree(raw), 'Copied gradient input differs')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        grads[label] = cpu_copy({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
        optimizer.step()
        values[label] = cpu_copy(dict(model.named_parameters()))
        states[label] = digest_tree(optimizer.state_dict())
        result['trials'][label].update(norm=float(norm), clip_coefficient=float((10./(norm+1e-6)).clamp(max=1.)),
                                     clipped_sha256=digest_tree(grads[label]), optimizer_after_sha256=states[label])
        persist()
    result.update(clipped_gradients_exact=digest_tree(grads['mother']) == digest_tree(grads['ror']),
                  optimizer_states_exact=states['mother'] == states['ror'],
                  parameters_exact=digest_tree(values['mother']) == digest_tree(values['ror']),
                  parameter_max_abs=max(float((v-values['ror'][n]).abs().max()) for n, v in values['mother'].items()))
    result['status'] = 'PASS' if all(result[k] for k in ('clipped_gradients_exact', 'optimizer_states_exact', 'parameters_exact')) else 'FAIL'
    persist()


def coordinate_details(name, flat, data, trials, initial, group):
    labels = list(data)
    p0 = float(initial.flatten()[flat]); observations = {}
    for label in labels:
        raw = data[label]['raw'][name]; g = float(data[label]['clipped'][name].flatten()[flat])
        rg = float(raw.flatten()[flat]); after = float(data[label]['after'][name].flatten()[flat])
        predicted = p0 * (1-group['lr']*group['weight_decay']) - group['lr']*g/(abs(g)+group['eps'])
        observations[label] = dict(raw_gradient=rg, clipped_gradient=g, raw_sign=int(np.sign(rg)), clipped_sign=int(np.sign(g)),
            clip_coefficient=trials[label]['clip_coefficient'], total_gradient_norm=trials[label]['total_gradient_norm'],
            clipped_abs_over_eps=abs(g)/group['eps'], raw_abs_over_tensor_max=abs(rg)/max(float(raw.abs().max()),1e-300),
            parameter_after=after, actual_update=after-p0, predicted_after_fp64=predicted, formula_residual=after-predicted)
    def compare(a, b):
        left, right = observations[a], observations[b]
        actual = left['actual_update']-right['actual_update']
        predicted = left['predicted_after_fp64']-right['predicted_after_fp64']
        gd = abs(left['clipped_gradient']-right['clipped_gradient'])
        return dict(a=a,b=b,actual_update_difference=actual,formula_predicted_difference=predicted,
                    difference_residual=actual-predicted, gradient_sign_flip=left['clipped_sign']*right['clipped_sign']==-1,
                    amplification_per_gradient_difference=abs(actual)/gd if gd else None,
                    fraction_of_2lr=abs(actual)/(2*group['lr']))
    a = [k for k in labels if k.startswith('mother')]; b = [k for k in labels if k.startswith('ror')]
    def worst(pairs): return compare(*max(pairs, key=lambda ab: abs(observations[ab[0]]['actual_update']-observations[ab[1]]['actual_update'])))
    cross = worst([(x,y) for x in a for y in b]); native = worst([(a[i],a[j]) for i in range(3) for j in range(i+1,3)])
    own = worst([(b[i],b[j]) for i in range(3) for j in range(i+1,3)])
    ranges = {label: [min(observations[k]['raw_gradient'] for k in keys), max(observations[k]['raw_gradient'] for k in keys)]
              for label, keys in (('mother',a),('ror',b))}
    return dict(index=[int(i) for i in np.unravel_index(flat, initial.shape)], parameter_before=p0,
                optimizer={k: group[k] for k in ('lr','eps','weight_decay','betas')}, trials=observations,
                worst_cross=cross, mother_repeat=native, ror_repeat=own, raw_gradient_ranges=ranges,
                mother_repeat_spans_both_signs=ranges['mother'][0]<0<ranges['mother'][1],
                raw_gradient_ranges_overlap=max(ranges['mother'][0],ranges['ror'][0])<=min(ranges['mother'][1],ranges['ror'][1]),
                cross_update_within_mother_observed_range=abs(cross['actual_update_difference'])<=abs(native['actual_update_difference']),
                interpretation='Sample evidence only: 3 trials cannot establish a distribution or justify a larger tolerance')


def summarize(data, trials, snapshot, prior):
    groups = {n: group for group in snapshot['groups'] for n in group['params']}
    labels = list(data); mother = [k for k in labels if k.startswith('mother')]; ror = [k for k in labels if k.startswith('ror')]
    parameters = {}; samples = {}; historical = {n for n, row in prior.items() if row['ror_vs_parent_max_abs']>row['bound']}
    require(historical <= set(groups), 'Historical failing names not found among parameters')
    for name in groups:
        a = torch.stack([data[k]['after'][name] for k in mother]); b = torch.stack([data[k]['after'][name] for k in ror])
        lo_a, hi_a = a.min(0).values, a.max(0).values; lo_b, hi_b = b.min(0).values, b.max(0).values
        cross = torch.maximum((hi_a-lo_b).abs(), (hi_b-lo_a).abs())
        bound = prior[name]['bound']; bad = cross>bound
        row = parameters[name] = dict(prior_failed=name in historical, frozen_bound=bound, max_cross_update=float(cross.max()),
            max_mother_repeat_update=float((hi_a-lo_a).max()), max_ror_repeat_update=float((hi_b-lo_b).max()),
            exceeded_coordinate_count=int(bad.sum()), cross_pair_max_abs={x+'/'+y: float((data[x]['after'][name]-data[y]['after'][name]).abs().max()) for x in mother for y in ror})
        if name not in data[labels[0]]['raw']:
            require(all(name not in data[k]['raw'] for k in labels) and not bad.any(), 'Inconsistent None gradients')
            continue
        require(all(name in data[k]['raw'] for k in labels), 'Inconsistent None gradients')
        raw_a = torch.stack([data[k]['raw'][name] for k in mother]); raw_b = torch.stack([data[k]['raw'][name] for k in ror])
        row.update(mother_raw_repeat_max_abs=float((raw_a.max(0).values-raw_a.min(0).values).max()),
                   ror_raw_repeat_max_abs=float((raw_b.max(0).values-raw_b.min(0).values).max()))
        if name in historical or bad.any():
            ranked = cross.flatten().masked_fill(~bad.flatten(), -1) if bad.any() else cross.flatten()
            count = min(SAMPLES, int(bad.sum()) if bad.any() else ranked.numel())
            samples[name] = dict(exceeded_now=bool(bad.any()), original_failure=prior[name],
                coordinates=[coordinate_details(name, int(i), data, trials, snapshot['state'][name], groups[name]) for i in ranked.topk(count).indices])
    newly_failed = [n for n,v in parameters.items() if v['exceeded_coordinate_count'] and n not in historical]
    # No recalibrated bound or statistical acceptance derived from three repeats.
    return dict(status='REVIEW_REQUIRED', original_failed_parameters=sorted(historical), newly_failed_parameters=newly_failed,
        exceeded_parameters=[n for n,v in parameters.items() if v['exceeded_coordinate_count']], parameters=parameters, samples=samples,
        reason='Original server failure remains open; current exceedances and unreproduced historical names are reported separately',
        threshold_policy='Use original stored bounds only; three-repeat ranges are descriptive, never a new acceptance threshold')


def run(args):
    output = args.output or ROOT/'outputs/ror_v1'/('update_control_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', conclusion='EVIDENCE_INSUFFICIENT', formal_training='NOT_RUN', gate_updated=False,
                  trials={}, output=str(output.resolve()), repeats_per_model=REPEATS, max_coordinates_per_parameter=SAMPLES,
                  amp=False, precision_scope='Original failing FP32 first-step check only; no change to formal AMP/TF32')
    def persist(): atomic_json(output/'diagnostic.json', report)
    try:
        torch.set_num_threads(4); init_seeds(42, deterministic=True)
        report.update(runtime=runtime(), backend=backend())
        report['source_hashes'] = {name: sha256(ROOT/name) for name in
            ('tools/ror_v1_update_diagnostic.py', 'ultralytics-main/ultralytics/models/utils/ror.py',
             'ultralytics-main/ultralytics/models/utils/loss.py', 'ultralytics-main/ultralytics/nn/tasks.py')}
        require(sha256(args.source)==SOURCE_SHA256, 'Wrong unified initialization source')
        prior, report['prior'] = read_prior(args.prior)
        cls, report['mother_source'] = mother_class()
        losses = parent_module('ultralytics-main/ultralytics/models/utils/loss.py','ultralytics.models.utils._ror_update_parent_loss')
        torch.manual_seed(42)
        unused, initial, _ = controlled_models(args.source)
        model, _ = build_training_model(initial.yaml, initial, dict(nc=1, channels=3))
        mother = cls(deepcopy(model.yaml),nc=1,verbose=False); mother.load_state_dict(model.state_dict(),strict=True)
        mother.criterion = losses.RTDETRDetectionLoss(nc=1,use_vfl=True)
        model = install(model,dict(check='three-repeat-update'))
        del unused, initial
        device = args.device
        batch = dict(img=torch.rand(2,3,160,160,device=device),
            bboxes=torch.tensor([[.25,.25,.2,.1],[.6,.6,.3,.2],[.7,.2,.1,.25],[.5,.5,.3,.3]],device=device),
            cls=torch.zeros(4,1,device=device),batch_idx=torch.tensor([0,0,0,1],device=device))
        models = dict(mother=mother.to(device).train(),ror=model.to(device).train())
        state = rng_state()
        # Reproduce the one no-grad diagnostic forward preceding the old check's
        # starting BN state. Warm only the mother; clone that complete state to ROR.
        with torch.no_grad():
            restore_rng(state); mother.predict(batch['img'],batch=targets_of(batch))
        model.load_state_dict(mother.state_dict(),strict=True)
        helper = RTDETRTrainer.__new__(RTDETRTrainer)
        optimizers = {k: helper.build_optimizer(v,name='AdamW',lr=.0005,momentum=.937,decay=.0001) for k,v in models.items()}
        specs = {k: group_spec(models[k], o) for k,o in optimizers.items()}
        require(specs['mother']==specs['ror'], 'Optimizer parameter grouping differs')
        snapshot = dict(state=cpu_copy(mother.state_dict()),buffers=cpu_copy(dict(mother.named_buffers())),rng=state,
                        optimizer=deepcopy(optimizers['mother'].state_dict()),groups=specs['mother'])
        snapshot['identity'] = dict(state=digest_tree(snapshot['state']),buffers=digest_tree(snapshot['buffers']),
            rng=digest_tree(state),optimizer=digest_tree(snapshot['optimizer']),groups=digest_tree(snapshot['groups']))
        require(set(prior)==set(snapshot['state']), 'Prior/current state names differ')
        report.update(initial_state=snapshot['identity'], optimizer_groups=snapshot['groups'], batch_sha256=digest_tree(batch))
        persist()
        data = {}
        # Interleave labels to avoid confounding model type with run order.
        for i in range(REPEATS):
            for kind in ('mother','ror'):
                label = kind+'_'+str(i); row = report['trials'][label] = {}
                print('Bounded update trial '+label,flush=True)
                try: data[label] = independent_trial(label,models[kind],optimizers[kind],snapshot,batch,row,persist)
                except BaseException as error:
                    row.update(status='FAIL',error=repr(error));raise
                require(digest_tree(batch)==report['batch_sha256'], 'Trial changed input batch')
                require(backend()==report['backend'], 'Trial changed backend/AMP configuration')
        keys = ('prediction_hashes','matcher_hashes','loss_terms','loss','shown','graph','pre_backward_rng','forward_buffers','gradient_none')
        reference = report['trials']['mother_0']
        report['zero_path_audit'] = {k: all(row[k]==reference[k] for row in report['trials'].values()) for k in keys}
        report['zero_path_audit']['no_ror_or_diagnostic_calls'] = True
        report['zero_path_audit']['status'] = 'PASS' if all(report['zero_path_audit'].values()) else 'FAIL'
        report['mother_repeat_controls_exact'] = all(report['trials']['mother_'+str(i)][k]==reference[k]
                                                      for i in range(REPEATS) for k in keys)
        report['same_gradient_optimizer'] = {}
        identical_gradient_control(models,optimizers,snapshot,data['mother_0']['raw'],report['same_gradient_optimizer'],persist)
        report['independent_backward'] = summarize(data,report['trials'],snapshot,prior)
        variable = [n for n,v in report['independent_backward']['parameters'].items() if v.get('mother_raw_repeat_max_abs',0)>0]
        report['native_nondeterminism_observed'] = bool(variable) and report['mother_repeat_controls_exact']
        report['native_varying_gradient_parameters'] = variable
        report['analytic_steps_all_pass'] = all(row['status']=='PASS' for row in report['trials'].values())
        report['conclusion'] = 'NATIVE_NONDETERMINISM_OBSERVED_BUT_ORIGINAL_FAILURE_ATTRIBUTION_INCOMPLETE' if report['native_nondeterminism_observed'] else 'EVIDENCE_INSUFFICIENT'
        if (report['zero_path_audit']['status']!='PASS' or report['same_gradient_optimizer']['status']!='PASS'
                or not report['analytic_steps_all_pass']):
            report['conclusion'] = 'WIRING_OR_DIAGNOSTIC_FAILURE_REQUIRES_INVESTIGATION'
        report.update(status='REVIEW_REQUIRED', backend_after=backend(),
            acceptance='Original failure remains open. Same-gradient PASS is not independent-backward PASS. No threshold increase.')
        persist()
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        raise
    finally: persist()
    print(json.dumps({k:report[k] for k in ('status','conclusion','output','formal_training')},ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prior',type=Path,default=ROOT/'docs/ror_v1/update_diagnostic/server_prior.zip')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,help='New directory; never overwrite original evidence')
    result = run(parser.parse_args())
    raise SystemExit(2 if result['status']=='REVIEW_REQUIRED' else 1)


if __name__=='__main__': main()
