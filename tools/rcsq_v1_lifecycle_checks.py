"""Bounded lifecycle rejection/recovery probes; mocks forbid any formal training dispatch."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import torch
import train_rcsq_v1 as lifecycle
from init_rcsq_v1 import ROOT, require, write_json


def run_checks(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Preserve prior lifecycle probe')
    output.mkdir(parents=True)
    authority = ROOT / 'docs/rcsq_v1/cbr_lif_authoritative_args.yaml'
    args, diff = lifecycle.recipe(authority, 'cbr_lif_rcsq_v1', output / 'init.pt', ROOT / 'configs/crack.yaml', output / 'runs')
    require(len(args) == 109 and args['amp'] is True and args['batch'] == 16 and args['epochs'] == 200, 'Recipe drift')
    altered = lifecycle.YAML.load(authority)
    altered['amp'] = False
    altered_path = output / 'bad_args.yaml'
    lifecycle.YAML.save(altered_path, altered)
    try:
        lifecycle.recipe(altered_path, 'cbr_lif_rcsq_v1', output / 'init.pt', ROOT / 'configs/crack.yaml', output / 'runs')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed AMP recipe accepted')
    plan_path, old_report = output / 'plan.json', output / 'edited_old_report.json'
    plan_path.write_text('{}', encoding='utf-8')
    write_json(old_report, {'status': 'PASSED', 'note': 'deliberately forged old status; must not authorize dispatch'})
    p = dict(output=str(output), args=args, init=str(output / 'init.pt'), source=str(output / 'source.pt'),
             variant='cbr_lif_rcsq_v1', code={'sha256': 'code'}, init_dir=str(output), main=str(output))
    dummy = torch.nn.Linear(1, 1)
    dispatched = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(lifecycle, 'load_plan', return_value=p))
        stack.enter_context(patch.object(lifecycle, 'verify_init'))
        stack.enter_context(patch.object(lifecycle, 'controlled_models', return_value=({'cbr_lif_rcsq_v1': dummy}, {})))
        stack.enter_context(patch.object(lifecycle, 'RTDETR', return_value=SimpleNamespace(model=dummy)))
        stack.enter_context(patch.object(lifecycle, 'code_fingerprint', return_value={'sha256': 'code'}))
        stack.enter_context(patch.object(lifecycle, 'sha256', return_value='digest'))
        stack.enter_context(patch.object(lifecycle, 'process_token', return_value='live-token'))
        stack.enter_context(patch.object(lifecycle.torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(lifecycle.torch, '__version__', '2.1.2+cu121'))
        stack.enter_context(patch.object(lifecycle.platform, 'python_version', return_value='3.10.13'))
        stack.enter_context(patch.dict(lifecycle.os.environ, {'CONDA_DEFAULT_ENV': 'rtdetr'}))
        fresh = stack.enter_context(patch('preflight_rcsq_v1.run_checks', return_value={'status': 'LOCAL_PASSED_SERVER_PENDING'}))
        stack.enter_context(patch.object(lifecycle, 'run_training', side_effect=lambda *a, **k: dispatched.append(True)))
        try:
            lifecycle.start(plan_path, old_report)
        except RuntimeError as error:
            require('Live server' in str(error), 'Unexpected rejection: ' + str(error))
        else:
            raise AssertionError('Forged report bypassed fresh gate')
        require(fresh.call_count == 1 and not dispatched, 'Start did not run/reject fresh gate')
        try:
            lifecycle.recover_failed_preflight(p, plan_path)
        except RuntimeError as error:
            require('still alive' in str(error), 'Unexpected active-owner rejection')
        else:
            raise AssertionError('Recovery accepted active owner')
    with patch.object(lifecycle, 'process_token', return_value=None):
        archived = lifecycle.recover_failed_preflight(p, plan_path)
    require((archived / 'state.json').is_file() and (archived / 'reservation/owner.json').is_file() and
            not (output / 'launch').exists(), 'Recovery lost evidence or left stale launch')
    write_json(output / 'checks.json', dict(status='PASSED', full_recipe_fields=109, altered_AMP_rejected=True,
               forged_old_PASSED_cannot_start=True, fresh_pending_rejected=True, active_owner_recovery_rejected=True,
               dead_failed_gate_preserved=str(archived), formal_training_calls=0, final_test_calls=0,
               scope='control-flow fixtures/mocks only; no replacement for CUDA or native Trainer checks'))
    return output / 'checks.json'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(run_checks(args.output))
