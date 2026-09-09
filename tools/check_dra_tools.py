"""Small synthetic evaluation and lifecycle/export regression checks; no real dataset run."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from init_dra import ROOT, build, require, runtime, sha256, write_json
import dra_results as results
import train_dra as train
from ultralytics.nn.modules import DRAAIFI
from ultralytics.utils import YAML


def formula_check():
    m = DRAAIFI(256)
    x = torch.rand(1, 256, 2, 3)
    z = torch.linspace(-3, 3, 48).reshape(1, 8, 2, 3)
    with patch.object(m.dra_predictor, 'forward', return_value=z):
        actual = m.directional_bias(x)
    d = z.reshape(4, 2, 6).transpose(1, 2).double()
    d = d / torch.sqrt(1 + d.square().sum(-1, keepdim=True))
    expected = torch.zeros(1, 8, 6, 6, dtype=torch.float64)
    import math
    for head in range(4):
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                dx, dy = ((j % 3) - (i % 3)) / 3, ((j // 3) - (i // 3)) / 3
                rho = dx * dx + dy * dy
                axis = torch.tensor([(dx*dx-dy*dy)/rho, 2*dx*dy/rho], dtype=torch.float64)
                a, b = (d[head, i] * axis).sum(), (d[head, j] * axis).sum()
                score = -.25 * (torch.logaddexp(-a/.25, -b/.25) - math.log(2))
                expected[0, head, i, j] = .5 * math.exp(-rho/(2*.35**2)) * score
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-6)
    m.directional_bias(torch.rand(1, 256, 3, 5))
    require(m._axis.shape == (15, 15, 2), 'Cache shape transition failed')
    return dict(scalar_float64_formula_max_abs=float((actual.double()-expected).abs().max()), shape_cache='passed')


def state_check(temp):
    p = dict(launch=temp / 'state', run=temp / 'run')
    p['launch'].mkdir()
    def read():
        stream = io.StringIO()
        with patch.object(train, 'paths', return_value=p), redirect_stdout(stream):
            train.status('dra_aifi')
        return next(line for line in stream.getvalue().splitlines() if line.startswith('state:'))
    states = [read()]
    write_json(p['launch'] / 'launch_state.json', dict(status='dispatched'))
    with patch.object(train.shutil, 'which', return_value=None):
        states.append(read())
    write_json(p['launch'] / 'process.json', dict(pid=99999999))
    with patch.object(train.os, 'kill', side_effect=ProcessLookupError):
        states.append(read())
    (p['launch'] / 'process_exit_code.txt').write_text('1')
    states.append(read())
    (p['launch'] / 'process_exit_code.txt').write_text('0')
    states.append(read())
    require(states == ['state: not_submitted', 'state: dispatched', 'state: failed_worker_missing', 'state: failed',
                       'state: successfully_exited'], 'Lifecycle state classification failed')
    holder = SimpleNamespace(_oom_retries=0)
    train.disable_oom_retry(holder)
    require(holder._oom_retries == 3, 'OOM guard failed')
    return states


def run(output):
    require(not output.exists(), 'Preserve previous tool evidence')
    output.mkdir(parents=True)
    report = dict(status='failed', runtime=runtime(), formal_training='NOT_RUN', real_dataset_val_test='NOT_RUN',
                  full_server_preflight='NOT_RUN')
    try:
        report['formula'] = formula_check()
        with tempfile.TemporaryDirectory(dir=output) as directory:
            temp = Path(directory).resolve()
            report['states'] = state_check(temp)
            # Exercise the real independent validator on exactly TWO synthetic images per split.
            data_root = temp / 'synthetic'
            for split in ('val', 'test'):
                (data_root / 'images' / split).mkdir(parents=True)
                (data_root / 'labels' / split).mkdir(parents=True)
                for i in range(2):
                    arr = np.full((128 + i*32, 192, 3), 70 + i*70, dtype=np.uint8)
                    arr[:, 80:85] = 10
                    Image.fromarray(arr).save(data_root / 'images' / split / f'{i}.jpg')
                    (data_root / 'labels' / split / f'{i}.txt').write_text('0 0.43 0.5 0.05 0.7\n')
            data = temp / 'data.yaml'
            YAML.save(data, dict(path=str(data_root), train='images/val', val='images/val', test='images/test', names={0:'crack'}))
            checkpoint = temp / 'synthetic_only.pt'
            torch.save(dict(epoch=-1, model=build(nc=1).eval(), train_args={'task':'detect'}, debug_only=True), checkpoint)
            debug_eval = dict(results.EVAL, imgsz=160, batch=1)
            report['debug_evaluation_overrides'] = dict(imgsz=160, batch=1, device='cpu', synthetic_images_per_split=2)
            with patch.object(results, 'EVAL', debug_eval):
                report['val'] = results.evaluate(checkpoint, data, 'val', temp/'eval_val', device='cpu')
                report['test'] = results.evaluate(checkpoint, data, 'test', temp/'eval_test', device='cpu',
                                                  val_report=temp/'eval_val/metrics.json')
            for split in ('val','test'):
                require(report[split]['images'] == 2 and report[split]['predictions'] == 600, 'Synthetic export coverage failed')
            # Archive missing real results honestly, and read back every member.
            fake_paths = dict(launch=temp/'absent_launch', run=temp/'absent_run')
            destination = temp / 'incomplete.tar.gz'
            with patch.object(results, 'paths', return_value=fake_paths):
                results.package(destination)
            verification = json.loads(Path(str(destination)+'.verification.json').read_text())
            require(not verification['complete'] and verification['missing_evidence'] and verification['all_member_hashes_verified'],
                    'Incomplete evidence falsely called complete')
            results.verify_archive(destination)
            report['archive'] = dict(complete=False, integrity='passed', missing_count=len(verification['missing_evidence']))
            try:
                with patch.object(results, 'paths', return_value=fake_paths):
                    results.package(destination)
            except RuntimeError:
                report['archive_overwrite_guard'] = 'passed'
            else:
                raise RuntimeError('Existing archive overwritten')
        args, rows = train.recipe(ROOT/'docs/dra/c2_args.yaml', 'dra_aifi', ROOT/'weights/dra_aifi_controlled_init.pt')
        require({r['field'] for r in rows if r['changed']} == {'model','name','save_dir'}, 'Default recipe has extra differences')
        write_json(output/'parameter_diff.json', rows)
        report['recipe_changed_fields'] = ['model','name','save_dir']
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(output/'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    run(args.output)
