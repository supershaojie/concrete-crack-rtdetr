"""Small BSC lifecycle contracts; no training or artifact creation on import."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from init_bscrep_v1 import ROOT, MODEL_DIR, VARIANTS, require, sha256
from ultralytics.utils import YAML

DEFAULT_VARIANT = 'cbr_lif_bscrep_v1'
BASE_SHA = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
BRANCH = 'exp-rtdetr-r18-lite-bscrep-v1'
DOCS = ROOT / 'docs/bscrep_v1'


def git(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT, text=True).strip()


def lf_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def code_hashes():
    # Includes reused parents and original loss/Trainer/AMP code, not just the new module.
    files = []
    for folder in (ROOT/'tools', ROOT/'ultralytics-main/ultralytics'):
        files.extend(p for p in folder.rglob('*') if p.suffix in {'.py', '.yaml', '.yml', '.sh'} and '__pycache__' not in p.parts)
    return {p.relative_to(ROOT).as_posix(): lf_sha(p) for p in sorted(files)}


def fingerprint(source, initialized, variant, data=None):
    result = dict(commit=git('rev-parse', 'HEAD'), variant=variant, source_sha256=sha256(source),
                  initialized_sha256=sha256(initialized), config_sha256=lf_sha(MODEL_DIR/VARIANTS[variant][0]),
                  recipe_sha256=lf_sha(DOCS/'parent_args.yaml'), code=code_hashes())
    if data:
        result['data_config_sha256'] = sha256(data)
    return result


def paths(variant=DEFAULT_VARIANT, main=None):
    main = Path(main or os.environ.get('BSCREP_V1_MAIN', '/root/autodl-tmp/projects/Crack_RTDETR')).resolve()
    name = VARIANTS[variant][2]
    return dict(main=main, name=name, run=main/'runs/c_series'/name, launch=ROOT/'outputs'/variant,
                init=ROOT/'weights'/f'{variant}_controlled_init.pt',
                source=main/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt',
                data=main/'configs/crack_autodl.yaml')


def recipe(variant, initialized, data, project):
    parent = YAML.load(DOCS/'parent_args.yaml')
    c2 = YAML.load(DOCS/'c2_args.yaml')
    allowed = {'model', 'name', 'project', 'save_dir', 'data'}
    require(set(parent) == set(c2) and all(parent[k] == c2[k] and type(parent[k]) is type(c2[k])
            for k in parent if k not in allowed), 'Successful parent differs from authoritative C2 recipe')
    name = VARIANTS[variant][2]
    target = dict(parent, model=str(Path(initialized).resolve()), data=str(Path(data).resolve()),
                  project=str(Path(project).resolve()), name=name, save_dir=str(Path(project).resolve()/name))
    rows = [dict(field=k, parent=parent[k], target=target[k], changed=parent[k] != target[k],
                 reason='model/output identity or verified environment path' if parent[k] != target[k] else 'inherited')
            for k in sorted(parent)]
    require({r['field'] for r in rows if r['changed']} <= allowed, 'Unexpected recipe change')
    return target, rows


def verified_data(data):
    from ultralytics.data.utils import check_det_dataset
    from c19_lif_v1_data import dataset_inventory
    raw, expected = YAML.load(data), YAML.load(DOCS/'c2_data.yaml')
    require(set(raw) == set(expected), 'Data fields changed')
    require(all(raw[k] == expected[k] for k in raw if k != 'path'), 'Data splits/classes changed')
    resolved = check_det_dataset(str(data), autodownload=False)
    inventory = dataset_inventory(Path(resolved['path']))
    archived = json.loads((DOCS/'parent_dataset_inventory.json').read_text())
    require(inventory == archived, 'Dataset identity differs from successful CBR+LIF archive')
    return resolved, inventory


def optimizer_coverage(model, optimizer):
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    require(len(ids) == len(set(ids)) and set(ids) == set(expected), 'Optimizer missing/duplicate/unexpected parameters')
    require(type(optimizer).__name__ == 'AdamW', 'Expected native AdamW')
    return dict(status='PASSED', trainable_tensors=len(expected), occurrences=1,
                bsc={name: ids.count(pid) for pid, name in expected.items() if '.bsc.' in name})
