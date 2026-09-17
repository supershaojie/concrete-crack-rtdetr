"""Shared SRE identities, immutable recipe and dataset/start-gate evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ultralytics-main'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
from init_c19_lif_v1 import source_contract
from init_lif_down import MODEL_DIR, SOURCE_SHA256, require, sha256, write_json
from ultralytics.utils import YAML

BASE_COMMIT = 'a0459d6a652cb702699087c88fa39a3e4c4087ec'
VARIANTS = {
    'cbr_lif_sre_v1': ('rtdetr-resnet18-lite-cbr-lif-sre-v1.yaml',
                       'rtdetr-resnet18-lite-cbr-lif-down.yaml',
                       'cbr_lif_sre_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
    'sre_v1': ('rtdetr-resnet18-lite-sre-v1.yaml', 'rtdetr-resnet18-lite.yaml',
               'sre_v1_rtdetr_r18_lite_e200_b16_onlineaug'),
}
MAIN = Path(os.environ.get('SRE_MAIN', '/root/autodl-tmp/projects/Crack_RTDETR'))


def paths(variant, output_dir=None):
    folder = Path(output_dir) if output_dir else ROOT / 'outputs/sre' / variant
    return dict(folder=folder, init=folder/'controlled_init.pt', initialization=folder/'initialization.json',
                preflight=folder/'preflight.json', plan=folder/'plan.json', state=folder/'training_state.json',
                run=MAIN/'runs/c_series'/VARIANTS[variant][2])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()


def code_identity():
    """Hash actual executable/config bytes, including new files before the first commit."""
    records = {}
    for directory, suffixes in [('tools', {'.py', '.sh'}), ('ultralytics-main/ultralytics', {'.py', '.yaml'})]:
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts:
                records[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    for name in ('parent_args.yaml','parent_data.yaml','parent_dataset_inventory.json'):
        path=ROOT/'docs/sre'/name
        records[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return dict(commit=commit(), files=records,
                manifest_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())


def runtime():
    import torch
    import ultralytics
    from ultralytics.nn.modules import sre
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT/'ultralytics-main'), 'Wrong ultralytics import')
    require(Path(sre.__file__).resolve() == ROOT/'ultralytics-main/ultralytics/nn/modules/sre.py', 'Wrong SRE import')
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), ultralytics=ultralytics.__file__, sre=sre.__file__,
                commit=commit(), module_hashes=source_contract())


def recipe(variant, initialized, data):
    parent = YAML.load(ROOT/'docs/sre/parent_args.yaml')
    require(len(parent) == 109, 'Authoritative 109-field parent recipe changed')
    target = dict(parent)
    target.update(model=str(Path(initialized).resolve()), data=str(Path(data).resolve()),
                  project=str(MAIN/'runs/c_series'), name=VARIANTS[variant][2],
                  save_dir=str(MAIN/'runs/c_series'/VARIANTS[variant][2]))
    rows = [dict(field=k,parent=parent[k],target=target[k],changed=parent[k]!=target[k]) for k in sorted(parent)]
    require({r['field'] for r in rows if r['changed']} <= {'model','data','project','name','save_dir'}, 'Recipe drift')
    return target, rows


def dataset_identity(data):
    from c19_lif_v1_data import dataset_inventory
    from ultralytics.data.utils import check_det_dataset
    if not Path(data).is_file(): raise FileNotFoundError('Data YAML missing: '+str(data))
    original = YAML.load(data)
    expected = YAML.load(ROOT/'docs/sre/parent_data.yaml')
    require({k:v for k,v in original.items() if k!='path'} == {k:v for k,v in expected.items() if k!='path'},
            'Data YAML semantics differ from successful parent')
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved['nc']==1 and resolved.get('channels',3)==3, 'Dataset class/channel mismatch')
    inventory = dataset_inventory(Path(resolved['path']))
    require(inventory == read_json(ROOT/'docs/sre/parent_dataset_inventory.json'), 'Split paths/labels differ from parent')
    require([inventory[s]['images'] for s in ('train','val','test')] == [6048,1728,864], 'Split counts differ')
    return dict(config=str(Path(data).resolve()), config_sha256=sha256(data), root=str(resolved['path']),
                inventory=inventory, identity_scope='Relative image paths and every label SHA256; no split redivision')


def optimizer_audit(model, optimizer):
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    require(len(ids)==len(set(ids)) and set(ids)==expected, 'Optimizer missing/duplicate trainable parameters')
    return dict(status='PASSED',trainable_tensors=len(ids),exactly_once=True,
                sre=[dict(name=n,occurrences=ids.count(id(p))) for n,p in model.named_parameters() if '.sre.' in n])


def require_clean():
    changes = subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True)
    relevant = [row for row in changes.splitlines() if any(x in row[3:] for x in ('tools/','ultralytics-main/','docs/sre/'))]
    require(not relevant, 'Commit executable/config changes before formal start: '+repr(relevant))
    require(subprocess.run(['git','merge-base','--is-ancestor',BASE_COMMIT,'HEAD'],cwd=ROOT).returncode==0,
            'Pinned successful parent is not an ancestor')


def gate(variant, data, output_dir=None):
    """Never edit a previous report to approve new code: re-run preflight instead."""
    p = paths(variant, output_dir)
    require_clean()
    report = read_json(p['preflight'])
    require(report.get('status')=='PASSED', 'Server preflight is not PASSED; inspect PENDING/FAILED checks')
    require(report['variant']==variant and report['identity']==code_identity(), 'Preflight commit/code/config identity changed')
    require(report['dataset']==dataset_identity(data), 'Data identity changed since preflight')
    init = read_json(p['initialization'])
    require(sha256(p['init'])==report['init_sha256']==init['output_sha256'], 'Controlled init hash changed')
    require(sha256(init['source'])==SOURCE_SHA256==report['source_sha256'], 'Unified source missing/changed')
    capacity = report.get('capacity',{})
    require(capacity.get('status')=='PASSED' and capacity.get('batch')==16 and capacity.get('imgsz')==640
            and capacity.get('native_amp') is True and capacity.get('effective_updates',0)>=2
            and capacity.get('upstream_gradient_after_nonzero_output') is True,
            'Real B16/640 native AMP/two effective updates gate not passed')
    require(capacity.get('resume',{}).get('status')=='PASSED', 'Native optimizer/scaler/epoch resume not verified')
    require(report.get('core',{}).get('status')=='PASSED', 'Mathematics/structure/lifecycle checks not passed')
    require(report.get('runtime',{}).get('cuda_available') is True, 'CUDA server preflight required')
    return report
