"""Content-bound TRC preflight gate shared by plan/start and bounded diagnostics."""
import hashlib
import json
from pathlib import Path
from init_trc_v1 import ROOT, VARIANTS, require, sha256


def source_files():
    # Include native training/loss/matching code, wrappers and both YAMLs.
    files=list((ROOT/'ultralytics-main/ultralytics').rglob('*.py'))
    files+=list((ROOT/'ultralytics-main/ultralytics/cfg').rglob('*.yaml'))
    files+=list((ROOT/'tools').glob('*.py'))
    files+=list((ROOT/'tools').glob('*trc*.sh'))
    files+=[ROOT/'docs/c19_lif_v1/c2_args.yaml', ROOT/'docs/trc_v1/cbr_lif_args.yaml']
    files+=list((ROOT/'ultralytics-main/tests').glob('*trc*.py'))
    return sorted(set(files))


def fingerprint(variant,source,initialized,data=None):
    require(variant in VARIANTS,'Unknown variant')
    return dict(variant=variant,source_sha256=sha256(source),initialized_sha256=sha256(initialized),
                source_files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in source_files()},
                data_config_sha256=hashlib.sha256(Path(data).read_bytes().replace(b'\r\n',b'\n')).hexdigest() if data else None)


def verify_preflight(path,variant,source,initialized,data=None):
    report=json.loads(Path(path).read_text(encoding='utf-8'))
    require(report.get('status')=='PASSED','Preflight is not PASSED (PENDING/skip cannot authorize start)')
    require(report.get('server_capacity',{}).get('status')=='PASSED','Real B16/640/AMP server capacity is mandatory')
    require(report.get('fingerprint')==fingerprint(variant,source,initialized,data),'Source/config/weights/init changed after preflight')
    require(report.get('formal_optimizer_steps')==0,'Formal initialization was updated')
    if data:
        from c19_lif_v1_data import dataset_inventory
        from ultralytics.utils import YAML
        config=YAML.load(data)
        require(report.get('dataset')==dataset_inventory(Path(config['path'])),'Dataset split/labels changed after preflight')
    return report
