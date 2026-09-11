"""Read original experiment records and supplied mincompat archives without extraction."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from sci_adapter import ROOT,require,sha256,write_json,YAML,SOURCE_SHA256,PARENT_COMMIT


def run(result_root):
    evidence=ROOT/'docs/sci_adapter/evidence';evidence.mkdir(exist_ok=True)
    report={}
    for label,folder,variant in [('v52_standalone','mincompat_cscef_v52','cscef_v52_compat'),
                               ('v52_pair','mincompat_cscef_v52_scca_compat','cscef_v52_scca_compat')]:
        candidates=list((Path(result_root)/folder).glob('*.tar.gz'));require(len(candidates)==1,'Ambiguous archive')
        path=candidates[0];digest=sha256(path)
        require(Path(str(path)+'.sha256').read_text().split()[0]==digest,'Archive SHA mismatch')
        selected={}
        wanted={'test/metrics.json','console/initialization.json','console/actual_train_args.yaml',
                'console/parameter_diff.json','console/source_record.json',
                'metadata/source/ultralytics-main/ultralytics/nn/modules/cscef_v52_compat.py'}
        with tarfile.open(path,'r|gz') as tar:
            for member in tar:
                if member.isfile() and member.name in wanted:
                    raw=tar.extractfile(member).read();dest=evidence/(label+'_'+Path(member.name).name)
                    dest.write_bytes(raw);selected[member.name]=dict(file=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())
        require(set(selected)==wanted,'Missing archived history evidence')
        metrics=json.loads((evidence/(label+'_metrics.json')).read_text())
        require(metrics['variant']==variant and metrics['split']=='test' and metrics['status']=='completed','Wrong test evidence')
        init=json.loads((evidence/(label+'_initialization.json')).read_text())
        require(init['status']=='passed' and init['source_sha256']==SOURCE_SHA256,'Initialization not audited')
        require(metrics['runtime']['commit']==PARENT_COMMIT and not metrics['runtime']['dirty'],'Wrong mincompat source')
        archived=(evidence/(label+'_cscef_v52_compat.py')).read_bytes().replace(b'\r\n',b'\n')
        current=(ROOT/'ultralytics-main/ultralytics/nn/modules/cscef_v52_compat.py').read_bytes().replace(b'\r\n',b'\n')
        require(archived==current,'Archived attenuation code mismatch')
        args=YAML.load(evidence/(label+'_actual_train_args.yaml'));c2=YAML.load(ROOT/'docs/sci_adapter/c2_args.yaml')
        diffs=[k for k in c2 if type(args.get(k)) is not type(c2[k]) or args.get(k)!=c2[k]]
        require(set(args)==set(c2) and set(diffs)=={'model','name','save_dir'},'Historical recipe drift')
        report[label]=dict(archive=str(path),sha256=digest,sidecar_verified=True,members=selected,
            attenuation_source_exact=True,C2_recipe_109_fields=True,recipe_differences=diffs,
            metrics={k:metrics[k] for k in ('precision','recall','mAP50','AP75','mAP50_95','images','checkpoint_sha256','runtime')})
    # Earlier source-verified evidence is already committed in the designated base.
    records=ROOT/'docs/mincompat_v2/evidence'
    for pattern in ('c17_c17_cscef_v51_1_metrics.json','c24_c24_scca_1_metrics.json','c25_c25_1_metrics.json',
                    'c17_cscef_v6_1_metrics.json','c24_scca_v2_1_metrics.json'):
        p=records/pattern;data=json.loads(p.read_text(encoding='utf-8'))
        report[pattern]=dict(file=str(p.relative_to(ROOT)),sha256=sha256(p),data=data)
    return dict(status='passed',evidence=report,causal_scope='Observed experiments support interface mismatch as a hypothesis; not causal proof')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();write_json(a.report,run(a.results))
