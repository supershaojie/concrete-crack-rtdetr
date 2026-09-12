"""Read trusted local experiment evidence without executing archive contents."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMITS = dict(c2='67c3078e54a657fd96d65fee657a75fbb1dae0d6',
               c24='f6e9dfda765046ae7691302cf5ec89d3f76cec5d',
               lif='0e95bbade3558b0d2b77c5531483c60810391d88')
MODULES = dict(c24='scca_aifi.py', lif='lif_down.py')
HASHES = dict(c24='67b0d347d5305c16d48bb031cd10fd1d23b7ea447b58b2423166eb68b81dedca',
              lif='26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def audit(c24, lif, reference):
    out = ROOT/'outputs/c24_lif_v1_audit'; out.mkdir(parents=True, exist_ok=True)
    docs = ROOT/'docs/c24_lif_v1'; docs.mkdir(parents=True, exist_ok=True)
    report = dict(commits=COMMITS, packages={}, reference_files=[], repository={
        k: git(*v).decode().strip() for k,v in dict(status=['status','--porcelain'],
        worktrees=['worktree','list'], remote=['remote','-v'], branch=['branch','--show-current']).items()})
    contents = {}
    for kind, path in [('c24',c24),('lif',lif)]:
        path=Path(path)
        rows={}
        with tarfile.open(path) as archive:
            if kind=='c24':
                prefix='c24_scca_complete_20260908_212957_GmHKX5/'
                names=['metadata/working_tree.patch','metadata/crack_autodl.yaml',
                       'launch_and_evaluation/actual_train_args.yaml','launch_and_evaluation/initialization.json',
                       'launch_and_evaluation/evaluation_val/metrics.json','launch_and_evaluation/evaluation_test/metrics.json']
                blob=archive.extractfile(prefix+'metadata/source_at_commit.tar.gz').read()
                rows['metadata/source_at_commit.tar.gz']=dict(sha256=digest(blob),bytes=len(blob))
                with tarfile.open(fileobj=io.BytesIO(blob)) as source:
                    for name in ['ultralytics-main/ultralytics/nn/modules/scca_aifi.py',
                        'ultralytics-main/ultralytics/nn/tasks.py','ultralytics-main/ultralytics/nn/modules/__init__.py',
                        'ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-scca.yaml',
                        'docs/scca/README.md','docs/scca/VALIDATION.md','docs/scca/c2_args.yaml','experiment_records/scca_aifi.md']:
                        b=source.extractfile(name).read(); rows['source/'+name]=dict(sha256=digest(b),bytes=len(b))
                        dest=out/kind/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
                assert archive.extractfile(prefix+'metadata/working_tree.patch').read()==b''
            else:
                prefix=''
                names=['metadata/source/'+n for n in ['ultralytics-main/ultralytics/nn/modules/lif_down.py',
                    'ultralytics-main/ultralytics/nn/tasks.py','tools/init_lif_down.py','tools/lif_down_topology.py',
                    'tools/check_lif_down.py','tools/train_lif_down.py','tools/lif_down_results.py']]
                names+=['metadata/launch/'+n for n in ['authoritative_c2_args.yaml','actual_train_args.yaml',
                    'initialization.json','data_config.yaml']]+['training/args.yaml','val/metrics.json','test/metrics.json']
            for name in names:
                b=archive.extractfile(prefix+name).read();rows[name]=dict(sha256=digest(b),bytes=len(b))
                dest=out/kind/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
                contents[kind+'/'+name]=b
        name='ultralytics-main/ultralytics/nn/modules/'+MODULES[kind]
        original=git('show',COMMITS[kind]+':'+name)
        package=(out/kind/('source' if kind=='c24' else 'metadata/source')/name).read_bytes()
        assert digest(original)==digest(package)==HASHES[kind]
        current=ROOT/name
        if kind=='lif': assert current.read_bytes()==original
        report['packages'][kind]=dict(path=str(path),sha256=digest(path.read_bytes()),read_files=rows,
            module_git_sha256=digest(original),module_package_sha256=digest(package),module_lf_sha256=digest(original.replace(b'\r\n',b'\n')))
    base=yaml.safe_load(contents['lif/metadata/launch/authoritative_c2_args.yaml'])
    assert len(base)==109
    report['recipes']={}
    for kind,name in [('c24','launch_and_evaluation/actual_train_args.yaml'),('lif','metadata/launch/actual_train_args.yaml')]:
        actual=yaml.safe_load(contents[kind+'/'+name]);assert actual.keys()==base.keys()
        differences=[dict(field=k,c2=base[k],parent=actual[k],c2_type=type(base[k]).__name__,parent_type=type(actual[k]).__name__)
            for k in base if type(base[k]) is not type(actual[k]) or base[k]!=actual[k]]
        assert {d['field'] for d in differences}=={'model','name','save_dir'}
        report['recipes'][kind]=dict(fields=109,differences=differences)
    (docs/'c2_args.yaml').write_bytes(contents['lif/metadata/launch/authoritative_c2_args.yaml'])
    (docs/'c2_data.yaml').write_bytes(contents['lif/metadata/launch/data_config.yaml'])
    report['parent_metrics']={}
    for kind,names in [('c24',['launch_and_evaluation/evaluation_val/metrics.json','launch_and_evaluation/evaluation_test/metrics.json']),('lif',['val/metrics.json','test/metrics.json'])]:
        report['parent_metrics'][kind]={}
        for split,name in zip(['val','test'],names):
            value=json.loads(contents[kind+'/'+name]);report['parent_metrics'][kind][split]={k:value[k] for k in value if k in
                ['status','precision','recall','mAP50','AP75','mAP50_95','policy','data_sha256','checkpoint_sha256','images','ground_truth','settings']}
            (docs/(kind+'_'+split+'_metrics.json')).write_text(json.dumps(report['parent_metrics'][kind][split],indent=2)+'\n')
    ref=Path(reference)
    for name in ['ultralytics/nn/modules/transformer.py','ultralytics/nn/extra_modules/block.py','ultralytics/nn/extra_modules/wtconv2d.py']:
        p=ref/name
        if p.is_file():
            b=p.read_bytes();report['reference_files'].append(dict(path=str(p),sha256=digest(b),bytes=len(b),status='READ'))
            # Evidence excerpts for the interface, four phases and frequency encoding.
            text=b.decode('utf-8',errors='replace'); lines=text.splitlines()
            tokens=('class AIFI','class TransformerEncoderLayer','class SPDConv','class WTConv','def create_wavelet','def wavelet_transform')
            excerpts=[]
            for i,line in enumerate(lines):
                if any(t in line for t in tokens):excerpts.extend(lines[max(0,i-1):i+95])
            (out/('reference_'+p.name+'.txt')).write_text('\n'.join(excerpts),encoding='utf-8')
        else:report['reference_files'].append(dict(path=str(p),status='MISSING'))
    report['reference_note']='Background only; no reference implementation copied.'
    spec=Path('D:/rtdetr跑结果/C24／SCCA ＋ LIF-Down v1/RTDETR_C24_LIF_v1_Codex_Spec.md')
    report['spec']=dict(path=str(spec),sha256=digest(spec.read_bytes()))
    (docs/'PROVENANCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(packages={k:v['sha256'] for k,v in report['packages'].items()},recipes=report['recipes'],references=report['reference_files']),ensure_ascii=True,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--c24',required=True);p.add_argument('--lif',required=True);p.add_argument('--reference',required=True)
    a=p.parse_args();audit(a.c24,a.lif,a.reference)
