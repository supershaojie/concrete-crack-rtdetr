"""Two-image-per-split pipeline check with copied existing samples; no complete dataset evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import torch
from mincompat_v2 import ROOT,require,sha256,write_json,YAML
from mincompat_v2_results import evaluate


def run(data_root,weights,output):
    output=Path(output);require(not output.exists(),'Preserve prior pipeline artifacts');output.mkdir(parents=True)
    original=Path(data_root);subset=output/'subset';manifest=[]
    for split in ('train','val','test'):
        files=sorted(p for p in (original/'images'/split).iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'})[:2]
        require(len(files)==2,f'Need two existing {split} images')
        for image in files:
            for source,category in ((image,'images'),(original/'labels'/split/(image.stem+'.txt'),'labels')):
                target=subset/category/split/source.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target);require(sha256(source)==sha256(target),'Sample copy mismatch')
                manifest.append(dict(source=source,path=target,sha256=sha256(source),original_split=split))
    cfg=subset/'data.yaml';YAML.save(cfg,dict(path=subset.resolve().as_posix(),train='images/train',val='images/val',test='images/test',names={0:'crack'}))
    write_json(output/'sample_manifest.json',manifest)
    reports={}
    for split in ('val','test'):
        report=evaluate('triad_mincompat_v2',weights,cfg,split,output/split,
            val_report=output/'val/metrics.json' if split=='test' else None)
        require(report['images']==2,'Pipeline exceeded bounded subset')
        reports[split]=dict(images=report['images'],export_complete=report['export_complete'],
            checkpoint_sha256=report['checkpoint_sha256'],settings=report['settings'],scope='pipeline only; no research performance claim')
    write_json(output/'pipeline_validation.json',dict(status='passed',scope='two copied images per original split; untrained controlled model',splits=reports))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4);run(a.data_root,a.weights,a.output)
