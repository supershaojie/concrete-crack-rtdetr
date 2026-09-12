"""Two-image evaluation/export fixture, explicitly not full val/test or experiment metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import torch
from init_c17_lif_v1 import ROOT, require, write_json
from c17_lif_v1_results import evaluate, postprocess
from ultralytics.utils import YAML


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    # Deliberately unsorted scores expose the historical sort/mask mismatch.
    pred=torch.tensor([[[.5,.5,.2,.2,.0001],[.5,.5,.4,.4,.9],[.2,.2,.1,.1,.2]]])
    selected,affected=postprocess(pred,640,.001)
    require(torch.equal(selected[0]['conf'],torch.tensor([.9,.2])) and affected==1,'Unified confidence filter')
    data=args.output/'data'
    images=sorted((args.dataset/'images/train').glob('*.jpg'))[:2]
    require(len(images)==2,'Two real samples required')
    for split in ('train','val','test'):
        (data/'images'/split).mkdir(parents=True);(data/'labels'/split).mkdir(parents=True)
        for p in images:
            shutil.copyfile(p,data/'images'/split/p.name)
            shutil.copyfile(args.dataset/'labels/train'/p.with_suffix('.txt').name,data/'labels'/split/p.with_suffix('.txt').name)
    config=args.output/'data.yaml'
    YAML.save(config,dict(path=str(data.resolve()),train='images/train',val='images/val',test='images/test',names={0:'crack'}))
    reports={}
    for split in ('val','test'):
        report=evaluate(args.weights,config,split,args.output/split,device='0' if torch.cuda.is_available() else 'cpu',
                        val_report=args.output/'val/metrics.json' if split=='test' else None,evidence_scope='two_real_train_images_fixture')
        require(report['images']==2 and report['predictions']==600 and report['export_complete'],'Incomplete export fixture')
        reports[split]={k:report[k] for k in ('status','images','predictions','ground_truth','checkpoint_sha256','export_complete','policy','evidence_scope')}
    # Wrong checkpoint must be rejected before output creation or inference.
    tampered=args.output/'tampered.pt';tampered.write_bytes(b'explicit invalid checkpoint fixture')
    try:
        evaluate(tampered,config,'test',args.output/'rejected',val_report=args.output/'val/metrics.json',evidence_scope='two_real_train_images_fixture')
    except RuntimeError:pass
    else:raise AssertionError('Checkpoint SHA mismatch accepted')
    require(not (args.output/'rejected').exists(),'Rejection performed inference')
    write_json(ROOT/'docs/c17_lif_v1/evaluation_checks.json',dict(status='PASSED',scope='two duplicated real training images only, no full val/test',
               reports=reports,sorted_filter='PASSED',checkpoint_mismatch_rejected=True,formal_metrics='NOT_RUN'))


if __name__=='__main__':main()
