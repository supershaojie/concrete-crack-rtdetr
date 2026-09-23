"""RDL 操作边界与真实Validator小样检查；使用隔离临时文件，不启动训练/正式评估。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import rdl_v1 as cli
from init_c19_lif_v1 import build, require, write_json
from rdl_v1_training import configure
from c19_lif_v1_results import postprocess
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML


def rejected(fn):
    try:fn()
    except RuntimeError:return
    raise AssertionError('Required rejection did not occur')


def run():
    results={}
    with TemporaryDirectory(prefix='rdl-ops-') as tmp:
        root=Path(tmp);asset=root/'identity.txt';asset.write_text('initial')
        identity=cli.file_identity([asset]);asset.write_text('changed');rejected(lambda:cli.verify_files(identity))
        results['changed_asset_rejected']='PASS'
        write_json(root/'preflight.json',dict(status='FAIL'))
        with patch.object(cli,'OUT',root),patch.object(cli,'prepared',return_value={}):rejected(cli.gate)
        results['incomplete_preflight_rejected']='PASS'
        run=root/'run';(run/'weights').mkdir(parents=True)
        torch.save(dict(epoch=19,ema=SimpleNamespace()),run/'weights/last.pt')
        rejected(lambda:cli.check_last(dict(run=str(run))))
        torch.save(dict(epoch=-1,ema=SimpleNamespace(rdl_config=cli.CONFIG,rdl_epoch=-1)),run/'weights/last.pt')
        rejected(lambda:cli.check_last(dict(run=str(run))))
        results['foreign_and_stripped_resume_rejected']='PASS'
        predictions=torch.tensor([[[.5,.5,.2,.2,.0005],[.5,.5,.2,.2,.9],[.5,.5,.2,.2,.2]]])
        rows,affected=postprocess(predictions,640,.001)
        torch.testing.assert_close(rows[0]['conf'],torch.tensor([.9,.2]))
        require(affected==1,'Mask fixture must exercise original sorting bug');results['corrected_sorted_mask']='PASS'
        # Actual standalone validator with an audited native model plus RDL metadata.
        # Two synthetic fixtures test execution only, never report their AP as experimental evidence.
        import cv2
        images=root/'images';labels=root/'labels';images.mkdir();labels.mkdir()
        for i in range(2):
            cv2.imwrite(str(images/f'{i}.jpg'),np.full((160,160,3),64+i*80,dtype=np.uint8))
            (labels/f'{i}.txt').write_text('0 0.5 0.5 0.2 0.1\n')
        config=root/'data.yaml';YAML.save(config,dict(path=str(root),train='images',val='images',names={0:'crack'}))
        model=configure(build(nc=1));model.nc=1
        class SmokeValidator(RTDETRValidator):
            def postprocess(self,preds):return postprocess(preds,self.args.imgsz,self.args.conf)[0]
        validator=SmokeValidator(args=dict(data=str(config),imgsz=160,batch=2,device='cpu',workers=0,
                                half=False,plots=False,save_json=False,conf=.001,rect=False,project=str(root/'val'),name='smoke'))
        with torch.no_grad():validator(model=model)
        require(validator.seen==2,'Validator did not process both fixtures')
        results['native_validator_with_RDL_metadata']='PASS (CPU B2/160 synthetic execution only)'
    return dict(status='PASS',checks=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=cli.OUT/'ops_checks.json')
    args=parser.parse_args();torch.set_num_threads(4);write_json(args.output,run())
