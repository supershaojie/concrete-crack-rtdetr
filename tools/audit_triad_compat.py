"""Reproducible semantic/parameter/source audit, with no external result-package dependency."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
from triad_compat import *


def audit():
    report=dict(runtime=runtime(),c2_commit=BASE_COMMIT,source_sha256=SOURCE_SHA256,variants={})
    report['baseline_parameters']=sum(p.numel() for p in build('c2').parameters())
    for variant,config in VARIANTS.items():
        model=build(variant);new=new_keys(model)
        groups=defaultdict(int)
        for name,p in model.named_parameters():
            if name in new:
                category='scca' if '.scca_' in name else 'cbr' if '.cbr.' in name else 'cscef'
                groups[category]+=p.numel()
        report['variants'][variant]=dict(**config,topology=topology(model,variant),new_parameters=dict(groups),
            measured_parameters=sum(p.numel() for p in model.parameters()),new_state_count=len(new))
    protected=['ultralytics-main/ultralytics/nn/modules/head.py','ultralytics-main/ultralytics/models/utils/loss.py',
        'ultralytics-main/ultralytics/models/utils/ops.py','ultralytics-main/ultralytics/models/rtdetr/train.py',
        'ultralytics-main/ultralytics/models/rtdetr/val.py','ultralytics-main/ultralytics/engine/trainer.py',
        'ultralytics-main/ultralytics/cfg/models/rt-detr/'+BASE_YAML]
    require(not git('diff',BASE_COMMIT,'--',*protected),'C2 protected behavior files changed')
    report['protected_C2_files']={p:sha256(ROOT/p) for p in protected}
    report['historical_audit_sha256']=sha256(ROOT/'docs/triad_compat/compatibility_audit.json')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    a=p.parse_args();torch.set_num_threads(4);write_json(a.report,audit())
