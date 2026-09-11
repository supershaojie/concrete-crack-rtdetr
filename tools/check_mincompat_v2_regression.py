"""Execute historical C2/C17/C19/C24 sources separately and compare same nonzero state/output."""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
from mincompat_v2 import *
from audit_mincompat_v2 import C17,C19,C24
from check_mincompat_v2 import nested_errors
from ultralytics.utils.patches import torch_load

CHILD = '''import os,sys
os.environ['YOLO_AUTOINSTALL']='false'
sys.path.insert(0,sys.argv[1])
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
torch.set_num_threads(4)
torch.manual_seed(42)
m=RTDETRDetectionModel(sys.argv[2],nc=1,verbose=False).eval()
with torch.no_grad():
 for n,p in m.named_parameters():
  if any(k in n for k in ('output_projection','scca_o','offset_out')):p.normal_(0,.02)
 image=torch.rand(1,3,160,192)
 out=m(image)
torch.save(dict(state=m.state_dict(),image=image,output=out),sys.argv[3])
'''


def run():
    records={}
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mincompat_historical_',dir=ROOT/'outputs') as directory:
        root=Path(directory)
        for label,commit,file in (('c2',BASE_COMMIT,BASE_YAML),('c17',C17,'rtdetr-resnet18-lite-cscef-v51.yaml'),
                                 ('c19',C19,'rtdetr-resnet18-lite-cbr.yaml'),('c24',C24,'rtdetr-resnet18-lite-scca.yaml')):
            folder=root/label;folder.mkdir()
            data=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','archive',commit,'ultralytics-main/ultralytics'],cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                for member in tar:
                    # Read ordinary source files into a new confined folder; never extract links.
                    if not member.isfile():continue
                    target=folder/member.name
                    require(target.resolve().is_relative_to(folder.resolve()),'Unsafe historical source path')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
            fileout=folder/'result.pt'
            subprocess.run([sys.executable,'-c',CHILD,str(folder/'ultralytics-main'),
                str(folder/'ultralytics-main/ultralytics/cfg/models/rt-detr'/file),str(fileout)],check=True,cwd=ROOT)
            ref=torch_load(fileout,map_location='cpu')
            model=RTDETRDetectionModel(str(MODEL_DIR/file),nc=1,verbose=False).eval()
            model.load_state_dict(ref['state'],strict=True)
            with torch.no_grad():out=model(ref['image'])
            records[label]=dict(commit=commit,strict_state_load=True,parameters=sum(p.numel() for p in model.parameters()),
                class_names=[type(m).__name__ for m in model.model],same_nonzero_state_forward=nested_errors(ref['output'],out))
    return dict(status='passed',scope='historical git sources in separate Python processes; synthetic nc1 B1 160x192, nonzero innovation outputs',models=records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
    torch.set_num_threads(4);write_json(a.report,run())
