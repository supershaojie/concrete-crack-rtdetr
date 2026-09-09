"""Six real images maximum, native debug DN/loss and same-pass exports; no full split evaluation."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import torch
from torch.nn import functional as F
from init_lcr import ROOT, controlled_models, require, runtime, sha256, write_json
from check_lcr import rebuild
import lcr_results as results
from ultralytics.cfg import get_cfg
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.utils import YAML


def formula_check():
    from ultralytics.nn.modules import LCRAIFI
    torch.manual_seed(71)
    m=LCRAIFI(256).eval()
    with torch.no_grad():
        m.dwconv.weight.normal_(0,.2)
        m.gate_out.weight.normal_(0,.2)
        m.gate_out.bias.normal_(0,.2)
    x=torch.randn(1,6,256)
    captured={}
    hook=m.gate_in.register_forward_pre_hook(lambda layer,args: captured.update(e=args[0]))
    actual=m._ffn(x,2,3)
    hook.remove()
    # Independent float64 reference, explicit replicated windows and grouped matrix products.
    u=F.linear(x.double(),m.fc1.weight.double(),m.fc1.bias.double()).transpose(1,2).reshape(1,1024,2,3)
    s=torch.zeros_like(u)
    for y in range(2):
        for z in range(3):
            for dy in range(-1,2):
                for dz in range(-1,2):
                    s[:,:,y,z]+=u[:,:,min(1,max(0,y+dy)),min(2,max(0,z+dz))]*m.dwconv.weight[:,0,dy+1,dz+1].double()
    mean=torch.zeros_like(s)
    for y in range(2):
        for z in range(3):
            for dy in range(-1,2):
                for dz in range(-1,2):
                    mean[:,:,y,z]+=s[:,:,min(1,max(0,y+dy)),min(2,max(0,z+dz))]/9
    d=s-mean
    e=d/(d.square().mean(1,keepdim=True)+1e-6).sqrt()
    def grouped(v,layer):
        parts=[]
        for g in range(16):
            ni,no=layer.in_channels//16,layer.out_channels//16
            parts.append(torch.einsum('oc,bchw->bohw',layer.weight[g*no:(g+1)*no,:,0,0].double(),v[:,g*ni:(g+1)*ni])+layer.bias[g*no:(g+1)*no].double()[None,:,None,None])
        return torch.cat(parts,1)
    t=grouped(F.gelu(grouped(e,m.gate_in)),m.gate_out)
    a=F.gelu(s)+.5*t.tanh()*(F.gelu(s)-F.gelu(mean))
    expected=F.linear(a.flatten(2).transpose(1,2),m.fc2.weight.double(),m.fc2.bias.double())
    torch.testing.assert_close(actual.double(),expected,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(captured['e'].double(),e,atol=3e-6,rtol=2e-5)
    actual.square().sum().backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),'Formula gradient')
    return dict(fp64_formula_max_abs=float((actual.double()-expected).abs().max().detach()), rms_max_abs=float((captured['e'].double()-e).abs().max().detach()), rms_input_dtype=str(captured['e'].dtype), atol=2e-6, rtol=2e-5)


def run(source,data_root,output):
    require(not output.exists(),'Preserve evidence')
    output.mkdir(parents=True)
    report=dict(status='failed',runtime=runtime(),scope='bounded_real_samples',full_val_test='NOT_RUN',formal_training='NOT_RUN',full_server_preflight='NOT_RUN')
    try:
        report['formula']=formula_check()
        if not data_root.is_dir():
            report.update(real_data='NOT_RUN: path unavailable',status='passed')
            return
        with tempfile.TemporaryDirectory(dir=output) as tempname:
            temp=Path(tempname).resolve(); subset=temp/'subset'
            files=[]
            for split in ('train','val','test'):
                images=sorted((data_root/'images'/split).glob('*.jpg'))[:2]
                require(len(images)==2,'Need exactly two real images per split')
                for image in images:
                    label=data_root/'labels'/split/(image.stem+'.txt')
                    require(label.is_file(),'Missing real GT')
                    for f,kind in ((image,'images'),(label,'labels')):
                        dest=subset/kind/split/f.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(f,dest)
                    files.append(dict(split=split,image=str(image.resolve()),image_sha256=sha256(image),label_sha256=sha256(label)))
            report['real_files']=files
            data=dict(path=str(subset),train='images/train',val='images/val',test='images/test',names={0:'crack'},nc=1)
            config=temp/'data.yaml'; YAML.save(config,data)
            _,target80,_mapping=controlled_models(source)
            model=rebuild(target80); model.nc=1; del target80  # Native trainer normally sets dataset attributes
            hyp=get_cfg(); hyp.imgsz=160
            dataset=RTDETRDataset(img_path=str(subset/'images/train'),imgsz=160,batch_size=2,augment=False,hyp=hyp,rect=False,cache=False,data=data)
            sample=dataset.collate_fn([dataset[0],dataset[1]])
            sample['img']=sample['img'].float()/255
            model.train()
            groups=[int((sample['batch_idx']==i).sum()) for i in range(2)]
            pred=model.predict(sample['img'],batch=dict(cls=sample['cls'].flatten().long(),bboxes=sample['bboxes'],batch_idx=sample['batch_idx'].long(),gt_groups=groups))
            loss,items=model.loss(sample,pred); loss.sum().backward()
            require(torch.isfinite(loss).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Real native loss nonfinite')
            require(pred[-1]['dn_num_split'][0]>0,'Native DN absent')
            report['real_native_loss']=dict(loss=float(loss.sum().detach()),items=items.detach().tolist(),gt_groups=groups,dn_num_split=pred[-1]['dn_num_split'],batch=2,imgsz=160,device='cpu',augmentation=False,note='bounded debug only, no optimizer step, no formal recipe change')
            checkpoint=temp/'debug_only.pt'
            torch.save(dict(epoch=-1,model=deepcopy(model).eval(),train_args={'task':'detect'},debug_only=True),checkpoint)
            report['checkpoint_sha256']=sha256(checkpoint)
            with patch.object(results,'EVAL',dict(results.EVAL,imgsz=160,batch=1)):
                for split in ('val','test'):
                    report[split]=results.evaluate(checkpoint,config,split,output/('limited_'+split),device='cpu',val_report=output/'limited_val/metrics.json' if split=='test' else None,evidence_scope='bounded_real_samples')
                    require(report[split]['images']==2 and report[split]['predictions']==600,'Export count differs')
            report['status']='passed'
    except BaseException as error:
        report['error']=repr(error); raise
    finally:
        write_json(output/'report.json',report)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); torch.set_num_threads(4); torch.manual_seed(42)
    run(a.source,a.data_root,a.output)
