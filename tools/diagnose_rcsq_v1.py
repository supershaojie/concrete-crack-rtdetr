"""Aggregate RCS-Q diagnostics on a disposable checkpoint copy; never train or evaluate test."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from init_rcsq_v1 import require, sha256, write_json, runtime, verify_model
from check_rcsq_v1 import synthetic_batch, targets
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def training_examples(data_path, device, imgsz):
    """Read two training images/labels, with resize only and no dataset cache writes."""
    import cv2
    import numpy as np

    data_path = Path(data_path).resolve()
    data = YAML.load(data_path)
    root = Path(data.get('path') or data_path.parent)
    if not root.is_absolute():
        root = data_path.parent / root
    entries = data['train'] if isinstance(data['train'], list) else [data['train']]
    images = []
    for entry in entries:
        entry = Path(entry)
        source = entry if entry.is_absolute() else root / entry
        require(source.exists(), 'Training image source missing: ' + str(source))
        if source.is_dir():
            images.extend(p.resolve() for p in source.rglob('*') if p.suffix[1:].lower() in IMG_FORMATS)
        elif source.suffix.lower() == '.txt':
            for line in source.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    path = Path(line.strip())
                    images.append(path.resolve() if path.is_absolute() else (source.parent / path).resolve())
        elif source.suffix[1:].lower() in IMG_FORMATS:
            images.append(source.resolve())
        else:
            raise ValueError('Unsupported training input: ' + str(source))
    images = sorted(set(images))[:2]
    require(len(images) == 2, 'Diagnostics require at least two training images.')
    labels = [Path(p) for p in img2label_paths([str(p) for p in images])]
    tensors, classes, boxes, indices = [], [], [], []
    for index, (image, label) in enumerate(zip(images, labels)):
        pixels = cv2.imread(str(image))
        require(pixels is not None, 'Training image unreadable: ' + str(image))
        pixels = cv2.resize(pixels, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensors.append(torch.from_numpy(np.ascontiguousarray(pixels[:, :, ::-1].transpose(2, 0, 1))).float() / 255)
        for line in label.read_text(encoding='utf-8').splitlines() if label.is_file() else []:
            if not line.strip():
                continue
            row = [float(v) for v in line.split()]
            require(len(row) == 5 and np.isfinite(row).all(), 'Expected finite YOLO detection labels: ' + str(label))
            require(row[0] >= 0 and row[0].is_integer() and all(0 <= v <= 1 for v in row[1:]) and
                    row[3] > 0 and row[4] > 0, 'Invalid normalized training label: ' + str(label))
            classes.append([row[0]]); boxes.append(row[1:]); indices.append(index)
    batch = dict(img=torch.stack(tensors).to(device), cls=torch.tensor(classes, device=device).reshape(-1, 1),
                 bboxes=torch.tensor(boxes, device=device).reshape(-1, 4),
                 batch_idx=torch.tensor(indices, device=device))
    return batch, dict(kind='real_training_examples',data=str(data_path),data_sha256=sha256(data_path),
                       files=[dict(image=str(image),image_sha256=sha256(image),label=str(label),
                                   label_sha256=sha256(label) if label.is_file() else None)
                              for image,label in zip(images,labels)],
                       transforms='deterministic RGB resize only; no augmentation; not a B16/640 capacity check')


def explicit_forward(model, batch):
    """Follow the native saved-layer graph and call the explicit diagnostic head API."""
    value, saved = batch['img'], []
    for layer in model.model[:-1]:
        if layer.f != -1:
            value = saved[layer.f] if isinstance(layer.f, int) else [value if j == -1 else saved[j] for j in layer.f]
        value = layer(value)
        saved.append(value if layer.i in model.save else None)
    head = model.model[-1]
    return head.forward_with_diagnostics([saved[j] for j in head.f], targets(batch))


def summary(value):
    value = value.detach().float().flatten()
    if value.numel() == 0:
        return dict(count=0,min=None,mean=None,max=None,p10=None,p50=None,p90=None)
    require(torch.isfinite(value).all(), 'Non-finite diagnostic statistic')
    quantiles = torch.quantile(value, value.new_tensor([.1,.5,.9])).cpu().tolist()
    return dict(count=value.numel(),min=float(value.min()),mean=float(value.mean()),max=float(value.max()),
                **dict(zip(('p10','p50','p90'),quantiles)))


def diagnose(checkpoint_path, output, data=None, device='cpu', backward=False, imgsz=160):
    output = Path(output)
    require(not output.exists(), 'Existing diagnostic report preserved: ' + str(output))
    require(imgsz >= 128 and imgsz % 32 == 0, 'Diagnostic imgsz must be >=128 and divisible by 32')
    require(device != 'cuda' or torch.cuda.is_available(), 'Requested CUDA unavailable')
    checkpoint_path = Path(checkpoint_path).resolve()
    digest = sha256(checkpoint_path)
    checkpoint = torch_load(checkpoint_path, map_location='cpu')
    key = 'ema' if checkpoint.get('ema') is not None else 'model'
    require(checkpoint.get(key) is not None, 'Checkpoint contains no model or EMA')
    model = deepcopy(checkpoint[key]).float().to(device)
    variant = 'cbr_lif_rcsq_v1' if hasattr(model.model[-1], 'cbr') else 'rcsq_v1'
    verify_model(model, variant, zero=False)
    # This copy can update local BN buffers and collect gradients; checkpoint bytes
    # and parameter values are never rewritten, optimized, or reinitialized.
    model.train();model.nc = model.model[-1].nc
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    parameters_before = {name: parameter.detach().clone() for name,parameter in model.named_parameters()}
    torch.manual_seed(42)
    if data is None:
        batch = synthetic_batch(device, groups=(0,3), shape=(imgsz,imgsz))
        source = dict(kind='synthetic',note='Random RGB fixture, not measured dataset performance or capacity')
    else:
        batch, source = training_examples(data, device, imgsz)
    require(batch['cls'].numel() == 0 or batch['cls'].max() < model.nc, 'Training labels exceed checkpoint classes')
    with (nullcontext() if backward else torch.no_grad()):
        predictions, details = explicit_forward(model,batch)
        detail = details.get('rcsq',details)
        loss = model.loss(batch,preds=predictions)[0] if backward else None
        if loss is not None:
            require(torch.isfinite(loss), 'Non-finite original loss');loss.backward()
    dn_count = predictions[-1]['dn_num_split'][0] if predictions[-1] is not None else 0
    groups = {}
    for name, section in (('DN',slice(0,dn_count)),('normal',slice(dn_count,None))):
        original = detail['query'][:,section].float().norm(dim=-1)
        correction = detail['delta'][:,section].norm(dim=-1)
        groups[name] = dict(null_weights=summary(detail['null_weights'][:,section]),
                            valid_points=summary(detail['valid_count'][:,section]),
                            valid_queries=int(detail['query_valid'][:,section].sum()),
                            original_query_norm=summary(original),correction_norm=summary(correction),
                            correction_to_query_norm=summary(correction/original.clamp_min(1e-12)))
    gradients = {}
    for name,parameter in model.model[-1].rcsq.named_parameters():
        grad = parameter.grad
        gradients[name] = dict(status='NOT_REQUESTED' if not backward else ('ABSENT' if grad is None else
                                    ('FINITE' if torch.isfinite(grad).all() else 'NONFINITE')),
                               norm=None if grad is None else float(grad.float().norm()),
                               nonzero=None if grad is None else bool(torch.count_nonzero(grad)))
    require(all(row['status'] != 'NONFINITE' for row in gradients.values()), 'Non-finite RCS-Q gradient')
    require(all(torch.equal(parameters_before[name],parameter) for name,parameter in model.named_parameters()),
            'Diagnostic unexpectedly changed a parameter')
    require(sha256(checkpoint_path) == digest, 'Checkpoint bytes changed during diagnosis')
    report = dict(status='PASSED',checkpoint=str(checkpoint_path),checkpoint_sha256=digest,checkpoint_component=key,
                  checkpoint_epoch=checkpoint.get('epoch'),
                  variant=variant,nc=model.nc,device=device,imgsz=imgsz,batch=2,source=source,runtime=runtime(),
                  query_counts=dict(DN=dn_count,normal=model.model[-1].num_queries),gt_groups=targets(batch)['gt_groups'],
                  groups=groups,out_proj_norm=float(model.model[-1].rcsq.out_proj.weight.norm()),
                  gradients=gradients,original_loss=None if loss is None else float(loss.detach()),
                  optimizer_steps=0,parameters_unchanged=True,checkpoint_unchanged=True,
                  formal_training='NOT_STARTED_BY_DIAGNOSTIC',final_test='NOT_RUN',features_saved=False,
                  note='Disposable training-mode diagnostic; optional backward is not an optimizer update.')
    write_json(output,report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,help='Read only two training images; omitted means a synthetic fixture')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--imgsz',type=int,default=160)
    parser.add_argument('--backward',action='store_true',help='Original loss/backward on a disposable copy, no optimizer step')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = diagnose(args.checkpoint,args.output,args.data,args.device,args.backward,args.imgsz)
    print(result['status'],str(args.output))
