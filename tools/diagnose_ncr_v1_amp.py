"""Bounded investigation of a scaled-gradient failure: ONE real batch, no optimizer step."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from ncr_v1_common import NCR, paths, write
from init_c19_lif_v1 import controlled_models, build_training_model
from c19_lif_v1_data import real_batch
from ultralytics.models.utils.ncr import configure_ncr, ncr_terms
from ultralytics.utils.torch_utils import init_seeds


def probe(main, output):
    torch.set_num_threads(4); init_seeds(42, deterministic=True)
    _, initialized, _ = controlled_models(paths(main)["source"])
    model, _ = build_training_model(initialized.yaml, initialized, dict(nc=1,channels=3))
    del initialized
    model.cuda().train(); model.nc=1
    batch,_=real_batch(Path(main)/"datasets/crack_det",size=640,count=16)
    batch={k:v.cuda() for k,v in batch.items()}
    targets=dict(cls=batch["cls"].long().reshape(-1),bboxes=batch["bboxes"],batch_idx=batch["batch_idx"].long(),
                 gt_groups=[int((batch["batch_idx"]==i).sum()) for i in range(16)])
    with torch.autocast("cuda"):
        preds=model.predict(batch["img"],batch=targets)
        base,_=model.loss(batch,preds)
        configure_ncr(model,NCR); model.ncr_epoch=19
        total,items=model.loss(batch,preds)
        regular=preds[0][-1,:,preds[4]["dn_num_split"][0]:]
        regular_scores=preds[1][-1,:,preds[4]["dn_num_split"][0]:]
        matches=model.criterion.matcher(regular.contiguous(),regular_scores.contiguous(),targets["bboxes"],targets["cls"],targets["gt_groups"])
        idx,gi=model.criterion._get_index(matches)
        extra=.25*ncr_terms(regular[idx],targets["bboxes"][gi],(640,640))[0]
    report=dict(scope="one B16/640 AMP forward, zero optimizer steps, scaled-gradient diagnosis only", loss=float(total.detach()),
                weighted_ncr=float(extra.detach()), cases={})
    named=list(model.named_parameters())
    for label,value,scale in (("parent_65536",base,65536.),("ncr_term_65536",extra,65536.),("total_parent_smoke_scale128",total,128.)):
        grads=torch.autograd.grad(value*scale,[p for n,p in named],allow_unused=True,retain_graph=True)
        bad=[n for (n,p),g in zip(named,grads) if g is not None and not bool(torch.isfinite(g).all())]
        report["cases"][label]=dict(finite=not bad,nonfinite_count=len(bad),first_nonfinite=bad[:10])
        del grads
        write(output,report)
    print(report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--main",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); probe(args.main,args.output)
