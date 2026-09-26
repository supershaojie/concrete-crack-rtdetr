"""Bounded read-only train-only response probe. Does not optimize or update BN."""
from __future__ import annotations

from tcr_v1_core import *


def probe(weights, dataset, output, limit=64, device="0"):
    import cv2
    require(1 <= limit <= 64,"Probe at most 64 images")
    weights, dataset, output = Path(weights),Path(dataset),Path(output)
    report = dict(status="PENDING", scope="train-only, random P, zero O; not learned effectiveness or AP",
                  sampling_rule="lexicographic relative train-image paths, first at most 64; frozen before inference",
                  weights=file_identity(weights), limit=limit, batch=1,imgsz=640,augment=False)
    if not weights.is_file() or not (dataset/"images/train").is_dir():
        report["reason"]="Missing trained mother weights or train images"
        write_json(output/"probe.json",report); return report
    require(not (output/"probe.json").exists(),"Existing probe evidence preserved")
    files = sorted(p for p in (dataset/"images/train").rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})[:limit]
    require(files,"Empty train split")
    manifest = []
    for p in files:
        label = dataset/"labels/train"/p.relative_to(dataset/"images/train").with_suffix(".txt")
        require(label.is_file(),"Missing probe train label")
        manifest.append(dict(image=p.relative_to(dataset).as_posix(),image_sha256=sha256(p),label_sha256=sha256(label)))
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"frozen_images.json",manifest)
    with isolated_rng():
        from ultralytics import RTDETR
        mother = RTDETR(str(weights)).model.float().eval()
        parent.verify_model(mother)
        require(mother.model[-1].nc==1,"Trained mother nc1 required")
        ckpt = torch_load(weights,map_location="cpu")
        require(bool(ckpt.get("train_results")) or ckpt.get("epoch",-1)>=0,"Weight lacks trained provenance")
        target = build().float().eval()
        state = target.state_dict(); public = mother.state_dict()
        require(set(state)-set(public)==NEW and not set(public)-set(state),"Mother weight inventory mismatch")
        target.load_state_dict({**state,**public},strict=True)
        verify_model(target,zero=True)
        dev = torch.device("cuda:0" if str(device)=="0" else device)
        target.to(dev)
        initial = {k:v.clone() for k,v in target.state_dict().items()}
        records=[]
        # Traverse the native saved graph only up to node 17; no decoder, GT loss or test access.
        with torch.no_grad():
            for p in files:
                im=cv2.imread(str(p)); require(im is not None,"Unreadable probe image")
                x=torch.from_numpy(cv2.resize(im,(640,640))[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float().to(dev)/255
                saved=[]
                for layer in target.model[:18]:
                    if layer.f != -1:
                        x=saved[layer.f] if isinstance(layer.f,int) else [x if j==-1 else saved[j] for j in layer.f]
                    if layer.i==17:
                        layer.tcr.capture=True
                    x=layer(x); saved.append(x if layer.i in target.save else None)
                stats=target.model[17].tcr.last_stats
                stats.update(image=p.relative_to(dataset).as_posix())
                append_json(output/"probe_samples.jsonl",stats); records.append(stats)
        require(all(torch.equal(v,target.state_dict()[k]) for k,v in initial.items()),"Read-only probe updated parameters/BN")
        total=sum(g["nonzero"] for r in records for g in r["groups"])
        report.update(status="PASS" if total else "NO_RESPONSE_FOUND",images=len(records),nonzero_elements=total,
                      finite=all(math.isfinite(r["z_rms"]) and all(math.isfinite(g["rms"]) for g in r["groups"]) for r in records),
                      bn_unchanged=True,optimized=False,weights_sha256=sha256(weights),manifest_sha256=sha256(output/"frozen_images.json"),
                      group_summary=[dict(direction=(0,45,90,135)[i//2],side_step=i%2+1,
                                          **{k:sum(r["groups"][i][k] for r in records) for k in ("elements","nonzero","positive","negative","sum_squares")}) for i in range(8)])
        if not report["finite"]: report.update(status="FAILED",reason="Nonfinite actual-feature response")
    write_json(output/"probe.json",report)
    return report
