"""SHA-locked C2 initialization, semantic mapping and native nc80-to-nc1 adaptation."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import torch
from mincompat_v2 import (ROOT, MODEL_DIR, BASE_COMMIT, SOURCE_SHA256, VARIANTS, build, common_key,
                          new_keys, topology, require, sha256, write_json, runtime, verify_zero)
import ultralytics
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import RTDETRDecoder
from ultralytics.utils.patches import torch_load


def rebuild_nc1(model):
    trainer=object.__new__(RTDETRTrainer)
    trainer.data=dict(nc=1, channels=3)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return trainer.get_model(cfg=model.yaml, weights=model, verbose=False)


def rebuild_audit(source, target, variant, zero=True):
    topology(target,variant)
    if zero: verify_zero(target)
    before,after=source.state_dict(),target.state_dict()
    require(set(before)==set(after), "Native rebuild changed state keys")
    different={k for k in before if before[k].shape!=after[k].shape}
    index=topology(target,variant)["roles"]["decoder"]
    prefix=f"model.{index}."
    expected={prefix+"denoising_class_embed.weight",prefix+"enc_score_head.weight",prefix+"enc_score_head.bias"}
    expected|={prefix+f"dec_score_head.{i}.{suffix}" for i in range(target.model[-1].num_decoder_layers) for suffix in ("weight","bias")}
    require(different==(expected if source.model[-1].nc!=target.model[-1].nc else set()), "Unexpected class adaptation")
    require(all(torch.equal(v.cpu(),after[k].cpu()) for k,v in before.items() if k not in different),
            "Native trainer failed to preserve common/new learned tensors")
    return dict(CLASS_ADAPTATION=sorted(different), loaded_exact=len(before)-len(different), MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[])


def controlled_models(source, variant):
    require(sha256(source)==SOURCE_SHA256, "Wrong C2 initialization SHA256")
    ckpt=torch_load(source,map_location="cpu")
    require(ckpt.get("epoch")==-1 and all(ckpt.get(k) is None for k in
        ("ema","optimizer","scaler","updates","train_metrics","train_results","best_fitness")), "Source is trained")
    require(type(ckpt["model"].model[-1]) is RTDETRDecoder, "Source is not native C2")
    original=ckpt["model"].float().state_dict()
    reference,target=build("c2",80),build(variant,80)
    topo=topology(target,variant); mapping=topo["common_layer_mapping"]
    public,new=reference.state_dict(),target.state_dict(); extra=new_keys(target)
    require(set(original)==set(public) and all(v.shape==public[k].shape for k,v in original.items()), "Source state/shape mismatch")
    mapped={common_key(k,mapping) for k in public}
    require(set(new)==mapped|extra and not mapped&extra, "Unexplained target tensors")
    require(all(torch.equal(v,new[common_key(k,mapping)]) for k,v in public.items()), "New constructors consumed public RNG")
    reference.load_state_dict(original,strict=True)
    target.load_state_dict({**new,**{common_key(k,mapping):v for k,v in original.items()}},strict=True)
    reference1,target1=rebuild_nc1(reference),rebuild_nc1(target)
    adaptation=rebuild_audit(target,target1,variant)
    common=[]; target_states=target1.state_dict()
    for k,v in reference1.state_dict().items():
        t=common_key(k,mapping); max_abs=float((v-target_states[t]).abs().max())
        require(torch.equal(v,target_states[t]), f"Common nc1 mismatch: {k} -> {t}")
        common.append(dict(source=k,target=t,shape=list(v.shape),max_abs=max_abs))
    report=dict(status="passed", variant=variant, c2_commit=BASE_COMMIT, source_sha256=SOURCE_SHA256,
        parameters_unfused=sum(p.numel() for p in target1.parameters()), public_constructor_equal=True,
        COMMON=common, NEW=sorted(extra), CLASS_ADAPTATION=adaptation["CLASS_ADAPTATION"],
        MISSING=[], UNEXPECTED=[], SHAPE_MISMATCH=[], topology=topo, native_rebuild=adaptation,
        mapping="Validated semantic main-graph correspondence; C17 insertion shifts C2 layers 18..26 to 19..27; CSCEF semantic is the existing upsampled Y4",
        storage="FP32; C2 source half storage promoted exactly; native seed42 class adaptation")
    return reference1,target1,report


def initialize(source, output, variant):
    from pathlib import Path
    output=Path(output); require(not output.exists(), f"Preserve existing initialization: {output}")
    _,target,report=controlled_models(source,variant)
    target.eval()
    target.args={**DEFAULT_CFG_DICT,"model":str(MODEL_DIR/VARIANTS[variant]["yaml"]),"task":"detect"}
    target.task,target.pt_path="detect",str(output.resolve())
    report["runtime"]=runtime()
    ckpt=dict(epoch=-1,model=deepcopy(target).float(),train_args=target.args,
              date=datetime.now(timezone.utc).isoformat(),version=ultralytics.__version__,license="AGPL-3.0",
              mincompat_v2_provenance=report)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as f: torch.save(ckpt,f)
    reloaded=RTDETR(str(output)).model
    rebuild_audit(target,reloaded,variant)
    report.update(output=str(output.resolve()), output_sha256=sha256(output),reload_exact=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=VARIANTS,required=True)
    parser.add_argument("--source",required=True); parser.add_argument("--output",required=True); parser.add_argument("--report",required=True)
    args=parser.parse_args(); torch.set_num_threads(4)
    write_json(args.report,initialize(args.source,args.output,args.variant))
