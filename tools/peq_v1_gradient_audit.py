"""Named same-forward gradient audit. Diagnostic precision/RNG always restore; no training."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import traceback
from peq_v1_common import *
from init_peq_v1 import build
from ultralytics.models.rtdetr.peq_train import parameter_partition, is_peq
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.peq_loss import PEQDetectionLoss, quality_targets


def fingerprint(value):
    digest=hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            item=item.detach().cpu().contiguous()
            digest.update(str((str(item.dtype),tuple(item.shape))).encode())
            digest.update(item.numpy().tobytes())
        elif isinstance(item,np.ndarray):
            digest.update(str((str(item.dtype),item.shape)).encode());digest.update(item.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):
                digest.update(repr(key).encode());visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode())
            for part in item:visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),
                cuda=[state.clone() for state in torch.cuda.get_rng_state_all()])


def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:torch.cuda.set_rng_state_all(state["cuda"])


def precision_state():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cuda_autocast=torch.is_autocast_enabled(),cpu_autocast=torch.is_autocast_cpu_enabled(),
                cuda_autocast_dtype=str(torch.get_autocast_gpu_dtype()),cpu_autocast_dtype=str(torch.get_autocast_cpu_dtype()))


@contextmanager
def diagnostic_precision(disable_tf32):
    before=precision_state()
    try:
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
        with ExitStack() as contexts:
            contexts.enter_context(torch.autocast("cpu",enabled=False))
            if torch.cuda.is_available():contexts.enter_context(torch.autocast("cuda",enabled=False))
            yield precision_state()
    finally:
        torch.backends.cuda.matmul.allow_tf32=before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32=before["cudnn_tf32"]
        torch.set_float32_matmul_precision(before["matmul_precision"])
        require(precision_state()==before,"Diagnostic precision state did not restore")


def gradient_snapshot(model):
    result={name:None if p.grad is None else p.grad.detach().clone()
            for name,p in model.named_parameters() if not is_peq(name)}
    require(all(value is None or value.data_ptr()!=dict(model.named_parameters())[name].grad.data_ptr()
                for name,value in result.items()),"Reference gradient aliases live .grad")
    return result


def differences(reference, actual, atol, rtol):
    rows={}
    require(reference.keys()==actual.keys(),"Gradient parameter inventory differs")
    for name,ref in reference.items():
        got=actual[name]
        if ref is None or got is None:
            rows[name]=dict(reference_none=ref is None,actual_none=got is None,pass_fixed=ref is None and got is None)
            continue
        require(torch.isfinite(ref).all() and torch.isfinite(got).all(),f"Nonfinite gradient: {name}")
        delta=(got-ref).abs()
        mismatch=delta>atol+rtol*ref.abs()
        index=int(delta.flatten().argmax())
        rows[name]=dict(shape=list(ref.shape),elements=ref.numel(),max_abs=float(delta.max()),
                        error_l2=float((got-ref).float().norm()),reference_l2=float(ref.float().norm()),
                        reference_max_abs=float(ref.abs().max()),mismatched_elements=int(mismatch.sum()),
                        pass_fixed=not bool(mismatch.any()),max_abs_flat_index=index,
                        reference_at_max=float(ref.flatten()[index]),actual_at_max=float(got.flatten()[index]),
                        atol=atol,rtol=rtol)
    return rows


def compare_snapshots(reference, actual, repeat_rows, device):
    # Preserve the old uncut CUDA bounds; the clipped bound is UNCHANGED and fixed.
    raw=differences(reference["raw"],actual["raw"],2e-6 if device.type=="cpu" else 2e-5,
                    2e-5 if device.type=="cpu" else 3e-4)
    for name,row in raw.items():
        row["pass_existing_uncut_policy"]=row["pass_fixed"]
        if device.type=="cuda" and "error_l2" in row:
            repeat=repeat_rows[name]
            bound=max(4*repeat["error_l2"],1e-6*row["elements"]**.5+2e-5*row["reference_l2"])
            atol=max(2e-5,4*repeat["max_abs"],2e-5*row["reference_max_abs"])
            ref,got=reference["raw"][name],actual["raw"][name]
            row.update(existing_l2_bound=bound,existing_element_atol=atol,
                       pass_existing_uncut_policy=row["error_l2"]<=bound and
                       bool(torch.all((got-ref).abs()<=atol+3e-4*ref.abs())))
    clipped=differences(reference["clipped"],actual["clipped"],2e-5,3e-4)
    failed_raw=[n for n,r in raw.items() if not r["pass_existing_uncut_policy"]]
    failed_clipped=[n for n,r in clipped.items() if not r["pass_fixed"]]
    return dict(raw=raw,clipped=clipped,failed_raw=failed_raw,failed_clipped=failed_clipped,
                passed=not failed_raw and not failed_clipped)


class MatcherRecorder(torch.nn.Module):
    def __init__(self,inner):
        super().__init__();self.inner=inner;self.calls=[]
    def forward(self,boxes,scores,*args,**kwargs):
        indices=self.inner(boxes,scores,*args,**kwargs)
        self.calls.append(dict(boxes=boxes.detach().clone(),scores=scores.detach().clone(),
                               indices=[(a.detach().clone(),b.detach().clone()) for a,b in indices]))
        return indices


def profile(model,batch,report,flush,device):
    report["precision"]=precision_state()
    boundary={};decoder_inputs={}
    def capture_boundary(module,args):
        boundary.update(zip(("F","q","b0","b1","z"),args))
    def capture_decoder(module,args,kwargs):
        decoder_inputs.update(embed=args[0].detach().clone(),reference=args[1].detach().clone(),
                              attention_mask=kwargs.get("attn_mask"))
    head=model.model[-1]
    hooks=[head.peq.register_forward_pre_hook(capture_boundary),
           head.decoder.register_forward_pre_hook(capture_decoder,with_kwargs=True)]
    before_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    before_rng=fingerprint(rng_state())
    targets=dict(cls=batch["cls"].long().flatten(),bboxes=batch["bboxes"],batch_idx=batch["batch_idx"],gt_groups=[2,0])
    try:
        raw=model.predict(batch["img"],batch=targets)
    finally:
        for hook in hooks:hook.remove()
    dboxes,dscores,eboxes,escores,dn,payload=raw
    dnbox,dboxes=torch.split(dboxes,dn["dn_num_split"],dim=2)
    dnscore,dscores=torch.split(dscores,dn["dn_num_split"],dim=2)
    preds=(torch.cat((eboxes[None],dboxes)),torch.cat((escores[None],dscores)))
    c0=RTDETRDetectionLoss(nc=1,use_vfl=True);c1=PEQDetectionLoss(nc=1,use_vfl=True)
    c0.matcher=MatcherRecorder(c0.matcher);c1.matcher=MatcherRecorder(c1.matcher)
    a=c0(preds,targets,dnbox,dnscore,dn)
    b=c1(preds,targets,dnbox,dnscore,dn,payload)
    require(list(b)==list(a)+["loss_peq"],"Changed native loss insertion order")
    report["loss_components"]={k:dict(native=float(v.detach()),peq_native=float(b[k].detach()),
                                      exact=torch.equal(v,b[k])) for k,v in a.items()}
    require(all(row["exact"] for row in report["loss_components"].values()),"Native loss components differ")
    calls0,calls1=c0.matcher.calls,c1.matcher.calls
    require(len(calls0)==len(calls1)==4,"Final/encoder/aux matcher call count changed")
    report["matcher_calls"]=[]
    for i,(left,right) in enumerate(zip(calls0,calls1)):
        equal=fingerprint(left)==fingerprint(right)
        report["matcher_calls"].append(dict(index=i,role="final" if i==0 else "encoder/aux",exact=equal,
                                             native_sha256=fingerprint(left),peq_sha256=fingerprint(right),
                                             indices=[(x.cpu().tolist(),y.cpu().tolist()) for x,y in left["indices"]]))
        require(equal,f"Matcher inputs OR assignments differ at call {i}")
    after_state=model.state_dict()
    report["forward"]=dict(count=1,shared_raw=True,parameter_unchanged=all(torch.equal(before_state[n],p) for n,p in model.named_parameters()),
                          bn_buffers_updated_once=[n for n,v in after_state.items() if not torch.equal(before_state[n],v)],
                          rng_before=before_rng,rng_after=fingerprint(rng_state()),dn_split=dn["dn_num_split"],
                          boundary_requires_grad={k:v.requires_grad for k,v in boundary.items()},
                          nonzero_peq_output=bool(head.peq.output.weight.count_nonzero()))
    require(report["forward"]["parameter_unchanged"],"Forward mutated weights")
    del before_state
    def conditions():
        return dict(parameters=fingerprint(dict(model.named_parameters())),
                    buffers=fingerprint(dict(model.named_buffers())),input=fingerprint(batch),
                    modes=fingerprint({n:m.training for n,m in model.named_modules()}),
                    rng={k:fingerprint(v) for k,v in rng_state().items()},
                    dn_inputs=fingerprint(decoder_inputs),dn_meta=fingerprint(dn),
                    raw=fingerprint(raw[:4]),precision=precision_state())
    common=conditions()
    report["conditions"]=common
    report["paths"]={}
    flush()

    def backward(name,loss):
        model.zero_grad(set_to_none=True)
        require(all(p.grad is None for p in model.parameters()),f"Residual gradient before {name}")
        before=conditions()
        require(before==common,f"Conditions changed before backward {name}")
        loss.backward(retain_graph=True)
        original,added=parameter_partition(model)
        raw_grads=gradient_snapshot(model)
        raw_hash=fingerprint(raw_grads)
        added_grads={n:None if p.grad is None else float(p.grad.detach().norm())
                     for n,p in model.named_parameters() if is_peq(n)}
        norm=torch.nn.utils.clip_grad_norm_(original,10.)
        added_norm=torch.nn.utils.clip_grad_norm_(added,10.)
        clipped=gradient_snapshot(model)
        require(fingerprint(raw_grads)==raw_hash,f"Clipping mutated raw reference: {name}")
        after=conditions()
        require(after==common,f"Conditions changed during backward {name}")
        report["paths"][name]=dict(loss=float(loss.detach()),conditions_before=before,conditions_after=after,
                                   original_preclip_norm=float(norm),original_clip_coefficient=min(1.,10./(float(norm)+1e-6)),
                                   peq_preclip_norm=float(added_norm),peq_gradient_norms=added_grads,
                                   raw_sha256=raw_hash,clipped_sha256=fingerprint(clipped),
                                   independent_clones=True,raw_reference_unchanged_after_clip=True)
        if name in ("native_reference","native_repeat","native_peq_criterion"):
            require(all(v is None for v in added_grads.values()),f"{name} trains PEQ")
        flush()
        return dict(raw=raw_grads,clipped=clipped)

    native=sum(a.values())
    peq_native=sum(v for k,v in b.items() if k!="loss_peq")
    reference=backward("native_reference",native)
    reference_hashes={k:fingerprint(v) for k,v in reference.items()}
    repeated=backward("native_repeat",native)
    repeat_rows=differences(reference["raw"],repeated["raw"],2e-5,3e-4)
    report["comparisons"]={"native_self":compare_snapshots(reference,repeated,repeat_rows,device)}
    del repeated
    native_new=backward("native_peq_criterion",peq_native)
    report["comparisons"]["native_criteria"]=compare_snapshots(reference,native_new,repeat_rows,device)
    del native_new
    combined=backward("native_plus_quality",peq_native+b["loss_peq"])
    report["comparisons"]["native_plus_quality"]=compare_snapshots(reference,combined,repeat_rows,device)
    del combined
    require({k:fingerprint(v) for k,v in reference.items()}==reference_hashes,"Saved reference was mutated across paths")

    model.zero_grad(set_to_none=True)
    require(all(v.requires_grad for v in boundary.values()),"Detach boundary fixture is not live")
    input_gradients=torch.autograd.grad(b["loss_peq"],tuple(boundary.values()),allow_unused=True,retain_graph=True)
    boundary_leaks=[n for n,g in zip(boundary,input_gradients) if g is not None]
    require(not boundary_leaks,f"L_Q crosses detach boundary: {boundary_leaks}")
    b["loss_peq"].backward(retain_graph=True)
    original_names=[n for n,p in model.named_parameters() if not is_peq(n)]
    leaked=[n for n,p in model.named_parameters() if not is_peq(n) and p.grad is not None]
    new_norms={n:None if p.grad is None else float(p.grad.detach().norm()) for n,p in model.named_parameters() if is_peq(n)}
    report["quality_only"]=dict(original_parameters_checked=len(original_names),original_parameters_with_grad=leaked,
                                 boundary_grad_none={n:g is None for n,g in zip(boundary,input_gradients)},
                                 peq_gradient_norms=new_norms,conditions_after=conditions())
    require(not leaked,f"L_Q directly trains original parameters: {leaked}")
    require(conditions()==common,"Quality-only backward changed comparison conditions")
    require(all(v is not None and math.isfinite(v) and v>0 for v in new_norms.values()),"Nonzero PEQ fixture cannot learn")
    model.zero_grad(set_to_none=True)
    # Retain the prior real G>Q matcher/target check.
    gt=torch.rand(7,4,device=device)*.5+.2
    assignment=c1.matcher.inner(torch.rand(2,3,4,device=device)*.5+.2,torch.zeros(2,3,1,device=device),
                                gt,torch.zeros(7,dtype=torch.long,device=device),[5,2])
    _,mask=quality_targets(torch.ones(2,3,4,device=device)*.5,gt,assignment,[5,2])
    require(int(mask.sum())==5,"G>Q matching/target count")
    report["g_greater_q"]=True
    report["passed_numeric"]=all(v["passed"] for v in report["comparisons"].values())
    report["failed_parameters"]={k:dict(raw=v["failed_raw"],clipped=v["failed_clipped"])
                                  for k,v in report["comparisons"].items()}
    flush()


def audit_gradients(device="cpu",evidence_dir=None):
    device=torch.device(device)
    root=Path(evidence_dir) if evidence_dir else OUT
    folder=root/"gradient_checks"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+"_"+device.type)
    folder.mkdir(parents=True)
    path=folder/"report.json"
    entry_precision=precision_state();entry_rng=fingerprint(rng_state())
    report=dict(status="FAILED",scope="BOUNDED_GRADIENT_INTERFACE_AUDIT",environment=environment(),
                source=source_identity()["sha256"],started=utc(),profiles={},entry_precision=entry_precision,
                reported_failure_parameter="model.0.conv.weight",reported_failure_shape=[32,3,3,3],
                tolerance_policy=dict(clipped_atol=2e-5,clipped_rtol=3e-4,unchanged=True,
                                      uncut_cuda="Existing native/native noise and vector bounds, unchanged"),
                server_B16_640_preflight="PENDING")
    def flush():write_json(path,report)
    print("Gradient evidence:",path,flush=True)
    try:
        with isolated_rng():
            torch.manual_seed(42)
            model=build(1).to(device).train()
            batch=dict(img=torch.rand(2,3,160,160,device=device),
                       bboxes=torch.tensor([[.5,.5,.35,.2],[.3,.4,.15,.3]],device=device),
                       cls=torch.zeros(2,1,device=device),batch_idx=torch.zeros(2,device=device,dtype=torch.long))
            # Exercise the detach graph with a learned/nonzero PEQ, not only its zero output layer.
            with torch.no_grad():model.model[-1].peq.output.weight.normal_(std=.01)
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            shared_rng=rng_state()
            report["fixture"]=dict(parameters=fingerprint(state),batch=fingerprint(batch),rng=fingerprint(shared_rng),
                                   nonzero_peq_output=True,batch_shape=list(batch["img"].shape))
            profiles=[("ambient_fp32",False),("controlled_fp32_no_tf32",True)] if device.type=="cuda" else [("cpu_fp32",False)]
            for name,disable_tf32 in profiles:
                model.load_state_dict(state,strict=True);restore_rng(shared_rng)
                item=report["profiles"][name]={}
                with diagnostic_precision(disable_tf32):
                    profile(model,batch,item,flush,device)
                print("GRADIENT",name,"numeric",item["passed_numeric"],"failures",item["failed_parameters"],flush=True)
            strict=report["profiles"][profiles[-1][0]]
            require(strict["passed_numeric"],f"Strict gradient audit failed: {strict['failed_parameters']}; evidence={path}")
            # Ambient noise is recorded, never described as a numerical PASS when it failed.
            if device.type=="cuda":
                ambient=report["profiles"]["ambient_fp32"]
                if not ambient["passed_numeric"]:
                    require(not ambient["comparisons"]["native_self"]["passed"],
                            f"Ambient mixed-path failure without native-self failure remains unexplained; evidence={path}")
                    report["ambient_interpretation"]="Native self also fails the unchanged bound; strict locally scoped FP32 passes. Numerical sensitivity, not an ambient PASS."
                else:
                    report["ambient_interpretation"]="All ambient native-self and mixed-path checks pass on this device."
            report["status"]="PASS"
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished=utc(),exit_precision=precision_state(),rng_restored=fingerprint(rng_state())==entry_rng)
        report["precision_restored"]=report["exit_precision"]==entry_precision
        if not report["precision_restored"] or not report["rng_restored"]:report["status"]="FAILED"
        flush()
        require(report["precision_restored"] and report["rng_restored"],f"Gradient diagnostic leaked state; evidence={path}")
    return dict(status=report["status"],report=identity(path),profiles={k:dict(passed_numeric=v["passed_numeric"],
                failed_parameters=v["failed_parameters"]) for k,v in report["profiles"].items()},
                isolated_quality_backward=True,strict_partitioned_clipping_equal=True,
                precision_restored=True,rng_restored=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--evidence-dir",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    print(json_bytes(audit_gradients(args.device,args.evidence_dir)).decode())
