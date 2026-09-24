"""Bounded PDS mechanism checks. No formal training and no full val/test."""
from __future__ import annotations
import argparse
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import torch
from pds_v1_common import *
from pds_v1_trainer import PDSTrainer, gradient_report, validate_resume
from pds_v1_deploy import strip_model, compare_outputs
from ultralytics.nn.modules.pds import (PDSHead, split_phases, scatter_phases, reference_grid, decode,
                                        labels, assign, dense_loss, ramp, xywh_to_xyxy)
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping
from ultralytics.utils.patches import torch_load


def batch(device="cpu"):
    return dict(img=torch.linspace(0, 1, 2*3*160*160, device=device).reshape(2, 3, 160, 160),
                bboxes=torch.tensor([[.38,.44,.32,.12],[.68,.70,.11,.23],[.46,.39,.29,.08]], device=device),
                cls=torch.zeros(3,1,device=device), batch_idx=torch.tensor([0.,0.,1.],device=device))


def expect_error(fn, classes=(ValueError, RuntimeError)):
    try:
        fn()
    except classes:
        return
    raise AssertionError("Expected rejection was not raised")


def scalar_iou(a, b, giou=False):
    ax = [a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2]
    bx = [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]
    inter = max(min(ax[2],bx[2])-max(ax[0],bx[0]),0)*max(min(ax[3],bx[3])-max(ax[1],bx[1]),0)
    union = a[2]*a[3]+b[2]*b[3]-inter
    iou = inter/(union+1e-7)
    enc = (max(ax[2],bx[2])-min(ax[0],bx[0]))*(max(ax[3],bx[3])-min(ax[1],bx[1]))
    return iou-(enc-union)/(enc+1e-7) if giou else iou


def mechanism():
    x = torch.arange(2*3*6*10).reshape(2,3,6,10)
    phases = split_phases(x)
    assert torch.equal(scatter_phases(phases), x)
    assert phases[0,1,0,0,0] == x[0,0,1,0]
    assert phases[0,2,0,0,0] == x[0,0,0,1]
    rng = torch.get_rng_state().clone()
    head = PDSHead()
    assert torch.equal(rng, torch.get_rng_state())
    assert len(head.state_dict()) == 11 and sum(p.numel() for p in head.parameters()) == 10821
    with torch.no_grad():
        z, r = head(torch.zeros(1,64,160,160), torch.zeros(1,256,80,80))
    assert z.shape == (1,1,160,160) and r.shape == (1,4,160,160)
    expect_error(lambda: head(torch.zeros(1,64,10,12),torch.zeros(1,256,6,6)))
    assert [ramp(e) for e in (0,5,6,20,200)] == [0,0,1/15,1,1]
    # Small two-image fixture: tiny conflicting GTs, clipped GT, sparse/dense n_j, empty image.
    b = dict(img=torch.zeros(2,3,16,20), cls=torch.zeros(6,1),
             batch_idx=torch.zeros(6),
             bboxes=torch.tensor([[.1,.1,.001,.001],[.1,.1,.001,.001],[.4,.5,.8,.8],
                                  [1.,.5,.4,.3],[.5,.5,0.,.2],[float("nan"),.5,.1,.1]]))
    z = torch.linspace(-3,1,40).reshape(2,1,4,5).requires_grad_()
    raw = torch.zeros(2,4,4,5,requires_grad=True)
    targets = labels(b)
    assert targets[0]["invalid"] == 2 and len(targets[1]["pixel"]) == 0
    original = b["bboxes"].clone()
    loss, stats = dense_loss(z,raw,b,details=True)
    pred, _ = decode(raw)
    ref = [0.,0.,0.]
    for bi, row in enumerate(stats["assignment"]):
        selected = row["selected"]
        all_ids = [i for ids in selected for i in ids]
        assert len(all_ids) == len(set(all_ids)) and all(len(ids)<=4 for ids in selected)
        m = max(sum(bool(ids) for ids in selected),1)
        terms = [0.,0.,0.]
        for j, ids in enumerate(selected):
            for i in ids:
                logit = float(z.flatten(1)[bi,i])
                p = 1/(1+math.exp(-logit))
                a, gt = pred[bi,i].tolist(), targets[bi]["xywh"][j].tolist()
                terms[0] += .25*(1-p)**2*math.log1p(math.exp(-logit))/len(ids)/m
                terms[1] += sum(abs(v-t) for v,t in zip(a,gt))/len(ids)/m
                terms[2] += (1-scalar_iou(a,gt,True))/len(ids)/m
        for i in torch.where(row["negative"])[0].tolist():
            logit = float(z.flatten(1)[bi,i])
            p = 1/(1+math.exp(-logit))
            terms[0] += .25*.75*p*p*math.log1p(math.exp(logit))/m
        ref = [a+t/2 for a,t in zip(ref,terms)]
    assert abs(float(loss)-(ref[0]+5*ref[1]+2*ref[2]))<2e-6
    assert torch.allclose(b["bboxes"], original, equal_nan=True)
    assert stats["assignment"][0]["stats"]["unmatched_gt"] >= 1
    assert stats["assignment"][0]["fallback"][:2] == [True,True]
    # A different first image must not change second-image background loss/assignment.
    bg = dict(img=b["img"][1:],bboxes=torch.empty(0,4),cls=torch.empty(0,1),batch_idx=torch.empty(0))
    bg_loss, bg_stats = dense_loss(z[1:],raw[1:],bg)
    assert bg_stats["assignment"][0] == stats["assignment"][1]["stats"]
    bg_loss.backward(retain_graph=True)
    assert raw.grad is None  # no fake regression update for an empty image
    loss.backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(raw.grad).all()
    # Cap selection and tie-breaking at all-zero IoU, >128 candidate pool.
    pts = reference_grid(16,16,"cpu")
    gt = dict(pixel=torch.tensor([[0.,0.,16.,16.]]),xywh=torch.tensor([[.5,.5,1.,1.]]),
              original_index=torch.tensor([0]),invalid=0)
    matched = assign(torch.zeros(256),torch.tensor([3.,3.,.01,.01]).expand(256,4),pts,gt,16,16,16,16)
    expected = [t*255//127 for t in range(128)]
    assert matched["pools"][0] == expected and matched["selected"][0] == expected[:4]
    bad = dict(b,cls=torch.ones(6,1))
    expect_error(lambda: labels(bad))
    bad = dict(b,bboxes=torch.ones(6,4,dtype=torch.long)*30)
    expect_error(lambda: labels(bad))
    # External CPU RNG and head state deterministic across different surrounding RNG.
    state = {k:v.clone() for k,v in head.state_dict().items()}
    torch.manual_seed(77)
    other = PDSHead()
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in state.items())
    return dict(status="PASS", phase_non_square=True, fine_grid_640=[160,160],
                parameter_count=10821, keys=11, independent_scalar_loss=ref,
                fixtures=["two_images","empty","thin_fallback","overlap_conflict","unequal_nj",
                          "zero_iou","clipping","invalid_skips","cap128","format_rejection"])


def rebuild(source):
    _, weights, source_report = controlled_models(source)
    rng = torch.get_rng_state().clone()
    native, native_report = build_training_model(weights.yaml, weights, dict(nc=1,channels=3))
    native.nc = 1
    torch.set_rng_state(rng)
    trainer = PDSTrainer.__new__(PDSTrainer)
    trainer.data = dict(nc=1,channels=3)
    wrapped = PDSTrainer.get_model(trainer, cfg=weights.yaml, weights=weights, verbose=False)
    audit = state_audit(native,wrapped)
    return native, wrapped, dict(audit, source_sha256=source_report["source_sha256"],
                                 native_adaptation=native_report)


def model_checks(source, device, folder):
    native, model, audit = rebuild(source)
    native, model = native.to(device).train(), model.to(device).train()
    b = batch(device)
    # Same DN RNG and same CPU arithmetic; CUDA nondeterminism is not asserted bitwise.
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    a, ai = native.loss(b)
    torch.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    model.pds_epoch = 5
    v, vi = model.loss(b)
    assert model.pds_head.calls == 0
    torch.testing.assert_close(a,v,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(ai,vi,rtol=1e-5,atol=1e-5)
    v.backward()
    assert all(p.grad is None for p in model.pds_head.parameters())
    del a, v, native
    model.zero_grad(set_to_none=True)
    model.pds_epoch = 20
    preds, features = model.training_forward(b["img"], None)
    del preds
    logits, raw = model.pds_head(*features)
    aux, stats = dense_loss(logits,raw,b)
    aux.backward()
    route = {}
    for name, index in (("P2",4),("P3_backbone",5),("AIFI",9),("Neck_P3",19),("LIF",20),("PAN",22),("decoder",26)):
        gs = [p.grad for p in model.model[index].parameters() if p.grad is not None]
        route[name] = dict(tensors=len(gs), norm=sum(float(g.float().square().sum()) for g in gs)**.5,
                           finite=all(bool(torch.isfinite(g).all()) for g in gs))
    assert all(route[k]["norm"]>0 and route[k]["finite"] for k in ("P2","P3_backbone","AIFI","Neck_P3"))
    assert all(route[k]["tensors"]==0 for k in ("LIF","PAN","decoder"))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pds_head.parameters())
    trainer = PDSTrainer.__new__(PDSTrainer)
    trainer.pds_effective_steps = 0
    optimizer = trainer.build_optimizer(model,name="AdamW",lr=.0005,momentum=.937,decay=.0001)
    old = model.pds_head.box.weight.detach().clone()
    old_p2 = next(model.model[4].parameters()).detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    optimizer.step()
    assert not torch.equal(old,model.pds_head.box.weight)
    assert not torch.equal(old_p2,next(model.model[4].parameters()))
    optimizer.zero_grad(set_to_none=True)
    total, _ = model.loss(b)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.model[20].parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.model[26].parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    optimizer.step()
    # A subsequent r=0 microbatch must not erase previously accumulated head gradients.
    previous = model.pds_head.box.weight.grad.clone()
    model.pds_epoch=5
    model.loss(b)[0].backward()
    assert torch.equal(previous,model.pds_head.box.weight.grad)
    del total, aux, features, logits, raw
    model.eval()
    calls = model.pds_head.calls
    with torch.no_grad():
        prediction = model(b["img"])
        model.loss(b,prediction)
    assert model.pds_head.calls == calls
    expect_error(lambda: model.train().loss(b,prediction) if setattr(model,"pds_epoch",20) is None else None)
    model.eval().cpu()
    deployed, deployment = strip_model(model)
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/"serialization.pt"
    torch.save(dict(model=model,deploy=deployed),path)
    # Separate interpreter proves the model class is importable without __main__ registration.
    script = ("import sys; sys.path.insert(0,'tools'); import pds_v1_common; "
              "from ultralytics.utils.patches import torch_load; "
              f"c=torch_load({str(path)!r},map_location='cpu'); "
              "assert len(c['model'].pds_head.state_dict())==11; "
              "assert not hasattr(c['deploy'],'pds_head'); print('PDS_SEPARATE_LOAD_OK')")
    result = subprocess.check_output([sys.executable,"-c",script],cwd=ROOT,text=True)
    assert "PDS_SEPARATE_LOAD_OK" in result
    write_json(folder/"public_state_audit.json",audit)
    write_json(folder/"optimizer_groups.json",trainer.pds_optimizer_groups)
    return dict(status="PASS",device=device,public_states=552,parameters=20160586,
                r0_equivalent=True,aux_gradient_route=route,total_gradient_route=True,
                true_optimizer_updates=trainer.pds_effective_steps,aux_and_p2_weights_changed=True,
                accumulated_gradient_retained=True,eval_calls=0,deploy_exact=deployment["raw_output_exact"],
                independent_process_load=True)


def run(source, device, folder):
    torch.set_num_threads(4)
    report=dict(runtime=environment(),server_B16_640="PENDING",formal_training_started=False)
    try:
        report["mechanism"]=mechanism()
        report["model"]=model_checks(source,device,folder)
        report["status"]="PASS"
    except BaseException as e:
        import traceback
        report.update(status="FAILED",error=repr(e),traceback=traceback.format_exc())
        raise
    finally:
        write_json(folder/"checks.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path,default=OUT/"local_checks")
    a=p.parse_args()
    print(json.dumps(run(a.source,a.device,a.output),ensure_ascii=False,indent=2))
