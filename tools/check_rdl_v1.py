"""RDL 有界数学/真实模型/生命周期检查；--help 查看选项，不启动正式训练。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from init_c19_lif_v1 import ROOT, build, controlled_models, native_rebuild, verify_model, require, write_json
from ultralytics.models.utils.rdl import CONFIG, RDLDetectionLoss, decompose, sides, ramp
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load
from rdl_v1_training import RDLTrainer, configure, epoch_start

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"


def close(a, b, atol=2e-6, rtol=2e-6):
    torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b), atol=atol, rtol=rtol, check_dtype=False)


def reference_loss_class():
    """Execute the actual pinned mother criterion, not a reimplementation of L0."""
    raw = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "show",
                                   BASE + ":ultralytics-main/ultralytics/models/utils/loss.py"], cwd=ROOT)
    name = "ultralytics.models.utils._rdl_pinned_reference"
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    exec(compile(raw, BASE + ":loss.py", "exec"), module.__dict__)
    return module.RTDETRDetectionLoss


def mathematical():
    b = torch.tensor([[50., 50., 40., 40.]], requires_grad=True)
    v = torch.zeros_like(b, requires_grad=True)
    out, inside, ev = decompose(b, b.detach(), v, (100, 200))
    close(out, 0); close(inside, 0)
    close(sides(torch.tensor([[.5, .4, .2, .6]])), [[.4, .6, .1, .7]])
    # Move one GT side at a time, independently reconstruct GT xywh from L,R,T,B.
    for side in range(4):
        for sign in (-1, 1):
            for error in (3., 11.):
                gs = sides(b.detach()).clone(); gs[0, side] += sign * error
                g = torch.stack(((gs[:, 0]+gs[:, 1])/2, (gs[:, 2]+gs[:, 3])/2,
                                 gs[:, 1]-gs[:, 0], gs[:, 3]-gs[:, 2]), -1).requires_grad_()
                out, inside, ev = decompose(b, g, v, (100, 200))
                close(ev["vstar"][0, side], sign * min(error / 4, 1))
                close(ev["eta"][0, side], min(1, 4/error))
                close(out, (1.25 if error == 11 else 0)/4)
                close(inside, (2/11 if error == 11 else .5*.75**2)/4)
                gv, gg = torch.autograd.grad(inside, (v, g), retain_graph=True, allow_unused=True)
                require(gv[0, side] * sign < 0 and gg is None, "Offset direction or detached GT wrong")
                close(torch.autograd.grad(inside, b, retain_graph=True)[0], torch.zeros_like(b), 0, 0)
                require(all(not ev[k].requires_grad for k in ("a", "ebar", "vstar", "eta", "S")), "Detached labels")
                if error == 11:
                    gb = torch.autograd.grad(out, b, retain_graph=True)[0]
                    # Fixed detached labels for finite difference; never reconstruct a or S.
                    def frozen_out(bb):
                        z = ((gs - sides(bb)).abs() - ev["a"]).relu() / ev["S"]
                        return torch.where(z <= 1, .5*z*z, z-.5).sum()/4
                    for j in range(4):
                        plus, minus = b.detach().clone(), b.detach().clone()
                        plus[0, j] += .01; minus[0, j] -= .01
                        numeric = (frozen_out(plus)-frozen_out(minus))/.02
                        close(gb[0, j], numeric, atol=2e-5, rtol=3e-4)
                    require(gb[0, 0 if side < 2 else 1] * sign < 0, "Outside gradient away from GT")
    tiny = torch.tensor([[.5, .5, 1e-8, 2e-8]], requires_grad=True)
    out, inside, ev = decompose(tiny, tiny.detach()+torch.tensor([[3e-7, 0., 0., 0.]]), torch.zeros_like(tiny), (320, 640))
    close(ev["S"], [[1/640, 1/640, 1/320, 1/320]])
    require(ev["a_protected"].sum() == 4 and ev["e_protected"].sum() == 2, "Protection counts wrong")
    empty = torch.empty(0, 4, requires_grad=True)
    z1, z2, _ = decompose(empty, empty.detach(), empty, (320, 640))
    require(z1.dtype == torch.float32 and z1.device == empty.device, "Empty device/dtype")
    (z1+z2).backward(); require(empty.grad is not None, "Empty loss disconnected")
    for bad in (torch.tensor([[0., 0., -1., 1.]]), torch.tensor([[float('nan'), 0., 1., 1.]])):
        try:
            decompose(bad, b.detach(), v, (100, 200))
        except (ValueError, FloatingPointError):
            pass
        else:
            raise AssertionError("Illegal geometry was hidden")
    require([ramp(e) for e in (0, 5, 6, 19, 20)] == [0, 0, 1/15, 14/15, 1], "Schedule off by one")
    return dict(status="PASS", side_sign_cases=16, finite_difference="fixed detached labels, atol=2e-5 rtol=3e-4")


class CountingMatcher(torch.nn.Module):
    def __init__(self, matcher):
        super().__init__(); self.matcher = matcher; self.calls = []

    def forward(self, boxes, scores, *args, **kwargs):
        result = self.matcher(boxes, scores, *args, **kwargs)
        self.calls.append((boxes.detach().clone(), [(i.clone(), j.clone()) for i, j in result]))
        return result


def indexing():
    reference = reference_loss_class()
    cases = []
    for groups in ([1, 2], [0, 0], [0, 1]):
        for dn in (False, True):
            torch.manual_seed(42)
            b = torch.rand(4, 2, 5, 4) * .4 + .25
            s = torch.randn(4, 2, 5, 1)
            gt = torch.tensor([[.3, .4, .2, .1], [.6, .3, .3, .2], [.4, .65, .1, .25]])[:sum(groups)]
            batch = dict(bboxes=gt, cls=torch.zeros(len(gt), dtype=torch.long), gt_groups=groups)
            details = dict(before=(b[-1]-.01).requires_grad_(), after=b[-1],
                           tanh_offsets=torch.full_like(b[-1], .2, requires_grad=True))
            meta = dict(dn_pos_idx=[torch.arange(n) for n in groups], dn_num_group=1) if dn else None
            db, ds = (b[1:].clone(), s[1:].clone()) if dn else (None, None)
            old, new = reference(nc=1, use_vfl=True), RDLDetectionLoss(nc=1, use_vfl=True, config=CONFIG)
            old.matcher, new.matcher = CountingMatcher(old.matcher), CountingMatcher(new.matcher)
            new.set_epoch(20)
            try:
                new((b,s),batch,enabled=True,image_hw=(320,640))
            except RuntimeError as error:
                require('same forward' in str(error),'Wrong missing-details failure')
            else:
                raise AssertionError('Active RDL silently accepted missing details')
            l0 = old((b, s), batch, db, ds, meta)
            losses = new((b, s), batch, db, ds, meta, details=details, image_hw=(320, 640), enabled=True)
            require(set(losses) == set(l0) | {"loss_rdl"}, "RDL leaked into auxiliary/DN losses")
            for k, val in l0.items(): close(losses[k], val, 0, 0)
            require(len(old.matcher.calls) == len(new.matcher.calls) == 4, "Extra/missing Hungarian calls")
            for left, right in zip(old.matcher.calls, new.matcher.calls):
                close(left[0], right[0], 0, 0)
                for a, c in zip(left[1], right[1]):
                    require(all(torch.equal(x, y) for x, y in zip(a, c)), "Matcher identities differ")
            indices = new.matcher.calls[0][1]
            # Independent scalar Python reference, including batch-flat GT offset.
            expected, offset = 0., 0
            for i, ((src, dst), n) in enumerate(zip(indices, groups)):
                require(all(offset <= int(j) < offset+n for j in dst), "GT batch offset wrong")
                offset += n
                for q, j in zip(src.tolist(), dst.tolist()):
                    bb, gg, vv = details["before"][i, q].tolist(), gt[j].tolist(), details["tanh_offsets"][i, q].tolist()
                    x = [bb[0]-bb[2]/2, bb[0]+bb[2]/2, bb[1]-bb[3]/2, bb[1]+bb[3]/2]
                    g = [gg[0]-gg[2]/2, gg[0]+gg[2]/2, gg[1]-gg[3]/2, gg[1]+gg[3]/2]
                    for edge in range(4):
                        a = .1 * bb[2 if edge < 2 else 3]; e = g[edge]-x[edge]
                        target = min(1, max(-1, e/a)); eta = min(1, a/abs(e)) if e else 1
                        o = max(0, abs(e)-a)/max(a, 1/(640 if edge < 2 else 320))
                        huber = lambda z: .5*z*z if abs(z) <= 1 else abs(z)-.5
                        expected += huber(o) + eta*huber(vv[edge]-target)
            close(losses["loss_rdl"], .05*expected/(4*max(sum(groups), 1)))
            if sum(groups): require(losses["loss_rdl"] > 0, "Inactive loss")
            cases.append(dict(groups=groups, DN=dn, matches=sum(groups), matcher_calls=len(new.matcher.calls)))
    return dict(status="PASS", cases=cases, original_losses="exactly equal to pinned criterion")


def targets(batch):
    return dict(cls=batch["cls"].long().view(-1), bboxes=batch["bboxes"], batch_idx=batch["batch_idx"].long(),
                gt_groups=[int((batch["batch_idx"] == i).sum()) for i in range(len(batch["img"]))])


def split_prediction(raw, details=None):
    b, s, eb, es, dn = raw
    db = ds = None
    if dn is not None:
        db, b = b.split(dn["dn_num_split"], dim=2); ds, s = s.split(dn["dn_num_split"], dim=2)
        if details is not None: details = {k: v.split(dn["dn_num_split"], dim=1)[1] for k, v in details.items()}
    return (torch.cat((eb[None], b)), torch.cat((es[None], s))), db, ds, dn, details


def model_checks(source, device, fusion_dir=None):
    torch.set_num_threads(4)
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(42, deterministic=True)  # same deterministic/warn-only policy as the real Trainer
    _, init, init_audit = controlled_models(source)
    torch.manual_seed(71)
    rng = torch.get_rng_state()
    base = native_rebuild(deepcopy(init.yaml), init, 1, 3)
    torch.set_rng_state(rng)
    trainer = RDLTrainer.__new__(RDLTrainer)
    trainer.data, trainer.resume = dict(nc=1, channels=3), False
    new = trainer.get_model(deepcopy(init.yaml), init, False)
    base.nc = new.nc = trainer.data["nc"]  # native Trainer.set_model_attributes before training
    require(len(trainer.rdl_loading_audit["ALLOWED_CLASS_ADAPTATION"]) == 9, "Expected 9 adapted keys")
    require(base.state_dict().keys() == new.state_dict().keys(), "State names changed")
    for k, value in base.state_dict().items(): close(value, new.state_dict()[k], 0, 0)
    del init
    # grid_sample CUDA backward uses atomics even under native deterministic=True.
    # Exact optimizer equivalence is checked on CPU; CUDA separately checks the
    # same-forward predictions/L0 plus FP32/AMP gradients, never a widened update tolerance.
    base, new = base.cpu().train(), new.cpu().train()
    torch.manual_seed(42)
    batch = dict(img=torch.rand(2, 3, 160, 192),
                 bboxes=torch.tensor([[.4,.4,.2,.1],[.6,.6,.3,.2],[.2,.7,.1,.25]]),
                 cls=torch.zeros(3,1), batch_idx=torch.tensor([0,1,1]))
    tb = targets(batch)
    # Zero-weight comparison includes equal RNG/DN, effective native AdamW/clip/EMA step.
    opt_base = trainer.build_optimizer(base, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    opt_new = trainer.build_optimizer(new, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    original = reference_loss_class()(nc=1, use_vfl=True)
    torch.manual_seed(123); p0 = base.predict(batch["img"], batch=tb)
    pp, db, ds, dn, _ = split_prediction(p0)
    old_losses = original(pp, tb, db, ds, dn)
    old_sum = sum(old_losses.values())
    torch.manual_seed(123); new_sum, shown = new.loss(batch)
    close(old_sum, new_sum, 0, 0)
    close(shown, torch.stack([old_losses[k].detach() for k in ("loss_giou","loss_class","loss_bbox")]), 0, 0)
    old_sum.backward(); new_sum.backward()
    gradient_differences = []
    for (name, left), (_, right) in zip(base.named_parameters(), new.named_parameters()):
        if left.grad is not None:
            difference = float((left.grad - right.grad).abs().max())
            if difference:
                gradient_differences.append(dict(name=name,max_abs=difference,
                    baseline_max_abs=float(left.grad.abs().max()),rdl_max_abs=float(right.grad.abs().max())))
    for model, optimizer in ((base, opt_base), (new, opt_new)):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step(); optimizer.zero_grad()
    maximum = max(float((v-new.state_dict()[k]).abs().max()) for k,v in base.state_dict().items())
    if maximum:
        write_json(ROOT/'outputs/rdl_v1/cuda_zero_comparison_failure.json',dict(device=device,maximum=maximum,
            gradients=sorted(gradient_differences,key=lambda row:row['max_abs'],reverse=True),
            updates=sorted([dict(name=k,max_abs=float((v-new.state_dict()[k]).abs().max()))
                            for k,v in base.state_dict().items() if not torch.equal(v,new.state_dict()[k])],
                           key=lambda row:row['max_abs'],reverse=True)))
    require(maximum == 0, f"Zero-weight effective update differed: {maximum}")
    del old_losses, old_sum, new_sum, pp, p0, opt_base
    base, new = base.to(device), new.to(device)
    batch = {k:v.to(device) for k,v in batch.items()};tb=targets(batch)
    for state in opt_new.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)
    # Same forward diagnostics gives identical predictions and original losses when active.
    torch.manual_seed(222); direct = base.predict(batch["img"], batch=tb)
    torch.manual_seed(222); raw, details = new.predict(batch["img"], batch=tb, return_cbr_details=True)
    for a, b in zip(direct[:4], raw[:4]): close(a, b, 0, 0)
    pp, db, ds, dn, details = split_prediction(raw, details)
    criterion = new.init_criterion(); criterion.set_epoch(20)
    l0, lr = original(pp,tb,db,ds,dn), criterion(pp,tb,db,ds,dn,details=details,image_hw=(160,192),enabled=True)
    for k in l0: close(l0[k], lr[k], 0, 0)
    require(lr["loss_rdl"] > 0 and criterion.last_stats["M"] == 3, "Real regular matching not connected")
    del direct, raw, pp, db, ds, dn, details, l0, lr, base
    new.rdl_epoch=20
    loss, _ = new.loss(batch); loss.backward()
    grads = {n:float(p.grad.norm()) for n,p in new.named_parameters()
             if p.grad is not None and ("cbr.offset_out" in n or "O_proj" in n)}
    require(all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None), "FP32 gradients")
    require(all(v>0 for v in grads.values()), "Original output modules lost gradients")
    opt_new.zero_grad(); del loss
    amp_report = dict(status="SKIPPED", reason="CPU selected")
    if device.startswith("cuda"):
        # Same bounded check scale as the mother's check_c19_lif_v1.py:175.
        # Does not alter the actual Trainer's native scaler initialization.
        scaler = torch.cuda.amp.GradScaler(init_scale=128.)
        with torch.cuda.amp.autocast(): loss,_=new.loss(batch)
        scaler.scale(loss).backward(); scaler.unscale_(opt_new)
        require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None), "Native AMP nonfinite")
        amp_report=dict(status="PASS", loss=float(loss), optimizer_update=False,diagnostic_scaler_init=128.,formal_scaler="unchanged native default")
        opt_new.zero_grad(); del loss
    # Save one native checkpoint in a bounded temporary directory; reload through real get_model/resume.
    trainer.model, trainer.ema, trainer.optimizer = new, ModelEMA(new), opt_new
    trainer.epoch, trainer.start_epoch, trainer.epochs, trainer.best_fitness = 19,0,200,.1
    new.rdl_epoch=19;trainer.ema.ema.rdl_epoch=19
    trainer.scaler=torch.cuda.amp.GradScaler(enabled=False)
    trainer.args=SimpleNamespace(close_mosaic=10,model="isolated-lifecycle-checkpoint.pt")
    trainer.metrics,trainer.fitness,trainer.save_period={},0.,-1
    trainer.read_results_csv=lambda: {}
    with tempfile.TemporaryDirectory(prefix="rdl-checkpoint-") as temporary:
        trainer.wdir=Path(temporary);trainer.last=trainer.wdir/"last.pt";trainer.best=trainer.wdir/"best.pt"
        trainer.save_model()
        ckpt=torch_load(trainer.last,map_location=device)
        require(ckpt["epoch"]==19 and ckpt["ema"].rdl_config==CONFIG,"Saved metadata")
        resume=RDLTrainer.__new__(RDLTrainer);resume.data=trainer.data;resume.resume=True
        resume.model=resume.get_model(ckpt["ema"].yaml,ckpt["ema"].float(),False).to(device)
        resume.model.nc = 1
        resume.epochs=200;resume.args=trainer.args
        resume.optimizer=resume.build_optimizer(resume.model,name="AdamW",lr=.0005,momentum=.937,decay=.0001)
        resume.scaler=torch.cuda.amp.GradScaler(enabled=False);resume.ema=ModelEMA(resume.model)
        resume.resume_training(ckpt);resume.epoch=resume.start_epoch;epoch_start(resume)
        require(resume.start_epoch==20 and resume.model.rdl_epoch==20,"Resume epoch")
        restored_loss,_=resume.model.train().loss(batch)
        require(resume.model.criterion.weight==.05 and resume.model.criterion.last_stats["active"],"Resume RDL lost")
        require(torch.isfinite(restored_loss),"Resume finite")
        del restored_loss,ckpt,resume
    # Native EMA inference + validation loss with precomputed preds must not need a second forward.
    ema=trainer.ema.ema.float().eval()
    from rdl_v1_fusion import check_ema_fusion
    fusion = check_ema_fusion(ema, batch, folder=fusion_dir, source_sha256=init_audit["source_sha256"])
    return dict(status="PASS", source_sha256=init_audit["source_sha256"], parameters=20149765,
                added_parameters=0, adaptation_keys=9, input_hw=[160,192], batch=2, device=device,
                zero_weight_update_max_abs=maximum, zero_weight_device="cpu", original_losses="exact equality", diagnostics_prediction="exact equality",
                gradient_norms=grads, native_amp=amp_report, native_save_reload_resume_e=20,
                inference_shape=[2,300,5], EMA_validation="L0 only", fusion_tolerance="atol=rtol=3e-5",
                fusion_original_status=fusion["original_status"], fusion_max_abs=fusion["original_assertion"]["max_abs_error"],
                fusion_acceptance=fusion["fusion_acceptance"],
                fusion_precision=fusion["precision_scope"])


def run_checks(source=None, device="cpu"):
    results={}
    for name, fn in (("mathematics",mathematical),("indices_and_L0",indexing),
                     ("real_model",lambda:model_checks(source,device))):
        if name=="real_model" and (source is None or not Path(source).is_file()):
            results[name]=dict(status="SKIPPED",reason="Verified unified source unavailable");continue
        try:
            results[name]=fn();print(name, "PASS",flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc();results[name]=dict(status="FAIL",error=repr(error));break
    return results


def native_cuda_repeat(source):
    """Isolate native-only CUDA atomic backward/scaler behavior; never enable RDL."""
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(42,deterministic=True);torch.set_num_threads(4)
    _,init,_=controlled_models(source)
    torch.manual_seed(71);model=native_rebuild(init.yaml,init,1,3);del init
    a,b=model.cuda().train(),deepcopy(model).cuda().train();del model
    trainer=RDLTrainer.__new__(RDLTrainer)
    torch.manual_seed(42)
    batch=dict(img=torch.rand(2,3,160,192,device='cuda'),bboxes=torch.tensor([[.4,.4,.2,.1],[.6,.6,.3,.2],[.2,.7,.1,.25]],device='cuda'),
               cls=torch.zeros(3,1,device='cuda'),batch_idx=torch.tensor([0,1,1],device='cuda'))
    tb=targets(batch);criterion=reference_loss_class()(nc=1,use_vfl=True)
    optimizers=[];losses=[]
    for model in (a,b):
        opt=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001);optimizers.append(opt)
        torch.manual_seed(123);raw=model.predict(batch['img'],batch=tb)
        pp,db,ds,dn,_=split_prediction(raw)
        loss=sum(criterion(pp,tb,db,ds,dn).values());losses.append(float(loss.detach()));loss.backward()
    gradient_max=max(float((x.grad-y.grad).abs().max()) for x,y in zip(a.parameters(),b.parameters()) if x.grad is not None)
    for model,opt in zip((a,b),optimizers):
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();opt.zero_grad()
    update_rows=sorted([dict(name=k,max_abs=float((v-b.state_dict()[k]).abs().max())) for k,v in a.state_dict().items()
                        if not torch.equal(v,b.state_dict()[k])],key=lambda r:r['max_abs'],reverse=True)
    scaler=torch.cuda.amp.GradScaler()  # original default, explicitly diagnose the earlier overflow
    with torch.cuda.amp.autocast():
        raw=a.predict(batch['img'],batch=tb);pp,db,ds,dn,_=split_prediction(raw)
        loss=sum(criterion(pp,tb,db,ds,dn).values())
    scaler.scale(loss).backward();scaler.unscale_(optimizers[0])
    bad=[n for n,p in a.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    return dict(status='OBSERVED',RDL_enabled=False,native_losses=losses,loss_exact=losses[0]==losses[1],
                native_repeat_gradient_max_abs=gradient_max,native_repeat_update_top10=update_rows[:10],
                default_amp_scale=scaler.get_scale(),amp_loss=float(loss.detach()),amp_loss_finite=bool(torch.isfinite(loss)),
                nonfinite_scaled_gradient_parameters=len(bad),first_nonfinite_parameters=bad[:10],
                conclusion='Native CUDA backward is not bitwise reproducible; default initial scaling can overflow. Exact optimizer equivalence is tested on CPU. Bounded AMP uses mother check scale=128; formal scaler unchanged.')


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path);parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/rdl_v1/checks.json")
    parser.add_argument('--native-cuda-repeat',action='store_true',help='仅隔离诊断原生CUDA非确定性和默认GradScaler；不启用RDL')
    parser.add_argument('--fusion-diagnostic',type=Path,help='仅重建原 real_model 生命周期现场并诊断融合；目录必须不存在，不执行数学/运营/容量/长训检查')
    args=parser.parse_args()
    if args.fusion_diagnostic:
        require(not args.native_cuda_repeat, "Choose only one diagnostic")
        require(not args.fusion_diagnostic.exists(), "Use a new diagnostic directory; preserve old evidence")
        require(args.source is not None and args.source.is_file(), "Verified unified source required")
        try:
            result=model_checks(args.source,args.device,fusion_dir=args.fusion_diagnostic)
        except Exception as error:
            import traceback
            traceback.print_exc()
            result=dict(status="FAIL",error=repr(error))
        write_json(args.fusion_diagnostic.parent/(args.fusion_diagnostic.name+"_result.json"),result)
        sys.exit(0 if result["status"]=="PASS" else 1)
    if args.native_cuda_repeat:
        write_json(args.output,native_cuda_repeat(args.source));sys.exit(0)
    results=run_checks(args.source,args.device);write_json(args.output,results)
    sys.exit(1 if any(r["status"]=="FAIL" for r in results.values()) else 0)
