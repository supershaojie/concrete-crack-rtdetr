"""Exercise real Trainer setup, EMA validator, save and epoch-boundary continuation."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from pds_v1_common import *
from pds_v1_trainer import PDSTrainer, validate_resume
from pds_v1_deploy import deploy
from ultralytics.utils.patches import torch_load


def fail_head(module, args):
    raise AssertionError("PDS head called from evaluation")


def fixture(dataset, folder):
    data=folder/"fixture"
    files=sorted((dataset/"images/train").glob("*.jpg"))[:2]
    require(len(files)==2,"Two real train images required for the train-derived lifecycle fixture")
    for split in ("train","val"):
        (data/"images"/split).mkdir(parents=True,exist_ok=True)
        (data/"labels"/split).mkdir(parents=True,exist_ok=True)
        for image in files:
            label=dataset/"labels/train"/(image.stem+".txt")
            shutil.copyfile(image,data/"images"/split/image.name)
            shutil.copyfile(label,data/"labels"/split/label.name)
    config=folder/"fixture.yaml"
    YAML.save(config,dict(path=str(data.resolve()),train="images/train",val="images/val",
                          test="images/val",names={0:"crack"}))
    return config


def compare_nested(a,b):
    if isinstance(a,torch.Tensor):
        require(torch.equal(a.cpu(),b.cpu()),"Resume tensor state differs")
    elif isinstance(a,dict):
        require(a.keys()==b.keys(),"Resume dictionary inventory differs")
        for k in a:
            compare_nested(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        require(len(a)==len(b),"Resume list size differs")
        for x,y in zip(a,b):
            compare_nested(x,y)
    else:
        require(a==b,"Resume scalar state differs")


def continuation(path, result):
    ckpt=torch_load(path,map_location="cpu")
    validate_resume(ckpt)
    args=ckpt["train_args"].copy()
    args.update(model=str(path),resume=str(path),device="cpu",workers=0,exist_ok=True)
    t=PDSTrainer(overrides=args)
    t._setup_train()
    require(t.start_epoch==7 and t.model.pds_epoch==7,"Epoch/ramp did not continue")
    compare_nested(ckpt["optimizer"],t.optimizer.state_dict())
    compare_nested(ckpt["scaler"],t.scaler.state_dict())
    compare_nested(ckpt["scheduler"],t.scheduler.state_dict())
    compare_nested(ckpt["stopper"],vars(t.stopper))
    for key,model in (("model",t.model),("ema",t.ema.ema)):
        compare_nested(ckpt[key].state_dict(),model.state_dict())
    t.optimizer.zero_grad()
    t._restore_pending_gradients()
    for n,p in t.model.named_parameters():
        expected=ckpt["pds_resume"]["pending_gradients"].get(n)
        require((expected is None and p.grad is None) or
                (expected is not None and p.grad is not None and torch.equal(expected,p.grad.cpu())),
                "Accumulated gradient not restored")
    t.epoch=7
    t.model.train()
    b=t.preprocess_batch(next(iter(t.train_loader)))
    before=t.model.pds_head.box.weight.detach().clone()
    with torch.enable_grad():
        loss,items=t.model(b)
        t.scaler.scale(loss).backward()
    t.optimizer_step()
    require(not torch.equal(before,t.model.pds_head.box.weight),"Continued optimizer did not update head")
    write_json(result,dict(status="PASS",epoch=7,ramp=(7-5)/15,optimizer_state_exact=True,
                           scaler_scheduler_stopper_exact=True,raw_model_ema_exact=True,
                           pending_gradients_exact=True,continued_real_update=True))


def lifecycle(source,dataset,folder):
    torch.set_num_threads(4)
    folder.mkdir(parents=True,exist_ok=True)
    report=dict(status="FAILED",scope="bounded CPU train-derived fixture; not official validation/test")
    try:
        data=fixture(dataset,folder)
        _,native,_=controlled_models(source)
        native.args=recipe()
        init=folder/"native_init.pt"
        torch.save(dict(model=native,ema=None,epoch=-1,train_args=native.args),init)
        args=recipe()
        args.update(model=str(init),data=str(data),device="cpu",batch=2,nbs=8,imgsz=160,workers=0,
                    amp=False,plots=False,project=str(folder),name="trainer",save_dir=str(folder/"trainer"))
        t=PDSTrainer(overrides=args)
        t._setup_train()
        require(type(t.model).__name__=="PDSDetectionModel" and t.model.nc==1,"Real Trainer lost PDS")
        t.model.train()
        b=t.preprocess_batch(next(iter(t.train_loader)))
        head_before={k:v.detach().clone() for k,v in t.model.pds_head.state_dict().items()}
        t.epoch=5
        t.model.pds_epoch=5
        t.loss,t.loss_items=t.model(b)
        t.scaler.scale(t.loss).backward()
        require(all(p.grad is None for p in t.model.pds_head.parameters()),"r=0 has auxiliary gradients")
        t.optimizer_step()
        require(all(p not in t.optimizer.state for p in t.model.pds_head.parameters()),"r=0 created auxiliary AdamW state")
        require(all(torch.equal(v,t.model.pds_head.state_dict()[k]) for k,v in head_before.items()),"r=0 decayed head")
        t.epoch=6
        t.model.pds_epoch=6
        t.loss,t.loss_items=t.model(b)
        t.scaler.scale(t.loss).backward()
        t.optimizer_step()
        # Save at an epoch boundary with a partial accumulation window.
        t.loss,t.loss_items=t.model(b)
        t.scaler.scale(t.loss).backward()
        t._last_opt_step=5
        t.scheduler.last_epoch=6
        t.stopper.best_epoch=4
        t.stopper.best_fitness=.2
        sentinel=t.ema.ema.pds_head.register_forward_pre_hook(fail_head)
        try:
            t.metrics,t.fitness=t.validate()  # actual per-epoch RTDETRValidator and loss(preds)
        finally:
            sentinel.remove()
        t.best_fitness=t.fitness
        t.stop=False
        t.save_model()
        ckpt=torch_load(t.last,map_location="cpu")
        validate_resume(ckpt)
        require(len(ckpt["pds_resume"]["pending_gradients"])>0,"No pending accumulation saved")
        result=folder/"continuation.json"
        subprocess.run([sys.executable,str(ROOT/"tools/check_pds_v1_lifecycle.py"),"--continue-from",str(t.last),
                        "--result",str(result)],cwd=ROOT,check=True)
        report["resume"]=read_json(result)
        # Nonzero training state -> native deploy -> common independent evaluator for both splits.
        # Both use the train-derived two-image fixture, never the actual held-out test split.
        deploy_path=folder/"deploy.pt"
        deployment=deploy(t.best,deploy_path,folder/"deployment.json")
        from c19_lif_v1_results import evaluate
        val=evaluate(deploy_path,data,"val",folder/"fixture_val",device="cpu",evidence_scope="train_derived_fixture",runtime_info=environment())
        test=evaluate(deploy_path,data,"test",folder/"fixture_test",device="cpu",
                      val_report=folder/"fixture_val/metrics.json",evidence_scope="train_derived_fixture",runtime_info=environment())
        # Explicitly verify incomplete/deploy checkpoints cannot resume.
        for incomplete in (torch_load(deploy_path,map_location="cpu"),dict(ckpt,optimizer=None)):
            try:
                validate_resume(incomplete)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Incomplete resume accepted")
        report.update(status="PASS",native_get_model=True,ema_has_all_auxiliary_keys=True,
                      r0_head_unchanged_no_adamw_state=True,per_epoch_validator_head_calls=0,
                      independent_fixture_val_test="PASS",actual_test_images_used=0,
                      deploy_source=deployment["selected"],deploy_exact=deployment["raw_output_exact"],
                      incomplete_resume_rejected=True)
    except BaseException as error:
        import traceback
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(folder/"lifecycle.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--dataset",type=Path,default=MAIN/"datasets/crack_det")
    p.add_argument("--output",type=Path,default=OUT/"lifecycle")
    p.add_argument("--continue-from",type=Path)
    p.add_argument("--result",type=Path)
    a=p.parse_args()
    if a.continue_from:
        torch.set_num_threads(4)
        continuation(a.continue_from,a.result)
    else:
        print(json.dumps(lifecycle(a.source,a.dataset,a.output),ensure_ascii=False,indent=2))
