"""Two bounded synthetic CPU epochs through the actual Trainer, interrupted/resumed once."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ultralytics-main"))
import cv2
import numpy as np
import torch
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from gpc_trainer import GPCTrainer


class InterruptedAfterEpoch(Exception):pass


def run(output,scratch=None):
    torch.set_num_threads(1)
    start=time.monotonic()
    report=dict(status="FAIL",scope="synthetic CPU B2/160 native Trainer, NOT formal recipe or real data")
    try:
        with tempfile.TemporaryDirectory(prefix="gpc_pipeline_",dir=scratch) as temporary:
            root=Path(temporary)
            for split,count in (("train",4),("val",2)):
                (root/"images"/split).mkdir(parents=True)
                (root/"labels"/split).mkdir(parents=True)
                for i in range(count):
                    pixels=np.random.default_rng(i).integers(0,256,(160,192,3),dtype=np.uint8)
                    cv2.imwrite(str(root/"images"/split/f"{i}.png"),pixels)
                    (root/"labels"/split/f"{i}.txt").write_text("0 0.5 0.5 0.25 0.20\n")
            data=root/"data.yaml"
            YAML.save(data,dict(path=str(root),train="images/train",val="images/val",names={0:"crack"}))
            model=str(ROOT/"ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml")
            settings=dict(model=model,data=str(data),epochs=2,patience=50,imgsz=160,batch=2,nbs=4,
                          device="cpu",workers=0,amp=False,optimizer="AdamW",lr0=.0005,seed=42,
                          deterministic=True,project=str(root/"runs"),name="synthetic",plots=False,save=True,
                          mosaic=0.,mixup=0.,close_mosaic=0,exist_ok=False,val=True)
            first=GPCTrainer(overrides=settings,preflight=True,identity=dict(scope="synthetic_fixture"))
            def stop_after_saved_epoch(trainer):
                if trainer.epoch==0:raise InterruptedAfterEpoch()
            first.add_callback("on_fit_epoch_end",stop_after_saved_epoch)
            try:first.train()
            except InterruptedAfterEpoch:pass
            saved=torch_load(first.last,map_location="cpu")
            assert saved["epoch"]==0 and saved["optimizer"] is not None and saved["scaler"] is not None
            prior_updates=saved["updates"]
            second=GPCTrainer(overrides=dict(model=str(first.last),resume=str(first.last)),preflight=True,identity=dict(scope="synthetic_fixture"))
            second.train()
            assert second.start_epoch==1 and second.epoch==1
            assert second.ema.updates>prior_updates
            assert second.best.is_file() and second.last.is_file()
            assert all(x.get("auxiliary_images")==4 for x in first.gpc_rows+second.gpc_rows)
            if torch.cuda.is_available():
                from gpc_evaluate import evaluate
                evaluated=evaluate(second.best,data,"val",root/"independent_eval")
                assert evaluated["counts"]["images"]==2 and evaluated["counts"]["queries"]==600
                assert (root/"independent_eval/ap_pr_inputs.npz").is_file()
                assert (root/"independent_eval/predictions_gt.jsonl.gz").is_file()
                report["independent_FP32_evidence_export"]=dict(status="PASS",synthetic_images=2,queries=600,
                                                              protocol=evaluated["protocol"],AP_not_performance_evidence=True)
            else:
                report["independent_FP32_evidence_export"]="PENDING: CUDA unavailable"
            report.update(status="PASS",initial_epoch=0,resumed_start_epoch=second.start_epoch,last_epoch=second.epoch,
                          checkpoint_optimizer_present=True,ema_updates_before=prior_updates,ema_updates_after=second.ema.updates,
                          synthetic_real_trainer_microbatches=len(first.gpc_rows)+len(second.gpc_rows),
                          ordinary_validation_loss_and_final_eval="PASS",context_ramp_override="GPC-only diagnostic epoch20",
                          algorithm_config_restored=second.model.gpc_config==first.model.gpc_config)
    except BaseException as error:
        report.update(error=repr(error),trace=traceback.format_exc())
        traceback.print_exc()
    report["seconds"]=time.monotonic()-start
    GPCTrainer._write_json(output,report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,help="Use a scratch path without quotes; the parent loader strips quote characters")
    args=parser.parse_args();sys.exit(0 if run(args.output,args.scratch)["status"]=="PASS" else 1)
