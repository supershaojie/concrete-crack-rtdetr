"""Actual Trainer setup/save/resume and Validator sentinel on a tiny disposable dataset."""
from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
import tempfile

from bmc_v1_common import OUT, MODEL, require, write, runtime
import cv2
import numpy as np
import torch
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import autocast
from bmc_v1_training import BMCTrainer


def optimizer_equal(expected, actual):
    require(expected["param_groups"] == actual["param_groups"], "Restored optimizer groups differ")
    require(expected["state"].keys() == actual["state"].keys(), "Restored optimizer states missing")
    checked = 0
    for key, state in expected["state"].items():
        for name, value in state.items():
            restored = actual["state"][key][name]
            if isinstance(value, torch.Tensor):
                require(torch.equal(value.to(restored.device, restored.dtype), restored), f"Optimizer restore differs: {key}/{name}")
            else:
                require(value == restored, "Optimizer scalar restore differs")
            checked += 1
    require(checked > 0, "Empty optimizer is not a resume test")
    return checked


def lifecycle_checks(device="cpu"):
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(status="FAIL", runtime=runtime(), scope="B2/160 disposable actual Trainer; not formal capacity", device=device)
    cuda = device != "cpu"
    with tempfile.TemporaryDirectory(prefix="lifecycle_", dir=OUT) as temporary:
        folder = Path(temporary)
        try:
            for split in ("train", "val"):
                (folder / "images" / split).mkdir(parents=True)
                (folder / "labels" / split).mkdir(parents=True)
                for i in range(2):
                    pixels = np.random.default_rng(i).integers(0, 256, (160, 160, 3), dtype=np.uint8)
                    cv2.imwrite(str(folder / "images" / split / f"{i}.jpg"), pixels)
                    (folder / "labels" / split / f"{i}.txt").write_text("0 0.4 0.5 0.1 0.2\n", encoding="utf-8")
            YAML.save(folder / "data.yaml", dict(path=str(folder), train="images/train", val="images/val", names={0: "crack"}))
            args = YAML.load(OUT / "train_args.yaml") if (OUT / "train_args.yaml").exists() else {}
            args.update(model=str(MODEL), data=str(folder / "data.yaml"), project=str(folder / "runs"), name="source",
                        save_dir=str(folder / "runs/source"), batch=2, imgsz=160, workers=0, device=device, epochs=200,
                        amp=cuda, plots=False, cache=False, save=True, val=True, optimizer="AdamW", lr0=.0005,
                        weight_decay=.0001, nbs=64, warmup_epochs=5, deterministic=True)
            trainer = BMCTrainer(overrides=args)
            trainer._setup_train()
            require(bool(trainer.amp) == cuda, "Native AMP check unexpectedly changed mode")
            if cuda:
                trainer.scaler = torch.cuda.amp.GradScaler(init_scale=128)
            trainer.epoch = 19
            trainer.model.set_bmc_epoch(19)
            trainer._model_train()
            batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
            with autocast(trainer.amp):
                loss, trainer.loss_items = trainer.model(batch)
            trainer.loss = loss.detach()
            trainer.scaler.scale(loss).backward()
            trainer.optimizer_step()
            require(any(float(s.get("step", 0)) > 0 for s in trainer.optimizer.state.values()), "Fixture has no effective optimizer update")
            trainer.fitness, trainer.best_fitness = .1, .1
            trainer.metrics = {"metrics/mAP50-95(B)": .1}
            trainer.save_model()
            checkpoint = torch_load(trainer.last, map_location="cpu")
            from ultralytics import RTDETR
            from bmc_v1_results import verify_selected_model
            wrapper = RTDETR(str(trainer.last))
            verify_selected_model(wrapper, expected_name="source")
            require("name" not in wrapper.model.args, "Fixture no longer reproduces inference args filtering")
            del wrapper
            # Explicitly synthetic epoch fixtures. Weight/optimizer/scaler/EMA are from a real tiny update.
            cases = []
            for epoch in (19, 57):
                fixture = dict(checkpoint, epoch=epoch)
                path = folder / f"resume_epoch{epoch}.pt"
                torch.save(fixture, path)
                restored = BMCTrainer(overrides=dict(model=str(path), resume=str(path), device=device, workers=0, plots=False))
                restored._setup_train()
                audit = restored.resume_audit
                require(restored.start_epoch == epoch+1 and restored.model.criterion.epoch == epoch+1, "Resume epoch reset")
                require(restored.model.criterion.active, "Resumed criterion warmup restarted")
                tensor_count = optimizer_equal(fixture["optimizer"], restored.optimizer.state_dict())
                require(restored.scaler.state_dict() == fixture["scaler"], "GradScaler did not restore")
                require(audit["ema_exact"] and restored.ema.updates == fixture["updates"], "EMA did not restore")
                restored._model_train()
                with autocast(restored.amp):
                    replay_loss, _ = restored.model(batch)
                restored.scaler.scale(replay_loss).backward()
                require(torch.isfinite(replay_loss) and all(torch.isfinite(p.grad).all() for p in restored.model.parameters()
                        if p.grad is not None), "Active resumed loss/gradient nonfinite")
                require(restored.model.criterion.last_report["state"] == "ENABLED", "Actual resumed loss failed to enable BMC")
                cases.append(dict(checkpoint_epoch=epoch, next_epoch=restored.start_epoch, optimizer_values_checked=tensor_count,
                                  scaler_restored=cuda, scaler_disabled_cpu=not cuda, scaler_state=restored.scaler.state_dict(),
                                  ema_exact=True, bmc_active=True, active_loss=float(replay_loss.detach()),
                                  active_backward_finite=True, active_amp=bool(restored.amp)))
                del restored
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
            # Nontrivial LIF/CBR weights; both ordinary val and independent val must bypass BMC.
            with torch.no_grad():
                trainer.ema.ema.model[-1].cbr.offset_out.weight.fill_(.01)
                trainer.ema.ema.model[20].O_proj.weight.fill_(.002)
            def forbidden(*args):
                raise AssertionError("BMC assignment invoked by validation/predict")
            trainer.ema.ema.criterion.route_observer = forbidden
            trainer.ema.ema.set_bmc_epoch(57)
            trainer.epoch, trainer.epochs = 57, 200
            values = trainer.validator(trainer)
            require(values is not None and trainer.ema.ema.criterion.last_report["state"] == "EVAL_NATIVE", "Epoch validator did not use native criterion")
            trainer.validator.args.half = False
            trainer.validator.args.plots = False
            independent = trainer.validator(model=trainer.ema.ema)
            require(independent is not None, "Independent validator failed")
            trainer.ema.ema.eval()
            with torch.no_grad():
                predictions = trainer.ema.ema.predict(batch["img"].float())
            require(torch.isfinite(predictions[0]).all(), "Nonfinite predict sentinel")
            report.update(status="PASS", resume_cases=cases, actual_trainer_constructed=True,
                          epoch_validator_sentinel=True, independent_validator_sentinel=True, predict_sentinel=True,
                          synthetic_epoch_metadata=True, diagnostic_only_scale=128 if cuda else None,
                          original_optimizer_scaler_ema_methods=True, selected_weight_identity_from_checkpoint=True)
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            write(OUT / ("lifecycle_" + str(device).replace(":", "_") + ".json"), report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    lifecycle_checks(parser.parse_args().device)
