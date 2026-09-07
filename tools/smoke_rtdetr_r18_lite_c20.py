"""Three real augmented training batches through the native trainer; isolated debug outputs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import torch

from init_rtdetr_r18_lite_c20 import require, sha256, write_json, verify_module
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML


def smoke(initialized, data, c2_args, output, expected_states):
    from audit_rtdetr_r18_lite_c20 import state_hashes, branch_gradients, optimizer_rows, compare_output
    from train_rtdetr_r18_lite_c20 import build_locked_args, DEFAULT_NAME

    output = Path(output).resolve()
    require(not output.exists(), f"Preserve previous smoke outputs; choose a fresh --smoke-dir: {output}")
    baseline = YAML.load(c2_args)
    recipe, _ = build_locked_args(baseline, Path(initialized), DEFAULT_NAME)
    debug = deepcopy(recipe)
    debug.update(data=str(Path(data).resolve()), project=str(output.parent), name=output.name, save_dir=str(output),
                 batch=2, workers=0, device="0" if torch.cuda.is_available() else "cpu", val=False, plots=False, save=False)
    report = {"status": "failed", "data": str(Path(data).resolve()), "data_sha256": sha256(data),
              "c2_args_sha256": sha256(c2_args), "steps": [], "debug_only": True,
              "recipe_differences": {k: {"C2": recipe[k], "smoke": debug[k]} for k in recipe if recipe[k] != debug[k]},
              "debug_scaler_init_scale": 128.0,
              "scaler_note": "Smoke-only initial scale 128; formal launcher retains the native C2 GradScaler default and dynamic scaling.",
              "schedule": "Native 200-epoch setup; only three explicit optimizer steps, no epoch loop or evaluation."}

    class SmokeTrainer(RTDETRTrainer):
        def _do_train(self):
            self._setup_train()
            require(state_hashes(self.model) == expected_states, "Real trainer rebuilt a different initialization.")
            verify_module(self.model)
            require(self.data["nc"] == 1, "Smoke requires the one-class crack dataset.")
            if self.device.type == "cuda":
                require(self.amp is True, "Native AMP check disabled AMP.")
            report["amp"] = self.amp
            report["actual_args"] = vars(self.args)
            report["optimizer"] = optimizer_rows(self.model, self.optimizer)
            report["initial_states_exact"] = len(expected_states)
            require(not self.optimizer.state, "Smoke optimizer was not initially empty.")
            require(self.ema.updates == 0 and state_hashes(self.ema.ema) == expected_states,
                    "Smoke initial EMA differs from clean initialization.")
            report.update(initial_optimizer_empty=True, initial_ema_updates=0, initial_ema_states_exact=True)
            # Audits may warm up FP16 separately; this is a real scaler/optimizer update.
            # Lower scale avoids an initial overflow being mistaken for branch inactivity.
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp, init_scale=128.)
            self.model.train()
            iterator = iter(self.train_loader)
            image = None
            for step in range(3):
                batch = self.preprocess_batch(next(iterator))
                image = batch["img"]
                count = len(batch["bboxes"])
                require(count > 0, "Expected GT and denoising queries in real smoke batch.")
                gt_groups = [(batch["batch_idx"] == i).sum().item() for i in range(len(image))]
                targets = {"cls": batch["cls"].long().flatten(), "bboxes": batch["bboxes"],
                           "batch_idx": batch["batch_idx"].long().flatten(), "gt_groups": gt_groups}
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    preds, details = self.model.predict(image, batch=targets, cbr_diagnostics=True)
                    loss, items = self.model.loss(batch, preds)
                require(preds[4] is not None and preds[4]["dn_num_split"][1] == 300, "Real DN split invalid.")
                require(preds[0].shape[2] == sum(preds[4]["dn_num_split"]), "Query order/count changed.")
                require(torch.isfinite(loss) and torch.isfinite(preds[0]).all(), "Nonfinite real batch output/loss.")
                require((preds[0][..., 2:] > 0).all(), "Illegal predicted width/height.")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grads = branch_gradients(self.model)
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        require(torch.isfinite(p.grad).all(), f"Real batch nonfinite gradient: {name}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
                self.scaler.step(self.optimizer); self.scaler.update()
                report["steps"].append({"step": step, "loss": float(loss.detach()), "gt_count": count,
                                        "dn_num_split": preds[4]["dn_num_split"], "module_gradients": grads,
                                        "image_files": batch["im_file"]})
            require(all(v > 0 for k, v in grads.items() if k != "score.bias"), "Real updates did not open the branch.")
            self.model.eval().float()
            # No debug weights are ever accepted by controlled init/formal launcher.
            checkpoint = {"model": deepcopy(self.model), "epoch": 0, "train_args": vars(self.args)}
            path = output / "smoke_only.pt"
            torch.save(checkpoint, path)
            restored = RTDETR(str(path)).model.eval().to(self.device)
            verify_module(restored, require_zero=False)
            with torch.inference_mode():
                before, diagnostic = self.model.predict(image.float(), cbr_diagnostics=True)
                after = restored(image.float())
            require((diagnostic["after"] - diagnostic["before"]).abs().max() > 0, "Trained CBR output unchanged.")
            report["checkpoint_reload"] = compare_output(before, after)
            report["debug_checkpoint_sha256"] = sha256(path)
            self.last = path  # Model.train's normal post-run reload must use the real debug checkpoint
            report["status"] = "passed"

    try:
        RTDETR(str(initialized)).train(trainer=SmokeTrainer, **debug)
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        write_json(output / "smoke.json", report)
    return report
