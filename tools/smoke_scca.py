"""Optional LOCAL two-update real-data exercise of RTDETR.train; never a server start prerequisite."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from init_scca import ROOT, VARIANTS, initialize, require, verify_model, write_json, sha256
from train_scca import rebuild_audit, disable_oom_retry
from check_scca import comparison, tensors
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def smoke(variant, source, data, output):
    require(not output.exists(), "Preserve old smoke output")
    output.mkdir(parents=True)
    init = output / "independent_smoke_init.pt"
    write_json(output / "initialization.json", initialize(source, init, variant))
    formal = YAML.load(ROOT / "docs/scca/c2_args.yaml")
    debug = {**formal, "model": str(init), "data": str(data), "imgsz": 320, "batch": 2, "workers": 0,
             "device": "0" if torch.cuda.is_available() else "cpu", "project": str(output), "name": "run",
             "save_dir": str(output / "run"), "plots": False, "val": False, "save": False}
    report = dict(status="failed", debug_only=True, formal_training_started=False, variant=variant,
                  debug_recipe_differences={k: [formal[k], v] for k, v in debug.items() if formal[k] != v},
                  debug_scaler_init_scale=128, schedule="Native train API + _setup_train + two direct optimizer updates; no epoch loop",
                  data_sha256=sha256(data), steps=[])

    class SmokeTrainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            m = super().get_model(cfg, weights, verbose)
            report["nc1_loading"] = rebuild_audit(weights, m, variant)
            return m

        def _do_train(self):
            self._setup_train()
            verify_model(self.model, variant, zero=True)
            require(self.amp or self.device.type == "cpu", "Native AMP check did not enable AMP")
            report["native_amp"] = bool(self.amp)
            require(not self.optimizer.state and self.ema.updates == 0, "Smoke must begin with fresh optimizer/EMA")
            report["fresh_optimizer_ema"] = True
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp, init_scale=128)
            self.model.train()
            iterator = iter(self.train_loader)
            added = {n: p for n, p in self.model.named_parameters() if ".scca_" in n}
            ids = [id(p) for g in self.optimizer.param_groups for p in g["params"]]
            require(all(ids.count(id(p)) == 1 for p in added.values()), "Optimizer registration failed")
            for step in range(2):
                batch = self.preprocess_batch(next(iterator))
                if step == 1:
                    batch["img"] = batch["img"][:1]
                    select = batch["batch_idx"] == 0
                    for key in ("cls", "bboxes", "batch_idx"):
                        batch[key] = batch[key][select]
                self.optimizer.zero_grad(set_to_none=True)
                targets = dict(cls=batch["cls"].long().flatten(), bboxes=batch["bboxes"], batch_idx=batch["batch_idx"].long(),
                               gt_groups=[int((batch["batch_idx"] == i).sum()) for i in range(len(batch["img"]))])
                with torch.autocast(self.device.type, enabled=self.amp):
                    preds = self.model.predict(batch["img"], batch=targets)
                    loss, items = self.model.loss(batch, preds)
                require(torch.isfinite(loss).all() and preds[-1] is not None, "Native loss/DN failed")
                self.scaler.scale(loss).backward()
                # Inspect scaled gradients; native optimizer_step owns unscale, clip, update and EMA.
                scale = self.scaler.get_scale()
                gradients = {n: float(p.grad.detach().abs().max()) / scale for n, p in added.items()}
                require(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None), "Nonfinite real gradients")
                require(gradients["model.9.scca_o.weight"] > 0, "O not learning")
                require(all(v == 0 if step == 0 else v > 0 for n, v in gradients.items() if n != "model.9.scca_o.weight"),
                        "Real-data staged gradients failed")
                before = {n: p.detach().clone() for n, p in added.items()}
                self.optimizer_step()
                report["steps"].append(dict(loss=float(loss.detach()), loss_items=items.detach().cpu().tolist(),
                    image_shape=list(batch["img"].shape), dn_num_split=preds[-1]["dn_num_split"], max_gradients=gradients,
                    max_updates={n: float((p-before[n]).detach().abs().max()) for n, p in added.items()},
                    images=batch["im_file"][:len(batch["img"])]))
            require(self.ema.updates == 2, "Expected two actual optimizer/EMA updates")
            report["optimizer_steps"] = sorted({int(v["step"]) for v in self.optimizer.state.values()})
            require(report["optimizer_steps"] == [2], "AMP skipped an update")
            self.model.eval().float()
            path = output / "smoke_only.pt"
            torch.save(dict(model=deepcopy(self.model), epoch=0, train_args=vars(self.args), optimizer=self.optimizer.state_dict(),
                            scaler=self.scaler.state_dict(), smoke_only=True), path)
            restored = RTDETR(str(path)).model.eval().to(self.device)
            verify_model(restored, variant)
            with torch.no_grad():
                image = batch["img"].float()
                report["updated_checkpoint_max_abs"] = comparison(self.model(image), restored(image))
                if self.device.type == "cuda":
                    result = deepcopy(restored).half()(image.half())
                    require(all(torch.isfinite(v).all() for v in tensors(result)), "Updated half checkpoint failed")
                    report["updated_half"] = "passed"
            saved = torch_load(path, map_location=self.device)
            opt = self.build_optimizer(restored, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
            opt.load_state_dict(saved["optimizer"])
            report["optimizer_reload_max_abs"] = comparison(self.optimizer.state_dict(), opt.state_dict())
            self.last = path
            report["status"] = "passed"

    try:
        m = RTDETR(str(init))
        m.add_callback("on_train_batch_start", disable_oom_retry)
        m.train(trainer=SmokeTrainer, **debug)
        return report
    finally:
        write_json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    smoke(args.variant, args.source, args.data.resolve(), args.output.resolve())
