"""Three real augmented training batches through the native trainer; isolated debug outputs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import time
import torch

from init_acr import require, sha256, write_json, verify_module, branch_modules
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML


def smoke(initialized, data, c2_args, output, expected_states, source):
    from audit_acr import state_hashes, branch_gradients, optimizer_rows, compare_output
    from train_acr import build_locked_args, DEFAULT_NAME

    output = Path(output).resolve()
    require(not output.exists(), f"Preserve previous smoke outputs; choose a fresh --smoke-dir: {output}")
    baseline = YAML.load(c2_args)
    recipe, _ = build_locked_args(baseline, Path(initialized), DEFAULT_NAME)
    debug = deepcopy(recipe)
    debug.update(data=str(Path(data).resolve()), project=str(output.parent), name=output.name, save_dir=str(output),
                 batch=int(os.environ.get("ACR_SMOKE_BATCH", "2")), workers=0,
                 device="0" if torch.cuda.is_available() else "cpu", val=False, plots=False, save=False)
    report = {"status": "failed", "data": str(Path(data).resolve()), "data_sha256": sha256(data),
              "amp_validation": "Explicit ACR AMP forward/backward and three optimizer updates; inspect console for native YOLO helper status.",
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
            for module in branch_modules(self.model).values(): module.acr_stats_interval = 1
            for step in range(3):
                batch = self.preprocess_batch(next(iterator))
                image = batch["img"]
                count = len(batch["bboxes"])
                require(count > 0, "Expected GT and denoising queries in real smoke batch.")
                gt_groups = [(batch["batch_idx"] == i).sum().item() for i in range(len(image))]
                targets = {"cls": batch["cls"].long().flatten(), "bboxes": batch["bboxes"],
                           "batch_idx": batch["batch_idx"].long().flatten(), "gt_groups": gt_groups}
                if step == 0:
                    report["real_initial_equivalence"] = real_initial_equivalence(self.model, batch, targets, source)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    preds = self.model.predict(image, batch=targets)
                    loss, items = self.model.loss(batch, preds)
                require(preds[4] is not None and preds[4]["dn_num_split"][1] == 300, "Real DN split invalid.")
                require(preds[0].shape[2] == sum(preds[4]["dn_num_split"]), "Query order/count changed.")
                require(torch.isfinite(loss) and torch.isfinite(preds[0]).all(), "Nonfinite real batch output/loss.")
                require((preds[0][..., 2:] > 0).all(), "Illegal predicted width/height.")
                # Independent FP32 check on the same real 640/DN batch at initialization.
                if step == 0:
                    with torch.autocast(device_type=self.device.type, enabled=False):
                        fp32_loss, _ = self.model(batch)
                    fp32_loss.backward()
                    report['real_640_fp32_backward'] = {'loss': float(fp32_loss.detach()),
                        'gradients': branch_gradients(self.model)}
                    self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grads = branch_gradients(self.model)
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        require(torch.isfinite(p.grad).all(), f"Real batch nonfinite gradient: {name}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
                self.scaler.step(self.optimizer); self.scaler.update()
                if self.device.type == 'cuda': torch.cuda.synchronize()
                report["steps"].append({"step": step, "loss": float(loss.detach()), "gt_count": count,
                                        "dn_num_split": preds[4]["dn_num_split"], "module_gradients": grads,
                                        "image_files": batch["im_file"], "image_shape": list(image.shape),
                                        "elapsed_ms": (time.perf_counter()-started)*1000,
                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device.type=='cuda' else None,
                                        "allocation": {n:m.acr_last_stats for n,m in branch_modules(self.model).items()}})
            require(all(v > 0 for v in grads.values()), "Real updates did not open every ACR parameter tensor.")
            first = report['steps'][0]['module_gradients']
            require(all(v > 0 for k,v in first.items() if k.endswith('acr_head.weight')), 'Zero head weight received no gradient.')
            require(all(v == 0 for k,v in first.items() if '.acr_head.' not in k), 'Upstream first-step gradient should be zero.')
            for module in branch_modules(self.model).values():
                module.acr_stats_interval = 0; module.acr_last_stats = None
            # A real one-image terminal batch, with an isolated model and no fourth update.
            terminal = deepcopy(batch)
            select = terminal['batch_idx'].flatten() == 0
            terminal['img'] = terminal['img'][:1]
            for key in ('bboxes', 'cls', 'batch_idx'): terminal[key] = terminal[key][select]
            terminal_model = deepcopy(self.model).train()
            terminal_model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                terminal_loss, _ = terminal_model(terminal)
            terminal_loss.backward()
            report['real_terminal_batch'] = {'batch_size': 1, 'loss': float(terminal_loss.detach()),
                                             'gradients': branch_gradients(terminal_model), 'optimizer_updates': 0}
            del terminal_model, terminal_loss, terminal
            self.model.eval().float()
            # No debug weights are ever accepted by controlled init/formal launcher.
            report["optimizer_step_counts"] = sorted({int(v['step']) for v in self.optimizer.state.values()})
            require(report["optimizer_step_counts"] == [3], "Expected three actual optimizer updates, not skipped AMP steps")
            checkpoint = {"model": deepcopy(self.model), "epoch": 0, "train_args": vars(self.args),
                          "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict()}
            path = output / "smoke_only.pt"
            torch.save(checkpoint, path)
            restored = RTDETR(str(path)).model.eval().to(self.device)
            from ultralytics.utils.patches import torch_load
            from audit_acr import real_optimizer
            saved = torch_load(path, map_location=self.device)
            optimizer_reloaded = real_optimizer(restored)
            optimizer_reloaded.load_state_dict(saved['optimizer'])
            report['optimizer_checkpoint_reload'] = compare_output(self.optimizer.state_dict(), optimizer_reloaded.state_dict())
            del saved, optimizer_reloaded
            verify_module(restored, require_zero=False)
            with torch.inference_mode():
                before = self.model.predict(image.float())
                after = restored(image.float())
            with torch.inference_mode():
                half_out = deepcopy(restored).half()(image.half())[0]
            require(torch.isfinite(half_out).all(), 'Trained ACR half inference failed.')
            report['trained_half_forward'] = {'shape': list(half_out.shape), 'finite': True}
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


def real_initial_equivalence(model, batch, targets, source):
    """Isolated copies: same real augmented batch, RNG, native loss, DN and every matcher result."""
    from init_acr import read_source
    from audit_acr import trainer_build, compare_output
    _, weights = read_source(Path(source))
    reference, _ = trainer_build(weights.yaml, weights, 1)
    reference.nc = reference.model[-1].nc  # Native trainer.set_model_attributes supplies this before training loss.
    reference.to(batch['img'].device).train()
    target = deepcopy(model).train()
    results=[];matches=[]
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        for candidate in (reference,target):
            if getattr(candidate,'criterion',None) is None: candidate.criterion=candidate.init_criterion()
            matching=[]
            hook=candidate.criterion.matcher.register_forward_hook(lambda m,a,o: matching.append(o))
            torch.manual_seed(719)
            prediction=candidate.predict(batch['img'],batch=targets)
            loss=candidate.loss(batch,prediction)
            hook.remove();results.append((prediction,loss));matches.append(matching)
    report={'outputs_loss_dn':compare_output(results[0],results[1]),
            'matcher_calls':len(matches[0]),'matching':compare_output(matches[0],matches[1]),
            'reference_loss':float(results[0][1][0]),'target_loss':float(results[1][1][0]),
            'dn_num_split':results[0][0][4]['dn_num_split'],'status':'passed'}
    require(report['matcher_calls']>0,'Native Hungarian matcher was not exercised')
    return report
