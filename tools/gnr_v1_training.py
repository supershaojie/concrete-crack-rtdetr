"""Native Trainer integration and bounded effective-update preflight."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import json
import time

from gnr_v1_common import ROOT, OUT, RUN, INIT, MODEL, read_json, write_json, require, sha256, now
import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.gnr_model import GNRDetectionModel
from ultralytics.models.rtdetr.gnr_loss import FORMULA, ramp
from ultralytics.utils.patches import torch_load


def amp_assets(main):
    """Native AMP check, with explicit local assets; no network-dependent fallback."""
    from ultralytics.utils import ASSETS
    records = []
    for target, candidates in (
        (ROOT / "yolo26n.pt", [main / "yolo26n.pt", main / "weights/yolo26n.pt"]),
        (ASSETS / "bus.jpg", [main / "ultralytics-main/ultralytics/assets/bus.jpg", main / "bus.jpg"]),
    ):
        if not target.is_file():
            source = next((p for p in candidates if p.is_file()), None)
            require(source is not None, f"Missing native AMP check resource: {target.name}; supply mother's existing asset")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as a, target.open("xb") as b:
                shutil.copyfileobj(a, b)
        records.append({"path": str(target), "sha256": sha256(target)})
    write_json(OUT / "amp_assets.json", records)


class GNRTrainer(RTDETRTrainer):
    """Original optimizer/scaler/EMA/scheduler/save/early-stop implementation."""

    def __init__(self, *args, gnr_binding=None, evidence_dir=None, **kwargs):
        self.gnr_binding = gnr_binding
        self.evidence_dir = Path(evidence_dir or OUT)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_batch_start", self._guard_batch)
        self.add_callback("on_train_epoch_end", self._record_epoch)
        self.add_callback("on_train_batch_end", self._record_batch)

    @staticmethod
    def _guard_batch(trainer):
        # Disable the mother's optional batch-halving OOM retry; never alter B16.
        trainer._oom_retries = 3

    @staticmethod
    def _record_epoch(trainer):
        write_json(trainer.evidence_dir / "progress.json", {"epoch_completed": trainer.epoch + 1,
            "gnr_epoch": trainer.model.gnr_epoch, "gnr": getattr(trainer.model, "gnr_diagnostics", None),
            "train_loss": trainer.tloss.detach().cpu().tolist(), "time": now()})

    @staticmethod
    def _record_batch(trainer):
        value = {"epoch": trainer.epoch, "microbatch": trainer.gnr_batch_index,
                 "gnr": getattr(trainer.model, "gnr_diagnostics", None) if ramp(trainer.model.gnr_epoch) else None}
        with (trainer.evidence_dir / "gnr_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        trainer.gnr_batch_index += 1

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(weights is not None, "GNR needs verified controlled initialization or same-run resume")
        if not self.resume:
            from init_c19_lif_v1 import build_training_model
            mother, audit = build_training_model(cfg, weights, self.data)
        else:
            mother = super().get_model(cfg, weights, verbose)
            require(set(mother.state_dict()) == set(weights.state_dict()), "Resume state keys changed")
            require(all(torch.equal(v, weights.state_dict()[k].float()) for k, v in mother.state_dict().items()), "Resume state values changed during rebuild")
            audit = {"resume": True, "keys": len(mother.state_dict())}
        # Reference construction/loading follows the original RNG. The wrapper
        # construction consumes no additional global RNG and adds no parameters.
        with torch.random.fork_rng(devices=[]):
            model = GNRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
        model.load_state_dict(mother.state_dict(), strict=True)
        require(set(model.state_dict()) == set(mother.state_dict()), "GNR added state keys")
        model.gnr_identity, model.gnr_formula = self.gnr_binding, deepcopy(FORMULA)
        write_json(self.evidence_dir / "trainer_loading.json", audit)
        return model

    def setup_model(self):
        # Avoid native download helper's quote stripping for this Windows checkout.
        # The normal server path delegates without modification.
        if isinstance(self.model, (str, Path)) and "'" in str(self.model):
            self.model = str(Path(self.model).resolve().relative_to(ROOT))
        return super().setup_model()

    def _setup_train(self):
        super()._setup_train()
        require(self.args.amp and self.amp, "Native AMP check disabled AMP; formal recipe cannot change")
        require(self.args.batch == 16 and self.args.imgsz == 640 and self.args.nbs == 64, "B16/640/nbs64 changed")
        require(type(self.optimizer) is torch.optim.AdamW, "Expected native AdamW")
        expected = read_json(OUT / "prepare.json")["args"]
        allowed = {"model", "resume", "save_dir", "project", "name", "exist_ok"} if self.evidence_dir != OUT else {"model", "resume"}
        changes = {k: [v, getattr(self.args, k, None)] for k, v in expected.items()
                   if type(getattr(self.args, k, None)) is not type(v) or getattr(self.args, k, None) != v}
        require(set(changes) <= allowed, f"Trainer changed recipe: {changes}")
        write_json(self.evidence_dir / "training_setup.json", {"args": vars(self.args), "allowed_changes": changes,
            "binding": self.gnr_binding, "amp": bool(self.amp), "optimizer": type(self.optimizer).__name__,
            "parameter_count": sum(p.numel() for p in self.model.parameters()), "start_epoch": self.start_epoch})

    def _model_train(self):
        super()._model_train()
        self.gnr_batch_index = 0
        self.model.gnr_epoch = self.epoch
        self.ema.ema.gnr_epoch = self.epoch

    def _load_checkpoint_state(self, ckpt):
        require(ckpt.get("optimizer") is not None and ckpt.get("scaler") is not None and ckpt.get("ema") is not None, "Cannot resume stripped/incomplete checkpoint")
        require(getattr(ckpt["ema"], "gnr_identity", None) == self.gnr_binding, "Checkpoint GNR binding differs")
        require(getattr(ckpt["ema"], "gnr_formula", None) == FORMULA, "Checkpoint formula changed")
        super()._load_checkpoint_state(ckpt)

    def save_model(self):
        self.ema.ema.gnr_identity = self.gnr_binding
        self.ema.ema.gnr_epoch = self.epoch
        self.ema.ema.gnr_formula = deepcopy(FORMULA)
        super().save_model()
        write_json(self.evidence_dir / "checkpoint_state.json", {"epoch": self.epoch, "best_fitness": self.best_fitness,
            "fitness": self.fitness, "binding": self.gnr_binding, "last": str(self.last), "best": str(self.best),
            "ema_updates": self.ema.updates, "scaler": self.scaler.state_dict()})

    def final_eval(self):
        selected = torch_load(self.best, map_location="cpu")
        record = {"status": "TRAINING_COMPLETE_EVAL_PENDING", "binding": self.gnr_binding,
                  "completed_epoch": self.epoch + 1, "completion": "budget_completed" if self.epoch + 1 >= self.epochs else "early_stopped",
                  "best_epoch": selected["epoch"], "best_train_metrics": selected.get("train_metrics"), "time": now()}
        write_json(self.evidence_dir / "training_completed.json", record)
        try:
            super().final_eval()
            record["status"] = "TRAINING_COMPLETE"
        finally:
            record["weights"] = {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                                 for name, path in (("best", self.best), ("last", self.last)) if path.exists()}
            write_json(self.evidence_dir / "training_completed.json", record)
            print("Training complete. Training best-val metrics (independent FP32 val/test still separate):", record, flush=True)


class PreflightDone(Exception):
    pass


class PreflightTrainer(GNRTrainer):
    def __init__(self, *args, budget=900, max_batches=16, **kwargs):
        self.budget, self.max_batches = budget, max_batches
        self.started = time.monotonic()
        self.observation = {"status": "PENDING", "microbatches": 0, "microbatch_attempts": 0, "step_attempts": [], "effective_updates": 0,
                            "scaler_skips": 0, "gnr_epoch_context": 20, "gnr_observed": []}
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_batch_start", self._before_batch)
        self.add_callback("on_train_batch_end", self._after_batch)

    def _save_observation(self):
        self.observation.update(elapsed_seconds=time.monotonic() - self.started, binding=self.gnr_binding,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        write_json(self.evidence_dir / "preflight.json", self.observation)

    @staticmethod
    def _before_batch(trainer):
        if time.monotonic() - trainer.started >= trainer.budget or trainer.observation["microbatches"] >= trainer.max_batches:
            raise PreflightDone("Time/microbatch budget exhausted before an effective update")
        trainer.model.gnr_epoch = 20  # GNR context only; epoch/scheduler/accumulation are unchanged.
        trainer.observation["microbatch_attempts"] += 1
        trainer._save_observation()

    @staticmethod
    def _after_batch(trainer):
        trainer.observation["microbatches"] += 1
        require(bool(torch.isfinite(trainer.loss)), "Nonfinite preflight loss")
        diag = getattr(trainer.model, "gnr_diagnostics", None)
        require(diag and diag["epoch"] == 20 and diag["ramp"] == 1, "Missing full-strength same-forward GNR evidence")
        trainer.observation["gnr_observed"].append(diag)
        trainer._save_observation()
        if trainer.observation["effective_updates"]:
            trainer.observation["status"] = "TECHNICAL_PASS"
            raise PreflightDone("Verified finite nonzero parameter update")
        if trainer.observation["microbatches"] >= trainer.max_batches:
            raise PreflightDone("Microbatch budget exhausted without an effective update")

    def optimizer_step(self):
        chosen = {n: p for n, p in self.model.named_parameters() if p.requires_grad and
                  (n.startswith("model.0.") or "dec_score_head.2" in n or "offset_out" in n)}
        before = {n: p.detach().clone() for n, p in chosen.items()}
        scale_before = self.scaler.get_scale()
        finite_grad = all(bool(torch.isfinite(p.grad).all()) for p in self.model.parameters() if p.grad is not None)
        super().optimizer_step()  # ORIGINAL unscale, clip, scaler.step/update, zero_grad, EMA
        scale_after = self.scaler.get_scale()
        require(all(bool(torch.isfinite(p).all()) for p in self.model.parameters()), "Nonfinite parameters after native optimizer step")
        changes = {n: float((p.detach() - before[n]).abs().max()) for n, p in chosen.items()}
        skipped = scale_after < scale_before
        updated = not skipped and any(delta > 0 for delta in changes.values())
        self.observation["scaler_skips"] += int(skipped)
        self.observation["effective_updates"] += int(updated)
        self.observation["step_attempts"].append({"microbatch": self.observation["microbatches"] + 1,
            "accumulate": self.accumulate, "scale_before": scale_before, "scale_after": scale_after,
            "scaled_gradients_finite": finite_grad, "skipped": skipped, "updated": updated, "parameter_max_abs_changes": changes})
        self._save_observation()


def run_preflight(bind, folder, budget, batches):
    if not torch.cuda.is_available():
        result = {"status": "PENDING", "reason": "CUDA required for real AMP/B16/640 preflight", "binding": bind}
        write_json(folder / "preflight.json", result)
        return result
    plan = read_json(OUT / "prepare.json")
    amp_assets(Path(plan["main"]))
    args = dict(plan["args"], project=str(folder), name="temporary_run", save_dir=str(folder / "temporary_run"), exist_ok=False)
    trainer = PreflightTrainer(overrides=args, gnr_binding=bind, evidence_dir=folder, budget=budget, max_batches=batches)
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train()
        raise RuntimeError("Preflight unexpectedly completed training loop")
    except PreflightDone as result:
        trainer.observation["reason"] = str(result)
    except torch.cuda.OutOfMemoryError as error:
        trainer.observation.update(status="RESOURCE_ERROR", reason=repr(error))
        raise
    except BaseException as error:
        trainer.observation.update(status="IMPLEMENTATION_ERROR", reason=repr(error))
        raise
    finally:
        trainer._save_observation()
    return trainer.observation
