"""Native RTDETR Trainer with strict loading and read-only sampled diagnostics."""
from __future__ import annotations

from copy import deepcopy
import logging
import traceback

from tcr_v1_core import *


class TCRTrainer(RTDETRTrainer):
    def __init__(self, *args, evidence=None, dispatch=None, **kwargs):
        self.evidence = Path(evidence) if evidence else None
        self.dispatch = dispatch
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(weights is not None, "TCR training requires audited controlled init or valid last")
        model, audit = native_rebuild(cfg, weights, self.data,
                                      resume=bool(self.resume or getattr(self, "fixture_allow_nonzero", False)))
        if self.evidence:
            write_json(self.evidence / "trainer_loading.json", audit)
        self.loading_audit = audit
        return model

    def _setup_train(self):
        # Observe the original check_amp result, including its "skipped" fallback.
        # Neither replace check_amp nor force its boolean result.
        from ultralytics.utils import LOGGER
        records=[]
        class Capture(logging.Handler):
            def emit(self,record):
                message=record.getMessage()
                if "AMP:" in message: records.append(message)
        handler=Capture(); LOGGER.addHandler(handler)
        try:
            result=super()._setup_train()
        finally:
            LOGGER.removeHandler(handler)
            if self.evidence:
                write_json(self.evidence/"native_amp_check.json",dict(messages=records,
                    requested=bool(self.args.amp),passed=any("checks passed" in r for r in records),
                    skipped=any("checks skipped" in r for r in records)))
        if self.args.amp and self.device.type=="cuda":
            require(any("checks passed" in r for r in records) and not any("checks skipped" in r for r in records),
                    "Native check_amp did not actually pass; inspect resources/environment; no bypass allowed")
        return result


class MechanismLogger:
    """Scalar-only opt-in observations, no RNG draw or retained autograd tensors."""
    def __init__(self, folder, dispatch):
        self.folder, self.dispatch = Path(folder), dispatch
        self.samples, self.gradients, self.handles = [], {}, []
        self.local_batch = 0
        self.last_summary = None

    def setup(self, trainer):
        verify_model(trainer.model, zero=not trainer.resume)
        require(trainer.amp, "Native AMP check disabled AMP; formal recipe cannot change")
        groups = optimizer_audit(trainer.model, trainer.optimizer)
        planned = YAML.load(OUT / "train_args.yaml")
        allowed = {"model", "resume"} if trainer.resume else set()
        differences = {k: [v, vars(trainer.args).get(k)] for k,v in planned.items()
                       if k not in allowed and (type(v) is not type(vars(trainer.args).get(k)) or v != vars(trainer.args).get(k))}
        require(not differences, f"Native training recipe drift: {differences}")
        YAML.save(self.folder / "actual_train_args.yaml", vars(trainer.args))
        criterion = trainer.model.init_criterion()
        write_json(self.folder / "training_setup.json", dict(dispatch=self.dispatch, runtime=runtime(),
                   topology=verify_model(trainer.model), groups=groups, loading=trainer.loading_audit,
                   criterion=type(criterion).__name__, loss_gain=criterion.loss_gain,
                   matcher=type(criterion.matcher).__name__, matcher_cost=criterion.matcher.cost_gain,
                   accumulate=trainer.accumulate, scaler=trainer.scaler.state_dict(),
                   source=source_identity(), resume=bool(trainer.resume), start_epoch=trainer.start_epoch,
                   ema_updates=trainer.ema.updates, optimizer_states=len(trainer.optimizer.state)))
        for name, p in trainer.model.named_parameters():
            if name in NEW:
                def observe(grad, key=name):
                    if trainer.model.model[17].tcr.capture:
                        with torch.no_grad():
                            g = grad.detach().float() / trainer.scaler.get_scale()
                            self.gradients[key] = dict(norm=float(g.norm()), finite=bool(torch.isfinite(g).all()))
                    return grad
                self.handles.append(p.register_hook(observe))

    def epoch_start(self, trainer):
        self.local_batch = 0
        self.samples = []

    def batch_start(self, trainer):
        trainer._oom_retries = 3  # Preserve B16; native OOM auto-downsizing is disallowed.
        self.step = trainer.epoch * len(trainer.train_loader) + self.local_batch
        trainer.model.model[17].tcr.capture = self.step % 100 == 0
        self.gradients = {}

    def batch_end(self, trainer):
        from tcr_v1_ops import update_state
        update_state(self.dispatch, phase="RUNNING", pid=os.getpid(), epoch=trainer.epoch + 1,
                     micro_batch=self.step + 1, loss=float(trainer.loss.detach()))
        module = trainer.model.model[17].tcr
        if module.capture:
            stats = deepcopy(module.last_stats)
            stats.update(dispatch=self.dispatch, epoch=trainer.epoch + 1, micro_batch=self.step,
                         gradient_rule="current backward contribution divided by GradScaler scale, before clipping",
                         gradients=self.gradients, weights={n:float(p.detach().float().norm()) for n,p in trainer.model.named_parameters() if n in NEW},
                         scale=trainer.scaler.get_scale(), accumulate=trainer.accumulate)
            values_finite = all(g["finite"] for g in self.gradients.values())
            stats["finite"] = values_finite and math.isfinite(stats["residual_ratio"])
            append_json(self.folder / "mechanism_samples.jsonl", stats)
            self.samples.append(stats)
            module.capture = False; module.last_stats = None
        self.local_batch += 1

    def epoch_end(self, trainer):
        if self.last_summary == trainer.epoch:
            return
        self.last_summary = trainer.epoch
        groups = []
        for i in range(8):
            rows = [s["groups"][i] for s in self.samples]
            count = sum(r["elements"] for r in rows)
            groups.append(dict(direction=(0,45,90,135)[i//2], side_step=i%2+1, elements=count,
                               **{k: sum(r[k] for r in rows) for k in ("nonzero","positive","negative","sum_squares")},
                               rms=math.sqrt(sum(r["sum_squares"] for r in rows)/count) if count else None))
            for key in ("nonzero", "positive", "negative"):
                groups[-1][key+"_fraction"] = groups[-1][key]/count if count else None
        ratios = [s["residual_ratio"] for s in self.samples]
        record = dict(dispatch=self.dispatch, epoch=trainer.epoch+1, observed_batches=self.local_batch,
                      samples=len(self.samples), sampling="global micro-batch index divisible by 100 including zero",
                      denominator="B*16*valid_interior_H*valid_interior_W per group, summed over sampled batches",
                      groups=groups, residual_ratio_mean=float(np.mean(ratios)) if ratios else None,
                      residual_ratio_max=max(ratios) if ratios else None,
                      P_weight_norm=self.samples[-1]["weights"]["model.17.tcr.P.weight"] if self.samples else None,
                      O_weight_norm=self.samples[-1]["weights"]["model.17.tcr.O.weight"] if self.samples else None,
                      gradient_samples=[s["gradients"] for s in self.samples],
                      event="ALWAYS_ZERO_SAMPLED_RESIDUAL" if trainer.epoch > 0 and ratios and max(ratios)==0 else None)
        append_json(self.folder / "mechanism_epochs.jsonl", record)
        if record["event"]:
            print("TCR DIAGNOSTIC:", record["event"], "epoch", trainer.epoch+1, flush=True)

    def close(self, trainer):
        for handle in self.handles: handle.remove()


def worker(dispatch, resume=False):
    from tcr_v1_ops import update_state, verify_prepared, ensure_gate
    folder = OUT / "dispatches" / dispatch
    code = 1
    try:
        prepared = verify_prepared()
        ensure_gate(prepared)
        update_state(dispatch, phase="SETTING_UP", pid=os.getpid(), started=utc())
        args = YAML.load(OUT / "train_args.yaml")
        if resume:
            args.update(model=str(RUN / "weights/last.pt"), resume=str(RUN / "weights/last.pt"))
        else:
            require(not RUN.exists(), "Formal results appeared after dispatch; preserved")
        trainer = TCRTrainer(overrides=args, evidence=folder, dispatch=dispatch)
        logger = MechanismLogger(folder, dispatch)
        for event, method in (("on_train_start",logger.setup), ("on_train_epoch_start",logger.epoch_start),
                              ("on_train_batch_start",logger.batch_start), ("on_train_batch_end",logger.batch_end),
                              ("on_train_epoch_end",logger.epoch_end), ("on_train_end",logger.close)):
            trainer.add_callback(event,method)
        trainer.train()
        code = 0
        update_state(dispatch, phase="COMPLETED", epoch=trainer.epoch+1, actual_epochs=trainer.epoch+1,
                     early_stopped=trainer.epoch+1 < args["epochs"], finished=utc())
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        update_state(dispatch, phase="FAILED", error=repr(error), traceback=traceback.format_exc(), finished=utc())
        raise
    finally:
        write_json(folder / "python_exit.json", dict(dispatch=dispatch, python_exit_code=code, finished=utc(),
                   shutdown_thread_warnings="See console; separate from main-process exit code"))
