# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Detached scalar/histogram telemetry. No random sampling, feature-map dumping or graphs."""
from __future__ import annotations
import numpy as np
import torch


class PEQStatistics:
    def __init__(self):
        self.total_batches = 0
        self.pending_gradient_sample = False
        self.reset()

    def reset(self):
        self.batches = self.queries = self.matched = self.unmatched = 0
        self.matched_batches = self.unmatched_batches = 0
        self.lq = self.lm = self.lu = 0.
        self.delta_sum = self.delta_abs = self.score_change = self.raw_sum = self.final_sum = 0.
        self.near_limit = self.monotonic_violations = self.monotonic_pairs = 0
        self.hist = np.zeros(400, dtype=np.int64)
        self.positives = np.zeros(10, dtype=np.int64)
        self.events = []

    def observe(self, payload, loss):
        self.batches += 1
        self.total_batches += 1
        delta = payload["delta"].detach().float().cpu().numpy()
        raw = payload["s_raw"].detach().float().cpu().numpy()
        final = payload["s_final"].detach().float().cpu().numpy()
        self.queries += raw.size
        self.matched += loss["matched_count"]
        self.unmatched += loss["unmatched_count"]
        self.lq += loss["loss_peq"]
        for key, attr, denominator in (("loss_matched", "lm", "matched_batches"),
                                       ("loss_unmatched", "lu", "unmatched_batches")):
            if loss[key] is not None:
                setattr(self, attr, getattr(self, attr) + loss[key])
                setattr(self, denominator, getattr(self, denominator) + 1)
        self.positives += np.asarray(loss["target_positive_count"], dtype=np.int64)
        self.delta_sum += float(delta.astype(np.float64).sum())
        self.delta_abs += float(np.abs(delta).astype(np.float64).sum())
        self.hist += np.histogram(delta, bins=400, range=(-1, 1))[0]
        self.near_limit += int((np.abs(delta) >= .95).sum())
        self.monotonic_violations += int((delta[..., 1:] > delta[..., :-1]).sum())
        self.monotonic_pairs += delta[..., 1:].size
        self.raw_sum += float(raw.astype(np.float64).sum())
        self.final_sum += float(final.astype(np.float64).sum())
        self.score_change += float(np.abs(final-raw).astype(np.float64).sum())
        if (self.total_batches - 1) % 100 == 0:
            gap = payload["evidence_difference"].detach().float().norm(dim=-1)
            oob = torch.stack((payload["oob0"], payload["oob1"]))
            self.events.append(dict(batch=self.total_batches, queries=gap.numel(),
                                    evidence_difference_norm_mean=float(gap.mean()),
                                    out_of_bounds_points=int(oob.sum()), sampled_points=oob.numel(),
                                    out_of_bounds_fraction=float(oob.float().mean())))
            self.pending_gradient_sample = True

    def gradient_event(self, model, step):
        if not self.pending_gradient_sample:
            return
        self.pending_gradient_sample = False
        norms = {}
        for name, parameter in model.named_parameters():
            if ".peq." in name or name in ("model.20.O_proj.weight", "model.26.cbr.offset_out.weight"):
                norms[name] = dict(parameter_norm=float(parameter.detach().float().norm()),
                                   unscaled_gradient_norm=None if parameter.grad is None else
                                   float(parameter.grad.detach().float().norm()))
        self.events.append(dict(type="gradient", optimizer_step=step, batch=self.total_batches, norms=norms))

    def report(self):
        n = self.queries * 10
        cumulative = self.hist.cumsum()
        quantiles = {str(p): None if not n else float(-1 + (min(
            int(np.searchsorted(cumulative, p * n, side="left")), 399)+.5)/200)
                     for p in (.01, .10, .50, .90, .99)}
        return dict(batches=self.batches, queries=self.queries, matched=self.matched, unmatched=self.unmatched,
                    loss_peq_batch_mean=self.lq/self.batches if self.batches else None,
                    loss_matched_batch_mean=self.lm/self.matched_batches if self.matched_batches else None,
                    loss_unmatched_batch_mean=self.lu/self.unmatched_batches if self.unmatched_batches else None,
                    matched_loss_batches=self.matched_batches, unmatched_loss_batches=self.unmatched_batches,
                    target_positive_count=self.positives.tolist(),
                    positive_rate_all_queries=(self.positives/self.queries).tolist() if self.queries else None,
                    positive_rate_matched=(self.positives/self.matched).tolist() if self.matched else None,
                    delta_elements=n, delta_mean=self.delta_sum/n if n else None,
                    delta_abs_mean=self.delta_abs/n if n else None,
                    delta_quantiles_histogram_approx=quantiles, delta_histogram_bin_width=.005,
                    delta_near_limit_count=self.near_limit, delta_near_limit_fraction=self.near_limit/n if n else None,
                    s_raw_mean=self.raw_sum/self.queries if self.queries else None,
                    s_final_mean=self.final_sum/self.queries if self.queries else None,
                    absolute_score_change_mean=self.score_change/self.queries if self.queries else None,
                    monotonic_violation_count=self.monotonic_violations,
                    monotonic_comparisons=self.monotonic_pairs,
                    monotonic_violation_rate=self.monotonic_violations/self.monotonic_pairs if self.monotonic_pairs else None,
                    events=list(self.events), event_count=len(self.events),
                    sampling="first and every 100 micro-batches; gradient sample at next optimizer step")
