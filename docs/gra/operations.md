# GRA operations and provenance

`tools/train_gra.py plan` only resolves the archived 109-field recipe. `start`
requires a full delivery SHA, controlled source/init audit, full mathematics and
learned-state checks, and a successful bounded capacity preflight from the actual
server worktree/environment. Source-content hashes, public/init weight hashes,
data configuration and complete split/label inventories must still match.
`resume` requires this run's unfinished native `last.pt` with optimizer, scaler,
epoch and EMA. Neither command silently chooses a new output name.

The archived parent args were read from
`D:/rtdetr跑结果/C19＋LIF：原 CBR＋LIF-Down v1/training/args.yaml` and compared
field by field with Appendix A in the requested prompt: all 109 fields match.
`recipe_audit.json` records the original file hash and both complete field diffs.
`resolved_formal_config.yaml` and `resolved_ablation_config.yaml` use POSIX server
paths; only model, name and save_dir differ from the parent. The ablation is
provided for future explicit use and is not launched by the main helper.

`parent_data_identity.json` is an unchanged copy of
`D:/rtdetr跑结果/C19＋LIF：原 CBR＋LIF-Down v1/metadata/launch/dataset_inventory.json`.
Its SHA256 is `8567e1b4d1aae5f168fb18b633ed4357c7f0aba56c0041a6570647fb542d0446`.
Data checking requires the complete original image-path and label hashes, beyond
the 6048/1728/864 split counts. Reading test label identity does not run test
inference or use test metrics to select the model.

`preflight_gra.py` uses the actual native Trainer loop and 200-epoch configuration,
including original warmup, accumulation, augmentation and GradScaler. Its private
callback exits after two actual optimizer updates or at most 16 batches, before
epoch validation or checkpoint saving. Optimizer post-step hooks count actual
updates, so skipped scaler steps never count. Its directory, model and optimizer
are disposable. GPU processes are displayed for context; concurrent experiments
are permitted. OOM is a failure with the original B16/640 configuration.

`eval_gra.py evaluate` is a separate, explicit command. It freezes native selected
best.pt by SHA, uses independent FP32 evaluation with
`corrected_sorted_conf_mask_v1`, and exports native fixed-pool TP flags with full
precision metrics. `test` requires completed same-weight/config/source `val`.
`fixed-p` only consumes complete existing val exports. It scans original unique
score thresholds, includes entire equal-score groups, and uses native one-to-one
matching at IoU 0.50. It is separate from formal maximum-F1 P/R and does not alter
the formal evaluation `iou=0.7`. A missing parent/candidate prediction export is
PENDING. Full prediction streams remain outside the lightweight archive.

`pack_gra_light.py` packages existing evidence only, includes missing-evidence
fields, verifies each member's checksum and enforces a compressed size below
20 MiB. Weights, data, reference ZIPs and complete predictions are excluded.
Native lifecycle records distinguish RUNNING, COMPLETED_200, EARLY_STOPPED,
INTERRUPTED and FAILED and record `final_eval` separately. A failed final val after
200 completed epochs is not an instruction to retrain.

`check_gra_ops.py` exercises CLI help/plan/rejection, the real validator matching
path on a small fixture, equal-score threshold grouping, no-achievement semantics,
and archive integrity. It does not run training, full validation or test.
