# SRE implementation and handoff

Branch: `exp-rtdetr-r18-lite-sre-v1`. Fixed successful base: `a0459d6a652cb702699087c88fa39a3e4c4087ec`. The main checkout and its uncommitted artifacts were preserved; all changes are in the independent linked worktree.

## Delivered experiment

| Variant | Configuration | Run name |
|---|---|---|
| `cbr_lif_sre_v1` | `rtdetr-resnet18-lite-cbr-lif-sre-v1.yaml` | `cbr_lif_sre_v1_rtdetr_r18_lite_e200_b16_onlineaug` |
| `sre_v1` | `rtdetr-resnet18-lite-sre-v1.yaml` | `sre_v1_rtdetr_r18_lite_e200_b16_onlineaug` |

See [STRUCTURE.md](STRUCTURE.md) for the mathematical/graph contract and [SPEC.md](SPEC.md) for the supplied specification. Original CBR and LIF-Down LF hashes match exactly. Only the complete original node19 output is enriched. Both variants add **16,704 parameters**; no extra detection branch or alternative third module is included.

`init_sre.py` reuses the successful parent's controlled initialization, audits common parameters and buffers individually, allows exactly nine native nc80-to-nc1 classifier adaptations, verifies actual RTDETR.train reconstruction without stepping training, and immediately reloads the saved FP32 initialization. It never initializes from trained parent best/last weights. Both SRE initial states are identical.

## Actual local evidence

The local environment is Python 3.9.25 / PyTorch 2.7.1+cu118 on Windows, with RTX 2060 6 GiB. The successful server environment was not entered or changed. No package upgrade was performed.

| Check | Result and scope |
|---|---|
| Source checkpoint | SHA256 matches `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e` |
| Both controlled initializations | PASSED common-state, new-state, native class-adaptation and immediate reload audits |
| Math and boundaries | PASSED independent per-pixel oracle, hand cases, positive differences before sum, fixed denominator, single-axis/1x1/odd/rectangular inputs, input/parameter gradients |
| Structure | PASSED 27 nodes, one wrapper, three original internal RepConv, hooks at full RepC3 output and both consumers |
| Parent identity | PASSED synchronized CPU/CUDA RNG, BN, valid GT/DN, exact initial training outputs/loss |
| Small detection-loss checks | PASSED CPU FP32, CUDA FP32 and native AMP, both variants, two real optimizer updates on disposable synthetic B2/160 copies |
| AMP startup | Six native scale backoffs from 65536 to 1024, then two effective updates; no forced scale=1 or AMP disable |
| Learned-state lifecycle | PASSED save/reload, native EMA formula, fusion state preservation, explicit CUDA half reload and fused-half finite inference |
| B1/640 geometry | PASSED node19 256x80x80; node20/22 256x40x40; node25 256x20x20; native 300-query output finite |
| Parameter counts | Main 20,166,469 / 19,961,669; ablation 20,099,476 / 19,894,420 unfused/fused |
| Original data identity | PASSED all train/val/test relative image-list and label hashes; counts6048/1728/864. Reading test inventory is not test evaluation |
| Recall and package checks | PASSED ties, no-achieved/empty cases, original one-to-one matching, synthetic two-image validator capture, archive exclusions and hash verification |
| Server B16/640/native AMP | PENDING until an actual server preflight passes; small local checks cannot replace it |
| Native save/resume methods | PASSED both variants on disposable synthetic CUDA copies: actual save_model/get_model/resume_training restored optimizer/scaler/EMA, start_epoch1 and learned SRE; full B16 preflight remains separate |
| Reference module ZIP | PENDING/not found in bounded local search; no archive contents or scripts were claimed read/executed |

Fusion diagnostics keep default TF32 behavior and separately use a scoped strict-FP32 comparison with flags restored. The main combination's native top-k returned the same candidate set in a different order: original rowwise output differences are recorded, while outputs aligned by their original candidate IDs match to about 1.19e-7. Fixed-candidate replay is labeled diagnostic only. Half comparisons use the same quantized checkpoint values and do not claim raw FP16/FP32 output equivalence.

All measurements are engineering evidence, not accuracy evidence. The projection subtotal 0.212992 GFLOPs is explicitly PARTIAL; functional neighborhood/GN/normalization/activation work is not silently omitted from a claimed complete total.

The actual B16 attempt first hit a Windows permission error writing the old label cache. Its failure is preserved. Preflight now redirects only its disposable cache writes to the temporary directory, retaining native data loading and augmentation. A second attempt passed native AMP checking and scanned6048 train/1728 val images, then stalled in original8-worker startup before any batch/update. It was interrupted; raw RUNNING evidence is preserved alongside a separate PENDING receipt, not rewritten into a pass. No B16 peak-memory or batch-time result is claimed. Server capacity now has a declared600-second setup/batch wall budget plus24 batches; the handoff also bounds the entire preflight to1800 seconds.

Committed `reports/` files preserve local measurements captured during development, including their original identities. Final-SHA evidence and handoff receipts are separate delivery artifacts. Never modify an old report's status or commit to satisfy the start gate; execute the server preflight at the delivered commit.

## Recipe and data

`parent_args.yaml` is copied from the actual successful parent training record and equals the supplied 109-field appendix. `parent_data.yaml` and `parent_dataset_inventory.json` archive its data identity. `train_sre.py plan` produces `planned_args.yaml` and a field-by-field `recipe_diff.json`; only model, output/name and verified equivalent data paths may differ. Original AdamW, lr0=.0005, 200 epochs, patience50, batch16, 640, seed42, native AMP and all online augmentation values remain fixed.

Start rejects stale commit/code/config/data/init/source identities, an existing run, a non-PASSED bounded preflight, incorrect optimizer coverage or missing native resume evidence. No automatic name2, lower batch, alternate recipe, extra seed or threshold tuning is used. Resume accepts only this run's genuine unfinished last.pt. Training completion and final validation are recorded separately; a validation failure after200 epochs does not authorize retraining200 epochs.

## Recall diagnostic

The original formal P/R retain each model's maximum-F1 operating point. Independent evaluation reuses `corrected_sorted_conf_mask_v1` with no added NMS. It stores full-precision P/R/AP50/AP75/mAP50-95, all IoU AP columns, curves, confusion matrix, speed and evidence hashes.

The additional diagnostic is val-only: fixed original-validator TP@IoU0.50 labels on the conf>.001 / max300 candidate pool; unique unrounded score thresholds use `score >= threshold` and include complete tie groups. Among points with P>=.8656, maximize R, then P, then threshold. No nonempty qualifying point yields NOT_ACHIEVED and null threshold/R_at_P. No per-threshold rematching, interpolation or empty-prediction shortcut is used.

The archived successful parent val predictions were replayed once through the original matcher without inference. Verified best SHA: `24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`. Derived R_at_P=.835202492211838, threshold=.6047021150588989, P=.8658162441466172, TP10724/FP1662/FN2116, GT12840. This was independently calculated even though its Recall happens to equal the historical maximum-F1 Recall. P/R/AP50 reconstruct exactly; tiny higher-IoU AP differences caused by archived FP32 coordinate round-trips are explicitly retained in the report. New evaluations capture TP labels directly.

Candidate comparison remains PENDING until training and independent val exist. Parent/candidate thresholds may differ. Test must use the same frozen best checkpoint after val; test is never used to select thresholds, modules or epochs.

## Commands and delivery

Each tool's actual `--help` was executed. Bash blocks are syntax-checked. [SERVER_COMMANDS.template.md](SERVER_COMMANDS.template.md) is the source template; `make_sre_handoff.py --output outputs/sre/delivery/SERVER_COMMANDS.md` renders the final committed40-digit SHA into an independent artifact. Follow its synchronization/environment block, then initialize/plan/preflight only. The explicit start, resume, val/test/diagnostic and light-package blocks are separate.

Remote push and SHA verification results are recorded with the generated handoff. If GitHub is unreachable, use the verified incremental bundle relative to the fixed successful base. The server synchronization checks the object before fetching, uses bounded HTTP/1.1 fetch retries, preserves existing checkouts, and verifies the exact detached-worktree HEAD. It never clones over the main directory, resets a checkout, stops another process, or upgrades the environment.

**Formal training NOT_STARTED. Final test NOT_RUN. No server login occurred. No accuracy improvement is claimed.**
