# C26 independent AutoDL lifecycle

Use the existing `rtdetr` environment: Python3.10, PyTorch2.1.2+cu121, RTX4090. No dependency upgrade is performed. The entrypoint sets PYTHONPATH to this worktree and YOLO_AUTOINSTALL=false.

## Synchronize the pinned delivery

Set `C26_SHA` to the full 40-character commit supplied in the delivery message. It is deliberately explicit instead of following a moving branch tip.

```bash
set -euo pipefail
C26_SHA=PASTE_FULL_DELIVERY_COMMIT_SHA
C26_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
git -C "$C26_MAIN" fetch origin codex/c26-cbr-scca
C26_SYNC=$(mktemp /tmp/c26-sync.XXXXXX.sh)
git -C "$C26_MAIN" show "$C26_SHA:tools/sync_c26.sh" > "$C26_SYNC"
bash "$C26_SYNC" "$C26_SHA"
git -C /root/autodl-tmp/projects/Crack_RTDETR-c26 rev-parse HEAD
```

`sync_c26.sh` checks origin and C24 ancestry, then creates the independent branch/worktree if absent. An existing directory must belong to the same Git repository, be clean, and already have the exact SHA. Existing modified checkouts, different commits, branches checked out elsewhere, or unrelated directories stop the command for inspection; there is no reset/clean/forced switch. The main checkout and C25 checkout/outputs/processes are not modified. The Git object store is shared by standard worktree semantics.

## Direct start (manual)

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c26
bash tools/autodl_c26.sh start-direct
```

This is the only command that starts training. It requires the main repository's authoritative C2 `runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`, unified initialization and `configs/crack_autodl.yaml`.

No full prepare/audit, paired full FP32/AMP models, real batch16 smoke, C19 audit marker, CBR box diagnostic or historical reevaluation is required. Records explicitly say `full_server_preflight=NOT_RUN` and `cbr_box_diagnostics=NOT_RUN`. Native lightweight Ultralytics AMP checking remains; its small reference model/image are reused locally when available. No C26 full-model GPU diagnostic is added before training. Normal forwards have no GPU quantiles or persistent activation statistics.

Separate C26 resources:

| Resource | Location/name |
|---|---|
| Worktree | `/root/autodl-tmp/projects/Crack_RTDETR-c26` |
| tmux session | `c26-training` |
| Init | worktree `weights/c26_cbr_scca_controlled_init.pt` |
| Launch records, PID, exits and console | worktree `outputs/c26/` |
| Complete command logs | worktree `outputs/c26_command_logs/` |
| Formal run | main repo `runs/c_series/c26_rtdetr_r18_lite_cbr_scca_e200_b16_onlineaug` |
| Persistent reservation | formal run path + `.c26.lock` |

The atomic C26 reservation prevents duplicate launches across worktrees on the same server; only the C26 tmux session is checked. C25 can run independently, including on another server. Shared GPU memory can still cause OOM: the process preserves logs and exits, without shrinking batch/imgsz, changing AMP/lr, retrying training, or killing other experiments. The Python exit record and shell `process_exit_code.txt` record failure; shell exit also captures interpreter-start failures or signals when the shell survives. Failed reservations/init files are preserved for inspection; the script does not silently reset/reuse them.

## Status and full log

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c26
bash tools/autodl_c26.sh status
```

```bash
less -R /root/autodl-tmp/projects/Crack_RTDETR-c26/outputs/c26/console.log
```

Follow live output with `tail -n 100 -F /root/autodl-tmp/projects/Crack_RTDETR-c26/outputs/c26/console.log`. `status` reads a bounded tail; the file itself is complete. No `tmux attach` is required.

## Independent val, then test (manual, separate)

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c26
bash tools/autodl_c26.sh val
```

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c26
bash tools/autodl_c26.sh test
```

Both require successful formal training and use the fixed `best.pt`. Existing evaluation directories are never overwritten. Settings: imgsz640, batch16, workers0, device0, half=False, conf0.001, iou0.7, max_det300, augment=False, rect=False. Test also verifies the independent val's checkpoint/data/source/settings. Never tune using test.

Outputs are `outputs/c26/evaluation_val/` and `evaluation_test/`. Each contains metrics.json with precision/recall/AP75/mAP50/mAP50–95 and the full `[class,10]` AP array for IoU 0.50…0.95, affected-image counts from the corrected confidence mask, requested/actual settings, weight/config hashes, source/environment, curves/matrices and predicted batch images. Parameter count is measured **before** native inference fusion; the console may also show a lower fused count.

Each same-pass `predictions_gt.jsonl.gz` has one line per split image, including images without GT or accepted predictions. It contains all 300 regular-query refined predictions, including scores below the metric threshold, plus all GT. `used_for_metrics` identifies the thresholded set. Paths are relative to the resolved dataset root, original dimensions are `[height,width]`, classes are zero-based, and boxes are continuous original-pixel `xyxy` after reversing RT-DETR's stretch resize. Values retain FP32 inference precision without decimal rounding/clipping; no second full inference or long-lived GPU result cache is used. Complete split coverage and stream SHA are checked.

## Complete package (manual; never invokes train/val/test)

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c26
bash tools/autodl_c26.sh pack-complete
```

This requires existing successful training and both independent evaluations, their complete prediction streams, curves/matrices/samples and C26 direct-start metadata. Missing output produces an explicit failure, never an automatic evaluation. It does **not** require old C19 audit evidence or pretend the full server preflight ran.

The timestamped archive has no 20 MiB cap and includes:

- `training/`: every existing training output recursively, including best.pt, last.pt, args, CSV, all curves/matrices/labels/train/val batch images.
- `evaluation/val/` and `evaluation/test/`: complete metrics, plots and same-pass predictions/GT streams.
- `metadata/`: direct-launch initialization/loading/optimizer/recipe evidence, actual args, source commit, pinned source snapshot and patch, model/module/tool source, dependency versions, environment, data config, hashes, full completed command/console logs and exit status.
- `MANIFEST.json`: every payload member's byte count and SHA256.

No whole dataset or environment is copied. Packaging streams from disk, rereads every archive member to verify integrity, and publishes the archive only after verification. Existing archives/sidecars are preserved; a failed write may leave a `.partial` file for inspection. The current packaging command's live log stays outside its own archive; all previously completed start/val/test logs are included.

Download the timestamped files from:

```text
/root/autodl-tmp/projects/Crack_RTDETR/downloads/c26/
  c26_complete_YYYYMMDD_HHMMSS_microseconds.tar.gz
  ...tar.gz.sha256
  ...tar.gz.inventory.json
```

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR/downloads/c26
ls -lh c26_complete_*
sha256sum -c c26_complete_*.tar.gz.sha256
```

Use AutoDL's file browser or your existing SCP connection to download these three files. Recheck the hash locally (`sha256sum -c FILE.tar.gz.sha256`, or PowerShell `Get-FileHash -Algorithm SHA256 FILE.tar.gz`). Weights, images and result archives remain on the server, outside Git.
