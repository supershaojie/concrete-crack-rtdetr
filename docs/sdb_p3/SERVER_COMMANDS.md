# SDB-P3 server commands

Use the existing `rtdetr` environment. These commands do not install/upgrade packages. The first run prepares only the main `cbr_lif_sdb_p3_v1` variant. The single-module YAML is available for a separately authorized ablation. Run each complete block separately; every block is a child bash so failures return to the interactive shell. All expensive checks run sequentially.

## Synchronization and environment

The full delivered SHA below is filled at delivery. The existing main repository must already contain the successful base `a0459d6a652cb702699087c88fa39a3e4c4087ec`. No clone/reset/force-push/delete is used. Existing worktrees with different commits are preserved; choose a new path explicitly if needed.

```bash
bash <<'SDB_SYNC'
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
SDB_SHA=FINAL_FULL_SHA
SDB_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SDB_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1
case "$(git -C "$SDB_MAIN" remote get-url origin)" in
  https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo 'Unexpected origin; stop before fetch'; exit 1 ;;
esac
if ! git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}" 2>/dev/null; then
  for SDB_TRY in 1 2 3 4 5; do
    GIT_TERMINAL_PROMPT=0 timeout 120 git -C "$SDB_MAIN" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags --no-write-fetch-head origin refs/heads/exp-rtdetr-r18-lite-sdb-p3-v1:refs/remotes/origin/exp-rtdetr-r18-lite-sdb-p3-v1 || true
    git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}" 2>/dev/null && break
  done
fi
git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}"
SDB_SCRIPT=$(mktemp /tmp/sdb-p3-sync.XXXXXX)
git -C "$SDB_MAIN" show "$SDB_SHA:tools/sync_sdb_p3.sh" > "$SDB_SCRIPT"
bash "$SDB_SCRIPT" "$SDB_SHA" "$SDB_MAIN" "$SDB_WORKTREE"
source "$SDB_WORKTREE/outputs/sdb_p3.env"
test "$(git rev-parse HEAD)" = "$SDB_P3_SHA"
python tools/train_sdb_p3.py --help
SDB_SYNC
```

If all five bounded fetches fail, copy the delivered `.bundle` and `.bundle.sha256` to `/root/autodl-tmp/sdb-p3-delivery/` using your normal file transfer, then import the verified incremental bundle. Replace `DELIVERED_BUNDLE_FILENAME` with the supplied filename, retaining its SHA256 sidecar. This uses the existing successful base and preserves the main checkout.

```bash
bash <<'SDB_BUNDLE'
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
SDB_SHA=FINAL_FULL_SHA
SDB_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SDB_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1
SDB_BUNDLE=/root/autodl-tmp/sdb-p3-delivery/DELIVERED_BUNDLE_FILENAME
(cd "$(dirname "$SDB_BUNDLE")" && sha256sum --check "$(basename "$SDB_BUNDLE").sha256")
git -C "$SDB_MAIN" bundle verify "$SDB_BUNDLE"
git -C "$SDB_MAIN" fetch --no-tags --no-write-fetch-head "$SDB_BUNDLE" "refs/heads/exp-rtdetr-r18-lite-sdb-p3-v1:refs/sdb-p3-delivery/$SDB_SHA"
git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}"
SDB_SCRIPT=$(mktemp /tmp/sdb-p3-sync.XXXXXX)
git -C "$SDB_MAIN" show "$SDB_SHA:tools/sync_sdb_p3.sh" > "$SDB_SCRIPT"
bash "$SDB_SCRIPT" "$SDB_SHA" "$SDB_MAIN" "$SDB_WORKTREE" "$SDB_BUNDLE"
source "$SDB_WORKTREE/outputs/sdb_p3.env"
test "$(git rev-parse HEAD)" = "$SDB_P3_SHA"
SDB_BUNDLE
```

## Controlled initialization and server preflight

Native AMP validation needs the compatible local `yolo26n.pt` check model and `bus.jpg`. Copy trusted resources from the main checkout when absent; verify that the selected weight is compatible with this checkout. The fallback official downloads below are explicit preparation commands, not automatic dependency upgrades. If resources are unavailable, leave server preflight PENDING.

```bash
bash <<'SDB_CHECK'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1/outputs/sdb_p3.env
if [ ! -f yolo26n.pt ]; then
  if [ -f "$SDB_P3_MAIN/yolo26n.pt" ]; then cp -n "$SDB_P3_MAIN/yolo26n.pt" yolo26n.pt;
  elif [ -f "$SDB_P3_MAIN/weights/yolo26n.pt" ]; then cp -n "$SDB_P3_MAIN/weights/yolo26n.pt" yolo26n.pt;
  else
    SDB_RESOURCE=$(mktemp ./sdb-amp-weight.XXXXXX)
    curl --fail --location --max-time 120 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt --output "$SDB_RESOURCE"
    test -s "$SDB_RESOURCE"
    mv -n "$SDB_RESOURCE" yolo26n.pt
  fi
fi
if [ ! -f ultralytics-main/ultralytics/assets/bus.jpg ]; then
  mkdir -p ultralytics-main/ultralytics/assets
  if [ -f "$SDB_P3_MAIN/ultralytics-main/ultralytics/assets/bus.jpg" ]; then
    cp -n "$SDB_P3_MAIN/ultralytics-main/ultralytics/assets/bus.jpg" ultralytics-main/ultralytics/assets/bus.jpg
  else
    SDB_RESOURCE=$(mktemp ./sdb-amp-image.XXXXXX)
    curl --fail --location --max-time 120 https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg --output "$SDB_RESOURCE"
    test -s "$SDB_RESOURCE"
    mv -n "$SDB_RESOURCE" ultralytics-main/ultralytics/assets/bus.jpg
  fi
fi
python tools/init_sdb_p3.py --source "$SDB_P3_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" --output weights/cbr_lif_sdb_p3_v1_controlled_init.pt --report outputs/sdb_p3/cbr_lif_sdb_p3_v1/initialization.json
python tools/check_sdb_p3.py --source "$SDB_P3_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" --initialized weights/cbr_lif_sdb_p3_v1_controlled_init.pt --variant cbr_lif_sdb_p3_v1 --server --data "$SDB_P3_MAIN/configs/crack_autodl.yaml" --output outputs/sdb_p3_preflight/cbr_lif_sdb_p3_v1
python tools/train_sdb_p3.py plan
SDB_CHECK
```

`plan` requires an exact current-code PASSED server report, B16/640/native AMP and at least two effective optimizer updates, the fixed source and initialization hashes, and the successful parent's actual split/label fingerprints. Counts alone cannot satisfy the data gate. A failed/PENDING check cannot start training. Preflight output and initial state are preserved, and formal run output is untouched.

For a repeat after a source fix, create a new uniquely named preflight output and pass that `checks.json` via `train_sdb_p3.py plan --preflight ...`; preserve the old failed report. A changed existing plan is intentionally refused and must be reviewed rather than overwritten.

## Explicit formal start

This is the only initial formal-training command. Run it only when you intend to start. The unique run directory is `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/cbr_lif_sdb_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug`. No `name2`, reduced batch or disabled AMP is accepted.

```bash
bash <<'SDB_START'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1/outputs/sdb_p3.env
python tools/train_sdb_p3.py start
python tools/train_sdb_p3.py status
SDB_START
```

The worker runs in `tmux`, writes real exit codes and preserves failed outputs. Inspect the `latest_attempt` path printed by `status`; its `console.log`, `state.json` and `process_exit_code.txt` are authoritative together. States distinguish `COMPLETED_200`, `EARLY_STOPPED`, `FAILED` and `INTERRUPTED`. Epoch 40/80 observations compare the successful parent's same epoch only, without changing patience or schedule.

## Resume or status

Resume requires an actual unfinished `last.pt` including epoch and optimizer state. It retains learned SDB state and restores native optimizer/scaler/EMA state. A final-validation failure after completed training is recorded separately and cannot silently restart training.

```bash
bash <<'SDB_RESUME'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1/outputs/sdb_p3.env
python tools/train_sdb_p3.py status
python tools/train_sdb_p3.py resume
SDB_RESUME
```

For status alone:

```bash
bash <<'SDB_STATUS'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1/outputs/sdb_p3.env
python tools/train_sdb_p3.py status
SDB_STATUS
```

## Independent val, then explicit test

Run only after training has finished. `val` evaluates training-selected `best.pt`; `test` demands the same hash, code, data and `corrected_sorted_conf_mask_v1` policy. Test is never run by training, preflight or packaging. Each evaluation writes full-precision JSON, per-IoU AP and plots. Keep the two blocks separate so test remains an explicit decision.

```bash
bash <<'SDB_VAL'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1/outputs/sdb_p3.env
python tools/eval_sdb_p3.py val
SDB_VAL
```

```bash
bash <<'SDB_TEST'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1/outputs/sdb_p3.env
python tools/eval_sdb_p3.py test
SDB_TEST
```

## Light evidence package

This packages only existing evidence. Missing val/test or unfinished training is listed explicitly. Weights, data, huge prediction exports and reference ZIPs are excluded; log tails are bounded to 64 KiB and stored as `.txt`. Every member is read back and checked against the SHA256 manifest. The target is less than 20 MiB; a larger package is retained and its largest members listed for review.

```bash
bash <<'SDB_PACK'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1/outputs/sdb_p3.env
python tools/pack_sdb_p3.py --output "$SDB_P3_MAIN/downloads/sdb_p3/cbr_lif_sdb_p3_v1_light_$(date +%Y%m%d_%H%M%S).tar.gz"
SDB_PACK
```

At implementation delivery: formal training **NOT_STARTED**; final test **NOT_RUN**. Server-only checks remain **PENDING** until actually executed on the server.
