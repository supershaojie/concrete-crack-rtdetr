# PSDB-P3 server commands

Use the existing `rtdetr` environment. These commands do not install/upgrade packages. The first run prepares only the main `cbr_lif_psdb_p3_v1` variant. The single-module YAML is available for a separately authorized ablation. Run each complete block separately; every block is a child bash so failures return to the interactive shell. All expensive checks run sequentially.

## Synchronization and environment

The full delivered SHA below is filled at delivery. The existing main repository must already contain the successful base `a0459d6a652cb702699087c88fa39a3e4c4087ec`. No clone/reset/force-push/delete is used. Existing worktrees with different commits are preserved; choose a new path explicitly if needed.

```bash
bash <<'PSDB_SYNC'
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
PSDB_SHA=FINAL_FULL_SHA
PSDB_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
PSDB_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1
case "$(git -C "$PSDB_MAIN" remote get-url origin)" in
  https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo 'Unexpected origin; stop before fetch'; exit 1 ;;
esac
if ! git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}" 2>/dev/null; then
  for PSDB_TRY in 1 2 3 4 5; do
    GIT_TERMINAL_PROMPT=0 timeout 120 git -C "$PSDB_MAIN" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags --no-write-fetch-head origin refs/heads/exp-rtdetr-r18-lite-psdb-p3-v1:refs/remotes/origin/exp-rtdetr-r18-lite-psdb-p3-v1 || true
    git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}" 2>/dev/null && break
  done
fi
git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}"
PSDB_SCRIPT=$(mktemp /tmp/psdb-p3-sync.XXXXXX)
git -C "$PSDB_MAIN" show "$PSDB_SHA:tools/sync_psdb_p3.sh" > "$PSDB_SCRIPT"
bash "$PSDB_SCRIPT" "$PSDB_SHA" "$PSDB_MAIN" "$PSDB_WORKTREE"
source "$PSDB_WORKTREE/outputs/psdb_p3.env"
test "$(git rev-parse HEAD)" = "$PSDB_P3_SHA"
python tools/train_psdb_p3.py --help
PSDB_SYNC
```

If all five bounded fetches fail, copy the delivered `.bundle` and `.bundle.sha256` to `/root/autodl-tmp/psdb-p3-delivery/` using your normal file transfer, then import the verified incremental bundle. Replace `DELIVERED_BUNDLE_FILENAME` with the supplied filename, retaining its SHA256 sidecar. This uses the existing successful base and preserves the main checkout.

```bash
bash <<'PSDB_BUNDLE'
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
PSDB_SHA=FINAL_FULL_SHA
PSDB_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
PSDB_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1
PSDB_BUNDLE=/root/autodl-tmp/psdb-p3-delivery/DELIVERED_BUNDLE_FILENAME
(cd "$(dirname "$PSDB_BUNDLE")" && sha256sum --check "$(basename "$PSDB_BUNDLE").sha256")
git -C "$PSDB_MAIN" bundle verify "$PSDB_BUNDLE"
git -C "$PSDB_MAIN" fetch --no-tags --no-write-fetch-head "$PSDB_BUNDLE" "refs/heads/exp-rtdetr-r18-lite-psdb-p3-v1:refs/psdb-p3-delivery/$PSDB_SHA"
git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}"
PSDB_SCRIPT=$(mktemp /tmp/psdb-p3-sync.XXXXXX)
git -C "$PSDB_MAIN" show "$PSDB_SHA:tools/sync_psdb_p3.sh" > "$PSDB_SCRIPT"
bash "$PSDB_SCRIPT" "$PSDB_SHA" "$PSDB_MAIN" "$PSDB_WORKTREE" "$PSDB_BUNDLE"
source "$PSDB_WORKTREE/outputs/psdb_p3.env"
test "$(git rev-parse HEAD)" = "$PSDB_P3_SHA"
PSDB_BUNDLE
```

## Controlled initialization and server preflight

先用 `nvidia-smi` 查看占用；保留正在运行的 SDB 进程，资源不足则暂缓本段。默认只预检主组合，B16/640/native AMP 不降配。预检成功不会启动正式训练。

原生 AMP 检查仅复用本地可信的 `yolo26n.pt` 和 `bus.jpg`；不下载或升级。缺资源保持 PENDING。

```bash
bash <<'PSDB_CHECK'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1/outputs/psdb_p3.env
nvidia-smi
if [ ! -f yolo26n.pt ]; then
  if [ -f "$PSDB_P3_MAIN/yolo26n.pt" ]; then cp -n "$PSDB_P3_MAIN/yolo26n.pt" yolo26n.pt;
  elif [ -f "$PSDB_P3_MAIN/weights/yolo26n.pt" ]; then cp -n "$PSDB_P3_MAIN/weights/yolo26n.pt" yolo26n.pt;
  else echo 'PENDING: trusted local AMP check weight yolo26n.pt missing'; exit 1; fi
fi
if [ ! -f ultralytics-main/ultralytics/assets/bus.jpg ]; then
  test -f "$PSDB_P3_MAIN/ultralytics-main/ultralytics/assets/bus.jpg" || { echo 'PENDING: AMP bus.jpg missing'; exit 1; }
  mkdir -p ultralytics-main/ultralytics/assets
  cp -n "$PSDB_P3_MAIN/ultralytics-main/ultralytics/assets/bus.jpg" ultralytics-main/ultralytics/assets/bus.jpg
fi
python tools/init_psdb_p3.py --source "$PSDB_P3_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" --output weights/cbr_lif_psdb_p3_v1_controlled_init.pt --report outputs/psdb_p3/cbr_lif_psdb_p3_v1/initialization.json
python tools/check_psdb_p3.py --source "$PSDB_P3_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" --initialized weights/cbr_lif_psdb_p3_v1_controlled_init.pt --variant cbr_lif_psdb_p3_v1 --server --data "$PSDB_P3_MAIN/configs/crack_autodl.yaml" --output outputs/psdb_p3/cbr_lif_psdb_p3_v1/preflight
python tools/train_psdb_p3.py plan
PSDB_CHECK
```

`plan` requires an exact current-code PASSED server report, B16/640/native AMP and at least two effective optimizer updates, the fixed source and initialization hashes, and the successful parent's actual split/label fingerprints. Counts alone cannot satisfy the data gate. A failed/PENDING check cannot start training. Preflight output and initial state are preserved, and formal run output is untouched.

For a repeat after a source fix, create a new uniquely named preflight output and pass that `checks.json` via `train_psdb_p3.py plan --preflight ...`; preserve the old failed report. A changed existing plan is intentionally refused and must be reviewed rather than overwritten.

## Explicit formal start

This is the only initial formal-training command. Run it only when you intend to start. The unique run directory is `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/cbr_lif_psdb_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug`. No `name2`, reduced batch or disabled AMP is accepted.

```bash
bash <<'PSDB_START'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1/outputs/psdb_p3.env
python tools/train_psdb_p3.py start
python tools/train_psdb_p3.py status
PSDB_START
```

The worker runs in `tmux`, writes real exit codes and preserves failed outputs. Inspect the `latest_attempt` path printed by `status`; its `console.log`, `state.json` and `process_exit_code.txt` are authoritative together. States distinguish `COMPLETED_200`, `EARLY_STOPPED`, `FAILED` and `INTERRUPTED`. Epoch 40/80 observations compare the successful parent's same epoch only, without changing patience or schedule.

## Resume or status

Resume requires an actual unfinished `last.pt` including epoch and optimizer state. It retains learned PSDB state and restores native optimizer/scaler/EMA state. A final-validation failure after completed training is recorded separately and cannot silently restart training.

```bash
bash <<'PSDB_RESUME'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1/outputs/psdb_p3.env
python tools/train_psdb_p3.py status
python tools/train_psdb_p3.py resume
PSDB_RESUME
```

For status alone:

```bash
bash <<'PSDB_STATUS'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1/outputs/psdb_p3.env
python tools/train_psdb_p3.py status
PSDB_STATUS
```

## Independent val, then explicit test

Run only after training has finished. `val` evaluates training-selected `best.pt`; `test` demands the same hash, code, data and `corrected_sorted_conf_mask_v1` policy. Test is never run by training, preflight or packaging. Each evaluation writes full-precision JSON, per-IoU AP and plots. Keep the two blocks separate so test remains an explicit decision.

```bash
bash <<'PSDB_VAL'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1/outputs/psdb_p3.env
python tools/eval_psdb_p3.py val
PSDB_VAL
```

```bash
bash <<'PSDB_TEST'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1/outputs/psdb_p3.env
python tools/eval_psdb_p3.py test
PSDB_TEST
```

## Light evidence package

This packages only existing evidence. Missing val/test or unfinished training is listed explicitly. Weights, data, huge prediction exports and reference ZIPs are excluded; log tails are bounded to 64 KiB and stored as `.txt`. Every member is read back and checked against the SHA256 manifest. The target is less than 20 MiB; a larger package is retained and its largest members listed for review.

```bash
bash <<'PSDB_PACK'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1/outputs/psdb_p3.env
python tools/pack_psdb_p3.py --output "$PSDB_P3_MAIN/downloads/psdb_p3/cbr_lif_psdb_p3_v1_light_$(date +%Y%m%d_%H%M%S).tar.gz"
PSDB_PACK
```

At implementation delivery: formal training **NOT_STARTED**; final test **NOT_RUN**. Server-only checks remain **PENDING** until actually executed on the server.
