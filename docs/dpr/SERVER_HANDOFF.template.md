# DPR fixed-commit server handoff

Branch `exp-rtdetr-r18-lite-dpr-v1`; fixed commit `@SHA@`;
base `a0459d6a652cb702699087c88fa39a3e4c4087ec`.
This template is rendered after commit; the delivered rendered copy has no placeholder.
Local checks and server checks are distinct. No server login, formal training,
independent val, or final test was performed during implementation.

## Sync and environment

Run the whole block in the existing server/tmux terminal. A child bash failure
returns to your terminal. It does not reset/clean or alter other worktrees.
The bootstrap obtains the sync script from the fixed commit, not an old checkout.

```bash
bash <<'DPR_SYNC'
set -eo pipefail
S=@SHA@
M=/root/autodl-tmp/projects/Crack_RTDETR
B=exp-rtdetr-r18-lite-dpr-v1
R=$(git -C "$M" remote get-url origin); R=${R%.git}
case "$R" in
  https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr|ssh://git@github.com/supershaojie/concrete-crack-rtdetr) ;;
  *) echo 'Unexpected origin; preserved.' >&2; exit 1 ;;
esac
if ! git -C "$M" cat-file -e "$S^{commit}" 2>/dev/null; then
  for i in 1 2 3 4 5; do
    echo "Fetch attempt $i/5"
    if GIT_TERMINAL_PROMPT=0 timeout 120s git -C "$M" -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags \
      --no-write-fetch-head origin "refs/heads/$B:refs/remotes/origin/$B"; then
      git -C "$M" cat-file -e "$S^{commit}" && break
    else
      rc=$?; echo "Fetch failed: exit $rc" >&2
    fi
    [ "$i" = 5 ] || sleep 3
  done
fi
git -C "$M" cat-file -e "$S^{commit}"
git -C "$M" show "$S:tools/sync_dpr.sh" | bash -s -- "$S"
bash /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1/tools/dpr_server.sh environment
DPR_SYNC
```

## Initialization and finite preflight

This prepares C2+DPR initialization and CPU module checks, then checks only the
main combination's full capacity. It does not start either formal experiment.
Each wrapper independently loads `/root/miniconda3/etc/profile.d/conda.sh`,
activates existing `rtdetr`, then enables nounset and sets the worktree PYTHONPATH.
Source and data come from the original main repository; no packages are upgraded.

```bash
bash <<'DPR_CHECK'
set -eo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
test "$(git rev-parse HEAD)" = @SHA@
DPR_VARIANT=dpr_v1 bash tools/dpr_server.sh init
DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh init-preflight
bash tools/dpr_server.sh plan
DPR_CHECK
```

Reports/logs/real process and tee exit codes are under
`outputs/dpr/<variant>/check_<timestamp>/`. The full check uses native real train
augmentation, B16/640/AMP and a maximum of 16 batches, targeting two actual
optimizer updates. Failed/PENDING leaves block start. OOM does not reduce batch.
No GPU-idle check is imposed; existing processes are preserved.

The server environment must be the existing Python3.10, PyTorch2.1.2+cu121,
RTX4090 setup. Local Windows RTX2060 evidence cannot authorize it. If a check
fails, preserve its report/log and fix the cause before rerunning the affected
check. Do not edit reports to obtain a pass or discard failures.

## Explicit formal start (not executed by this handoff)

After independently reviewing a complete server preflight, invoke:

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1/tools/dpr_server.sh start
```

The wrapper rechecks current evidence and identity. This is the only new formal
start command. It uses the unchanged 200e/online-augmentation parent recipe and
the controlled public initialization. The ablation is not automatically started.

## Explicit resume

Only an actual unfinished checkpoint from this experiment, with the
`optimizer_fp32_v1` original FP32 moments, is accepted. Initialization/deploy
files and completed runs are rejected. A final validation failure after 200
completed epochs is not authorization to train another 200 epochs.

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1/tools/dpr_server.sh resume \
  --checkpoint /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/cbr_lif_dpr_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/last.pt
```

## Independent val then final test (not executed)

Run only after the experiment finishes and structure is frozen. Val creates a
frozen best SHA record. Existing/failed result directories are preserved; choose
a fresh timestamp for another attempt. Test requires the successful val JSON
for the same weights, source, data identity and fixed evaluation policy.

```bash
bash <<'DPR_EVAL'
set -eo pipefail
W=/root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
BEST=/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/cbr_lif_dpr_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt
STAMP=$(date -u +%Y%m%dT%H%M%S)_$$
V="$W/outputs/dpr/cbr_lif_dpr_v1/evaluation_val_$STAMP"
T="$W/outputs/dpr/cbr_lif_dpr_v1/evaluation_test_$STAMP"
bash "$W/tools/dpr_server.sh" val --weights "$BEST" --output "$V"
bash "$W/tools/dpr_server.sh" test --weights "$BEST" --val-report "$V/metrics.json" --output "$T"
DPR_EVAL
```

This evaluation block explicitly invokes both steps; do not run it during
preflight. P/R are each model's own max-F1 point. Historical parent test mAP is
52.2009%; no DPR accuracy measurement exists before these runs.

## Light package

Can be invoked independently without triggering train/val/test. The archive is
strictly <20 MiB and includes missing/omitted evidence explicitly. Full weights,
data and predictions stay on the server. The tool prints the absolute downloadable
path, byte size and SHA256, and rechecks all archive members.

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1/tools/dpr_server.sh pack
```

Use `DPR_VARIANT=dpr_v1` with the corresponding tools only when the later single
module experiment is explicitly scheduled; its metadata/run paths are independent.
