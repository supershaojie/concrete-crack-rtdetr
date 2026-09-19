# DPR THOP compatibility: fixed-commit server handoff

Branch `exp-rtdetr-r18-lite-dpr-v1`; new commit `@SHA@`;
expected existing server commit `bf23c6d4ad43aa2182e11e8a0726cef950425728`.
This template is rendered after commit. Do not execute the placeholder template.

## Sync and environment

Run the complete block in the existing server/tmux terminal. It fetches only
if the target commit is absent, uses finite HTTP/1.1 retries, and updates only
the verified clean DPR worktree from the stated old SHA. Existing reports,
weights, untracked and ignored files are preserved; any checkout collision
causes refusal. It does not reset/clean, change dependency versions or stop
other processes. A failure returns to the terminal without closing its session.

```bash
bash <<'DPR_SYNC'
set -eo pipefail
S=@SHA@
P=bf23c6d4ad43aa2182e11e8a0726cef950425728
M=/root/autodl-tmp/projects/Crack_RTDETR
W=/root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
B=exp-rtdetr-r18-lite-dpr-v1
R=$(git -C "$M" remote get-url origin); R=${R%.git}
case "$R" in
  https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr|ssh://git@github.com/supershaojie/concrete-crack-rtdetr) ;;
  *) echo 'Unexpected origin; preserved.' >&2; exit 1 ;;
esac
MAIN_HEAD=$(git -C "$M" rev-parse HEAD)
MAIN_STATUS=$(git -C "$M" status --porcelain)
if ! git -C "$M" cat-file -e "$S^{commit}" 2>/dev/null; then
  for i in 1 2 3 4 5; do
    echo "Fetch attempt $i/5"
    if GIT_TERMINAL_PROMPT=0 timeout 120s git -C "$M" \
      -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head \
      origin "refs/heads/$B:refs/remotes/origin/$B"; then
      git -C "$M" cat-file -e "$S^{commit}" && break
    else
      rc=$?; echo "Fetch failed: exit $rc" >&2
    fi
    [ "$i" = 5 ] || sleep 3
  done
fi
git -C "$M" cat-file -e "$S^{commit}"
git -C "$M" merge-base --is-ancestor "$P" "$S"
if [ -e "$W" ]; then
  test -f "$W/.git"
  test "$(git -C "$W" rev-parse --show-toplevel)" = "$W"
  test "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)" = \
       "$(git -C "$M" rev-parse --path-format=absolute --git-common-dir)"
  test -z "$(git -C "$W" status --porcelain --untracked-files=no)"
  H=$(git -C "$W" rev-parse HEAD)
  case "$H" in
    "$S") ;;
    "$P") git -C "$W" checkout --detach --no-overwrite-ignore "$S" ;;
    *) echo 'Unexpected DPR HEAD; preserved.' >&2; exit 1 ;;
  esac
fi
git -C "$M" show "$S:tools/sync_dpr.sh" | bash -s -- "$S" "$M" "$W"
test "$(git -C "$M" rev-parse HEAD)" = "$MAIN_HEAD"
test "$(git -C "$M" status --porcelain)" = "$MAIN_STATUS"
bash "$W/tools/dpr_server.sh" environment
DPR_SYNC
```

## Initialization and complete bounded preflight

Each wrapper independently loads the existing Conda activation script and
`rtdetr` environment, sets the worktree PYTHONPATH and checks its import path.
The first action re-audits existing C2+DPR initialization and runs its complete
640 module checks. The second re-audits main initialization, runs its module
checks and CPU/CUDA/AMP lifecycle plus B16/640/AMP capacity preflight. Reports
and tee exit codes use fresh timestamped directories. Any failure remains a
failure and blocks formal admission. Existing initialization is verified in
place instead of overwritten. Neither action starts formal training or test.

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

Exact server environment and complete preflight: **PENDING until executed**.
Local THOP 2.1.6 count verification used PyTorch 2.7.1, not server 2.1.2.
Earlier AMP/half failures and old reports are retained, not overridden by this
FLOPs fix. Formal training **NOT_STARTED**; final test **NOT_RUN**; no server
login was performed for this change.
