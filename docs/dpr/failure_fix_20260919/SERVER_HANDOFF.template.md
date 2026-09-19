# DPR failure diagnostics: fixed-commit server handoff

Branch `exp-rtdetr-r18-lite-dpr-v1`; new commit `@SHA@`;
expected existing server commit `2f4319281cac92a1e56663c3c4a48ba7bafc6ccb`.
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
P=2f4319281cac92a1e56663c3c4a48ba7bafc6ccb
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

## Run now: initialization/module audits and finite diagnosis

The current status is BLOCKED. These commands collect evidence only. Both init
calls verify existing controlled weights without overwriting them and run each
variant's 640 mathematical/module checks. They may revoke an old permit as the
existing wrapper requires. The diagnosis itself never writes a start permit,
never runs B16 capacity and never starts formal training/test. It uses fresh
output directories, real B2/160 GT/DN data and the unchanged native step.

```bash
bash <<'DPR_DIAG'
set -eo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
test "$(git rev-parse HEAD)" = @SHA@
DPR_VARIANT=dpr_v1 bash tools/dpr_server.sh init
DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh init
DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh diagnose --device all --scope all --timeout 900
DPR_DIAG
```

The diagnostic prints its absolute output and light-package path. Preserve
`diagnostic.json`, `worker.log`, `exit_status.json`, `package_manifest.json` and
`diagnostic_light.tar.gz`. Send only the light package and manifest for review;
replay weights remain on the server. Exit 0 means diagnostic collection without
recorded raw blockers, **not** admission; exit 3 retains failure/BLOCKED; timeout
returns 124. Shell tee/command exit statuses are recorded separately. Partial
reports and missing outputs are explicitly recorded. Do not append `|| true`.

## Separate entry: complete preflight (DO NOT RUN WHILE BLOCKED)

Only after the targeted causes have been resolved under the original contract,
run one complete preflight on the then-approved fixed SHA. The command below is
the entry for this delivery; it is documented, not executed or claimed passed.
It includes new initialization/math, CPU/CUDA/AMP lifecycle A/B and B16/640/native
AMP capacity. Historical A/capacity passes are not a new-SHA permit.

```bash
bash <<'DPR_FULL_PREFLIGHT'
set -eo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
test "$(git rev-parse HEAD)" = @SHA@
DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh init-preflight
bash tools/dpr_server.sh plan
DPR_FULL_PREFLIGHT
```

## Separate entry: formal start (NOT EXECUTED)

`DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh start` is the existing formal
entry. It requires explicit authorization to start plus a fresh complete
preflight that passes strict validation for the exact current code/source/data/
initialization identities. The current BLOCKED diagnosis cannot authorize it.
No formal start or final test command is part of the run-now block.

Exact server Python/PyTorch/THOP rerun: PENDING. Original CPU backend cause:
PENDING. New complete server admission: BLOCKED/PENDING. Local validation used
PyTorch 2.7.1+cu118, not server 2.1.2+cu121. Formal training NOT_STARTED; final test
NOT_RUN. No dependency upgrade/downgrade, server login or other experiment action
was performed. Any contract alternative remains UNAPPROVED/NOT_IMPLEMENTED.
