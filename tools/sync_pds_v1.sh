#!/usr/bin/env bash
# Official pinned sync; does not switch the main checkout or touch another experiment.
set -Eeuo pipefail
SHA="${1:-}"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Usage: bash tools/sync_pds_v1.sh FULL_SHA" >&2; exit 2; }
MAIN="${PDS_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
TARGET="${PDS_WORKTREE:-${MAIN}-pds_v1}"
BRANCH=exp-rtdetr-r18-lite-pds-v1
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
MAIN="$(cd -- "$MAIN" && pwd -P)"
[[ "$TARGET" != "$MAIN" && "$TARGET" == /* ]] || { echo "Unsafe PDS worktree path" >&2; exit 2; }
ORIGIN="$(git -C "$MAIN" remote get-url origin)"
[[ "$ORIGIN" == https://github.com/supershaojie/concrete-crack-rtdetr.git ]] || { echo "Wrong origin: $ORIGIN" >&2; exit 2; }
MAIN_HEAD="$(git -C "$MAIN" rev-parse HEAD)"
# Bootstrap can reuse exactly one already-completed official branch fetch.
if [[ "${PDS_FETCHED_SHA:-}" != "$SHA" ]]; then
    echo "PDS sync: fetching official branch"
    git -C "$MAIN" fetch origin "$BRANCH"
fi
[[ "$(git -C "$MAIN" rev-parse "refs/remotes/origin/$BRANCH")" == "$SHA" ]] || {
    echo "Remote-tracking PDS branch differs from delivered full SHA" >&2; exit 3;
}
git -C "$MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SHA"
if [[ -e "$TARGET" ]]; then
    [[ -f "$TARGET/.git" ]] || { echo "Existing target is not a linked worktree" >&2; exit 4; }
    [[ "$(git -C "$TARGET" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo "Different repository" >&2; exit 4; }
    [[ -z "$(git -C "$TARGET" status --porcelain --untracked-files=no)" ]] || { echo "Preserve tracked target modifications" >&2; exit 4; }
    CURRENT_BRANCH="$(git -C "$TARGET" branch --show-current)"
    [[ -z "$CURRENT_BRANCH" || "$CURRENT_BRANCH" == "$BRANCH" ]] || { echo "Target belongs to another branch" >&2; exit 4; }
    # Reject a live dedicated PDS session; LBC and all other sessions are irrelevant.
    if command -v tmux >/dev/null && tmux has-session -t =pds-v1-training 2>/dev/null; then
        echo "PDS training session is active; refusing sync" >&2; exit 4
    fi
    "$PYTHON" - "$TARGET" <<'PY'
import pathlib, sys, psutil
target=str(pathlib.Path(sys.argv[1]).resolve()/"tools/pds_v1.py")
for p in psutil.process_iter(["cmdline"]):
    args=p.info["cmdline"] or []
    if target in args and "_worker" in args:
        raise SystemExit("Verified PDS worker is active; refusing sync")
PY
    git -C "$TARGET" merge-base --is-ancestor HEAD "$SHA"
    if [[ -z "$CURRENT_BRANCH" ]]; then
        git -C "$TARGET" checkout --detach "$SHA"
    else
        git -C "$TARGET" merge --ff-only "$SHA"
    fi
else
    git -C "$MAIN" worktree add --detach "$TARGET" "$SHA"
fi
[[ "$(git -C "$TARGET" rev-parse HEAD)" == "$SHA" ]]
[[ "$(git -C "$MAIN" rev-parse HEAD)" == "$MAIN_HEAD" ]] || { echo "Main HEAD changed concurrently" >&2; exit 5; }
export PDS_MAIN="$MAIN"
export PYTHONPATH="$TARGET/ultralytics-main:$TARGET/tools"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
cd -- "$TARGET"
"$PYTHON" -u "$TARGET/tools/pds_v1.py" sync "$SHA"
echo "PDS official sync complete: $SHA"
