#!/usr/bin/env bash
# Run from `git show FULL_SHA:tools/run_saved_dpr_b7.sh | bash -s -- FULL_SHA`.
# Only the DPR worktree is synchronized. No training/admission entry is invoked.
set -eo pipefail
S=${1:?Supply the delivered full SHA}
P=f0eace10a882049e7053fbe9f69cad0c38149554
M=/root/autodl-tmp/projects/Crack_RTDETR
W=/root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
[[ $# == 1 && "$S" =~ ^[0-9a-f]{40}$ ]] || exit 2
case "$(git -C "$M" remote get-url origin)" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr) ;;
  *) echo 'Unexpected origin; preserved.' >&2; exit 1 ;;
esac
git -C "$M" cat-file -e "$S^{commit}"
git -C "$M" merge-base --is-ancestor "$P" "$S"
MAIN_HEAD=$(git -C "$M" rev-parse HEAD)
MAIN_STATUS=$(git -C "$M" status --porcelain)
test -f "$W/.git"
test "$(git -C "$W" rev-parse --show-toplevel)" = "$W"
test "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)" = \
     "$(git -C "$M" rev-parse --path-format=absolute --git-common-dir)"
test -z "$(git -C "$W" status --porcelain --untracked-files=no)"
H=$(git -C "$W" rev-parse HEAD)
case "$H" in
  "$S") ;;
  "$P") git -C "$W" checkout --detach --no-overwrite-ignore "$S" ;;
  *) echo 'Unexpected DPR HEAD; preserved without reset/clean.' >&2; exit 1 ;;
esac
git -C "$M" show "$S:tools/sync_dpr.sh" | bash -s -- "$S" "$M" "$W"
test "$(git -C "$M" rev-parse HEAD)" = "$MAIN_HEAD"
test "$(git -C "$M" status --porcelain)" = "$MAIN_STATUS"
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
set -u
cd -- "$W"
test "$(git rev-parse HEAD)" = "$S"
export PYTHONPATH="$W/ultralytics-main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
set +e
python tools/compare_dpr_b7.py
RC=$?
set -e
test "$(git -C "$M" rev-parse HEAD)" = "$MAIN_HEAD"
test "$(git -C "$M" status --porcelain)" = "$MAIN_STATUS"
printf 'Diagnostic exit=%s; DPR remains BLOCKED. Read the NEW_DIAGNOSTIC directory printed above.\n' "$RC"
exit "$RC"
