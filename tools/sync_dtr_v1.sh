#!/usr/bin/env bash
# Independent worktree only; bounded Git transport, no raw.githubusercontent.com.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  printf '%s\n' "Usage: bash tools/sync_dtr_v1.sh COMMIT" "Fetch and safely create/reuse the isolated DTR worktree at an exact 40-digit commit."
  exit 0
fi
commit="${1:?A full committed 40-digit SHA is required}"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected full 40-digit SHA" >&2; exit 2; }
main=/root/autodl-tmp/projects/Crack_RTDETR
target=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1
branch=exp-rtdetr-r18-lite-dtr-v1
python=/root/miniconda3/envs/rtdetr/bin/python
[[ -x "$python" ]] || { echo "Missing expected Python: $python" >&2; exit 2; }
[[ "$(git -C "$main" remote get-url origin)" == https://github.com/supershaojie/concrete-crack-rtdetr.git ]] || exit 2
success=0
for attempt in 1 2 3; do
  if timeout 120 git -c http.connectTimeout=15 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 -C "$main" fetch origin "$branch"; then
    success=1
    break
  fi
  echo "Fetch attempt $attempt failed" >&2
done
[[ "$success" == 1 ]] || exit 2
git -C "$main" cat-file -e "$commit^{commit}"
git -C "$main" merge-base --is-ancestor "$commit" "origin/$branch"
if [[ -e "$target" ]]; then
  [[ "$(git -C "$target" rev-parse --show-toplevel)" == "$target" ]] || exit 2
  [[ -z "$(git -C "$target" status --porcelain)" ]] || { echo "Existing target dirty; preserved" >&2; exit 2; }
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]] || {
    echo "Existing DTR worktree is at another commit; preserved. Inspect its run identity before updating." >&2; exit 2;
  }
else
  if git -C "$main" show-ref --verify --quiet "refs/heads/$branch"; then
    [[ "$(git -C "$main" rev-parse "$branch")" == "$commit" ]] || { echo "Existing branch differs; preserved" >&2; exit 2; }
    git -C "$main" worktree add "$target" "$branch"
  else
    git -C "$main" worktree add -b "$branch" "$target" "$commit"
  fi
fi
cd "$target"
export PYTHONPATH="$target/ultralytics-main"
export YOLO_AUTOINSTALL=false
"$python" -c 'import pathlib,ultralytics; root=pathlib.Path.cwd(); p=pathlib.Path(ultralytics.__file__).resolve(); assert p.is_relative_to(root/"ultralytics-main"), p; print(p)'
"$python" tools/dtr_v1.py --help
printf '%s\n' "DTR worktree ready: $target at $commit. No prepare, preflight, training or evaluation started."
