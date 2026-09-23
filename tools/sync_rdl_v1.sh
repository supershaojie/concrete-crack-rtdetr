#!/usr/bin/env bash
# 从已核验 remote 同步完整交付SHA，独立detached worktree，保留主项目和其他实验。
set -euo pipefail
if [[ "${1:-}" == "--help" || $# != 1 ]]; then
  echo '用法: bash tools/sync_rdl_v1.sh 完整40位commitSHA'; exit 0
fi
rdl_commit="$1"
[[ "$rdl_commit" =~ ^[0-9a-f]{40}$ ]] || { echo '必须给完整commit SHA'; exit 1; }
rdl_main=/root/autodl-tmp/projects/Crack_RTDETR
rdl_worktree=/root/autodl-tmp/projects/Crack_RTDETR-rdl_v1
rdl_branch=exp-rtdetr-r18-lite-rdl-v1
rdl_remote="$(git -C "$rdl_main" remote get-url origin)"
[[ "$rdl_remote" == 'https://github.com/supershaojie/concrete-crack-rtdetr.git' || "$rdl_remote" == 'git@github.com:supershaojie/concrete-crack-rtdetr.git' || "$rdl_remote" == 'ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git' ]] || { echo 'origin仓库不匹配'; exit 1; }
rdl_main_head="$(git -C "$rdl_main" rev-parse HEAD)"
git -C "$rdl_main" fetch origin "$rdl_branch"
test "$(git -C "$rdl_main" rev-parse FETCH_HEAD)" = "$rdl_commit"
git -C "$rdl_main" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$rdl_commit"
if [[ -e "$rdl_worktree" ]]; then
  test -f "$rdl_worktree/.git"
  test "$(git -C "$rdl_worktree" rev-parse HEAD)" = "$rdl_commit"
  test -z "$(git -C "$rdl_worktree" status --porcelain --untracked-files=no)"
else
  git -C "$rdl_main" worktree add --detach "$rdl_worktree" "$rdl_commit"
fi
test "$(git -C "$rdl_main" rev-parse HEAD)" = "$rdl_main_head"
mkdir -p "$rdl_worktree/outputs"
printf '{"commit":"%s","branch":"%s","main":"%s","worktree":"%s"}\n' "$rdl_commit" "$rdl_branch" "$rdl_main" "$rdl_worktree" > "$rdl_worktree/outputs/rdl_v1_delivery.json"
echo "同步完成: $rdl_worktree @ $rdl_commit；未初始化、未训练"
