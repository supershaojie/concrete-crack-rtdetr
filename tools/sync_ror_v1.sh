#!/usr/bin/env bash
# 中文：只定位 ROR 独立 worktree，不改变主项目或其他实验；必须传入交付完整 SHA。
set -euo pipefail
if [[ "${1:-}" == "--help" || $# -ne 1 ]]; then
  echo '用法: bash tools/sync_ror_v1.sh 完整40位交付SHA'
  echo '可选 ROR_V1_MAIN 指定主项目；默认 /root/autodl-tmp/projects/Crack_RTDETR'
  exit 0
fi
ror_sha="$1"
[[ "$ror_sha" =~ ^[0-9a-f]{40}$ ]] || { echo '必须是完整 SHA'; exit 1; }
ror_main="${ROR_V1_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
ror_worktree="$(dirname "$ror_main")/Crack_RTDETR-ror_v1"
ror_branch='exp-rtdetr-r18-lite-ror-v1'
ror_origin="$(git -C "$ror_main" remote get-url origin)"
case "$ror_origin" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo "仓库来源不匹配: $ror_origin"; exit 1 ;;
esac
ror_main_before="$(git -C "$ror_main" rev-parse HEAD)"
git -C "$ror_main" worktree list
git -C "$ror_main" fetch origin "$ror_branch"
[[ "$(git -C "$ror_main" rev-parse FETCH_HEAD)" == "$ror_sha" ]] || { echo '远端分支与交付 SHA 不同，先核对交付版本'; exit 1; }
git -C "$ror_main" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$ror_sha"
if [[ -e "$ror_worktree" ]]; then
  [[ "$(git -C "$ror_worktree" branch --show-current)" == "$ror_branch" ]] || { echo '目标不是 ROR 分支'; exit 1; }
  [[ -z "$(git -C "$ror_worktree" status --porcelain)" ]] || { echo '目标有未提交更改'; exit 1; }
  git -C "$ror_worktree" merge --ff-only "$ror_sha"
else
  if git -C "$ror_main" show-ref --verify --quiet "refs/heads/$ror_branch"; then
    [[ "$(git -C "$ror_main" rev-parse "$ror_branch")" == "$ror_sha" ]] || { echo '已有同名分支未指向交付 SHA；不覆盖'; exit 1; }
    git -C "$ror_main" worktree add "$ror_worktree" "$ror_branch"
  else
    git -C "$ror_main" worktree add -b "$ror_branch" "$ror_worktree" "$ror_sha"
  fi
fi
[[ "$(git -C "$ror_main" rev-parse HEAD)" == "$ror_main_before" ]]
[[ "$(git -C "$ror_worktree" rev-parse HEAD)" == "$ror_sha" ]]
printf 'ROR worktree: %s\nCommit: %s\n主工作区 HEAD 保持原值。\n' "$ror_worktree" "$ror_sha"
