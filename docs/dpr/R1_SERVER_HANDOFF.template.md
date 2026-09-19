# R1 固定版本服务器交付（模板）

固定 SHA：`@SHA@`。此模板提交后替换 SHA，勿直接执行占位符。

## 1. 同步

先检查本地目标 commit；缺失才 HTTP/1.1 fetch 指定分支，单次 120 秒、最多 5 次。仅接受已知的 DPR 旧 HEAD 或目标 HEAD，保护 tracked 修改及全部历史输出；不 reset/clean、不覆盖 ignored 文件、不改变主实验工作树。同步块本身不执行补证或训练。

```bash
bash <<'DPR_SYNC'
set -eo pipefail
S=@SHA@
P=bb7687874a13782a768335adc60a38d00f150706
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
    "$P"|bd3906bece58dede1fa55ce2d5e04d3cb02d8c0d) git -C "$W" checkout --detach --no-overwrite-ignore "$S" ;;
    *) echo 'Unexpected DPR HEAD; preserved.' >&2; exit 1 ;;
  esac
fi
git -C "$M" show "$S:tools/sync_dpr.sh" | bash -s -- "$S" "$M" "$W"
test "$(git -C "$M" rev-parse HEAD)" = "$MAIN_HEAD"
test "$(git -C "$M" status --porcelain)" = "$MAIN_STATUS"
bash "$W/tools/dpr_server.sh" environment
DPR_SYNC
```

## 2. 现在执行：新初始化/数学审计与定向补证

每个已批准配置分别留新目录。`init` 验证已有 controlled init，不覆盖其权重；包含原 640 数学/结构检查。定向补证不运行旧完整诊断、CPU/容量矩阵或正式训练/test。主配置先运行；退出 3/124 或日志出现失败即停止，回传 light 包及 manifest。完整 state/grad 留服务器，不上传 GB 文件。

```bash
bash <<'DPR_R1_TARGETED'
set -eo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
test "$(git rev-parse HEAD)" = @SHA@
test -z "$(git status --porcelain --untracked-files=no)"
DPR_VARIANT=dpr_v1 bash tools/dpr_server.sh init
DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh init
R1_OUT="$PWD/outputs/dpr/cbr_lif_dpr_v1/r1_$(date -u +%Y%m%dT%H%M%S)_$$"
DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh supplement --output "$R1_OUT" --timeout 900
printf 'Supplement JSON: %s/supplement.json\nLight evidence: %s/r1_light.tar.gz\nManifest: %s/package_manifest.json\n' "$R1_OUT" "$R1_OUT" "$R1_OUT"
DPR_R1_TARGETED
```

单独消融 `dpr_v1` 若也需要完整准入，必须对该配置另跑同样的 supplement；不能复用主配置证据，也不自动启动该实验。

## 3. 补证通过后才执行：完整准入

本块当前不执行。先审查补证 JSON/light 包，再把下列路径替换成该固定 SHA 下的实际 supplement.json。`init-preflight` 再次独立校验合同、所有事实、文件哈希、环境、输入及源码，只有补证合格才开始完整生命周期与有限容量。它依然不启动训练。失败不得换报告标签、借用旧证据或继续 start。

```bash
bash <<'DPR_R1_FULL_AFTER_REVIEW'
set -eo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-dpr-v1
test "$(git rev-parse HEAD)" = @SHA@
export DPR_R1_SUPPLEMENT=/REPLACE_WITH_REVIEWED_FIXED_SHA_REPORT/supplement.json
test -f "$DPR_R1_SUPPLEMENT"
DPR_VARIANT=cbr_lif_dpr_v1 bash tools/dpr_server.sh init-preflight
DPR_R1_FULL_AFTER_REVIEW
```

0 仅表示当前命令通过；定向补证即使为 0 也不是训练准入。补证 3=BLOCKED，124=预算超时；`exit_status.json` 保留实际 worker 退出码。缺项和新检查失败继续 BLOCKED。所有环境/数学/补证/完整准入服务器结果在执行前均为待验。
