# 当前 AutoDL 流程：共享 P3 门禁

本次Codex只做本地有限验证。把FULL_DELIVERED_SHA替换为最终回复中真实的完整40位SHA。新根目录是待创建约定，不代表已经部署。[诊断与边界](parent_gate_fix/DIAGNOSIS.md)。

在服务器执行一次以下同步、只读状态和启动流程。旧工作树和报告保持；运行目录已存在、tmux/worker活跃或锁归属不明时入口停止，不删除锁来绕过它。旧preflight-only建议已被本次流程替代。

```bash
(
  set -euo pipefail
  C24_SHA=FULL_DELIVERED_SHA
  export C24_LIF_V1_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
  C24_OLD=/root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1-preflightfix
  C24_ROOT=/root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1-parentgatefix
  git -C "$C24_LIF_V1_MAIN" fetch --no-write-fetch-head origin refs/heads/codex/rtdetr-c24-lif-v1:refs/remotes/origin/codex/rtdetr-c24-lif-v1
  bash <(git -C "$C24_LIF_V1_MAIN" show "$C24_SHA:tools/sync_c24_lif_v1.sh") "$C24_SHA" "$C24_LIF_V1_MAIN" "$C24_ROOT"
  if [[ -f "$C24_OLD/tools/autodl_c24_lif_v1.sh" ]]; then
    bash "$C24_OLD/tools/autodl_c24_lif_v1.sh" status
  fi
  bash "$C24_ROOT/tools/autodl_c24_lif_v1.sh" status
  bash "$C24_ROOT/tools/autodl_c24_lif_v1.sh" start-direct
)
```

实际同步签名：`bash tools/sync_c24_lif_v1.sh SHA MAIN WORKTREE`。包装脚本在入口及tmux worker中激活已有rtdetr环境。只调用一次start-direct，不另跑preflight-only。内部运行当前SHA的22个必需阶段，包括原C24四路证书、native初始化和非零分支各自真实B16/640校准；任一不接受则不派发并自动保存小包。旧报告不能替代本次验收；正式GradScaler仍从默认65536开始。

另一个终端查看真实日志；console.log只有派发后生成，tail -F会等待：

```bash
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1-parentgatefix/outputs/c24_lif_v1/preflight.log /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1-parentgatefix/outputs/c24_lif_v1/console.log
```

只读状态、手动故障小包：

```bash
export C24_LIF_V1_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
C24_ROOT=/root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1-parentgatefix
bash "$C24_ROOT/tools/autodl_c24_lif_v1.sh" status
bash "$C24_ROOT/tools/autodl_c24_lif_v1.sh" pack-light
```

包位于`/root/autodl-tmp/projects/Crack_RTDETR/downloads/c24_lif_v1/`，脚本打印实际路径、字节数和SHA256，硬上限8,000,000 bytes。标准库打包，不加载模型、不重跑预检。

仅在未来正式训练完成且状态SUCCESS后，按需评估与打完整分析包。本次没有执行这些操作：

```bash
export C24_LIF_V1_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
C24_ROOT=/root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1-parentgatefix
bash "$C24_ROOT/tools/autodl_c24_lif_v1.sh" val
bash "$C24_ROOT/tools/autodl_c24_lif_v1.sh" test
bash "$C24_ROOT/tools/autodl_c24_lif_v1.sh" pack-complete
```
