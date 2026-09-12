# 单次预检后派发

用本次交付回复中的真实完整 SHA 替换第一行占位符。sync 的实际接口为 `FULL_SHA MAIN_REPO WORKTREE`；它验证远端和完整提交、创建 detached worktree、保护已有目录和主 checkout。

```bash
SHA='<本次交付的40位完整SHA>'
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
OLD=/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-preflightfix
NEW=/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix
bash "$OLD/tools/sync_c19_lif_v1.sh" "$SHA" "$MAIN" "$NEW" &&
cd "$NEW" &&
CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c19_lif_v1.sh status &&
CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c19_lif_v1.sh start-direct
```

status 只读显示当前 C19 worker、专属 tmux 和共享 lock owner。用户已归档原失败启动锁；这次旧 manual preflight 没有正式 dispatch 锁，**不再要求执行旧 recover --apply**。若出现新锁/活跃任务，start-direct 会拒绝并保留现场；应按新 owner 的实际状态判断，不能沿用旧截图删锁。C17 worktree 和进程不作任何操作。

start-direct 内仅启动一次必要 checker（含 `--capacity-batch 16`），接受全部融合证据后继续实际 CUDA loss/DN/AMP、原生验证精度及 B16/640 AMP loss/backward；全部通过才派发 tmux。保持当前 SSH 至显示 DISPATCHED；正式 worker 在 tmux 内继续。无需先跑 diagnose 或 preflight-only，后者仅供用户主动要求只测试时使用。

另一 SSH 查看只读状态：

```bash
CUDA_VISIBLE_DEVICES=0 bash /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/tools/autodl_c19_lif_v1.sh status
```

预检实时日志（文件尚未生成时 tail -F 会继续等待）：

```bash
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/outputs/c19_lif_v1/preflight.log
```

派发后的正式训练日志：

```bash
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/outputs/c19_lif_v1/console.log
```

状态区分 CHECKING / DISPATCHED / RUNNING / FAILED / SUCCESS；CHECKING 校验 PID 的启动 token，PID 被复用不能冒充原预检。任何失败保留 launch、partial report、init 和 reservation，不自动重启。

失败会自动尝试生成小包及 SHA256 文件：

```text
/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/outputs/c19_lif_v1_LIGHT.tar.gz
/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/outputs/c19_lif_v1_LIGHT.tar.gz.sha256
```

如需重新导出，使用新的文件名；工具拒绝覆盖旧包，并强制小于等于 8,000,000 字节。完整张量继续留服务器，不上传。

```bash
NEW=/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix
/root/miniconda3/envs/rtdetr/bin/python "$NEW/tools/pack_c19_lif_v1_light.py" \
  --preflight "$NEW/outputs/c19_lif_v1_preflight" \
  --launch "$NEW/outputs/c19_lif_v1" \
  --output "$NEW/outputs/c19_lif_v1_LIGHT_$(date +%Y%m%d_%H%M%S).tar.gz"
```

仅在正式训练 SUCCESS 后才按原规则独立 val、锁定 best SHA、独立 test、pack-complete。本次没有执行这些命令：

```bash
NEW=/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix
CUDA_VISIBLE_DEVICES=0 bash "$NEW/tools/autodl_c19_lif_v1.sh" val &&
CUDA_VISIBLE_DEVICES=0 bash "$NEW/tools/autodl_c19_lif_v1.sh" test &&
bash "$NEW/tools/autodl_c19_lif_v1.sh" pack-complete
```

pack-complete 是原训练结果完整归档，不作为本次故障上传包；排错仅上传 LIGHT 小包。本地有限检查不等于服务器已通过，不保证新服务器会产生相同自然候选。
