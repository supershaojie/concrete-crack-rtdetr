# 服务器有限诊断、安全恢复与后续命令

先将交付回复中的新 **40 位完整 SHA** 设置为 `C19_LIF_FIX_SHA`。本页不把自己的提交 SHA 写进自身；具体 SHA 以交付回复和 sync 生成的 delivery JSON 为准。旧目录永远不就地改代码，原初始化不复制为新正式初始化。

```bash
export C19_LIF_FIX_SHA='在这里填交付回复中的新完整SHA'
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
OLD=/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1
NEW=/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-preflightfix
bash "$OLD/tools/sync_c19_lif_v1.sh" "$C19_LIF_FIX_SHA" "$MAIN" "$NEW"
cd "$NEW"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export C19_LIF_V1_MAIN="$MAIN" CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$NEW/ultralytics-main:$NEW/tools" PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
```

只读加载旧初始化、重放旧 CPU RNG 前缀，采集原环境的实际失配。每次使用不重名目录；如果存在就拒绝覆盖。命令只执行有限融合、父退化与单图 predict，不运行正式训练或完整评估。

```bash
DIAG="$NEW/outputs/c19_lif_v1_fusion_$(date +%Y%m%d_%H%M%S%N)_$$"
set -o pipefail
python -u tools/diagnose_c19_lif_v1_fusion.py \
  --source "$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --initialized "$OLD/weights/c19_lif_v1_controlled_init.pt" \
  --devices cpu --output "$DIAG" 2>&1 | tee "${DIAG}.console.log"
cat "$DIAG/checks.json"
cat "$DIAG/cpu_FP32_fusion/fuse_diagnostic.json"
tar -C "$NEW/outputs" -czf "${DIAG}.tar.gz" "$(basename "$DIAG")" "$(basename "${DIAG}.console.log")"
sha256sum "${DIAG}.tar.gz"
```

有限重放失败 fixture（同样在新目录；不能用训练 best/last 替代）：

```bash
python -u tools/diagnose_c19_lif_v1_fusion.py \
  --source "$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --initialized "$OLD/weights/c19_lif_v1_controlled_init.pt" \
  --fixture "$DIAG/cpu_FP32_fusion/fixture.pt" \
  --output "${DIAG}_replay_$(date +%s%N)"
```

旧锁恢复默认只读。必须在服务器检查实际 owner；包内没有 owner，不能猜 PID。若 PID 仍活跃（包括可能重用）、身份未知、tmux 无法读取、正式结果或派发证据存在，命令拒绝且保留现场。C17 worker/session 不会被清理或停止。

```bash
python tools/recover_c19_lif_v1_preflight.py \
  --main "$MAIN" --old-worktree "$OLD" \
  --old-sha c27064ce6b06d0c32d38e644fcff64394a1a5992
```

仅当上一步显示 `SAFE_TO_ARCHIVE`，并决定恢复时，显式执行下面这一条。它会再次核验，在共享 runs/c_series 下保留式归档原 lock/owner，记录旧路径、归档路径、SHA 和原因；不会删除旧工作树、初始化、报告或日志。没有 start-direct 自动抢锁。

```bash
python tools/recover_c19_lif_v1_preflight.py \
  --main "$MAIN" --old-worktree "$OLD" \
  --old-sha c27064ce6b06d0c32d38e644fcff64394a1a5992 --apply
```

原有限工程预检加 B16/640 AMP 容量门禁，不派发训练。新的独立预检初始化在 `outputs/c19_lif_v1_manual_preflight_时间_随机值/controlled_init.pt`，每次新目录。终端会打印准确目录，内部日志为该目录的 `preflight.log`；总控制台日志在 `outputs/c19_lif_v1_command_logs/preflight-only_*.log`。

```bash
bash tools/autodl_c19_lif_v1.sh preflight-only
```

实时日志在另一个终端使用命令打印出的准确目录：

```bash
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-preflightfix/outputs/c19_lif_v1_manual_preflight_实际时间_实际随机值/preflight.log
```

只有全部门禁通过且之后决定开始正式训练时，执行下列启动命令。本次开发没有执行。start-direct 使用新工作树自己的 `weights/c19_lif_v1_controlled_init.pt`，重复有限预检和 B16 容量门禁，再原子预约保护下派发 tmux。再次失败仍保留现场；不要删除后重试。使用下一份独立目录和显式恢复流程。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-preflightfix
CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c19_lif_v1.sh start-direct
bash tools/autodl_c19_lif_v1.sh status
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-preflightfix/outputs/c19_lif_v1/preflight.log
# 正式 worker 派发后：
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-preflightfix/outputs/c19_lif_v1/console.log
```

正式训练 SUCCESS 后才按顺序执行（本次未执行）：

```bash
bash tools/autodl_c19_lif_v1.sh val
bash tools/autodl_c19_lif_v1.sh test
bash tools/autodl_c19_lif_v1.sh pack-complete
```

这些命令继续使用原评估协议、同一 best 的 SHA 锁、完整证据与打包校验。不要用开发机或 CPU 小图通过替代 AutoDL CUDA/AMP/true-half 和 B16 原生容量检查。
