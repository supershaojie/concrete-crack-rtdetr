# DTR v1 服务器命令

固定功能提交：`bb5cc3987c6b0df3c65ee9253e2bc63ba725a461`。分支：`exp-rtdetr-r18-lite-dtr-v1`。
本文件在功能提交之后由纯文档提交补入；bootstrap 使用已推送的固定功能 SHA，不依赖本文自引用 hash。

以下主项目、Python、数据路径来自历史约定，本地未连接服务器核实。同步脚本和 prepare 会验证实际仓库、解释器、导入来源、公共权重和数据配置，缺失时明确退出。本次没有自动启动这些服务器命令。

## 1. 同步独立 worktree（可整块复制）

只更新 Git 远端引用并创建/核对 DTR worktree，不切换主项目或其他实验的工作区。已有 DTR 目录/分支不一致或有改动时保留并报错，不 reset、不强推。网络每次限 120 秒，最多三次，不用 raw.githubusercontent.com。

```bash
set -euo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR
fetched=0
for attempt in 1 2 3; do
  if timeout 120 git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 fetch origin exp-rtdetr-r18-lite-dtr-v1; then
    fetched=1
    break
  fi
done
test "$fetched" = 1
git show bb5cc3987c6b0df3c65ee9253e2bc63ba725a461:tools/sync_dtr_v1.sh > /tmp/sync_dtr_v1_bb5cc3987c6b0df3c65ee9253e2bc63ba725a461.sh
bash /tmp/sync_dtr_v1_bb5cc3987c6b0df3c65ee9253e2bc63ba725a461.sh bb5cc3987c6b0df3c65ee9253e2bc63ba725a461
```

脚本将工作区固定到 `/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1`，输出实际 `ultralytics.__file__`，并执行真实入口 --help。正式 run 为：
`/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/runs/c_series/dtr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。

## 2. Prepare：公共初始化、109 字段配方和数据身份

首次有已确认 ARG 数据快照则复用，否则仅首次建立全内容快照。数据无变化时再次 prepare 不重新全量扫描。不会继承其他实验或 smoke 权重。

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py prepare
```

仅当数据确实改变、且尚未开始本 run 时，显式重建快照：

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py prepare --refresh-data
```

## 3. 有界 preflight：不会接着自动长训

需要 GPU 0 空闲，保持原 B16/640、AMP、AdamW、nbs=64 和 e=20 对应的累积规则。总上限 16 个训练 micro-batch / 900 秒，阶段与进度持续输出。诊断 run、checkpoint、RNG、optimizer/scaler/EMA 与正式训练隔离。

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py preflight --seconds 900 --micro-batches 16
```

必须得到 PASS/start_eligible=true 才能正式 start。PENDING（Python 退出 2）不是 PASS。若原生初始 AMP overflow 在边界内没有产生有效更新，保留该次证据；先用下面 status 查看 scale_before/after、skipped 和有效更新，再分析对应 folder 下 gpu.json。不要反复重跑同一失败条件、替换低 scale scaler、降低 batch 或自动扩大预算。当前交付不会做这些变更。

## 4. Status：仅读已有证据

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py status
```

显示 preflight 结果/证据目录/scale 更新，实际 worker PID、dispatch、退出记录和进度。tmux 存在、tee 成功或 best.pt 存在均不等于训练完成。

## 5. 正式 tmux start（仅在 preflight 通过后手动执行）

从公共初始化、零基 e=0 独立开始。原 epochs=200/patience=50 生效，不自动延长。启动会核对代码/配方/数据/初始化与预检绑定；已有活动 run 拒绝覆盖，其他 GPU 作业只报告占用。

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py start
```

会创建 `dtr-v1-training`，保存命令、run identity、PID、开始/结束、console.log、真实 Python 和 tee 的独立退出码。训练自然结束后仅记录完成；正式 FP32 val/test 由下方 finish 执行。

## 6. 查看进度

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py status
tmux attach -t dtr-v1-training
```

退出查看但保留训练：Ctrl+B，然后 D。终端日志也可独立查看：

```bash
log=$(find /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/outputs/dtr_v1/dispatches -mindepth 2 -maxdepth 2 -name console.log | sort | tail -n 1)
test -n "$log"
tail -n 100 -F "$log"
```

## 7. 中断后的原生 resume

只允许本实验尚未完成、身份相同、保留完整 epoch/optimizer/scaler/EMA 的 last。会继续真实 epoch 和 DTR 日程。被 strip 的 checkpoint 明确拒绝，不将其伪装成训练 resume。

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py resume
```

如果训练已经完成但评估失败，直接走下面的评估恢复，不 resume 训练。

## 8. 训练完成后的 finish：一次完整评估分析包

**这一步会在必要时运行正式 val 和 test，只在训练完成后由你执行。**

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py finish
```

依次核对同 best/同数据/同 FP32 协议的成功记录，复用完整结果（包括完整报告缺锁时补锁），只补缺失/失败 split。第一次必要推理默认同时导出全部 query+GT、曲线和原始统计；随后离线用 val 阈值计算 test 同阈值 TP/FP/FN/P/R/F1。不会用 test 选阈值或再选 best，不为补预测/曲线重推理。

完整包输出到 `/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/outputs/dtr_v1/packages/DTR_v1_COMPLETE_*.tar.gz`。只产生一个完整包；同身份和 hash 的再次 finish 复用该包。默认不含图片全集及权重本体，含 best/last 路径、大小、SHA256。失败/历史记录保留。

## 9. 独立评估恢复（不重新训练）

若 val 失败，可单独恢复，再完成缺项与打包：

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py val
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py finish
```

若只有 test 失败，直接再次 finish 即可。也提供独立 test 入口；它要求已有同 best 的完整正式 val 锁：

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py test
```

## 10. 只打现有材料的 pack

不推理、不扫描原始数据、不更换 best。缺项明确打成 INCOMPLETE，不能当作完成正式实验。为避免打包中读取正在变化的日志，请在本实验 worker 退出后执行。

```bash
env PYTHONPATH=/root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/ultralytics-main YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/rtdetr/bin/python -u /root/autodl-tmp/projects/Crack_RTDETR-dtr_v1/tools/dtr_v1.py pack
```

## 本地已验证与服务器边界

数学/autograd/DN 关联、L0 路由、公共初始化映射、CPU 全参数梯度、同权重推理、checkpoint 新进程加载、局部 CPU/CUDA FP32 有效更新、新进程一批真实 val/有限零 warmup/全部查询导出、CLI 和离线流程已检查。

CUDA 严格逐元素全参数梯度重现性仍为 PENDING；母版同一计算图重复反传也出现同量级差异，记录原容差而未放宽为 PASS。本地 prepare 已真实 PASS；本地 preflight 正确返回 PENDING/start_eligible=false，因为这里是 Windows + 6GB GPU。服务器 B16/640 AMP、原生完整状态 resume、正式训练及 val/test 全部留待上述命令实测。详见 DELIVERY.md 和 local_validation.json。
