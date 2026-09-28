# ARG v2 服务器命令

分支 `exp-rtdetr-r18-lite-arg-v2`，固定训练代码提交
`5e7fde5e8393107a4f6c9df9d8282e81a7d03b62`。
该提交包含这里调用的全部脚本；后续交付文档提交不替换这个训练代码身份。
公式 `ARG-v2-q2`，lambda=0.20、epsilon_w=0.05、gamma=2、eps=1e-7。

本页按项目实际脚本和用户给定服务器路径生成。本轮未连接服务器；现场的路径、
origin、源权重、数据配置、Python 和 CUDA 由下面的同步、prepare、preflight 核验。
主仓库为 `/root/autodl-tmp/projects/Crack_RTDETR`，独立 v2 worktree 为
`/root/autodl-tmp/projects/Crack_RTDETR-arg_v2`。
现有 v1 worktree、prepare、预检、run 和训练状态保留，v2 从同一公共源重新初始化。

## 1. 同步到固定代码提交

整块在服务器 Bash 执行。两次 fetch 各自最多 120 秒，失败即停止；临时文件仅保存
从该提交提取的同步工具，退出时清理。已有 v2 同 SHA 且干净可复用；不同 SHA、
其他仓库或未提交改动会被保留并拒绝同步，不能通过删输出或强制重置绕过。

```bash
bash <<'ARG_V2_SYNC'
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SHA=5e7fde5e8393107a4f6c9df9d8282e81a7d03b62
BRANCH=exp-rtdetr-r18-lite-arg-v2
[[ "$(git -C "$MAIN" remote get-url origin)" == https://github.com/supershaojie/concrete-crack-rtdetr.git ]]
timeout 120s git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
git -C "$MAIN" cat-file -e "$SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$SHA" FETCH_HEAD
script=$(mktemp -t arg-v2-sync.XXXXXXXX.sh)
trap 'rm -f -- "$script"' EXIT
git -C "$MAIN" show "$SHA:tools/sync_arg_v2.sh" > "$script"
bash "$script" "$SHA"
ARG_V2_SYNC
```

同步工具还校验母版和 ef9cb7e 起点的祖先关系，并保存 outputs/arg_v2/sync.json。
它不切换主仓库分支。正式解释器始终为 `/root/miniconda3/envs/rtdetr/bin/python`，
`arg_v2.sh` 固定本 worktree 的 Ultralytics/tools 导入路径。

## 2. prepare

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh prepare
```

默认先复用本 v2 已有快照；首次会尝试
`/root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/prepare.json`
及其相邻数据清单。核对保存清单、路径映射、配置/哈希、计数、目录元数据，记录
`reused_from` 和核验范围；没有可用记录才执行一次完整图像/标签 inventory。
显式要求仅使用 v1 快照、不可回退扫描时，可将上述 prepare 换成：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh prepare --reuse-snapshot /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/prepare.json
```

两种 prepare 选一种即可。公共源固定为主仓库
`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
独立 v2 init 使用原初始化与 nc80→nc1 映射、seed42；不加载 v1 best/last。
完整 109 字段配方保存到 outputs/arg_v2/train_args.yaml，保留原 B16/640、AMP、
200 epoch 上限、patience50、workers8、nbs64、AdamW 和全部增强。

快照复用要求数据冻结。轻量检查不能证明同路径文件内容刚被重新哈希；
发现或怀疑原地修改时使用末尾的显式重验命令。

## 3. 运行新的有界 preflight

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh preflight --seconds 900 --micro-batches 16
```

总边界 900 秒、至多 16 个训练 micro-batch，按正式 B16/640/AMP 检查实际更新、
有限梯度、显存、保存后新进程 FP32 零输入 warmup/真实 val 和 resume 状态。
不复用 v1 或本地小测结果。缺母版现有离线 AMP 资源时停止，不能跳过 AMP 检查。
输出保存在独立 preflights 子目录，最新汇总为 outputs/arg_v2/preflight.json。

## 4. 查看状态和预检资格

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh status
/root/miniconda3/envs/rtdetr/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/root/autodl-tmp/projects/Crack_RTDETR-arg_v2/outputs/arg_v2/preflight.json')
r = json.loads(p.read_text())
print(json.dumps({k: r.get(k) for k in ('status', 'start_eligible', 'checks', 'error', 'boundary_error')}, indent=2))
required = ('cpu', 'cuda_b16_amp', 'mechanism', 'new_process_val', 'resume')
assert r.get('status') in ('PASS', 'PENDING') and r.get('start_eligible'), 'Preflight is not eligible'
assert all(r.get('checks', {}).get(k, {}).get('status') == 'PASS' for k in required), 'A required check did not pass'
if r['checks'].get('native_scale', {}).get('status') != 'PASS':
    assert r['checks'].get('fallback_scale', {}).get('status') == 'PASS'
    assert r.get('gpu', {}).get('effective_update_arm') == 'diagnostic_init_scale_128'
    print('Native scale adaptation remains PENDING; isolated scale128 evidence is diagnostic only.')
print('Required preflight checks passed. Start will also revalidate the current binding.')
PY
```

必要项均须 PASS、start_eligible=true，并有实际有效更新。
仅原生初始 scale 适应可按既有规则保持 PENDING，前提是隔离 scale128 分支有效更新
及其余必要项均通过。正式 scaler 仍为原生初值，不能将诊断分支写成原生全过程 PASS。
本次交付时服务器以上项目均为 PENDING。

## 5. 正式 start：单独执行

仅第 4 步必要资格通过后，由用户执行下面命令。它再次校验代码/配方/初始化/缓存
数据身份与 preflight 绑定，然后创建 tmux `arg-v2-training`。
不会继承诊断权重、optimizer、scaler 或 RNG；从 e=0 开始。

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh start
```

正式 run：
`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/arg_v2_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。
已有活动 worker/tmux/run 时拒绝重复 start。patience50 允许原生提前结束。

## 6. 查看进度、日志和中断恢复

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh status
tmux attach-session -t arg-v2-training
```

退出查看、保持训练运行：先按 Ctrl-b，再按 d。只跟踪最后一次 dispatch 日志可用：

```bash
bash <<'ARG_V2_LOG'
set -euo pipefail
shopt -s nullglob
logs=(/root/autodl-tmp/projects/Crack_RTDETR-arg_v2/outputs/arg_v2/dispatches/*/console.log)
((${#logs[@]})) || { echo 'No dispatch log yet'; exit 2; }
tail -n 80 -f -- "${logs[-1]}"
ARG_V2_LOG
```

只有训练未完成、活动 worker/tmux 已结束且本实验 last 的 epoch/optimizer/scaler/
训练身份有效时，执行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh resume
```

训练完成但 final_eval 失败时不 resume。原 worker exit.json 和 Python/tee 退出码保留。
训练完成以 training_completed.json 的原生结束记录为准，不能依据 best.pt 或 tmux。

## 7. 训练结束后只需 finish

确认训练 worker/tmux 已结束，再执行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh finish
```

finish 复用同一个 best 的完整正式 FP32 final_eval val 锁；若尚未成功完成，则执行
缺少的首次正式 val。随后只执行尚未完成的锁定 test，最后一次完整打包。
首次正式 val/test 已默认同步导出每图全部 300 queries 和 GT，不需要补导出选项。
成功重入只复用记录/包；失败重入保留历史，恢复未完成阶段，不用重新训练。
test 只用于最终报告，不重新选择 best 或参数。

成功要求 finish.status=PASS 且 package.analysis_complete=true。
包路径和 SHA256 记录在
`/root/autodl-tmp/projects/Crack_RTDETR-arg_v2/outputs/arg_v2/package.json`，
包位于同级 packages/ARG_v2_ANALYSIS_*.tar.gz。
包含现有全量预测/GT、十阈值 AP、P/R/F1/AP50/AP75/mAP、曲线、源码、身份、
预检/训练/退出/恢复证据、best/last 大小与哈希、逐成员 manifest；不含权重或原始数据本体。

## 8. 故障评估恢复和纯 pack

训练结束、final_eval 推理失败但没有成功 val 时，可直接重试 finish；
若只想先恢复 val，则执行下面命令，成功后再 finish：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh val
```

若已有成功评估，但导出/曲线等被删除或不完整，普通 finish 会明确拒绝隐式补推理。
根据报错的 split 仅执行对应的显式恢复；保持原 best 和数据身份，旧记录保留：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh val --recover-export
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh test --recover-export
```

两条不是固定顺序步骤；只恢复被报告不完整的 split，test 始终要求完整 val 锁。
之后运行 finish 复用已完成阶段并打包。失败证据不会改写成成功训练退出码。

纯打包现有材料（也适用于预检/训练失败）：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh pack
```

pack 不启动模型、不读取数据集、不自动补评估；缺项列在 missing 中，
analysis_complete=false 时只是故障证据包。中断半包保留；重试只重新完成打包。

## 9. 数据显式重验：按需执行

仅数据疑似改变、发生原地编辑或明确要求全量核验时运行，不作为日常状态/启动步骤：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh recheck-data
```

该命令执行完整哈希/计数，保存独立重验和原快照。真实差异会将快照置 INVALID 并
阻止继续，不能借重验替换训练数据身份；恢复原数据后再次重验完全一致才恢复 PASS。
