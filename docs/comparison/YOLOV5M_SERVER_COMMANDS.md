# YOLOv5m：AutoDL 运行说明

代码接入已完成，尚未启动正式训练。以下训练命令由用户随后执行。完整固定参数见 `benchmarks/comparison/yolov5m/recipe.json`、`hyp.yaml` 和 `YOLOV5M_AUGMENTATION.md`。

## 1. 独立工作树

在 AutoDL 终端执行。已有目标目录时先核实，不覆盖；不切换主项目或其他创新工作树。

```bash
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m
git -C "$MAIN" fetch origin refs/heads/bench/yolov5m:refs/remotes/origin/bench/yolov5m
git -C "$MAIN" rev-parse origin/bench/yolov5m
# 核对上行完整 SHA 与本次交付回执相同。
test ! -e "$WORK" && git -C "$MAIN" worktree add --detach "$WORK" origin/bench/yolov5m
cd "$WORK"
git rev-parse HEAD
```

已存在且就是交付提交的工作树可直接 cd 使用。公共 comparison-base 的 detached HEAD 无需更改。原数据入口始终是主项目的 configs/crack_autodl.yaml 和 datasets/crack_det。

## 2. 解释器与资产

先读现有解释器版本，不改变正在运行实验的环境：

```bash
export YOLOV5_PYTHON=/root/miniconda3/envs/rtdetr/bin/python
"$YOLOV5_PYTHON" -c 'import sys,torch,torchvision,numpy; print(sys.executable,sys.version); print(torch.__version__,torchvision.__version__,numpy.__version__)'
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/assets.py --assets "$WORK/outputs/yolov5m/assets"
```

资产入口只下载官方 v7.0 源码与 yolov5m.pt，复用前校验完整 commit、补丁、权重大小和 SHA256。错误资产不覆盖；下载中断的 .partial 留待检查。需要网络代理时显式配置当前终端 HTTPS_PROXY，勿改运行中实验。不要换 mu/u 模型或 latest 链接。

prepare 若报告缺包，可在**单独 venv**中补齐；这一步不会升级原 rtdetr。已满足依赖时直接复用解释器，不必另装：

```bash
BASEPY=/root/miniconda3/envs/rtdetr/bin/python
VENV=/root/autodl-tmp/envs/comparison-yolov5m-v7
# 新建；已有目录请先核实，勿覆盖。
test ! -e "$VENV" && "$BASEPY" -m venv --system-site-packages "$VENV"
export YOLOV5_PYTHON="$VENV/bin/python"
"$YOLOV5_PYTHON" -m pip install -r benchmarks/comparison/yolov5m/requirements-server.txt
```

固定的普通依赖配置面向 Python3.10/NumPy1.26.4；复用服务器现有匹配的 torch/torchvision。历史服务器记录为 torch2.1.2+cu121 / torchvision0.16.2；当前实际状态必须以上面的导入检查为准。本地另用 Python3.9、torch2.7.1+cu118、NumPy2.0.2 完成 CPU 验证，不代表服务器 CUDA 已验证。requirements 文件不包含 torch 安装，也不安装 wandb/clearml/comet/albumentations 或新版 ultralytics。所有子进程清除外来 PYTHONPATH，固定 upstream 搜索路径，关闭自动依赖安装和远程跟踪。

## 3. 只检查（不训练）

输出必须是新目录；这一步统计实际列表/标签，不哈希或解码全部图片，不要求 AUDITED。配置、包版本、官方权重加载和模块导入位置会写入检查目录。

```bash
PREFLIGHT="$WORK/outputs/yolov5m/preflight_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/run.py prepare \
  --run "$PREFLIGHT" --source-project "$MAIN"
```

也可独立运行仅数据检查的 data.py（--data、--data-root、--source-project、--output 都是实际支持的参数）。从其他工作树运行不会复制数据或用旧 split_manifest 覆盖目录。新数据身份是路径+标签字节+图片大小的轻量身份，明确不同于已有全图字节审计身份；旧原图族重叠事实仍见公共报告。

## 4. 用户随后启动正式顺序实验

使用新的 RUN_NAME，避免与 preflight 目录相同：

```bash
RUN_NAME="yolov5m_e200_b16_s42_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash scripts/autodl_yolov5m.sh "$RUN_NAME"
tmux attach -t comparison-yolov5m
```

顺序为：轻量检查/冻结 → 训练 → 同一 best 的 val/test FP32 导出 → 公共 CPU 评测 → 汇总。退出 tmux 用 Ctrl+B 再 D，不等于任务完成。完成后的 pane 通过 remain-on-exit 保留。已有同名 session、run、launcher log 或 active lock 会拒绝启动，不终止其他任务。需要当前终端前台执行时，显式使用 --foreground。

不等待 GPU 完全空闲，不调整 batch16/640/精度；显存不足明确失败并保留日志。Ctrl+C/SIGTERM 被记录为 interrupted/failed。日志中出现 NOT_EXECUTED 表示尚未执行，绝不填参考结果。

输出在 outputs/yolov5m/runs/RUN_NAME，包括：

- frozen.json、resolved_options.json、data/manifest.json、data/data.yaml 和三组绝对路径列表。
- status.json：每个独立阶段的命令、起止时间、真实子进程 exit code、Python tee exit code；外层 .launcher_exit.json 另记 pipeline 和 shell tee 退出码。
- native/results.csv、hyp.yaml、opt.yaml、pretrained_load.json、autoanchor.json、effective_training.json、epoch_state.json、training_complete.json。
- native/weights/best.pt、last.pt：保留优化器以支持显式恢复，保留 EMA；不会自动 strip 后丢掉 epoch 信息。
- selected_best.json 固定同一权重身份；evaluation/ 下 val/test GT、全精度 JSONL、导出校验回执和 *_unified_metrics.json；summary.json 给出实际完成 epoch、best epoch、结束原因和五项统一指标。

数据加载器自身仍正常执行首次标签/尺寸缓存；缓存位于本 run 的列表目录，未伪造。遇到损坏 JPEG/缺失标签会失败，不自动改原图或丢样本。禁止改数据凑数量。

## 5. 分步、恢复与 CPU 重算

若沿用一个已 prepare 成功的目录，可明确分别执行：

```bash
RUN="$PREFLIGHT"
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/run.py train --run "$RUN" --source-project "$MAIN"
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/run.py export --run "$RUN" --source-project "$MAIN"
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/run.py evaluate --run "$RUN"
```

只有训练正常结束并存在 training_complete.json 时才导出。已完成的缓存不会重新 GPU 导出。旧缓存自动优先在主项目及兄弟公共工作树的 outputs/comparison_prepare/*/coco 中查找；也可给 export 传 --gt-cache /已有结果/coco。缓存需有同级父目录 dataset_manifest.json，路径/ID/标签哈希/图片大小必须对应实际目录，且标注逐项吻合；旧绝对 Windows 图像路径会拒绝。有效旧 GT 不需要重新读图；无可用缓存时只读 val/test 图片头与必要 EXIF 取得宽高。无论哪条路径均不先把 train 转为 COCO。

中断后只允许明确的同一 run 恢复：

```bash
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/run.py train --run "$RUN" --source-project "$MAIN" --resume
```

严格核对代码/补丁/配置/环境/数据身份、本 run 的 last.pt、checkpoint epoch 与 epoch_state、关键 opt；保存新 stage/log，保留最初 pretrained_load 记录。正常完成的 run 不允许 resume。硬断电若 checkpoint 与状态不一致会拒绝恢复，不自动猜测。恢复沿用上游随机流重启行为，不承诺逐位等同不中断训练。残留 active.lock 先核实该 run PID 已结束，再人工处理；不自动删除活跃锁或输出。

纯 CPU 指标重算无需再推理，用一个**新的**输出文件：

```bash
"$YOLOV5_PYTHON" benchmarks/comparison/evaluation/evaluate.py \
  --gt "$RUN/evaluation/test_gt.json" \
  --predictions "$RUN/evaluation/test_predictions.jsonl" \
  --output "$RUN/evaluation/test_unified_metrics_recomputed.json"
```

统一评测沿用 corrected_sorted_conf_mask_v1，类别 YOLO0→公共1，每图包括空预测；坐标为原图 FP32 浮点值，经真实 x/y resize gain 与整数 letterbox padding 反变换，无展示文本舍入/额外裁剪。独立评测 640、batch16、workers0、FP32、conf>.001、class-aware NMS IoU=.7、max_det300、max_nms30000，无 TTA；NMS 超时提前退出关闭，防止并行任务拖慢时静默漏图。AP IoU 为 .50:.05:.95，P/R 为公共最大 F1 工作点。JSON 是0–1值，百分数需乘100。

参数量/GFLOPs 入口在训练后手动调用，不是训练门槛，也不测速：

```bash
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/run.py resources --run "$RUN"
```

使用实际 nc=1 的 best，CPU 输入1×3×640×640，分别测 fused/unfused，thop MACs×2 记 FLOPs。并行硬件条件下未提供论文速度结论。
