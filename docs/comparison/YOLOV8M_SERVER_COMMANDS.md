# YOLOv8m AutoDL 运行入口

本次交付是代码；尚未启动正式训练或完整 val/test 推理。下列启动命令供用户随后执行。
YOLOv5m 与 YOLOv8m 可以各自使用 GPU 0；没有 GPU 空闲检查、整卡锁、另一模型完成依赖或自动缩 batch。

## 1. 核对并进入独立工作树

公共基点为 `529c456b9404f1d9ab66d82d2b2f9ec7e0c98545`，本次分支为 `bench/yolov8m`。
交付提交的完整 SHA 以本次最终汇报和远端核验为准。首次创建可执行：

```bash
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
git -C "$MAIN" remote -v
git -C "$MAIN" status --short
git -C "$MAIN" worktree list
git -C "$MAIN" fetch origin bench/yolov8m
git -C "$MAIN" worktree add -b bench/yolov8m \
  /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m origin/bench/yolov8m
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m
git merge-base --is-ancestor 529c456b9404f1d9ab66d82d2b2f9ec7e0c98545 HEAD
git rev-parse HEAD
git ls-remote origin refs/heads/bench/yolov8m
```

已有同名分支或工作树时，先核对其来源和未提交改动，再进入已有工作树；不要重复创建、reset、删除或强推。
训练和导出要求本工作树干净，并把实际代码提交写入结果。不要从 YOLOv5m 未完成 HEAD 派生。

## 2. 一键顺序运行（由用户执行）

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m
bash scripts/autodl_yolov8m.sh start \
  --run-id "yolov8m_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)" \
  --base-python /root/miniconda3/envs/rtdetr/bin/python \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml
tmux attach -t comparison-yolov8m
```

依次执行 bootstrap → 轻量 preflight → train → val 导出/CPU 评测 → test 导出/CPU 评测 → summary。
`run` 替代 `start` 可在当前终端执行同一流程。设备为 0，训练/训练内验证 batch 均为16，640，AMP开启。
真实 OOM 会失败退出并保留日志；没有 AutoBatch、OOM 重试缩 batch 或尺寸、等待 GPU 利用率归零。

启动脚本只拒绝已有 `comparison-yolov8m` 会话、重复 run ID 或重复日志。不会查看或操作 `comparison-yolov5m`。
会话在任务释放前已设置 `remain-on-exit`，完成和失败界面均保留。结束后需要新会话时，由用户先检查并处理该同名会话。

可选 `--public-coco /实际已完成公共准备目录/coco`。只有路径、ID、维度、类别和全部框与当前标签对应的 val/test GT 才复用；
已有但不匹配的 GT 直接报错。未提供或文件缺失时，仅从 YOLO 标签和图片头生成 val/test GT，不转换 train，不要求公共完整审计完成。
`--data-root` 只用于明确的数据根映射；服务器默认直接采用真实 YAML 的绝对根。Windows 的本地验证清单不能复制过去冒充服务器路径。

## 3. 分步入口

先准备官方固定资产和独立环境，此步骤没有训练：

```bash
/root/miniconda3/envs/rtdetr/bin/python benchmarks/comparison/yolov8m/bootstrap.py \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
PY=$(cat .runtime/yolov8m/python_path.txt)
RUN="$PWD/outputs/yolov8m/manual_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
"$PY" benchmarks/comparison/yolov8m/run.py preflight --run "$RUN" \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml
"$PY" benchmarks/comparison/yolov8m/run.py train --run "$RUN"
"$PY" benchmarks/comparison/yolov8m/run.py export --run "$RUN" --split val
"$PY" benchmarks/comparison/yolov8m/run.py evaluate --run "$RUN" --split val
"$PY" benchmarks/comparison/yolov8m/run.py export --run "$RUN" --split test
"$PY" benchmarks/comparison/yolov8m/run.py evaluate --run "$RUN" --split test
"$PY" benchmarks/comparison/yolov8m/run.py summary --run "$RUN"
```

默认在 `.envs/yolov8m` 建立 `--system-site-packages` 私有 venv，先记录基环境依赖探测，再按固定 overlay 补齐 NumPy1.26.4、OpenCV等小依赖。
使用 `pip --no-deps`，继承已有 Torch/TorchVision，不会在 rtdetr/YOLOv5 环境安装、卸载或升级。缺少 overlay 之外的依赖会报告实际错误，不能盲装最新版。
`--reuse-base` 仅在依赖探测通过时只读复用基环境，不会往该环境 pip install。服务器当前 Torch/CUDA 兼容性由探测和首次实际 AMP 初始化确认。

官方源码放在 `.vendor/yolov8m/ultralytics-v8.3.20`，初始权重放在 `.runtime/yolov8m/assets/yolov8m.pt`。
断网时可提前从锁定 URL 获取此文件；大小和 SHA256 不符即拒绝。源码准备中断不会覆盖已有不同内容，查看错误后修复对应独立目录。
没有随机初始化回退或浮动 latest 下载入口。

## 4. 冻结版本与配方

实际固定实现为官方 **v8.3.20 / `f4d8f7765a490f3920e2d14c592a2967e347f185`**。
拟定 v8.3.0 在 nc=1 重建时使用深度可分离分类分支，改变旧官方 YOLOv8 头；v8.3.20 已有官方 legacy 分支修复。
选择发生在任何正式训练/指标产生前。详细证据、重新核查的参数及增强含义见 `YOLOV8M_INTEGRATION_REPORT.md`、`YOLOV8M_AUGMENTATION.md`。

`recipe.yaml` 冻结200上限、patience50、batch16、640、seed42、SGD、lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、
cosine、warmup5、nbs64、AMP、deterministic。nbs64通过梯度累积实现，常规阶段累积4步；并非单次装64张。
loss保持官方box7.5、cls.5、dfl1.5，完整展开值、实际优化器/参数组和每轮 LR/累积都写入 run。

best和patience只使用训练val的未作日志舍入的mAP50–95；原生 `0.1*AP50+0.9*mAP50–95` fitness另存日志。
best采用官方的相等时更新规则；训练内验证为FP16、batch16、workers0、conf.001、NMS IoU.7、max_det300、rectFalse。
最终主表由独立FP32导出和公共 `corrected_sorted_conf_mask_v1` 评测产生。测试集不参与选择或调参。

## 5. 输出、恢复与 CPU 重算

输出根：`outputs/yolov8m/RUN_ID/`；启动日志：`outputs/yolov8m_launch/RUN_ID/`（resume有独立attempt后缀）。

| 产物 | 含义 |
|---|---|
| `manifest.json`、`data.yaml`、split清单、`input_checksums.json` | 本run实际数据入口与轻量身份；未宣称全量内容审计 |
| `gt/val.json`、`gt/test.json` | 全覆盖原图浮点GT，含真实来源/复用校验 |
| `identity.json`、`source_identity.json`、`initialization.json` | 本仓库/官方/补丁/预训练/类别适配身份和参数转移 |
| `frozen_recipe.json`、`expanded_train_args.json`、`actual_training_setup.json` | 冻结、完整展开与实际执行参数 |
| `environment.json`、`pip_freeze.txt`、`epoch_trace.jsonl` | 环境、实际batch/AMP/LR/累积与增强状态 |
| `cache/SPLIT/*.labels.json` | 本模型只读标签及尺寸生成的真实缓存，原子写+局部锁 |
| `train/weights/best.pt`、`last.pt` | 官方EMA/优化器信息及本实验身份，保留显式续训能力 |
| `predictions/*.jsonl.gz`、`*_complete.json` | 包括空预测图片的全精度公共缓存、完整性hash |
| `metrics/*.json`、`summary.json` | 五项指标的0–1原值与百分数；缺失字段为null |
| `*_status.json`、启动日志的`*.exit.json` | Python阶段状态与command/tee各自真实退出码；信号单列 |

`stage.py`给每个阶段建立自己的子进程组，信号只转发给该阶段及其worker，记录command与tee真实返回值。
任一阶段或tee失败后停止后续步骤，Ctrl+C/TERM标为interrupted。原生负信号返回码保留在`command_exit`，外层退出码转换为128+signal。

中断续训必须显式选择原run：

```bash
bash scripts/autodl_yolov8m.sh start-resume --run-id 原RUN_ID \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
```

会校验模型、last/best配对、原始权重、代码/补丁、数据和冻结配置；恢复早停的best epoch/fitness，且恢复关闭后的worker状态。
如果CSV领先于last checkpoint，先保存旧CSV再恢复checkpoint中的已完成行。已完成训练、已达到停止条件或不同身份会拒绝自动续训。
正式resume的完整GPU恢复仍需服务器验证；本地验证覆盖其早停与worker恢复逻辑。

如果训练已完成，仅导出/评测失败，直接重执行相应`export`/`evaluate`命令；不要再train或resume。
完整且身份一致的预测缓存会复用，失败的`.partial`保留，重试另建临时文件。
仅需格式调整/指标重算时执行`evaluate`和`summary`，全部在CPU完成。不同重算结果另存带后缀文件。

参数量/GFLOPs测量入口（不要求GPU独占，不作为训练前置）：

```bash
"$PY" benchmarks/comparison/yolov8m/run.py measure \
  --output "outputs/yolov8m_measure_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).json"
```

测量的是nc=1、FP32、CPU、640×640的模型，分融合前后，以THOP MACs×2为FLOPs口径；不输出论文测速。
本地已做的小尺寸CPU smoke不等于正式640训练已验证。真实AutoDL GPU/AMP、CUDA显存及tmux/信号最终行为待用户运行时核验。
