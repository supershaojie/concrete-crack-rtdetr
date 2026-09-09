# AutoDL 固定版本运行

最终交付回复提供已推送的真实完整 SHA 及首次同步命令。
`tools/sync_lif_down.sh` 从主仓库 fetch `codex/lif-down`，确认指定 SHA 属于分支历史且包含 C2 基点，
随后在 `/root/autodl-tmp/projects/Crack_RTDETR-lif-down` 创建 detached worktree。
远端前进不改变固定 SHA；同 SHA 干净 worktree 可重复同步。
目标有修改、SHA 不同、不是同一仓库或不是 linked worktree 时，拒绝并保留现场；不 reset/clean/stash/delete。
不要求 `branch --show-current` 返回分支名。

同步后在独立目录执行：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-lif-down
bash tools/autodl_lif_down.sh start-direct
bash tools/autodl_lif_down.sh status
tail -n 80 outputs/lif_down/console.log
tail -f outputs/lif_down/console.log
```

start-direct 使用已有 rtdetr conda 环境；可通过 LIF_DOWN_PYTHON 指定现有解释器。
不安装或升级依赖。核对 Python/Torch/CUDA、ultralytics 与 LIFDown 来源、源码投递记录、YAML、初始化 SHA、
C2 109 字段和数据、已有 tmux/进程/输出，再独占认领 run 并启动 `lif-down-training` tmux。
原生 AMP helper 需要 bus.jpg/yolo26n.pt，优先复制主仓库已有资源，缺失时仅下载官方资源，不更换环境。
OOM 直接失败并保留输出；禁止原生自动减半 batch 的重试，AMP 检查若关闭 AMP 也停止。
训练 worker 独立记录 Python 退出码及 shell 退出码，命令 tee 保留 PIPESTATUS。

状态只有 NOT_STARTED / DISPATCHED / RUNNING / SUCCESS / FAILED。
SUCCESS 必须两个退出码均为 0，且存在 best.pt、last.pt、results.csv；tmux 存在只表示会话存在。
原生 patience=50 保留，因此 SUCCESS 表示原生训练流程正常结束，可包含其正常早停，不伪称恰好完成 200 轮。
失败锁、日志和输出不会自动清除，需人工审计后另行安排。

默认目录：

- 主仓库：`/root/autodl-tmp/projects/Crack_RTDETR`
- 训练：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/lif_down_rtdetr_r18_lite_e200_b16_onlineaug`
- 日志：`/root/autodl-tmp/projects/Crack_RTDETR-lif-down/outputs/lif_down/console.log`
- 包：`/root/autodl-tmp/projects/Crack_RTDETR/downloads/lif_down`

正式训练成功后，使用同一 best.pt 独立评估和打包：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-lif-down
bash tools/autodl_lif_down.sh val
bash tools/autodl_lif_down.sh test
bash tools/autodl_lif_down.sh pack-complete
ls -lh /root/autodl-tmp/projects/Crack_RTDETR/downloads/lif_down
```

val/test 默认 imgsz640、batch16、workers0、FP32、conf=.001、iou=.7、max_det300、augment=false。
采用项目已使用的 `corrected_sorted_conf_mask_v1` 独立评估口径：置信度排序后重算阈值 mask。
训练 best 选择仍是 C2 原验证器，训练逻辑没改；历史原 mask 口径数字不可直接作为同口径差值，
比较时需让 C2 使用相同独立评估口径。两份报告必须有相同 checkpoint/data/code SHA 及配置。
每张图同一轮推理导出全部 300 个预测和 GT，保留原图像素 xyxy、score、class、指标筛选标记。
包含 P/R/mAP50/mAP50-95/AP75/完整 IoU AP、PR/F1/P/R 曲线、混淆矩阵与样本图。

pack-complete 要求训练、val、test 和全部必要证据完成，缺失则报错，绝不补跑。
按时间戳创建 `.tar.gz`、`.sha256`、`.inventory.json`、`.verification.json`，逐成员重新读取校验散列。
包内有 training/、val/、test/、console/command logs、best/last、训练表与图、初始化/映射/解析配置、
完整 Git SHA、YAML、源码快照与相对 C2 patch、环境清单及 MANIFEST。
只接收已知生成物，不采集数据集或任意额外文件；未知文件/符号链接会报错保留现场。
包及同名 sidecar 已存在则拒绝覆盖，无 20MiB 上限。

本地测试不是 AutoDL 验证。服务器 Python/PyTorch2.1.2、batch16 显存、实际 tmux、完整训练和完整评估尚未执行。
