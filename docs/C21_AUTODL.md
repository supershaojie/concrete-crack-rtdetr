# C21 AutoDL 操作

使用服务器已有 `rtdetr` 环境，不安装或升级共享依赖。工作树独立，保留主仓库和其他实验。以下创建命令只需执行一次，已有同名目录/分支时会停止。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR
git fetch origin refs/heads/codex/c21-rtdetr-r18-lite-sala:refs/remotes/origin/codex/c21-rtdetr-r18-lite-sala
git worktree add -b codex/c21-rtdetr-r18-lite-sala \
  /root/autodl-tmp/projects/Crack_RTDETR_sala \
  origin/codex/c21-rtdetr-r18-lite-sala
cd /root/autodl-tmp/projects/Crack_RTDETR_sala
git log -1 --oneline
bash tools/autodl_c21.sh prepare
```

`prepare` 检查干净提交、导入路径、PyTorch 2.1.2/CUDA、C2 源权重 hash、全部 args、真实数据路径；生成独立干净初始化，审计 CPU/CUDA/AMP/half、nc=1/80、optimizer 分组、三层替换，执行三个真实 640、batch2、DN smoke 更新，然后生成启动计划并退出。smoke 使用独立 optimizer/EMA/debug checkpoint；batch16 配方不改。再次 prepare 会保留旧 smoke 目录，以时间戳创建新审计。

源权重：`/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。

C2 配方：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`。

数据：`/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`。若路径或 109 字段与核验 C2 不符，工具停止并给出原因，不使用其他实验权重/默认值替代。

用户决定正式训练后，单独执行：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_sala
bash tools/autodl_c21.sh start
```

启动前重查提交、源码/权重/数据/审计 hash、运行目录、tmux 和进程；原子锁阻止并发重复启动。Trainer 首批前检查实际参数、完整状态、空 optimizer 和 EMA。训练位于独立 tmux，日志和子进程真实退出码保留。启动后的结果目录或锁存在时，禁止再次 prepare/start 覆盖。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_sala
bash tools/autodl_c21.sh status
tail -n 80 outputs/c21/launch_c21/console.log
cat outputs/c21/launch_c21/process_exit_code.json
# 需要交互查看时：Ctrl-b d 可退出查看而继续训练
tmux attach -t c21_c21_rtdetr_r18_lite_sala_e200_b16_onlineaug
```

未启动或尚未退出时，日志/退出码文件可能还不存在；status 会展示准备状态、进程和已有文件。结果目录为：

`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c21_rtdetr_r18_lite_sala_e200_b16_onlineaug`。

成功退出后依次评估和打包：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_sala
bash tools/autodl_c21.sh val
bash tools/autodl_c21.sh test
bash tools/autodl_c21.sh pack
```

val/test 都固定 `weights/best.pt`、显式 split、640、batch16、conf=0.001、iou=0.7、max_det=300、half=False、augment=False、rect=False；test 必须匹配 val 的权重/数据 hash 和评估设置。iou 是记录的原生验证参数，RT-DETR 原生后处理不额外引入 NMS。形状诊断额外采用 confidence≥0.25，不改变主 AP 指标。

评估报告：`outputs/c21/evaluation_val/metrics_summary.json` 和 `outputs/c21/evaluation_test/metrics_summary.json`。

压缩包：`outputs/c21/c21_sala_small.tar.gz`，附 `.sha256` 和 `.inventory.json`。包含训练日志、allocation、args、results.csv、独立指标/前32图预测与GT、初始化审计、启动/退出证据、模块源码和 C2 diff。排除权重、图片、缓存和参考模块包；超过 20 MiB 时列出主要来源并停止，不静默删减证据。仅在未来正式训练和两次独立评估均结束后打包，本次没有伪造这些结果。

本机无法替代服务器运行验证。AutoDL/tmux/200轮训练/正式val-test 在本次交付时仍未执行。
