# AutoDL操作

交付回复提供远端核对后的真实40位SHA；同步必须固定该SHA。不要把父LIF SHA当作组合SHA。以下脚本只服务本组合，不接受variant。

同步签名：`bash tools/sync_c19_lif_v1.sh FULL_SHA [MAIN_REPO] [WORKTREE]`。
默认主库 `/root/autodl-tmp/projects/Crack_RTDETR`，独立worktree `/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1`。
首次从主库fetch本任务ref，使用 `git show "$C19_LIF_V1_SHA:tools/sync_c19_lif_v1.sh"` 导出唯一临时脚本，在子shell内执行。最终回复含完整可复制命令。

同步校验origin身份、本任务ref可达性、LIF成功基点祖先关系、固定提交和必要文件，再创建detached worktree。同SHA且tracked clean可幂等核验，保留downloads/runs等未跟踪内容。不同SHA、tracked dirty、非本库linked-worktree均拒绝，原现场不变；已有训练时同SHA核验不会checkout或改源码。fetch使用独立ref和no-write-fetch-head，不污染别组FETCH_HEAD。同步不初始化、不安装、不派发。

唯一正式启动：

```bash
(cd /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1 && CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c19_lif_v1.sh start-direct)
```

shell和tmux worker均显式source `/root/miniconda3/etc/profile.d/conda.sh`、activate rtdetr，并设置本worktree PYTHONPATH及PYTHONUNBUFFERED=1。硬件仍是同一张4090、逻辑device=0；没有自动分配第二张GPU。

启动检查固定SHA/source、109字段配方、数据清单/标签fingerprint、公共init SHA及原始状态、原模块回归、非零融合、原loss/DN/optimizer，再在独立子进程检查B16/640 AMP原loss前后向并记录显存。启动审计打印Python/Torch/CUDA/GPU、源码位置、HEAD、YAML/recipe/data/init哈希。预期Python3.10.13/torch2.1.2+cu121，不符即停止报告，不升级依赖。

有限预检日志：`outputs/c19_lif_v1/preflight.log`。正式run固定为主库 `runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug`，受控init仅写本worktree `weights/c19_lif_v1_controlled_init.pt`。预检副本不用于正式训练，正式进程重新seed42加载未经训练且锁定哈希的init。已有init、launch、run、lock或同名活跃worker会保护现场并停止，不隐式续训/重试/加后缀。预检失败也保留证据和本任务reservation，应先检查错误和资源再显式处理本任务现场；脚本不会清理它或其他任务。

会话固定 `c19_lif_v1-training`。status通过PID启动token、退出码和结果文件判别NOT_STARTED/DISPATCHED/RUNNING/SUCCESS/FAILED，tmux存在不代表成功。训练主进程日志直写，shell退出trap保留包括启动失败在内的真实退出码；外层tee也保留Python退出码。

```bash
(cd /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1 && bash tools/autodl_c19_lif_v1.sh status)
tail -n 80 -f /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1/outputs/c19_lif_v1/console.log
```

训练SUCCESS后：

```bash
(cd /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1 && \
 bash tools/autodl_c19_lif_v1.sh status && \
 bash tools/autodl_c19_lif_v1.sh val && \
 bash tools/autodl_c19_lif_v1.sh test && \
 bash tools/autodl_c19_lif_v1.sh pack-complete)
```

评估固定imgsz640/batch16/device0/workers0/half=False/conf0.001/iou0.7/max_det300/augment=False/rect=False/plots=True/seed42，原RT-DETR预处理和corrected_sorted_conf_mask_v1。test要求本实验同best SHA的独立val完成，配置、split及标签fingerprint与训练证据一致。P/R为各模型自身最大F1工作点；没有新NMS/TTA、重复sigmoid或缩放。

pack-complete不补跑评估。缺关键来源、预检、曲线、checkpoint、预测GT或完整split证据即拒绝。输出默认主库 `downloads/c19_lif_v1`，按时间戳exclusive创建tar.gz及.sha256/.inventory.json/.verification.json，并打印4个绝对路径。包内包括best/last、training args/results、原日志、独立val/test完整指标图/预测GT、源码快照/patch、父版本证据、运行来源及所有哈希/映射/预检报告；不包含dataset、预检更新权重或凭证。内部MANIFEST逐文件读回校验。

本地重跑工程验证（已有依赖）：

```text
python tools/init_c19_lif_v1.py --source PATH_TO_PUBLIC_INIT --output weights/c19_lif_v1_controlled_init.pt --report outputs/init.json
python tools/check_c19_lif_v1.py --source PATH_TO_PUBLIC_INIT --initialized weights/c19_lif_v1_controlled_init.pt --real-dataset PATH_TO_CRACK_DET --output outputs/unique_checks
python tools/check_c19_lif_v1_ops.py --output outputs/ops.json
```

初始化和验证目录要求新路径，已存在内容不会被覆盖。`--capacity-batch 16`只在已有CUDA服务器显式启用；该选项本地未执行。
