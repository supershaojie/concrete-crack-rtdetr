# YOLOv5m 原生预处理与固定配置训练工具

本分支从指定 pilot `07d77c16ec168bd9547cd5aac04817c1d5d46e39` 派生。固定 v7.0 上游、原生 resize/LetterBox/RandomPerspective/乘法 HSV、原生检测头/loss/EMA/更新顺序；没有隐式 Albumentations，CutMix/CopyPaste 必须为0。上一版 b19 文件仅保留来源归档，不进入本次运行管线。

默认值在 config.py，用户 YAML 覆盖数值和已支持选项。`v5_ft_01.yaml` 是首组待验证候选。运行准备后，所有阶段使用 run 内 `resolved_config.yaml`/`train_hyp.yaml`，不重读外部候选。recipe.json/hyp.yaml 只说明固定默认值，worker 不用它们覆盖 run 参数。

启动：`bash scripts/autodl_yolov5m_coco_native_ft_v1.sh start --run-id v5_ft_01 --config /absolute/path/v5_ft_01.yaml`。同 run 恢复使用 `resume --run-id v5_ft_01`；选定完整 run 后 `finalize --run-id v5_ft_01` 只补公共 test。start 只做公共 val。旧启动器和旧 COCO/scratch 文档是历史归档，不用于本分支。

完整路径、固定 SHA、两组候选、tmux、恢复、finalize 和归档命令见 [服务器命令](../../../docs/comparison/YOLOV5M_COCO_NATIVE_FT_V1_SERVER_COMMANDS.md)，实现和验证结论见 [交付报告](../../../docs/comparison/YOLOV5M_COCO_NATIVE_FT_V1.md)。

输出：`outputs/yolov5m-coco-native-ft-v1/runs/<run-id>/`。候选写到 gitignored `runtime_configs/`。环境只读复用；一次新建源码运行目录，旧已补丁源码不覆盖，原始官方权重核验后直接读取。原图只读，已有尺寸清单经轻量核对后生成 run 内原生标签缓存，AutoAnchor 核验 train shapes 并记录真实 BPR。

必要检查：`python -m unittest discover -s benchmarks/comparison/yolov5m -p test_config.py -v`，`python -m unittest discover -s benchmarks/comparison/yolov5m -p test_control.py -v`，`python -m unittest discover -s scripts -p test_yolov5m_native_launcher.py -v`。`test_native_ft.py --assets PATH --output NEW_PATH` 只做小样本合成生命周期/预处理检查；其 scope 明确为 SMOKE_ONLY_SYNTHETIC，不构成正式实验或双任务 batch16/640 显存验证。
