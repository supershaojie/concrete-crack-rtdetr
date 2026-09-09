# DRA-AIFI 实测验证

2026-09-09 本机：Windows、Python 3.9.25、PyTorch 2.7.1+cu118、CUDA 11.8、NVIDIA GeForce RTX 2060、Ultralytics 8.4.21。没有连接服务器执行训练或真实 val/test。

运行 `tools/check_dra.py`，退出码 0；`tools/check_dra_tools.py`，退出码 0。原始详细结果分别在 [validation.json](validation.json)、[tools_validation.json](tools_validation.json)。测试时 HEAD 为分支基点、DRA 修改尚未提交，不能将报告中的 base commit 误读为独立已提交实现；交付源码的逐文件 LF 散列见 [validation_source_hashes.json](validation_source_hashes.json)。调试临时模型已经由测试清理，不用于正式初始化。

## 数值、结构与学习

| 项目 | 实测 |
|---|---|
| nc=1 未融合参数 | C2 20,082,772；DRA 20,087,148；新增 4,376 |
| 模块尺寸 | batch2、20×20、5×7、7×5、1×1，pre/post norm、train/eval；CPU/CUDA 全通过 |
| 零初始化模块比较 | 最大绝对差 0；atol=2e-5、rtol=2e-5 |
| 非零偏置 | 有限、绝对值≤0.5、严格对称、对角为0，后4头为0 |
| 几何 | 横向/纵向/对角二倍角、max(H,W)归一化、尺寸变换、half→float清缓存通过 |
| 独立公式对照 | 对 2×3 输入逐 i/j/head 用 FP64 标量公式重算；最大差 2.3589253078659667e-08 |
| 公共权重 | 全部533个 C2状态 identity 映射，值完全相同；无异常 missing/extra/shape mismatch |
| 原生类别重建 | nc80→1 精确加载528/537状态，9个预期分类shape skip；公共初始化与受控C2一致 |
| 原生 train API | 通过真实 get_model 重建，在 train setup 前主动停止，未进入正式 epoch 循环 |
| Decoder/DN | 3层、300常规queries；小样本 DN split=[200,300]；训练5项输出语义保持 |
| 评估层选择 | eval_idx=2；最终 boxes/sigmoid(scores) 拼接与原生返回逐元素相同 |

整网比较使用受控公共初始化、batch2、160×192 合成图，固定 DN 随机种子71，分别显式 train/eval。执行原生 predict、loss、DN 和 Hungarian 路径。

| 模式 | eval最大差 | train输出最大差 | 原生loss最大差 | 容差 atol/rtol |
|---|---:|---:|---:|---|
| CPU FP32 | 0 | 0 | 0 | 2e-5 / 2e-5 |
| CUDA FP32 | 0 | 0 | 0 | 2e-5 / 2e-5 |
| CUDA AMP | 0 | 0 | 0 | 0.005 / 0.005 |

这仅描述本次实测，不保证跨设备/框架版本逐 bit 相同。

每种整网模式各做3次 AdamW 合成样本更新（lr0.0005、decay0.0001；AMP调试GradScaler初始scale128）。全部 trainable 参数恰好进入 optimizer 一次，四个新增参数张量无缺失/重复。第一步末层 weight/bias 有限且非零梯度，前两层梯度为0；随后前两层获得有限非零梯度。权重衰减引起的变化没有被误当作学习梯度，不要求随机样本 loss 单调下降。

三步后新增 bias 最大绝对值：CPU `0.0008454740745946765`、CUDA FP32 `0.000779781665187329`、CUDA AMP `0.0007348601939156651`。保存/重载后输出最大差均为0；已更新整网在 CPU 和 CUDA 上实际 `.half()` 前向均输出有限 FP16 `[2,300,5]`。

## 工具链

- 109字段默认配方对比只有 model/name/save_dir 改变；完整对比 [parameter_diff.json](parameter_diff.json)，初始化逐键证据 [initialization_mapping.json](initialization_mapping.json)。
- 原生验证器低置信度、非排序分数、相同分数测试通过；导出未改变排序或原生mask实际选中的预测。
- 实际 `RTDETR.val` 独立入口在每 split 两张合成图片上验证，CPU、imgsz160、batch1；val/test 各导出2图、600条预测与GT。同一 checkpoint/data/source 校验通过，图像为128×192和160×192，覆盖坐标还原路径。
- 合成模型/图像的零AP只说明工具运行，不是 DRA 训练精度；没有绘制虚构实验图，也未把临时合成图留作方法结果。
- 状态分类 not_submitted/dispatched/进程失踪/失败退出/成功退出检查通过。OOM guard 设置核对通过，并阅读原 trainer 确认重试阈值3会直接抛出异常。
- 两个 shell 脚本 `bash -n` 通过；所有新增 Python AST 解析通过，CLI/模块导入检查通过。
- 可用材料打包、成员散列回读、已存在包拒绝覆盖通过。缺少正式材料时 complete=false；最终 missing_evidence 枚举包含曲线/混淆矩阵/样例，[package_validation.json](package_validation.json) 记录这次针对性检查。未执行真实结果的 complete=true 打包，因为正式结果尚不存在。

## 未运行

`formal_training=NOT_RUN`，真实完整 val/test=`NOT_RUN`，`full_server_preflight=NOT_RUN`。没有正式200e、真实batch16全套预检、多种子或交叉验证。AutoDL 4090/PyTorch2.1.2、真实SSH/tmux投递与服务器资源适配未在本机冒充验证；提供的服务器入口将在实际运行时记录版本、资源、配置、PID和退出码。
