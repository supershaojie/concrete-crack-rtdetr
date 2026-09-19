# 本次 PBI 验证记录

本机：Windows、现有 rtdetr Conda、PyTorch 2.7.1+cu118、RTX 2060。
服务器要求：Python 3.10、PyTorch 2.1.2+cu121、RTX4090；**未登录服务器**。
下列本机检查不替代目标服务器的完整预检或启动许可。

## 已实测通过

- 母版祖先、仓库身份、参考 ZIP 和公共未训练源 SHA256 核对。
- 原 CBR/LIF CRLF→LF SHA256 保持合同值；原 AIFI、Decoder、loss、query 选择源码未改。
- 真实母版 109 字段配方与附件附录完全一致；两变体仅允许身份字段差异，完整记录见 `recipe_diff.json`。
- 全部实际 train/val/test 路径清单与标签哈希等于父数据记录；没有交换或重新划分数据。
- 两份受控初始化保存并立即重载，公共 key/shape/value 逐项相同。
- 实际 `RTDETRTrainer.get_model`、checkpoint `setup_model` 和 `Model.train` 分发的 nc=1 重建。
  分发审计在数据/优化器/训练循环前显式停止，更新数为 0，非正式训练。
- 两配置 27 节点、唯一第17层投影后 PBI、640 输入逐节点张量与父模型输出等价。
- W1/W2 独立 Xavier、Wo 零初始化、CPU/CUDA RNG 隔离、三权重无共享 storage、两变体新增权重相同。
- CPU/CUDA 的非零显式公式及输入/三个参数梯度；native AMP 与显式 CUDA half 的模块公式。
- 原生 optimizer_fp32_v1 保存入口的差分检查：原 FP32 moments 被完整保留，活动状态不变。
- 各 Python CLI help、Python 语法、三个 shell 语法与 help；本机真实 Conda 激活后恢复 nounset。
- corrected_sorted_conf_mask_v1 排序掩码回归、同分整组阈值、可实现 recall 和空集合诊断逻辑。
- 19 项早期拒绝门禁夹具、完整字段兼容夹具和 5 项嵌套原始证据破坏拒绝检查通过。
  完整兼容夹具的服务器容量/环境是显式合成值，标记 `TEST_FIXTURE_NOT_PERMISSION`，不构成启动许可。
- 实际轻量打包烟测通过：369 个成员逐项回读验证，4,491,527 bytes（约4.28 MiB），
  完整列出尚不存在的正式 args/results、训练状态及 val/test。该包是提交前工具烟测，
  最终服务器包由用户后续运行 pack 生成。

参数计数口径为 **nc=1**：

| 配置 | 未融合父模型 | 未融合 PBI | 融合后 PBI | PBI 增量 |
|---|---:|---:|---:|---:|
| 原 C2＋PBI | 20,082,772 | 20,107,348 | 19,902,292 | 24,576 |
| 原 CBR＋LIF＋PBI | 20,149,765 | 20,174,341 | 19,969,541 | 24,576 |

PBI 单模块理论成本：0.3145728 GFLOPs 投影 + 0.0018432 G 逐元素操作。
原生 Trainer 的自动日志仍可能显示通用 hook 统计（如 58.9 GFLOPs），
该数字没有通过 functional Wo 完整计数审计，**不作为本实验整网性能结论**。

## 真实检测 loss 与非零生命周期

使用当前实际 train 的固定两张图、有效 GT/DN，B2/160 有限工程烟测。
它使用原检测 loss、原生 AdamW/scaler/clip/step/EMA；固定学习率的烟测
不替代原在线增强、梯度累计与 B16/640 正式容量预检。

| 模式 | 观察 batch | 有效更新 | A：保存/恢复硬检查 | B：独立反向 | 非零融合/half |
|---|---:|---:|---|---|---|
| CPU FP32 | 3 | 3 | PASSED | PASSED | FP32 PASSED；不要求 CPU half |
| CUDA FP32 | 3 | 3 | PASSED | PRECISION_NOTE | strict FP32 PASSED；half PRECISION_NOTE |
| CUDA native AMP | 7 | 3 | PASSED | PRECISION_NOTE | strict FP32 PASSED；half PRECISION_NOTE |

AMP 初始 scale=65536，发生 4 次原生 overflow 跳步并回退至 4096，之后得到
3 次有效更新。有效更新来自真实 optimizer step 状态和权重变化，不是调用次数。
首个有效更新 Wo 梯度非零，W1/W2 为零；后续全部获得有限非零梯度。

A 包括保存字节 oracle、实际 FP32 optimizer 副本、分组参数名/顺序/超参数、
完整模型/buffers/moments/step/scaler/epoch/EMA/updates、独立 storage、
同梯度原生路径重放、CPU 一步恢复、AMP 有效更新和溢出跳步。全部通过。
非零 Wo 的 state_dict/full-model、实际 Trainer 重建、EMA 自身副本、原生恢复、
融合后分支执行一次和非零残差均有实际证据。一次性 .pt 位于本次 TemporaryDirectory，
只保留必要哈希与报告，未删除用户权重。

## 保留的精度差异

所有 A/B 比较保留 atol=2e-5、rtol=2e-4；不把 raw false 改为 true。
CUDA FP32 父 live 有 1 个公共参数更新超限；PBI live 和 checkpoint 更新的 raw allclose 为 true。
CUDA AMP 父 live／PBI live／checkpoint 的公共模型更新超限张量数为 6／7／8，
梯度超限张量数为 128／139／138。完整名称、max_abs、relative_L2、超限比例均在原始报告。

尤其 AMP PBI live 的 **Wo 梯度 raw_allclose=false**：max_abs=8.647516369819641e-5，
relative_L2=0.001446596535980398，超限比例=0.2166748046875；checkpoint 对照也保留失败值。
W1/W2 梯度满足原阈值。三个 PBI 参数、EMA 参数和 Adam moments 更新均满足原阈值。
所有对照初值、前向连续张量/候选/loss 相同，保存与同梯度恢复完整；公共超限更新在
同名参数的父/候选 live 梯度差异中有对应证据。因此按 A/B 合同单列 PRECISION_NOTE，
不声称独立 CUDA 反向或长期训练轨迹一致，也不声称已确定具体 CUDA kernel 根因。

half 的自然输出和连续路径仍保留原阈值下的原始比较。严格完整 FP32 融合通过；
在实际 half 投影输入上独立重算三个投影/乘法/残差公式，融合前后两份 PBI 都逐元素相等；
三权重不丢失，相同输入的 PBI 输出相等、分支确实非零、输出有限。
half 差异在 PBI 之前的原投影输入已出现，记录为上游 half 融合量化与候选选择诊断说明。
没有采用放宽到 3% 的容差，也没有修改正式 query 选择。

`local_evidence/` 保存完整初始化与数学报告，以及生命周期原文的无损 gzip、原始路径/字节数/
SHA256 清单。运行期间的报告保留当时源码身份；后续门禁/交付脚本修改没有被反填到旧证据。
本机报告从不生成服务器启动许可。原生 resume 日志会打印剩余 200 轮目标，
但这些审计只恢复状态并执行有界步骤，没有启动正式训练循环。

## PENDING / 未启动

- **PENDING**：目标服务器实际环境、Conda 激活、全套重新核验、真实增强 B16/640/native AMP
  最多16 batches／至少2次有效更新的容量、目标 GPU 显存与延时。
  本机容量入口已实际返回 PENDING（0 batches/0 updates、非零退出码），因为环境不匹配；未降 batch。
- **PENDING**：正式训练结果、独立 val/test 指标与曲线、精度收益、父完整预测的可选固定 P 诊断。
- **NOT_STARTED**：主组合正式训练；单模块正式消融。
- **NOT_RUN**：最终 test。仅核对 test 文件身份，没有 test 推理。
- 没有 `passed_gates.sh` 或正式启动许可。服务器 `init-preflight` 和 `start` 必须各自完整核验。

门禁回归使用明确标注的合成夹具验证拒绝路径/字段兼容性，不能充当实际服务器或模型通过证据。
固定 SHA 的同步、预检与用户后续显式 start/resume/val/test/pack 命令在提交后生成的交接文档中。
