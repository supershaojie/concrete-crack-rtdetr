# 本机验收记录

母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。测试时 HEAD 仍为母版，
工作区包含本次新实现；没有将旧报告的 HEAD 改写成交付提交。
`tested_tree_identity.json` 保存内容哈希与后续仅 CLI/报告工具变化的说明。
所有数值模块、解析器、模型 YAML 与被测源码一致；初始化非 CLI 函数 AST 一致。

本机 Python 3.9.25、PyTorch 2.7.1+cu118、CUDA 11.8、RTX 2060。
这不代替服务器 PyTorch 2.1.2+cu121、RTX 4090 的实测。

| nc=1 模型 | 未融合参数 | 融合参数 | 相对父模型新增 |
|---|---:|---:|---:|
| 原 CBR＋LIF | 20,149,765 | 19,944,965 | — |
| CBR＋LIF＋GRA | 20,158,509 | 19,953,709 | 8,744 |
| 原 C2 | 20,082,772 | 19,877,716 | — |
| C2＋GRA | 20,091,516 | 19,886,460 | 8,744 |

已实际通过：

- 主组合552、单模块533个公共状态逐项精确相等；9项nc适配父/目标相等。
- 两变体新增5个state tensor逐项相同，8744参数；offset零初始，其余卷积非零。
- 真实27节点和15/16/17输入对象、训练与推理接线；初始共同特征最大误差0。
- 非方形及非整数目标ramp，最大误差≤9.54e-7；零偏移误差0。
  目标像素分母、轴交换和组错配三个故障注入均被拒绝。
- 两变体CPU/CUDA FP32各2次小样本真实检测loss有效更新；CUDA native AMP各6个
  batch内2次有效更新、4次原生scaler回退。使用2张真实train图、160输入，不是B16容量试验。
- 已学offset最大约0.00029–0.00032源像素，全部有限，tanh整体饱和比例0。
- 非零状态保存/重载、EMA自身对照、真实checkpoint→Trainer setup/get_model重建、
  原生optimizer/scaler/epoch/EMA恢复、CUDA half/float同量化比较。
- 两变体真实BaseValidator一次epoch-half路径：2张train图、有效GT与真实loss，
  不聚合验证指标；AutoBackend融合→half、全零warmup与真实图前向有限且GRA继续执行。
- CLI帮助、plan、错误start拒绝、原Validator匹配fixture、固定P同分整组/
  NOT_ACHIEVED、轻量包逐成员哈希核验；shell经`bash -n`检查。

精度说明（不改母版Decoder、不放宽严格容差）：

1. 主组合已学CUDA FP32融合中，连续特征最大误差1.67e-6且严格容差通过，固定候选
   replay通过，但原生top-k变序导致最终行序输出不allclose。因此记录PRECISION_NOTE，
   不声称原生最终输出完全相等；完整验证集上的融合影响仍PENDING。
2. 首次检查失败已原样保留于`local_first_failure.json.gz`。独立诊断确认：仅调用
   native resume状态恢复时，EMA可继承即时模型的shape-only anchor缓存，其中保留AMP
   计算值。缓存与AMP重新生成值完全相等，与FP32值最大差0.00410366。
   学习参数、共同特征、候选索引完全相等；双方诊断性重建FP32缓存后输出误差0。
   EMA自身保存重载及完整实际Trainer重建路径另行通过。完整证据在
   `cache_diagnostic.json.gz`，没有把原失败记录改成通过。

成本中的卷积MAC估计不包含两次grid_sample、坐标及逐元素运算，标记PARTIAL。
服务器原增强B16/640/native AMP的显存、耗时、有效更新与Torch2.1边界仍PENDING；
`gra_server.sh init-preflight`显式执行补检，结束后不会训练。若失败，保留目录，
按各CLI的`--output`指定新目录补检；禁止手工把报告改成通过。

可选固定P诊断已仅使用历史父模型**已有val导出**计算，没有运行新推理：
P目标0.8656，实际可实现阈值≥0.6047021150588989，P=0.8658162441466172，
R=0.835202492211838，TP/FP/FN=10724/1662/2116，GT=12840。
候选模型尚未训练，候选val诊断PENDING；这些数字不是本次GRA精度结果。

正式训练 **NOT_STARTED**，最终test **NOT_RUN**。参考ZIP本机有界查找未发现，
未声称读取其源码；数学定义按用户附件完整合同实现。
