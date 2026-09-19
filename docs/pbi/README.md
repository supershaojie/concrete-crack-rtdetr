# PBI-v1：固定 P3 点位双线性交互实验

本分支从成功母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 派生。
主组合 `cbr_lif_pbi_v1` 保留原 CBR 与 LIF-Down；单模块 `pbi_v1`
保留原 C2 下采样和原 Decoder。每份 YAML 仅替换第 17 层类型及其新增参数，
维持 27 节点、原 `.conv/.bn/.act`、节点来源和后续 Concat 顺序。

第 5 层特征 → 第 17 层原 128→256 Conv/BN/Identity → PBI → 原 Concat。
640 输入的 PBI 特征为 `[B,256,80,80]`，PBI 执行一次。

```
U = W1(X), V = W2(X)
Y = X + Wo(U * V)
W1,W2: Conv2d(256,32,1,bias=False)
Wo:    Conv2d(32,256,1,bias=False)
```

无归一化、激活、注意力矩阵、额外门控或空间分支。W1/W2 独立 Xavier uniform
（gain=1），仅 Wo 为零。新增构造使用私有 CPU RNG 区间，两种变体使用同一初值；
不改变公共随机序列或 CUDA RNG。原公共参数均可训练。

W1/W2 沿用外层 native autocast；乘法、functional 输出卷积与残差加法使用局部
FP32，再转回输入 dtype。W1/W2 或末尾转换仍可能产生真实数值风险，不能由局部
FP32 推断无条件有限性。融合后的 `forward_fuse` 继续执行 PBI。

新增参数设计值为 **24,576**。B1/256×80×80 三个投影为 157,286,400 MACs，
按 2 FLOPs/MAC 为 0.3145728 GFLOPs；乘法与残差加法另计 0.0018432 G 操作。
这里不含 dtype 转换和内存成本。Wo 使用 functional conv2d，普通 Conv2d hook
可能漏计，因此不把默认整网 profiler 输出当作本实验完整 GFLOPs。
实测参数和验证结果见 `VALIDATION.md`；参数不能推导延时或显存。

## 来源与配方

`provenance.json` 记录已读取的提示词、真实母版 args、公共源权重及参考包路径/哈希。
参考包实际名称为 `RTDETR-20260623.zip`，SHA256 与用户给出的两种历史名称完全相同。
只读取 `RTDETR-main/ultralytics/nn/backbone/starnet.py`，Block 位于 36–54 行，
未执行参考包脚本、未导入 extra_modules、未复制完整 Block、未安装其依赖。

元素乘法机制已有研究：[Rewrite the Stars](https://arxiv.org/abs/2403.19967)。
PBI 是本项目对 P3 侧向特征的固定低秩适配候选，不与原始 StarNet Block 等价，
不宣称首创乘法交互，也不宣称已获得精度收益。被读取成员没有许可证声明；
本提示词不构成第三方许可证。新实现保留项目 AGPL-3.0 声明。

`parent_args.yaml` 是实际成功母版的完整 109 字段快照，与附件附录逐字段相等。
正式配方仅替换模型/输出身份和经核实等价的数据路径；200e、B16、640、seed42、
AdamW、在线增强、AMP、原损失与 best 选择规则保持原样。`train_pbi.recipe`
输出完整逐字段差异。公共源 SHA256 固定为
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，
不使用已训练母版 best/last 初始化。

`parent_dataset_identity.json` 来自真实成功母版记录，`local_dataset_identity.json`
为本机重新读取全部路径清单和标签得到，两者一致。train/val/test 为
6048/1728/864；GT 为 45573/12840/6663。读取 test 文件身份不执行 test 推理。

## 检查与启动边界

入口：`init_pbi.py`、`check_pbi_math.py`、`check_pbi.py`、
`check_pbi_checkpoint.py`、`preflight_pbi.py`、`train_pbi.py`、
`eval_pbi.py`、`pack_pbi_light.py`；服务器统一入口为 `tools/pbi_server.sh`。
所有完整 CLI 以各工具 `--help` 为准。两个 variant 独立初始化、元数据和 run 名称；
默认只选择主组合，单模块不自动训练。

数学验收使用非零 Wo 检查显式公式和梯度；真实检测 loss 验收覆盖 DN。
首个有效更新中 Wo 非零梯度、W1/W2 零梯度符合零起点合同；Wo 更新后要求
W1/W2 有限非零梯度。临时 hook 在 deepcopy/保存前移除，finally 再清理。

保存策略为 `optimizer_fp32_v1`：仅保存的独立 AdamW moments 保持 FP32；
活动优化器、native scaler/unscale/clip/step、EMA、epoch 和 best/last 选择不变。
EMA 仍按原生 half 保存。保存/恢复以保存字节为参考，不将历史 FP16 moments
转 float 冒充原精度恢复，不宣称历史或长期续训轨迹完全等价。

A（必须通过）检查完整保存/恢复、参数名与 optimizer 分组顺序/超参数、moments、
step/scaler/EMA/updates、无可变 storage 共享、同梯度原生重放和 AMP 跳步。
B 单独记录同输入/RNG/精度的父/候选独立 CUDA 反向重复性，保留
atol=2e-5、rtol=2e-4、原始 allclose 和失败名单。只有完整证据支持有限的反向
差异时才可报告 PRECISION_NOTE，不能把缺证据、前向不一致或丢状态当作精度说明。

融合诊断保留连续路径、候选分数和索引。固定候选重放只用于定位，不能改变正式
query 选择或代替正式评估。严格 FP32 诊断局部关闭 TF32，结束后恢复 flags。

服务器有限容量预检要求 Python3.10／torch2.1.2+cu121／RTX4090，真实原增强
B16/640/native AMP、最多 16 batches、至少 2 次有效更新；到预算退出，不接着训练。
记录 scaler 跳步、GT/DN、loss/梯度、实际更新、显存、耗时。GPU 有其他进程不会
导致人为拒绝，实际 OOM 如实失败，禁止自动减 batch 或关闭 AMP。

`init-preflight` 先归档撤销旧许可，全部必需检查通过后原子生成新许可；
`start` 重新核对当前完整 HEAD、源码/配置、variant、源和初值、配方、数据和环境。
文件存在本身不构成通过。正式训练必须由用户之后显式运行 `start`。

## 独立评估与打包

训练后冻结原规则选中的 best.pt 及 SHA，先独立 val，再对同一权重独立 test。
协议 `corrected_sorted_conf_mask_v1`：640/B16/workers0/FP32/conf=.001/iou=.7/
max_det300/augment=False/rect=False/seed42；不另加 NMS 或调 test 阈值。
报告完整精度、十个 IoU AP、图数/GT、速度、曲线和混淆矩阵。

母版历史独立 val mAP50–95 为 0.5245427221919051；test 为
**0.5220090191444802（52.2009%）**。母版 best SHA 为
`24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`。
这些是历史参照，不是本次重测或 PBI 成果。val 可选固定 P 诊断没有父完整预测时
仍为 PENDING；test 不参与阈值、结构或轮次选择。

轻量包小于 20 MiB，核对每个成员大小与哈希，列明 missing/omitted；不含权重、
数据、参考 ZIP、完整逐图预测，不触发训练或 test。

本次交付状态：**正式训练 NOT_STARTED；最终 test NOT_RUN；未登录服务器。**
固定 SHA 的完整服务器命令在提交和推送核实后单独生成，避免文档 HEAD 自引用。
