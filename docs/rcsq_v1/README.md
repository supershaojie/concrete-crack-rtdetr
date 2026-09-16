# RCS-Q v1

首轮主候选为原 RT-DETR-R18-Lite + original CBR + 原 LIF-Down + RCS-Q，variant 为 `cbr_lif_rcsq_v1`。`rcsq_v1` 是单模块消融准备；本次不训练任何组，不运行最终 test。正式训练状态为 **NOT_STARTED**，最终 test 为 **NOT_RUN**。诊断副本上的优化不属于正式训练更新，不能用于正式初值。

## 代码与配方来源

基点为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。`base_audit.json` 记录 Git 历史、实际已完成 CBR+LIF 结果目录、完整文件 SHA 和 13 个数值源码/入口的 LF 内容比较：归档内容均与该基点相同。历史 CSV 有 200 行且最后为 epoch 200，退出码为 0，最终打包和历史 test 记录均报告此 SHA，历史 test checkpoint SHA 与历史 training best 相同。历史 `training_state.json` 遗留 `training`，因此不把单个状态文件或启动初始化 SHA 当成训练完成证明。本任务没有重新执行历史训练/test，也没有读取其训练权重作为初始化。

`cbr_lif_authoritative_args.yaml` 来自成功组合的真实 `training/args.yaml`，共 109 字段；与已归档 C2 完整参数仅 `model/name/save_dir` 不同。正式计划继承全部字段，限定身份和环境路径差异，并输出逐字段比较。保持 200 epochs、patience 50、640、batch 16、seed 42、AdamW lr0 0.0005、AMP 与原在线增强；不因 OOM 自动改 batch/imgsz/AMP/queries。`c2_data.yaml` 和 `base_audit.json` 保留原数据描述和 train/val/test 路径、标签指纹；不创建新切分。

源初始化只允许 `rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。沿用项目既有 ImageNet 来源初始化；原三层 3×3 stem 并非直接复制 torchvision 7×7 stem。没有 SVD、factors 文件、冻结骨干、蒸馏或额外预训练。

## 结构与数学定义

RCS-Q 为 Region-Centered Support Query（区域对比支持查询初始化）。保留原 R18、AIFI、CCFM、三层 Decoder、300 normal queries、原分类回归头和采样点数。组合 YAML 的 Decoder 输入为 `[19,22,25]`，slot 0 为最终 Neck P3；LIF 保留在 layer 20 的 P3→P4 下采样，CBR 保留 Decoder 后原有 36 点四边内外采样。只有 Decoder 类替换为 RCS-Q 对应子类；新增成员为 `rcsq`，不再包装已有 head。

原 `_get_decoder_input` 先完成 top-k 和 detach，再将其完整 DN+normal `embed/refer_bbox` 显式传入 RCS-Q。采样框为 `sigmoid(refer_bbox.detach().float())`，原 logits 不修改地继续交给 Decoder。分支结果不再次 detach，Neck P3 的采样支路保持可微；训练后会增加到 P3 的梯度路径。

固定 256→64 P3 投影、256 维 query LayerNorm、64 维 Q/K/V、2 heads×32 维和 5×5 bin centers `[-0.4,-0.2,0,0.2,0.4]`。每框 row-major 顺序，最后维度 `(x,y)`；`grid=(2u-1,2v-1)`，双线性 `grid_sample(align_corners=False,padding_mode='border')`。图外点排除，不能先 clamp 后重复采样边界。合法边缘点的插值核使用 border，避免零填充值制造差异。

令 `m` 为有效点掩码，`z` 为投影后采样特征：

```text
mu = sum(m*z) / max(sum(m), 1)
c  = where(m, z-mu, 0)
qn = LayerNorm(q)
L_real = q_proj(qn) · k_proj(c) / sqrt(32)
A = softmax([null_logit(qn), masked(L_real)])
r = sum(A[...,1:] * v_proj(c))
q_new = q + out_proj(concat_heads(r))
```

空槽 index 0 的 value 恒为 0，不新增 query，无效点 logits 为 `-inf`。空槽永远有效；全无效区域保留原 q。真实位置权重不再次归一化到 1。区域均值仅是共有成分对照，不是真实背景；attention 不是裂缝分割，null 权重不是校准置信度，也不能直接恢复未进入 Top-300 的目标。若没有空槽，单独给线性 K 减均值仅改变共同 logit 偏移；待验证机制依赖 **V 中心化的差异证据和空槽竞争**。

新增参数精确为 `16,384+512+16,384+4,096+4,096+514+16,384=58,370`，没有额外 BN、激活、offset、门控、dropout、辅助损失或全局 token。K/V 无 bias，常量区域产生零证据。

## 初始化、DN 与精度

首次新建时，P3/Q/K/V 使用 Xavier uniform，LN weight=1/bias=0，null weight=0/bias=ln(25)，out weight=0 无 bias。out_proj 是唯一关闭贡献的层。初始化使用局部 RNG 隔离和公共状态迁移，最终以逐值比较证明公共参数及 BN buffers 相同；仅设同一 seed 不算证明。两个新 variant 的 RCS-Q 状态一致。

首步只要求 out_proj 的有限非零梯度；内部投影因 out=0 首步梯度为零符合链式法则。诊断副本设置非零 out 或进行少量更新后核验内部投影、null 与 P3 可学习。正式初值不使用这些更新。load/EMA/eval/fuse/resume 不重新清零；加载非零分支必须保持其值。

沿用真实 `get_cdn_group` 的 noisy reference boxes、排列、attention mask 和损失，不读取 DN 正负身份/干净 GT 作为特征。仅以 gt_groups 判定 padding：`D=dn_num_split[0]`、`G=dn_num_group`、`M=D/(2G)`，有效条件 `(k % M)<gt_groups[b]`，再拼接 300 个 True。必须先断言维度和整除关系，且与原生成器实际 scatter 索引独立核对。padding logits=0 仍会 sigmoid 为合法几何框，因此不能靠框值或 embedding 是否全零判断 padding。

小分支局部禁用 autocast，全程 FP32，函数式运算对输入和注册参数使用可微 `.float()`，最后将 delta 转回原 query dtype；forward 不改参数对象。CPU 预检使用 FP32。CUDA 诊断融合时暂时关闭 matmul/cudnn TF32并恢复原设置；正式 AMP 和原 deterministic=True/warn_only 行为保留，不保证 CUDA 逐 bit 确定。

## 参考归属与区别

实际阅读参考包 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA 与规格中的上传文件一致；详见 `reference_audit.json` 的类、函数、文件 SHA 与语义。读取 RTDETRDecoder 的选择/detach、MSDeformAttn 的坐标处理、DySample 的 `(x,y)` 与 align_corners=False/border 约定，不导入另一套 Ultralytics 或整个 extra_modules。

[SACQ 原论文](https://arxiv.org/html/2405.03318v1) 已研究内容查询增强：全局自注意力池化初始化和后续层局部 ROI 内容增强，配合类似预测聚合。RCS-Q 测试初始化时的区域中心化值与零值空槽，保留原 matching，无跨 query 聚合。不宣称首次内容查询增强、首次区域采样、必然涨点或已测得速度提升。

## 检查与交付

初始化和预检 JSON 保存在被忽略的 `outputs/rcsq_v1`，包括代码/配置/源权重/初始化指纹、环境、状态和诊断摘要。本地结果应与 `LOCAL_VALIDATION.md` 一同查看；没有执行的服务器项目始终为 PENDING，不能把 CPU/小图功能检查等同 B16/640 真数据容量通过。复杂度区分未融合/原生融合参数和 THOP 主体算量，grid_sample、归约、softmax 等未覆盖算子须单列。规格参考新增主要算量约 0.3545 GFLOPs（2×MAC，Q=300），并非整网实测延迟。

实际入口与完整按顺序命令见 `SERVER_HANDOFF.md`。提交后执行：

```bash
python tools/make_rcsq_v1_delivery.py --output outputs/rcsq_v1/delivery
```

输出含真实完整 SHA 的交付目录、精简 bundle、校验清单、统一 task.env 及 `SERVER_HANDOFF_DELIVERED.md`。仓库文档使用替换占位符以避免提交自引用 HEAD 循环。不得把权重、数据集、参考 ZIP 或大型输出加入 Git；同步与正式 start 留给用户在服务器执行。
