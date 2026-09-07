# ACR-SA：C22 / C23 实现说明

ACR-SA（Axis-wise Coverage Relation Self-Attention，轴向覆盖关系自注意力）替换三层 Decoder 的 query self-attention。C22 为 C2＋ACR-SA，先开展正式实验；C23 为 C17 原 CSCEFv51＋ACR-SA，先完成构建、初始化和 smoke，等待 C22 完整 val 支持独立收益后再决定训练。

本次没有启动正式训练，没有上传权重、原始图片或数据集。实现从 C17 提交 `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139` 创建独立分支 `codex/acr-query-attention`。C2 来源为 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。现有 C2/C17/C19/C20/C21 和其他实验分支、运行结果保留。

## 接入与公式

实现文件 `ultralytics-main/ultralytics/nn/modules/acr.py` 中的 `CoverageRelationSelfAttention` 继承原生 `nn.MultiheadAttention`，保留 `self_attn.in_proj_weight`、`in_proj_bias`、`out_proj.weight/bias`。`RTDETRDecoderACR` 直接在原构造位置选择 ACR，不先构建再重建整套 decoder。可选 `acr=False` 追加在原接口末尾；旧模型默认行为保持。

执行顺序仍为 self-attention、残差/norm1、原 MSDeformAttn、残差/norm2、FFN/残差/norm3。Q/K 仍来自 query＋query_pos，V 仍来自 query 内容；新增向量不加入 query_pos。基线已经有参考框位置编码，新增的是候选两两关系。

`DeformableTransformerDecoder.forward` 先对输入参考框 logits 做 sigmoid，然后将当前层 `[B,N,4]` 的 cxcywh 传给 ACR；后续层读取当前 refined_bbox。仅新增路径 `refer_bbox.detach().float()`，宽高下限 `1e-4`；原 query_pos、采样、框更新的梯度不改变。角点不裁剪到 `[0,1]`。

定义 `l=c-s/2, r=c+s/2`，每轴 `overlap=max(0,min(r_i,r_j)-max(l_i,l_j))`，`gap=max(0,max(l_i,l_j)-min(r_i,r_j))`。固定十维顺序如下：

| 索引 | 描述 | clamp |
|---|---|---|
| 0 | asinh((cx_j-cx_i)/(0.5*(w_i+w_j))) | [-4,4] |
| 1 | asinh((cy_j-cy_i)/(0.5*(h_i+h_j))) | [-4,4] |
| 2 | log(w_i/w_j) | [-4,4] |
| 3 | log(h_i/h_j) | [-4,4] |
| 4 | log1p(gap_x/min(w_i,w_j)) | [0,4] |
| 5 | log1p(gap_y/min(h_i,h_j)) | [0,4] |
| 6,7 | overlap_x/w_i, overlap_y/h_i | [0,1] |
| 8,9 | overlap_x/w_j, overlap_y/h_j | [0,1] |

长宽 0.8/0.2 的包含例子，纵向完全一致时，长→短覆盖项为 `[.25,1,1,1]`，反向为 `[1,1,.25,1]`。这是轴对齐包围框关系，不能解释为真实裂缝走向、拓扑或连通性；包含/相邻也不等于同一目标。信息来自当前框，是显式归纳偏置，没有新增观测。平移和统一尺度不变性有宽高下限及截断的适用范围。

每层独立：`Linear(10,32) → SiLU → Linear(32,H)`，`R=0.5*tanh(...)`；转为 `[B,H,Qr,Qr]`，自身对角置零，保留方向性。权重为 `softmax(QKᵀ/sqrt(dh)+R_full+M)`，再使用原 MHA attention dropout、V 聚合及输出投影。训练和推理均启用相同关系公式，无 NMS、删框、合框、坐标后修正或评分阈值变更。

## DN、dtype 和初始化

常规 query 数取 head 的真实 `num_queries`。原 `_get_decoder_input` 将 DN 放在前缀，代码同时检查 `dn_num_split`、DN 输入长度、真实 embedding/reference 前缀。ACR 仅为常规后缀计算 `[B,Qr,Qr,10]` 和 32 维隐藏特征，然后将偏置补到总 N；其他三个 DN 区域新增偏置严格为零。regular→DN 必须仍为禁止连接。

bool mask 的 True 转为 `-inf`；浮点 mask 保留加性数值并转换到 attention dtype。支持 `[N,N]`、`[B,N,N]`、原生 `[B*H,N,N]`、`[B,H,N,N]` 及 B/H 广播，按 B 再 H 展平；padding mask 合并后仍生效。合法 `-inf` 不属于数值失败。

几何运算显式关闭局部 autocast，采用 FP32；几何描述进入 Linear 前转为参数 dtype，因此兼容 AMP 和真正 `model.half()`。正式 AMP 仍沿用 C2；prepare 恢复并核验官方 `bus.jpg`，确认原生 AMP helper 的日志确实是 passed，skipped 不可作为正式启动依据。

`acr_geometry_proj.weight` 为 Xavier uniform，bias 为零；`acr_head.weight/bias` 全零。额外随机初始化在 `fork_rng` 内进行，恢复外部 CPU 和已初始化 CUDA RNG。三层 deepcopy 参数独立。初始 R=0，但仍实际计算整个分支，第一步输出头得到梯度，更新后上游几何投影得到梯度。

每层新增 `10*32+32+32*8+8=616`，三层新增 **1,848**。这是相对各自基线的增量；C23 另外继承 C17 已有 26,912 个 CSCEF 参数，相对 C2 总增量为 28,760。

统一源权重 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。初始化脚本校验干净训练状态、全部 533 个 C2 状态、形状和逐键值，完整字典 strict load；C23 使用 C17 原层偏移映射，并保留其五个参数张量和两个 Scharr buffer。两组均不读取训练后 C17 best。存储 FP32，保留原 C2 half 数值的精确提升。nc=80→1 的九个分类状态、实际 train API 重建、optimizer/EMA 在审计中单独核对。

## 配置与训练口径

| 编号 | YAML（位于 `ultralytics-main/ultralytics/cfg/models/rt-detr/`） | Decoder 输入 | 输出名 |
|---|---|---|---|
| C22 | rtdetr-resnet18-lite-acr.yaml | [19,22,25] | c22_rtdetr_r18_lite_acr_e200_b16_onlineaug |
| C23 | rtdetr-resnet18-lite-cscef-v51-acr.yaml | [20,23,26] | c23_rtdetr_r18_lite_cscef_v51_acr_e200_b16_onlineaug |

C23 在 layer18 保持 `CSCEFv51([17,16], [])`，拼接 `[16,18]`、原输入顺序、算法、构造参数、buffer、初始化均保持；没有 CBR，没有冻结 CSCEF。CSCEF 做跨尺度融合，ACR 做候选交互；二者由共享损失联合训练，可能相互影响，组合收益需实验检验。

权威配方为服务器原 C2 `runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`。本地读取的归档副本 SHA256 为 `ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`，与提交的 109 字段 fixture 完全一致。prepare 必须读服务器原文件，拒绝缺项、类型/值变化，不能用 fixture 代替服务器原文件。

正式配方仅允许 `model/name/save_dir` 三字段变化；`project`、数据划分、200 epochs、640、batch16、seed42、workers8、AdamW、lr0=.0005、weight_decay=.0001、warmup5、cos_lr、AMP 与全部在线增强保持。完整差异表在 `docs/acr_c2_109_field_comparison.json`，服务器 prepare 重新生成实际路径的 `launch_plan.json`/`train_args.yaml`。loss、query300、decoder3、采样点4、评估阈值均不因模块而调整。

## 诊断、评估和研究边界

正式训练默认关闭额外关系统计（`acr_stats_interval=0`），不会调用 `_record_stats`，不计算分位数或额外的注意力熵。原来的每 200 次调用自动开启行为已取消。完整 prepare 后的普通 start 可显式设置 `ACR_STATS_INTERVAL=200` 按需记录；`start-direct c22` 固定关闭统计。开启时仍按原方法记录常规非对角偏置分位数、`abs(bias)>.45` 比例和允许位置的注意力熵；诊断熵从 FP32 QK 的 dropout 前 softmax 重算，只保存标量，不参与模型输出，也不逐批保存矩阵。独立 val 工具保持原有按需诊断流程。固定 val 顺序前 32 张保存预测框、分数、GT、文件 hash，不保存图片。重复候选诊断定义为：score≥.25、同类、预测对 IoU≥.7、两者各自最佳同类 GT IoU≥.5 且关联同一 GT，计无序对；所有重叠框不自动视为错误。

公共原 RTDETRValidator 存在“排序后的 pred 用排序前 score mask”问题。原文件保持以保留训练/历史口径；`tools/acr_results.py` 默认 `--policy corrected`，排序后用对应分数过滤，记录 `historical_mask_issue_triggered/postprocess_affected_images`。同一入口支持 C2、C17、C22、C23，支持 `--policy historical` 另目录复现历史规则。**不能将 corrected 新模型指标直接与历史旧指标对比**；必须以同工具、同 settings 对 C2/C17 重评。完整 val mAP50–95 为主指标，另报 AP75、Precision、Recall、速度，资源取审计和实际运行记录。训练 CSV/最佳权重选择仍沿用 C2 原训练验证口径；该局限必须与独立 corrected val 一起解释。

C23 start 需要明确设置 `ACR_COMBINATION_REVIEWED=1`，提供同口径 C2/C17/C22 完整 val；要求 C22 mAP50–95 高于 C2 才允许排期。这是排期条件，不能证明统计显著性。组合完成后应与同口径 C17 和 C22 比较。结构和选择固定后手动 test；不使用 test 调关系上限/版本，不自动做大批内部消融。

参考关系先验的研究边界如下，均实际查看 arXiv 摘要页，不声称已复现论文代码：

- [RT-DETR](https://arxiv.org/abs/2304.08069)：原检测架构与 decoder 背景；具体接入以当前源码为准。
- [Relation-DETR](https://arxiv.org/abs/2407.11699)：已研究显式位置关系先验作为注意力偏置。不能凭 ACR 名称宣称这一整体思想首次提出。
- [CrossFormer](https://arxiv.org/abs/2108.00154)：动态位置偏置背景。
- [Dual-R-DETR](https://arxiv.org/abs/2512.13876)：预印本摘要讨论候选竞争、成对 routing 和训练期偏置；ACR 的推理期也启用关系公式。
- [MDS-DETR](https://arxiv.org/abs/2605.23507)：预印本摘要讨论基于置信度的非对称 self-attention mask 与重复候选抑制；本实现没有引入该方法的 mask 或监督方案。

模块包 ZIP 的实际路径、SHA、成员一致性及实际读取位置见 `docs/acr_source_trace.json`。包内 DPB 处理特征网格相对位置，本次仅参考“几何描述→逐头偏置”方式；十维候选框轴向覆盖设计按本次请求独立实现。

验证细节和局限见 [ACR_LOCAL_VALIDATION.md](ACR_LOCAL_VALIDATION.md)，服务器步骤见 [ACR_AUTODL.md](ACR_AUTODL.md)。
