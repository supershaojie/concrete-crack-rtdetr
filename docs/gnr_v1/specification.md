# GNR v1：Codex 完整实施提示词

> 使用方式：把本文件全文发给负责代码仓库的 Codex。本文件已经包含方案、母版锚点、实施边界和交付要求。
> 本次工作是实现一个待验证的损失函数候选，并交付可执行的服务器入口。没有证据保证涨点；不要把工程检查、数学检查写成真实训练结果。

> 2026-09-29 参数勘误：根据用户提供的 Codex 核验截图，源码、实际模型 criterion 和 CPU autograd 确认母版实际使用 alpha=0.25、gamma=1.5。先前把参考模块包的 alpha=0.75、gamma=2 写成母版固定值，是本文档错误。保留母版损失参数，让 GNR 从实际生效的最终普通 query VFL 中读取参数并计算对应负梯度。本条覆盖旧方案/旧提示词中冲突的参数要求；GNR 其余公式、beta=0.5、渐入及全部母版保护要求不变。
>
> 若你已因该差异停在参数契约核验，现明确授权按此勘误继续现有 GNR 分支的完整实施，无需为这项已解决差异再次请求确认。当前核验提交不是完整 GNR 功能交付；继续完成接入、必要测试、脚本、提交推送及对应服务器命令。已有核验证据可复用，报告需清楚指向实际生效的 VFL 调用链。

请直接完成下面已经授权的代码实现、必要验证、文档、普通 Git 提交及推送，并给我与你最终提交完全对应的服务器命令。不要只做分析或停在计划，不要让我重复确认已经明确的常规步骤。不要在开发端自动发起 200 轮训练，不自动跑多种损失、消融矩阵或超参数搜索。正式训练由我在服务器执行你交付的启动命令。

如果开发环境缺少服务器数据、母版 best 或 CUDA，完成所有不依赖这些资源的工作，把尚未执行的服务器检查明确标为 PENDING，并提供完成它们的命令。不能冒充在 AutoDL 上运行过，也不要因为缺少可选诊断资源就放弃代码和脚本交付。只有涉及真实的算法契约冲突、缺失且无法恢复的必要来源，才报告具体阻塞。

## 一、实验目标与不可变边界

实验名称：GNR v1。
英文工作名：GT-conditioned Negative-gradient Redundancy Reweighting。
中文工作名：GT 条件负梯度冗余校正损失。

唯一实验组合：

**RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + GNR。**

研究假设：同一 GT 周围存在多个特征与框都相近的未匹配普通 query；它们可能在共享的最后线性分类头上提供重复背景梯度，妨碍仍低于质量目标的匹配正 query。GNR 只有限减弱这部分重复负监督，并完整保留组内最强负项。

这不是已经证实的母版瓶颈。母版独立 FP32 val mAP50–95 为 52.454272%，test 为 52.200902%。GEO、QCC 已完成的 best 正式 test 截图约为 48.5%、48.4%，用户已决定不下载；不要重跑。ARG v1 test 52.2180%，基本持平。DTR 目前没有可核实的正式结果，不可将其称为失败，不能修改或停止它。其它未恢复最终成绩的方案也不能编造结果。

必须保持：

1. 原 LIF-Down、CBR 的结构、初始化、修框公式、参数与推理路径。
2. 原 Hungarian 匹配与代价、正负标签、query 数量、训练增强和数据划分。
3. 原所有回归损失、encoder 损失、中间 decoder auxiliary 损失和全部 DN 损失。
4. 原匹配正样本的 VFL 质量目标与分类项。
5. 原优化器、AMP、梯度累积、学习率、EMA、检查点与早停语义。

唯一算法改动是：**最后 decoder 层普通 query 分类损失中，部分未匹配负项乘以 detached 的 GNR 权重。** 使用 CBR 修正后的最终框建立关系。不能将最终匹配传播给辅助层或 DN。

禁止引入：新检测头、新训练参数、额外教师、第二次模型 forward、额外回归/排序/margin 损失、多对一正样本、DN 参照、查询筛选、推理 NMS、动态类别统计或 EMA 权重统计。不得混入 GEO/QCC/ARG/DTR/RMD/RDL/ROR/BMC/CQS 等算法代码。

新增可训练参数为 0。正常无标签推理不运行 GNR，不需要 GT，不增加部署计算。关闭 GNR 时必须恢复母版原路径。

## 二、先核对真实母版，建立独立分支

母版记录的完整提交：

```text
a0459d6a652cb702699087c88fa39a3e4c4087ec
```

模型 YAML：

```text
rtdetr-resnet18-lite-cbr-lif-down.yaml
```

服务器约定：

```text
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-gnr_v1
PY=/root/miniconda3/envs/rtdetr/bin/python
BRANCH=exp-rtdetr-r18-lite-gnr-v1
RUN=/root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/runs/c_series/gnr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug
OUT=/root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/outputs/gnr_v1
```

这是服务器目录约定，不代表你的当前开发容器已经存在这些目录。请根据实际仓库位置在独立分支/worktree 开发，最终服务器使用上述固定目录。若同名 worktree 已存在，先检查其分支、HEAD、脏文件与正在运行的进程，安全复用兼容状态；不要覆盖未知内容。

先读取适用的 AGENTS.md，确认母版提交确实存在，查看该提交下实际的 forward、最后分类 Linear、CBR 输出、普通/DN 拆分、匹配与 VFL 归约。不能从一个失败实验分支继续叠加，也不能整体导入通用模块包。

参考附件 RTDETR-20260623.zip 的 SHA256：

```text
b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc
```

附件只是一套通用模块实现集合；其源码检查提示 VFL 的负项调制因子参与反传、最后分类头是 Linear，但**实际母版才是实现权威**。拿不到附件不妨碍核对真实母版。不要因附件索引里有某个热门 loss 就替换本方案。

允许参考已有实验中独立的训练、评估和打包基础设施修复，但须逐项说明来源并审查差异，不能把旧算法、GPU 独占检查或错误路径一并复制。不能切换、清理或重置用户正在工作的主目录，不能 force push，不能删除旧实验。

## 三、GNR 数学定义：按此实现，不自行改公式

以下只支持当前单类别 nc=1。真实母版若不是这个条件，报告不兼容，不能悄悄推广。

### 3.1 同一次 forward 的数据

对每张图：

- M：母版最后普通 query 的原 Hungarian 匹配；U：未匹配的普通 query。
- g_j：GT 框；i_j：原匹配中与 g_j 对应的 query。
- b_i：CBR 后最终框；z_i：最终分类 logit；p_i=sigmoid(z_i)。
- h_i：产生 z_i 的最后分类 Linear 的**真实输入特征**。
- v_i=concat(h_i, 1)：将 Linear bias 的梯度纳入分析。
- q_j：母版最终 VFL 已使用的 detached IoU 质量目标，等于 IoU(b_i_j,g_j) 的原实现值。

必须先核实真实分类头有该 Linear 与 bias，核实 z_i 与该层输入、参数一致。不要把 CBR 采样特征、其它 decoder 层的输出、分类权重、logit 或代理特征当成 h_i。若母版实际结构与该假设不一致，不要构造虚假的 v_i。

复用**同一次原始匹配**的索引和原质量目标，不二次匹配、不重新计算一个近似 q_j。正确处理 batch 内 GT 展平偏移、普通/DN 分界与 query 原始索引。如果某个 GT 确实没有原正匹配，不杜撰 i_j；归属于它的负 query 保持权重 1，并记录计数。

sg 表示停止梯度。所有权重计算使用 detached 的框、logit、特征与关系，在局部 no_grad + FP32 下进行。原分类损失使用原来活跃的 logits 与数值路径反传。

### 3.2 GT 归属

仅对 i∈U 计算：

```text
Q_ij = sg(IoU(b_i, g_j))
j(i) = argmax_j Q_ij
```

最大 IoU 为 0 或最大值并列时，a_i=0，该 query 保持母版原损失。

否则：

```text
a_i = Q_i,j(i) - max_{k != j(i)} Q_ik
```

只有一个 GT 时第二大值定义为 0。每个 query 最多进入一个 GT 组 C_j，已匹配到任何 GT 的 query 都不进入负组。无 GT 图像所有权重为 1。

不用额外 IoU 阈值、top-k、框中心范围或伪正标签。并列按所用 FP32 结果的真实相等判断，不用任意 epsilon 增加新的算法阈值。

### 3.3 正 query 欠拟合与局部方向冲突

原正 VFL：

```text
ell_pos_j = q_j * BCEWithLogits(z_i_j, q_j)
d ell_pos_j / d z_i_j = q_j * (p_i_j - q_j)
```

定义：

```text
u_j = sg(relu(q_j - p_i_j) / max(q_j, 1e-8))
C_ik = sg(max(0, cosine(v_i, v_k)))
c_i = a_i * u_j(i) * C_i,i_j(i)
```

q_j=0 时 u_j=0。cosine 先数值 clamp 到 [-1,1]，再取正部；v 包含常数 1，范数非零。u_j、a_i、C_ik、c_i 均在 [0,1]。

p_i_j≥q_j、方向余弦非正或归属不明确时，不削弱对应负监督。这里描述的是共享最后 Linear 上的欧氏梯度关系，不能写成“全网络梯度已无冲突”。

### 3.4 负梯度强度、顺序与冗余

原负 VFL 与其完整 logit 导数：

```text
alpha = alpha_from_actual_final_vfl  # 母版核验值：0.25
gamma = gamma_from_actual_final_vfl  # 母版核验值：1.5
ell_neg_i = alpha * p_i**gamma * softplus(z_i)

d_neg_i = alpha * p_i**gamma * (
    p_i + gamma * (1 - p_i) * softplus(z_i)
)

s_i = sg(d_neg_i * norm(v_i, 2))
```

以上 alpha_from_actual_final_vfl/gamma_from_actual_final_vfl 是数学记号，不是假定仓库里已有的函数或属性。读取最终普通 query 分类损失真正使用的参数来源，包括实际运行时属性、调用参数或对应配置；不能读取一个未使用的 FocalLoss 实例，也不能拿参考模块包或构造函数默认值代替实际执行值。母版核验预期为 alpha=0.25、gamma=1.5，记录实际来源和数值；若又出现差异，应报告具体调用链与证据，不能静默回退到旧数值。

不修改母版损失参数，不在 GNR 内维护另一组独立可调的 alpha/gamma。这里修正的是解析负梯度与原损失的对应关系，通用导数公式保持不变。实际损失中的 p_i**gamma 必须保留原有梯度，绝不能 detach 成常量。不能将 d_neg_i 错写成 alpha*p_i**(gamma+1)。使用稳定 softplus，不用在极端概率下不稳定的 log(1-p)。

在每个 C_j 内按 s_i 降序排序；完全相等时按原普通 query 索引升序。使用真实稳定排序，不给分数添加扰动来制造顺序。k≺i 表示 k 在 i 前面。

```text
R_i = sum over k in C_j(i), k precedes i:
      a_k * sg(IoU(b_i, b_k)) * C_ik
```

R_i≥0，所有量 detached。只能比较同图、同 GT 组内更强或按规则在前的负 query，不能混入匹配正 query、DN、其它图或其它 GT 组。

组内第一项必定 R_i=0，保留原权重。无前驱、框不相交或方向无正相似时 R_i=0。注意使用前驱 a_k，而不是误用 a_i 或遗漏这个因子。

### 3.5 权重、渐入与总损失

首版固定：

```text
beta = 0.5
r(e) = clip((e - 5) / 15, 0, 1)
w_i = 1 - beta * r(e) * c_i * R_i / (1 + R_i)
```

e 是已经完成的 epoch 数，与真实 trainer 的 zero-based epoch 对齐：e=0 到 5 为 0，e=6 为 1/15，e=20 及以后为 1。恢复训练必须使用真实恢复后的 epoch，不能重新开始渐入或根据调用次数更新。

所有负标签仍为 0；0.5≤w_i≤1；未分组或不满足条件的负项、所有正项权重为 1。

```text
L_total = L_mother
        + (g_cls / Z) * sum_{all images, i in U} (w_i - 1) * ell_neg_i
```

这等价于在原最终普通 query 分类逐项损失中，将对应负项替换为 w_i*ell_neg_i。不能保留原负项，再额外加一整份加权负项。

g_cls 与 Z 完全沿用母版的增益、batch 归约及分布式语义。参考包 nc=1 情形的 Z 为 max(最终匹配数,1)，但必须以真实母版代码为准。禁止改成权重和、负项数量、每图分别平均或另一个分布式归约。

差分项可能为负，这是减少原负项的数值；总目标仍由非负的原始逐项损失及其非负权重组成。不能因差分为负就 clamp 成 0，也不能把其作为独立 loss 又重复加入总和。

beta=0 或 r(e)=0 时，直接使用母版 criterion 原路径，跳过 GNR 特征采集、关系计算及新增随机操作，不能仅算完新公式再乘 0。开启且 r>0 时缺少必要特征/索引，应明确报错，不能静默返回 0 冒充正常启用。

### 3.6 理论解释与边界

固定特征、只对最后 Linear 做普通 SGD 一步时，某负项对正 query logit 的作用为：

```text
delta_z_pos_from_negative_i
    = -learning_rate * w_i * d_neg_i * dot(v_pos, v_i)
```

内积为正时降低 w_i，会减轻这一项的局部抑制。这不是实际 AdamW、整网更新、CBR 精度或 mAP 改善的证明。几何全部 detach 意味着 GNR 权重没有新增直接框梯度，不意味着 backbone、CBR、LIF-Down 不再受分类训练间接影响。

保留最强负项和至少 0.5 权重只能限制干预，不能保证重复误检不增加。禁止在代码注释、报告或最终回答中宣称已保证超过母版、全局创新或一定没有副作用。

## 四、最小侵入的工程接入

1. 明确列出修改的 forward/decoder/criterion 路径以及每个张量的形状、坐标系、普通/DN 索引范围。
2. 最后 Linear 输入通过显式、按需的数据返回通道传给 criterion；可采用与现有接口兼容的训练附加上下文。不要用 forward hook、全局变量、模块上的跨 batch 张量缓存或再次 forward。
3. 正常 inference/export 与 GNR 关闭状态保持母版输出契约；诊断如需 eval 特征，应由明确的内部请求开启，不能让日常推理携带训练状态。
4. 若需要模型/criterion 包装类，使用可导入的顶层实现，保证 Trainer 重建模型、EMA、保存/加载、resume 都能找到它。不要只给临时 Python 对象 monkey-patch 一个字段，随后在 Trainer 构建时丢失。
5. 不重复注册原模型模块，不增加 state_dict 参数键；GNR 不需要持久统计 buffer。配置和公式版本放实验元数据，恢复时与代码身份核对。
6. 将纯 detached 权重计算与原 VFL 接入分开，便于核对数学与真实 loss。按 GT 组使用 GPU 张量运算；避免每 query .item()/CPU 往返、Python 成对循环或整网逐样本 autograd。
7. 计算代价最多与实际普通 query 的组内两两关系对应；核实实际 query 数，参考包通常 300，不能把 DN 或所有层拼进去形成巨大矩阵。无需二阶梯度和额外 backward。
8. 原训练日志主损失名与总和口径保持。GNR 原始/加权负损失、差分、权重统计用独立 detached diagnostics 记录，不能加入会被自动求和的 loss 字典造成双算。
9. 所有坐标运算明确 normalized cxcywh/xyxy 转换，复用母版可靠 IoU 工具。NaN/Inf、索引越界和形状错配要报出可定位错误，不能用 nan_to_num 掩盖实现问题。
10. 不因 GNR 的 FP32 权重计算关闭母版 AMP，也不全局改变 autocast、默认 dtype 或优化器精度。解析梯度是 FP32 数学参考，真实 AMP 路径需实际验证。

## 五、母版初始化、完整配方与数据口径

### 5.1 公共未训练初始化

正式实验必须从以下公共源生成与母版一致的受控初始化，不能从训练好的母版 best 微调：

```text
/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt

SHA256:
fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
```

历史公共源 epoch=-1、nc=80。复用母版 init_c19_lif_v1.py 的受控映射与 CBR/LIF 初始化逻辑，再按原 Trainer 转换到 nc=1。原记录中 nc 转换涉及 9 个分类相关键，必须在真实源码中核实名单、形状和允许差异，不可仅用 strict=False 吞掉其它缺失键。

母版同构状态曾记录 unfused 参数量 20,149,765、fused 19,944,965。以相同构造/融合状态比较，报告实际值和差异；不能把融合差异误判成 GNR 新参数，也不能在预检中融合训练模型改变语义。

记录公共源哈希、受控初始化哈希、结构/参数键摘要、映射差异及原零初始化验证。生成 GNR 工作目录自己的受控初始化文件，不覆盖其它实验的文件。已核验且身份一致的初始化直接复用，不每次运行都重新生成。

### 5.2 完整训练配方

读取并核对：

```text
docs/c19_lif_v1/resolved_formal_config.yaml

/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml
```

若文件在实际仓库中存放位置有变，先按文件名和母版元数据定位，不猜一个默认配置替代。需要完整保留归档配置的全部字段及类型；历史为 109 项，不能只核对下面这张摘录。

| 字段 | 母版要求 |
|---|---|
| epochs / patience | 200 / 50 |
| imgsz / batch / nbs | 640 / 16 / 64 |
| seed / workers / device | 42 / 8 / 0 |
| optimizer | AdamW |
| lr0 / lrf | 0.0005 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0001 |
| warmup_epochs / warmup_momentum / warmup_bias_lr | 5 / 0.8 / 0.1 |
| cos_lr / deterministic / amp | true / true / true |
| cache / freeze / rect | false / null / false |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.5 / 0.35 |
| degrees / translate / scale | 5 / 0.1 / 0.4 |
| shear / perspective | 1.5 / 0.0002 |
| flipud / fliplr | 0.2 / 0.5 |
| mosaic / mixup / close_mosaic | 0.8 / 0.05 / 10 |

不要用早期 150 轮、lr0=0.01、关闭在线增强等旧设置。GNR 配置独立存放，不把未注册参数硬塞进 Ultralytics 配置验证器；通过明确支持的入口传入 criterion，并在归档中完整记录。

只允许经明确白名单解释的实验身份与路径差异，如本实验 model/data/project/name/save_dir、resume 状态；这些字段仍需分别验证目标路径及身份，不能全部跳过检查。

**必须处理已知的母版历史路径漂移：**

```text
归档路径：
/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1/weights/c19_lif_v1_controlled_init.pt

曾实际使用的路径：
/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/weights/c19_lif_v1_controlled_init.pt
```

这两个已知字符串可以在“比较用副本”中做精确别名归一，并记录双方原值、归一原因及其它字段/类型比较结果。不要修改母版 args.yaml 原件，不忽略所有 model 字符串，更不能因此放松训练超参数检查。新实验当前真正加载的公共源和受控初始化仍需独立校验哈希，不要求旧目录必须存在。

### 5.3 数据

当前固定数据记录：

| split | 图片数 | GT 数 |
|---|---:|---:|
| train | 6048 | 45573 |
| val | 1728 | 12840 |
| test | 864 | 6663 |

nc=1，类别 crack。定位原 data YAML，固定原 split 和解析结果，保持图片/标签处理逻辑；计数不符时输出具体差异，不修改数据来凑数。

已有增强来源家族可能跨 split 的情况属于既有评估局限，记录即可；不能在本实验重新划分数据、去重后改变训练集或创建一个无法和母版直接比较的 holdout。

首次 prepare 建立数据/标签/split 指纹与解析快照；若存在已验证的 ARG/母版快照，可以在确认路径、协议、版本与文件元数据对应后复用。没有可信快照才做一次完整核验。之后同一运行复用带身份的快照，关键文件/目录元数据不符时显式使其失效，需要用户主动刷新准备结果；不能默默继续使用陈旧指纹或训练过程中改写快照。

status 和 pack 只能读取已有产物及快照，不能触发模型加载、数据集扫描、重新推理或后台补测。

## 六、必要且有界的验证

不要用大量与实现逐行同构的测试制造“验证充分”的假象。请集中覆盖以下真实风险，记录测试环境与实际结果；没有运行的检查标 PENDING。

### 6.1 数学与索引单元验证

使用 PyTorch 的实际 autograd 及独立构造例子：

1. 使用母版实际 alpha=0.25、gamma=1.5，将解析负 VFL 导数与实际生效 criterion 负项的 autograd 对照，保留 p**gamma 的梯度；不能只与另写的一份同公式实现互相验证。覆盖合理 logit 范围及极端值有限性。已有的源码/criterion/CPU autograd 核验可以复用并补齐接入检查，不能因参数勘误重复整个诊断流程。float64 可用于导数参考，实际权重实现仍按规定 FP32。
2. Linear 的 weight/bias 梯度对应 d_neg*v，确保取得正确 h，并核对 bias 项。
3. 零 GT、零负项、单 GT、单负项、多图多 GT、GT 最大 IoU 并列、完全不相交、q=0、p_pos≥q、非正余弦、完全相同强度等情况。
4. 组内稳定顺序、前驱方向、a_k 因子、普通/DN 分界和 batch GT 偏移均正确；最强负项权重严格为 1，所有权重在 [0.5,1]。
5. detached 权重没有到框、特征或 logits 的新增梯度通路；母版原 logits 的分类梯度仍存在。
6. 在人工给定全部门控、框重合与余弦为 1 的 8 项例子中，满强度权重应为：

```text
[1, 0.75, 2/3, 0.625, 0.6, 7/12, 4/7, 0.5625]
sum = 5.358928571428571
```

这是权重构造例子，不把门控恰为 1 宣称为有限 logit 下真实数据的典型状态，也不把系数和称为实际梯度总量。

7. 原 loss 与“原正项 + 加权原负项”的归约一致，能检出重复相加、错误归一、错误分类增益与错误 log loss 总和。
8. e=0、5、6、20 和恢复后的 e 验证渐入；beta=0/r=0 走原函数路径。

先前方案做过 NumPy 数学检查，但这不是本次 PyTorch、真实 batch、CUDA 或 AMP 已通过的证据。不能照抄旧检查结果作为新测试结果。

### 6.2 母版等价与模型集成

- 在相同权重、输入、RNG、DN 随机状态和模型模式下，比较真实母版与关闭 GNR 的路径。FP32 确定性场景优先要求 loss、匹配、输出和梯度逐项一致；不能只比较总 loss。
- 对固定张量的 criterion 零权重比较应直接复用原路径并达到严格一致。实际硬件或 AMP 的容差若有必要，必须事先说明依据并报告绝对/相对差，不能失败后任意放宽容差。
- 对启用 GNR 的同一 forward，正项、回归、encoder、auxiliary、DN 的数值应保持原值；区别只出现在指定最终普通负分类项。未来迭代中预测会变化，不要求两种训练轨迹长期相同。
- 确认训练、普通推理、EMA 保存/加载和中断恢复接口均有效；checkpoint 中 GNR 配置与 epoch 能被正确恢复，母版参数键不变。
- 至少做一个小规模、临时目录中的保存/恢复验证，不覆盖正式 run；校验 optimizer、scaler、EMA、epoch 和渐入状态，而不是只检查能 load_state_dict。
- 若当前只有 CPU，完成 CPU 可验证项并生成 CUDA/AMP 待执行列表，不伪造 GPU 结果，也不要为此更改正式配方。

以上成对比较允许分别运行参考模型与待测模型；“禁止第二次 forward”针对生产损失为取特征额外重跑模型，不是禁止必要对照测试。

### 6.3 服务器 preflight：确实完成一次有效更新

实现有界 preflight，默认最多 900 秒、最多 16 个真实训练 microbatch，以任一预算先达到为止。读取真实数据与增强，使用母版 B16、640、AMP、AdamW、nbs64 和原梯度累积/GradScaler 逻辑。

将 GNR 的**预检用 epoch 上下文**设为 e=20，使新机制满强度参与；这不代表正式训练已进行 20 轮，也不能把预检状态作为正式初始化。母版训练日程与累积规则不为通过测试而改写。

必须看到至少一次实际 optimizer step 后某个应更新的可训练参数发生有限的非零变化，才可标 TECHNICAL_PASS。记录：

- microbatch 数、实际累积边界与 optimizer step 尝试数；
- scaler 的 scale_before/scale_after、跳过次数和真正更新次数；
- loss/关键梯度/参数有限性及参数变化量；
- GNR 是否得到真实特征、索引与满强度参数；
- 用时、峰值显存和当前代码/初始化/配方/数据身份。

GradScaler 因非有限梯度跳过一次 step，不等于完成一次有效更新。budget 用尽仍无有效更新：exit=2、PENDING，输出原因；不要写 PASS，也不要自动扩大预算、重启试到通过、换低 scaler、关闭 AMP、减 batch 或改图像大小。确定实现错误或真实资源错误须明确分类，不能都包装成 PENDING。

preflight 使用独立临时 run，不修改正式权重、EMA、优化器状态或 RNG 快照。真实 OOM 不得杀其它实验或清理别人的显存；记录资源错误，保留日志，停止这次预检。

## 七、先用已训练母版做有限现象诊断

实现 diagnose 动作。它回答“这个机制是否在母版上实际作用”，不能代替性能实验。

1. 从母版归档及 checkpoints 定位**成功母版 best**，验证来源、SHA256、结构和所选 epoch。必要时允许明确的 --mother-best 路径参数。不要把公共未训练初值、别的实验 best 或 last 自动冒充母版 best。
2. 在固定种子和固定图像清单下，最多检查 64 张 val 图像，及最多 8 个使用原在线增强、B16 的真实 train batch。禁止使用 test 预测/标签做方案诊断或调权重。
3. 默认诊断总时间预算 900 秒。若资源/时间不足，保存已完成样本清单与覆盖情况，标明部分结果；不能把缺项当零，也不能自动反复扫描。
4. val 使用确定性 eval 路径；train 增强检查使用隔离的母版模型实例/进程，真实前向但不做 optimizer step。BN、dropout、DN 和 RNG 的变化不能污染正式训练或原 checkpoint。
5. 显式采集同一次 forward 的 h、CBR 最终框、logits、母版匹配及原质量目标，计算满强度 r=1 下的反事实权重。不要重新匹配一份更容易激活的结果。
6. 输出机器可读 JSON 与简洁 Markdown，分别报告 val 和 train 统计：组大小、a_i、p_pos<q 比例、正负局部方向关系、R_i、w_i 分位数、发生改动的数量、最强项保留检查，以及

```text
removed_negative_strength_fraction =
    sum_i((1 - w_i) * s_i) / sum_i(s_i)
```

分母为 0 时写清“无可计量负梯度强度”，不要输出伪造百分比。区分所有普通负项的分母和分组负项的分母，主指标使用所有普通负项；如报告两种口径，分别命名。

7. 可同时记录负项 loss 减少份额和计算开销，但不能用它们替换真正的梯度强度份额，也不能把上述标量当作整网向量梯度抵消的量。
8. 若完整诊断中全程没有实际干预，或实现退化，应明确不建议长训，start 对已确认的零激活/实现失败状态拒绝启动并解释原因。非零但极弱、样本不足或解释不清的结果给出原始证据，标为 REVIEW/PENDING；不能擅自发明某个百分比阈值宣布“有效创新”。
9. 技术是否通过与机制报告分开保存。诊断完成不代表精度会提高。diagnose 和 preflight 都不能自动串联 start。

没有母版 best 时继续完成实现、静态/单元测试及服务器脚本，报告诊断 PENDING 和准确缺失项。我拿到诊断报告后再决定是否执行你已交付的 start 命令，不需要你提前发起 200 轮。不要把人工再次确认设计成每个可逆准备步骤的阻塞弹窗。

## 八、服务器入口、共享 GPU 与运行恢复

请优先复用可靠的现有基础设施，必要时提供：

```text
tools/gnr_v1.py
tools/sync_gnr_v1.sh
docs/gnr_v1/README.md
docs/gnr_v1/server_commands.md
docs/gnr_v1/resolved_formal_config.yaml
```

命名可在不影响清晰度的情况下适应仓库，但最终文档和命令必须统一，不能写出不存在的文件或尚未实现的 action。

至少支持 prepare、diagnose、preflight、status、start、resume、val、test、finish、pack。--help 能列出真实参数和默认目录。入口职责：

| 动作 | 要求 |
|---|---|
| prepare | 核对代码、配方、数据快照和公共初始化，准备独立实验目录 |
| diagnose | 只做上一节的有界母版现象诊断 |
| preflight | 只做有界真实更新预检，生成绑定身份的结果 |
| status | 只读状态，显示真实进程身份、阶段、最新指标和缺项 |
| start | 校验当前身份和有效技术预检，从公共受控初始化启动正式训练 |
| resume | 只恢复同一未完成 run 的有效完整训练 checkpoint |
| val / test | 对已锁定的同一 val-selected best 做独立完整评估；合格结果可复用 |
| finish | 依次补齐必要正式 val/test、离线统计与完整包；已有成功项复用 |
| pack | 纯离线整理现有产物；不自行补测或重新加载模型 |

### 8.1 不写“GPU 必须空闲”的限制

**我基本总是两个实验共享 GPU0。prepare/diagnose/preflight/start/resume/val/test/finish 全部遵守：**

- GPU 上存在其它进程、利用率不为 0、显存已被其它实验使用，都不是拒绝启动或等待的条件。
- 不写“空 GPU 才能运行”、显存空闲阈值闸门、GPU 进程数量闸门、预约 GPU、全局 GPU 文件锁或等待空卡的循环。
- 不杀其它 Python/CUDA/tmux 任务，不自动占满显存，不修改其它实验的环境。
- 可以只读打印 nvidia-smi 作为遥测；不能把其输出用于上述阻塞。
- 可以锁定**本实验同一 run**以防重复写入，并防止同一输出目录的并发评估；锁不代表独占 GPU。
- 真实 CUDA OOM 必须如实报错、留日志，不能自动改 B16、640、AMP、累积或配方来强行运行。
- 共享显存仍可能因实际需求而 OOM；脚本不保证两项任务一定装得下，也不能声称“已有任务完全不会受影响”。

全仓扫描本次新入口及实际调用链，确认没有继承旧 GPU 独占检查。不能仅修改顶层脚本，下面的 helper 仍在等空卡。

### 8.2 tmux 与结束页面

正式训练放在本实验独立 tmux worker，建议：

```text
训练 worker：gnr-v1-training
结果 viewer：gnr-v1-view
finish worker：gnr-v1-finish
```

用户断开 SSH/网页或关闭本地电脑，服务器任务继续运行。训练与 finish 的日志持续写到本实验目录，日志查看页面独立于工作进程：

- worker 完成后记录真实结束状态并退出；
- viewer 保留最后输出、指标表、best 路径、包路径及 Python/日志管道退出码，页面不自动消失；
- viewer 可以继续停留在交互 shell 或明确的持续日志查看状态，用户能自行离开；
- 不能把“viewer 或 tmux 会话还存在”误判成训练仍在执行；
- 不用影响其它实验的全局 remain-on-exit 设置，不 kill-server，不重建/杀掉别人会话；
- shell 管道必须正确保存 Python 和 tee 的真实退出码，不能只报告最后一个命令的 0。

训练结束时展示训练完成/早停状态和训练记录中的 best val；明确它与后续独立 FP32 val/test 的区别。finish 结束后显著展示正式 test 的最终指标，即使较差也完整显示，供我决定是否下载；不自动传输大包到本地。

### 8.3 身份校验和恢复

run 状态必须绑定实际算法/功能代码身份、配置、数据快照、公共源/受控初始化、模型结构和输出目录。预检结果不能在改了 loss 或配置之后继续冒充有效；文档-only 变动可通过明确的功能代码摘要区别处理。

start 要求有效的 TECHNICAL_PASS；展示对应母版诊断结论和覆盖情况。已确认实现错误或完整诊断零干预时不直接开始长训。缺少诊断的准备工作可以完成，但不要自动把 PENDING 填成“机制有效”。对于有数据但效果较弱的 REVIEW 报告，交付命令和证据，由我是否执行明确的 start 命令作决定；不要添加无依据的自动收益阈值或层层确认交互。

不得把新 start 当 resume：正式新 run 从 e=0 开始、200 轮预算、patience50；预检和诊断权重不能流入正式训练。

resume 仅用于同一身份、尚未完成的 run，恢复 optimizer/scaler/EMA/epoch 及渐入；不可仅加载模型权重再称为续训。optimizer 已 strip 的完成 checkpoint 不能用于真正 resume。若训练已完成而最终验证报错，保留 best，用 finish 修复评估，不重训。

进程活性检查同时核对 PID 是否存在和实际命令、工作目录/run 身份、启动信息；不能只看 pid 文件或 tmux 名字。失败、正常完成、早停、被中断与评估待补分别记录，日志与退出码一致。

锁与状态更新避免竞态，过期锁必须根据真实进程身份判断后处理。不要将同卡其它实验误认为本实验重复进程。status 不进行自动恢复或任何隐含训练/评估。

## 九、一次完整评估、离线分析与打包

### 9.1 固定评估协议

best 由完整 val mAP50–95 选择，test 不参与 checkpoint 或 beta/门控选择。同一 GNR best 的 SHA256 必须贯穿 val、test、统计和打包。

正式独立评估沿用母版协议：

```text
precision = FP32
imgsz = 640
batch = 16
workers = 0
conf = 0.001
iou = 0.7
max_det = 300
augment = false
rect = false
seed = 42
protocol = corrected_sorted_conf_mask_v1
```

这里 FP32 仅指独立评估，正式训练继续 AMP=True。核对该协议在现有 validator 中的实际实现，特别是排序、置信度掩码、预测/GT 索引对齐和 AP 匹配；不能只贴一个协议名字就宣称已修复。不能顺便引入新的 NMS、阈值或匹配口径。

母版参考值：

```text
val mAP50-95 = 52.454272%
test mAP50-95 = 52.200902%
test P = 86.0239%
test R = 83.5359%
test F1 = 84.7617%
test AP50 = 89.1997%
test AP75 = 54.0024%
```

上述 P/R/F1 的比较必须使用母版归档的同一报告口径。若无法核实阈值口径，清楚标注为历史参考，不能强行比较新的固定阈值指标。报告涨跌使用百分点，避免将 0–1 小数、百分数和相对百分比混为一谈。

### 9.2 第一次正式推理就保存后续需要的数据

第一次必要的完整 val 和 test，各做一次网络推理，同时导出：

- 实际参与评估的图片 ID、路径标识、原尺寸、预处理/缩放信息及 GT。
- 所有普通 query 的原索引、原始分数、类别、最终框；在置信度裁剪前保存，包含低于 conf 的 query。核实实际 query 数，通常为 300。
- 明确框坐标系，保存足够信息复原到原图坐标；不能把 DN、encoder 或中间层预测混入。
- 实际指标使用的过滤/排序结果及其与原 query ID 的映射。
- IoU 0.50:0.05:0.95 的 AP、原始正确性/匹配统计、PR 曲线、P/R/F1 随置信度曲线所需数据。
- checkpoint、代码、数据/split、评估配置与协议的身份摘要和实际退出状态。

不要误把“导出所有 query”变成指标改用所有未过滤预测；正式指标仍按固定协议。导出完整数据是为了离线分析，无需再跑一次模型。

如需研究重复预测，可用已保存 query 与 GT 做离线统计，并清楚定义 IoU 与置信度；它是诊断数据，不能替代官方 AP。没有完全同协议的母版逐图预测时，不编造配对比较，也不要自动重跑母版整套评估。

### 9.3 可复用评估与阈值分析

为每个 split 保存成功记录，绑定 best 哈希、配置、数据快照、代码/协议、图片清单、产物完整性和退出码。完全一致且产物齐全时，val/test/finish 直接复用，显示 REUSED 和来源，不重复推理。

成功记录缺失但历史产物完整且身份、哈希与成功状态可独立核实时，可重建索引元数据；不能只因少了一个锁文件就重跑。身份不明或产物确实不完整时，说明具体缺口后只补必要 split，保留旧产物，不伪造 COMPLETE。

离线同时提供两类不同口径：

1. 既有协议的常规 AP、最大 F1 报告及曲线，明确阈值来源。
2. 只在 val 上选定的置信度，再固定用于 test，输出 TP/FP/FN/P/R/F1；计数默认明确标注 IoU=0.5，不能混用 AP 的十阈值均值。

第二类从完整保存的 query/GT 按相同排序、过滤及匹配规则重新做**离线**统计。不要简单截断一份不适用于新阈值的旧 TP 数组，导致匹配错误。阈值只由 val 确定并落盘，test 不再选阈值。常规 test 最大 F1 仅作为注明口径的报告值，不用于配置选择。

已有 test 被多轮实验使用，微小单次提升不能宣称统计显著或证明全新外部泛化。首版固定 beta=0.5，不根据 test 继续搜索门控、渐入或挑 checkpoint。

### 9.4 完整包与最终显示

默认输出：

```text
OUT/GNR_v1_COMPLETE_<timestamp>.tar.gz
```

完整包一次收齐：

- 实际 Git/功能提交身份、差异补丁、GNR 配置及公式版本；
- 母版/实验配方快照、允许差异与历史路径别名审计；
- 公共初始化、受控初始化、结构/参数键和加载映射记录；
- 数据与 split 的指纹/解析快照；
- 数学/集成测试、母版 diagnose、真实 preflight 报告；
- 训练 args、results.csv、日志、best/last 的路径/哈希/epoch 与选择依据；
- 正式 val/test 指标、全部普通 query、GT、曲线、原始匹配统计和阈值分析；
- 运行阶段、退出码、成功复用记录和包内文件清单；
- 对包内每个文件的 SHA256 manifest，并校验归档可读取。

默认不塞入原始图片或 best/last 权重文件本体；权重留在服务器，包中记录其位置与哈希。若用户以后明确要权重，再提供单独选项。完整包不是只带一张 summary 图，也不应因加入原始数据变得巨大。

pack 必须纯离线，不 import 模型触发 GPU，不扫描原始数据，不调用 val/test。缺必需产物时输出 INCOMPLETE 和缺项，不能把不完整包命名为 COMPLETE。未执行的可选环境检查如实标注，不伪造成已执行结果。实际开始正式训练的运行，必须有真实有效预检与应有诊断证据。

finish 的终端最后保留一张清晰表：val/test 的 P、R、F1、AP50、AP75、mAP50–95、与母版主指标的百分点差、best 身份、每个阶段真实退出码、包路径和完整性。显示原始未舍入值或保存其机器可读原值，页面可简洁舍入；不能由截图舍入数反推高精度数值。

## 十、交付、Git 与服务器命令

### 10.1 完成顺序

1. 核对真实母版与适用仓库规则，建立独立分支/worktree。
2. 实现 GNR、必要诊断以及可靠的服务器入口。
3. 完成当前环境可执行的必要测试，明确服务器待执行项。
4. 审查相对母版的差异，确认保护的模型机制和共享 GPU 要求。
5. 普通 commit、push 到实际仓库 remote，并核实远端包含相应提交。若实际网络/权限阻塞，记录真实失败，不编造推送成功。
6. 使用你实际产生并已核实的提交身份，交付完整服务器命令文档。

不要输出虚构 SHA、示意仓库 URL、假定已经存在的远端分支或“把这里换成你自己的提交”这种让我再补的占位命令。可以先产生功能提交并推送，再用其真实 SHA 补齐命令文档做一个 docs-only 提交，避免文档引用自身提交哈希的问题；说明功能 SHA 与文档 SHA 的关系。

服务器同步脚本只使用已确认的 Git remote。fetch 需有界，例如最多 3 次、单次 120 秒；明确网络失败，不无限重试。不要 curl 未固定版本的远程脚本直接执行。同步到你实际交付的功能身份后再 prepare；现有 worktree 脏或正在运行时不能 reset/clean 覆盖，报告具体不兼容状态。

### 10.2 我要的服务器命令文档

请按顺序给出可直接复制的命令，使用最终真实脚本、参数、remote 与提交：

1. 同步到本实验独立 worktree，并确认当前代码身份。
2. prepare。
3. diagnose 与 preflight；说明它们有界且不会自动开始长训。
4. 查看诊断、预检与运行状态。
5. 单独的 start 命令：从公共初始化正式训练。
6. attach 本实验 viewer；断开后重新查看；说明如何离开 viewer 而不结束服务器训练。
7. 中断后的 resume 命令及它适用的状态。
8. 训练结束后 finish 命令：同一 best 的完整 val/test、离线统计与完整打包。
9. 独立 val/test/pack 的重入命令与已有成功产物复用行为。
10. 指标和包位置；不自动下载，我先看结果再决定。

不要把 start 拼在 prepare/preflight 的同一条自动流水线里。不要在命令里要求先清空 GPU、等待其它实验结束、停止 DTR 或更改 CUDA 设备隔离设置。默认 device=0，保留 B16 与母版配方。

### 10.3 最终回复我时

用简洁中文说明：

- 实际修改了什么，GNR 的唯一作用范围及母版保护结果；
- 真实运行过哪些验证，哪些还是 PENDING；不要把 CPU 测试说成 CUDA 已通过；
- 分支、实际功能提交、推送结果、关键文件与服务器命令文档位置；
- 下一步首先运行的完整服务器命令；
- 有无真实阻塞；没有则继续完成全部交付，不再问“是否开始实现”。

代码中记录必要技术细节，服务器页面只显示能帮助我判断运行状态和指标的信息。不要追加无关论文综述、自动新建更多实验、替我下载所有旧包，或把候选命名当作已经获得性能创新。

## 十一、研究来源与表述边界

方案借鉴而非复刻：

- The Equalization Losses，TPAMI 2023：用梯度而非样本数描述监督失衡。https://arxiv.org/html/2210.05566v1
- Bucketed Ranking-based Losses，ECCV 2024：负预测存在可聚合结构；其等价加速结论不能证明本方法涨点。https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7634_ECCV_2024_paper.php
- DEIM，CVPR 2025：定位质量与分类监督关系；本实验不引入 Dense O2O/MAL。https://arxiv.org/html/2412.04234v2

梯度重加权、余弦相似度、停止梯度和背景降权都有先例。GNR 的待验证点是本文件定义的完整组合，不宣称单个部件原创或已证明全球新颖。当前任务以实现这份固定首版为准，不需要为“找新论文”改变公式。

后续若正式结果有明确收益，再考虑一个与整体负项削弱量相近的统一负项降权对照，用于分辨收益是否来自选择性机制。这不属于本次自动启动范围。

**现在直接开始实施，完成后交付实际可运行的代码和与提交一致的服务器命令。**
