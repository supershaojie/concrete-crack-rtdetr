# PDS v1：Codex 完整实施提示词

版本：2026-09-24。本文是待实施的实验合同，不是已实现、已预检或已证明涨点的报告。

请完整阅读并执行：在独立分支实现 PDS，完成必要且有界的检查、提交、普通推送，交付真实完整 SHA 和可独立复制的服务器指令。不要只回复计划。本轮由你完成本地实施和交付，不登录服务器、不启动正式长训、不运行最终 test；服务器训练由我执行你交付的命令。

## 1. 目标、实验范围与用户已经确认的含义

唯一正式主实验是：**原 RT-DETR-R18-Lite＋原 LIF-Down＋原 CBR＋PDS 辅助训练，即实验 F。**

PDS = Phase-preserving Dense Supervision，暂译“相位保留的密集辅助监督”。训练时从原模型同一次前向读取 P2 和 Neck P3，以额外检测分支提供框监督；辅助特征和预测不回灌主检测前向。训练后的主模型权重包含 PDS 对学习过程的影响。

用户已明确确认：

- 正式训练中启用 PDS 辅助损失；**每轮 val、最终独立 val、test、普通 predict 均自动关闭辅助分支计算。**
- best.pt 按上述主检测分支的原验证 fitness 选择，不能按辅助头的指标挑权重。
- 实验 F 的指标就是“经 PDS 辅助训练后的 LIF-Down＋CBR 主模型”的指标。
- 用户只需正常训练和执行 val/test，不额外执行一条手动“关闭 PDS”的命令，不再训练一遍母版。
- 从训练 checkpoint 转成部署模型时，保留本轮选定的全部主模型权重；不能重新载入旧母版 best、源初值或随机主模型。
- 目前只做 F 的正式训练准备。实验 E（原基线＋PDS）、普通辅助头和内部机制消融仅记录后续建议，不自动实施整套消融，不启动这些训练。
- LBC 是另一个独立实验，用户最新说明它仍在运行。不得改动、停止或占用它的工作树、输出路径、tmux 名称。PDS 不叠加 LBC、LCD、MDR、RDL/ROR 或其他第三创新。

不承诺涨点，不把重排、辅助检测监督或一对多分配本身宣称为全球首创。模块包只提供参考算子，不能整包替换项目 Ultralytics。

## 2. 固定身份与工作方式

| 项目 | 固定值 |
|---|---|
| 仓库 | https://github.com/supershaojie/concrete-crack-rtdetr.git |
| 源码母版 SHA | a0459d6a652cb702699087c88fa39a3e4c4087ec |
| 母版 YAML | rtdetr-resnet18-lite-cbr-lif-down.yaml |
| 新分支 | exp-rtdetr-r18-lite-pds-v1 |
| 建议本地 worktree | 主仓库同级 Crack_RTDETR-pds_v1，实际从本机位置解析 |
| 服务器主仓库 | /root/autodl-tmp/projects/Crack_RTDETR |
| 服务器实验 worktree | /root/autodl-tmp/projects/Crack_RTDETR-pds_v1 |
| 服务器 Python | /root/miniconda3/envs/rtdetr/bin/python |
| 正式 tmux | pds-v1-training |
| 正式 project | /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series |
| 正式 run name | pds_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug |
| 本实验工程输出 | 本实验 worktree 下 outputs/pds_v1/ |
| Shell 操作入口 | tools/pds_v1.sh |
| 同步入口 | tools/sync_pds_v1.sh FULL_SHA |
| Python 操作入口 | tools/pds_v1.py，内部文件按仓库风格合理拆分 |

先读适用 AGENTS.md，核验仓库身份、remote、HEAD、git status、worktree 和已有同名分支。新分支从上述母版派生；同名任务已存在时核对是否本次工作的续做，不覆盖现有修改。保留用户改动和所有旧结果，不 reset --hard、不 clean、不强推、不切换主工作区，不合并其他候选整条分支。

本任务授权本实验的独立实现、必要修复、提交和普通推送。一般可逆实现选择自行完成；遇到真正的缺失资产或权限，完成可做部分后准确报告，不伪造检查或推送结果。

## 3. 先读取真实源码和环境

从母版读取：完整模型 YAML；nn/tasks.py 的 RTDETRDetectionModel.predict/loss；ResNet Blocks/BasicBlock；原 cbr.py、lif_down.py；models/rtdetr/train.py、val.py；实际 engine/trainer.py 的 AMP、梯度累积、clip、optimizer、EMA、保存和 resume；tools/init_c19_lif_v1.py、train_c19_lif_v1.py 及 docs/c19_lif_v1 中的受控初始化、配方、独立评估策略。

记录解释器、Python、torch、CUDA、GPU、实际 ultralytics.__file__、criterion 路径和 Git SHA。历史服务器是 Python 3.10.13 / torch 2.1.2+cu121 / RTX4090 / 仓库定制 Ultralytics 8.4.21；历史本地为 Windows、RTX2060 6GB、torch 2.7.1+cu118。以本次真实探测为准，不自动升级环境，不使用 torch 2.1 没有的新 API 作为唯一实现。

当前可见的母版路由为：

| 节点 | 含义 | 640 输入时形状 |
|---|---|---|
| 4 | backbone P2 / S2 | B×64×160×160 |
| 5 | backbone P3 / S3 | B×128×80×80 |
| 19 | 最终 top-down Neck P3 | B×256×80×80 |
| 20 | 原 LIFDown | B×256×40×40 |
| 22、25 | 最终 P4、P5 | 原形状 |
| 26 | 原 RTDETRDecoderCBR，输入 [19,22,25] | 原输出契约 |

**PDS 读取第 4 和第 19 层，不能把第 5 层 backbone P3 当作第 19 层 Neck P3。** 原图仍为 27 个节点，原拓扑、公共 state keys、CBR 的 36 点采样及 rho=0.10、3 层 decoder、300 个常规 query 都保持。

LF 换行规范化后的原文件 SHA256：

- lif_down.py：26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
- cbr.py：d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787

记录原始字节哈希与 LF 哈希，保持两模块内容。若实际图或来源不符，说明具体差异，不凭猜测改接点。

模块包历史同内容 ZIP 的 SHA256 为 b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc；文件名曾为 c090a941-2c78-42c5-b566-27ec43639daf.zip、dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip。只读其中 extra_modules/block.py::SPDConv 的四相位切片，以及 utils/tal.py::TaskAlignedAssigner 的思想。包不可见时如实记录，按本文独立实现；不要把 ChatGPT 的临时路径当作 Codex 主机路径。

## 4. PDS v1 辅助头：唯一固定结构

本节及后面的取值是为首轮实验预先固定的实现起点，没有调参或性能验证。实现时不得自行更换成 YOLO Detect/DFL、多尺度辅助头、蒸馏、分割、中心线监督或另一个方法。

### 4.1 输入与相位

设 F2 为第 4 层输出，形状 B×64×2H×2W；F3 为第 19 层输出，B×256×H×W。检查真实比例和通道，不做静默 resize 来掩盖接错。

相位顺序固定为：

~~~text
t=0: (dy,dx)=(0,0), F2[..., 0::2, 0::2]
t=1: (dy,dx)=(1,0), F2[..., 1::2, 0::2]
t=2: (dy,dx)=(0,1), F2[..., 0::2, 1::2]
t=3: (dy,dx)=(1,1), F2[..., 1::2, 1::2]
~~~

每相位为 B×64×H×W。四相位使用同一套头参数，不能分别初始化四个头。不得先平均四相位。

### 4.2 数学前向

r=32；GN 为 GroupNorm(groups=4, channels=32, eps=1e-5, affine=True)。所有四相位共享如下模块：

~~~text
p2_proj = Conv1x1(64 -> 32, bias=False)
p3_proj = Conv1x1(256 -> 32, bias=False)
norm1   = GN(4,32)
dw      = DWConv3x3(32, padding=1, stride=1, bias=False)
norm2   = GN(4,32)
cls     = Conv1x1(32 -> 1, bias=True)
box     = Conv1x1(32 -> 4, bias=True)

S       = p3_proj(F3)
Z_t     = SiLU(norm1(p2_proj(F2_t) + S))
T_t     = SiLU(norm2(dw(Z_t)))
z_t     = cls(T_t)                  # B×1×H×W
braw_t  = box(T_t)                  # B×4×H×W
~~~

可将 B 和 phase 合并为 B×4 一次计算，但 GN 的每个样本必须对应一个图像的一个相位。P3 投影可共享计算后广播，梯度正常累加。GN 不统计跨图像批量均值，无新的 BN running buffers。

将输出散回 fine grid：第 t 相位 (y,x) 对应 (2y+dy,2x+dx)，得到 logits B×1×2H×2W 和 raw_boxes B×4×2H×2W。按这个 fine grid 的 row-major 顺序展平后再做分配。不要误用与上述顺序不匹配的默认 pixel_shuffle 通道顺序。

新增可训练参数固定为：

~~~text
64×32 + 256×32 + 32×3×3 + 2×(32+32) + (32+1) + (4×32+4)
= 10,821
~~~

注册在根节点 pds_head 下，预计 11 个参数 state keys，无新增持久张量 buffers。主模型公共参数不重命名、不移到 wrapper.base 前缀下。

### 4.3 初始化与精度

辅助头构造和初始化使用隔离 RNG，固定新增 seed=424003；退出后恢复外部 RNG。两个投影与 dw 用 PyTorch Conv2d 默认初始化；GN weight=1、bias=0；cls.weight 用 N(0,0.01²)，cls.bias=logit(0.01)；box.weight 用 N(0,0.001²)，box.bias=[0,0,logit(0.1),logit(0.1)]。所有值记录到版本配置。

辅助头、坐标解码、样本分配、IoU/GIoU 和损失归约在局部关闭 autocast 的 FP32 中计算；F2/F3.float() 必须可导，不 detach。若实现用函数式 conv/GN，权重转 float 也要保留 autograd，不重新创建 Parameter。主网络仍使用原生 AMP，不能全局关 AMP/TF32 或改变其 dtype 策略。

### 4.4 辅助框解码

设当前训练画布为 Himg×Wimg，fine grid 为 Hq=2H、Wq=2W。位置 (v,u) 的归一化参照点为 ((u+0.5)/Wq,(v+0.5)/Hq)。这是标签映射约定，不宣称是严格的卷积感受野中心。

对 raw 输出 (tx,ty,tw,th)：

~~~text
cx = (u+0.5)/Wq + 0.1*tx
cy = (v+0.5)/Hq + 0.1*ty
w  = sigmoid(tw)
h  = sigmoid(th)
~~~

用 cxcywh 转 xyxy 计算 IoU/GIoU。中心偏移有符号、没有硬边界，允许参照点在极细 GT 外但预测同一 GT；不使用 ltrb 正距离带来的点必须在框内限制。GT 用画布裁剪后的合法框；预测框不做破坏梯度的画布裁剪。计算交并时分母 epsilon=1e-7，非有限值要定位而不是 nan_to_num 掩盖。

## 5. 标签、正样本分配与负样本

### 5.1 标签来源

只使用当前训练 batch 中，已经经过原 mosaic/mixup/perspective/flip/letterbox 等增强后的 GT，以及实际 img 的尺寸。原主任务 batch 和标签不改；为 PDS 生成独立副本。

核验 batch_idx/cls/bboxes 格式：本任务 nc=1、cls=0；bboxes 为归一化 cxcywh。转换成像素 xyxy，裁到当前画布，非法/非有限/退化框在辅助监督中跳过并计数，不改原 criterion 收到的标签。错误的标签格式、接线缺失必须报错，不能当作合法空目标略过。

候选点、GT 和离散分配均不求梯度；被分配预测的 logits/boxes 保留梯度。

### 5.2 每个 GT 的候选池

用 fine-grid 参照点的像素坐标判断是否在 GT 内，区间左/上包含、右/下不包含。

1. 若框内至少有一个点，以所有框内点作为初始候选。最多保留 128 个：按 fine-grid row-major 索引升序，n>128 时取序号 floor(t*(n-1)/127)，t=0..127；否则全保留。
2. 若一个框内点也没有，候选只取全图离 GT 中心最近的一个参照点；距离定义为 ((x-cx)/max(w,sx))²+((y-cy)/max(h,sy))²，sx=Wimg/Wq、sy=Himg/Hq；并列取较小 fine-grid 索引。GT 不膨胀、不改框。此种辅助参考点允许在框外，单独记录 fallback。
3. 不能从 val/test 搜集位置统计再修改分配，也不能用阈值图、SAM 或颜色规则伪造裂缝 mask。

### 5.3 质量排序及一对多分配

对候选 i 与 GT j，用 detached 的辅助预测计算：

~~~text
a_ij = sigmoid(z_i)^0.5 * (IoU(pred_box_i, GT_j) + 1e-6)^2
K = 4  # 每个 GT 最多四个正样本
~~~

IoU 是普通非负 IoU，不把 GIoU/CIoU 塞进这个公式。同一 GT 按 a_ij 降序排序，完全相等时按 fine-grid 索引升序。全零 IoU 也不能导致 GT 被静默筛掉；epsilon 只用于排序稳定，不改回归目标或回归损失。

同一图像中每个预测点最多归属一个 GT。按如下确定算法处理冲突：

- 将 GT 按候选数升序、框面积升序、该图原标签索引升序排列。
- 共做最多 4 个轮次。每个轮次依次访问上述 GT，为它分配其排序池中尚未被任何 GT 使用的第一个点；没有可用点则本轮跳过。
- 因此每个 GT 最多得到 4 点。候选都被占用的 GT 可能最终 0 点，必须如实计数；不可复制同一个点监督两个冲突框，也不能悄悄扩候选来凑通过。

GT 数、候选覆盖、正样本数、完全未匹配数和 fallback 数都需记录。稠密/重叠 GT 的有限合成用例应检验这套规则。分配按图独立，不能跨图匹配。

只对最多 128 个候选/GT 计算框关系，不构造 B×GT×全 fine-grid 的巨大可导矩阵。允许按 GT 的小循环，禁止逐像素 Python 循环。此分配仅用于 PDS，不替换或更改原 DETR Hungarian/DN 匹配。

### 5.4 辅助分类负样本与忽略区

每个合法 GT 左右扩 max(sx,0.1*w)，上下扩 max(sy,0.1*h)，裁到画布。所有扩展框的并集为排除区。

- 被分配的点为辅助正样本。
- 位于所有排除框之外且不是任何正样本的点，为辅助负样本。
- 其余点忽略辅助分类损失，包括框内没有被选中的位置。

这些是框级检测责任点，不是裂缝像素标签。不把整个 GT 框当像素前景，不把未选中的框内位置强制当背景。排除区外可能包含 padding 或漏标，不宣称像素真值完全可靠，不按 RGB==114 猜 padding。用少量真实 train 增强图核验实际覆盖即可。

## 6. 精确损失和调度

固定 focal gamma=2、alpha=0.25：

~~~text
p = sigmoid(z)
FL_pos(z) = 0.25*(1-p)^2 * softplus(-z)
FL_neg(z) = 0.75*p^2 * softplus(z)
~~~

对图像 b，令 J_b 为分到了正样本的 GT 集合，M_b=|J_b|，每个 GT 的正样本集合 S_j 大小为 n_j；N_b 为上述负样本点集：

~~~text
L_cls_b = [sum_{j in J_b} mean_{i in S_j}(FL_pos(z_i))
           + (1/4)*sum_{i in N_b} FL_neg(z_i)] / max(M_b,1)

L_l1_b  = sum_{j in J_b} mean_{i in S_j}(
             sum_{d in cxcywh} abs(pred_i[d] - gt_j[d])
          ) / max(M_b,1)

L_giou_b = sum_{j in J_b} mean_{i in S_j}(1-GIoU(pred_i,gt_j)) / max(M_b,1)

L_PDS = mean_b(L_cls_b + 5*L_l1_b + 2*L_giou_b)
lambda_pds = 0.25
r(e) = clip((e-5)/15, 0, 1)       # e 是原 Trainer 的零基 epoch
L_total = L0 + lambda_pds*r(e)*L_PDS
~~~

这里 L1 在归一化 cxcywh 上计算，四维取 sum，不能偷偷改成 mean；各 GT 的正项先按 n_j 平均，再按 M_b 平均。负项的 1/4 对应固定 K=4，不能另按 fine-grid 点数平均或重复乘 batch。

M_b=0 时，该图回归项为0；若有负样本，则仍计算该图合法的背景分类项。完全空集合用当前图可导的零标量处理，不能 mean(empty) 得到 NaN，也不能清除以前 micro-batch 累积的梯度。仅有背景分类时，box 参数可没有梯度；不要为通过检查制造假的回归更新。

r=0 时完全跳过辅助特征捕获、头、分配和新增损失，直接执行原母版 loss。e=5 时仍为0，e=6 开始渐增，e>=20 为完整权重。调度随 resume 恢复真实 epoch，不能从0重新启动。

L0 必须是原 RT-DETR criterion 的完整求和，包括原 VFL/L1/GIoU、encoder、decoder auxiliary、DN 等。历史主项为 VFL＋5×L1＋2×GIoU、matcher cost 2/5/2；原通用 args 中的 box/cls/dfl 字段不能被拿来重写 RT-DETR criterion。

在原 L0 求和后、母版已有外层缩放之前，只加一次 PDS 标量。核验真实调用链，不额外乘/除 batch、accumulate 或 DDP world_size。显示用的原三项 loss_items 保持原契约；PDS cls/l1/giou/raw/weighted 和分配统计另记，不能放入会被 criterion 自动 sum 的诊断字典。

## 7. 模型与 Trainer 接入

### 7.1 只在训练 loss 中读取同一次前向

建议在稳定可导入模块定义 PDSDetectionModel 和 PDSTrainer，而非 __main__ 动态类。优先复用并审计母版模型，再注册 pds_head。原 predict 返回类型、输出数量、顺序和主图完全不变。

当 self.training 且 r(e)>0 且处于正式 loss 调用时，用一个训练专用 traversal 在原网络同一次前向显式取得第4、第19层输出和原检测 preds。原 head/DN 只调用一次；将 preds 交给原 loss 算 L0，将两个局部特征用于 PDS。不对原特征做原地写入、detach 或第二次 backbone forward。

不要把 live 特征缓存到永久 hook、module 属性、全局变量或跨 batch 列表，避免 EMA deepcopy/checkpoint 持有计算图。局部 capture 返回后及时释放引用；统计只保留 detached 数值。

eval/predict 路径继承母版行为，不取 PDS 特征、不执行 pds_head、不做分配。验证器传 preds 再调用 loss 时，只计算原 L0，不二次 forward。若真实训练入口会传入预计算 preds，应显式支持同时携带的当前特征；本 v1 固定 compile=false，不能在训练传 preds 时默默跳过 PDS 后仍声称激活。

PDS 梯度应进入 head、P2 路径和 P3 上游的骨干/top-down Neck（包括实际依赖的 AIFI）；PDS 单独的梯度不应直接进入第20层 LIF、后续 PAN 层或第26层 decoder/CBR。原 L0 仍正常训练这些模块。不得将“没有直接 PDS 梯度”宣传成完全没有间接影响。

### 7.2 原生优化流程

保留母版原 AdamW、所有原组 LR/decay、warmup、cosine schedule、nbs/梯度累积、GradScaler、EMA 和梯度裁剪流程。读取原实际 max_norm（当前历史源码为10）并记录，不套用其他实验的阈值。

新增卷积 weight、GN 参数、bias 按原 optimizer 分组规则各加入恰好一次，列清实际所属组和 LR/decay。不引入第二个优化器、特殊 LR 倍率、分支单独 scaler 或 LBC 的分组裁剪改法。本 v1 使用原生的全体参数裁剪规则；辅助头对裁剪范数的影响属于该实验，需记录诊断，不能悄悄改策略。

保持 micro-batch 累积边界。r=0 且整窗口未启用辅助时，head.grad 应为 None，不强制填0而触发 AdamW 状态/decay；单个无监督 micro-batch 不能清除先前累积梯度。原生 AMP overflow 如实记为跳步，不把 backward 或 scaler.step 调用次数当作有效 optimizer 更新。

### 7.3 真实模型重建

必须覆盖 RTDETRTrainer.get_model 对 nc=80→1 的真实重建流程，不能仅给传入 Trainer 之前的临时对象加头。先在原生母版对象上运行原严格类型/拓扑检查，再对训练包装做公共状态与行为审计，不删除原审计来放行。

当前任务只要求单卡原配方；不要引入 DDP/compile 作为新依赖。如果实现未支持这些路径，显式拒绝非合同配置，不能悄悄产生未训练的 auxiliary 参数。

## 8. 统一初始化、配方和数据

### 8.1 初值

从主仓库 weights/rtdetr_r18_lite_imagenet_backbone_init.pt 的统一未训练源构建母版。源 SHA256：

fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e

复用原受控构建及 nc 适配工具。先核验母版公共参数，再加入 PDS；新增 RNG 不改变公共初始化或后续 DN/数据增强随机序列。正式初值不能使用任何已经训练过的母版、LCD/LBC/SDB 或预检 checkpoint。

审计每个公共 parameter/buffer 名称和值与同源控制母版相同；类别适配只允许真实枚举出的对应 keys，不笼统使用 strict=False 吞掉所有差异。历史 nc=1 未融合母版为20,149,765参数、552公共状态项；本设计同口径训练参数预计20,160,586，新增10,821。部署后回原公共模型参数数。计数仅是辅助核验，逐 key 身份不能省。

### 8.2 完整配方

用母版完整 args 解析并锁定全部字段，附录给出已保存的成功组合样本。model/name/save_dir 等身份字段、经核验等价的数据路径允许改为本实验；其余不靠新版本默认值补齐。PDS 参数单独存配置和 checkpoint metadata，不塞入 Ultralytics 不认识的通用 args。

关键固定值：epochs=200、patience=50、batch=16、imgsz=640、nbs=64、seed=42、workers=8、AdamW、lr0=.0005、lrf=.01、momentum=.937、weight_decay=.0001、warmup_epochs=5、warmup_momentum=.8、warmup_bias_lr=.1、cos_lr=true、amp=true、deterministic=true、cache=false、freeze=null、rect=false、compile=false。

增强：hsv_h=.015、hsv_s=.5、hsv_v=.35、degrees=5、translate=.1、scale=.4、shear=1.5、perspective=.0002、flipud=.2、fliplr=.5、mosaic=.8、mixup=.05、close_mosaic=10，其余见原完整配方。

沿用数据配置 /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml 及其真实物理 split。历史 train/val/test 图像数6048/1728/864、框数45573/12840/6663，nc=1。核对清单/指纹，不重划、不搬动 val/test，不重生成数据。已有增强同源图可能跨 split 的历史限制保留记录，不能称已解决。

保持 B16；若母版 Trainer 有 OOM 自动减 batch，PDS 正式入口必须阻止该静默改配方行为并明确失败。不用改 nbs、关 AMP、换优化器或减输入分辨率来把容量检查做成通过。

## 9. 保存、恢复、验证与部署

训练 checkpoint 要保存 pds_head 全部参数、原主模型、EMA、PDS定义/epoch调度和正常 resume 所需的 optimizer/scaler/scheduler/epoch/stopper 等状态；检查原版本实际保存什么并补齐本实验所需字段。EMA 覆盖辅助头。恢复只支持本实验完整的未结束 epoch 边界 checkpoint，不拿无 optimizer 的 deploy.pt 当 resume，不随机补丢失的辅助头后宣称恢复成功。

正常恢复按实际 epoch 连续调度，说明多 worker 数据与原 CUDA 非确定性限制，不承诺跨进程训练逐位完全一致。无需为本任务实现任意 batch 中点的精确重放。

每轮验证使用训练所得主模型/EMA、原指标与 fitness，PDS 调用计数必须为0。训练最佳权重的选择完全不依赖辅助头成绩。

提供自动部署转换，推荐由独立 val/test 入口内部完成：

1. 按母版评估策略从已经训练的 best.pt 选择 EMA 或 model，并记录来源和哈希。不得重新取统一源或旧母版训练权重。
2. 重建标准 RTDETRDetectionModel，严格加载全部公共参数和 buffers，只移除 pds_head.* 及训练包装依赖。原 decoder 的 shapes/anchors/valid_mask 等非持久缓存按真实实现一致处理，避免 FP16 缓存被静默改精度。
3. 保留 names、nc、主 YAML、必要模型元数据，另存 deploy.pt，不覆盖训练 best/last。
4. 从相同来源、相同 dtype/cache 状态，在 eval 下比较包装模型与标准部署模型的原始检测输出；证明没有 head 调用和 GT 依赖。最终推理参数/FLOPs按这一标准模型统计，训练开销另报。

test 只对独立 val 已选定、锁定 SHA256 的权重运行一次；不由 prepare/preflight/start/pack 自动执行。重复调用同一已完成 test 时返回既有结果/身份，不擅自换权重重新测试。原评估阈值、precision、max_det、数据划分和统计口径沿用母版。

F 对 D 的主要比较以 mAP50–95 为主，同时保留 P/R/AP50/AP75。已知母版独立 val约52.4543%、test约52.2009%，只是历史参考，不能把它们写成此次新复测结果，也不能据此预填 PDS 成绩。

## 10. 有限且有针对性的验证

不新建庞大的通用诊断框架；优先复用母版工具，针对本改动完成以下检查。缺本地 CUDA、数据或公共源时将依赖项列为 PENDING，继续可做工作；6GB 本地卡不要求跑服务器 B16/640。

### 10.1 本地检查

1. **相位/坐标**：用唯一编号张量和非方形合法尺寸检查四相位拆分、回填和 fine-grid 索引逐点对应；验证640时160×160监督网格，不能意外退回80×80。
2. **分配/损失独立小例**：包含两个图像、空 GT、极细无内部点、重叠 GT 争用点、不同 n_j、全零 IoU、边界裁剪。手算/独立参考核对候选截取、分配无重复、实例均衡、负样本排除及归一化，证明一个图的标注不影响另一个图的分配。GT框外 fallback 用原GT回归，不能改GT。
3. **同源主模型等价性**：公共 keys和值相同；PDS 构造不消耗外部 RNG；r=0 时原前向/L0/三项日志保持，新增参数不更新。CUDA 本来存在的非确定性不能用“逐位相同参数更新”做假阻断。
4. **真实梯度路由**：在非退化小例分别看附加损失和总损失，确认 head、P2/P3 上游有限有效梯度，LIF/decoder 不存在独立 PDS 损失的直接路径；至少一个实际 optimizer 更新后相关参数确实变化。不可用复制梯度、手工写权重替代真实更新。
5. **生命周期**：真实 Trainer nc=1重建、EMA、保存/独立进程加载、epoch边界 resume、r(e)在5/6/20的连续语义。不能只测 torch.load 成功就宣称 resume 完整。
6. **评估关闭**：为辅助头设置调用计数或临时抛错哨兵，实际每轮 validator、独立 val/test 的共用评估路径和无标签 predict 都应0调用；无需为检查执行完整test数据。前向包装与剥离模型使用同一非零训练状态核对输出。
7. **真实增强覆盖**：最多2个 train mini-batch，至多8张可视化，显示GT、四相位点、分配正点和排除区；输出GT匹配覆盖率、fallback与冲突数。机制覆盖不足必须报告，不能调标签制造通过，不访问test进行调参。

若使用纯 FP32 进行严格推理等价性检查，在隔离副本上局部控制 TF32/autocast 并 finally 恢复原状态。先比较同精度、未融合的相同权重；保留母版 LIF 的特殊融合保护。只有真实需要时才使用已有候选 ID 对齐工具，不以放宽容差/改变生产输出顺序掩盖差异。

### 10.2 服务器一次短检

真实 B16/640、原 AMP、原 AdamW 与累积规则，使用独立模型/optimizer/scaler、独立临时输出。PDS 的受控诊断 epoch设为20，使辅助权重完整激活；调度和累积状态一致设置。最多16个micro-batch、最多900秒；取得2次**有效** optimizer 更新即结束。到微批次或时限边界时，只有已经取得至少1次有效更新，且其余梯度/参数变化检查全部通过，才可判定短检通过；0次必须失败。

为避免历史 LBC 的“原生默认初始 scale 在有限窗口内全跳步”问题，这个隔离容量/梯度诊断副本的 GradScaler init_scale固定为128，其余保持原生机制；报告显式写 diagnostic_only_scale=128。**正式训练初始 scaler、恢复策略保持母版，不继承诊断 scale 或任何诊断权重。** 这个短检不宣称已经完整验证正式初始 scale 的自适应过程。

记录实际shape、训练参数、AMP状态、allocated/reserved峰值、L0/PDS分项、匹配覆盖、公共/辅助梯度有限性和范数、scale前后值、overflow skips、真实optimizer step计数、主模型和head参数变化证据。不能把 scaler 调用、backward 或 epoch编号当作有效更新。

未完成就保留失败报告和原异常，不擅自无限延长、降低正式配方或更改lambda来“修通过”。非有限报告写严格JSON时用null并附明确 nonfinite 标志/字段路径，不用0伪装正常，也不能让 JSON 的 inf 序列化错误覆盖原始异常。

LBC 正在同卡运行不自动构成阻断理由，不停止它；按实际容量做短检，若确实OOM就准确报告资源冲突。只在源码/配置/源权重/数据身份或影响能力的环境发生变化时让预检失效，start 不重复自动跑重预检。

## 11. 避免已发生过的启动问题

- prepare 先读取原 check_amp 的真实资源依赖，确保本工作树需要的 assets/bus.jpg 等图片可读取，以及原检查需要的权重文件在可用位置。可从现有主仓库的已验证副本补齐并记录哈希；不能等正式 tmux 启动后才因缺 bus.jpg 失败，也不能将原 AMP 检查直接跳过或改成恒真。
- 启动前核验导入路径指向本工作树；Shell入口设置绝对Python、PYTHONPATH、PYTHONUNBUFFERED=1、YOLO_AUTOINSTALL=false，并 cd 到本工作树。不要依赖上一个命令块里的临时变量。
- AutoBackend 预热采用真实有效的有限输入；如原代码使用可能含非有限值的 torch.empty，作最小、有出处的修复并记录，不用它推断模型权重NaN。
- 同步必须执行官方实验同步/准备流程，生成合法交付身份；不能只手工 worktree add 后就假设 delivery.json 已存在。
- 不复制其他实验的预检 PASS，不把“已派发”“tmux存在”当作已经训练到epoch。

## 12. CLI、tmux 与交付

操作入口应实现并实际验证其参数，不交付未实现的命令：

| 动作 | 行为 |
|---|---|
| sync FULL_SHA | 获取本实验分支，核对完整SHA，安全建立/前进本实验worktree，生成交付记录 |
| prepare | 环境/资产/源权重/配方/数据核验，生成独立受控初值与配置 |
| preflight | 上述本地可执行检查及服务器一次有界短检，不启动正式长训 |
| start | 读取本次有效预检，独立tmux派发正式F训练 |
| status | 显示会话/真实PID及子进程、运行阶段、日志尾部、epoch进展和退出状态 |
| resume | 从本实验可恢复last恢复，继续同名专用tmux，不作为任意启动失败的替代 |
| val | 评估选定主模型；内部自动剥离/关闭PDS并锁定评估权重 |
| test | 对val锁定权重执行一次最终test，主分支评估 |
| pack | 只打包身份、配置、源码差异、必要报告/日志摘要及指标；默认LIGHT，不包含大权重/数据 |

同步使用普通 fetch/安全 checkout 或 fast-forward，保持主工作区 HEAD 和其他 worktree 不变。固定 FULL_SHA，不使用会漂移的分支名替代交付身份。一次同步流程避免不必要的重复联网 fetch；网络失败报出具体阶段。首次提取同步脚本后也要执行本实验正式同步逻辑。拒绝覆盖本实验已跟踪修改或冲突文件，不 reset/clean。旧元数据归档后再写新记录。

**tmux 必须直接显示训练输出，并同时保存日志。** 不能把 stdout/stderr 全重定向到文件，导致 attach 后是一片空白。可使用带 pipefail 的 tee，正确、立即保存 Python 的 PIPESTATUS/退出码，不能把 tee 成功当训练成功。日志位于 outputs/pds_v1/console_<UTC>.log。根据真实worker进程和训练事件更新状态；实际开始batch处理后打印 PDS_TRAINING_RUNNING。

进程/status至少区分：NOT_STARTED、DISPATCHED、SETTING_UP、RUNNING、COMPLETED（注明实际epoch和是否early stop）、FAILED。原 epochs=200、patience=50，真实早停不能写成COMPLETED_200。保留Python traceback、退出码、失败时间与对应dispatch/log，不由旧exit.json覆盖新运行状态。

同名活跃PDS训练拒绝重复启动；不要因为lbc-v1-training或其他实验存在而拒绝。所有PID操作必须核验完整命令及本实验身份，不能用全局 pkill python。

若正式输出目录只因初始化失败留下args.yaml、尚无任何已完成epoch/权重，要有安全处理办法：核对本实验无活跃进程、无checkpoint、无results数据行后，将该失败目录改名归档，再允许start。保留日志，不删除目录、不覆写已有训练；有有效last则走真正resume。把实际实现的归档重启入口交付给用户，不让其猜测删除目录。

## 13. 收尾交付与研究边界

完成后普通提交和push本实验分支，核对远端真实完整SHA；没有网络/权限时如实报告，不编造远端一致。交付内容：

1. 分支、母版SHA、交付完整SHA、实际变更文件及原LIF/CBR完整性核验。
2. 中文说明：固定数学、位置/相位、参数数、初始化、PDS仅训练、主分支验证/test、部署权重来源和已知限制。
3. 必要检查结果，区分CPU/本地CUDA、服务器PENDING；不把诊断scale128写成正式训练配置。
4. 每块独立可复制的服务器命令：同步、prepare、preflight、start、status、tmux attach、resume、val、test、pack。使用已实现入口和真实SHA，不留模板SHA，不把ChatGPT临时路径写进服务器命令。
5. 明确本轮没有启动正式训练，没有PDS性能结论。用户自己执行start，预检失败时明确原因。

只读参考：

- Co-DETR，ICCV 2023：https://arxiv.org/abs/2211.12860 。借鉴训练期辅助密集监督；本实现不添加其定制decoder正query。
- YOLOv9，ECCV 2024：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04462.pdf 。借鉴辅助信息路径帮助训练的思想，不移植完整PGI/GELAN。
- TOOD，ICCV 2021：https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html 。借鉴分类/定位质量共同参与分配，本合同的候选、冲突处理及损失是独立固定定义，不宣称原样复现TAL。

四相位拆分/回填本身是等价重排，不创造新信息；共享预测器的部分计算可能等价于高分辨率的相应卷积操作，必须如实说明，不能用重新命名制造新颖性。待F确实有收益，再考虑E和机制消融。普通P3辅助头只能作为整体监督对照；若单独论证相位机制，还需输入/分辨率/容量匹配的对照。当前不启动这些实验。

本任务的成功交付是实现正确、可恢复、评估口径正确、服务器命令可用；是否超过母版必须等真实训练和独立评估。发现本文具体矛盾时说明证据与最小修正，不悄悄更换方法或塞入额外创新。

## 附录：成功母版完整 args 样本

下面仅用于逐字段核对。旧 model/name/save_dir 是历史身份，必须换成本实验身份；其他值按真实母版核验并保持。PDS新参数另存研究配置。

~~~yaml
task: detect
mode: train
model: /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/weights/c19_lif_v1_controlled_init.pt
data: /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml
epochs: 200
time: null
patience: 50
batch: 16
imgsz: 640
save: true
save_period: -1
cache: false
device: '0'
workers: 8
project: /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series
name: c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug
exist_ok: false
pretrained: true
optimizer: AdamW
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: true
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
half: false
dnn: false
plots: true
end2end: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
int8: false
dynamic: false
simplify: true
opset: null
workspace: null
nms: false
lr0: 0.0005
lrf: 0.01
momentum: 0.937
weight_decay: 0.0001
warmup_epochs: 5
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.35
degrees: 5
translate: 0.1
scale: 0.4
shear: 1.5
perspective: 0.0002
flipud: 0.2
fliplr: 0.5
bgr: 0.0
mosaic: 0.8
mixup: 0.05
cutmix: 0
copy_paste: 0
copy_paste_mode: flip
auto_augment: null
erasing: 0
cfg: null
tracker: botsort.yaml
save_dir: /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug
~~~
