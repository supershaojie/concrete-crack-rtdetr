# RMD v1：Codex 完整执行提示词

版本：2026-09-28。独立实验：RT-DETR-R18-Lite＋原 LIF-Down＋原 CBR＋RMD。

请将本文作为实施任务执行：阅读真实母版源码，完成代码接入、必要且有界的验证、文档、提交和普通推送，最后提供真实完整提交 SHA 与分段服务器命令。不要停在方案介绍或伪代码。本轮不由你启动正式长训；服务器正式训练由我执行你交付的 start 命令，并必须进入独立 tmux。

本实验仅增加/调整一种损失机制，保持已有两项创新的结构与前向公式。候选尚未通过正式训练，不能宣称保证涨点或全球首创。本文已固定首版公式和参数，不自行搜索超参数、替换候选、增添其他模块或扩展成整套消融实验。

## 1. 实验身份与母版

| 项目 | 固定身份 |
|---|---|
| 项目 | RT-DETR 混凝土裂缝检测；本轮不做 YOLO |
| 仓库 | https://github.com/supershaojie/concrete-crack-rtdetr.git |
| 源码母版 | a0459d6a652cb702699087c88fa39a3e4c4087ec |
| 本实验分支 | exp-rtdetr-r18-lite-rmd-v1 |
| 本地 worktree | 主仓库同级 Crack_RTDETR-rmd_v1，按当前机器真实路径解析 |
| 服务器主目录 | /root/autodl-tmp/projects/Crack_RTDETR |
| 服务器 worktree | /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1 |
| Python | /root/miniconda3/envs/rtdetr/bin/python |
| tmux | rmd-v1-training |
| 正式 run 名 | rmd_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug |
| run 父目录 | /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series |
| 工程产物 | 本 worktree 的 outputs/rmd_v1/ |
| 操作入口 | tools/rmd_v1.py，可配 tools/rmd_v1.sh |
| 同步入口 | tools/sync_rmd_v1.sh FULL_SHA |

先检查仓库 origin、当前 HEAD、工作区状态、AGENTS.md、已有分支与 worktree。从上述母版创建独立实验，不从其他实验的 HEAD 派生。保留用户未提交工作，不 reset --hard、clean、强推或覆盖已有 run。同名实验存在时识别是否为本任务续做，不能重新初始化覆盖训练。不要操作另一实验的分支、进程、tmux 或权重。

原有两个模块必须同时保留。禁止把实验做成原始 base＋RMD；也禁止附带 RDL、ROR、LBC、LCD、PDS、CQS、BMC、PEQ、TCR 等旧第三方案。

实施前读取母版中的模型 YAML、nn/tasks.py、nn/modules/cbr.py、lif_down.py、RTDETRDecoder 及 decoder 本体、models/utils/loss.py 与 ops.py、utils/loss.py、RT-DETR Trainer/Validator、optimizer/AMP/clip/EMA/checkpoint/AutoBackend warmup 路径，以及 tools/init_c19_lif_v1.py、train_c19_lif_v1.py 和 docs/c19_lif_v1/。以真实源码为准，不以最新公开 Ultralytics 代替母版。

记录实际解释器、torch/CUDA、GPU、ultralytics.__file__、criterion 来源和完整 Git SHA。历史服务器 Python3.10.13 / torch2.1.2+cu121 / RTX4090、定制 Ultralytics8.4.21；本地曾为 Windows＋RTX2060 6GB。环境不一致时记录实际差异，不自动升级依赖。

## 2. 原模型与训练配方约束

母版为 3 层 decoder、300 个常规 query。LIF 为原第20层，CBR decoder 为原第26层、读取 Neck [19,22,25]，CBR 36点采样、rho=0.10。以实际母版核验层号；不要改层号、通道、采样、修框公式、条件梯度、query 数量或推理接口。原 LIF 的融合排除规则保留。

两个原文件 LF 换行规范化后的历史哈希：

- lif_down.py：26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
- cbr.py：d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787

记录原始字节与 LF 哈希。优先把接线放在实验专用 model/trainer/criterion 包装中，保持两个原模块文件和所有原计算不变。不得更换 CBR/LIF 实现来通过测试。参数集合、形状、优化器分组、部署 state keys 不应增加可训练参数；实验配置与 checkpoint 元数据可单独保存。

母版完整损失 L0 包含 VFL＋5×L1＋2×GIoU，以及 encoder/decoder auxiliary 和 DN 项。matcher 原 class/bbox/giou cost=2/5/2。普通最终层匹配应只计算一次并明确复用给新机制；中间层和 encoder 仍按母版各自匹配。不要把最终层 match_indices 一次传给整个 super.forward，导致所有辅助层被强制使用最终层匹配。

从母版权威完整 args 与初始化脚本复现所有参数，不只抄下面摘要。公共初始化来源为 rtdetr_r18_lite_imagenet_backbone_init.pt，经原组合初始化流程构建训练模型；不把已训练母版 best.pt 当正式初始化。历史公共源权重 SHA256 为 fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e，实际文件需核验。

关键原配方：epochs=200、patience=50、batch=16、imgsz=640、nbs=64、seed=42、workers=8、device=0、AdamW、lr0=0.0005、lrf=0.01、momentum=0.937、weight_decay=0.0001、warmup_epochs=5、warmup_momentum=0.8、warmup_bias_lr=0.1、cos_lr=true、AMP=true、deterministic=true、cache=false、freeze=null、compile=false、rect=false、close_mosaic=10。

原增强：hsv_h=0.015、hsv_s=0.5、hsv_v=0.35、degrees=5、translate=0.1、scale=0.4、shear=1.5、perspective=0.0002、flipud=0.2、fliplr=0.5、mosaic=0.8、mixup=0.05、cutmix=0、copy_paste=0、bgr=0、erasing=0、auto_augment=null、multi_scale=0.0。以完整权威 args 的类型和值复核。通用 YAML 的 box/cls/dfl 字段不能拿来替换实际 RT-DETR criterion 权重。

正式数据沿用 crack_det：train6048/45573GT、val1728/12840GT、test864/6663GT 为历史计数。核对实际 data.yaml、划分、标签清单和身份，不重新划分、不加入新数据。归一化标签使用当前 batch 完成原增强后的 GT，不能读取原始标签替代增强后坐标。

输出 recipe_diff：原参数仅允许模型/实验路径、run名等身份变化，以及本方案明示的新损失参数；不擅改 batch、学习率、增强或 AMP。初始化 nc=80 到实际单类别训练模型的构建/权重映射沿用母版，不重新设计分类头初始化。训练前证明实际重建模型的公共参数值、优化器组和可训练参数数与母版一致。

## 3. RMD v1：唯一损失定义

RMD = Residual-Matched Denoising Loss，真实残差匹配去噪损失。只重分配原有正 DN 框回归监督。普通 query 所有原损失、DN 分类、DN 负例、query 数量、噪声采样、注意力掩码、GT身份和原匹配成本均保留。

固定参数：eta_max=0.50，tau=0.50，eps_num=1e-7。e 为 epoch 开始前已完成的 epoch 数，r(e)=clip((e−5)/15,0,1)。e≤5 完全回母版，e=6 为1/15，e≥20 完全启用。所有原DN decoder监督层使用同一组权重，不只改最后一层；不增加 encoder DN 或新监督层。

### 3.1 取对真实初始参考框

对 GT g_j，复用最终 CBR 后普通 Hungarian 匹配的 query i_j，取这个普通 query 实际进入 decoder 的初始参考框 a_j。取同 GT 的第k个正 DN query 实际进入 decoder 的初始受扰动框 d_jk。这里不是最终框、CBR前框或倒数第二层框。

母版 _get_decoder_input 返回 refer_bbox（常为logits）及 enc_bboxes；普通候选在 decoder 输入中顺序与 enc_bboxes 对应，DN 在前，分割长度以 dn_meta['dn_num_split'] 为准。以真实 decoder 对 refer_bbox 的 sigmoid 方式得到实际初始框；避免将 logits 当cxcywh，也不要对已经sigmoid的框再次sigmoid。

不能重新调用 get_cdn_group 生成一批“相同噪声”，那会改变 RNG/样本且失去对应关系。捕获同一次原前向已经生成的参考框；权重用的副本 detach，原 encoder/decoder 数据流继续保持原梯度。

优先用实验专用训练 model 包装传递局部只读上下文，保持 cbr.py/lif_down.py 原文件。若需捕获 decoder 的输入，可在单次 training predict 调用范围临时注册原 decoder 的 forward-pre-hook，读取实际 refer_bbox，finally 移除，把 detached 数据仅通过该次原 dn_meta 的局部副本传给 criterion。正常和异常返回都必须清空；不采用永久 hook、全局 monkey-patch、跨步实例缓存或第二次前向。原训练 forward 与原 RNG 序列必须相同，公共输出接口不变。

包装类必须可导入且不引入参数。hook 只用于读输入，不改任何参数、输入、输出或训练状态。eval/predict 不注册 hook，不依赖参考框元数据。若使用更直接的等价训练接线，可采用，但须提供同前向/同RNG证明，不能借机重写原 CBR。

### 3.2 正 DN 与 GT 对应

使用 dn_pos_idx、dn_num_group 和母版 get_dn_match_indices 的真实映射。最终普通匹配给出 query→GT；DN元数据给出正DN→GT；在同一张图、同一个全局GT身份连接两者，不按预测 IoU 另配GT。

以 (image_index, GT_identity) 分组计算均值，不能把不同图局部GT编号相同的目标混为一组。复用母版 DN 正索引顺序和GT重复顺序；不要自行假定正负交错或按总张量前一半选正例。padding和负DN不得进入残差相似度、均值、回归分子和分母。

若某个有效 GT 因常规 query 数量限制未获得最终普通匹配，该GT所有正DN权重设为1并记录；不重新匹配补一个普通query。batch没有GT或没有DN时按母版处理。r>0且应有DN但本次捕获/元数据丢失，应报接线错误，不能悄悄全置1掩盖。

### 3.3 残差相似度与有界权重

使用当前增强后归一化cxcywh，局部关闭autocast、FP32计算：

```text
rho(b,g) = [
    (b_cx-g_cx)/(g_w+1e-7),
    (b_cy-g_cy)/(g_h+1e-7),
    log((b_w+1e-7)/(g_w+1e-7)),
    log((b_h+1e-7)/(g_h+1e-7)),
]
s_jk = exp(-sum((rho(d_jk,g_j)-rho(a_j,g_j))**2) / (2*0.5**2))
sbar_j = mean_k(s_jk)
w_jk = 1 + 0.5*r(e)*(s_jk-sbar_j)
```

所有用于 w 的量停止梯度；不增加可学习尺度、置信度gate、IoU阈值、softmax、EMA统计、样本删除或额外正样本。这个方案按有符号误差相似度分配，不改成仅按噪声大小/IoU/当前loss难度加权。

w∈[0.5,1.5]，同GT权重均值为1；只守恒系数总和，不保证实际loss或梯度范数相同。全部相似度接近零时权重自然回到1。指数下溢到0允许；NaN/Inf或非法宽高必须定位报错，不能nan_to_num掩盖。

### 3.4 替换原 DN 框损失

对每个原有 DN decoder 层 ell，原匹配正DN框为 b_DN[ell,j,k]：

```text
Ndn = 当前实际有效正DN匹配数（不含padding和负DN）
loss_bbox_dn_at_layer = 5 * sum_jk(w_jk * sum_coord(abs(b_DN[ell,j,k]-g_j))) / max(Ndn,1)
loss_giou_dn_at_layer = 2 * sum_jk(w_jk * (1-GIoU(b_DN[ell,j,k],g_j))) / max(Ndn,1)
```

L1仍为母版归一化cxcywh误差，GIoU仍用母版函数与几何定义；保持各层原归一化和层间求和，最终层使用原CBR修正后的DN框。FP32加权归约；与原函数数值差异要记录。不要再额外除以K、batch或GT数，不更改母版外层DDP/batch缩放。

数学上 L_total=L0−原DN框损失＋上述加权DN框损失。工程上直接替换原loss_bbox_dn/loss_giou_dn和对应aux_dn的分子；不同时累计“原DN＋整份新DN”。DN分类soft IoU标签及其detach沿用母版，新机制不改变它。

不要仅按 postfix='_dn' 区分最后层，所有原DN层都需使用匹配顺序一致的权重；普通encoder/aux最终三项均不得被重加权。各层复用初始参考框生成的同一组 w，不用每层预测重算新权重。

r=0 时直接调用原损失路径，不捕获/计算残差。K=1或全部权重精确为1时走原DN框损失路径，以验证退化等价。正常训练没有可用普通匹配的GT保留w=1。验证不生成DN，原val/test/predict计算与模型输出保持。

### 3.5 epoch与模型生命周期

实际 Trainer.get_model 重建的对象必须使用RMD训练model/criterion，epoch应每轮同步到真实模型。resume恢复e和原optimizer/EMA/scaler；不能把保存时某个r值永久用于后续epoch。独立eval无训练上下文时直接用原验证路径，不为计算RMD切回train或二次forward。

普通初始框只为计算detached权重提供信息，新加权分支不通过w直接拉动普通初始框；但DN预测与普通预测共享网络，不能要求共享参数梯度与母版相同，也不能宣称无干扰。不要冻结LIF/CBR或修改原裁剪规则。

## 4. RMD 数学、接线与生效检查

### 4.1 必要正确性测试

1. 同GT相似度[0.9,0.4,0.1]在r=1时权重约[1.2166667,0.9666667,0.8166667]；权重范围、均值为1、停止梯度。
2. K=1严格w=1；全部s相等或全0时w=1；e≤5退回原路径；无GT、某图无GT、GT未获普通匹配时合理回退。不能把缺元数据当作合法无GT。
3. 两张图GT数量不同、动态K、padding、正负DN混合的人工索引例：逐项验证 GT身份、DNidx、初始框、w 与每层 b_DN 对齐。shuffle同GT的副本及对应预测后结果应不变，跨GT混合必须被测试发现。
4. 核实初始输入 logits→sigmoid 的语义，与原decoder实际入口对照；训练前向次数和get_cdn_group调用次数均为1，原RNG推进相同，临时hook退出后为0且异常路径也清理。
5. 相同参数/RNG/batch下，r=0或K=1与母版原loss、输出、梯度及一次原生optimizer更新一致；r=1,K>1时只有DN框项改变。原普通loss和DN分类项相同；总loss为精确替换式，不是双计数。
6. 新进程checkpoint加载/真实验证/恢复训练；自定义类可导入，零新增可训练参数，公共state keys保持，无捕获图或hook被序列化。

### 4.2 训练前生效检查：不能只证明代码可运行

当前母版 get_cdn_group 采用 K=max(1,num_dn//batch内最大GT数)，实际以源码和dn_meta复核。K=1时组内重分配完全退化，不能增加num_dn或改变原batch/增强来制造激活。

服务器preflight必须包含最多16个真实B16/640原增强训练batch的机制探测，时间边界与第5节共用。记录每个batch的GT数量、K、真实正DN数、不同K覆盖的GT次数，以及在r=1诊断状态下的s、w分布、mean(abs(w-1))、std(w)、abs(w-1)>=0.01的比例。至少8个有效batch才对生效性下结论，否则PENDING。

首版的保守可执行门槛，写入配置并在运行前固定：

- 按GT出现次数计，K>=2覆盖率须≥50%；
- 使用已训练母版权重的只读诊断时，全部有效正DN的mean(abs(w-1))须≥0.01。

这些是“机制不能几乎不工作”的工程门槛，不是涨点预测或统计显著性标准，也不能据结果反复降低门槛。

随机初始化下相似度很低不能据此宣布方案无效。优先在隔离模型中加载已有母版已训练权重，在train数据原增强上做无optimizer更新的机制探测；允许train模式生成原DN，但仅修改隔离诊断副本的BN缓冲，不保存回原权重。记录来源SHA、模型身份和文件hash，这个探测权重绝不作为正式初始化，也不读取test。

若可用母版权重不存在，仍完成实现、CPU正确性、普通容量检查和交付，生效性标APPLICABILITY_PENDING；给出明确的probe命令/缺失路径。不要伪造PASS，不能仅因为当前Codex本机无GPU而放弃实现。可自动发现并核对主目录历史母版run的best.pt，禁止下载未知权重或用TCR/PEQ结果替代。

若K覆盖或已训练母版探测的权重差异不足，标NOT_APPLICABLE并让正式start明确拒绝；保留已完成代码/报告并交付，说明具体数值。不要“修好检查”成自动改tau、num_dn、门槛或batch，也不要提交一条会白跑200epoch的无效训练命令。

诊断必须与正式权重、optimizer、scaler、RNG、dataloader和输出隔离。正式start仅在正确性/容量PASS且APPLICABILITY_PASS时允许。已通过的生效报告绑定代码、母版权重、数据和原配方身份，身份变化需明确重新探测。

训练时按epoch记录上述机制量、各层原DN框loss与加权后值、普通定位loss、r和K；不要再算第二次前向。记录真实训练中的激活变化，不因DN训练loss下降就判定有效。

这两个数值门槛是本提示词对设计稿“先确认生效”的具体化，并未修改RMD公式；在交付文档中明确记录。相邻论文包括DN-DETR、WACV2025 Adaptive Query Denoising、AAAI2026 MonoDLGD；不能宣称首次提出自适应去噪。首轮不扩展到embedding对比或困难样本挖掘。

## 5. 一次有界服务器短检与推理生命周期

本机缺数据、CUDA或B16容量时，完成本机可执行项并将服务器项标PENDING，不把缩小batch/分辨率小测记为正式容量PASS。使用原训练数据、B16/640、AMP、原优化器和累积规则，服务器整个preflight默认最多16个训练micro-batch、900秒；日志持续输出阶段、耗时和剩余边界，不自动重试无限循环或跑多套配置。

短检使用隔离副本，至少一次真实有效optimizer更新且新损失相关梯度有限，才可确认训练路径可工作；scaler.step调用或backward不等于完成更新。记录scale、overflow跳过、更新计数、参数变化、loss与关键梯度、显存峰值。若原始初始scale导致有限边界内没有有效更新，可在明确标注的隔离诊断副本用init_scale=128取得一次有界梯度/更新证据；不能改正式scaler、清除真实错误或宣称已验证原生scale全部适应过程。

为检查启用后的机制，诊断副本设置e=20；正式训练仍从e=0开始，诊断权重、optimizer、scaler、RNG及增强序列不继承到正式训练。RMD的生效探测与梯度容量检查是不同结论，共用总时间边界；时间不足将未完成项标PENDING，不绕过检查。

损失候选没有新增推理结构。复用母版已有验证工具，只补充分别覆盖“新进程加载训练模型checkpoint”和“独立val/AutoBackend/fuse/warmup”的必要检查，不把旧实验几十套融合矩阵全部再跑一遍。推理差异如果来自原路径，保留证据并定位，不随意放宽容差、删检查或改原模块。

此前PEQ在训练200epoch后，自动final_eval的AutoBackend warmup阶段报错。因此本实验必须在长训前用初始化为有限值的输入验证实际warmup→真实val batch路径。核对母版是否直接把torch.empty的未初始化内容送入warmup；若是，限定地将warmup输入改为同shape/dtype/device的torch.zeros（不改其他torch.empty、不改模型计算），记录这项评估可靠性修复并在隔离母版对照使用相同修复。两个实验采用相同内容；这不属于损失创新贡献。

独立评估固定使用FP32；训练期间原生AMP/验证策略保持母版。测试保存checkpoint→新Python进程→加载→按实际fuse规则→warmup→真实一批val，确保缺少训练上下文时也能评估。若没有实际权重/数据只可标PENDING，不能写“已通过正式test”。

## 6. 可恢复的实验操作流程

实现或复用本实验独立入口，至少支持 prepare、preflight、status、start、resume、val、test、pack。RMD另提供probe/生效检查入口。入口必须有--help；显式使用正确解释器和PYTHONPATH，不把主仓库已安装的另一版ultralytics当当前worktree代码。

- prepare：只读核对配置/数据/源权重；创建本实验初始化与身份记录。已存在时校验并复用，不覆盖训练产物。
- preflight：执行第4、5节有界检查，输出实际PASS/FAIL/PENDING及原因。不要让某一个PASS掩盖其他必需项PENDING。
- status：同时报告训练进程、dispatch、epoch/结束原因、tmux、日志、best/last是否存在、final_eval状态；读取为主，不自动启动新实验。
- start：验证当前代码SHA、工作区、配方、数据和preflight绑定。只启动一次本实验独立tmux，run已训练不得以exist_ok=true覆盖。保存明确的worker命令、PID、开始时间、dispatch、日志和退出码。
- resume：仅从本实验有效last恢复完整训练状态，不从best、诊断权重或另一方案权重续训；保留epoch和新增损失schedule。原训练已正常结束但末尾评估失败时，不走resume重训。
- val：对由原val fitness选择的best做独立评估，保存checkpoint hash、源码身份、数据身份、配置和指标，生成val锁。
- test：核对同一个best及val锁，使用相同口径，只做最终测试；不重新选权重或遍历阈值。
- pack：生成可下载LIGHT包，包含代码、配置、训练/评估/机制证据。失败时也应能打包故障证据，不能因缺少test而让pack完全不可用。

正式start必须在tmux rmd-v1-training中，关闭用户电脑不影响训练。日志tee链路使用pipefail并保存真实Python退出码；不要用tmux会话存在或tee返回0当作训练成功。PID/tmux冲突只处理本实验确认过的实例，不用pkill -f python或全局kill。训练可因原patience=50合法提前结束，不强行拉满200来改变配方。

训练完成与final_eval分别记录。如果训练已正常达到结束条件并保存权重，但final_eval失败，应保留原异常/exit1，状态明确TRAINING_COMPLETED＋FINAL_EVAL_FAILED。提供仅重跑评估的恢复路径：核对训练结束证据、无活动worker、权重及训练身份，再进行真实warmup/val；不能靠改exit.json、伪造COMPLETED、重写训练fingerprint或放宽校验来“修复”。评估代码修复使用独立eval身份并记录允许差异，原训练SHA永远保留。

## 7. 评估与分析包

沿用母版校正后的评估实现（历史corrected_sorted_conf_mask_v1），核对按置信度排序后的mask/预测/GT对应，不换用另一套默认评估器。独立val/test：imgsz640、batch16、workers0、FP32、conf0.001、iou0.7、max_det300、augment=false、rect=false、seed42；数据split明确。实际配置写入结果，不能把终端训练CSV峰值当独立评估。

母版参照：独立val mAP50–95=52.454272219190514%，独立test=52.20090191444802%。只作同协议比较，不根据test调lambda/tau/权重/门槛。原best选择应核对当前fitness函数，历史当前实现主要按mAP50–95，不套用旧版本0.1*AP50+0.9*mAP公式。

训练主日志保留原giou_loss/cls_loss/l1_loss列，机制诊断单列；实际优化总loss包含aux/DN，不能误说终端三项就是全部。失效与非有限诊断写严格JSON：非有限数值可用null附标志，并保留原异常，不用0伪装成功。

LIGHT包至少包含：README；完整训练/eval Git SHA与source snapshot或可复现patch；公式版本；数据与源权重/初始化/最终checkpoint hash；完整实际args和recipe_diff；preflight与机制检查；训练日志/results.csv/曲线/epoch统计；best/last大小与身份；独立val/test配置、指标及十IoU阈值AP；val锁；所有退出/故障/恢复记录；manifest文件大小与SHA256。LIGHT默认不塞数据集和巨大权重，不误标包含best.pt。

pack --include-predictions另导出压缩的val原始预测与GT（每图ID、尺寸、坐标空间、boxes/scores/classes/query索引、GT、所用best和配置身份），便于分析定位误差；不要事先按0.25筛掉低分预测。若输出test预测需单独命名和身份，不与val混杂。可另做可选FULL权重包，默认不上传大文件到Git。

训练产物、数据、权重、日志包不提交Git；提交源码、必要测试、配置模板和文档。不要把大压缩包或图像数据误加到提交中。

## 8. Git交付与服务器命令要求

完成实现与可执行检查后，对本实验独立分支普通commit并push，不强推、不合并主分支。不因为“服务器GPU检查待执行”而把已完成代码停在未交付状态。若网络/认证阻塞，保留本地提交并准确说明阻塞，不能伪造远端已存在。

推送成功后核对远端分支提交与本地完整SHA一致。最终交付 docs/rmd_v1/DELIVERY.md 与 server_commands.md，写明：变化、公式/接入位置、原模块/参数不变证据、已PASS项目、服务器PENDING项目、实际新增训练开销测量状态、完整SHA、分支、所有路径及下一条可执行命令。

服务器同步优先使用git fetch＋git show提取已提交的tools/sync_rmd_v1.sh，绑定完整SHA。此前raw.githubusercontent.com拉脚本多次超时，不再将curl raw脚本作为唯一入口。为网络命令设置有界超时、低速超时；失败退出，不自动循环。

最终server_commands.md里的命令必须使用此次真实40位SHA，不能留<FULL_SHA>、伪造哈希、模糊目录或让我自己从截图找名称。同步脚本应安全创建/复用本实验worktree并核验commit，不把用户主工作区checkout到实验分支。

分段提供：同步→prepare→preflight/必要probe→status→tmux正式start→查看日志/进度→训练后val→锁定best的test→pack。标清哪些必须等上一步完成；不要用一条命令把preflight和200epoch训练无条件串起来。Python环境设PYTHONUNBUFFERED=1、YOLO_AUTOINSTALL=false；确认必要的离线AMP检查资源，网络不可用时不跳过原检查或静默关AMP。

正式server工作不是你本轮自动执行的部分。交付必须自包含，公式和操作不能依赖本聊天截图、ChatGPT scratch路径或未随repo提供的附件。最后用简洁中文说明：已实施内容、验证事实、尚待服务器验证的条件、完整SHA及首条服务器命令；不要声称未跑过的训练/test已经成功。
