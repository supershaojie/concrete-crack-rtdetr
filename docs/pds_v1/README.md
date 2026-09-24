# PDS v1：实验 F 实施说明

本分支只实现原 RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + PDS 训练辅助监督（F）。
母版为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`；分支为
`exp-rtdetr-r18-lite-pds-v1`。没有叠加 LBC/LCD/MDR/RDL/ROR，也没有实施 E 或机制消融。
本地未启动正式训练、未使用真实 test 图像推理、未登录服务器。没有 PDS 性能结论。

## 固定实现

主 YAML 不变，共27个节点。PDS从同一次训练前向读取第4层P2（64通道）
和第19层top-down Neck P3（256通道），没有把第5层接成Neck P3。
四相位按 `(dy,dx)=(0,0),(1,0),(0,1),(1,1)` 拆分，使用同一个共享预测头。
P3投影计算一次并广播；GN对每个图像的每个相位独立计算。
输出显式散回fine grid，640输入时为160×160，row-major展平。

共享头：64→32和256→32的无bias 1×1投影，相加后GN(4,32,eps=1e-5)+SiLU，
再无bias DW3×3、GN+SiLU，然后1类logit和4维box投影。
新增10,821参数、11个 `pds_head.*` 状态项，无新增持久buffer。
nc1未融合主模型20,149,765参数，训练模型20,160,586参数。
原552个公共状态项逐名逐值审计，公共参数未换前缀。

PDS seed=424003，仅在隔离CPU RNG范围构造。投影和DW使用Conv默认初始化，
GN为weight1/bias0，cls weight N(0,.01²)、bias logit(.01)，
box weight N(0,.001²)、bias `[0,0,logit(.1),logit(.1)]`。
正式统一源仅允许SHA256
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
prepare保存独立未训练nc80母版checkpoint；在真实Trainer的nc80→nc1重建、
原母版严格类型/拓扑/初始化审计之后附加PDS，避免临时模型的头被Trainer丢掉。
新增构造不消耗外部RNG，原9项类别适配有明确枚举。

PDS头、解码、分配、IoU/GIoU、损失均在局部关闭autocast的FP32中计算，
特征和权重 `.float()` 保留autograd，主模型AMP不变。
中心为 `((u+.5)/Wq,(v+.5)/Hq)+.1*(tx,ty)`，宽高为sigmoid(tw,th)。
预测框不裁剪；GT只使用当前增强batch的独立副本，裁到实际画布后跳过非法/退化框并计数。
检查标签形状、浮点cxcywh格式、batch_idx和单类cls；主criterion收到原标签。

每GT候选优先取框内点，超过128按 `floor(t*(n-1)/127)` 等距截取；
无框内点才取归一化距离最近一点，不改变GT。质量为
`sigmoid(z)^.5*(IoU+1e-6)^2`，同分按fine-grid索引。
GT按候选数、面积、原标签索引排序，四轮轮流各取一个尚未占用点；
绝不复制点监督冲突GT，未匹配GT如实统计。
负点仅在所有扩展GT框之外，每边扩展max(网格步长,.1*框宽高)；
未选中的框内/扩展框位置忽略。这是框级检测责任点，不是裂缝像素标签。

Focal alpha=.25/gamma=2。每GT正项先按其正点数平均，再按匹配GT数M平均；
负focal总和乘1/4后除max(M,1)。归一化cxcywh的L1四维取sum。
逐图计算后取batch均值：
`LPDS=Lcls+5*Ll1+2*Lgiou`，
`Ltotal=L0+.25*clip((epoch-5)/15,0,1)*LPDS`。
epoch零基，e5仍关闭、e6开始、e20全开。r=0完全走原母版loss；
L0保留原criterion完整求和及三项loss_items，诊断另存。
不增加batch/accumulate/world_size缩放。

## 优化、保存与恢复

原AdamW公共分组、LR/decay、warmup、cosine、nbs=64、梯度累积、
GradScaler、EMA及全参数max_norm=10裁剪不变。新增参数各出现一次：

| 组 | PDS参数 | 初始LR | decay |
|---|---|---:|---:|
| weight | p2_proj/p3_proj/dw/cls/box.weight | .0005 | .0001 |
| bn | norm1/norm2.weight | .0005 | 0 |
| bias | norm1/norm2/cls/box.bias | .0005 | 0 |

实际原warmup/cosine仍生效。没有第二个优化器或独立裁剪。
有效更新通过optimizer实际step hook计数，overflow独立记录。
全部辅助关闭的窗口保持head.grad=None，不产生AdamW状态或decay；
空监督/r=0微批次不清除已累积梯度。

PDS Trainer补存完整FP32原模型、EMA、optimizer、scaler、scheduler、early stopper、
epoch、PDS定义、Python/NumPy/torch RNG、最近optimizer边界和待累积梯度。
母版默认EMA-only和半精度optimizer保存不足以满足本任务。
BaseTrainer只有三个可选恢复接点；非PDS路径行为不变。最终验证不strip训练best/last。
仅支持完整、未结束的epoch边界last；拒绝deploy、丢头/optimizer、结束checkpoint和源码SHA变化。
多worker预取及CUDA非确定性意味着恢复不承诺跨进程逐位复现数据顺序。

正式入口拒绝DDP/compile，禁止OOM时静默减batch。
原NaN恢复会把EMA装入原模型并丢弃累积状态；PDS遇到非有限loss/fitness明确失败、
保留异常，使用完整resume路径，不默默采用EMA-only回退。
一般AMP梯度overflow仍按原scaler正常跳步。

## 验证、部署与独立评估

只覆盖训练loss，不覆盖predict。eval、每轮validator的loss(preds)、无标签predict均零PDS调用；
best按母版主分支fitness选择。PDS梯度进入P2/P3上游及AIFI，
没有直接进入第20层LIF、后续PAN或第26层decoder；联合训练仍可间接影响它们。

独立val内部从本轮best的EMA（缺EMA才model）重建标准RTDETRDetectionModel，
严格加载全部公共参数/buffer，仅去掉PDS。保留names/nc/YAML/args、decoder非持久缓存
和EMA的requires_grad执行状态。同精度未融合原始输出要求完全相同，不重排query或放宽容差。
已修正冻结EMA与可求导重建模型导致本机CPU AIFI算子差异的问题。
不重新载入统一初值或旧母版best，不覆盖训练checkpoint。

复用母版独立评估策略 `corrected_sorted_conf_mask_v1`：
FP32、640、B16、workers0、conf=.001、IoU=.7、max_det300、无增强。
原评估helper仅增加可选runtime_info参数，默认调用不变。
独立val完成后锁定deploy和来源best SHA256；test只对此身份运行一次，
已完成的相同test返回既有结果；失败/中断保留开始标记，不自动重跑。
prepare/preflight/start/pack都不运行真实最终test。
检查中的val/test用两张train图复制成的独立工程fixture，不是数据集test。

参数/FLOPs按标准部署模型统计。THOP为multiply-add=2、有限B1/640输入，
自定义算子覆盖有限，不是精确硬件指令数。
本地未融合标准模型为20,149,765参数、58.672512 GFLOPs（THOP口径），
训练开销由服务器报告单独给出。保留LIF特殊融合保护。
AutoBackend仅将warmup的torch.empty改为torch.zeros，避免未初始化非有限输入。

## 配方、数据与操作

109字段来自母版归档并由当前get_cfg逐字段核验，只改model/name/save_dir及等价路径。
200epochs、patience50、B16/640、AdamW、seed42、AMP、workers8等不变，PDS定义单独保存。
train/val/test清单和标签指纹与母版一致：6048/1728/864图、45573/12840/6663框。
未改split，历史增强同源图跨split限制仍存在。不自动升级环境。

入口为 `tools/pds_v1.sh`、`tools/sync_pds_v1.sh FULL_SHA`，
设置绝对Python、PYTHONPATH、PYTHONUNBUFFERED、YOLO_AUTOINSTALL并cd到worktree。
同步只对本实验worktree安全快进，拒绝已跟踪修改和冲突文件；
不切主工作区、不reset/clean/force-push。bootstrap提取脚本后必须执行官方sync，
生成delivery.json并归档旧metadata。

prepare核验初值、原AMP的bus.jpg/yolo26n.pt可读性及哈希、配方、数据、导入路径，
生成初值与配置。资源用主库已验证本地副本，缺失明确报错。
preflight执行有界CPU机制/生命周期/增强覆盖，再做一个隔离B16/640 CUDA诊断。
诊断epoch20、原AMP/AdamW/accumulate4，最多16micro-batch/900秒，2次有效更新即止；
至少1次有效更新且梯度/参数变化全通过才PASS，0次失败。
**diagnostic_only_scale=128只用于隔离副本，正式scaler仍用母版默认或完整resume。**
不复用诊断权重，超时只终止直接创建的PDS诊断进程组。
身份不变时复用已通过预检，start不重复重预检。

start/resume创建独立 `pds-v1-training`，日志为本worktree
`outputs/pds_v1/console_<UTC>.log`，tee显示并保存输出，立即捕获Python PIPESTATUS。
实际处理batch才打印PDS_TRAINING_RUNNING。status核验命令/dispatch/PID/子进程，
区分未启动、派发、设置、运行、完成、失败，记录实际epoch/早停，旧exit不覆盖新dispatch。
不检查、停止或占用LBC输出与会话；同卡竞争只在实际OOM时报告。
初始化失败仅留args等、没有checkpoint和任何results数据行时，
显式 `archive-failed-start` 改名保存整个失败目录，再允许start；有效last走resume。
pack默认LIGHT，只有身份、配置、差异、报告和日志摘要，无权重/数据。

含交付完整SHA的独立可复制服务器命令在最终交付的 `server_commands.md`。
服务器B16容量、Linux tmux和正式训练为PENDING，本地PASS不能替代。
正式sync脚本已在隔离Git夹具中实际运行：生成交付记录、原metadata归档、保留主HEAD和已跟踪修改、
拒绝活跃PDS而忽略LBC会话均通过；只替换解释器绑定和离线fetch运输。
同一夹具中实际prepare和LIGHT pack通过，真实初值与配置已生成，包内没有权重。

## 证据与研究边界

开发机Python3.9.25 / torch2.7.1+cu118 / RTX2060 6GB / 仓库Ultralytics8.4.21。
CPU和本地CUDA机制、逐key初值、真实梯度/更新、部署一致性通过；
实际CPU Trainer保存/跨进程恢复、原AMP检查、FP16 validator、AutoBackend预测通过。
详见validation.json与精简报告。
两批真实增强train共4图，23/23 GT匹配，92正点，7个候选重叠点，0 fallback。
真实小样本未覆盖fallback，合成极细框/冲突用例补充验证，未修改标签制造覆盖。

LIF/CBR内容不变，raw/LF哈希见provenance.json。
未取得参考ZIP，按固定合同独立实现，没有整包替换Ultralytics。
四相位重排不创造信息，共享头部分计算可能与相应高分辨率卷积等价；
密集辅助监督和质量分配有已有工作，不主张这些概念全球首创。
母版历史val约52.4543%、test约52.2009%只是历史参考，不能预填PDS成绩。
是否有效需等待F正式训练与独立评估，再考虑E和机制消融。
