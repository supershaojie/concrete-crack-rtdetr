# GNR v1：实现与运行说明

唯一组合为 RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + GNR。
这是待验证的损失候选；工程检查和母版机制诊断不能证明精度提升。
正式 200 epoch 实验由用户在服务器单独执行 start。

母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
分支：`exp-rtdetr-r18-lite-gnr-v1`。修订需求全文见 `specification.md`。
原参数冲突证据见 `history/initial_contract_conflict.json`；用户 2026-09-29 勘误已解决该冲突。
旧核验提交 `8773be2` 不是功能交付。完整操作顺序见 `server_commands.md`。

## 算法与保护范围

唯一算法改动：最后 decoder 层普通 query 的部分未匹配负分类项乘以 detached 权重。
原正项和 IoU 质量目标、最终 Hungarian 匹配、回归、encoder、auxiliary、DN 均保留。
没有新增参数、buffer、教师、forward hook、第二次模型 forward 或推理计算。
原模型 YAML、LIF-Down、CBR、decoder、matcher、VFL、Trainer 核心源码均未修改。

实际参数读取 `criterion.vfl.alpha/gamma`，预期为 0.25/1.5。
调用链是 `RTDETRDetectionModel.init_criterion` → `RTDETRDetectionLoss` →
`DETRLoss.__init__(gamma=1.5,alpha=0.25)` → `VarifocalLoss(gamma,alpha)`。
VFL 类自身的默认 0.75/2 不代表运行值；实际参数变化会明确报错，GNR 没有独立可调 alpha/gamma。
sigmoid 调制因子保留梯度，解析式包含它的导数。

权重严格使用修订文档的 GT 最大 IoU 间隔、欠拟合门控、含 bias 的特征余弦、完整负梯度强度、
组内稳定排序、前驱 a_k、框 IoU 及冗余和。GT 最大 IoU 精确并列不参与；强度精确并列按原 query ID。
beta=0.5，r(e)=clip((e-5)/15,0,1)，最强项权重为 1。
beta=0 或 e≤5 直接走母版完整 loss 路径，跳过额外特征返回和关系计算。
batch 全无匹配时母版 FocalLoss fallback 保留。
归约仍为原 mean(query).sum() / (max(匹配数,1)/query数) * class_gain；不重复加入负项。
独立 detached diagnostics 不进入自动求和的 loss 字典。

## 张量和文件

| 路径/张量 | 契约 |
| --- | --- |
| `ultralytics/models/rtdetr/gnr_model.py` | 可导入模型包装类；原 head、模块注册和普通推理不变 |
| `forward_with_features` | 用原模块执行母版 CBR forward 的同样运算，显式返回已有 final_query |
| 训练 decoder 输出 | 框 [3,B,DN+300,4]、logits [3,B,DN+300,1]；normalized cxcywh |
| 最后 Linear 输入 | [B,DN+300,256]，对应 Linear(256,1)，bias=[1] |
| `criterion_inputs` | 按原 dn_num_split 切开 DN 前缀；普通 h=[B,300,256] |
| criterion 输入 | encoder 放在前面，得到 [4,B,300,4/1]；GNR 只处理最后层 |
| `gnr_loss.py` | 最终 matcher 一次、原 bbox_iou 质量目标一次；权重直接复用 |
| GT 索引 | 原 matcher 的 batch 展平索引按 gt_groups 校验；不向辅助层传播 |
| 普通 eval/export | 完全继承母版输出；无 GT 和 GNR 运算 |
| 显式诊断 eval | 同一次前向返回 h；最终输出 [1,B,300,*] |

权重几何/特征/logit 计算局部 no_grad+FP32，活跃分类保留原 AMP 数值路径。
按 GT 组做张量两两运算，无逐 query CPU 往返或 Python 成对循环。
未融合参数 20,149,765，state_dict 552 项；新增参数为 0，不混比融合/未融合计数。

## Prepare 和身份

复用母版 `init_c19_lif_v1.py` 的公共映射与 CBR/LIF 零初始化核验。
公共源哈希必须是 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
本实验独立初值为 WT/weights/gnr_v1_controlled_init.pt，不从母版 best 微调。
nc=80→1 的九个允许形状变化键逐一核验：encoder 分类 weight/bias、三层 decoder 分类 weight/bias、DN class embedding。
其余键、形状和加载值必须一致。已经核实的初值复用；中断后遗留文件只在全部受控张量严格一致后恢复元数据。

`resolved_formal_config.yaml` 保留完整 109 字段及类型；GNR 配置独立见 `gnr_config.json`。
实际母版 args 与归档逐项比较，仅对文档列出的 c19-lif-v1 / c19-lif-v1-gatefix 两个 model 路径做比较副本别名归一。
原 args 不改写。实验路径/身份差异单独核验并记录，不放松训练超参数检查。

首次数据扫描记录图片/标签哈希、GT 解析、尺寸、split 和文件/目录元数据。
固定计数：train 6048/45573、val 1728/12840、test 864/6663（图片/GT）。
以后复用快照，检查已记录元数据；失配必须显式 prepare --refresh-data，正式开始后不可改写快照。
已有增强家族跨 split 是原评估局限，本实验不重划分数据。
绑定包含功能源码、配方、数据、公共源/初值、结构、运行目录与环境。
功能变动使预检失效；纯说明文档变化不改变功能摘要。
status/pack 不导入 torch、不加载模型、不扫描原始数据。

## Diagnose / preflight

成功母版 best 哈希固定为 `24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`，
来自成功母版正式 val 归档，见 `mother_reference.json`。
diagnose 默认总预算 900 秒，seed42，最多 64 张 val、8 个原在线增强 B16 train batch。
val/train 使用隔离实例，无 optimizer step，不用 test 诊断机制。
每批落盘样本清单、覆盖、组大小、a/R/w/方向分位数、欠拟合比例、最强项保护和强度份额。
主分母是全部普通负项，分组口径另列；无分母为 null 并说明。
完整零干预为 ZERO_INTERVENTION，start 拒绝；非零证据为 REVIEW，供用户审阅，不设任意收益阈值。

preflight 默认最多 900 秒和 16 microbatch，以先到为准，使用独立临时 run。
保留原 B16/640/AMP/AdamW/nbs64/warmup/累积/GradScaler，只把 GNR 上下文设 e=20。
必须观察 native optimizer step 后有限、非零参数变化才是 TECHNICAL_PASS。
记录 loss、关键梯度、参数、累积边界、scale 前后、跳过次数、更新量、峰值显存和绑定。
无更新且预算耗尽为 exit2/PENDING；真实 OOM 为 RESOURCE_ERROR；实现错误单独标记。
不会降低 scaler、减 batch/尺寸、关闭 AMP、扩大预算或自动重试。两个动作都不会自动 start。

## 运行、恢复与共享 GPU

入口 `python tools/gnr_v1.py --help`，动作：prepare/diagnose/preflight/status/start/resume/val/test/finish/pack。
默认 device=0，无空 GPU、显存阈值、GPU 进程数量、GPU 预约或等待空卡逻辑。
内核锁只保护本实验同一 run，死亡后自动释放；进程核实命令、cwd、PID 和创建时间。
不终止其它实验，不保证共享显存一定容纳两项任务；真实 OOM 原样报告。

tmux worker：gnr-v1-training；finish worker：gnr-v1-finish；viewer：gnr-v1-view。
Python 与 tee 退出码分别保存；worker 退出后 viewer 仍保留输出。
SSH 断开不结束训练，Ctrl+B 后按 D 只 detach；不改全局 remain-on-exit，不 kill-server。
start 要求当前 TECHNICAL_PASS 与对应 REVIEW 诊断，执行 start 就是用户审阅后的决定。
新 run 从公共初值 e=0 开始，200 epoch/patience50；resume 仅接受相同身份、未完成的完整 last。
恢复 optimizer/scaler/EMA/真实 epoch/渐入；strip 后的完成 checkpoint 不能真 resume。
结束验证异常时记录训练已完成和 best，使用 finish 补评估，不重训。
原 Trainer 自带训练验证及结束验证保留，与显式独立 FP32 val/test 分开记录。

## 独立评估、离线阈值和包

best 由完整训练 val mAP50–95 选择；test 不选 checkpoint 或参数。
独立协议：FP32/640/B16/workers0/conf=.001/iou=.7/max_det300/augment=false/rect=false/seed42。
排序逻辑源于母版 `c19_lif_v1_results.py`：先排序，再对排序后的分数做置信度掩码，保留原 query ID，不加 NMS。
只在独立评估入口修正历史掩码错位；训练验证行为保持母版。

首次必要 val/test 各一次推理，同时保存全部 300 query（含低置信度）、GT、图片 ID/尺寸/预处理、
指标 query ID 和 IoU .50:.05:.95 TP 矩阵。框为输入图像像素 xyxy，记录 stretch 的 x/y 逆缩放。
另存 raw_stats.npz、AP 和 PR/P/R/F1 曲线原值；指标仍只用固定过滤后的预测。
身份与产物全部一致时重入显示 REUSED；success 索引缺失可由完整可信 metrics 恢复。
其它缺口保留旧目录，只补必要 split，不自动重跑母版。

离线从 val 的原生平滑最大 F1 曲线选择置信度，固定用于 test。
TP/FP/FN/P/R/F1 为 IoU=.5，并对新过滤集合重新匹配，不能截断旧 TP 数组。
test 自身最大 F1 仅作为报告值；历史母版 P/R/F1 不强行与 val 固定阈值口径比较。
涨跌用百分点，机器文件保存未舍入值。

finish 补 val/test、离线分析，保存其 Python/tee 退出码后再离线 pack。
完整包名 GNR_v1_COMPLETE_<timestamp>.tar.gz；缺项只能命名 INCOMPLETE。
包含源码/补丁、配方/别名、初始化/数据身份、检查/诊断/预检、训练、全部评估和阈值产物。
不含 best/last 权重本体或原始数据图片，记录权重路径/哈希/选择 epoch。
MANIFEST 覆盖全部载荷并逐文件回读 SHA256；manifest 自身由外部包哈希覆盖。
本次打包结束后的退出码/包哈希是外部 receipt，避免自引用。
viewer 最终显示全部指标、母版百分点差、best、退出码和包路径，不自动下载。

## 验证与研究边界

复现：`python tools/check_gnr_v1.py --cuda`，报告 `local_validation.json`。
覆盖真实 VFL autograd、Linear weight/bias、边界/并列/前驱/GT 偏移、关闭等价、活跃范围、
无新增框梯度、真实 Linear 输入、DN 切分、推理、EMA 和 native 恢复。
CPU 严格比较使用单线程确定性执行；此前多线程累积差异 2.7940e-09 已记录，没有放宽精确比较容差。
CUDA 固定张量检查与真实 B16/640 AMP 更新预检严格分开。
实际开发端记录见 `VALIDATION.md`：CPU/CUDA/固定张量AMP与恢复检查通过；
母版诊断为REVIEW；真实B16/640 preflight在首个backbone前向遇到CUDA OOM，有效更新为0，未通过。
AutoDL的有效更新预检、Linux tmux全流程、正式训练和独立完整val/test均待执行。

复用仅限指定母版的初始化/Trainer/IoU/匹配及已修正独立评估排序逻辑；没有导入其它实验算法或 GPU 独占 helper。
Windows 用户路径含 apostrophe 时，原权重下载 helper 删除引号；此入口必要时以同文件相对路径调用母版逻辑。
局部最后 Linear 的 SGD 解释不证明 AdamW 整网更新或 AP 改善；权重下界也不保证重复误检不增加。
已有 test 被多轮使用，微小单次改善不能宣称统计显著或外部泛化。
