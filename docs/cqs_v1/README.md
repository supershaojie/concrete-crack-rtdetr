# CQS v1：原 RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR

本实验直接从 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 派生，分支为
`exp-rtdetr-r18-lite-cqs-v1`。实现、有限本地验证和交付不代表已完成正式训练或超过母版。
正式 200 epochs / patience50 训练和最终 test 由用户在服务器显式启动。

## 模型实现

`ultralytics-main/ultralytics/nn/modules/cqs.py` 中的 `RTDETRDecoderCBRCQS`
继承完整的原 `RTDETRDecoderCBR`，只替换 `_get_decoder_input`，并提供显式、有界诊断上下文。
新 YAML `rtdetr-resnet18-lite-cbr-lif-down-cqs.yaml` 仅改变节点26的类名；构造时保留
nc80 以复用公共 ImageNet 初始化，实际选择入口强制 nc1。真实 Trainer 按数据 nc1 重建。

固定机制见 [research.yaml](research.yaml)：原 dtype、原 `topk` 先产生300个原生位置；
显式排除后补至最多600个互异候选；保留原顺序的前240个，再按
`s * (1 - 0.5 * max_j(IoU(i,j)^2 * max(cos(i,j),0)^2))` 依次加入60个。
每次加入都更新冗余关系；并列选择池中最前的位置。局部 detached FP32 选择不改变主 AMP。
候选 bbox head 的600个前向保留梯度，encoder loss 只 gather 最终300个原始预测。
所有 embedding、reference、bbox、class、anchor 使用同一图像的同一索引。

训练、EMA 每轮验证、独立 val/test、predict 和 resume 均保留 CQS。
beta=0 或 enabled=false 仅用于诊断，并直接进入父类原实现；正式入口拒绝非固定配置。
N<300报错；300≤N<600取全部剩余位置。没有尺度配额、GT筛选、阈值NMS或额外参数。

原文件按 LF 归一化 SHA256 完全保留：

| 文件 | SHA256 |
|---|---|
| lif_down.py | `26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7` |
| cbr.py | `d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787` |

节点19是最终 Neck P3；640实测为 B×256×80×80。节点20仍为原LIF，decoder仍接
[19,22,25]，3层、300普通query、原DN；CBR只修正最后一层框，36点、rho=.10不变。
`head.py`、`transformer.py`、criterion、Trainer核心均无修改。
唯一额外的框架修复是 AutoBackend 预热由未初始化 `torch.empty` 改为 `torch.zeros`。
本地真实保存/最终验证链复现了 empty 输入导致525个非有限encoder logit；异常证据随本地交付保留。
没有关闭 AMP 检查、融合或有限性检查。LIF仍保留自身BN，其余模块按母版正常融合。

## 初始化与配方

公共未训练源 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
复用母版 `init_c19_lif_v1.controlled_models`，先验证母版/CQS原构造状态完全一致，再严格加载
同一公共源；未用母版best初始化。原参数和buffer名称、shape、value逐key审计，新增参数/状态为0。
真实 nc1 未融合模型为 **20,149,765 参数、552状态项**。Trainer的9项原生nc80→nc1适配
与同RNG母版逐值一致，其余543项精确加载。全部应训练参数继续训练。

[parent_args.yaml](parent_args.yaml) 保存合同109字段；真实成功母版args逐字段核对。
只允许model/name/save_dir身份变化。历史model路径是否带`-gatefix`的差异会明确记录；其余字段
不吸收新框架默认值。研究配置单独保存，不塞入框架未知args。
AdamW分组、warmup、cosine、nbs64与原动态accumulate、clip10、GradScaler、EMA、best fitness、
原训练val统计和保存/恢复实现均继承。实际criterion loss gain为class1/bbox5/giou2，matcher为2/5/2，
不是用通用box/cls/dfl字段替换RT-DETR loss。

固定正式字段：epochs200/patience50/B16/640/workers8/device0/seed42/AdamW/lr0=.0005/lrf=.01/
weight_decay=.0001/warmup5/cos_lr=true/amp=true/deterministic=true/cache=false/freeze=null/compile=false。
其余在线增强完整继承。母版路径/标签清单见 [parent_data_identity.json](parent_data_identity.json)。
本地所有split身份与真实归档一致；6048/1728/864张，45573/12840/6663实例。
prepare另外记录全部图片内容SHA256，不重新划分或修改数据集。

## 已执行验证与范围

详细报告见 [validation.json](validation.json)。实际本机为Python3.9.25、torch2.7.1+cu118、
Ultralytics8.4.21、RTX2060 6GB。实现采用torch2.1已具备的API，没有升级依赖。

| 检查 | 实际结果 |
|---|---|
| CPU/CUDA池构造、并列、核心/唯一性、N边界 | PASS；包含N=300/301/525/599/600/8400 |
| 独立CPU标量关系与逐步贪心oracle | PASS；包含零feature、近零面积、两个不同j的max陷阱、互补替换 |
| 两图所有gather、原分数、原DN/mask、detach和encoder辅助loss梯度 | PASS |
| 原构造/公共源/序列化/真实Trainer重建 | PASS；nc1 20,149,765参数、552状态项、0新增 |
| 真实Trainer保存、EMA/每轮val、resume、独立进程val/predict | PASS；B2/160合成图接口小测 |
| 有效optimizer更新 | 诊断首阶段1次，真实resume阶段1次；LIF/CBR/encoder/backbone梯度有限，参数确实变化 |
| 原生AMP检查 | 本机实际执行并通过；bus.jpg和yolo26n.pt身份已记录 |
| 正式B16/640容量与服务器torch2.1执行 | PENDING，必须服务器preflight |
| Linux tmux派发 | PENDING；Bash语法和真实pipeline退出码夹具已验证 |
| 正式200轮训练、完整独立val/test、AP收益 | NOT_RUN |

接口诊断显式使用`diagnostic_only_scale=128`；原生65536首步overflow的尝试保留证据，不能把该
隔离诊断说成验证了原生scale完整适应过程。正式start/resume不设置诊断scale、不继承诊断状态。
原生cuDNN/grid_sample的deterministic warn_only行为继承，不作多种子显著性或独立原图泛化声明。

融合前后的小样本自然CQS选择和最终输出均记录。有一组同集合、不同顺序，直接行对行最大差
不具有同query意义；另一固定检查索引完全一致、最终输出最大绝对差2.22e-9。
没有套用其他实验的截止位豁免；融合改变离散顺序/集合仍应检查并记录。

同GPU、B1/640、2次预热/3次重复、CUDA同步，实际短前向：

| 精度 | 同源母版路径 ms | CQS ms | 母版/CQS峰值allocated bytes |
|---|---:|---:|---:|
| FP32 | 43.615 | 48.638 | 161984000 / 161984000 |
| AMP | 28.145 | 41.345 | 180931072 / 180931072 |

这是本机短前向测量，不是B16训练容量、稳定FPS或零开销声明。相同总峰值可能由更早的层决定，
CQS仍增加bbox计算、关系矩阵与60步选择。框架自动打印的GFLOPs未计全离散选择，不能用作CQS总计算量结论。

## 机会诊断

使用真实母版best（SHA256 `24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`），
固定排序前64张train图，RGB stretch640 /255、无随机增强；共174个GT。
文件清单及图片/标签SHA在probe报告中。指标允许多个候选覆盖同一GT，名称为候选覆盖率。

| IoU | 原300 | CQS300 | 池600 | 池新增GT | CQS新增/丢失GT |
|---|---:|---:|---:|---:|---:|
| 0.3 | 100% | 100% | 100% | 0 | 0/0 |
| 0.5 | 99.4253% | 99.4253% | 99.4253% | 0 | 0/0 |
| 0.7 | 81.6092% | 81.6092% | 82.7586% | 2 | 0/0 |

平均最高IoU：原300=.805771、CQS300=.805926、池600=.807439；平均引入原Top300之外16.890625个。
**这份有界机会诊断暂不支持该机制**：池的新增覆盖很少，CQS未在所测阈值取得新增覆盖。
没有据此调整240/60、候选池或beta；这不构成最终detector AP结论。

## 操作与结果保全

Shell入口`tools/cqs_v1.sh`，同步入口`tools/sync_cqs_v1.sh FULL_SHA`，Python入口`tools/cqs_v1.py`。
正式服务器路径、Python和tmux均按合同固定；worktree为`/root/autodl-tmp/projects/Crack_RTDETR-cqs_v1`，
project为主仓库`runs/c_series`，name为`cqs_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。

sync普通fetch，核对remote/完整SHA/母版祖先/分支/共用Git身份；拒绝脏工作树、分叉历史或本实验活跃进程。
prepare审计初值/109字段/数据内容与AMP资源。preflight在独立模型、optimizer、scaler、目录中执行
最多16 micro-batch、最多900秒、目标2次有效optimizer更新；边界至少1次且相关检查完整通过才能PASS。
start只复用与源码、配置、源权重、数据、相关环境一致的本实验PASS，不因报告时间戳重复预检。
日志直接出现在tmux并由tee保存，立即读取PIPESTATUS分别记录Python和tee状态。

每100 step采样选择标量和少量索引，每epoch汇总；不保存整批feature/关系矩阵。
状态区分NOT_STARTED、DISPATCHED、SETTING_UP、RUNNING（真实batch处理后TRAINING_RUNNING）、COMPLETED、FAILED。
每次dispatch独立退出记录；COMPLETED记录实际epoch和早停，不声称COMPLETED_200。
同实验活跃拒绝重复启动；不停止或阻止其他实验。status检查完整命令与子进程，不使用全局pkill。

`resume`要求本实验有效last，原生恢复optimizer/scaler/EMA/epoch；没有有效last的初始化失败，使用
`archive-failed-run`在确认无活跃进程、无.pt、无results数据行后重命名保全，再执行start。
不删除结果、不exist_ok覆盖、不把所有失败都当作resume。

独立val用本实验best的新YAML严格重建，FP32/640/B16/workers0/conf.001/iou.7/max_det300/augment=false。
实际复用母版`c19_lif_v1_results.postprocess`的`corrected_sorted_conf_mask_v1`实现，输出原CBR最终框。
保存mAP50–95/AP50/AP75/P/R与完整配置；P/R仍为各模型自身最大F1工作点。空split/缺GT报错。
val锁定权重SHA256和代码/数据身份；test必须显式调用，相同已完成身份只返回已有结果。
pack不会启动训练/评估；LIGHT包含机制epoch汇总和受限代表事件，排除数据和大型.pt，保留完整原日志和权重。
清单包含大小和SHA256，归档逐项读回核验，FileZilla路径写入`outputs/cqs_v1/FILEZILLA_PATH.txt`。

实际交付SHA确定后，运行`tools/cqs_v1_delivery.py FULL_SHA`生成
`outputs/cqs_v1/server_commands.md`的全部独立复制命令；服务器sync也会生成该文件。
tmux脱离键为Ctrl+B，然后D。报告和LIGHT内未执行的test明确标记NOT_RUN。

## 设计边界

CQS按本轮合同固定设计，不是DDQ的直接复现，也不声称distinct query selection或外观/几何关系为新概念。
工程验证与研究效果分别报告；母版参考val52.454272%、test52.20090191444802%没有被本轮重新运行或超越。
后续可建议原母版/CQS和内部核心/互补消融，但本轮只实施这一个组合，未新增成套消融、多种子或holdout。
