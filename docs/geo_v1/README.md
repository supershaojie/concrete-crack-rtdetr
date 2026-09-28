# GEO v1：固定研究候选

唯一实验是 RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + GEO。源码直接基于母版
`a0459d6a652cb702699087c88fa39a3e4c4087ec`；独立分支为
`exp-rtdetr-r18-lite-geo-v1`。不导入 ARG/ARG v2、QCC、RMD、RDL、ROR、PEQ、TCR、DTR
损失、模型类或训练状态。保留母版 YAML、任务损失、训练器主体和两项原模块。

## 固定公式和边界

`GEO-v1-envelope-K3`：enabled=true，lambda_geo=0.20，topk=3，eps_area=1e-9，
ramp_start=5，ramp_full=20。e 是零基 `trainer.epoch`，r(e)=clip((e-5)/15,0,1)。
e<=5 为零，e=6 为 1/15，e>=20 为 1；显示轮次为 e+1。

仅对最终 decoder 普通查询原 Hungarian 匹配选出的 **CBR 后框**，将原归一化 cxcywh
转成 xyxy。自身与邻居 GT 均来自本次真实增强 batch，按 GT 身份排除自身，同图所有 GT
均可作邻居，包括没有被查询匹配到的 GT。框不额外裁剪、排序或加宽。

H 的左/上边用 `where(B.detach()<G, B, G)`；右/下边用严格 `>`。
相等或偏内时为 GT 常量分支。H 去掉 G 的 left/right/top/bottom 四条互不重叠条带，
与邻居求交并相加，除以 `max(area(G_i),area(G_j),1e-9)` 得 E_ij。
相交宽高使用严格 raw_length>0 的正部分，零面积接触梯度为零。

按 `E.detach()` 稳定降序排序，平局按原 GT 索引升序。取 min(3,N_gt-1) 项，
包含零值，不补假 GT。用原 E gather 保留梯度。每匹配先除真实邻居数，再求和除
全部最终普通匹配数 M（包括无作用匹配）；loss_geo = 0.20*r(e)*raw_geo。
GT、分母和索引不带梯度，包络只给偏外边直接梯度；共享网络参数仍可得到间接梯度。
几何在局部关闭 autocast 后使用 FP32。CPU 参考与有限差分使用 FP64。

固定样例 G_i=[0,0,1,1]，G_j=[0.8,0,1.8,1]，B=[0,0.2,1.2,0.8]：
raw=0.2，满日程 weighted=0.04，xyxy raw 梯度=[0,0,1,0]，加权梯度=[0,0,0.2,0]。
继续缩窄已偏内上下边不能降低此项。没有邻居、没有匹配、没有越界返回有限可导零。
enabled=false/ramp=0 直接使用原损失，不做新增包络或邻居排序、不消耗 RNG。
诊断中未计算的 raw/E 字段为 null 并注明 geometry_skipped，不能解读为实际几何全零。

## 实际接线

- `ultralytics/models/rtdetr/geo_loss.py`：继承本仓库原 RTDETRDetectionLoss。
  首次无 postfix 的 `_get_loss` 捕获最终普通层完整 GT 和原匹配，然后立即禁用捕获；
  不给整个父类 forward 传最终匹配。encoder/aux/DN 仍走各自原路径。
  父类 DN/无 DN 补零结束后只加入 `loss_geo`；finally 清理上下文，不保留跨 batch 图。
- `geo_model.py`：GEOTrainer.get_model 先执行原生模型构建和 nc80→nc1 权重加载，
  再做无张量/RNG 的 Python 类专门化；实际 Trainer 重建对象使用可导入 GEODetectionModel。
  原模型 loss 内的 sum(loss.values()) 自然包含一次 GEO，三个原日志项保持原语义。
  每次 loss 同步当前 epoch/训练开关；eval 中关闭 GEO，包括 lazy criterion 和 EMA。
- 训练期仍用原 RTDETRValidator、原 fitness 和 best 保存条件。fitness 权重是
  [0,0,0,1]，即 val mAP50-95；同 fitness 的后续轮会覆盖 best。patience=50。
- final_eval 只记录原训练循环合法停止并保留完整 last/optimizer/scaler/EMA；正式 FP32
  val/test 延后到用户显式执行 `finish`，避免框架默认 final val 无导出后再补跑。
  不扩预算、不改变 best 选择、不自动运行 test。
- `geo_val.py`：独立 corrected_sorted_conf_mask_v1，分数排序后使用同序 conf mask，
  不加 NMS；同时按原 query 索引导出全部 300 项和 GT。

损失本身无新可训练参数或部署 tensor。原未融合 nc1 模型 20,149,765 参数、552 个
state_dict 键；参数名/shape/数量及同权重推理与母版相等，融合前后分别核对。
3 层 decoder、300 普通 queries、CBR 36 点、rho=0.10 保持。

## 原配方与身份

公共初始化 SHA256：
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
直接复用母版 `tools/init_c19_lif_v1.py`，先构造 nc80 母版并语义映射公共权重，
随后真实 Trainer 按原 RNG 构造 nc1。只允许原九个分类 tensor 的维度适配，
其余 543/552 tensor 精确加载；不使用任何训练 best/last/smoke 作为正式初始化。

权威完整配方来自 `docs/c19_lif_v1/c2_args.yaml`，交付完整 `formal_args.yaml` 和
逐字段 `recipe_diff.json`。差异仅为 model/data/project/name/save_dir 实验路径身份。
epochs200、patience50、B16/640、nbs64、seed42、workers8、device0、AdamW、lr0=.0005、
lrf=.01、momentum=.937、weight_decay=.0001、warmup5/.8/.1、cos_lr/AMP/deterministic
及所有原增强均保持。检测框损失保留 VFL、5L1、2GIoU、encoder、auxiliary、DN。
VFL alpha/gamma=.25/1.5；matcher alpha/gamma=.25/2.0、class/bbox/giou=2/5/2。

prepare 首次核对公共源 hash、模块 hash、环境、完整 args 和数据 YAML。
可复用 ARG 已确认的 **数据清单证据**，但不复用其训练状态；清单本身和分 split 指纹
会重新校验。无现存快照时才扫描数据一次。train=6048/45573，val=1728/12840，
test=864/6663（图像/GT）。后续只检查 YAML、已存清单及 split 目录变化标记，
不反复全量读原图/标签。数据须在确认后保持不变；文件原位改动需要用户在 dispatch
前显式 `prepare --refresh-data`，不能假称目录时间戳能发现所有内容改动。

## 可靠性来源与实现范围

参考 `ef9cb7e05e5557f7dd06c95cf2361998a284adc9` 的非算法可靠性模式：
LF 归一化源码身份、JSON class key 规范化、原公共初始化审计、独立模型可导入、
原 AMP 明确通过检查、tmux pipefail/真实 Python 退出码、进程树有界退出、FP32
零张量 warmup、排序后 conf mask 和原图像素导出。没有复制 ARG 损失/模型算法。
直接复用母版初始化工具；GEO 的 snapshot/result-lock/finish/pack 按本次要求实现。

母版共享代码只有两处可靠性修复：AutoBackend warmup 的 empty 改为 zeros（源自参考），
downloads 对已存在本地路径先返回，保留 Windows 用户名中的单引号（本次发现、测试）。
原 lif_down.py/cbr.py LF SHA256 分别为：

```
26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787
```

每轮至多前 4 个真实 micro-batch 记录 M、邻居/m_i、偏外边、raw/weighted、E 摘要、
GT 原重叠、激活统计、L0 分量、ramp 和有限性。按总数/计数汇总，原损失另按图数
加权；明确仅覆盖采样 batch。1e-8/.01 只用于统计。正式训练不额外 forward/backward。
单 GPU device0；DDP 未验证，无新增 world_size/all_reduce 补偿。

## 有界检查和训练后交付

`preflight` 整个独立进程树上限 900 秒、16 个训练 micro-batch；持续输出阶段。
原 B16/640/AMP/AdamW、e20 的原生累积4，保留原 GradScaler 初始 scale 和跳步逻辑。
只允许原 scaler 得到真实、有限、参数发生变化的 optimizer 更新后通过。
没有有效更新则 PENDING，不降低 scale，不关闭 AMP，不自动扩预算/重试训练。
本地小张量 CUDA autocast 和 B2/160 CPU 生命周期检查不能冒充该服务器门槛。
GEO start 不接受 PENDING preflight，预检绑定实际源码、配方、数据和初始化。
GPU0 被其他计算进程占用时明确拒绝，不杀进程、不减正式 batch。

`finish` 核对同一原生 best 的完整 val/test 结果及锁；完整同身份结果直接复用，
完整 metrics+导出+曲线缺锁时核验后补锁。只补必要评估，失败不写成功锁。
首次每 split 评估直接导出原 query 索引、分数/类别/xyxy 框、GT、图像 ID/路径、
尺寸和坐标定义；不按 conf 丢弃导出项。原指标仍在 score>0.001 边界计算。
正式协议 FP32/B16/640/workers0/conf.001/iou.7/max_det300/augment=false/rect=false/seed42。

提供 Precision、Recall、F1、AP50、AP75、mAP50-95、十 IoU AP 和 PR/P/R/F1 曲线数据。
P 为检测精确率，不是含 TN 的 Accuracy。保留各 split 原生最佳 F1 点 P/R；另外只用
val 的最佳 F1 置信度（不低于 .001）锁定共同阈值，离线计算 val/test 的 TP/FP/FN、
P/R/F1（IoU=.5，单类，原匹配）。离线整数计数与原插值 P/R 可能不完全相同，均标注。
不使用 test 调参或选择部署阈值，不为这些汇总重复推理。

正常 finish 交付一个 COMPLETE tar.gz，包含源码快照、Git 身份、公式、修复清单、
完整 args/diff/环境/导入、数据及初始化身份、预检、机制日志、results.csv/训练曲线、
start/end/PID/命令/退出码/恢复历史、正式结果/曲线/全量预测GT、best/last 身份与 epoch、
锁和文件 SHA256 清单。manifest 不自包含自身 hash。默认无原始图片和大权重本体。
`status` 与 `pack` 仅读取已有材料，不发起推理、不重扫数据、不重选 best；缺项为
INCOMPLETE，可打故障包。训练已结束后评估失败只恢复评估，不恢复或重跑训练。

GEO 是研究候选。Repulsion Loss 已研究实例间排斥，DeepACEv2/TNRL 已考虑扣除 GT
原有重叠；这里的候选贡献是包络限制直接梯度边界，以及本项目固定归一化和聚合。
不宣称首次 repulsion、首次允许 GT 重叠或全球首创；本次没有正式训练/AP 收益结论。
