# CEA v1：原 CBR + 原 LIF-Down 的训练辅助损失

母版为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`，分支为
`exp-rtdetr-r18-lite-cea-v1`，仓库为
https://github.com/supershaojie/concrete-crack-rtdetr.git 。
新分支直接从该母版建立，没有继承其它实验算法。服务器步骤见
[server_commands.md](server_commands.md)，验证证据见 [VALIDATION.md](VALIDATION.md)。

## 实现与保护

- 唯一组合：RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + CEA。
- 新增可训练参数为 0；辅助损失直接梯度仅到原 `model.26.cbr.score.weight` 的 64 个参数。
  未融合参数量为 20,149,765；独立评估实际融合后为 19,944,965。
  没有模块重复注册、`base.*` 前缀、forward hook 或跨 batch 张量缓存。
- `cbr.py` 只增加默认关闭的显式上下文返回，`tasks.py` 只增加一个默认关闭的传递参数。
  原 CBR 的数学运算、36 个采样点、rho=0.10、detach 范围、FP32 边界和零初始化保持。
  `lif_down.py` 完全未改。36 点属于 CBR；LIF 是 Haar lifting 下采样。
- `ultralytics.models.rtdetr.cea` 提供顶层 model/criterion 类，
  `cea_trainer` 提供可序列化、可重建的 Trainer 类。普通模型推理和导出不取 CEA 上下文。
- 禁用、lambda=0、e<=5、普通推理、正式独立 val/test 直接走母版 loss/forward 路径。
  启用后的长期优化轨迹本来就会改变，不保证两条训练轨迹长期相同。

## 固定数学契约

配置独立保存在 [algorithm_config.yaml](algorithm_config.yaml)，不传未知字段给 Ultralytics。
训练调度取 `trainer.epoch`：`r(e)=clip((e-5)/15,0,1)`，总损失为
`L0 + 0.10*r(e)*L_CEA`。0.10 是固定工程起点，未做超参数搜索。

同一次主 forward 显式返回 b0/hidden/p_forward/b1，按与 decoder 相同的 `dn_num_split`
剔除 DN，再取最终普通 query 的原 Hungarian 正匹配。最终匹配只调用一次；
encoder 和两个中间 decoder 仍分别按原逻辑匹配，DN 使用原指定索引。
没有把最终匹配传给公共 `DETRLoss.forward` 从而影响所有辅助层。

学生用 `F.linear(hidden.detach().float(), live_score_weight.float())` 和 softmax。
候选按当前分配、混合位置 1/2/3 排序，eta=0.5。每个候选只换一条边，另外三条边固定为
同一 FP32 小网络算出的基线位移。教师分支全程 no_grad，functional linear 使用 detached
FP32 offset 参数；实际主路框完全保留。

代价是 normalized cxcywh 上 `5*L1 + 2*(1-GIoU)`。逐对 GIoU 使用母版 `bbox_iou` 默认 epsilon，
显式消除 `[N,1]` 维度，禁止两两广播。stable argmin 的第 0 个候选优先。
`A=(E0-E*)/(E0+1e-8)`，`L_CEA=sum(A*KL(t||p))/(4M)`；无 GT、全零改善都返回合法零。
没有有效边归一化、sum(A) 归一化、额外门槛、候选重匹配或随机扰动。
只有正常 FP32 舍入允许将 A 保护到 [0,1]，明显越界/NaN/Inf 直接报错。

母版完整 VFL/5L1/2GIoU、encoder/aux/DN 归约不变，VFL alpha/gamma=0.25/1.5；
matcher 仍为 class/bbox/giou=2/5/2，matcher 自身 gamma=2.0。
CEA 在所有原损失（含 DN）完成之后仅追加一次，不产生 `loss_cea_dn`。
原三项日志、原 val mAP50-95 best 选择、一次 backward、累积、GradScaler、clip、optimizer、EMA 时机保持。

## 初始化、配方、数据

公共未训练源 SHA256 为
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
复用 `init_c19_lif_v1.py` 的 build/topology/key 规则，逐项验证原公共键、19 个原 LIF/CBR 新键、
父模块随机初值和零输出头。原 helper 锁定 CBR 老文件哈希，因此 CEA 在薄适配层重新执行
其受控映射审计，而不禁用该 helper 的原保护。

本实验提前完成受控 nc=80→nc=1 转换，与原 Native Trainer 在 seed=42 的转换逐张量相等，
9 个分类张量是唯一允许的形状适配。最终装载使用 strict=True，不吞任意缺失键。
本实验初值只写自身 `weights/cea_v1_controlled_init.pt`，母版 best 仅用于诊断。

`prepare` 比较母版归档 args 和母版 resolved config 的全部 109 个字段及 Python 类型。
只允许文档给出的两条旧 model 路径精确归一；实验配置仅更改 model/project/name/save_dir。
完整配方见 [resolved_formal_config.yaml](resolved_formal_config.yaml)，正式 epochs/patience=200/50、
B16/640、AMP、AdamW、nbs64、workers8、device0、原增强全部保持。
CPU 合成测试的缩小配置单独标注，不是正式或服务器预检配置。

数据划分已与母版归档的图片相对路径和标签内容 SHA256 核对，计数为
train 6048/45573、val 1728/12840、test 864/6663（图/GT）。
`prepare` 检查实际 YAML、路径、清单和标签指纹；已有文件 size/mtime/清单及 YAML 完全一致时
复用审计，不再读完全部标签。图片身份是路径和文件元数据，不宣称执行了图片内容全量哈希。

## 有界检查与阶段管理

`diagnose` 默认最多 64 张固定 val 图、8 个原增强 B16 train batch、900 秒，不读 test、不更新 optimizer。
两个 split 用独立模型，BN/随机状态/梯度不污染正式训练。
只在两个隔离 train batch 上比较同一 score.weight 的 L0 与加权 CEA 梯度范数、比例和余弦；
零分母记录 null/undefined。保存覆盖率、各分位数、选择频率、原始数组、锚点误差、时间和额外峰值显存。
本地已核实母版 best 为
`24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`，来源见
[mother_evidence.json](mother_evidence.json)。缺失或哈希不符不会冒充已诊断。

`preflight` 默认最多 16 个真实 microbatch、900 秒，保留 B16/640/AMP/AdamW/nbs64、原增强、
原 epoch0 warmup 和累积。仅 CEA 上下文设 e=20。先在独立子进程运行 CPU/CUDA 契约检查，
再跑原 Trainer 真实更新，记录 GradScaler 跳步、scale、有效更新、loss、有限梯度与参数、显存。
随后检查原生 CUDA optimizer/scaler/EMA checkpoint 重建与 epoch 恢复。
隔离预检的 partial-epoch checkpoint 仅验证序列化，不能当正式初值或声称完整 epoch 已完成。
只有必要检查通过且至少一次有效更新才写 `TECHNICAL_PASS`。
超时/缺证据为 PENDING，错误为 FAIL，真实 OOM 为 RESOURCE_ERROR，绝不自动减 B、减 imgsz 或关 AMP。

功能摘要覆盖实际 Python/YAML/shell、算法/完整配方和母版证据，文本换行归一为 LF；纯 Markdown 变更不使预检失效。
正式身份还绑定初值、数据、输出目录与运行环境。start/resume 校验同一身份的有效预检。
start 展示现有诊断结论，缺诊断不假称机制通过；完整无改善明确不建议长训。
用户执行 start 就是运行意图，无二次确认和空卡闸门。

所有操作只使用本 run 范围的文件锁；不查空闲显存、不等空卡、不停止别的 Python/tmux。
原 Trainer 的 OOM 自动减 batch 路径通过每批回调禁用；AMP 检查若关闭 AMP 则停止。
原母版 GPU 独占、自动 pack 的启动 helper 没有进入调用链。

正式训练与独立 val/test 在本实验 tmux 中执行，仅给自己的窗口设置 remain-on-exit。
Python 和 tee 退出码分别落盘，成功/失败都保留页面。status 核对真实 PID、内核启动时间、argv、cwd、run_id；
保留的 tmux pane 不会被算作仍在训练。`attach` 定位最新的本实验窗口。
训练结束保存早停信息、最佳训练 val 及对应 epoch；独立 FP32 test 另行执行。

resume 仅接受同一未完成 run 的完整 checkpoint，使用母版实际支持的 epoch 粒度，
不声称任意 microbatch 精确恢复。完成且剥离 optimizer 的 checkpoint 禁止假续训。
训练后不自动做第二次隐式最终验证；保存原训练 best 选择后，FP32 val/test 通过独立入口完成。

## 评估、复用与打包

独立 FP32 评估固定 imgsz640/B16/workers0/conf0.001/iou0.7/max_det300/seed42，
augment=false、rect=false。复用母版 `corrected_sorted_conf_mask_v1` 的实际修复：
先按 confidence 排序，再在排序后的行上生成置信度 mask。原 query index 与同次预测显式一起导出。
原训练阶段 validator 和 best 选择逻辑保持，不把独立评估修复混入训练协议。

首次完整评估同步保存 metrics.json、全部 300 普通 query 的原始索引和原图像坐标、GT、
用于正式指标的 mask、十 IoU 阈值的 TP/conf/class 数组、AP/PR 曲线数组和各产物 SHA256。
低分导出不会改变指标口径。成功且身份/完整产物一致时显示 REUSED，不重复网络推理。
AutoBackend 原源码 warmup 使用 torch.empty，现仅改为同 shape/dtype/device 的 zeros；
真实预测非有限值检查仍保留，没有 nan_to_num。

P/R/F1 使用与母版归档相同的“各模型原生最大 F1 报告点”口径，数值阈值不必相同；
不会使用 test 来选择训练 checkpoint、lambda、算法或部署阈值。涨跌统一显示百分点。
test/finish 打印指标后结束。finish 当前等价于补必要 test；需要独立 val 时单独执行 val。

只有用户单独执行 pack 才离线打包已有结果；不会触发推理或下载。
默认不含数据图片和权重本体，记录 best/last 路径与哈希；缺项标 INCOMPLETE。
包内文件清单逐项 SHA256，打包后完整读回校验；同样内容复用同一包。
正式训练的任何性能收益都尚未验证，小样本候选改善不等于 AP 涨点。
