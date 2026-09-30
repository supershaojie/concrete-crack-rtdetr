# GPC v1：RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR

母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`；分支：`exp-rtdetr-r18-lite-gpc-v1`。
仓库：<https://github.com/supershaojie/concrete-crack-rtdetr>。
正式训练由用户独立执行 `start`，开发交付不启动 200 轮训练、搜索或消融。
发布后的完整复制命令在 `server_commands.md`；它由功能提交之后的文档提交补入真实 SHA。

## 实现与不变项

`ultralytics.models.rtdetr.gpc.GPCDetectionModel` 是可保存、可导入的顶层子类，继承原 tensor predict/export；原 criterion 是 `RTDETRDetectionLoss(nc=1,use_vfl=True)`。没有模型包装层、重复注册、hook、跨 batch 特征缓存、教师模型或新增可训练参数。训练态 unfused 参数仍为 **20,149,765**，原融合逻辑的参数量为 **19,944,965**。

母版的九个指定核心文件均逐字节按 LF 归一核对未改，见 `source_audit.json`。CBR 的 36 点采样、rho=0.10、detach 范围、tanh 位移，LIF 的 Haar lifting、零初始化，三层 decoder、300 query、主路 DN 全部保留。主 loss 的 VFL alpha=0.25/gamma=1.5，增益 class/bbox/giou=1/5/2；matcher cost=2/5/2，matcher 自身 gamma=2.0 沿用原值。

`loss()` 先直接调用母版完整损失，再加一次 `0.10*r(e)*L_GPC`。新增量不进入父类 loss 字典，避免其通用 DN 分支复制；原三项日志和训练 val 选择 best 的规则不变。`enabled=false`、lambda=0、e<=5、eval 或 `loss(preds=...)` 跳过新增视图/RNG；训练缺失显式 epoch 上下文时报错。只有 GPC 的预检上下文设为 e=20，原训练器的 epoch、warmup、学习率、累积和 EMA 时机不改。

启用后从主 batch 均匀无放回选 K=min(4,B)，拼接 `[K 张原图重放, K 张平移图]`。B16 通常额外处理 B8，相当于主 batch 图像数量的 50%，**不是**实测时间/显存增幅。网络、decoder 保持 training；仅本次辅助前向的 BN 使用主路更新后的 running statistics，finally 恢复所有 BN training 标志。affine、两侧框和网络参数均可反传；batch=None 禁用辅助 DN，辅助不计算检测损失。

主预处理后的 `[0,1]` 图像按整数 ±8 像素复制，空白填充 114/255；无插值、环绕或再次增强。原 GT 不改。辅助标签以 `(源图 batch 索引, 该图 GT 原序号)` 为身份；部分裁切 GT 仍参与匹配，完全移出 GT 从位移侧匹配标签删除。只监督双侧完整可见且双侧各有唯一 Hungarian 匹配的同身份 GT。预测不裁剪；GT 尺度下界从实际 W/H 取 1/W、1/H，FP32 Huber 均值分母是 4M。空配对返回同设备零，非有限值定位报错。

辅助选择使用局部 CPU `torch.Generator`。种子是 UTF-8 JSON 数组 `["gpc_v1", seed, epoch, batch_index, 有序 im_file 字符串列表]`（无空格、保留 Unicode）的 SHA256 前 8 字节、大端、截为 63 位。辅助 forward 使用 scoped fork_rng 恢复 CPU/本 GPU RNG。epoch checkpoint 恢复后按真实 epoch/batch_index 重建；不宣称任意 microbatch 或跨不同 torch 版本的精确重放。

## 配置、初始化与来源

- `algorithm_config.yaml`：固定首版全部字段；正式 prepare 拒绝关闭或缺字段配置。模型 API 的关闭模式仅用于工程对照。
- `resolved_formal_config.yaml`：母版完整训练配方，只改 model/project/name/save_dir 为 GPC 身份；data 保持主数据 YAML。box/cls/dfl 通用字段原样保留，不覆盖 RT-DETR criterion。
- prepare 将真实母版 `runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml` 与母版归档逐字段、逐类型比较。仅在比较副本归一文档指定的两个 c19 model 精确别名，未知 model 路径仍拒绝。
- 复用母版 `init_c19_lif_v1.initialize`、`build_training_model` 的严格公共源映射与 nc80→nc1 转换。公共 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，初值写本实验 `weights/gpc_v1_controlled_init.pt`；从不使用母版 best 或预检权重初始化正式训练。
- 数据检查 train=6048/45573 GT、val=1728/12840、test=864/6663、nc=1 crack，核对解析路径/列表/标签；首次计算内容指纹，不解码全量图像。已审计清单的相对路径、大小、mtime_ns 及 YAML 完全相同时复用内容审计；变更后重新核对。文件时间/大小的缓存假设明示于此，不声称抵抗故意保留这些属性的文件篡改。
- 功能摘要覆盖 Ultralytics Python/YAML、新实验工具、用到的原初始化/数据 helper 和两份配置，文本按 LF 归一。代码/配置/运行环境版本/初值/数据/run/output 身份变化会使旧预检失效，单纯 README/命令文档更改不失效。

未导入任何 DTR/GNR/GEO/QCC 等旧损失。母版工具中只复用以上读写与初始化函数，不调用旧 start/preflight/pack 调用链。独立评估的排序后掩码公式来自母版 `c19_lif_v1_results.postprocess`，新增 query index 与离线统计导出。

唯一母版通用改动是 `AutoBackend.warmup` 的 `torch.empty` 改为同 shape/dtype/device 的 `torch.zeros`：源码原先给网络传未初始化内存。测试实际调用 warmup 检查有限输入，正式预测非有限检查仍保留。未更改主路网络输出、匹配或训练验证协议。

## 已执行验证及边界

详细机器记录：`local_cpu_checks.json`、`local_cuda_checks.json`、`local_ops_checks.json`、`local_pipeline_checks.json`。
开发机实际环境为 Python 3.9.25、torch 2.7.1+cu118、RTX 2060 6 GiB；没有升级环境。历史服务器 Python3.10.13/torch2.1.2+cu121/4090 尚须现场探测。

| 检查 | 证据与范围 |
|---|---|
| CPU 核心/真实模型 | 八个用例中七个通过，CUDA 专项单独跳过。固定单线程归约后，关闭路径的各主 loss/输出/匹配/梯度/AdamW 更新为零差异；active 主 loss 仍相同 |
| 小规模 CUDA | 八个用例通过，B2×160×192 主路、B4 辅助；零位移为零；参数、推理/导出输出契约、BN/RNG、序列化检查 |
| AMP | 默认 GradScaler，最多八个合成 microbatch，记录真实跳步与 scale；成功更新必须按 AdamW step 计数，跳步不计通过 |
| Trainer 集成 | 合成 CPU B2/160，真实 Trainer 跑完整第 1 轮并保存，中断后从有效 checkpoint 恢复第 2 轮；optimizer/scaler/EMA、普通验证及 final_eval 通过 |
| 独立评估导出 | 对两张合成图实际运行 FP32/640 验证器，保存 600 个 query、GT、十 IoU 的 AP/PR 输入和曲线；不是精度证据 |
| 运维 | 十个离线故障/契约检查：配方类型及别名、评估掩码/query 对齐、有限 warmup、功能摘要、缓存产物身份、tmux 脚本/退出码、PID 身份、预算超时不冒用旧证据、已完成训练补评估、离线不完整打包；Bash 语法通过 |
| 服务器 | 真实公共初值/数据/母版 best、B16 主路+B8 辅助、正式 AMP/AdamW/nbs64 预检、Linux tmux 实际运行与真实 val/test 均 PENDING |

CPU 初始四线程严格比较曾出现约 5.6e-9 的反向归约差异，改用单线程 reference 后得到严格零差异。CUDA `grid_sample` backward 的非确定性在母版自身也存在：最初的 1e-6 绝对/1e-5 相对逐元素标准失败，**未将其说成严格等价通过**。后续另做母版对母版重复对照；声明使用最多三倍母版重复最大误差的独立数值波动检查，同时保存原容差、实际梯度与一次 optimizer 参数差值。CPU 严格验证与 CUDA 波动验证的结论分开。不能据此要求两个开启后的长期训练轨迹一致。

本机 Windows 用户目录含单引号，母版 checkpoint loader 会剥除路径引号。实际 Trainer fixture 改放不含引号的 `D:/Temp` 后验证成功；服务器固定 `/root/...` 路径不受此问题影响，没有借此重构母版 loader。

## 服务器动作

| 动作 | 行为 |
|---|---|
| prepare | 验证真实版本、全配方、数据、公共源；严格初始化与 nc1 加载审计；不训练 |
| diagnose | 隔离母版 best；至多 32 固定 val 图、4 个原增强 B16 train batch、900 秒；val 8 像素轴向/对角及可选 16 像素参照，train 主路与 B8 BN-eval 辅助、首 batch 关键梯度量级；无 optimizer step、不用 test |
| preflight | 隔离子进程；总 900 秒内完成算法集成与最多 16 真实训练 microbatch；原 B16/640/AMP/AdamW/nbs64、原 warmup/累积；只将 GPC 的 epoch 置 20 |
| status | 只读显示报告、真实 PID/start token/cmd/cwd/run、shell Python/tee 退出码、最近训练指标；不把保留 pane 当作训练中 |
| start | 要求同功能身份 TECHNICAL_PASS；显示诊断结论；从公共受控初值 epoch0 开始，只在独立 tmux 运行 |
| resume | 仅同身份未完成 run 的有效 last，要求完整 optimizer/scaler/EMA/epoch；已完成/strip checkpoint 拒绝 |
| val / test | 锁定训练 val 选出的 best 的路径/epoch/SHA256；独立 FP32；同身份且全部产物校验成功则 REUSED |
| finish | 补独立 val/test 并显示指标；不打包 |
| pack | 用户单独执行；纯离线收集已有资料，缺项标 INCOMPLETE，校验归档与 SHA256；不评估、不下载 |

preflight 的 TECHNICAL_PASS 要求：必要工程检查通过、B8 辅助实际执行、至少一次相同 microbatch 的有效 AdamW 更新、关键梯度有限且参数确实变化。完美对应或零合格 GT 可使 GPC 为零，不用伪 GT/修改初值来强行激活；不足时 PENDING。真实实现错误 FAIL，实际 OOM RESOURCE_ERROR。记录更新数、跳步、scale、raw/weighted GPC、时间和峰值 allocated/reserved 显存。watchdog 只终止本次创建的私有进程组，不碰其他实验。

所有动作允许共享 GPU0，不查空卡/显存阈值/利用率、不占全局 GPU 锁。仅本实验输出的文件锁和 pending dispatch 防止重复写入。主训练器 OOM 自动减 batch 已在回调作用域禁用。共享允许性不保证显存足够或互不减速。

tmux 仅对本实验窗口设置 remain-on-exit；重试/恢复在同 session 新开窗口，保留旧页。shell 分别记录 Python 与 tee 退出码。训练完成 marker 在 native final_eval 之前写出：若只是最后评估失败，补 val/test，不能重训整个实验。

已完成训练另保存 training_source_sha256（模型/损失/Trainer/初始化/配方等，排除仅独立评估使用的 AutoBackend 及入口/诊断/检查工具）。只修复这些评估/工具代码时，val/test 保留原训练与 best 身份，并使用当前 evaluation_functional_sha256 重新判定评估缓存；不重新训练。模型/损失/配方/初值/数据变化仍拒绝冒用原身份。start/resume 一律要求完整功能身份匹配，不适用此补评估例外。

## 指标、归档与研究解释

独立评估固定 FP32、640、B16、workers0、conf0.001、IoU0.7、max_det300、augment=false、rect=false、seed42。掩码在 confidence 排序后的同一行计算；GT/预测都按 RT-DETR stretch 在同输入像素坐标计算指标，输出记录映射回原图连续 xyxy，并保留原始 query 索引和正式指标使用标志。保存未四舍五入的全 300 query、全部 GT、十阈值 TP/score/class/target 数组、AP/PR 曲线和各类哈希。

P/R/F1 明示母版脚本的“各模型原生最大平滑 F1 报告点”口径及实际 threshold；它与后续仅允许在 val 上选部署置信度阈值是不同用途。test 只报告既定协议的结果，不用来挑 checkpoint、lambda 或视图。母版历史 val mAP50–95=52.454272%，test=52.200902%；test P=86.0239%、R=83.5359%、F1=84.7617%、AP50=89.1997%、AP75=54.0024%。增减以百分点计算。

pack 不包含原始数据图片或权重本体，权重留服务器；记录 best/last 路径与 SHA256。已有评估产物直接入包，不为了补包推理；相同资料重用同一归档文件，缺必要诊断/预检/评估或身份不符时标 INCOMPLETE。

GPC 是有监督的跨视图定位正则。相位尺度参考 LIF 的 stride8/Haar 分组，但整个网络和上下文也受到平移影响；不证明 LIF 是唯一来源或全网严格等变。一致地预测错误仍可得到零损失，过强正则可能压制细节，因此只看真实 AP 能判断效果。lambda=0.10 未经性能优化，当前没有真实涨点证据。完整诊断若无作用会明确不建议长训；非零或很弱只报告量级/覆盖率，不自动改算法。
