# BMC v1：原 RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR

本实验从 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 独立派生，分支为
`exp-rtdetr-r18-lite-bmc-v1`。没有合并其他候选分支。BMC 是训练期离散标签分配策略，新增可训练参数 **0**。
模型 YAML、LIF-Down、CBR 前向、final_query、36 点采样、rho=0.10 和融合保护均保持母版。

## 实际接入

| 文件 | 职责 |
| --- | --- |
| `ultralytics/models/utils/ops.py` | 原 matcher 的可选 `return_costs=True` 接口，返回实际送入 SciPy 的逐图矩阵；默认调用保持原路径 |
| `ultralytics/models/utils/bmc.py` | 确定性并查集、逐 GT IQR 预算、BMCLoss、BMCDetectionModel 子类 |
| `tools/bmc_v1_training.py` | 原 Trainer 构造对照、真实 epoch 传递、原生恢复、逐轮统计和受限事件 |
| `tools/bmc_v1.py` / `tools/bmc_v1.sh` | prepare / preflight / probe / start / status / resume / val / test / pack |
| `tools/sync_bmc_v1.sh FULL_SHA` | 校验 remote/母版/实验分支及完整 SHA，安全建立或快进独立 worktree |
| `tools/bmc_v1_diagnostics.py` | 隔离容量检查、最多 64 张训练图的只读机会探针 |
| `tools/bmc_v1_results.py` | 原已核验置信度排序筛选、FP32 评估锁定、LIGHT 打包 |

普通 loss stack 由原 `RTDETRDetectionModel.loss` 在拆分 DN 后拼接：
`[encoder, decoder1, decoder2, final_CBR]`，索引为 `[0,1,2,3]`。
仅索引 **2** 可改用 BMC。构造时断言原 head 的 3 层、300 queries、输入 `[19,22,25]`，
训练时断言实际 4 项张量；旧代码中的“7 层”注释不作为依据。

同一次预测的原最终层匹配 A3 仅计算一次并用于原主项。A2 与 C2 从原 matcher 一次返回，
原分类/L1/GIoU 代价及系数 `2/5/2` 和 AMP 数值路径均复用，VFL 没有被误作匹配 cost。
新判断使用这份 CPU C2 的 detached FP32 副本；线性插值 IQR，floor=0.001，预算=0.02*scale。
并集图 query 与 GT 节点不相交，分量内所有 GT 均通过才整体采用 A3；负 delta 允许通过。
无实际改变返回原 A2 对象/顺序，有改变按 query 排序并恢复 batch 全局 GT 偏移。
空图保持原路径；G>Q 整图回退且不影响其他图；重复/越界/错位索引直接报错。

原 matcher 存在非有限 cost 置零的历史行为。本实验做的最小处理是：
默认接口仍保持该行为；显式代价审计接口在已有 CPU 转移后检查并报出非有限矩阵上下文，
不新增修补值，也不全局改为 FP32 matcher。有限输入的代价/匹配与原实现完全一致。

分类 target 继续用该层自己的预测与真实 GT；L1/GIoU 回归真实 GT。
原 12 个 loss key、系数及求和次数不变，DN 只走原 `get_dn_match_indices`。
没有教师、跨 batch 预测缓存、额外 forward、伪标签或新 loss。

## 配方、初始化和恢复

固定研究配置在 `configs/bmc_v1.yaml`，与框架 109 个训练 args 分开保存。
真实零基 epoch 0..19 为 `DISABLED_WARMUP`，epoch 20 起开启；学习率 warmup 仍是原 5 轮。
`enabled=False` 在 criterion 的等价性检查中直接走母版路径；零预算仍可接受同代价替换。
正式生命周期要求研究配置与本版本完全一致，避免把消融或调参混成同一次正式实验。

`parent_args.yaml` 来自用户本地成功母版 `training/args.yaml`，与合同附录 109 字段一致。
prepare 逐字段比较权威文件，再仅替换 model/name/save_dir；显式迁移 project/data 时另记差异。
服务器仍为 epochs=200、patience=50、B16/640、nbs=64、seed42、AdamW、lr0=.0005、
weight_decay=.0001、cosine、AMP、原全部在线增强，所有原参数继续训练。
没有升级依赖，也没有更改优化器、native GradScaler、EMA、clip_grad_norm=10 或 best fitness 策略。
本版本正式入口明确限制单 GPU；没有未经验证的 DDP 支持声明。

受控初值直接复用母版 `init_c19_lif_v1.initialize`，公共未训练源的 SHA256 为：
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
该源 epoch=-1、无 optimizer/EMA/scaler/训练指标，并带同一 ImageNet 骨干初始化。
母版 best 仅用于 probe。正式初值不使用任何训练后 checkpoint。
真实 Trainer.get_model 的 native 对照在隔离 RNG 作用域构建，再逐 key 核对新模型，
没有在核验后补拷头部；仅明确列出的 9 个类别相关 tensor 允许原生 nc80→nc1 构造适配。
nc1 实测 **20,149,765 参数、552 状态项**，名称/shape/value 均与同 SHA 母版重建一致。

原模块按 LF 规范化 SHA256：

```text
lif_down.py  26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
cbr.py       d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787
```

Trainer 在 setup_model 时设定初始/恢复 epoch，在每轮前重新传递真实 epoch。
首次 criterion 构造没有 epoch 时禁止训练，不能默认已开启。resume 复用 native
`check_resume → setup_model → resume_training → _load_checkpoint_state`，恢复 optimizer、scaler、EMA 和 epoch。
母版 checkpoint 保存 EMA 和 FP16 optimizer state 的原生语义保持，不宣称恢复为未量化的原始训练参数。

数据仍是 crack_det，train/val/test 数量 `6048/1728/864`，实例数 `45573/12840/6663`。
`parent_dataset_inventory.json` 固定母版路径和标签清单。prepare 额外计算所有图像字节 SHA，
并要求服务器路径迁移复现已验证本地内容摘要；历史母版没有完整图像字节清单，因此不把这份新摘要冒充历史证据。
不划分新 split、不覆盖数据、不追加增强副本。

## 必要验证与证据

本地环境实际为 Python 3.9.25、torch 2.7.1+cu118、RTX2060 6GB、定制 Ultralytics 8.4.21。
历史服务器 Python3.10.13 / torch2.1.2+cu121 / RTX4090 仅作参考，代码使用 torch2.1 可用接口，
服务器实际结果由 prepare/preflight 记录。没有在本地冒充服务器验证。

* `check_bmc_v1.py`：独立 BFS/手算分位数参考；相同、接受/拒绝环、交替路径、独立分量、负 cost、零 IQR、空 GT、G>Q、GT 偏移和零预算。
* 刻意构造近代价真实预测，通过原 Hungarian/原 loss，实际改动 2 个 GT；只改变普通 slot2，主项、encoder、decoder1、全部 DN 不变；loss/梯度有限。
* CPU/CUDA B2/160 原模型 forward/backward 对照：关闭 loss 一致；公共梯度按 FP32 每 tensor 最大范数误差核对。CUDA grid_sample 的原子累加不保证逐位一致，报告实际误差与阈值。
* `check_bmc_v1_lifecycle.py`：可丢弃两图 fixture 的真实 Trainer/原 AMP 检查/有效 AdamW 更新/原 save_model。用该真实状态构造 epoch19、57 的明确合成元数据 checkpoint，再实际重建并恢复到20、58；逐值核对1035个 optimizer字段和 scaler/EMA。
* 非零 LIF/CBR 权重上，以抛错观察器检查每轮 Validator、独立 Validator、predict 均不调用 BMC。fixture 的 AP 没有性能含义。
* `check_bmc_v1_ops.py`：所有公开子命令解析、Bash 语法、完整 SHA 拒绝、脏 worktree 保留、Python=7/tee=0 退出码传播、严格非有限 JSON、已核验排序筛选口径。

本地 prepare、检查与 probe 原始 JSON 在 `outputs/bmc_v1/`；可审阅副本在本目录 `validation/`。
验证报告中的 Git commit 可能为提交前母版 HEAD，`runtime.bmc_source` 另记实际被测工作树源码哈希，
交付 SHA 由提交完成后的 delivery 记录及远端核对给出。

本次固定前64张 train、无增强、640 FP32 的母版 best 探针结果：174 matched GT，3处分歧，2处接受，
均为原背景 query 转正；changed/matched=1.1494%，分歧中接受=66.6667%。
64图最终 CBR 修正均非零，所有参数和 BN buffer 保持不变。完整名单/哈希、分位数和事件见 probe 报告。
这是机会诊断，不是工程预检 PASS，也不是 AP 提升。没有修改预算或启用时间。

## 服务器操作

固定 worktree：`/root/autodl-tmp/projects/Crack_RTDETR-bmc_v1`。
Python：`/root/miniconda3/envs/rtdetr/bin/python`；tmux：`bmc-v1-training`。
正式 run：主仓库 `runs/c_series/bmc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。
完整可独立复制的、填入真实 SHA 的命令在交付的 `outputs/bmc_v1/server_commands.md`。
首次取得 sync 脚本后必须执行它，让服务器生成自己的 delivery 记录。

顺序是 sync → prepare → preflight → 可选 probe → 用户显式 start。
preflight 整个子进程组有900秒外部看门狗，隔离容量检查最多16个micro-batch，达到2次有效更新即结束，
边界时至少1次且检查全部通过才可 PASS。以 Adam step 计数和参数变化判断实际更新，记录 overflow/scale/相关梯度/显存/P3 shape。
容量诊断可用 `diagnostic_only_scale=128`；正式训练及 resume 保持 native scaler 设置，丢弃全部诊断状态。
本地6GB卡的 B2/160 检查不能作为 B16/640 容量 PASS。服务器 preflight 当前 **PENDING**。

start 只消费本实验有效 preflight，按源码、配置、初值/公共源、数据内容、环境和 AMP 资产身份使其失效，
不会因报告时间戳重复运行检查。另一实验占用 GPU 不构成预先拒绝理由；真实 OOM 原样报出且不降低batch/分辨率/AMP。
prepare 保证原 check_amp 使用的 bus.jpg/yolo26n.pt 可读并记录 SHA，不绕过 AMP 检查。
默认 AutoBackend 及 LIF 融合保护保持母版。

tmux 直接显示 Python 输出，tee 同时记录 `outputs/bmc_v1/console_<UTC>_<dispatch>.log`。
`pipefail` 和立即读取 PIPESTATUS 保留 Python 退出码；每次启动使用独立 dispatch/state/exit，旧 exit 不能覆盖新状态。
status 区分 NOT_STARTED、DISPATCHED、SETTING_UP、RUNNING、COMPLETED、FAILED，
完成真实 batch 后才出现 TRAINING_RUNNING，完成写实际 epoch，早停不会标成 COMPLETED_200。
同实验活跃时拒绝重复启动；不停止其他进程或实验。

`start --archive-failed` 只允许在无本实验进程、无任何 .pt、results.csv无数据行时将失败目录重命名归档。
有有效 last 必须 resume，不删除目录、不覆盖旧证据、不 reset/clean、不强推。

## 日志、评估和 LIGHT

每轮机制汇总保存在 `outputs/bmc_v1/mechanism/epoch_*.json`，包含分歧/替换/分量/背景转正/目标互换、
delta/scale 分位数和原 decoder2 三项损失。每100step记录受限详细事件，每轮另保存首个接受/拒绝代表性事件。
前20轮明确 DISABLED_WARMUP，未计算量和接受率为 null，已知 matched GT 数量单独记录。
记录共识计算、统计/IO 额外耗时及整轮时间，不把“推理图不变”说成训练无成本，也不冒充成对基线计时。

每轮 val 使用母版 CBR 最终输出及原 fitness。独立 val/test 复用母版 `c19_lif_v1_results.postprocess`
的 `corrected_sorted_conf_mask_v1` 实现：FP32/640/B16/workers0/conf.001/iou.7/max_det300/augment=false。
P/R 是 native ap_per_class 在平滑平均 F1 最大位置的统计，不是另选阈值来制造增益。
val 完成后锁定 best SHA、数据/源码/设置；test 必须用户显式调用，同身份已完成结果直接复用。
prepare/preflight/probe/start/pack 不自动执行最终 test。

pack 输出可读取且逐成员核验的 `BMC_v1_LIGHT_<UTC>.tar.gz`，并写 `LATEST_LIGHT.txt` 绝对路径供 FileZilla 查找。
包含源码/criterion/启用配置、Git差异、完整args及diff、初始化和验证报告、权重SHA、数据清单身份、CSV、
日志摘要、机制epoch汇总/事件/代表性事件、独立val/test结果与manifest。默认不含数据集、大型.pt或完整预测流。
服务器完整日志/权重均保留；未做test明确 NOT_RUN。

正式200轮训练、本实验独立val/test及服务器容量验证均未在本轮替用户执行。
历史母版独立val 52.454272%、test 52.20090191444802%仅是同协议参考，不与训练CSV或本地fixture指标混用。
