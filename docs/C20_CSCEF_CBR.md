# C20：C2 RT-DETR-R18-Lite + CSCEF + CBR

本轮只增加两模块同时启用的 C20。正式训练由用户在 AutoDL 手动启动；本地不启动 200 轮训练。C20 尚无正式性能结果，不能据零初始化等价或 smoke 推断涨点。

## 来源与隔离

- 原工作区 `D:/MyProjects/Crack_RTDETR` 保留在 CSCEF v4；已有未跟踪文件、其他 worktree、权重和实验结果不改动。
- C20 分支 `exp-rtdetr-r18-lite-cscef-cbr`，本地 worktree `D:/MyProjects/Crack_RTDETR/outputs/worktrees/cscef-cbr`，以 C19 的 `025997e3c51eaf6933534308a95da6ebf97bff53` 为基础。
- C2 公共基础为 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。C17 训练包中的 source_commit、干净 git_status、audit.runtime 共同确认训练提交 `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`；C19 训练结果包 source/git.json 确认上述 C19 提交。
- 从 C17 精确复制 `cscef_v51.py`、其必需父类 `cscef_v5.py` 及原 YAML。比较包内源码和提交时仅标准化 LF/CRLF，内容完全相同。只注册 CSCEFv51；父类文件是其运行依赖，不是额外创新模块。
- C19 的 `cbr.py`、显式 final-query 接口所在 `transformer.py` 原样保留。原 head、loss、DN、trainer、在线增强及默认配置由哈希清单保护。没有改两个模块的算法、超参数、损失、插入位置，也没有新增缩框对照或内部消融。
- 仓库及上级未找到 AGENTS.md；读取了 README、C17/C19 实施说明及实际源码。原 README 的数据准备阶段描述已过时，实施依据实际源码、训练包和 C2 配方。
- 附件中的 Linux 路径是服务器路径。本地 C17/C19 实际在 `outputs/worktrees/cscef-v51`、`outputs/worktrees/cbr`；本地权威 C2 配方副本还与 C17 结果包的 `baseline_c2/training/args.yaml` 逐值逐类型核对。
- 两个 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/` 参考路径均存在。本轮只检查其可访问性，未重新读取或移植模块包算法；优先使用已训练 C17/C19 实现。详见 `c20_source_trace.json`。

## 组合连接与初始化

组合 YAML：`ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-cbr.yaml`。

| 部位 | 实际连接 |
|---|---|
| CSCEF | 第 18 层，输入 `[17,16]`，即投影 S3 lateral 与上采样 P4 semantic |
| 原 Concat | 第 19 层，输入 `[16,18]`，保持 `[S,F_out]` 顺序 |
| 最终 Neck P3 | 第 20 层原 RepC3 输出 |
| Decoder / CBR | 第 27 层，输入 `[20,23,26]`，对应 P3/8、P4/16、P5/32 |

CBR 原实现读取 decoder 输入列表第 0 项。审计按实际 YAML 的连接推导层号、比较 C2 图结构，并在临时观测中确认传入 CBR 的张量对象就是最终 P3，且不是 CSCEF lateral；观测后立即移除测试 hook。生产 forward 不依赖 hook 或 query 缓存。

36 点采样、有符号 inside-outside evidence、SiLU 后打分、每边三位置 Softmax、rho=0.1、detach 范围、零初始化和末层常规/DN框处理沿用 C19。AIFI、三层 decoder、300 常规 query、分类分数与原损失不变。默认 `return_final_query=False` 的二元组返回保持兼容。

原始初始化 SHA256 固定为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。拒绝任何 C17/C19 best.pt、smoke checkpoint 或 SHA 不同的源。C2 源为 nc=80；原 half 存储值精确提升为 FP32。

映射按连接图生成：原 0–17 层不动，原 18–26 层映射至组合 19–27 层。严格加载 533 个公共状态，检查目标无遗漏、无重复、无意外状态；新增 CSCEF 7 个状态及 CBR 14 个状态，共 554 个。三个重载入口逐项比较值、形状、dtype。不能用 `strict=False` 或“Transferred”打印代替核对。

正式 nc=1 由原 RTDETRTrainer 按 C2 seed42 重建，9 个分类状态与同流程 C2 逐位一致。新增参数使用模块原有 fork_rng/zero-init，初始化工具也保护调用方 CPU/CUDA RNG；审计单独检查新建公共状态、构造后 RNG、原始权重映射、新增 nc=80 参数和 train API 重建。

nc=1 参数量：C2 20,082,772；CSCEF 新增 26,912；CBR 新增 45,889；C20 共 20,155,573。原生 AdamW 分组：81 个无衰减 norm 参数张量、130 个衰减 weight 参数张量、134 个 bias 参数张量；新增 19 个参数张量全部覆盖，无遗漏或重复。公共参数分组经层号映射后与 C2 一致。

## 配方与启动保护

服务器唯一权威文件：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`。读取完整 109 字段并逐值逐类型核对；仅 model/name/save_dir 改动，project 保持 C2 原值。200 epochs、batch16、640、seed42、workers8、AdamW lr0=.0005、weight_decay=.0001、warmup5、cos_lr、AMP 和所有在线增强全部保留。

`tools/autodl_c20.sh prepare` 只初始化、审计、少量 smoke 和生成计划。使用现有 rtdetr 环境，强制 PyTorch2.1.2、CUDA、正确工作树导入、完整配方、源码保护、nc=1/80 精确初始化、FP32/AMP/half、梯度、重载、optimizer 以及真实数据 smoke。任何失败有具体异常及 bootstrap/console 日志；没有跳过或放宽等价性条件的选项。

`start` 重新绑定提交、代码哈希、环境、数据配置、C2args、初始化及 audit SHA。没有服务器通过的 audit 不能启动。本轮只允许新初始化；训练前回调再次检查实际 109 args、全部模型状态、空 optimizer、EMA updates=0 且 EMA 全部状态等于模型。smoke 后重读正式初始化并比较 SHA 和全部状态。

Smoke 沿用原 trainer 构建、真实在线增强和 DN，独立目录仅进行三次优化。batch2、workers0、调试输出、关闭 val/plots/save 及 GradScaler 初始 scale128 都明确记录为 smoke 差异；正式训练保持 C2 原生 scaler 和完整配置。smoke checkpoint epoch=0，不可作为正式初始化。

启动用跨进程独占 `.c20.launch.lock` 防止重名训练；已有结果、tmux、进程或退出记录均阻止重复 start。失败锁/日志保留，不自动清理。训练 console 实时镜像到 tmux；子进程结束后保留最后输出，等待 Enter，再打开交互 shell。`exit_code.json` 与 `process_exit_code.json` 保留真实训练/子进程退出码，bootstrap 覆盖早期导入错误。未生成退出码不能当作成功。

## 评估与导出

val/test 只评估 C20 best.pt，不重跑前三组。两个 split 都显式使用 640、FP32、batch16、conf=.001、iou=.7、max_det300、无增强、原 RTDETRValidator，口径沿用 C19。test 要求已有独立 val 且 checkpoint/data SHA 与关键评估参数一致，不使用训练末尾 val 充当 test。

报告保存 P、R、mAP50、AP75、mAP50–95、十个 IoU 的逐类 AP、权重 SHA、未融合参数量、请求及实际评估 args、数据配置、解析数据路径、环境/提交/代码哈希和速度。保留固定原生评估顺序的前32张图的全精度预测/GT、图像/标签 SHA；所有图像都参与官方指标计算，32张仅限制证据包体积。

pack 白名单保留训练 CSV/args、完整审计、实际 args、初始化/preflight/退出记录、独立 val/test 指标、关键预测/GT、必要源码和来源记录；日志仅尾64KB。排除权重、图片、整个仓库与旧实验，压缩后超过20 MiB 会报错并列出最大成员，不静默丢弃证据。

## 验证边界与主消融表

本地验证摘要见 `c20_local_validation.json`。本地运行环境为 Windows、Python3.9.25、PyTorch2.7.1+cu118、RTX2060 6GB，不能写成服务器2.1.2已通过。服务器版本验证、Linux/tmux实际运行、完整服务器数据 smoke、正式200轮训练及正式 val/test 均由用户在 AutoDL 执行。

| 实验 | CSCEF | CBR | 已有 test mAP50–95 |
|---|---|---|---|
| C2 | 无 | 无 | 0.46963623190802783 |
| C17 | 有 | 无 | 0.5120450850441516 |
| C19 | 无 | 有 | 0.5038924194554971 |
| C20 | 有 | 有 | 待正式训练和独立 test |

前三组指标为用户提供的已核对结果，本轮未重算。具体服务器操作见 `C20_AUTODL.md`。
