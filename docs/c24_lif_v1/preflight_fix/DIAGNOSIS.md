# C24 + LIF v1：预检修复与证据边界

本次交付修复工程验收，不表示模型已获训练放行。服务端尚未执行新版本；不得启动正式训练或完整 val/test。本次以本文替代旧 AUTODL 文档中的下一步操作。

## 原故障包审计

输入 `c24_lif_v1_LIGHT_20260913_133412_448272.tar.gz`：260,391 字节；SHA256 `2acde183614281cce7549fa7374c6124fc1b5d64547bf0e06841df495123d156`。独立检查 52 个 MANIFEST 条目的名称、大小、SHA256，全部匹配。仅读取 JSON/文本，不执行包内代码、不反序列化权重。证据来自 `metadata/preflight/`，没有把 `source_docs/checks.json` 当作服务器结果；包内结构化摘要缺失的字段不补猜。

原服务端提交 `e314860a45abebee226935fc1dfdc87bff98e005`；Python 3.10.13、torch 2.1.2+cu121、RTX 4090、既有 rtdetr 环境。本机是 Python 3.9.25、torch 2.7.1+cu118、RTX 2060；没有升级服务器或连接 AutoDL。

两个独立阻塞必须同时保留：

| 原服务端阶段 | 已有证据 | 结论 |
| --- | --- | --- |
| fusion_cuda_amp | 640² 下 250 个位置不一致、5 个候选替换；连续最大绝对差 0.002888798713684082；A/B 重放通过 | REQUIRES_REVIEW |
| fusion_cuda_half | 640² 下 248 个位置不一致、7 个候选替换；连续最大绝对差 0.0029296875；A/B 重放通过 | REQUIRES_REVIEW |
| server_B16_640 | 旧非零压力副本第一次 AMP 反向报 `Loss/backward nonfinite` | BLOCKED，根因未知 |

AMP 差集：A 独有 `[260,2328,2499,5029,5251]`；B 独有 `[576,985,2279,2513,3170]`。
half 差集：A 独有 `[281,1801,2328,2499,2510,2513,3963]`；B 独有 `[290,1186,1193,1408,1632,3170,4515]`。
CPU FP32 的 640² 候选完全一致；CUDA FP32 只有 4 个位置换序、没有集合替换。160×192 低精度用例为同集合换位。

`first_unaccepted_stage=fusion_cuda_amp`，`terminal_exception_stage=server_B16_640`。原包没有真实 B16 loss 数值、scaled loss、非有限梯度名称、第一次更新或显存结果。真实 loss 是否有限、哪个梯度异常、最早出错算子、是否仅 scale 校准，均为 **UNKNOWN/UNPROVEN**。不能因最后一条异常被修复而宣称候选问题也通过。

## 实际代码问题与修复

旧 `loss_checks()` 对所有副本调用 activate，把 SCCA.scca_o 与 LIF.O_proj 的零初始化改为 N(0,0.01)，容量阶段只尝试一次；在 unscale 后、scaler.step/update 前抛异常。旧小尺寸 smoke 则先 FP32 更新两次，再在同一个副本上用 scale=128 更新两次。因此旧 smoke 是经过 FP32 更新的压力测试，不是原生冷启动。

现在分别执行并命名：

- `native_initialization_B16_640`：真实公共权重与父模块初值映射，原生 Trainer.get_model 重建 nc=1 后的干净副本；先验证两个输出投影为零，不调用 activate，不继承其他测试的权重、BN、optimizer 或 scaler。
- `nonzero_branch_stress_B16_640`：独立激活的原压力副本，仍要求全部 10 个新增参数及相关主路得到非零梯度。
- 小尺寸 CPU FP32、CUDA 原始初始化冷 AMP、非零冷 AMP，以及单独命名的 FP32-warmed AMP。只有最后一种沿用旧 smoke 的 128 起点。

原始零初始化允许上游支路第一步梯度为零，仍检查梯度存在和主路梯度；没有改梯度公式。源模型前后参数、buffers、输出投影摘要必须完全相同。

每个 AMP 冷检查使用原生 GradScaler 默认 65536，固定最多 12 次尝试，要求至少两次连续真实 optimizer.step。每次重新 forward/backward、清梯度、只 unscale 一次。有限前向但非有限梯度时，让 scaler.step 跳过底层 optimizer，scaler.update 原生回退；核对参数及 optimizer 状态未变、实际更新次数为零和 backoff 正确。有限梯度才按原规则 clip=10 后更新；更新后的参数、buffers、optimizer 必须有限。NaN/Inf 前向立即阻塞；耗尽预算仍失败即阻塞，不扩大次数、不手改 scale/found_inf、不替换坏梯度。

通过受控 optimizer.step 包装计数，不根据返回 None 判断更新；finally 恢复包装。使用 torch 2.1.2 已有的公开 CUDA GradScaler API，并共同保存/重载 model、optimizer、scaler 状态。没有复制最新版 GradScaler 实现。[PyTorch v2.1.2 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/cuda/amp/grad_scaler.py) 中的跳步、unscale 与 update 规则已核对；[2.1 AMP 示例](https://docs.pytorch.org/docs/2.1/notes/amp_examples.html) 用于核对裁剪顺序。

若冷 AMP 失败，最多另做一次相同初始参数/BN、同一增强 batch、同一首次前向 Torch RNG 的 FP32 控制。该控制只对应第一次尝试，不能解释后来已演化状态的因果，也不会覆盖 AMP 失败。资源不足保留 OOM 异常和显存记录，不降低 batch、不杀其他进程。

## 原生 Trainer 审计

审计的是此仓库原文件，而非网络上的最新版训练器：

- DetectionTrainer.preprocess_batch 将 tensor 搬到设备，img 转 FP32 后除以 255；原配方 multi_scale 为 0。容量数据仍由 RTDETRTrainer.build_dataset 构造真实 train 增强，只取一个 B16/640 batch，诊断 workers=0，正式 workers=8 不变。
- RTDETRDetectionModel.loss 用原 DN targets、原 decoder/encoder/aux/DN loss，返回所有 loss 项的总和及三个显示项。有限循环继续调用原 model.predict 与 model.loss，并使用 Trainer 相同的 `loss.sum()`；通过临时 criterion hook 记录所有真实 loss 项，异常时也移除 hook。
- _do_train 在 autocast 中预处理/前向/loss，随后 scaler.scale(loss).backward；单卡不乘 world_size。初始 accumulate=4；warmup 随 ni 从 1 调到 nbs/batch，bias lr 与其他 lr 分别插值。
- optimizer_step 顺序为 unscale、clip_grad_norm_(10)、scaler.step、scaler.update、zero_grad、EMA.update。此有限诊断使用同一原生 AdamW 分组和正常 lr=0.0005，每次尝试一次更新机会，明确不模拟完整 warmup/accumulation/EMA 训练轨迹，不是 epoch 或训练性能实验。生产 Trainer、EMA、配方均未改。

## 候选集合的独立验收

生产 top-k、gather、排序与 decoder 不变。FP32 容差仍为 2e-5/2e-4；AMP/half 仍为 .008/.04。记录真实返回且被消费的候选 ID；同集合按 ID 对齐，SET_DRIFT 保留自然输出、截止位证据和 A/B 双重放。

增加原始 SCCA 与原始 LIF 父模型的独立数值审计：原架构、可追踪公共权重/seed-42 初值，在同一环境、输入、精度和压力条件下比较各自未融合/融合状态。记录初始/激活状态摘要与真实 encoder 输入指纹。没有把组合模型前缀复制给 C2 后称作父模型。本地两个父模型也出现低精度漂移，但相似现象不满足严格比较证明，不能放行组合。

fusion、preflight、训练前 require_preflight 共用 fail-closed 详细验证器，核对 schema、全部连续键、原容差、真实候选范围/唯一性、ID 关系及双重放。保留原有 half 的严格父模型证据例外；没有新增 AMP SET_DRIFT 例外。缺少合法父模型可比性证明仍 REQUIRES_REVIEW，未知状态/不完整证据 BLOCKED。合法 warning 的一致性只用明确标记的合成测试验证，不能当作真实父模型证据。

仍需要用户根据新服务端证据决定后续候选验收政策或进一步调查；本提交不会自动改变规则来放行。

## 失败证据与有限验证

每次尝试在关键断言前原子记录输入/目标合法性、图片身份和增强 tensor/labels SHA256、初始参数/buffer dtype/摘要、投影零值、全部 loss、预测有限性、DN/query、scaled loss、缩放前后全部非有限梯度及计数、实际更新数、更新前后状态、CUDA allocated/reserved peak。参数枚举顺序不代表最早故障算子。

attempt/gradient JSON 每条不超过 64,000 字节；超过时使用带路径、字节数和 SHA256 的完整结构化分片。LIGHT 中大 JSON 也采用分片或显式摘要，MANIFEST 逐项验证。只含文本/JSON/源码，不含权重、激活、图片、完整预测或嵌套包，压缩后硬上限 8,000,000 字节。

`blocking_summary.json` 独立保存 first_unaccepted_stage、terminal_exception_stage、所有 unresolved_stages、两种容量结果、amp_calibration、training_dispatched=false；后面失败不覆盖前面候选阻塞。stage 异常保存 traceback 和部分记录；可独立的阶段继续完成。start-direct/preflight-only 的异常路径自动打 LIGHT，仍返回原异常的非零退出；打包失败也保留原证据与原异常。预检锁在 finally 按原 owner token 释放。

本地回归覆盖：真实 CUDA 溢出跳步后更新、前向 NaN/Inf、持续失败预算、漏参/重复参、参数和 optimizer 污染、零初始化禁止 activate、零上游梯度、冷/暖 AMP 分离、完整 loss 项、保存重载三种状态、异常后 hooks/top-k/RNG/backend/dtype/grad-mode 恢复、数值负例、warning 语义、一次预检/派发的 stub 测试及自动失败包。派发测试拦截 tmux，未启动实际训练。

详细结果见同目录 JSON。本地小尺寸合成结果不能替代 RTX4090/torch2.1.2 上的真实 B16/640。服务端新原始容量、压力容量和 AMP 校准均 **NOT_RUN**；完整 val/test 和正式训练均 **NOT_RUN**。

## 不变项

SCCA SHA256 `67b0d347d5305c16d48bb031cd10fd1d23b7ea447b58b2423166eb68b81dedca`；LIF SHA256 `26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`。LIF 残差在 BN 前，原融合排除条件保留。原 nn/tasks、head、transformer、训练器、组合 YAML 和 109 字段 C2 配方均与交付父提交逐字节核对。仍为 20,169,416 参数、543 states（533 公共+10 新增）；原生 nc=1 加载 534/543，9 个类别相关参数按原规则重建。

## 服务器下一步：一次有限诊断

使用最终回复中的新完整 40 位 SHA，经原 `sync_c24_lif_v1.sh SHA MAIN WORKTREE` 创建新的 detached 工作树：
`/root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1-preflightfix`。

先只读检查旧状态、owner/PID/start-time、tmux 与正式结果目录；不删除锁或旧初始化/报告。新工作树报告独立；既有正式 run/live owner 仍阻断。sync 只更新本分支 remote ref，使用 `--no-write-fetch-head` 保护其他任务的共享 FETCH_HEAD；已有不同 SHA/dirty/无关目录均保留并拒绝更新。

只运行新工作树 `bash tools/autodl_c24_lif_v1.sh preflight-only` 一次；不接着运行 start-direct。真实日志是新工作树 `outputs/c24_lif_v1/preflight.log`，可以 `tail -n 80 -F`；此阶段没有训练 console.log。失败自动打印 LIGHT 路径，也可以单独 `bash tools/autodl_c24_lif_v1.sh pack-light`，不会再次预检。包位于主项目 `downloads/c24_lif_v1`，下载日志打印的那一个文件。

结论：原服务端真实 loss/最早故障算子未知，不能断言只有 scale 问题；本地小尺寸原始初始化和非零压力均已获得有限更新，但真实 B16/640 尚待运行；AMP/half 候选审查仍未解决；不能正式派发训练。
