# YOLOv13-L Flash 可配置实验

2026-10-06（Asia/Shanghai）。新分支 `bench/yolov13l-flash-configurable` 从原训练实施提交 `64f6639847a0b6a1c18f8f246ba27e274a4de3ad` 派生。旧 native 工作树、环境、run 和冻结提交继续保留。训练固定 SHA、完整服务器命令和同内容 `.server.sh` 见 [交付文档](YOLOv13L_FLASH_HANDOFF.md)。本机没有启动服务器正式训练。

默认首轮为 `v13l_aug_x13_flash_01`。`default_config.yaml`、`recipe.yaml`、`configs/v13l_aug_x13_flash_01.yaml` 经同一 resolver 展开后相等；与旧 `v13l_aug_x13_01` 展开的差异只有 `train_attention_backend: flash`。完整数值配方在已提交的 preset；200 上限、patience50、physical batch16/nbs64、640、workers8、seed42、deterministic/AMP true、SGD、warmup5、loss/HSV/几何/MixUp/CutMix 数值全部保留。原生 warmup/累积、分组衰减、Nesterov、EMA、未舍入训练 val mAP50–95 和相等时较晚轮规则保持原实现。

| 阶段 | 后端 | 精度与证据 |
|---|---|---|
| 正式训练 forward/backward | resolved Flash | 原生 AMP；作者 Q/K/V `.half()`；观测真正 `flash_attn_func` 的 shape/dtype/calls/gradient flow；显式 deterministic backward |
| 训练内 val | native | 作者 validator 原有 CUDA half/AMP 语义，实际输入 dtype 单独保存；物理 batch 随 run、workers0；不强制 FP32 |
| 模型 CPU 构建、迁移、AMP 参考 | native | 参考模型副本 native FP32/AMP；模型/BN/全部 RNG 保护；不创建原模型 optimizer/EMA/scaler |
| 独立公共 val/test | native | 真 FP32；autocast/TF32 关闭；内部半精度 cast 和 Flash 调用为0；640/batch16/workers0/seed42 及原协议固定 |

作用域异常安全地恢复 `USE_FLASH_ATTN`、确定性标志和阶段。`native_fp32()` 可以在训练 Flash 外层中使用并恢复旧状态。CPU 在严格 Flash 作用域中显式拒绝。native 继续执行作者原来的显式 matmul、max-subtracted exp/sum，不替换为 SDPA/xFormers。作者历史 warning 中的 SDPA 字样不描述实际 native 算术。

`train_attention_backend` 支持 flash/native/auto；`eval_attention_backend` 固定 native。flash 要求 CUDA capability≥8、固定版本导入/ABI、真实 CUDA forward/backward、AMP 和 deterministic true，失败退出。auto 只在 prepare 解析一次，写入 requested/resolved/原因及检查收据；训练、resume、finalize 读取同一冻结结果，不因 OOM 或运行中导入状态改变而切换。配置/CLI、source/patch、环境、后端收据、完整 native args 和原 COCO 身份全部冻结并校验。原候选 YAML 的后续编辑不影响已有 run。克隆只继承配方；新的 auto 候选可在新 run 的准备阶段重新解析，模型仍从官方 COCO 开始。

作者源码仍为 `73289949533efac82bb5f72ec19b746618656bd2`，L/nc1/crack；COCO 资产仍是111819790字节，SHA256 `f95ad5bbf3aa80a3df2a28ff4c623582ded6e02bb43d7cea9063c2b248e316cb`。实测 1562/1568 张量迁移，仅6个类别输出张量改变形状；773个注意力/HyperACE/gate张量保留源值，learned FullPAD gates 不重置。模型 YAML、结构、损失、分配器、CutMix/MotherHSV、data/evaluator/NMS/整数 letterbox 反变换不变。新 patch/source 的真实哈希在 `upstream.lock.json`；patch 仅新增受控确定性参数、严格 CPU guard，保留原 Q/K/V 布局、缩放、dropout0、causal=false。

Flash 固定 [官方 v2.7.3](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3)，源码提交 `89c5a7dd4e6a8644575bd0c04a286f48c42763ec`。服务器 bootstrap 使用经检查有 PyYAML 的 `/root/miniconda3/envs/rtdetr/bin/python`（必须3.10），新 copies venv `.envs/yolov13l-configurable-flash`，不继承系统 site-packages。先从正常 PyPI/镜像安装 pip24.3.1/setuptools75.6.0/wheel0.45.1/typing_extensions4.12.2；再从 cu121 官方索引安装 Torch2.2.2/Vision0.17.2，其他运行依赖和 NumPy1.26.4 走正常索引。没有将整份作者 requirements 安装到母环境。

安装器读取官方 release metadata，按实际 cp310/torch2.2/cu12/Linux x86_64/CXX11 ABI 筛选唯一 wheel；记录 asset URL/id/size、下载 SHA256、package/extension 路径与哈希。该历史 release API 的资产 digest 为 null 时明确说明下载哈希为本机测量，不冒充独立签名。没有匹配 wheel 才检查 nvcc12/g++，在新私有目录从固定源码构建，MAX_JOBS/NVCC_THREADS=2；失败保留状态，不升级 Python/Torch 或回退 native。所有安装、NMS、Flash ABI 和真实确定性 CUDA forward/backward 检查通过后才冻结环境；冻结后不再 pip install。旧 Windows lock 在此分支改为 `SERVER_NOT_RUN` 的要求说明；实际服务器身份在 `.runtime/.../environment_identity.json`，包含 pip freeze 和真实 GPU/driver/Flash 收据。

`flash_checks.py` 是单独进程/SMOKE_ONLY run，默认 **640、physical batch16、真实 train 标签/图像**。它依次执行 operator、AAttn area1/4 输出/输入与参数梯度 parity、完整模型 native AMP 参考、至少3个真实训练批次（保留原生动态 scaler/warmup，要求真实 optimizer 和 EMA 更新；最多16批仍无更新则失败）、一个实际 native 训练 val batch及恢复、释放训练 GPU 状态后的 native FP32 实际 val 图像批次。显式新候选修改 physical batch 时检查该候选本身；默认首轮一直是16，不在 OOM 后减少。

Parity 容差预先固定：输出 rtol=.03/atol=.003，梯度 rtol=.05/atol=.003，考虑 FP16 Q/K/V 量化、归约次序和近零梯度；记录 max absolute error，失败不放宽。只说明小 AAttn 的测量，完整网络/训练曲线/最终精度及逐比特等价均 NOT_VERIFIED。正式 run 保存成功 parity/readiness 收据的哈希，并明确 AAttn 已通过/全网络未验证。

训练和验证各阶段同步 CUDA、重置峰值并记录 peak/current allocated/reserved、mem_get_info、GPU/driver 与并行进程。训练基线没有 profiler 或保存 tensor 的层观察器；公共 FP32 单独有算术/dtype 观察，复杂度仅为已执行 Conv/mm/bmm 的部分 MACs×2，Flash opaque kernel 等排除项明确记录。并行耗时不能作为论文独占速度。Readiness 的子集 val/前向输出不记为正式公共指标，也不评估 test/保存正式 best/last。

readiness 成功收据与代码/完整配方/环境/数据路径及报告哈希绑定，prepare 要求匹配。旧或修改后的收据不能启动新的 Flash 配方。训练 start 仍是 preflight→train→同一 best 独立 FP32 val→公共 CPU评价→summary；test保持not_requested。合法完成后 finalize 使用冻结 native FP32 test，不训练、不安装、不拉新代码。resume 只读原冻结后端/配方/环境/last，完整 FP32 状态、optimizer/scaler/EMA/RNG/早停/增强恢复范围保持原可验证语义，不声称 bitwise replay。

当前本机 Windows/RTX2060（capability7.5）不满足 Flash 环境：真实 Flash kernel、AAttn GPU parity、Linux 私有安装、4090/640/batch16 smoke、真实 tmux/POSIX 信号为 NOT_RUN。实际执行的 native CUDA AMP 参考、CPU 完整模型迁移/反向/精度保护、作用域/冻结/严格拒绝/ABI选择/生命周期回归结果见 [交付证据](evidence/yolov13l_flash_delivery_validation.json)。伪造的 observer 函数仅用于单元测试，不计入真实 Flash 验证。

用户提供的旧服务器失败记录为首轮 forward 的 C3 `torch.cat` OOM，尚无完成 epoch/best/last；17.57GiB本进程、5.89GiB另一实验、约43MiB空闲。native 注意力可能影响此前的总显存占用；本机没有独立核验该服务器记录。新工具不保证 Flash 消除所有 OOM，必须看实际640/batch16检查。旧 run 不 resume、覆盖或删除，其他实验继续并行，本 run/session 的锁不要求 GPU 空闲。

服务器所有耗时安装/build/smoke/train 在 tmux。worker 开启 `/etc/network_turbo`，source 时临时关闭 nounset/errexit后恢复旧 shell 选项；PyPI/Python包和PyTorch域名绕过代理。Git 网络调用使用单次 HTTP/1.1 与最多3次尝试，不改全局配置。stage.py 保存实际 command/tee退出码、日志和中断状态。

默认 pack 是只读审查证据，排除 .pt；`--include-weights` 在合法完成后包含真实 best/last。轻量 archive-config 明确先生成文件，再在独立归档工作树 Git commit/push；活跃训练工作树保持原固定 SHA/clean。大权重、数据、vendor、缓存、venv不进普通Git提交。
