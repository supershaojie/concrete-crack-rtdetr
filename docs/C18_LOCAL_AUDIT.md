# C18 本地审计记录

本记录总结实际执行结果，不包含原始训练日志或权重。审计命令仅使用干净初始化和合成张量，没有读取真实训练/val/test 数据。完整机器报告位于本 worktree 的 `outputs/gsdr_aifi_v3/development/final_audit.json`（Git 忽略）。代码和配置 hash 已在文档生成时与该报告再次核对一致。

| 项目 | 实际结果 |
| --- | --- |
| 环境 | Python 3.9.25；2.7.1+cu118；NVIDIA GeForce RTX 2060 |
| CPU FP32 | 模块前后向、完整 640 模型嵌套输出精确相等 |
| CUDA FP32 / CUDA AMP FP16 / 显式 CUDA half | 模块前后向均通过；零输出和打开输出投影的副本均检查，五个新投影梯度有限，打开后均非零 |
| 实际敏感运算 dtype | hook/拦截记录 LayerNorm、atanh、tanh、softmax、grid_sample、聚合均为 FP32 |
| 整模型 CUDA half | nc=1/80 的 640×640 前向均通过 |
| 坐标与边缘 | CPU/CUDA：20×20、17×23、1×7、7×1、1×1，raw=0/±1e6 均有限、xy 正确、每轴局部且有效；常量 V 边缘不损失幅度（容差 3e-6） |
| 非恒定内容 | 独立像素空间双线性插值 oracle 与实际聚合一致；batch/head 不串位，不同查询输出不同 |
| AIFI | pre/post norm、train/eval、dropout RNG 及打开辅助分支后的残差接入顺序通过 |
| 初始化存储 | epoch=-1；optimizer/EMA/updates/scaler/best_fitness/训练指标均为空；FP32 保存 |
| 重载 | 原始 checkpoint、RTDETR(checkpoint)、RTDETR(YAML).load(checkpoint) 全部 539 状态精确；640 完整重载输出一致 |
| 保护 | 886 个已有 C16 文件内容一致；仅两个旧注册文件增加 V3 |

| 类别数 | 共同状态 | 分类相关状态 | 总状态 | unfused 参数 | 完整输出张量数 | 真实 train API |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 533 精确 | 9 精确（含 5 个权重及 4 个偏置） | 539 | 20,170,836 | 5，逐张量 torch.equal | 539 状态、109 args、CPU RNG 均精确；未执行数据/优化步 |
| 80 | 533 精确 | 9 精确（含 5 个权重及 4 个偏置） | 539 | 20,272,272 | 5，逐张量 torch.equal | 539 状态、109 args、CPU RNG 均精确；未执行数据/优化步 |

新旧构造在加载权重前已比较共同状态和 CPU RNG；随后比较实际 trainer get_model 构造/加载，防止已加载权重掩盖 RNG 漂移。nc=1 的分类层是 C2 的相同随机初始化顺序，并非用不匹配 nc=80 的权重强行覆盖。

新增状态为五个权重和一个不可训练 anchors buffer。optimizer 实际组数：

| 组 | C2 | C18 |
| --- | --- | --- |
| weight | 117 | 122 |
| bn | 81 | 81 |
| bias | 128 | 128 |

所有共同参数分组相同，全部 331 个参数张量无遗漏、无重复。五个新增权重总计 88,064，均在普通 weight 组，weight_decay=0.0001，warmup 起始 lr=0；未进入 bias 组。原始 bias 组规则及 warmup_bias_lr=0.1 不变。

合成 AdamW 4 步审计：第 0 步仅 output_proj 有非零梯度，上游四个投影梯度为零，符合零边界设计；第 1 步起五个投影均获得有限非零梯度。正式初始化保持 output_proj 全零。

| 测试文件 | 实跑 | 失败/错误/跳过 |
| --- | --- | --- |
| `test_gsdr_aifi.py` | 17 | 0/0/0 |
| `test_gsdr_aifi_v2.py` | 5 | 0/0/0 |
| `test_gsdr_aifi_v2_tools.py` | 8 | 0/0/0 |
| `test_gsdr_aifi_v3.py` | 8 | 0/0/0 |
| `test_gsdr_aifi_v3_tools.py` | 11 | 0/0/0 |

合计 49 项通过（包含 V1/AIFI、V2 和 V3 回归）。首轮 CUDA 坐标 oracle 对照存在 1.1920929e-7（一个 FP32 ULP）差异，最终允许 2e-7，未改坐标设计；边缘常量保持另外检查。

CLI 帮助信息已实际运行；Bash 启动文件经 `bash -n` 通过。负向 CLI 检查实际证实：本机用 `--require-torch 2.1.2` 会失败；本机 2.7.1 的 passing audit 无法生成正式服务器启动计划；`--project` 被 argparse 拒绝。缺字段、非关键字段/类型漂移、已有 run、重复原子 claim、实际 trainer 状态变化、bootstrap 失败均有测试。

未运行：服务器 Python 3.10 / PyTorch 2.1.2+cu121 / RTX 4090 的实际兼容性审计、Linux tmux 派发、真实数据与首批 preflight、正式训练和 val/test、未来已训练 C18 的 128 图诊断。服务器命令要求原环境原配方执行；失败即停，不自动改 batch/workers/AMP。

Git 提交只包含本次源代码、测试、启动工具与说明文档。初始化权重、数据、原始日志及本地机器报告均保持在被忽略的目录中；具体最终提交和远端一致性以交付消息记录为准。
