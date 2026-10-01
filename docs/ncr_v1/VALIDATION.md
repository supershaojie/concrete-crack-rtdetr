# NCR v1 验证记录（2026-10-01，Asia/Shanghai）

代码交付范围：本地数学/接线/操作测试、最多两步的独立真实批次冒烟、已有母版 val 导出的离线几何调查。正式服务器训练、正式 NCR val/test、完整服务器包、真实 tmux 派发均 **NOT_RUN**。没有连接服务器或升级环境；本地结果不能替代服务器同身份预检。

## 本地实际环境

Windows；`D:/miniconda3/envs/rtdetr/python.exe`；Python 3.9.25、torch 2.7.1+cu118、NumPy 2.0.2、CUDA 11.8、NVIDIA GeForce RTX 2060。服务器预期环境为 Python3.10 / torch2.1.2+cu121 / NumPy1.26.4，待服务器 preflight 实查。

已核验公共初始化 SHA256 正确；直接沿母版初始化函数完成 nc80 严格逐键合并、保存/重新加载和 nc1 原生适配。父模型/CBR/LIF 参数结构、300 queries、3 层 decoder、20,149,765 参数均通过。

数据路径来自现有 `configs/crack.yaml`。train/val/test 图像数与框数全部一致，三个 split 的路径 SHA256、标签清单 SHA256 与母版正式归档**完全相同**，见 `parent_dataset_inventory.json`。没有重分数据，没有 test 推理。

## 测试结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 核心数学、零系数母版逐项 loss/梯度/RNG、aux/encoder/DN 隔离、实际总 loss、CBR 梯度、参数/推理不变、新进程 checkpoint 加载 | 11 项 PASS | evidence/unit_tests.json |
| 完整 109 字段、原生 resume epoch 与 EMA、导出及离线 TP 重放、Python/tee 退出码、PARTIAL 复用、只读 status、缺导出/训练失败保护、manifest JSON 身份复用 | 10 项 PASS | evidence/ops_tests.json |
| 真实 B16/640 FP32，独立临时模型，1 optimizer step | PASS | evidence/smoke.json |
| 同一真实 B16/640 AMP，另一个重新初始化临时模型，1 optimizer step | PASS，检查 scale=128 | evidence/smoke.json |
| 母版公共初始化与逐键加载 | PASS | evidence/initialization.json |
| Python compileall / Bash syntax / Git whitespace | PASS | evidence/delivery_checks.json |
| Linux tmux 实际生命周期、服务器环境与同身份预检 | NOT_RUN | 本机 Windows；未连接服务器 |
| 200 轮正式训练、正式独立 NCR val/test、COMPLETE 分析包 | NOT_RUN | 按任务边界留给服务器显式 start/finish |

真实批次固定取 train 排序前 16 张，35 个匹配正样本，均为工程冒烟而非有效性试验。FP32 loss=91.3257064819336，NCR raw=0.0014691639225929976、加权=0.0003672909806482494；AMP loss=91.37799072265625、raw=0.0014797274488955736、加权=0.0003699318622238934。两者参数与反向均有限。峰值 CUDA allocated 分别 9,609,775,104 / 6,464,080,384 字节，仅为本地临时模型记录，不是双实验并跑容量保证。

## 已遇到并定位的问题

第一次初始化检查遇到母版 Ultralytics 的 checkpoint 路径清理会删除 Windows 用户目录中的单引号。本地临时权重改放主仓库被忽略的独立临时输出目录，仍复用原加载实现；服务器路径不含单引号。没有修改母版加载严格性。

最初 AMP 冒烟使用默认 GradScaler scale=65536，出现非有限梯度，保留 `evidence/initial_amp_failure.json`。随后在同一 B16 AMP 前向图上、**零 optimizer step** 定位：母版 L0×65536 已有 32 个参数梯度非有限；单独 NCR×65536 梯度均有限；总 loss×128 梯度均有限。见 `amp_scaler_diagnosis.json`。

母版 `tools/check_c19_lif_v1.py` 原有损失冒烟明确使用 `GradScaler(init_scale=128.)`，因此最终冒烟继承这一固定检查规则，并以 FP32/AMP 各一步重新核验。正式训练仍使用原生 trainer 的 GradScaler，未改初值或恢复规则；原生动态缩放可能跳过初期 overflow 步，本地检查不声称默认 scale 首批一定有限。没有改 batch、分辨率、AMP 或任何模型模块来“通过”检查。

## 母版真实 val 的离线调查

只读已有正式 val 导出的前 64 张，未重新推理。210 对 IoU50 检测匹配中，横轴有嵌套且偏心的比例约 54.29%，纵轴约 63.33%。精确样本 ID、源导出/母版 best SHA 与聚合统计见 `parent_val64_geometry.json`。

母版旧导出缺原始 logits/query index，因此该调查使用检测匹配，**不是训练 Hungarian 正样本发生率**；原图坐标回算存在阈值附近的数值边界限制。它只说明有可观察的几何空间，不证明 NCR 检测更好。正式实验应同时查看独立 mAP50–95、AP75/高 IoU AP、中心误差、尺寸误差。

局部验证发生于独立 worktree 的未提交实现阶段；最终交付 SHA 由普通 Git commit 和远端核验给出。`delivery_checks.json` 记录交付源文件 LF SHA256，避免把母版 HEAD 加工作区修改误写成“母版已经含 NCR”。本地 smoke 后仅改进身份、打包检查和 OFF 诊断记录，未改变激活 NCR 数学路径或训练前向；无需反复运行同一 GPU 冒烟。
