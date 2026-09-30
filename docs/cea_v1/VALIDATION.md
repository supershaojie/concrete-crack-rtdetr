# CEA v1 开发验证（2026-09-30）

这些记录是工程与小样本机制证据，不是新实验 AP/涨点验证。没有启动正式 200 轮训练。

| 范围 | 实际状态 | 证据 |
|---|---|---|
| CPU PyTorch 契约及集成 | PASS，11 项 | `validation_cpu.json` |
| CUDA/AMP 算法与主路检查 | PASS，8 项；完整生命周期在 CPU 检查 | `validation_cuda_core.json` |
| CUDA 全模型逐位梯度/更新一致 | 未通过零容差；不声称通过 | `validation_cuda_exact_limitation.json` |
| 原母版与自身 CUDA 对照 | loss 完全相同，但梯度有非确定性差异 | `validation_cuda_mother_control.json` |
| 真实 Trainer/checkpoint/resume | CPU PASS；实际 epoch0→1，optimizer/EMA/config/key 恢复 | `validation_cpu.json` |
| CUDA GradScaler 真实状态 resume | PENDING，服务器 preflight 将执行 | 服务器 `outputs/cea_v1/preflight.json` |
| 独立评估完整导出/REUSED/离线 pack | CPU 合成 2 图/600 query PASS；重复评估未推理，打包未推理 | `validation_cpu.json` |
| 公共初始化映射及序列化 | PASS，公共 SHA 验证、原 nc1 映射逐张量一致 | `initialization_audit.json` |
| 完整母版配方 | PASS，109 字段和类型，仅记录精确路径别名 | `recipe_audit.json` |
| 本地真实数据划分 | 与母版三 split 路径/标签 SHA 完全一致 | `mother_evidence.json`，本地 `outputs/cea_v1/dev_data_audit.json` |
| 母版权重来源 | best SHA 与完整 val/test 归档、母版提交一致 | `mother_evidence.json` |
| 本地母版 val 小样本机制探针 | 完成，8 张固定真实 val，B1/640/FP32 | `mother_val_probe.json` |
| 正式服务器 B16/640/AMP 真实更新 | PENDING；不把本地缩小测试当成 TECHNICAL_PASS | 独立执行 preflight |
| 64 val + 8 原增强 B16 train 的完整诊断 | PENDING | 独立执行 diagnose |
| Linux tmux 页面/进程端到端运行 | PENDING；脚本已通过 Bash 语法检查，调用链已审查 | 服务器 status/attach |
| 新实验完整训练、FP32 val/test | PENDING | 由用户显式 start，训练后独立 val/test |

开发环境为 Windows、Python 3.9.13、torch 2.7.1+cu118、RTX 2060。
服务器按真实环境记录，不升级原 Python/torch/CUDA/Ultralytics。

## CPU 与 CUDA 误差

事先声明的 CPU 关闭路径输出、原损失、各层匹配、全部参数梯度和一次 AdamW 更新容差为 0，实测差异均为 0。
检查覆盖 enabled=false、lambda=0、e=5；主损失日志仍为三项。
非退化全模型仅 backward CEA 时，唯一有 grad 的参数是原 `model.26.cbr.score.weight`。
真实 Native Trainer 构造、保存和 resume 均参与测试，未以孤立 NumPy 公式代替 PyTorch 集成。

CBR 原文件从母版 Git 对象读出做同权重、同输入比较；FP32/AMP 主路输出均逐位一致。
学生概率与同次原聚合概率的声明绝对容差为 1e-6：CPU 最大 2.980232e-8，CUDA 最大 0。
同精度 FP32 锚点重建绝对容差为 1e-7：CPU 最大 0，CUDA 最大 5.960464e-8。
CUDA AMP 实际主输出与 FP32 诊断锚点的最大差异为 1.859665e-5，单列报告，不计入候选改善。
固定 A/t 的解析 logit 梯度与 autograd 对照使用 atol=1e-8、rtol=1e-6。

最初 CUDA 全网零容差梯度比较失败，记录未删除、未扩大容差。补上与原训练一致的
CUBLAS/cudnn 确定性设置后仍不是逐位相同。随后只用母版类（完全未使用 CEA 类）做相同
权重/输入/RNG 对照：loss 差为 0，梯度最大差 0.00048065185546875，全网相对 L2 差
4.6518126e-6；PyTorch 明确报告 CUDA grid_sample backward 无确定性实现。
因此 CUDA 核心集不冒充逐位全网反传/更新证明，这部分严格等价性由 CPU 集验证。
完整失败原始日志保留在本地 outputs/cea_v1，仓库保存其摘要与 SHA。

## 已有真实母版小样本信号

8 张固定 val 图共有 19 个匹配 GT、76 条边，75 条边找到比 FP32 基线代价更低的候选，19 个 GT 均涉及改善。
原始 CEA loss 的逐图范围为 0.0006336004～0.0162350759；乘 lambda=0.1、r=1 后为
0.0000633600～0.0016235076。没有改 lambda，也没有按 sum(A) 归一化。

前两图同一 score.weight 的加权 CEA/L0 梯度范数比分别为 0.01102139 与 0.0004345294，
余弦分别为 0.99932909 与 0.92838804。实际 FP32 框与锚点最大差为 2.980232e-8。
这些数值仅属于 B1、eval 模式的小样本探针，不代表增强 train B16 的量级，不能替代完整 diagnose，
也不证明参数更新后定位改善或最终 AP 提升。逐图原始分位数完整保存在 `mother_val_probe.json`。

## 复现开发检查

从实验工作树运行：

```bash
python tools/check_cea_v1.py --device cpu --output outputs/cea_v1/dev_cpu.json
python tools/check_cea_v1.py --device cuda:0 --core-only --output outputs/cea_v1/dev_cuda_core.json
bash -n tools/sync_cea_v1.sh
```

服务器 preflight 自动在子进程运行这些有界检查，再执行真实 B16/640/AMP 更新和原生 CUDA resume 校验。
CPU Native Trainer 会临时改变其进程的 CUDA_VISIBLE_DEVICES，因此它必须与真实 CUDA worker 隔离；
这个开发检查中发现的问题已在服务器调用链修复。

合成 Windows checkpoint fixture 使用独立临时目录，避免原库 check_file 对路径中的单引号做清理；
没有为此修改 Linux 正式路径行为。
