# ARG v1 数据身份修复交付

本次训练代码完整 SHA：`ef9cb7e05e5557f7dd06c95cf2361998a284adc9`。
服务器待升级旧 SHA：`e6d6ce11783e57cdd3377f6b5a103665218df065`，适用状态为已 prepare、names 键类型导致 preflight 失败、尚未正式 start。
独立分支：`exp-rtdetr-r18-lite-arg-v1`；母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
代码已普通commit/push；封版时通过 `git ls-remote --heads origin refs/heads/exp-rtdetr-r18-lite-arg-v1` 核对远端与本地完整SHA一致。
随后单独提交此交付文档和验证摘要，以便文档能真实引用已经存在的代码SHA；服务器命令固定使用上面的代码提交。

本地worktree：`D:/MyProjects/Crack_RTDETR-arg_v1`。主仓库的另一实验分支和用户原有未提交文件保留。
服务器worktree：`/root/autodl-tmp/projects/Crack_RTDETR-arg_v1`。
服务器主仓库：`/root/autodl-tmp/projects/Crack_RTDETR`。
Python：`/root/miniconda3/envs/rtdetr/bin/python`；tmux：`arg-v1-training`。
正式run：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/arg_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。
证据目录：本实验worktree的 `outputs/arg_v1/`。

## 本次修复与验证

`tools/arg_v1_common.py` 在生成 inventory 时调用同一个 `data_identity_config()`，复制配置并将 names 类别编号转为字符串键。prepare 与 binding 都经过该路径，JSON 保存/读取前后身份相同；`0` 与 `"0"` 发生键冲突时明确报错，即使类别名相同也不会覆盖。
完整 `data == plan["data"]` 校验保留，YAML 文件本身及其原始字节哈希不变。没有修改 ARG 损失、母版结构、初始化、训练/评估配方、预检门禁或同步脚本。

本轮验证如下；详细范围见 [identity_fix_validation.json](identity_fix_validation.json)：

- 旧 SHA 的真实 `arg_v1_common.py` 在同一夹具中复现 `Data identity changed since prepare`。
- 新增 7 项身份回归测试 PASS：实际 prepare/inventory/binding 与 JSON 往返；字符串键；冲突拒绝；类别名变化；真实图像/标签/YAML 字节变化；真实图像/GT 数量变化；独立修改 names、路径、各级哈希和数量均拒绝。
- 既有 6 项生命周期/启动门禁测试 PASS。
- 交付命令的 4 项本地临时 Git worktree 测试 PASS：实际 fast-forward、完整备份/哈希/旧记录迁移、实际重新 prepare；脏目录、已有训练目录、初始化变化分别拒绝。服务器 sync 网络、tmux/进程控制、模型环境与配方使用夹具，未在服务器执行。
- Python 语法、16 个 Bash 命令块语法、3 个内嵌 Python 块语法和 `git diff --check` PASS。

服务器升级按 [server_commands.md](server_commands.md) 第 1～4 段完成“备份与升级 → 重新 prepare → 新 preflight → status”。命令处理原 sync 的 HEAD/sync.json 双重限制，保留整个旧输出的独立备份与失败证据。原初始化文件及 provenance 按哈希复用，seed42 与全部参数、数据身份在重新 prepare 后逐项对照旧记录。正式 start 单列，只在必需预检通过后执行，仍使用 `arg-v1-training`。

本轮未运行服务器 GPU 预检或长训练，也未重复运行下述历史数学/模型/GPU检查。

## 原 v1 已实施（历史记录）

- `arg_loss.py` 固定ARG公式、显式repair几何、停止权重梯度、1D GIoU、最终常规层唯一接线、r=0原路径和DN补零顺序。
- `arg_model.py` 可导入model/trainer，实际get_model重建后生效，原生优化/AMP/clip/EMA/checkpoint，epoch恢复与有界机制日志。
- `arg_val.py` 复用母版corrected_sorted_conf_mask_v1计算顺序；独立FP32 val/test与全query导出。
- `tools/arg_v1.py` 及sh入口支持prepare/preflight/probe/status/start/resume/val/test/pack；固定解释器/PYTHONPATH、tmux、真实退出码、身份锁和失败证据。
- warmup唯一共享修复：该处 `torch.empty` 改为同shape/dtype/device的 `torch.zeros`。原模块和原预测计算未改。

公式与接入细节、原参数依据、异常恢复和打包约定完整见 [README.md](README.md)。无需访问聊天截图或附件才能运行。

## 原 v1 已验证事实（历史记录，本轮未全部重跑）

| 范围 | 结果 |
|---|---|
| CPU数学/自动微分 | 5组测试PASS：重合、单轴、交换、包含、分离梯度、窄框、空框、极小保护、教学例/归一化、GT/权重停止梯度、固定权重有限差分 |
| 匹配与损失路由 | 有/无DN、混合空GT、全空GT、r=0/r=1和验证禁用PASS；4个原匹配调用一致，仅多loss_arg，无loss_arg_dn |
| 原生模型重建 | 552 state keys、20,149,765参数；公共初始化映射与母版一致；9项nc80→nc1分类适配保持原生 |
| r=0完整模型对照 | 单线程CPU：raw forward/DN元数据、原loss、全部梯度和一次原生AdamW更新逐位一致 |
| 优化器分组 | 原345个张量，130 weight/81 norm/134 bias，分组及超参一致 |
| ARG单独反传 | CBR offset/query路径、最终回归头、共享Neck有有限非零梯度；独立最终分类head无直接梯度 |
| CPU检查点新进程恢复 | 原生resume恢复epoch20/ramp1、optimizer与EMA状态，禁用CPU scaler状态精确恢复；CUDA scaler仍待服务器 |
| 本机实际Trainer/AMP接线 | B2/160/workers0诊断设置下，原生get_model→setup与离线AMP自检PASS；没有容量或训练更新声明 |
| 新进程真实val生命周期 | 本机CUDA B1/640 FP32，实际加载→fuse→零warmup→1张真实val及全300query导出PASS；不是完整val/test |
| 母版推理对照 | 融合前/后各自逐位相等；LIF BN保留；母版原融合容差内，最大绝对差约1.04e-7 |
| 控制/故障包 | 6项测试PASS：各必需门禁、scale128单独记录、活跃锁、评估身份、缺test/权重的实际打包及hash回读、真实子进程超时 |
| 静态检查 | Python编译、Bash语法、git diff --check PASS |
| 数据与源权重 | 本地原train/val/test图像与GT计数吻合；逐图/标签SHA清单完成；公共权重hash符合要求 |

环境实际为 Windows、Python3.9.25、torch2.7.1+cu118、CUDA11.8、RTX2060 6GiB、仓库内定制Ultralytics8.4.21。
与历史服务器Python3.10.13/torch2.1.2+cu121/RTX4090不同，没有升级依赖。
多线程CPU embedding归约最初出现末位差异；最终逐位对照固定单线程，不改正式训练配方或放宽容差。
本地setup检查还发现并修复了清理DataLoader时不应对iterator做布尔求值的问题；失败日志保留于outputs，重跑通过。

原模块的LF哈希未变，原始字节hash及完整证据摘要见 [local_validation.json](local_validation.json)。
完整109字段的服务器模板见 [formal_args.yaml](formal_args.yaml)，差异摘要见 [recipe_diff.json](recipe_diff.json)。
检查实际执行时HEAD仍为母版，工作区包含待提交实现；摘要额外记录已封版代码SHA与源码LF清单，不把母版SHA冒充实现SHA。

## 服务器尚待验证

以下均为PENDING，没有声称已完成：服务器实际环境与离线资源、B16/640 AMP有效更新和峰值显存、原生高scale适应、CUDA optimizer/EMA/scaler新进程恢复、服务器16张真实val生命周期、真实tmux正式训练、完整独立val/test、训练新增开销与精度收益。

短检默认900秒/16训练micro-batch。必须观察真实optimizer更新与参数变化；必要时仅在隔离副本用scale128，并保留原生scale适应的PENDING。
正式训练仍使用原配方、原scaler、e=0和独立RNG；短检权重和增强序列不继承。

终端三项loss不代表全部L0；不能把ARG下降当作mAP改进。
后续只有有效结果时再考虑等系数等权1D GIoU对照。本次未增加消融长训、旧第三方案或其他检测模块。

下一条可执行命令是 [server_commands.md](server_commands.md) 第1段完整旧 worktree 升级块，已包含新旧真实40位SHA。
后续按“升级→重新prepare→新preflight→status→单独start→日志→val→test→pack”分段执行，不能无条件连跑preflight与训练。
若训练正常结束但final_eval失败，用val恢复评估，保留原异常与exit1，不用resume重训。
