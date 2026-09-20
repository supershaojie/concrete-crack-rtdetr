# RDM AMP 预检提前中止修复

继续分支 `exp-rtdetr-r18-lite-rdm-v1`，部署旧 SHA 为
`d6e4aa96db7ffdb749ac433c4bb8faa2c39a537d`。只修改预检观测与分类、小型测试及交接说明。
修复已实施，**服务器新验证待执行；正式训练 NOT_STARTED，完整 val/test NOT_RUN**。

## 原因和证据边界

附件提供的摘要显示：3 个实际 batch、0 个有效更新、3 次 scale 回退，65536→32768→16384→8192，
错误来自本工具的“连续三次”断言。它支持“AMP 启动回退被预检提前截断”，不证明模型稳定或能够涨点。
本次未登录服务器，未取得服务器完整 JSON；不能从摘要中的空列表推断完整 7 项梯度记录均存在且有限。
旧 FAILED 报告必须按原字节保留，不重新解释成 PASSED。提供的服务器命令先校验其身份和初始化哈希，再归档。

核对了 PyTorch **v2.1.2** 官方源码：

- [GradScaler.step / update](https://github.com/pytorch/pytorch/blob/v2.1.2/torch/cuda/amp/grad_scaler.py)：标准 AdamW 的非有限梯度会使原生 scaler 跳过 optimizer.step，update 按 backoff_factor 降低 scale。
- [Optimizer.register_step_pre_hook](https://github.com/pytorch/pytorch/blob/v2.1.2/torch/optim/optimizer.py)：公开 hook 在真实 optimizer.step 被调用时触发，支持移除，不依赖 step 的返回值。

## 行为

预检专用 `ProbeTrainer.optimizer_step` 使用观测上下文包住**一次**原生 `super().optimizer_step()`。
不修改正式 RDMTrainer，不增加 unscale/backward/update，不改变 scaler 初值、回退系数、梯度裁剪、优化器或 EMA。
每次观测的 optimizer pre-hook 在 finally 移除；它只计数，并读取 Wo 实际即将应用的梯度，返回 None 不修改参数或调用参数。

- `AMP_BACKOFF`：本批 RDM/decoder 前向、loss/loss_items、更新前后全模型参数有限；存在 scaled 非有限梯度；
  整个 optimizer.step 实际调用数为0；scale 按原生 backoff_factor 降低；Wo 状态和数值未动。
  连续次数只记录，不单独提前终止，不计有效更新。
- `EFFECTIVE_UPDATE`：梯度有限、实际调用1次、scale 未回退、Wo 的实际应用梯度有限且非零，Wo 值和 state.step 均推进。
  仅 weight decay 导致 Wo 变化不计有效更新。
- `NO_EFFECTIVE_UPDATE`：合法调用但没有达到上述有效更新条件（例如初始 lr=0），继续原预算。
- `FAILED`：当前前向/loss/参数非有限，缺失当前批证据或 RDM 梯度，非有限梯度却实际执行了更新，
  调用/scale/状态证据矛盾或真实运行异常。记录在断言前加入步骤数组，最终 JSON 保留失败原因。
- `PENDING`：启动8批或总16批到达仍未满足原有效更新和上游梯度要求。非零退出，start 拒绝；不自动重试。

启动仍至多8个实际 batch且至少2次有效更新，resume 至少1次，总预算16包括跳步。
后续原生保存/half保存源核验、真实resume、单批half EMA val和各一次FP32/half AutoBackend保持。
当前批前向标志在 batch_start 重置；不会用前一批 last_loss 替代当前 loss。

每步仅保存标量：当前 batch/loss、scale、全局 scaled 梯度统计、实际 optimizer.step 调用数、完整7项RDM梯度信息、
Wo变化及状态、分类。非有限梯度记录参数项总数和最多20个名称；None梯度单独计数。
不保存 tensor、长期 checkpoint 或原始梯度；TemporaryDirectory 及原≤10MiB报告/日志目标保持。

## 快速验证

`python tools/check_rdm_amp_backoff.py`：CPU **fixture**，使用模拟 scaler 决策和真实微型 AdamW 公开 hook。
`fixtures.json` 不是服务器 AMP 成功证据。验证：

- 三次连续合法回退全部为 AMP_BACKOFF，有效更新仍为0，之后两次有限更新正常计数。
- 16项错误/缺失证据案例被拒绝；原生调用异常时仍移除 hook、保留失败记录。
- 仅 weight decay 的变化不计有效更新。
- 第8/16批预算耗尽为PENDING，原start准入仍阻断。
- 正式RDMTrainer源码完全不变；模型、YAML、初始化方法、109项配方、数据身份、rdm.py及原shell入口不变。
- 本机已有受控初值SHA仍为 `5dac7ce5dae8ad6f94890c9d5f70efa9e37d4fac855dae0d2fe5d25bc070a443`；
  不拿此机器序列化文件哈希要求服务器重建。服务器以其原失败报告绑定的原初值哈希为准。

已做 Python 语法检查与 Git diff 范围检查；未运行完整模型训练、母版数值矩阵、CUDA容量或独立val/test。
新增fixture报告约13KiB，无测试checkpoint。

## 服务器更新与一次预检

提交完成后交接文档绑定新完整SHA，聊天直接提供两段命令：

1. 对现有 worktree 做固定旧SHA→新SHA的安全 detached checkout。
   不调用会拒绝现有不同HEAD的旧sync_rdm.sh；检查tracked修改、公共仓库、祖先、主目录状态和原初值哈希。
   checkout使用`--no-overwrite-ignore`，保护ignored/untracked结果；不reset/clean/改origin。
   只在目标提交缺失时按HTTP/1.1、无交互、120秒×最多5次fetch指定分支。
2. 利用现有action.lock约定排他归档已完成FAILED报告，记录原字节SHA256，不碰旧日志。
   只执行一次`bash tools/rdm_server.sh preflight`，不运行init，成功后plan。
   PASSED、启动/恢复有效更新达标、plan.pending为空才表示通过；否则输出阶段/错误/报告路径并停止。

旧本机报告和服务器旧失败报告均不能放行修复后start。运行内容哈希变化使旧证据失效；新预检入口会按现有机制
检查当前内容。工程通过不代表精度或收敛保证。
