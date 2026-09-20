# RDM 单批 half EMA 诊断

本轮为**诊断增强，数值问题尚未修复**。服务器验收待执行；正式训练、完整 val/test 均未启动。
基于 `f01878ebd944c793257f058031fe13d71ee2fc46`，母版、模型、配方与已有初始化不变。

## 已知证据与未知项

附件提供的服务器摘要：start 5批/2次有效更新/3次合法 AMP_BACKOFF，save_reload 通过，resume 1批/1次有效更新通过；
总训练6批，half_ema_val 因 `Nonfinite native half EMA prediction` 失败，rdm_calls=1、batches=0，临时文件已清理。
本机未获得该服务器完整 JSON、未复现这次数值失败；旧临时 checkpoint 已清理，不能重新加载。

旧代码中 rdm_calls 只有在 RDM 直接输出 half、有限且 Wo 非零后才增加；batches 只有批次完成后才增加。
因此本次第一批 forward 已执行，但没有完成。根输出检查递归覆盖 `(y, raw)`，旧报告不能确定失败张量。
`cbr.py` 中 y 用于预测，raw 中 dec/enc scores 与 boxes 进入原生验证 loss，不能只检查 y。
当前没有证据把失败归因于 RDM、CBR、LIF、EMA、BN 或某个 attention 算子，未做推测性数值修法。

## 实现

- `tools/rdm_half_diagnostic.py`：预检专用只读 observer。`preflight_rdm.py` 仅在原有一次 half EMA val 周围启用。
- 原生 validator 在 `model.half()`、`model.eval()` 后调用 `on_val_start`；利用该现有回调采集转换后状态，
  在调用 validator 前采集转换前状态。分别汇总 parameters/buffers、dtype、整数项、非有限项和超出 FP16 范围的名称。
  记录转换前有限而转换后非有限的名称，最多20项；整数/bool buffers 不按 half 浮点溢出判断。
- root pre-hook 记录真实验证输入 shape/dtype/batch size、train/eval、AMP/autocast、梯度/inference 状态和 EMA updates。
  不改 loader 或 batch，仍是现有的一批（原生通常为 B32，以实测为准）。
- root post-hook 在原完整 finite 断言之前记录 y、raw.dec_bboxes/dec_scores/enc_bboxes/enc_scores/dn_meta 中每个张量；
  只保存路径、shape、dtype、numel、NaN/+Inf/-Inf 数量与有限元素最大绝对值，无有限元素则 null。
  None 元素不是张量，不伪造统计。JSON 使用 `allow_nan=False`。
- 在所有 named_modules 上挂 pre/post 观测，覆盖骨干、neck 和 decoder 内部模块。记录首个非有限边界、输入是否有限、
  模块类型/路径/包围模块、最多8个异常事件及各自最多20项张量摘要、最近3个有限事件；不保存正常逐层全量日志。
  共享模块使用规范名称及包围模块，函数式操作只能由模块边界缩小区间，首个接收异常的模块不等于根因。
- 显式 attention mask 参数中只有 -Inf 且无 NaN/+Inf 时单列为 mask sentinel；不把其判为激活溢出。
  对 decoder 已有 anchor cache 只读统计，并检查 +Inf 是否恰落在 invalid mask；不生成新 anchors。
  未激活 reference logits 中 Inf 的来源仍未确定，不能据 cache 就证明所选 reference 的 Inf 合法，
  更不能据此豁免最终 y/raw 的有限性检查。内部观测本身不替代原输出检查或判定模型失败。
- hooks 返回 None，不修改输入、输出、参数、buffers、RNG 或 forward 次数。诊断错误保留在 diagnostic_errors；
  原生异常保持主错误。原验证正常返回但诊断不完整或 EMA 状态非有限时，预检拒绝通过。
- 所有新增 hooks、val_start 回调与原 val_batch/输出断言 hooks 均在 finally 清理。
  旧代码异常路径未移除验证 hooks 的清理缺口已修复；这不是本次数值问题的根因修复。

PyTorch 2.1.2 的 [公开 hook API 与 None 返回语义](https://github.com/pytorch/pytorch/blob/v2.1.2/torch/nn/modules/module.py)
已核对；使用 `with_kwargs=True`，不重写 native forward、half 或 EMA.update。

## 保持的预算与约束

正式 RDMTrainer、RDM 公式/插入点、CBR/LIF、AMP_BACKOFF 分类和完整配方不变。
200e/B16/640/seed42/原生 AMP；启动≤8个实际训练batch且≥2次有效更新，恢复≥1次，总计≤16含skip，原上游梯度要求保留。
half EMA仍单批，不增加整网 FP32 replay、精度组合矩阵或母版重跑；失败即停止后续 backend 检查。
旧 FAILED 报告保持原字节，原临时目录清理机制保留，不保留 checkpoint/图像/特征/梯度数组。

## 本机验证与服务器交接

`python tools/check_rdm_half_diagnostic.py` 只用微型 CPU 模块和张量，覆盖 y/raw 分离、参数/buffer half 溢出、
原已存在异常、整数 buffer、有限输入→异常输出、hook 清理、诊断失败时原始异常保留、标准 JSON、mask/anchor 语义和限量。
检查正式 RDMTrainer、已修复 AMP 观测/分类、有界训练函数、原模型/配方/数据文件与 f01878 完全一致；原本机初值哈希不变。
fixture PASSED 不等于服务器 half EMA PASSED；`fixtures.json` 只记录此小型测试。

交接命令绑定最终完整 SHA，先安全更新 worktree、核对主目录及初值不变，再用 action.lock 排他归档当前 f01878 的已完成
half_ema_val FAILED 报告；检查原错误、variant 和 init/source/data/recipe 身份，不要求旧新代码内容哈希相等。
旧日志不动，归档记录 SHA256；调用官方预检前释放外层锁，只运行一次 `bash tools/rdm_server.sh preflight`。
失败通过 `python tools/rdm_half_diagnostic.py <preflight.json>` 输出首个异常边界、y/raw路径、EMA转换前后状态和报告路径；
无足够记录明确 UNLOCATED。只有整体 PASSED、原有效更新/预算/临时清理全部达标且 plan.pending=[] 才算通过。
源码/fixture/文档实际留存字节见 `storage.json`；服务器命令另外打印本次输出目录实际留存增量，不设置 GPU 必须空闲的条件。
