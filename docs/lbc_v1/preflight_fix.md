# LBC v1 预检修复（服务器待验证）

修复基于原分支 `exp-rtdetr-r18-lite-lbc-v1` 的
`571fc7587cda9f7098092048edf0c8ed0796498e`。
输入证据为 `lbc_v1_preflight_debug_20260924T070921Z.zip`，
SHA256：`44c8369a05198864ef18922725a54acc3e908f593fa0387e05796f663eb58dcd`。

原服务器记录：B16/640/AMP、nbs=64，每 4 个微批次一次更新尝试；16 个微批次、
0 次有效更新、4 次 overflow 跳步。最后 total loss 约 38.0188、LBC weighted loss 约
0.08381、有效 pairs=124/127。容量断言失败后，严格 JSON 因 inf 再次失败。
这些证据不能证明原因只是初始 scale 太大，也不能证明真实网络中的 LBC 梯度完全正常。
输入压缩包及既有报告未修改。历史 local_evidence.json 等报告仍是此前代码的历史证据。

## 修复范围

- 在 `bounded_check` 创建的独立 Trainer 完成原生 setup 后，只初始化一次原生 GradScaler，
  `init_scale=128`。报告记录原生默认初始值（65536）、原实例 scale/state、诊断值和适用范围。
  动态缩放继续由原生 scaler 控制；没有额外降低初始 scale 的重试循环。
  上限仍为 16 个微批次、900 秒、2 次有效更新；低 scale 下仍失败即保存 FAILED 并抛出原异常。
- 原生训练器的 scaler 创建/恢复代码、正式配置、batch、nbs、累积、优化器、学习率、
  两组 clip 阈值 10.0、LBC 公式及权重不变。新增观察器仅由容量诊断启用。
- 每次更新边界保存 unscale 后、clip 前的原网络/辅助头梯度有限性、缺失/非有限参数名、
  FP64 观测 L2 范数及原生 clip 返回的裁剪前范数；还保存 scale 前后值、累积量、有效/跳步计数、
  更新后全部参数有限性、辅助头及明确命名的骨干 S3 参数变化。
  两种范数分开记录，避免把 FP32 范数溢出误判为所有梯度元素都非有限。
- PASS 仍须实际完成 1～2 次有效更新，且有效更新有有限梯度/范数、头和骨干真实变化，
  同时通过原有覆盖率、EMA 头状态覆盖、scaler roundtrip 和严格融合检查。
  AMP 跳步可以按原生机制恢复，但不计入有效更新。

## 报告格式及异常

`bounded.json`、`bounded_progress.json`、`epochs.jsonl` 使用同一个严格序列化器。
只复制日志结构，不修改原数值或张量。NaN/+Inf/-Inf 分别保存为：

```json
{
  "clip": {"head": {"__nonfinite_float__": "+Inf"}},
  "nonfinite_fields": [{"path": "/clip/head", "value": "+Inf"}]
}
```

路径是相对当前报告根的 JSON Pointer；数组使用数字下标，路径中的 / 和 ~ 按规范转义。
状态、原异常及其 traceback 保留，非有限值不替换为 0。已保存的嵌套容量报告仍保留自身路径根。
JSON 始终使用 `allow_nan=False`；普通报告先严格序列化再原子替换，JSONL 先序列化完整一行再追加。
finally 写盘异常输出到 stderr；若已有计算异常，继续抛原计算异常。
若只有写盘异常，仍失败退出。外层预检即使子进程失败，也收集其已保存的容量失败报告。

## 有针对性的本地验证

运行：

```text
python tools/lbc_v1_preflight_checks.py docs/lbc_v1/preflight_fix_validation.json
```

6 项回归验证，结果由脚本生成，含源码 LF SHA256。环境为 RTX 2060、torch 2.7.1+cu118。
CUDA 测试使用小型合成骨干、真实 LBC 头/公式、真实 AdamW 和原生 GradScaler：

1. 人为注入 Inf 梯度后原生 AMP 跳步、scale 128→64、有效更新保持 0，参数不变；
   随后两个有限梯度更新确实改变头和骨干。
2. 启用/关闭只读观察器的更新结果及 scaler 状态逐位相同。
3. 诊断覆盖不影响另一个正式实例的默认 65536；实际原生 checkpoint 恢复函数
   精确恢复 scale=64 和 growth tracker=1；恢复实例拒绝诊断覆盖。
   同时检查旧版和新版原生 GradScaler 构造接口。
4. 持续注入 overflow 的独立容量测试在 16 个微批次/4 次跳步后失败；
   两份严格 JSON 均保存原容量异常、traceback、全部尝试和非有限数值位置。
5. NaN/+Inf/-Inf 及真实 epoch_end 回调生成的 JSONL 能严格解析，源数值未变。
6. 模拟磁盘写失败不覆盖原计算异常；没有计算异常时写失败仍抛出。

以上为回归测试，不是 RTX4090 上真实 RT-DETR B16/640 的容量或梯度结论。
**服务器待验证。正式训练未启动，LCD 未修改或停止。**

## 服务器更新顺序

使用交付的完整 SHA 调用已有官方 `tools/sync_lbc_v1.sh`，然后分别运行官方
`tools/lbc_v1.py prepare` 和 `tools/lbc_v1.py preflight`，解释器固定为
`/root/miniconda3/envs/rtdetr/bin/python`。同步工具保留原交付记录；
prepare/preflight 自动归档历史摘要并创建新诊断目录，不需要删除旧失败目录。
不要手改 delivery/preflight 状态，不自动调用 start。服务器完成一次修复后的预检，
核对新报告及所有更新尝试后，再决定是否启动正式训练。
