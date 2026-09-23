# RDL 融合检查：按真实候选 ID 验收纯排列

## 根因和服务器证据

已读取用户附件 `rdl_fusion_review.zip` 的 fusion.json、preflight.json 和两份检查源码；源码与
`ddd2b49cf426892be068e108b59195a2879429ed` 一致。
原 fusion.json SHA256：`34f39f3ff6cd6c2f4046ec3c419f567c0f328cbccaa01da68af07359ce26a117`。

严格 FP32 作用域已生效，仍有正常舍入扰动导致同一候选集合内部交换：第二张图位置 115/116，
未融合为 `[532,535]`，融合后为 `[535,532]`。两个候选分数分别为：

| ID | 未融合 | 融合后 |
|---|---:|---:|
| 532 | -0.6539422273635864 | -0.6539422869682312 |
| 535 | -0.6539424061775208 | -0.6539421677589417 |

原逐行比较 4/3000 超差，最大差 0.2499469816684723；按实际 ID 对齐后，boxes 最大差
1.7881393432617188e-7、scores 最大差 9.5367431640625e-7，均通过原 atol=rtol=3e-5。
两图集合差均为空；pre_selection、固定 A/B ID、共同完整 head 输入都通过；母版同状态各 39 项逐位相等。
原始失败重放误差为 0，精度设置恢复成功。

因此本次根因是检查把“输出行号不变”误当作融合等价的必要条件。旧代码已经证明纯排列，却仍无条件
抛出逐行异常。它不证明生产 top-k 或 RDL 计算需要改动，也不能将候选集合变化一并放行。

## 最小行为变更

- 原 ordered assertion、容差和 FAIL 统计保留在 `original_status/original_assertion/original_error`。
- 新增独立 `fusion_acceptance`：`PASS_ORDERED`、`PASS_CANDIDATE_PERMUTATION` 或 `FAIL`。
  判定在实际精度恢复之后执行，不依据 finding 字符串。
- 纯排列必须通过完整 schema、有限性、实际分数重算的唯一 ID/同集合/同数量、所有 gather 轨迹、
  全部必要 ID 对齐张量、pre-selection、固定 A/B、共同输入、母版精确对照、缓存/输入/权重/LIF/精度不变量。
  缺失、空集合、非有限、集合变化、真实超差全部拒绝。ID 对齐仅在诊断副本进行。
- 调用方分别记录 `fusion_original_status` 和 `fusion_acceptance`。已证实的纯排列不会再被原逐行异常阻断；
  其他异常仍失败。preflight 在独立验收失败时停止，不进入容量检查。
- 没有改生产 top-k、输出顺序、模型/损失、初始化、正式训练/推理；没有改变容差或 TF32/AMP 恢复策略。
  不重写旧 FAIL 报告，新的 preflight 仍自动保存旧报告及 SHA。

本谓词只支持当前 B2/160×192、nc1、3 层/300 query 工程 fixture，保守拒绝其他形状或不完整证据。
JSON 离线审查不是新 preflight 授权；正式现场重放还会核对 fixture 模型状态 SHA 和旧 before/after 的逐位一致性。

## 针对性验证

本地没有重跑整套生命周期，也没有服务器容量/正式训练：

1. 对附件原报告逐项审查：原始 FAIL 保留，最终为 PASS_CANDIDATE_PERMUTATION。
2. 44 项证据测试通过，含集合变化、重复/错误数量 ID、gather 未验证、对齐真实超差、NaN/Inf、
   各组为空/字段缺失、母版不精确、输入/LIF/精度恢复失败、扩大容差等拒绝分支。
3. 对已有真实张量按 ID 构造双射并反向对齐：纯排列通过；实际框值误差和 NaN 被拒绝。
4. 调用方集成检查：原始异常存在且 original_status=FAIL，独立证据通过时正常返回；设置恢复。
5. 复用本地既有 fixture 做 CUDA 重放，before/after 均与原文件逐位一致，正常路径 PASS_ORDERED。
6. 真实隔离模型的 CBR 故障注入仍被拒绝，精度恢复且 caller 模型未改变。
7. preflight 边界检查：拒绝独立融合验收失败或精度恢复失败；旧 FAIL 字节保留。
   使用容量替身及小型 AMP 探针，不冒充 B16/640 容量通过。

结果摘要见 `fusion_permutation_validation.json`。附件未包含服务器 fixture.pt/records.pt，它们仍在服务器旧失败目录。

## 分段服务器操作

第一段：按最终回复提供的新 SHA 更新当前 ddd2b49 的工作树。必须先检验旧 HEAD/干净状态、fetch 验证远端 SHA，
再 checkout 新 SHA，最后调用 sync。更新前备份 preflight/prepared/environment/delivery JSON；保留全部失败目录。
旧 sync 本身不会更新已有工作树。

第二段：只重放已有现场，不运行前面的生命周期步骤。以下段落可独立执行：

```bash
(
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
test "$(command -v python)" = /root/miniconda3/envs/rtdetr/bin/python
cd /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1
export PYTHONPATH="$PWD/ultralytics-main"
rdl_old=outputs/rdl_v1/fusion_failure_20260923T184359913121Z
rdl_replay="outputs/rdl_v1/permutation_replay_$(date -u +%Y%m%dT%H%M%SZ)_$$"
timeout --signal=TERM --kill-after=30s 600s python tools/replay_rdl_v1_fusion.py \
  --fixture "$rdl_old/fixture.pt" --report "$rdl_old/fusion.json" \
  --output "$rdl_replay" --device cuda:0
)
```

应看到新 `_result.json` 中 status=PASS、original_status=FAIL、
fusion_acceptance.status=PASS_CANDIDATE_PERMUTATION、saved_fixture_replay 两项误差为 0、precision_scope.restored=true。
若旧输出不能精确重放，直接失败，不能扩大容差。新 fusion.json 包含完整候选证据；旧文件不变。

第三段：第二段通过后，仅运行一次新提交的最终门禁，补齐尚未通过的真实服务器 B16/640 AMP 容量及完整证据。
先前 preflight 在 real_model 失败处退出，operations/capacity 尚未执行；旧 prepared 又绑定旧 SHA，不能手改为 PASS。

```bash
set -euo pipefail
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh prepare
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh preflight
```

这一次完整预检用于生成新 SHA 的真实 gate 结果；开发/定位阶段不反复执行整套生命周期。
不要加 NVIDIA_TF32_OVERRIDE=0。检查 preflight.json：status=PASS、real_model 的独立融合验收已通过、
fusion_precision.restored=true、capacity.status=PASS、capacity.batch=16/imgsz=640/amp=true。
capacity 的 precision_before_amp 应与融合进入前一致，precision_inside_amp.cuda_autocast=true，退出后恢复。
若任一步失败，保留新报告停止，不自动开始训练。

本轮不执行 start/resume，ROR 状态未变。用户后续明确执行 start 且 gate 通过后，才会创建独立后台
tmux 会话 **rdl-v1-training**；同名会话存在时拒绝重复启动。
