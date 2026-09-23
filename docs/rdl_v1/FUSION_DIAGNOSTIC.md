# 服务器融合失败：定位状态与一次有界取证

## 结论边界

服务器在 `fe89760788e91405fe584bc5bbedc27731fa0890` 的 ordered fusion assertion 报
621/3000 不一致、max abs 0.8499179482460022。本机未复现这个失败，当前没有服务器执行通道，
因此这次交付是**诊断补全，服务器根因与修复仍待现场证据**，不是已经修好服务器 preflight。
原 `atol=rtol=3e-5`、逐行输出比较和失败退出保留；没有引入任何候选变化的自动豁免。

静态核查及本地实际运行确认：

- before 是 `trainer.ema.ema.float().eval()` 的同一输入预测，经 validation 后 clone；随后在同一 EMA 上原生 fuse，再预测。
- 先前的小批次 AMP loss 已退出 autocast 上下文。融合段只有 no_grad；实际记录模型/图像/encoder score head 输入为 FP32，CUDA/CPU autocast 均 false，所有模块 eval。没有新增关闭 AMP 的代码。
- validation 没有修改预测、模型 state_dict、输入或 anchors/valid_mask 缓存；诊断副本 state hash 完全相同。
- 母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 的 BaseModel.fuse AST 保持一致；对 LIFDown 的排除仍在。
  LIF、CBR、head、transformer、conv 源码与母版逐文件相同。LIF 状态融合前后精确相等，BN 保留；普通 BN 数量 69→27。
- 新训练包装仍使用原生 RTDETRDetectionModel，没有覆写 fuse。用母版的原 predict/fuse 方法、同一组当前生命周期后的权重/输入/缓存运行，两边所有采集张量逐位一致。
  这是**推理/融合控制实验**，不是母版训练历史的重放，也不是另一个训练权重的比较。

母版早已有 `tools/c19_lif_v1_probe.py` / `c19_lif_v1_diagnostic.py` 的候选 ID 诊断。
母版接受候选集合不变的置换，需要连续层、按实际 ID 对齐和固定 ID 重放同时通过。
本轮 RDL 的原检查没有这种候选语义；但这一静态差异并不能证明截图由置换导致。
新诊断复用母版采集工具，**不采用其不同的容差或放行谓词**。

## 本地实际证据

Python 3.9.25 / torch 2.7.1+cu118 / RTX 2060（服务器原环境为 Python 3.10.13 / torch 2.1.2+cu121）。
只重跑失败的 real_model 生命周期及融合现场；没有重跑数学/运营/容量检查或正式长训。

| 项目 | CPU | CUDA |
|---|---:|---:|
| 原融合断言 | PASS | PASS |
| 最终输出最大绝对差 | 1.7881393432617188e-7 | 7.450580596923828e-8 |
| top-k 集合/顺序 | 两图均完全一致 | 两图均完全一致 |
| 候选分数最大扰动（图0/图1） | 6.5565e-7 / 7.1526e-7 | 8.9407e-7 / 7.7486e-7 |
| 未融合 k/k+1 间隔（图0/图1） | 1.7726e-4 / 2.6864e-4 | 2.5147e-4 / 5.7995e-5 |
| 融合 k/k+1 间隔（图0/图1） | 1.7709e-4 / 2.6858e-4 | 2.5141e-4 / 5.8591e-5 |
| 固定 A/B 两组 ID、各自 encoder 输入 | PASS | PASS |
| 相同三层 head 输入＋相同 ID 的隔离检查 | PASS | PASS |
| 母版方法与 RDL 方法在同一现场 | 所有采集张量精确相等 | 所有采集张量精确相等 |

故障注入：仅在隔离模型中给融合后的 CBR offset bias 加 3；原断言拒绝 max abs
0.020596489310264587，保存 FAIL、fixture、records，common-input replay 也报告差异。
诊断没有把失败变成 PASS。语法检查和 `git diff --check` 通过。
具体机器可读摘要见 `fusion_local_evidence.json`。完整本地现场在忽略目录
`outputs/rdl_v1/fusion_investigation_{cpu,cuda_final}`；不是服务器结果。

## 安全更新服务器已存在的旧工作树

最终交付回复给出完整新 SHA。以下 `rdl_new` 要填该 **40 位 SHA**；不要把旧 sync 命令当更新命令。
旧 sync 只验证“已有 HEAD 已等于目标”，并不更新它。这里先明确检验旧 HEAD/干净状态/快进关系，
切换该独立工作树，再用原 sync 核验并登记交付。主仓库 HEAD 和原 outputs 保留。
该工作树应处于此次 preflight 失败后的停止状态。

```bash
set -euo pipefail
rdl_main=/root/autodl-tmp/projects/Crack_RTDETR
rdl_wt=/root/autodl-tmp/projects/Crack_RTDETR-rdl_v1
rdl_old=fe89760788e91405fe584bc5bbedc27731fa0890
rdl_new=填写最终回复中的完整新SHA
test -f "$rdl_wt/.git"
test "$(git -C "$rdl_wt" rev-parse HEAD)" = "$rdl_old"
test -z "$(git -C "$rdl_wt" status --porcelain --untracked-files=no)"
rdl_main_head="$(git -C "$rdl_main" rev-parse HEAD)"
git -C "$rdl_main" fetch origin exp-rtdetr-r18-lite-rdl-v1
test "$(git -C "$rdl_main" rev-parse FETCH_HEAD)" = "$rdl_new"
git -C "$rdl_main" merge-base --is-ancestor "$rdl_old" "$rdl_new"
rdl_backup="$rdl_wt/outputs/delivery_before_${rdl_new}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$rdl_backup"
cp -p "$rdl_wt/outputs/rdl_v1_delivery.json" "$rdl_backup/"
git -C "$rdl_wt" checkout --no-overwrite-ignore --detach "$rdl_new"
test "$(git -C "$rdl_wt" rev-parse HEAD)" = "$rdl_new"
test "$(git -C "$rdl_main" rev-parse HEAD)" = "$rdl_main_head"
bash "$rdl_wt/tools/sync_rdl_v1.sh" "$rdl_new"
```

没有 reset/clean/stash、强制推送或删除旧现场。若检查失败，保留输出并定位，不能强制覆盖。

## 一次有界诊断

更新后只运行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/diagnose_rdl_v1_fusion.sh
```

固定原 conda 环境、原公共初始化、原 seed、B2/160×192 的工程 fixture。
重建原 real_model 的 CPU 一步更新、FP32/AMP/保存恢复顺序，因为它们会影响 BN 状态；
不能从一份全新模型跳到 fusion 冒充相同现场。再执行最多 10 次隔离 eval 前向。
总时限 600 秒，超时 TERM 后 30 秒 KILL；没有重试、关闭 AMP、B16/640 容量检查、数据评估或正式训练。
200e/B16/640 训练配方及公共初始化不变。原断言失败仍非零退出（通常 1）；超时通常 124。

输出均使用唯一目录，保留旧 preflight：

- `outputs/rdl_v1/fusion_diag_<UTC>_<PID>.log`：完整运行日志及紧凑摘要。
- 同名前缀 `_result.json`：原 real_model 结果，不能作为整个 preflight 授权。
- 目录内 `fusion.json`：运行版本/代码 SHA、state/hash、真实 dtype/autocast、缓存、分数差、k/k+1 间隔、ULP、完整 630 个候选 logits、300 个 ID 的集合及顺序、固定输入比较、母版控制。
- `fixture.pt`（约 82 MB）与 `records.pt`（约 128 MB）：原失败权重、输入、原 before/after、RNG 和逐层张量；留在服务器，不提交 Git。超时/早期失败可能仅有 log/result。

请保留并返回 `.log`、`_result.json` 和 `fusion.json`；一般无需传两个 pt。重点字段：
`original_assertion`、`inside_score_head`、`validation_invariants`、`before_fuse/after_fuse/after_predict.cache`、
`fusion_invariants`、`reproduce_original`、`selection.images`（全部索引、分数扰动、临界间隔）、
`pre_selection`、`candidate_id_alignment`、`fixed_ids_A/B`、`common_candidate_inputs`、
`common_head_outputs`、`mother_vs_rdl`、`finding`。`traces` 证明实际 top-k 调用和 gather 使用的 ID。

解释限制：

- `reproduce_original` 必须逐位一致，否则重放不能解释原失败。
- ID 对齐是按实际 encoder 候选身份，不排序最终框；集合变化时不取交集宣称通过。
- fixed_ids 保留各自 encoder 数值；common_head 另用逐位相同三层输入和 ID 隔离 head/decoder/CBR，
  同时检查 selected_features/anchors/memory 是否相同。任何连续差异都保留原容差统计。
- 即使 `finding` 观察到母版置换/集合变化，原断言仍 FAIL；需要现场证据审查后才决定最小判据修复。
- 原 prepared/preflight 绑定旧提交，不挪用为新提交 PASS。本次诊断不会更新训练授权。
