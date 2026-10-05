# YOLOv13-L 可配置实验：服务器交付命令

日期：2026-10-05（Asia/Shanghai）。**本次没有启动服务器正式训练**。首轮 `v13l_aug_x13_01` 为 NOT_RUN 计划。

| 身份 | 实际值 |
|---|---|
| 分支 | `bench/yolov13l-configurable` |
| 实施代码完整 SHA | `64f6639847a0b6a1c18f8f246ba27e274a4de3ad` |
| 真实父提交 | `a7b3d842df60adf28e5cf35c086c58ef752674f8` |
| 作者源码 | `73289949533efac82bb5f72ec19b746618656bd2`，iMoonLab YOLOv13-L/l |
| 实施推送核验 | `git ls-remote origin refs/heads/bench/yolov13l-configurable` 实际返回上述 `64f6639...`，之后才补本文档 |
| 首轮配置哈希 | `a41c21caf0e514b40a4bc9aff047c4fea1fa50cc155e77c707b959a0bc2dc604` |
| 官方 yolov13l.pt | 111819790 字节；SHA256 `f95ad5bbf3aa80a3df2a28ff4c623582ded6e02bb43d7cea9063c2b248e316cb` |

后续交付提交只补本文档、NOT_RUN 配方归档和验证记录。训练检出 **64f6639 完整 SHA**；不能用之后的文档 HEAD 替换该训练身份。实现、默认配方、schema、模型/初始化/增强/环境锁、测试和脚本已在这个实施提交中。细节见 [实现说明](YOLOv13L_CONFIGURABLE.md) 与 [验证记录](evidence/yolov13l_configurable_delivery_validation.json)。

本机实测：46 个测试通过，1 个 POSIX 信号测试因 Windows 跳过；官方模型迁移 1562/1568 张量、773 个 gate/HyperACE/注意力张量保留源值；数据 6048/45573、1728/12840、864/6663（图/框）及参考指纹一致。独立 RTX2060 CUDA SMOKE_ONLY 用 2轮/batch2/64 像素合成数据完成 loss、反向、4 次真实 optimizer 更新、中断恢复、实际 resume/finalize guard、相同 best 的独立 FP32 val/test 和公共 CPU 评估；不是正式精度结果。默认审查包、含 best/last 包、只读校验和轻量归档已实测。

AutoDL 当前环境、真实 Linux tmux/POSIX 信号、正式数据 batch16/640 训练和全量 val/test、Flash/native parity、独占测速均未执行；具体 NOT_RUN/NOT_VERIFIED 原因和复验入口保存在验证记录。Bash 语法、CLI help、默认/SGD覆盖、AdamW YAML 生成/读取、配置冻结和重复 run 拒绝已实际验证。以下 start 等生产命令的接口经检查，**没有拿它们启动服务器训练**。

## 1. 固定代码建立独立 worktree，准备环境和官方权重

在普通服务器 SSH Bash 执行此段。仓库母工作树保持原状；新路径已存在时会停止，先检查该目录身份，不要覆盖或删除已有实验。

```bash
set -euo pipefail
REPO='/root/autodl-tmp/projects/Crack_RTDETR'
WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-configurable'
CODE_SHA='64f6639847a0b6a1c18f8f246ba27e274a4de3ad'

git -C "$REPO" fetch origin bench/yolov13l-configurable
git -C "$REPO" cat-file -e "${CODE_SHA}^{commit}"
test ! -e "$WT"
git -C "$REPO" worktree add --detach "$WT" "$CODE_SHA"
cd "$WT"
test "$(git rev-parse HEAD)" = "$CODE_SHA"

BASE_PY="$(command -v python3)"
bash scripts/autodl_yolov13l_configurable.sh bootstrap --base-python "$BASE_PY"
PY="$(cat "$WT/.runtime/yolov13l-configurable/python_path.txt")"
"$PY" -c 'import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)'
```

`bootstrap` 只准备新 `.vendor`、`.runtime`、`.envs`、核验模型和权重，不启动正式训练。默认只读继承兼容母依赖并在新 copies venv 安装锁定 overlay；源码固定 SHA，下载官方原资产后先校验大小/SHA256再加载，native 固定，不装 Flash wheel/部署套件。

如果母 Torch/TorchVision 不在已验证兼容对内，bootstrap 会保留失败记录并停止。使用新的环境目录准备作者参考的 2.2.2/0.17.2 cu121，不覆盖失败目录或修改母环境：

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh bootstrap \
  --base-python "$BASE_PY" \
  --env-dir "$WT/.envs/yolov13l-configurable-fresh" --fresh-torch
PY="$(cat "$WT/.runtime/yolov13l-configurable/python_path.txt")"
```

已有经过同一锁验证的作者源码或原始 COCO 资产可在 bootstrap 显式添加 `--reuse-source /absolute/author/tree --reuse-asset /absolute/original/yolov13l.pt`，只读复制到新位置。不要传已有训练 best/last；检查会拒绝不同资产。已冻结环境不自动升级或重新安装。后文显式 `--python "$PY"` 保留真实解释器与 prefix，适用于可选 fresh 环境。

新 SSH 登录时重新设置变量，无须重跑 bootstrap：

```bash
WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-configurable'
cd "$WT"
PY="$(cat "$WT/.runtime/yolov13l-configurable/python_path.txt")"
```

## 2. 查看和校验默认计划

```bash
cd "$WT"
cat benchmarks/comparison/yolov13l/default_config.yaml
cat benchmarks/comparison/yolov13l/configs/v13l_aug_x13_01.yaml
bash scripts/autodl_yolov13l_configurable.sh check-config --python "$PY"
bash scripts/autodl_yolov13l_configurable.sh check-config --python "$PY" \
  --config "$WT/benchmarks/comparison/yolov13l/configs/v13l_aug_x13_01.yaml"
```

这两条 check-config 不创建模型、不训练。优先级为默认 < YAML < --set。实际导入/模型检查可以另行执行，输出文件使用新名称：

```bash
"$PY" benchmarks/comparison/yolov13l/run.py check \
  --output "$WT/outputs/yolov13l-configurable-validation/manual_import_check.json"
```

## 3. 用户决定开始时，默认首轮 start

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh start --python "$PY" \
  --run-id v13l_aug_x13_01
```

自动创建并进入 `comparison-yolov13l-v13l_aug_x13_01`；若已在 tmux，则 switch-client。加 `--detach` 可返回 SSH 提示符。每个 run/session 单独锁定，允许与另一个 GPU0 实验并行，不等待整卡空闲、不停止其他模型、不在 OOM 后换参数。此命令依次做 preflight、train、best 的独立 FP32 val、公共 CPU evaluate、summary；真实 test 保持 `not_requested`。

默认母数据：`/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml` 与 `.../datasets/crack_det`。若服务器实际数据路径不同，仅在新 start 时显式传 `--data /absolute/data.yaml --data-root /absolute/crack_det`；实际划分、类别、数量和指纹须一致，不能重划分或凑数量。

## 4. 在服务器生成 YAML，启动另一 run

无需上传本地 YAML。候选文件可复用作为新 run 的输入，已经冻结的 run 不再读取它。

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh config --python "$PY" \
  --out "$WT/runtime_configs/v13l_from_yaml_02.yaml" \
  --set lr0=0.005 --set momentum=0.9 --set mixup=0.18
bash scripts/autodl_yolov13l_configurable.sh check-config --python "$PY" \
  --config "$WT/runtime_configs/v13l_from_yaml_02.yaml"
bash scripts/autodl_yolov13l_configurable.sh start --python "$PY" \
  --run-id v13l_from_yaml_02 \
  --config "$WT/runtime_configs/v13l_from_yaml_02.yaml"
```

`config --out` 拒绝覆盖现有候选文件；生成新候选时换文件名。YAML 也可只写合法部分字段，其他字段从默认值展开。

## 5. 只用 --set 启动新的 SGD 候选

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh start --python "$PY" \
  --run-id v13l_sgd_lr005_m09_mix018 \
  --set lr0=0.005 --set momentum=0.9 --set mixup=0.18
```

这仍使用默认 SGD，仅修改明确的三项；新 run 从官方 COCO 权重初始化。可选配方克隆同样不继承模型/optimizer 状态：

```bash
bash scripts/autodl_yolov13l_configurable.sh check-config --python "$PY" \
  --clone-config-from v13l_aug_x13_01 --set lr0=0.005
bash scripts/autodl_yolov13l_configurable.sh start --python "$PY" \
  --run-id v13l_cloned_recipe_03 \
  --clone-config-from v13l_aug_x13_01 --set lr0=0.005
```

## 6. AdamW 候选示例

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh start --python "$PY" \
  --run-id v13l_adamw_lr001_m09 \
  --set optimizer=AdamW --set lr0=0.001 --set momentum=0.9
```

这改变了首轮共同 SGD 配方，**不是更优推荐或已验证涨点**。原生 AdamW beta1=.9、beta2=.999、eps=1e-8；默认 SGD warmup_momentum 在 AdamW 中不生效，必须保留默认值以显式记录该语义。每个候选换 run-id，不改源码/默认模板，不覆盖已有输出。

## 7. status、tmux 和合法中断后的显式 resume

```bash
cd "$WT"
RUN_ID='v13l_aug_x13_01'
bash scripts/autodl_yolov13l_configurable.sh status --python "$PY" --run-id "$RUN_ID"
tmux list-sessions
tmux attach -t "comparison-yolov13l-$RUN_ID"
```

若当前已经在 tmux，用 `tmux switch-client -t "comparison-yolov13l-$RUN_ID"`；`Ctrl-b` 后按 `d` 只离开界面，训练继续。需要中断时在该训练窗口按 Ctrl-C，随后查看状态/日志及 checkpoint；保留已完整保存的 last。不要给 resume 新的 config、--set、路径或其他模型权重。

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh status --python "$PY" --run-id "$RUN_ID"
bash scripts/autodl_yolov13l_configurable.sh resume --python "$PY" --run-id "$RUN_ID"
```

仍使用同一冻结代码/配置/数据/环境和 last；成功恢复后继续训练和独立 val，不自动 test。已完成、epoch 上限或 patience 已到达、损坏/未成对保存的 best/last、不同配置身份都会拒绝。正常停止后的 val 失败应检查日志并用 finalize，不把完成的训练伪装成继续训练。

## 8. finalize、两种 pack 和真实配置/结果的 Git 归档

确认训练合法完成后补做 test；finalize 不训练、不下载、不拉取新版代码，验证并复用同一个 best 与 val。

```bash
cd "$WT"
RUN_ID='v13l_aug_x13_01'
bash scripts/autodl_yolov13l_configurable.sh status --python "$PY" --run-id "$RUN_ID"
bash scripts/autodl_yolov13l_configurable.sh finalize --python "$PY" --run-id "$RUN_ID"
```

finalize 自动进入本 run 的 tmux；完成后脱离界面，status 的 public_val/final_test 应为 completed。然后打包：

```bash
cd "$WT"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash scripts/autodl_yolov13l_configurable.sh pack --python "$PY" --run-id "$RUN_ID" \
  --out "$WT/outputs/yolov13l-packages/${RUN_ID}_${STAMP}_review.tar.gz"
bash scripts/autodl_yolov13l_configurable.sh pack --python "$PY" --run-id "$RUN_ID" \
  --include-weights \
  --out "$WT/outputs/yolov13l-packages/${RUN_ID}_${STAMP}_weights.tar.gz"
```

第一份为 `.pt` 排除的审查证据，**不是权重备份**；第二份包含真实 best/last。stdout/包内 manifest 明确 scope、含权重状态、原位置、SHA256 和各文件清单。两种模式均只读检查原 run。训练中只允许审查状态/配置快照，明确标 incomplete；正式权重包须合法完成。

`outputs/` 和 `runtime_configs/` 不会自动进入 GitHub。下面创建独立归档 worktree，把真实 run 的最终配置、CLI、args、代码/数据/环境/权重身份及轻量结果生成到 Git 可跟踪路径；保持训练 worktree 的原 SHA 和 clean 状态，以便以后仍能核验该 run。归档不包含 .pt、数据集、缓存或虚拟环境。

```bash
set -euo pipefail
REPO='/root/autodl-tmp/projects/Crack_RTDETR'
ARCHIVE_WT="/root/autodl-tmp/projects/Crack_RTDETR-yolov13l-archive-${RUN_ID}-${STAMP}"
ARCHIVE_BRANCH="codex/yolov13l-archive-${RUN_ID}-${STAMP}"
git -C "$REPO" fetch origin bench/yolov13l-configurable
test ! -e "$ARCHIVE_WT"
git -C "$REPO" worktree add -b "$ARCHIVE_BRANCH" "$ARCHIVE_WT" origin/bench/yolov13l-configurable

cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh archive-config --python "$PY" --run-id "$RUN_ID" \
  --out "$ARCHIVE_WT/docs/comparison/archives/yolov13l/${RUN_ID}_actual.json"

# 上面只生成文件；下面才提交和推送归档分支。
git -C "$ARCHIVE_WT" add -- "docs/comparison/archives/yolov13l/${RUN_ID}_actual.json"
git -C "$ARCHIVE_WT" commit -m "docs(bench): archive YOLOv13-L ${RUN_ID} actual configuration and results"
git -C "$ARCHIVE_WT" push --set-upstream origin "$ARCHIVE_BRANCH"
git -C "$ARCHIVE_WT" rev-parse HEAD
git -C "$ARCHIVE_WT" ls-remote origin "refs/heads/$ARCHIVE_BRANCH"
```

最后两条 SHA 应一致，才是服务器归档已推送的证据。训练 worktree 不提交/拉取改变其冻结 HEAD。首次训练前也可在这样的独立归档 worktree 生成候选计划，给一个尚不存在的 run-id，并显式标 NOT_RUN；例如把上面的 archive-config 改为 `--run-id v13l_new_plan_04 --set lr0=0.005 --out "$ARCHIVE_WT/docs/comparison/archives/yolov13l/v13l_new_plan_04_NOT_RUN.json"`。新候选仍从官方 COCO 开始，生成归档本身不会启动训练。

所有正式结果以实际阶段状态、真实完成轮/best_epoch、停止原因、公共指标与文件哈希为准。合法早停无需补足200轮；test不选 best 或调参。并行GPU日志的耗时不作为论文独占测速。
