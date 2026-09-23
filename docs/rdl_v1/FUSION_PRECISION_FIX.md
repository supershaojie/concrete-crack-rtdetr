# RDL 严格融合检查的局部精度控制

## 依据和范围

用户提供的服务器对照：`NVIDIA_TF32_OVERRIDE=0 bash tools/diagnose_rdl_v1_fusion.sh` 后，
所示样本 top-k changed_positions=0、分数最大扰动约 5.96e-7、连续比较失败项为空，mother_exact=true。
此前母版/RDL 在未融合和已融合状态各 39 项比较误差为 0，共同完整 head 输入的 30 项比较误差也为 0。
这些是服务器回传证据，不是本地代跑服务器；完整默认报告继续留在原目录。

问题在于此前“FP32”断言只保证 tensor dtype 和 autocast 状态，没有隔离 cuDNN 的 TF32 运算路径。
这个精度控制缺口会使严格融合比较受低精度扰动及候选选择敏感性影响。母版控制不支持将其归因于 RDL 模型/损失回归。
本次按证据只修复工程检查作用域；仍需服务器以新提交完成 preflight，不能用本地 PASS 替代服务器容量检查。

## 改动

- `check_ema_fusion` 对独立副本执行 FP32/eval 检查；在这一作用域临时令 cudnn.allow_tf32、cuda.matmul.allow_tf32=false，关闭 CPU/CUDA autocast。
- 上下文在正常/异常退出时恢复原 TF32、矩阵乘法精度、autocast enabled/dtype/cache 设置。
  特别保留 `float32_matmul_precision=medium`，避免仅还原 allow_tf32 布尔值时意外变成 high。
- 检查内所有融合副本（含母版重放）在 fuse 后再次 eval。断言同一设备/FP32 输入与权重，caller 模型不被融合或修改。
- 原 ordered assertion、atol=rtol=3e-5、原候选身份诊断、固定 ID/共同 head 输入和母版对照保留。
- `precision_scope.before/inside/after/restored` 在正常和异常路径均记录；日志打印，显式诊断写入 fusion.json，普通 preflight 的 real_model 结果也保留。
- 正式容量检查前强制确认当前配置与融合检查进入前相同；容量报告记录 AMP 前/内/后的实际设置。恢复失败会阻止容量检查。
- 覆盖当前 `preflight.json` 前先逐字节保留 `preflight_history_<UTC>.json`，新报告记录旧文件路径与 SHA256。独立诊断仍用唯一目录；旧 FAIL、log、fixture 不覆盖。

LIF/CBR/RDL、公共未训练初始化、200e/B16/640 配方、正式推理策略及训练启动脚本均未修改。
没有向正式脚本写入 NVIDIA_TF32_OVERRIDE，没有启动训练，没有更新 ROR 的任何状态。

## 实际验证

本地 Python 3.9.25 / torch 2.7.1+cu118 / RTX 2060：

- 精度恢复 24 组通过：highest/high/medium × cuDNN TF32 开关 × 外层 autocast 开关 × 正常/异常退出。
  实测严格作用域 GEMM 为 FP32，退出后 enabled/dtype/cache 和 TF32 设置完全相同。
- 原 CUDA real_model 生命周期加完整融合诊断通过，输出最大绝对差 7.450580596923828e-8。
  候选 ID 集合和顺序相同，各连续比较失败数为 0，母版各张量精确一致，融合后所有模块 training=false。
- 局部 CBR 故障注入仍由原断言拒绝（max abs 0.020596489310264587），异常后设置恢复，原模型状态未改变。
- preflight 边界测试通过：容量入口看见恢复后的设置；恢复失败时不会调用容量；旧 FAIL 字节和 SHA 保留。
  此测试用容量替身及 CUDA 8×8 AMP 前后向，**不是 B16/640 容量 PASS**。
- 本地原配置：cudnn TF32=true、matmul TF32=false、autocast=false；检查内均 false；退出后与进入前完全一致。

机器可读结果见 `fusion_precision_validation.json`；旧 `fusion_local_evidence.json` 保留原始结果。
本轮只验证改动影响的部分，没有重跑无关实验或正式长训。服务器 B16/640/native AMP 必须由以下 preflight 实际执行。

## 已有服务器 worktree 更新和预检

在已停止检查/训练的旧工作树执行。最终回复提供可直接复制、已填新 SHA 的命令；这里的 rdl_new 填最终回复的 40 位 SHA。

```bash
set -euo pipefail
rdl_main=/root/autodl-tmp/projects/Crack_RTDETR
rdl_wt=/root/autodl-tmp/projects/Crack_RTDETR-rdl_v1
rdl_old=b19e58c3e2461d9606423656376662442c727593
rdl_new=填写最终回复中的完整新SHA
test -f "$rdl_wt/.git"
test "$(git -C "$rdl_wt" rev-parse HEAD)" = "$rdl_old"
test -z "$(git -C "$rdl_wt" status --porcelain --untracked-files=no)"
rdl_main_head="$(git -C "$rdl_main" rev-parse HEAD)"
git -C "$rdl_main" fetch origin exp-rtdetr-r18-lite-rdl-v1
test "$(git -C "$rdl_main" rev-parse FETCH_HEAD)" = "$rdl_new"
git -C "$rdl_main" merge-base --is-ancestor "$rdl_old" "$rdl_new"
rdl_backup="$rdl_wt/outputs/before_strict_fp32_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$rdl_backup"
cp -p "$rdl_wt/outputs/rdl_v1/"{preflight.json,prepared.json,environment.json} "$rdl_backup/"
cp -p "$rdl_wt/outputs/rdl_v1_delivery.json" "$rdl_backup/"
git -C "$rdl_wt" checkout --no-overwrite-ignore --detach "$rdl_new"
test "$(git -C "$rdl_wt" rev-parse HEAD)" = "$rdl_new"
test "$(git -C "$rdl_main" rev-parse HEAD)" = "$rdl_main_head"
bash "$rdl_wt/tools/sync_rdl_v1.sh" "$rdl_new"
bash "$rdl_wt/tools/rdl_v1.sh" prepare
bash "$rdl_wt/tools/rdl_v1.sh" preflight
```

不要对这次 preflight 添加全局 NVIDIA_TF32_OVERRIDE=0；需要在恢复后的原配置验证容量。
之前仅作为单次命令前缀使用该变量不会影响后续命令。若另行 export 过，应先撤销那个临时实验设置。
prepare 刷新新提交身份，preflight 才验证新提交；旧 prepared/preflight 不能冒充新 SHA 的授权。
旧 sync 不负责更新已有 HEAD，所以明确 checkout 在前、sync 在后。预检不会创建训练 tmux 或启动长训。

如需完整严格 FP32 候选轨迹，可单独运行 `bash tools/diagnose_rdl_v1_fusion.sh`，不加全局 TF32 环境覆盖。

## start 行为

只有用户随后明确调用 `bash tools/rdl_v1.sh start`，且新提交全部 gate 通过，才会创建独立后台
tmux 会话 **rdl-v1-training**（`tmux new-session -d -s rdl-v1-training ...`）。同名会话已存在则拒绝重复启动。
本轮没有执行 start/resume；ROR 待解决项保持原状态。
