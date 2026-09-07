# CBR rescue step 1 — inference-only diagnosis

本工具从 C20 `81d5f0175e0168e022f29feb1ceb54de8a52deb5` 开发。保留 C17 CSCEF-v5.1 结构，只收集 CBR 补救依据；不修改任何现有模型、YAML、损失、后处理、初始化或训练脚本，不创建 optimizer，不执行 prepare/start/smoke 或正式训练。未使用 C21/C22 编号。没有加入 CBR-v2。

入口：`tools/autodl_cbr_rescue.sh run|pack`；Python 入口：`tools/diagnose_cbr_rescue.py run|pack`。包装脚本显式使用现有 `rtdetr` conda 环境，核对 PyTorch 2.1.2 和 CUDA，不安装或升级依赖。Python 入口提供 `--device cpu` 仅用于有明确记录的统一 CPU 诊断，服务器交付命令使用 GPU 0。

## 输入与保护

- 默认主项目 `/root/autodl-tmp/projects/Crack_RTDETR`，通过 `--main` 指定。
- 默认 data：`configs/crack_autodl.yaml`；初始化：`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
- C2/C17/C19/C20 best 的相对路径与本次实施文档一致，见脚本 `RUNS`。可使用 `--c2/--c17/--c19/--c20/--data/--initialization` 显式指定真实文件，不会搜索后自动替换权重。
- 锁定 C20 best SHA256：`6eda2d56209a4e714490ab11297a9a92b9cbc509dc8baa646eb742eac3f208ba`。
- 锁定原初始化 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。仅只读哈希核验，不拿它启动训练。
- 运行开始/结束核对全部 C20 已有受版本控制文件；保存全部当前代码哈希及相对 C20 的差异。CSCEF 两份源码逐一核对 C17 `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`；CBR 核对 C19 `025997e3c51eaf6933534308a95da6ebf97bff53`。保存原始字节 SHA256 和跨平台 LF 规范化 SHA256。
- 权重、data、已有证据文件和完整 val/test 图片/标签读取前后核对 SHA256；缺失标签明确记录 `null`。所有新报告、设置与日志都写在新诊断目录及它的同级 console log；禁止覆盖已存在输出或压缩包。
- 对数据使用原生图片清单、图片读取/拉伸、RT-DETR transforms、`preprocess` 和指标。只覆盖标签元数据读取，复用原生 `verify_image_label`，不读写旧标签 cache；损坏图片报错，禁止自动修复 JPEG。发现旧 `.npy` 图像缓存时中止，请现场核查缓存并提供无此缓存的真实图像路径，不自动删文件。
- 缺失文件、哈希不符、无效样本、重复图像、val/test 路径重叠、输出非有限、模型类型不符都会失败退出；失败 `summary.json` 不能打包成成功结果。

## 统一评估与排序观察

统一参数：imgsz=640、batch=16、conf=0.001、iou=0.7、max_det=300、half=False、augment=False、rect=False、workers=0、device=0、seed=42。未融合 FP32，`eval()` + `torch.inference_mode()`，不使用 autocast，关闭 TF32，记录确定性设置。没有调用会自动融合的 `model.val()`/AutoBackend 外层；直接复用原生 validator 的预处理、`update_metrics/get_stats` 和 `DetMetrics`。P/R 使用官方 best mean-F1 工作点，AP 为官方 10 个 IoU 阈值结果。`iou`/`max_det` 如实保留；此 RT-DETR 原生后处理不做 NMS 或额外 max_det 截断，本模型有300查询。

`raw_tensor` 检查 B×300×(4+nc)，框为归一化 xywh，类别分数已是 sigmoid 概率。先观察 `max(class_scores)` 的排序、阈值掩码及排序后对齐掩码，再把独立 clone 交给后处理；不重复 sigmoid，不重复缩放。`legacy` 直接调用未改动的 `RTDETRValidator.postprocess`；`aligned` 独立 validator 仅修正掩码与排序的对齐。原源码不动。每一路都保存原流程和对齐流程指标，互不污染。

顺序固定：C20 完整 val → C20 完整 test → C17 完整 val → C19 完整 val。只要任一检查发现错位，就补 C2 val/test、C17 test、C19 test；这样四个模型都有同模式的旧/新 val/test。后续比较统一取 `aligned`，旧/新分别存储。若无错位，则选 `legacy`；此时本轮 old/aligned 等价，历史结果保留。本轮未融合与历史融合评估可能有数值差异，不能把这个差异解释为模块收益；没有在历史融合模式复现时，不声称证明其所有浮点边界都一致。

Mask 报告包含真实图像数、总查询数、每图查询数集合、最低分类分数、<=阈值的图像/查询数、错位图像/查询数和至多16张例图（原始分数、排序位置、query 索引、两种 keep 标记）。test 历史864张与实际清单数同时报告；以实际全量清单为准，不硬凑864张。test 只用于公共评估问题，结论函数不读取 test。

## 固定 val 样本与模块观察

用 `--evidence` 显式传入旧 C20 worktree 的 `outputs/c20/evaluation_val/key_predictions_gt.jsonl.gz`。新 worktree 不会带入这些结果。文件存在时必须恰好32张并逐一核对图片与标签哈希，允许按唯一文件名+哈希解析数据迁移后的路径；任何内容或身份不符都报错，不偷偷退回另一批图。文件不存在时按当前 val 绝对路径稳定排序取前32张，先保存清单和哈希，明确写为新清单。所有模型读取同一清单，样本不依赖模型表现。

CSCEF 只统计 C17/C20，通过实际 `CSCEFv51` 对象发现模块，hook 观察真实 lateral、输出及 `lateral_norm` 输出。使用实际规范化 lateral 调用原模块的 Scharr/结构置信度方法，无须替换 forward。记录每图 L/F_out/残差 RMS、`RMS(F_out-L)/(RMS(L)+1e-9)`、结构置信度空间均值、输出投影 Frobenius/L2 范数、非有限检查。结构置信度不是分类概率。逐图 CSV 与均值、中位数、p05/p25/p75/p95/min/max 保存，另写 C20−C17 的描述性变化。所有临时 hook 在 `finally` 中移除，验证异常路径也能移除。

CBR 只统计 C19/C20，按 `RTDETRDecoderCBR` 实际类型定位，不写固定层号。`model.predict(..., cbr_diagnostics=True)` 返回 before/after、tanh_offsets、displacement、aggregation_weights；最后的 raw 分数和原始0–299 query 索引直接保存。普通推理与诊断 after 在第一批、每批固定证据图上逐项比较（atol=1e-6，rtol=1e-5），再使用其输出计算指标。首批验证通过前不会累积指标。

每个 query 的 CSV 保留归一化 before/after xywh、分类概率/类别/query、L/R/T/B 位移（带符号）、左右除以原宽/上下除以原高的带符号与绝对值、tanh 与 abs(tanh)、`abs(tanh)>=0.95`、正/负/零修正、每边三个位置的聚合权重、面积比、中心偏移及归一化偏移。位置顺序与原 CBR 的 `[-.25,0,.25]` 一致。分别统计全部300查询和分类分数>=0.25查询；均值是查询汇总的描述统计，空集合 n=0/mean=null。也保存四边合并的平均绝对 tanh 与饱和率等。水平标注框短边不等于裂缝像素宽度。

固定匹配对全部查询、>=0.25查询分别进行：类别相同，按修正前 IoU 从高到低贪心一对一，数值相同按 GT/query 索引升序；最低 IoU=0，因此零交并比也可配对并明确标记 `<0.5` 低质量配对。匹配只运行一次；after 用完全相同 GT/query。空 GT/无候选产生0对，未匹配数量逐图逐范围保存。IoU delta=after−before，>1e-6 改善、<−1e-6 恶化，其余基本不变。各边绝对定位误差及变化使用归一化图像坐标，负的 error_delta 表示误差减小。这是定位行为描述，与官方 AP 匹配实现回答不同问题。

完整 val 的 C19/C20 before/after 来自同一前向、相同分类分数和 query 索引，分别用官方 P/R、AP50、AP75、mAP50–95 计算。C20 before 仍是联合训练网络的输出，不能当成 C17，也不能当作重新训练的去 CBR 消融。完整 val 与32图固定匹配两类证据都保留。

## 结果与结论

主要文件：

- `summary.json`：完成/失败状态、输入/前后哈希、runtime/commit/代码、全量指标、模块比较和结论。
- `full_val_metrics.json`、`mask_statistics.json`、`module_comparison.json`：统一指标、排序影响和跨模型描述性变化。
- `evaluations/C*_val/report.json`：每模型各路指标、输出一致性、模块分布与固定匹配统计；相应 test 报告只用于公共评估问题。
- `cscef_fixed.csv`、`cbr_queries_fixed.csv`、`cbr_pairs_fixed.csv`、`cbr_matching_images.csv`：各模型目录中的固定图证据。
- `val_manifest.json`、`test_manifest.json`、`fixed_manifest.json`、`source_before.json`、`code_diff.patch`、`console.log`、`framework.log`、`conclusion.md`。

结论先列本轮同口径 val 事实，再给一个优先检查方向。不因平均位移或饱和机械选择 rho、不声称梯度冲突、不同时提出一串训练版本。AP 下降时先核对固定配对误修正幅度是否支持单项约束调整；末端仍有收益且 before 较差时先审查训练连接；无可靠线索则保留 C17，停止组合路线。32图本身不足以确定因果或保证改动涨点。若以后审查梯度，必须区分 P3、query、原始框主路径、宽高缩放路径和匹配/质量目标；仅 P3 detach 不是完全隔离。当前工具不测梯度、不加入 detach，也不自动选定 CBR-v2。

## 服务器命令

最终交付消息会给出固定完整提交 SHA。以下 `REV` 使用实际远端诊断分支解析，执行时核对它与交付 SHA 一致。若 worktree 已存在则退出，不覆盖。

```bash
set -Eeuo pipefail
MAIN='/root/autodl-tmp/projects/Crack_RTDETR'
WT='/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_rescue'
BRANCH='codex/diag-c20-cbr-rescue'
git -C "$MAIN" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
# C17 is not an ancestor of C20: explicitly fetch both module-source commits for hash verification.
git -C "$MAIN" fetch origin \
  '0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139' \
  '025997e3c51eaf6933534308a95da6ebf97bff53'
REV="$(git -C "$MAIN" rev-parse "refs/remotes/origin/$BRANCH")"
printf 'Diagnosis commit: %s\n' "$REV"
test ! -e "$WT"
git -C "$MAIN" worktree add --detach "$WT" "$REV"
test "$(git -C "$WT" rev-parse HEAD)" = "$REV"
git -C "$WT" status --short
```

```bash
set -Eeuo pipefail
WT='/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_rescue'
OUT="$WT/outputs/cbr_rescue_step1"
bash "$WT/tools/autodl_cbr_rescue.sh" run \
  --main '/root/autodl-tmp/projects/Crack_RTDETR' \
  --evidence '/root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr/outputs/c20/evaluation_val/key_predictions_gt.jsonl.gz' \
  --output "$OUT" --device 0
```

如默认路径失效，先 `find '/root/autodl-tmp/projects' -type f -name 'best.pt' -print` 定位，再用对应显式参数；不要用其他 best 替代。哈希不符必须调查真实权重来源。重复运行需要自己选新的 `--output`，旧目录保留。

```bash
set -Eeuo pipefail
WT='/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_rescue'
PKG='cbr_rescue_step1.tar.gz'
DL='/root/autodl-tmp/cbr_rescue_download'
bash "$WT/tools/autodl_cbr_rescue.sh" pack \
  --input "$WT/outputs/cbr_rescue_step1" --output "$WT/outputs/$PKG"
(cd "$WT/outputs" && sha256sum -c "$PKG.sha256")
test "$(stat -c %s "$WT/outputs/$PKG")" -le 20971520
tar -tzf "$WT/outputs/$PKG"
mkdir -p "$DL"
for suffix in '' '.sha256' '.inventory.json'; do
  test ! -e "$DL/$PKG$suffix"
done
for suffix in '' '.sha256' '.inventory.json'; do
  cp -- "$WT/outputs/$PKG$suffix" "$DL/$PKG$suffix"
done
(cd "$DL" && sha256sum -c "$PKG.sha256")
```

打包采用完成报告的显式清单，排除权重、原图、数据集、整个仓库和 runtime 设置。包内 `CONTENTS.json` 和 `MANIFEST_SHA256.txt` 逐文件记录字节数与 SHA256；包外另有 archive SHA256 和 inventory。压缩后超过20 MiB直接拒绝，不静默丢弃证据。下载后可在新解压目录运行 `sha256sum -c MANIFEST_SHA256.txt` 验证内容。

可从 AutoDL 的文件浏览器下载 `autodl-tmp/cbr_rescue_download/` 三个文件。也可在本地 PowerShell 用平台实际显示的 SSH 主机及端口下载；这里不猜连接信息，也不索取密码：

```powershell
$DiagHost = Read-Host 'AutoDL SSH 主机（平台 ssh 命令中 @ 后的主机名）'
$DiagPort = Read-Host 'AutoDL SSH 端口（平台 ssh 命令中 -p 的值）'
$DiagDir = Join-Path (Get-Location) ('cbr_rescue_download_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $DiagDir -ErrorAction Stop | Out-Null
foreach ($File in @('cbr_rescue_step1.tar.gz','cbr_rescue_step1.tar.gz.sha256','cbr_rescue_step1.tar.gz.inventory.json')) {
    scp -P $DiagPort "root@${DiagHost}:/root/autodl-tmp/cbr_rescue_download/$File" $DiagDir
    if ($LASTEXITCODE -ne 0) { throw "下载失败：$File" }
}
$Expected = ((Get-Content -LiteralPath (Join-Path $DiagDir 'cbr_rescue_step1.tar.gz.sha256') -Raw) -split '\s+')[0]
$Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $DiagDir 'cbr_rescue_step1.tar.gz')).Hash
if ($Actual -ne $Expected) { throw 'SHA256 校验失败' }
Write-Output "校验通过：$DiagDir"
```

## 本地验证边界

`python tools/test_cbr_rescue.py` 使用已有环境的 unittest，不要求安装 pytest。检查错位例、阈值边界、原输出不变、独立缩放、固定匹配与空集合、原生预处理一致、官方指标、旧清单哈希、所有四类实际模型的合成推理、异常 hook 清理、源码来源保护、压缩包哈希/排除/拒绝覆盖/大小上限。合成模型只用来验证接口与数据管线，绝不冒充最终 checkpoint 的 val/test。CUDA 可用时另核对 C20 的640输入、300查询以及普通/诊断输出一致性。

本地最终验证记录在 `docs/cbr_rescue_local_validation.json`。服务器 PyTorch2.1.2/4090、真实权重哈希、实际图像数、真实 full-val/test 指标、原32图身份及真实证据包大小，只有服务器成功执行后才算验证。没有真实诊断结果前，不确定任何 CBR-v2 改法或收益。
