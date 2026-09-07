# CBR 最后一次小规模梯度检查

本步骤基于第一步提交 `d73391f1b8496b16d9b6ba06429c6393b1a21eda`，只读 C19/C20 的训练计算图；不重复完整 val/test，不启动正式训练，不更新参数、BN、EMA，不创建 optimizer 或 GradScaler，不改原 CSCEF/CBR 源码、模型 YAML、loss 或训练配置。B 路是临时梯度探针，不是已经选定的 CBR-v2。

已阅读并沿用 [C19_CBR.md](C19_CBR.md) 的真实连接、原生损失权重与 detach 边界，以及 [CBR_RESCUE_DIAGNOSTICS.md](CBR_RESCUE_DIAGNOSTICS.md) 的读取、哈希和小包能力。没有读取或移植本轮不需要的外部模块包代码。

## 第一步证据

本地实际核验 `cbr_rescue_step1.tar.gz`：7,986,300字节，SHA256=`8b90e4e28f42ca44581271b9820c8cb5d5a14f427a5054b02f78f6038be05a9f`，24个清单成员大小和哈希全部匹配。包内 completed summary SHA256=`44f1272ab52408fefb8f4ceb663eaa41fc965654f767939533df1220f1472523`。运行时要求读取同一 summary，从其中获取 C19/C20/data 哈希并核对真实文件；不替换 checkpoint。

C19 best SHA256=`1a5a850c5cb1dadaed34ebff86b20c3f31312f44ed7839149864b3befab19f3d`；C20 best SHA256=`6eda2d56209a4e714490ab11297a9a92b9cbc509dc8baa646eb742eac3f208ba`。输入路径可用 `--c19/--c20/--data/--step1-summary` 显式定位，但内容身份不能绕过。

第一步同口径完整 val：C19 before/after mAP50–95=41.4150%/50.9321%，C20=42.1262%/49.1636%，C17=52.5060%。CBR 末端总体有效，没有据此缩小 rho、加限幅或改 CSCEF 的依据。CBR 前的框没有额外独立最终层监督，因此不能把其 AP 较低直接解释成特征被破坏。第一步不含梯度证据，本工具不把它改写成梯度冲突结论。

## 固定输入与状态保护

从原生 train 文件清单解析路径，稳定排序后，用 Python `random.Random(42).sample(range(N),16)` 一次无放回抽取，保留抽取顺序分4批，每批4张；不根据标签数量或模型表现改选。只对这16张图片构建已有的只读 RT-DETR dataset，保留原生 resize/format/collate/preprocess，无在线增强。先写完整排序路径清单及其 SHA256、选中图片/标签 SHA256 和明确批次顺序。两个模型使用同一清单，无 val/test 推理。

服务器条件固定：640、batch4、FP32、无增强、workers0、TF32关闭，使用既有 rtdetr 环境 PyTorch2.1.2+CUDA。正式训练的 batch16/在线增强/109字段配方不变；本次 batch4 不作为训练方法变化或性能实验。

加载时 `fuse=False`。模型、CBR head、deformable decoder 三个路由对象设 training=True；其余模块全部 eval，包括 BatchNorm、Dropout、MultiheadAttention 和 RNN。这保留原生三层训练输出、DN 和辅助损失，同时排除 BN 更新和 dropout 噪声。记录精确模式和所有参数原始/诊断期 requires_grad，内存中开启所有浮点参数梯度，退出后恢复原标记，不把 stripped checkpoint 的冻结标记当成训练配置。若 checkpoint 缺少原生 trainer 通常赋予的 `model.nc` 元数据，仅从实际 head.nc 补齐并记录，仍核对 names/head/model 一致。

每个模型保存全部参数、registered buffers 和既有 `.grad` 的逐张量 dtype/shape/SHA256；每批结束及模型结束都验证完全未变。全部训练标记和 requires_grad 也恢复。外部权重/data/第一步 summary 及16张图片/标签前后核对哈希。继承全部 C20 源码保护，并额外保护第一步7个文件。只增加本步骤工具和文档。

## 同一主网络计算图的 A/B

每模型每批 seed=`42+batch_index`（42、43、44、45），设置 Python、NumPy、Torch CPU/CUDA RNG，记录状态指纹。通过类型寻找 `RTDETRDecoderCBR`，P3生成模块从实际 `head.f[0]` 获取，最后 decoder/原框头从实际 `head.decoder.layers[-1]`/`head.dec_bbox_head[-1]` 获取。C19 与 C20 的模型层号不写死。

只执行一次 `model.predict(img,batch=targets,cbr_diagnostics=True)`；临时 hook 捕获实际 CBR 输入 P3/query/boxes，并核对 P3 生成模块和最后 decoder 的对象身份。记录 decoder 输入的 DN noisy boxes、DN embeddings、attention mask 和 dn_meta，随后移除这些 hook。

- A：原生训练输出和原生 `model.loss(batch,preds=...)`。
- B：只调用同一个 `head.cbr(p3.detach(),query.detach(),boxes,return_diagnostics=True)`，在返回的训练输出中替换最后一层 refined boxes，其余层、分类 scores、encoder 和 DN 对象复用。

CBR 的 rho=.10、normal_fraction=.10、36采样点、MLP/聚合和原坐标公式不变。boxes 保持原始 tensor/计算图，采样几何与宽高条件原先的 detach 不动，原框 identity 和 w/h 位移缩放梯度保留；CBR 内部参数仍启用梯度。B 不切断 query/P3 经其他主干路径或原始框产生的梯度，不能称为与共享网络完全隔离。

调用原生 model.loss 两次，复制输入 tensor/storage 时保留计算图。检查原输入和副本均未被原地修改。临时观察原生 criterion 的每层 loss、最终12项 loss、所有 Hungarian 匹配、DN匹配和实际分类 target/IoU质量 target，不替换任何损失或匹配数学。正样本例子中常规 matcher 4次（encoder+3层 decoder），DN 3层使用原生固定 DN 配对。

以下必须逐项、逐位一致，否则在求导前中止：原始框/logits/encoder/DN、CBR before/after/tanh/displacement/aggregation、全部 loss、全部匹配及分类质量目标。B 与 loss 之后 RNG 必须未变；没有重新生成 DN。两模型还核对同批 DN 元数据、初始 noisy boxes 和 attention mask 一致；DN embeddings 来自各自已学权重，不要求跨模型相同。所有临时 hook/方法包装都在 finally 中恢复，包括异常路径。

## 梯度与噪声

使用 [`torch.autograd.grad`](https://docs.pytorch.org/docs/2.1/generated/torch.autograd.grad.html) 读取梯度，`create_graph=False`、`allow_unused=True`，不累计到参数 `.grad`。首次 A 读取保留图，首批额外重复 A 一次测数值噪声，最后 B 读取释放共享图。不增加批次、重复采样或受控第二次主网络前向。每模型4次主网络前向、8次 loss、9次 gradient read。

统计 `g_all=grad(L_A)`、`g_keep=grad(L_B)`、`g_condition=g_all−g_keep`。后者只是在本次等价前向/损失下被移除条件路径的贡献，不是另加损失的梯度。[`detach`](https://docs.pytorch.org/docs/2.1/generated/torch.Tensor.detach.html) 共享数据存储，因此脚本禁止原地改输入。

预先固定的角色组：最终 Neck P3 生成模块、最后 decoder 层、C20 的 CSCEF 输出投影和 CSCEF 全参数、CBR 自身、原最后框预测头，以及 CBR 实际输入 boxes。CSCEF 两组重叠，单独报告且绝不相加；C19 CSCEF 记不适用。不在不同模型的参数空间之间计算余弦，仅按相同模块角色比较各自的标量统计。

每组以 float64 计算三个范数、condition/keep 比、condition·keep、condition/keep余弦、零/unused张量数、有限性和首批重复噪声。近零门槛为 `max(1e-12,10*首批A重复差范数,1e-7*max(norm_all,norm_keep))`；condition 或 keep 不超过门槛时余弦记 null，不机械解释方向。后3批复用本模型该组首批噪声估计，并明确标记来源；一次重复不是正式置信区间，也不保证界定所有批次噪声。

CBR 参数、原框头、boxes 的 A/B 梯度逐元素满足固定 FP32 容差 atol=1e-7、rtol=1e-5，CBR 还逐参数输出范数、零/unused和差异。需要观察到 B 的 CBR/原框头/boxes 非零梯度才能确认本次输入具有可学习证据；空 GT 会明确产生无框损失/unused状态，不直接宣称冻结。四批仍无法确认则输出“诊断实现/输入问题”。

额外用真实 `grad(L_B,refined_B)` 验证原始 boxes 的局部 Jacobian。例如

`dL/dw = v_w*(1+rho*(t_R-t_L)) + v_cx*rho*(t_L+t_R)/2`。

高度同理，cx/cy保留 identity。这样不仅核对非零梯度，还验证宽高缩放路径未被顺手 detach。CUDA grid_sample backward 存在数值不确定性，保留原生警告并记录首批重复读数，不循环尝试直至“显著”。

## 预先固定的有限判读

输出只取“有尝试依据”“没有明确依据”“诊断实现/输入问题”之一。任何前向/loss/匹配/DN/状态不等价，必须停止并报告第三类，不能据不等价梯度作建议。

为避免看到真实数据后选组/阈值，工具提交时固定一个保守的工程筛选规则；它不是通用的梯度冲突标准，也不用负余弦阈值证明原因。主要比较角色预先限定为 Neck P3 和最后 decoder。一个角色需要至少3/4批同时满足：

1. C20 condition 与 keep 均超出所记录的噪声门槛；condition/keep >=.10。
2. C20 的反向投影比例 `−dot(condition,keep)/(norm_keep²+1e-12)` >=.05。
3. 该比例比同批 C19 至少大 .05，同时大于两模型相对噪声门槛之和；C19 keep 本身也必须可辨识。

再确认两模型 B 路 CBR/原框均有非零梯度，才标“有尝试依据”；含义只是值得唯一一次受控组合训练，不能证明200轮下降的原因或保证收益。原始全部组/批次读数会保留，工程界线附近仍应谨慎判断。C19也有相同现象、差异小、同向、噪声难区分或只有一次负余弦，均不据此认定组合特有问题。无明确线索则建议停止补救、保留 C17，不新增幅度版本或修改 CSCEF。

如果真实结果支持候选，未来只考虑 P3/query 条件分支限制回传，CSCEF 和 CBR 前向/rho不变。仍按 C2 权威109字段及统一 ImageNet 初始化，只做一次组合训练，先看同口径 val 是否超过 C17；有效后才补单模块。当前入口没有任何训练子命令。

## 输出与运行

主要输出：`summary.json`、固定 train清单、`gradient_groups.csv`、每模型/批次完整小型统计、`state_before/after.json`、A/B所有输出/匹配/DN指纹及 loss、CBR逐参数梯度摘要、原框 Jacobian、重复噪声、runtime/commit、源码差异、日志和三类结论。仅保存哈希和小型统计，不保存梯度数组、图片或权重。`pack` 复用第一步白名单打包，拒绝未完成结果、覆盖、目录穿越和压缩后>20 MiB；包内成员清单+SHA256，包外 archive SHA256+inventory。

服务器准备：最终交付消息提供完整固定 SHA；下面从诊断分支读取 REV，可与交付 SHA 核对。C17不在C20祖先链中，必须显式获取来源对象。

```bash
set -Eeuo pipefail
MAIN='/root/autodl-tmp/projects/Crack_RTDETR'
WT='/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_gradcheck'
BRANCH='codex/diag-c20-cbr-gradcheck'
git -C "$MAIN" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git -C "$MAIN" fetch origin \
  '0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139' \
  '025997e3c51eaf6933534308a95da6ebf97bff53'
REV="$(git -C "$MAIN" rev-parse "refs/remotes/origin/$BRANCH")"
printf 'Gradient commit: %s\n' "$REV"
test ! -e "$WT"
git -C "$MAIN" worktree add --detach "$WT" "$REV"
test "$(git -C "$WT" rev-parse HEAD)" = "$REV"
```

```bash
set -Eeuo pipefail
WT='/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_gradcheck'
bash "$WT/tools/autodl_cbr_gradcheck.sh" run \
  --main '/root/autodl-tmp/projects/Crack_RTDETR' \
  --step1-summary '/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_rescue/outputs/cbr_rescue_step1/summary.json' \
  --output "$WT/outputs/cbr_gradient_step2" --device 0
```

包装脚本激活现有 conda rtdetr，核对 torch2.1.2 和 CUDA，不安装依赖。缺失权重需现场定位并传 `--c19/--c20`，哈希不符应调查来源，不用替代 checkpoint 继续。不存在 imgsz/batch/样本数扫描参数。输出存在就退出，保留第一次检查证据，不自动重试。

```bash
set -Eeuo pipefail
WT='/root/autodl-tmp/projects/Crack_RTDETR_diag_c20_cbr_gradcheck'
PKG='cbr_gradient_step2.tar.gz'
DL='/root/autodl-tmp/cbr_gradient_download'
bash "$WT/tools/autodl_cbr_gradcheck.sh" pack \
  --input "$WT/outputs/cbr_gradient_step2" --output "$WT/outputs/$PKG"
(cd "$WT/outputs" && sha256sum -c "$PKG.sha256")
test "$(stat -c %s "$WT/outputs/$PKG")" -le 20971520
tar -tzf "$WT/outputs/$PKG"
mkdir -p "$DL"
for suffix in '' '.sha256' '.inventory.json'; do test ! -e "$DL/$PKG$suffix"; done
for suffix in '' '.sha256' '.inventory.json'; do cp -- "$WT/outputs/$PKG$suffix" "$DL/$PKG$suffix"; done
(cd "$DL" && sha256sum -c "$PKG.sha256")
```

AutoDL 文件浏览器可以直接下载 `autodl-tmp/cbr_gradient_download/` 三个文件。或在本地 PowerShell 使用平台实际 SSH 地址和端口，不猜地址、不提供密码：

```powershell
$GradHost = Read-Host 'AutoDL SSH 主机名'
$GradPort = Read-Host 'AutoDL SSH 端口'
$GradDir = Join-Path (Get-Location) ('cbr_gradient_download_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $GradDir -ErrorAction Stop | Out-Null
foreach ($File in @('cbr_gradient_step2.tar.gz','cbr_gradient_step2.tar.gz.sha256','cbr_gradient_step2.tar.gz.inventory.json')) {
    scp -P $GradPort "root@${GradHost}:/root/autodl-tmp/cbr_gradient_download/$File" $GradDir
    if ($LASTEXITCODE -ne 0) { throw "下载失败：$File" }
}
$Expected = ((Get-Content -LiteralPath (Join-Path $GradDir 'cbr_gradient_step2.tar.gz.sha256') -Raw) -split '\s+')[0]
$Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $GradDir 'cbr_gradient_step2.tar.gz')).Hash
if ($Actual -ne $Expected) { throw 'SHA256 校验失败' }
Write-Output "下载并校验通过：$GradDir"
```

## 本地验证边界

`python tools/test_cbr_gradients.py` 使用已有 unittest，验证独立 CBR 叶子张量只切断条件梯度、CBR和boxes仍可学习；实际 C19/C20 原生 DN/辅助/VFL图、一次predict与精确loss/匹配、一致性失败拒绝求导、异常hook/标记恢复、空GT说明、四批固定顺序与文件/参数/buffer保护、噪声/判读规则和复用小包。CUDA可用时验证合成C20的640输入、batch1及首批重复噪声。

所有本地模型/图片均为明确标注的合成工具夹具，不是实际训练 checkpoint。本地 torch2.7.1/RTX2060 通过不等于服务器 torch2.1.2/RTX4090、batch4、最终 C19/C20 权重通过。详见 `cbr_gradient_local_validation.json`。真实梯度结果尚未取得时，不给出“值得训练 CBR-v2”的实测结论。
