# 零权重 CUDA 独立更新检查：结论与有界复核

## 当前结论

**未发现 ROR 零权重生产接线错误；实际观察到母版原生 CUDA 反传非确定性。但服务器原 10 个参数的超界是否全部由近零梯度符号翻转造成，现有证据仍不足。继续保留 REVIEW_REQUIRED，不放宽门槛。**

本次只新增更新诊断、针对性测试和证据，没有修改生产模型、ROR/LIF/CBR、初始化、Trainer、正式训练配方、AMP/TF32 开关、已有预检或 start 门禁。没有再运行融合、AMP、数据评估或完整预检。没有启动正式训练、创建/停止 tmux 会话。后续正式训练仍须显式 start/resume，以独立的 `ror-v1-training` tmux 会话运行。

## 服务器原始证据

输入文件 `D:/rtdetr跑结果/ROR损失函数/ror_latest_diagnostic.zip` 原样保存为 `update_diagnostic/server_prior.zip`，SHA256：

`19b7d167a1e6811acb4db02e8f4ada6ffd9adb9fdd3bbb93aa5199505372dd4c`

`server_prior_manifest.json` 记录每个成员的原始字节哈希。仅读取 JSON，没有执行包内附带的 Python 文件。其版本为 `f15be2dfe39f18a5ec012a0e378e4c56042ba645`，Python3.10.13 / torch2.1.2+cu121 / RTX4090。

包内确认：native 融合失败统计与原 fixture 精确复现；母版/ROR 共有误差在临时禁用 TF32 后消失，并恢复了后端。独立更新仍有 10 个参数超过各自的原门槛。最大 ROR/母版差 0.0009998567402362823，母版自身重复最大差 0.00099988654255867。两者全局最大值接近不能证明同一参数、同一坐标由同一原因造成，更不能据此放行。原报告没有逐坐标原始/裁剪梯度，无法反推出全部符号变化。

旧检查母版重复反传前只清空梯度和恢复 RNG，没有先恢复最初的全部 BN buffers；其后才恢复一次反传后的 buffers。训练 BN 通常不依赖 running mean/variance 计算本次输出，**没有证据证明这就是更新超界根因**，但旧协议不满足本次严格独立性要求。新诊断每次完整恢复并验证内容哈希；不改写旧检查结果。

## 零权重生产路径核查

- `RTDETRDetectionModel.loss` 仅在 training 且 `ror_weight>0` 时请求 CBR 诊断。零权重仍走原 `predict(img,batch=targets)`，原 DN split、encoder 拼接和 `sum(loss.values())`。
- `RORLoss.forward` 在有效权重为零时直接调用父损失并返回原字典，没有 `matched_loss`、额外最终匹配、`loss_ror`、`0*loss_ror` 或连到总损失的零张量。额外 `isfinite(...).all()` 是检查分支，不连接返回损失的反传图。
- 当前 `RTDETRDetectionLoss` 的默认 `final_match_indices=None` 与母版原匹配等价；aux/DN 的匹配和归一化代码未变。没有发现需要进行生产修复的具体接线错误。
- 对真实 criterion 的 e=0/e=5 测试，包含两张图不同 GT 数和 DN positives；比较固定母版源码与 ROR 的损失项、匹配结果、连接计算图和预测张量梯度，全部精确相等。测试会拒绝调用零权重 `matched_loss`。另有测试证明计算图指纹能识别人为添加的 `0*p.sum()` 分支。
- 完整模型的六次本地反传中，预测、所有原 L0 项、最终/aux 匹配、1790 节点的有序连接图、反传前 RNG、前向后的 buffers 均一致。诊断明确阻止零权重请求 `forward_with_diagnostics`。

## 新诊断协议

入口 `tools/ror_v1_update_diagnostic.py`。母版 tasks.py 与 loss.py 从固定 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 加载；共享生产模块必须仍与母版一致。

固定 B2/160、原统一初始化源、原旧检查 GT/RNG 构造、FP32、原 AdamW 分组、lr=0.0005、betas=(0.937,0.999)、原按组 weight_decay、eps=1e-8、clip_norm=10。这个 FP32 首步对照不是正式 AMP 训练，不替代已完成的 AMP 检查。只调用原 seed/deterministic 初始化，保留现有 TF32/cuDNN 设置，前后验证后端相同。

先在母版做一次无梯度前向以构造旧检查开始时的 BN 状态，复制给 ROR。随后按 mother_0、ror_0、mother_1、ror_1、mother_2、ror_2 交替执行，**各恰好 3 次独立反传**。每次前向/反传试验前：

1. 恢复同一初始参数、state_dict 及全部 registered buffers（含 nonpersistent buffers），清除梯度。
2. 恢复同一优化器 state_dict 和有序参数分组；这里是首步，初始 optimizer state 为空。
3. 恢复 Python、NumPy、CPU 和所有 CUDA RNG，逐项验证内容哈希。
4. 跑原损失和 backward，保存原始梯度到内存；使用原 `clip_grad_norm_`，记录真实返回范数与裁剪系数，再保存实际裁剪梯度和更新后参数到内存。所有大张量只驻内存，不落盘。

逐参数保留母版/ROR 9 种交叉比较的最大差，以及母版内部 3 对重复、ROR 内部 3 对重复各自的最大更新差。**门槛固定读取原诊断包的逐参数 bound，不重新估计、不扩大 4 倍系数。** 对原 10 个失败参数与本次新超界参数，每项最多保存 3 个坐标；历史参数若没有复现超界则明确标记，并保留当前差异最大的少量坐标。母版的三次取值范围只作描述，不用作新容差或统计置信区间。

每个坐标包含六次试验的更新前/后参数、实际更新、原始/裁剪梯度及符号、clip 系数与范数、lr/eps/weight_decay/betas、`abs(g_clip)/eps`、相对该张量最大梯度的量级。还记录最差交叉对与同坐标母版/ROR 重复对的实际更新差、FP64 首步公式预测差和残差，以及是否符号翻转、母版重复是否跨零、取值范围是否重叠。保留量化证据，不预设所有差异都是符号翻转。

单独的 `same_gradient_optimizer` 恢复相同初始模型/优化器状态，将 mother_0 的原始梯度逐位复制到两套模型，分别调用原裁剪和 AdamW；严格比较裁剪梯度、更新参数、优化器状态。**该项只检查优化器接线，不能把 independent_backward 标为 PASS。** 六次 trial 的 PASS 仅表示该次运行及自己的 AdamW 解析式通过，不表示两次独立更新一致。

每次试验和最终汇总都保存 JSON；异常保留已完成试验、出错阶段和 traceback。原报告、原 plan/preflight 状态不会被覆盖。

## 实际本地结果及边界

本地为 RTX2060 / torch2.7.1+cu118，不能替代服务器 RTX4090 / torch2.1.2+cu121 的逐坐标复现。原始本地结果保存在 `update_diagnostic/local_diagnostic.json`；简表见 `local_summary.json`，其哈希关联原始报告。报告来源版本是本次诊断开发时的工作树，记录了实际源文件哈希；最终仅增加了阶段、结论与状态作用范围等说明字段，没有重新进行六次测量。

| 检查 | 结果 |
| --- | --- |
| 六次初始参数/buffers/RNG/optimizer/分组哈希 | 全部一致 |
| 零权重完整模型前向/L0/匹配/连接计算图 | 全部精确一致 |
| 每次使用实际裁剪梯度的 AdamW 首步解析式 | 6/6 通过原 atol=2e-7、rtol=2e-6 |
| 相同梯度输入的独立优化器 | 参数、裁剪梯度、optimizer state 逐位一致，最大参数差 0 |
| 母版自身梯度变化 | 226 个参数张量；有原 `grid_sampler_2d_backward_cuda` 非确定性警告 |
| 原服务器 10 项门槛的本地复现 | 本地这次未超界，不等于服务器问题解决 |
| 其他新超界参数 | 2 项，继续报告失败 |
| 逐坐标样本 | 12 个参数共 32 个坐标；此次采样没有梯度符号翻转 |
| 针对性单元检查 | 5 项通过；没有重跑不受影响的预检 |

具体反例：`model.22.m.1.conv1.bn.weight[99]`，更新前 1.0，wd=0，lr=0.0005，eps=1e-8。

| 量 | mother_0 | ror_2 |
| --- | --- | --- |
| 原始梯度 | -6.3521361e-7 | -5.4506779e-7 |
| clip 系数 | 0.0141084790 | 0.0141084772 |
| 实际裁剪梯度 | -8.9618979e-9 | -7.6900761e-9 |
| 更新后参数 | 1.0002362728 | 1.0002173185 |

两者**同为负号**，但裁剪梯度与 eps 同量级，`g/(abs(g)+eps)` 对变化敏感。实测更新差 1.8954277e-5，公式预测差 1.8957661e-5，差异残差约 -3.38e-9；同坐标母版重复更新差 4.1723251e-6。另一个新超界为 `model.25.cv3.bn.bias[113]`，实际差约 3.8777944e-7，预测约 3.8768158e-7，同样没有符号翻转。不得把它们改写成“所有超界都是近零梯度变号”。

因此，生产零权重路径和同梯度优化器接线有通过证据；本地母版原生反传波动也有直接证据。服务器原 10 项的全部具体机制仍须下面一次同机运行，总体保持证据不足/REVIEW_REQUIRED，不用两个全局最大差相近进行放行。

## 服务器一次有界命令

先将原分支 fast-forward 到交付完整 SHA（可用既有 `tools/sync_ror_v1.sh`），然后运行：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ror_v1
PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false timeout 900 \
  /root/miniconda3/envs/rtdetr/bin/python tools/ror_v1_update_diagnostic.py \
  --source /root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt
```

默认读取随提交保留的 `docs/ror_v1/update_diagnostic/server_prior.zip`，覆盖本次已知 10 项，并自动扫描新超界项。固定 6 次反传和两次共享梯度的 optimizer step；不包含 epoch 循环、融合或整套预检，外部上限 900 秒。输出在新建的 `outputs/ror_v1/update_control_<UTC>/diagnostic.json`，通常不足 2 MiB，只有统计和少量坐标，没有大权重或完整梯度文件。计算时约需 1.5 GiB 主存存放六次试验的参数和梯度，结束释放。

正常完成仍以退出码 **2 / REVIEW_REQUIRED** 保留原问题；异常以失败退出并保存详情。先看 `zero_path_audit`、`same_gradient_optimizer`、`analytic_steps_all_pass`，再看 `independent_backward.samples` 的同坐标反传与更新证据。此命令不会启动正式训练，也不会修改原门禁。
