# ROR v1 实际检查与待办

本地执行目录：`D:/MyProjects/Crack_RTDETR/outputs/worktrees/Crack_RTDETR-ror_v1`。
解释器：`D:/miniconda3/envs/rtdetr/python.exe`；Python3.9.25 / torch2.7.1+cu118；RTX2060。
导入仓库内自定义 Ultralytics。运行时母版 HEAD 为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`，实现尚未提交；`source_audit.json` 记录具体实现内容哈希，最终交付 SHA 另存外部记录。没有安装/升级环境，没有找到适用 AGENTS.md；有限范围内未定位独立迁移包，已直接使用成功母版完整本地归档。

## 状态（不能合并为“全部预检通过”）

| 检查 | 状态 | 证据/限制 |
|---|---|---|
| 公共初值与成功母版 best SHA | PASS | 两个 SHA 均与任务书相同 |
| 数据图数/GT/路径清单/标签指纹 | PASS | 三 split 完全等于成功组合归档；没有重新划分 |
| 受控 nc80 初始化、nc1 适配、参数名称 | PASS | 原工具审计；仅明确列出的9个分类 keys 形状改变；ROR 参数/buffer=0 |
| 完整109字段配方差异 | PASS（本地归档） | 已对成功组合保存的权威 C2 args 核对；服务器实际文件仍由 prepare 再核验 |
| 数学例子、边界、归约、独立参考循环、梯度 | PASS | 例子0.06433845，梯度-0.09/+0.09；FP32、空集合、端点、极大有限logit、多图/类别 |
| 最终匹配复用 / DN / aux | PASS | 受控fixture一次最终+3次aux匹配；新项只出现一次且非零；真实B2不同GT数、DN有无、空/单GT |
| CPU 零权重对照 | PASS | 原母版源码预测、原损失、梯度、AdamW一步更新逐位一致；539个state发生有效更新 |
| 正权重路径的原L0与诊断预测 | PASS | CPU/CUDA均与原母版输出逐位一致；非零fixture也保留所有L0项 |
| CUDA 直接 optimizer 更新一致性 | **NOT_PASSED / REVIEW_REQUIRED** | 见下文；不把CPU或解析式通过替代成CUDA直接一致性通过 |
| CUDA 原生 AMP B2/160 | PASS | 原GradScaler动态回退后一次有效更新；真实EMA半精度验证及L0有限；非正式AP评估 |
| 保存/加载/resume / 真正get_model | PASS | 稳定criterion类随标准模型序列化；调用实际重建方法和原resume恢复；epoch6存档恢复e7，e20权重0.1；EMA同步 |
| 结构、正常推理和fusion | PASS | 参数20,149,765 / fused19,944,965；CPU发生候选置换，经母版同ID与共同输入重放通过；CUDA自然输出通过 |
| 已训练母版固定 train/val 现象 | OBSERVED_SUPPORT | 每split128张；具体数字/样本/原始量级见diagnosis.json及sample JSON；不证明AP收益 |
| Python编译、CLI帮助、Bash语法 | PASS | compileall / --help / bash -n |
| Linux实际tmux派发、服务器2.1.2+cu121、B16/640 | SKIPPED | 当前仅可访问Windows本地；已提供服务器入口，不能声称容量通过 |
| 正式200epoch、正式val/test | NOT_RUN | 均需用户显式调用；没有额外实验矩阵 |

## CUDA 差异的具体界限

真实全模型CUDA零权重前向和所有原L0项逐位一致；反向有原`grid_sampler_2d_backward_cuda`的非确定性，PyTorch在原`deterministic=True, warn_only=True`模式也明确输出警告。用同一个母版、相同权重/RNG/输入再反向一次，记录了本身的波动。

最终本次测量：ROR与母版梯度最大绝对差约0.000310898；重复母版最大差约0.000307083。各参数ROR梯度差均在本次测得的 `max(2e-7,4×母版重复反向差)` 范围内。

Adam第一步在梯度接近eps=1e-8时会放大这种变化；模型参数最大更新差约0.000191543。CUDA直接更新的逐state `max(2e-7,4×母版重复更新差)` 检查**未通过**。随后对两套实际更新各自使用实际裁剪后梯度检验独立第一步AdamW解析式，atol=2e-7、rtol=2e-6，均通过。这定位了反向波动和Adam敏感性，但不把未通过的直接一致性检查改成通过，不普遍放宽所有误差。

`cuda_backward_variation.json`、`cuda_step_variation.json`保留每项原始差值。服务器preflight会再次记录该检查；若仍未通过，状态为REVIEW_REQUIRED，start拒绝启动。不得手工改JSON为PASS或降低delta绕过。CPU一致性已完整通过，不能据此声称CUDA更新逐位一致。

CPU融合最初直接按query行比较失败；复用母版`fusion_protocol`后证实只是相同候选集合的顺序置换，按同候选ID以及共同输入重放，原FP32容差atol2e-5/rtol2e-4通过。没有放宽fusion阈值。

初版smoke错误地把GradScaler初始scale下的溢出当成网络非有限失败。已修正为原生scaler的有界校准并保留其skip/backoff步骤。最终CUDA完整smoke从65536回退到1024后完成一次有效更新；单独smoke从65536回退到2048。实际scale依赖本次RNG/数值，不能硬编码成新配方。正式入口保持原GradScaler步骤，并把scale下降记录到amp_overflows.jsonl；真正非有限损失直接报错，OOM不降batch。

## 实际命令

PowerShell在上述工作树中运行；由于隔离账户与Git工作树所有者不同，仅向当前进程及子进程传入精确safe.directory，没有更改全局Git配置：

```powershell
$env:GIT_CONFIG_COUNT='1'
$env:GIT_CONFIG_KEY_0='safe.directory'
$env:GIT_CONFIG_VALUE_0='D:/MyProjects/Crack_RTDETR/outputs/worktrees/Crack_RTDETR-ror_v1'
& 'D:/miniconda3/envs/rtdetr/python.exe' tools/check_ror_v1.py --source 'D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt' --device cpu --report outputs/ror_v1/checks_cpu_final.json
& 'D:/miniconda3/envs/rtdetr/python.exe' tools/check_ror_v1.py --source 'D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt' --device cuda:0 --report outputs/ror_v1/checks_cuda_final.json
& 'D:/miniconda3/envs/rtdetr/python.exe' tools/ror_v1.py prepare --main 'D:/MyProjects/Crack_RTDETR' --data outputs/ror_v1/local_data.yaml --c2-args 'D:/rtdetr跑结果/C19＋LIF：原 CBR＋LIF-Down v1/metadata/launch/authoritative_c2_args.yaml'
& 'D:/miniconda3/envs/rtdetr/python.exe' tools/ror_v1.py diagnose --main 'D:/MyProjects/Crack_RTDETR' --data outputs/ror_v1/local_data.yaml --weights 'D:/rtdetr跑结果/C19＋LIF：原 CBR＋LIF-Down v1/training/weights/best.pt' --batch 2 --output outputs/ror_v1/diagnose_final
& 'D:/miniconda3/envs/rtdetr/python.exe' -m compileall -q tools/ror_v1.py tools/ror_v1_training.py tools/ror_v1_diagnose.py tools/check_ror_v1.py ultralytics-main/ultralytics/models/utils/ror.py
& 'D:/miniconda3/envs/rtdetr/python.exe' tools/ror_v1.py --help
& 'C:/Program Files/Git/bin/bash.exe' -n tools/sync_ror_v1.sh
& 'C:/Program Files/Git/bin/bash.exe' -n tools/autodl_ror_v1.sh
```

local_data.yaml仅把原data.path改为同一数据在本地的绝对路径，train/val/test/names均保留。两个最终检查报告完成后，对CUDA总状态按已测得的false标志归类为REVIEW_REQUIRED；没有重写测量数值。代码/报告身份在source_audit.json保存。
