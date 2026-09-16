# 本地检查记录

2026-09-16 本地最终 CPU/CUDA 两份报告均为 **LOCAL_PASSED_SERVER_PENDING**，两个 variant 的本地检查均 PASSED。环境为 Windows、Python 3.9.25、torch 2.7.1+cu118、RTX 2060；不代替服务器 Python 3.10.13 / torch 2.1.2+cu121 / RTX 4090 的实机补检。

完整 JSON 为 `outputs/rcsq_v1/initialized_final/initialization.json`、`cpu_final/checks.json`、`cuda_final/checks.json`。提交内 `validation_summary.json` 保存必要结果、完整报告 SHA、初始化 SHA 与类别适配白名单。三份报告共用内容指纹 `db2d74f6e28be2c66bc5726aec09a7a202674272e00fa9d3d1b600f88fad216d`。测试时 HEAD 为基点、工作树含待提交实现；最终完整交付 SHA 在提交后生成，内容指纹绑定受检源码。

| 项目 | 状态 | 证据/限制 |
| --- | --- | --- |
| 正确 CBR+LIF 基点 | PASSED | `base_audit.json`：13/13 数值源码和原入口内容与历史完整结果归档一致；历史 CSV 200 epochs、退出码与最终 package/test 记录交叉核对 |
| 完整训练配方继承来源 | PASSED | 成功组合的 109 字段真实 args；与 C2 仅 model/name/save_dir 不同，详见 `base_audit.json` |
| 参考模块包 | PASSED | SHA256 与指定上传包一致；实际读取 Decoder/MSDeformAttn/DySample，见 `reference_audit.json` |
| SACQ 归属核对 | PASSED | 实际阅读官方 arXiv v1 Abstract、3.2、3.3，记录差异，不声称首创 |
| 同步脚本语法、帮助 | PASSED | Git Bash `bash -n tools/sync_rcsq_v1.sh`、`--help` |
| 交付生成器帮助 | PASSED | Python 3.9 `tools/make_rcsq_v1_delivery.py --help`；生成需已提交、干净的独立分支 |
| 后续 eval/pack CLI 帮助 | PASSED | 两个实际入口 `--help` 均成功；未执行评估 |
| 四模型受控初始化 | PASSED | 固定源 SHA 核验；公共参数/BN buffers 逐值相同；两个 variant 的 RCS-Q 参数相同；无未处理 missing/unexpected/shape mismatch |
| 真实 Trainer nc=1 与优化器 | PASSED | native constructor/setup_model；精确 9 个类别变化键白名单；所有注册可训练参数恰入 AdamW 一次；fixture setup 不启动 epoch/optimizer step |
| 两配置结构/初始等价 | PASSED | Decoder 26，输入 `[19,22,25]`；300 normal queries；eval/export/train+DN 初始最大绝对误差均 0；重置 RNG、独立模型副本 |
| 几何/中心化/null/梯度 | PASSED | 非方形、部分越界、全无效、常量与非均匀区域；独立像素中心坐标 oracle；首步 out 梯度非零，隔离副本一步后内部梯度非零；正式更新 0 |
| 真实 DN scatter 与 padding | PASSED | gt_groups `[0,0]`、`[0,3]`、`[1,3]`、`[2,7]`；真实 scatter 索引独立核对；DN 长度 0/198/198/196，padding 保留原 query |
| CPU 与 CUDA 原损失反传 | PASSED | 两 variant FP32 和 CUDA native AMP 小图 B2 功能检查，合成图 160×192；不能当作真实 B16/640 容量 |
| 保存/重载/EMA/eval | PASSED | 零和非零 out 均验证；非零已学参数不被重新清零；EMA 原生更新保留 |
| 非零分支融合 | PASSED，有条件说明 | FP32 原规则通过；AMP/true-half 为候选排列等价，通过同候选集对齐和 replay，不声称原始逐行/逐 bit 相等，见下文 |
| 本地真实数据指纹 | PASSED | train/val/test 图数 6048/1728/864，框数 45573/12840/6663；路径与标签指纹逐项匹配正式对照，无新切分 |
| 生命周期拒绝/恢复 | PASSED | 受限控制流 fixture：改 AMP、伪造旧 PASSED、实时 PENDING、活进程占用均拒绝；死进程失败门禁保留后可重试；正式训练/test 调用 0 |
| 服务器同步 | NOT_RUN | 按任务要求不登录或自动同步；交付可复制命令和 bundle 备用 |
| 服务器实际环境的原生 AMP 与 B16/640 真数据容量 | PENDING | 本地功能/小图不能替代服务器完整训练批次；用 handoff 的 `--device cuda --capacity` 补检，0 optimizer steps |
| 正式训练 | NOT_STARTED | 首轮主候选 CBR+LIF+RCS-Q；诊断副本更新不计作正式训练 |
| 最终 test | NOT_RUN | 只有入口，无新实验 test 指标 |

CPU 模块首步 out_proj 梯度范数 0.0772711，其余新增投影首步为 0 符合零输出投影的链式法则；隔离副本诊断更新 1 次后内部投影/null 均有限非零。真实 CUDA 观察到 `grid_sampler_2d_backward_cuda` 无确定性实现的 warn_only 警告，保留 deterministic=True；没有承诺 CUDA 逐 bit 确定。

融合诊断使用受控非零 RCS-Q，并在组合副本上激活原 CBR/LIF 分支；matmul/cudnn TF32 均关闭，结束恢复。两个 variant 的 CUDA FP32 模式为 PASSED，AMP 和 true-half 为 **PASS_WITH_CANDIDATE_PERMUTATION**：同候选集合发生行排列，按候选 ID 对齐并进行原有 replay 验证通过。没有静默放宽全局容差，也没有把未对齐原始行差异误记为零误差。

同设备/工具、nc=1、640、batch=1 的参数与主体算量如下。THOP 2.0.18 不统计函数式 RCS-Q 投影，故单列解析主要增量；grid_sample、中心化/归约、函数式归一化、softmax、逐元素和索引操作仍未全部计入。以下是 2×MAC 的 GFLOPs，不是 FPS。

| 模型 | 未融合参数 | 融合参数 | 未融合 THOP GFLOPs | 融合 THOP GFLOPs | RCS-Q 解析主要增量 GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原基线 | 20,082,772 | 19,877,716 | 58.276608 | 57.1657728 | 0 |
| 基线 + RCS-Q | 20,141,142 | 19,936,086 | 58.276608 | 57.1657728 | 0.3544832 |
| 原 CBR + LIF | 20,149,765 | 19,944,965 | 58.672512 | 57.5649536 | 0 |
| CBR + LIF + RCS-Q | 20,208,135 | 20,003,335 | 58.672512 | 57.5649536 | 0.3544832 |

每个新模型相对其对应原模型增加 **58,370** 参数，未融合/融合统计分别配套。报告还记录本地合成推理的 1 次 warmup、3 次计时样本；样本很少且运行环境不同，不作为服务器速度或延迟提升结论。历史训练结果只用于基点/配方核验，不当作 RCS-Q 指标。

服务器仍需：核实实际 import/环境和固定源、重新生成初始化、原生 check_amp 及其真实资源、真实训练批 B16/640/DN/AMP 前向反向和显存峰值、该环境非零融合、实际数据指纹与原配方 plan。检查入口不得执行正式 optimizer.step；所有项目通过后，由用户另外运行主组合 tmux start。
