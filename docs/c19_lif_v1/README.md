# 原 C19/CBR + 原 LIF-Down v1

2026-09-13 FP16 截止位门禁追加修复见 [gate_fix/README.md](gate_fix/README.md)，当前单次预检后派发命令见 [gate_fix/AUTODL.md](gate_fix/AUTODL.md)。旧锁已归档，不重复 apply；历史文档保留。

2026-09-13 预检融合工程修复与安全重试见 [fusion_fix/README.md](fusion_fix/README.md)；新 worktree 操作见 [fusion_fix/AUTODL.md](fusion_fix/AUTODL.md)。本目录原检查报告是历史开发机证据，不能当作旧 AutoDL 失败已通过。

本实验只组合两个成功原版，工程准备完成后等待用户启动固定配方训练。没有正式训练、完整 val/test 或涨点结论。独立分支为 `codex/rtdetr-c19-lif-v1`，直接基于 LIF 成功提交 `0e95bbade3558b0d2b77c5531483c60810391d88`。

源码依据：C2 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`、C19 `025997e3c51eaf6933534308a95da6ebf97bff53`。两个成功包及18个归档源码文件与相应 Git 历史比对一致；原始包哈希、实际 Windows 参考路径和 LF/字节哈希见 [PROVENANCE.json](PROVENANCE.json)。

新 YAML 为 `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml`。相对 C2 只改变原两个节点，所有工具从原拓扑定位：

| 层 | 结构 |
|---|---|
| 9 | 原 AIFI |
| 18 → 19 | 原 Concat → RepC3，最终 Neck P3 |
| 20 → 21 → 22 | LIFDown(P3) → Concat([20,15]) → RepC3/P4 |
| 23 → 24 → 25 | 原 Conv k3/s2 → Concat([23,10]) → RepC3/P5 |
| 26 | 原 RTDETRDecoderCBR([19,22,25]) |

CBR 的 `p3=x[0]` 仍指最终 Neck P3；rho/normal_fraction 均为0.10，36点、signed evidence、FP32采样、border、align_corners=False、原几何detach、box宽高梯度、14个参数及原bias均保留。LIF 的 r16、Haar/Align、5个无bias权重、O零初始化及 `act(BN(conv(X)+R(X)))` 均保留。LIF 会影响后续 P4/P5、query/box，并通过联合训练改变共享表征，不保证收益叠加。

核心实现差异仅为原 `cbr.py`、原 C19 YAML/单元测试、新组合 YAML、注册和 parser 各一处，以及原 C19 的可选 final_query 接口。`head.py` 与所有原 LIF 文件、工具保持 Git 无差异。`transformer.py` 与成功 C19 完全一致，只比 LIF 多可选参数和返回分支。没有引入 SCI、CSCEF、SCCA、triad/mincompat 或其他创新模块。

LIF 融合仍沿用父版：BaseModel.fuse 排除 LIFDown，保留自己的 BN 和 forward；其余普通 Conv、RepConv 正常融合。没有全局禁用融合。

| nc=1、未融合 | 参数量 | 相对 C2 |
|---|---:|---:|
| C2 | 20,082,772 | 0 |
| 原 C19 | 20,128,661 | 45,889 |
| 原 LIF | 20,103,876 | 21,104 |
| 本组合 | 20,149,765 | 66,993 |

统一初始化仅使用 SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e` 的公共 nc80 初始化，epoch=-1，无 optimizer/EMA 等训练状态。两个父模型 best/last 未参与初始化或回归。533 COMMON + 19 NEW_TRAINABLE + 0 NEW_BUFFER，异常 MISSING/UNEXPECTED/SHAPE_MISMATCH 均为0。两模块新初值与各自 seed42 独立父构造逐项一致。

保存/重载后，原生 RTDETRTrainer.get_model 按nc1重建：543/552个同形状态精确加载，9项原分类适配；全部533个公共nc1状态与相同RNG的C2重建一致。真实 Model.train → 原生get_model链路在进入训练前中止审计，通过552项精确值核对。正式 RecordingTrainer 也调用同一个原生get_model，仅记录审计，不改变构造和加载逻辑。

9项类别适配包括 denoising_class_embed.weight（Embedding原正态初始化）、enc_score_head.weight及3项dec_score_head权重（nn.Linear原构造器初始化）、对应4项score bias（原 `bias_init_with_prob(0.01)/80*nc`）。名称、形状和C2逐项等值证据在 checks.json 的 trainer_rebuild 中。

从真实父包重新读取的6份配置均为109字段，含类型核对；差异只有model/name/save_dir。完整正式配置见 resolved_formal_config.yaml，逐字段记录见 recipe_diff.json。全部应训练参数保持requires_grad=True。原优化器、warmup、动态accumulate及增强不改。

审计入口：init_c19_lif_v1.py、check_c19_lif_v1.py；生命周期入口：train_c19_lif_v1.py、autodl_c19_lif_v1.sh、sync_c19_lif_v1.sh；评估打包：c19_lif_v1_results.py。辅助probe/data文件只服务可复现验证和清单读取。

数值与未验证范围见 [VALIDATION.md](VALIDATION.md)、[summary.json](summary.json)，完整张量误差见 checks.json。服务器操作见 [AUTODL.md](AUTODL.md)。正式训练SUCCESS后，对同一best独立val、锁定SHA、独立test，再pack-complete。

历史test参考线：C2 0.46963623190802783、C19 0.5038924194554971、LIF 0.5081882458799715。用于固定规则下的最终比较，不能用于选epoch或扫描阈值。P/R按各模型自身最大F1工作点记录。原C19归档validator存在排序前mask问题，但该次864图均导出300预测，最低保存score=0.00295>0.001，因此该次mask全通过；证据见 parent_eval_policy.json。新入口复用LIF的corrected_sorted_conf_mask_v1，不改训练val规则。
