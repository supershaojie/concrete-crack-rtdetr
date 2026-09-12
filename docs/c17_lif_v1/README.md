# C17 + LIF-Down v1

本分支从成功 LIF-Down v1 `0e95bbade3558b0d2b77c5531483c60810391d88` 创建，仅迁入成功 C17 `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139` 的 CSCEFv5、CSCEFv51、原 YAML 和最小注册。C2 对照为 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。

三个模块文件与成功 Git 对象及原结果包逐字节相同，AST 也相同。LIF 普通融合排除条件沿用父版本；两个父工具及 YAML 均保留。没有迁入失败兼容分支、梯度缩放、额外 detach 或新创新模块。原 CSCEF 结构置信度的 no_grad/detach 保留，content 的 lateral/semantic 梯度正常。版本封存不等于冻结训练参数。

拓扑：9 AIFI → 15 Y4 → 16 Upsample；17 backbone P3 projection；18 CSCEFv51([17,16])；19 Concat([16,18])；20 RepC3=P3；21 LIFDown；22 Concat([21,15])；23 RepC3=P4；24 原 Conv(k3/s2)；25 Concat([24,10])；26 RepC3=P5；27 Decoder([20,23,26])。工具从 Decoder/source 边发现位置，再逐层核对后生成映射，未对全部 key 无条件 +1。

nc1 未融合参数：C2 20,082,772；C17 20,109,684；LIF 20,103,876；组合 20,130,788。新增 26,912 + 21,104 = 48,016。LIF 不会在同次前向倒改 CSCEF 的 Y4，但两者共享上游梯度；组合增益仍需固定配方实验验证。

受控公共源 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。533 COMMON、10 NEW_TRAINABLE、2 NEW_BUFFER；nc80→nc1 原生重建有9项分类适配。536/545同形状态精确载入，适配后的全部533项公共状态与独立C2相同，两个新分支的12项状态也精确保留。没有 best/last 热启动。

审计：`PROVENANCE.json`、`repository_audit.json`、`recipe_diff.json`、`dataset_manifest.json`、`common_mapping.tsv`、`initialization_mapping.json`、`checks.json`、`validation_summary.json`、`ops_checks.json`、`evaluation_checks.json`、`sync_checks.json`。工程验证详见 [VALIDATION.md](VALIDATION.md)，服务器命令详见 [AUTODL.md](AUTODL.md)。

历史 test 主比较线为原 C17 mAP50–95=0.5120450850441516；原 LIF=0.5081882458799715，C2=0.46963623190802783。此标准不用于挑 epoch/阈值/结构。正式训练仍按原 val 选择 best，然后独立 val→同 checkpoint test→完整打包。统一评估沿用成功 LIF 的 corrected_sorted_conf_mask_v1，seed42显式记录，P/R来自各模型自身F1工作点。C17历史包使用旧原生评估代码；没有重新运行父模型，比较结果前仍须确认筛选口径，不能把历史验证器seed0当成训练seed。

本次仅工程准备和本地有限验证；没有200e训练、完整val/test、正式性能结果或涨点结论。第三项轻量化不在本任务范围。
