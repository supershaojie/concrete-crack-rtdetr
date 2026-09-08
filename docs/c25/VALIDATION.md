# C25 本轮验证

本轮运行环境：Windows / Python3.9.25 / PyTorch2.7.1+cu118 / RTX2060；结构、初始化和4图评估回归在CPU执行，SCCA原模块测试包含本机CUDA AMP/half检查。未在AutoDL Python3.10 / PyTorch2.1.2+cu121 / RTX4090执行；没有启动正式训练、完整val/test或额外服务器完整预检，`full_server_preflight=NOT_RUN`。

## 本轮实测

| 项目 | 结果 |
|---|---|
| C17 cscef_v51.py、cscef_v5.py与指定原提交 | LF规范化后逐字节一致 |
| C24 scca_aifi.py与f6e9dfd完整来源提交 | LF规范化后逐字节一致 |
| C25 YAML与C17原YAML | 仅第9层 AIFI→SCCAAIFI，参数[1024,8]不变 |
| nc1未融合 C2/C17/C24/C25 | 20,082,772 / 20,109,684 / 20,148,312 / 20,175,224 |
| 统一初始化源 | SHA256与指定fe8501…完全一致，无EMA/optimizer等训练状态 |
| 公共映射 | 533源状态逐键形状/值通过，保留CSCEF7状态及SCCA5状态 |
| 初始化保存与RTDETR重载 | 全部状态逐键精确相等 |
| 原生RTDETRTrainer.get_model重建nc80→nc1 | 536/545精确加载，9项分类重初始化逐项列出及核对 |
| 双模块原生AdamW | SCCA5与CSCEF5个可训练参数张量全部各出现一次；遗漏和重复会失败 |
| C2配方 | 109字段及类型核对，只改变model/name/save_dir；服务器真实args仍由start-direct校验 |
| 独立评估小流程 | 原有隔离副本每split4张图，CPU640/batch1/FP32；val/test均完成 |
| 同次逐图导出 | 每split4条记录、1200条完整query，val8个GT/test12个GT；split路径多重集一致 |
| AP、日志与产物 | 每split [1,10] AP数组、实际参数、日志、退出码、PR/P/R/F1与混淆矩阵、样图生成 |
| 完整包 | 22,029,095字节测试包（超过20 MiB），含weights与train图片；逐文件读回校验、SHA、缺失声明、防覆盖通过 |
| CLI/Python语法 | Bash -n、Python3.10 AST解析通过 |

29项单元/回归通过：SCCA原模块5项、既有工具5项、完整导出新增6项、原C17测试13项。测试覆盖非方形坐标、浮点无舍入、空GT/预测、重复/缺失split、保持排序阈值指标、双模块参数遗漏/重复、test设置/源码漂移在推理前拒绝、包超限/损坏清单/防覆盖、直接派发及OOM策略。tmux派发使用mock，不是真实训练。

## 证据

- [结构、参数量、533项映射、545项重建记录、双模块优化器及配方报告](evidence/local_delivery.json)
- [109字段差异](evidence/parameter_diff.json)
- [本轮小流程完整日志](evidence/local_delivery.log)
- [16项SCCA/工具测试](evidence/scca_tests.log)、[13项C17回归](evidence/c17_regression.log)
- [四图val指标/实际参数](evidence/tiny_val_metrics.json)、[val覆盖与导出哈希](evidence/tiny_val_export_manifest.json)
- [四图test指标/实际参数](evidence/tiny_test_metrics.json)、[test覆盖与导出哈希](evidence/tiny_test_export_manifest.json)
- [本轮源码LF规范化SHA256](source_sha256.json)
- [完整包实际CLI读回校验](evidence/complete_pack_cli_verification.json)：本地没有正式产物，明确列出54项缺失；仅打包现有源码与证据，没有推理。

报告生成在commit之前，runtime.commit显示来源父提交f6e9dfd；源文件哈希对应交付时实际内容，最终完整commit由交付回复给出并核对远端。临时初始化和评估checkpoint已经删除，输出路径仅用于追溯，不能用于正式训练。逐图原始文件和绘图仅在本地忽略的outputs保留，没有把图像、权重或压缩包提交到Git。

小流程使用原有 `outputs/worktrees/acr/outputs/local_data/data.yaml` 的每split4图隔离副本，未改划分、图像或标签。使用未训练的临时初始化checkpoint，不运行optimizer step，AP为零没有研究意义；它只验证评估/导出流程。训练器原生API完整 `_setup_train` 的既有两步真实数据smoke、组合模型FP32对照和AMP梯度记录直接复用下述历史证据，不重复跑完整预检。

## 复用的历史验证

[SCCA原交付验证](../scca/VALIDATION.md)、[C25历史smoke](../scca/evidence/c25_smoke.json)、[历史整网对照和小步梯度报告](../scca/local_validation.json) 均保留。它们已覆盖原生train API/_setup_train、真实小样本两步loss/DN/AMP/EMA、SCCA梯度开启顺序、保存重载、整网FP32与C17初始输出最大误差0，以及原模块half行为。原模型计算文件本轮无diff，所以复用这些结论；不把历史本机测试说成AutoDL正式batch16预检。

## 复现命令

```bash
export PYTHONPATH="$PWD/ultralytics-main"
export YOLO_AUTOINSTALL=false
python -m unittest discover -s ultralytics-main/tests -p 'test_scca*.py' -v
python -m unittest discover -s ultralytics-main/tests -p test_cscef_v51.py -v
python tools/check_c25_delivery.py \
  --source /实际主仓库/weights/rtdetr_r18_lite_imagenet_backbone_init.pt \
  --output outputs/c25_delivery_new
```

可选 `--tiny-data /已有每split最多4张图的隔离副本/data.yaml` 仅用于本地工具回归；不要传正式完整数据集，该脚本会拒绝。正式启动不调用此脚本，直接使用AUTODL文档的start-direct。
