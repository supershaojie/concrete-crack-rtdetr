# 本地验证记录

环境：Windows、Python 3.9.25、PyTorch 2.7.1+cu118、RTX 2060。
报告生成于 commit 前，runtime 中的 SHA 是父提交；最终源码逐文件 LF 规范化 SHA256 见 [source_sha256.json](source_sha256.json)，交付回复给出最终完整 commit 与远端核对结果。

## 已完成

| 验证 | C24 | C25 |
|---|---|---|
| 相对同初始化 C2 / C17，整网 FP32，batch1 640×640 | 最大误差 0 | 最大误差 0 |
| 整网 FP32，batch1 160×192 | 最大误差 0 | 最大误差 0 |
| 统一初始化533状态完整映射、无意外缺项/形状差异 | 通过 | 通过 |
| nc80→1 原生 train API 重建，精确值加载 | 529状态，9个分类 shape 跳过 | 536状态，9个分类 shape 跳过 |
| CPU 原生 loss/DN/backward/AdamW，160×160，batch2后batch1 | 两次更新通过 | 两次更新通过 |
| CUDA AMP 原生 loss/DN/backward/AdamW，同规模 | 两次更新通过 | 两次更新通过 |
| 实际整网 model.half()，batch1 160×192 | 有限输出 | 有限输出 |
| 真实数据 RTDETR.train API/_setup_train，320×320，batch2后batch1 | 原生 AMP、两次更新通过 | 原生 AMP、两次更新通过 |
| 更新后模型和 optimizer 保存重载 | 最大误差 0 / 0 | 最大误差 0 / 0 |
| 更新后整网真实 half 推理，batch1 320 | 通过 | 通过 |
| 原生optimizer包括全部5个SCCA参数张量且各仅一次 | 通过 | 通过 |
| 独立 val/test 工具链，每split四张真实图，640、batch1 | 完成 | 完成 |
| 按需四图诊断 | 完成，正常forward无持久激活缓存 | 同模块，未额外重复 |

FP32 预设容差 `atol=1e-6, rtol=1e-5`，上述最大误差均为实测。模块级也检验 pre/post-norm、dropout随机流一致、动态/非方形和 N=1，单次 MHA。

真实数据来自已有本地隔离副本 `outputs/worktrees/acr/outputs/local_data/data.yaml`：32张训练图和各4张val/test；与原 crack_det 的图片/标签保持一致。训练器构建该小样本 loader，本次仅获取两批（第二批截为末批1张），实际进入 loss 的图数为3，在线增强可能读取额外图。没有改变划分或原标签，没有完整 epoch loop。
真实两步 loss：C24 `51.2544097900 → 42.9126586914`；C25 `51.2544097900 → 42.8857307434`。DN、每参数梯度/更新及重载证据见 [C24](evidence/c24_smoke.json)、[C25](evidence/c25_smoke.json)。第一步 O 梯度非零，上游Q/K/V/温度梯度零；第二步全部非零，EMA和AdamW均实际更新2次。

本地梯度诊断为保证有限步内观察到更新，将测试 GradScaler 初始 scale 设为128；这只存在 `check_scca.py` / `smoke_scca.py`，正式 `train_scca.py` 使用原生默认动态 GradScaler。真实 `_setup_train` 的原生 YOLO AMP 辅助检查通过（日志保留），未用额外整网 FP32 大batch检查替换。原 PyTorch grid_sampler 的 deterministic warn-only 提示仍在日志中，没有修改 C2 确定性设置。

## 单元和流程测试

- SCCAAIFI 5项：字面公式对照、矩阵 `[B,4,16,16]`/行和、N=1/常量均匀有限、固定X改变S关系发生变化、独立LN、参数计数、CPU随机流、单次MHA、pre/post及dropout对齐、可微X/S、分步梯度、CUDA AMP FP32核心及真实half。
- SCCA工具5项：109字段/配方漂移拒绝、原训练器OOM分支适配、mock tmux直接派发/无prepare/防重复、排序mask修正、≤20MiB包/排除权重图片/拒绝覆盖。
- 原 C17 `test_cscef_v51.py` 13项通过，含原baseline/parser和CPU旧autocast兼容回归。
- 原 ACR `test_acr.py` 6项通过。
- Bash `-n` 语法、Python3.10 AST语法、两份YAML逐字符仅替换AIFI、C17受保护文件相对来源提交无diff：通过。
- status及CLI帮助可执行；服务器 tmux/conda 启动用隔离 mock 验证派发和退出脚本，**未在真实 AutoDL 执行**。

共29项单元/回归测试；测试命令：

```bash
export PYTHONPATH="$PWD/ultralytics-main"
python -m unittest discover -s ultralytics-main/tests -p 'test_scca*.py' -v
python -m unittest discover -s ultralytics-main/tests -p test_cscef_v51.py -v
python -m unittest discover -s ultralytics-main/tests -p test_acr.py -v
python tools/check_scca.py --source /实际/统一初始化.pt --output outputs/scca_check_new
```

四图 val/test 的指标仅用于工具流程验证，低步数 smoke 权重没有研究意义，不能参与模型选择或替代完整指标；报告标明 pipeline。已验证 PR/混淆矩阵生成和输出路径。独立评估沿用已明确记录的排序mask修正，原公共训练验证不改。
诊断只用四张val图，不涉及完整注意力导出；比例不能据此推断背景消除或单模块收益。

## 实测开销与边界

同本地GPU，160×160、batch1、AMP eval，3次warmup+10次计时，含同步墙钟平均；不是4090或640正式训练指标：

| 对照 | 平均 ms/forward | 峰值已分配 bytes |
|---|---:|---:|
| C2 | 24.20684 | 156,069,376 |
| C24 | 25.66124 | 155,774,976 |
| C17 | 25.46934 | 156,642,304 |
| C25 | 26.59189 | 156,347,904 |

微型输入、短测量受算子临时张量和运行波动影响，不由此推断训练显存下降或稳定速度差。
未验证：AutoDL Python3.10/PyTorch2.1.2+cu121/RTX4090、正式640 batch16容量、完整200轮训练、完整val/test和真实收益。也未执行额外服务器完整预检。正常 start-direct 无需本地报告或额外smoke通过。

## 全字段配方差异

| 字段 | C2 | C24 / C25 |
|---|---|---|
| model | 原统一初始化.pt | 对应新SCCA干净初始化.pt |
| name | c2_rtdetr_r18_lite_e200_b16_onlineaug | c24_rtdetr_r18_lite_scca_e200_b16_onlineaug / c25_rtdetr_r18_lite_cscef_v51_scca_e200_b16_onlineaug |
| save_dir | 原C2目录 | 原project下各自新run目录 |
| 其余106字段 | 实际C2 args | 类型和值全部相同 |

完整逐字段表：[C24](c24_parameter_diff.json)、[C25](c25_parameter_diff.json)。正式运行还会生成实际 args、运行时差异、nc1加载与optimizer记录。
