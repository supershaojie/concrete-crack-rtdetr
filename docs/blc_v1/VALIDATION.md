# 本次 BLC 验证记录

所有本机原始报告（含失败）保存在 `local_reports/`，以报告内代码 LF 内容哈希和环境关联。
旧 DPR/PBI/C19 报告没有改写，也没有计作本次 BLC 通过。

| 检查 | 状态 | 证据/边界 |
|---|---|---|
| 原 CBR/LIF 源码哈希、两份模型拓扑及参数量 | PASSED | topology_main_01 / topology_single_01 |
| 合成数学参考、clamp 边界、带符号响应与零选项 | PASSED | math_03；1×1、2×5、5×3、7×9、11×13 |
| FP64/FP32 前向与输入/Wd/Wg/Wo 梯度 | PASSED | FP64 atol1e-8/rtol1e-6；FP32 atol2e-5/rtol2e-4；原误差入 JSON |
| FP64 固定随机点数值导数 | PASSED | gradcheck fast_mode，eps1e-6、atol1e-5、rtol1e-3；不重采样找通过 |
| Wo=0 恒等、初始 P3/P4/P5/整网输出 | PASSED | B1/640；两变体父/候选精确相同；两个 P3 消费者同一输出 |
| 公共源/全状态迁移、九项分类适配、保存重载 | PASSED | 两变体 initialization、init-preflight 报告 |
| 实际 RTDETR.train 重建 | PASSED | 两变体 actual_train_api，零训练 batch |
| 母版109字段与数据清单/标签身份 | PASSED | parent_provenance、recipe_diff、dataset_identity_local |
| 原生真实 GT loss 与三步梯度启动 | PASSED | CPU/CUDA B2/160 固定 seed42；不代表 B16/640 容量 |
| 非零 state_dict/完整模型、native half checkpoint、optimizer/EMA/scaler恢复 | PASSED | 同文件状态比较，双副本无共享 storage，相同外部梯度原生更新一致 |
| CPU/CUDA FP32 原生融合、非零分支保留 | PASSED | CUDA natural max_abs≈8.94e-8；BLC调用一次 |
| AutoBackend 相同融合/精度路径保存加载 | PASSED | CUDA FP32、AMP、half 的重载输出各自精确一致 |
| 实际原生 half EMA epoch-validator 路径 | PASSED（有限） | 2 张真实 val 图、160尺寸、非零测试副本；非完整val/最终test |
| CUDA AMP/half 融合前后自然输出 | PENDING | 详情如下；没有修改正式查询选择或阈值来放行 |
| 指定4090服务器 B16/640 native AMP容量、实际resume后有效更新 | PENDING | 本次未登录服务器；提供有界入口 |
| 参考 ZIP 中四个模块源码核对 | PENDING | 本机未找到指定包；完整数学合同已实现 |
| 正式训练 / 独立最终 val / 最终 test | NOT_STARTED / NOT_RUN / NOT_RUN | 仅实现入口 |

FP16 数学梯度报告保留 **PRECISION_NOTE**：Wd 梯度原始 max_abs=0.0009765625（1个该量级FP16间距），
relative_L2≈2.59e-5。把同一组量化后的输入/参数提升到 FP32 后，实际/参考梯度满足原 FP32 严格阈值；
各自梯度再转回叶子 dtype，与各自 half 梯度逐元素完全一致。前向一致，没有放宽 FP32/FP64阈值。

CUDA 固定 seed42 非零模型：AMP融合自然输出 max_abs≈0.804448，relative_L2≈0.505703；half 为≈0.804656、≈0.480449。
原骨干 P3/P4/P5 连续特征仍完全一致；encoder分数 max_abs分别0.001220703125和0.0009765625，
300个候选中分别195/197个位置发生变化，候选集合也有差异。
固定候选重放后 max_abs 分别0.0001220703125、0.0002445504069328308；这是定位结果，不替代自然验证。
同一融合/精度路径 AutoBackend 重载精确一致；没有丢 BLC 状态或绕过分支。
由于尚未满足当前严格自然输出诊断条件，这两项仍为 **PENDING**，不自动升级为 PRECISION_NOTE 或准入。

## 原始失败及修复/未解决事项

- 首次 Git runtime 读取遇到 Windows sandbox 不同所有者：仅本次进程的 `safe.directory` 配置，未改全局 Git；后续读取通过。
- math_01/02：把 FP32 严格梯度 allclose 直接用在 FP16 叶子梯度上失败；math_03 添加同量化值 FP32/回写的独立定位，保留原误差。
- loss_cpu_01：独立 loss 探针未设置 Trainer 原本会注入的 `model.nc`；修复探针，未改公共源码。
- loss_cpu_02/03：旧探针未固定外部随机种子，恢复自然输出有差异，原失败保留且不用于准入；修正为 seed42，记录连续特征/encoder/top-k及原误差，不挑选重采样结果。初始未定种子运行的异常不宣称已经完全复现解释。
- loss_cpu_04：定位期间的 PENDING 原样保留；固定 seed42 完整CPU报告通过。
- loss_cuda_fixed_seed_01：检查器直接比较 CPU 文件张量和 CUDA 张量报 device 错误；比较转到 CPU 后复验。
- loss_cuda_fixed_seed_02：检查器尝试 half 后再 fuse，触发母版 bias dtype 不兼容；改为实际 `load_checkpoint/AutoBackend` 的 FP32→fuse→half 顺序，未改母版融合源码。
- loss_cuda_fixed_seed_03：执行完全部 CUDA 检查，整体 PENDING 原因是上述 AMP/half 自然融合差异。

没有修改数学公式、初始化、训练数据、模型拓扑、正式精度或训练配方来消除这些记录。
最终受控初值始终保持未训练；本机小批梯度测试和非零副本不作为正式训练起点。
