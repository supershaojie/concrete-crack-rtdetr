# BLC v1 实现交付

母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。实验分支：`exp-rtdetr-r18-lite-blc-v1`。
仓库：`supershaojie/concrete-crack-rtdetr`。只在本实验独立 worktree 实施。

`model.5` 改为 `BLCBlocks`，先执行原 `.blocks.0`、`.blocks.1`（包含残差与末端 ReLU），再执行一次 `.blc`。
增强后的 128×80×80 P3 同时送入 `model.6` 和 `model.17`。层号及其他 from 连接不变。

| 变体 | YAML | 未融合参数 | 原生融合参数 | 新增 |
|---|---|---:|---:|---:|
| CBR + LIF + BLC | rtdetr-resnet18-lite-cbr-lif-blc-v1.yaml | 20,158,326 | 19,953,526 | 8,561 |
| C2 + BLC | rtdetr-resnet18-lite-blc-v1.yaml | 20,091,333 | 19,886,277 | 8,561 |

设计值与实际 nc=1 构造计数一致。融合只处理原网络可融合层，BLC 保留完整前向和全部参数。
本机 THOP 2.0.18 的普通子 Conv2d hook 对 functional conv2d 计数为 0；显式 BLC 投影 hook 得到
54,732,800 MACs，即 0.1094656 GFLOPs（B1、P3=128×80×80）。这仅包含三个投影，不含固定算子，
不是 BLC 完整 FLOPs，也不是整网增量或速度。原生日志的模型 GFLOPs 会漏计这些投影，不能直接用于论文表格。

核心源码严格按 `CONTRACT.md`：方向/间距固定，最终合成坐标 replicate clamp，带符号双侧 minimum，
八个响应加一个零响应选项，9 路空间 softmax，输出不加 ReLU。FP16/BF16 输入时分支工作精度为 FP32，
FP32/FP64 保持；functional 参数转换可微，不替换 Parameter。Wd Xavier gain=1，Wg 与 Wo 为零。
仅新增层构造隔离 CPU RNG；两个变体新值相同且 storage 独立。

公共修改只有 `nn/modules/__init__.py` 注册与 `nn/tasks.py` 的 Blocks 同类解析。
原 `cbr.py`、`lif_down.py` 的 LF 归一 SHA256 完全保持。未改 AIFI、decoder、DN、matcher、loss、查询选择或训练循环。
独立评估复用母版 `tools/c19_lif_v1_results.py` 已有 `corrected_sorted_conf_mask_v1` 算法；母版原生 epoch 验证和 best 选择不变。

## 初始化与配方

`tools/blc_common.py` 复用母版 `init_c19_lif_v1.py` 的公共源审计，构造各自父模型后严格迁移全部公共参数和 buffers。
指定公共源 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，本机已核对。
保存的是受控 nc=1 未训练初值；nc80→nc1 的九项分类适配通过原生 Trainer 构造并逐项核对父/候选一致。
实际 `RTDETR.train` 重建探针执行到 Trainer.train 边界即退出，零训练 batch；不是只验证推理加载。

母版原始 `training/args.yaml` 与附件 109 字段完全一致，来源和哈希在 `parent_provenance.json`。
正式 Linux 配方在两个 `*_args.yaml` 中；`recipe_diff.json` 仅有 model、name、save_dir 三个身份变化。
保持 200e、B16/640、AdamW、lr0=.0005、seed42、原在线增强、不冻结公共参数。
本机数据 train/val/test=6048/1728/864，图像清单及标签哈希逐项等于母版 `parent_dataset_inventory.json`。
本次仅对 test 清单和标签做身份核查，未执行 test 推理。

## 验证边界

详见 `VALIDATION.md` 和 `local_reports/` 原始记录。服务器 Python3.10 / PyTorch2.1.2+cu121 / RTX4090 的
B16/640 native AMP 容量与真实恢复续更仍为 PENDING。本机是 Python3.9.25 / torch2.7.1+cu118 / RTX2060 6GiB。
CUDA AMP/half 的融合自然输出存在未放行的数值差异，保持 PENDING，不以有限性或固定候选重放替代自然输出验收。
`start` 会检查代码/配置、数据清单与标签、初值、变体、环境、配方及最新完整证据；PENDING/FAILED 均不放行。

有限服务器预检使用实际 Trainer 原生循环，最多总计 16 个训练 batch（包括真实 resume 诊断），
至少两次 Wo 有实际变化的有效更新，并检查后续 Wd/Wg 非零梯度。记录 loss、scaler 跳步/回退、
真实 optimizer step 状态、显存与耗时。保持原 AMP、累计和优化器分组；OOM 不减 batch，不终止其他进程。
预检独立输出，不覆盖 controlled_init。原生 half EMA checkpoint 与 optimizer moment 量化策略不变。
预算内没有剩余 batch 做 resume 时如实 PENDING，不增加重试预算。

`tools/blc_server.sh` 提供 environment/init/init-preflight/preflight/plan/start/resume/val/test/pack。
单模块消融只准备，不自动训练；后续需先审阅主组合 val 收益并显式使用 `--ablation-after-review`，且完成该变体自身预检。
完整操作在 `AUTODL.md`；提交后生成的固定 SHA 交接在交付文件中，不通过修改 HEAD 反复追逐文档 SHA。

## 来源与限制

参考 ZIP 在已搜索的本地范围没有找到：**本机参考包未核对**，见 `reference_package.json`；没有执行包内脚本或安装依赖。
本实现来自用户完整数学合同，没有拷贝 LSKBlock/CAA/DySnakeConv/DSConv。
线/脊检测、方向处理和自适应上下文已有研究：[Steger](https://mv.in.tum.de/_media/members/steger/publications/1996/fgbv-96-03-steger.pdf)、
[Dynamic Snake Convolution](https://arxiv.org/abs/2307.08388)、[LSKNet](https://arxiv.org/abs/2303.09030)。
这些链接是思想背景，不是本实现已经证明的论文新颖性。BLC 是待验证组合，不承诺涨点，不宣称达到旧的骨干减参 10% 目标。

**正式训练 NOT_STARTED；最终 test NOT_RUN；本次未登录服务器。**
