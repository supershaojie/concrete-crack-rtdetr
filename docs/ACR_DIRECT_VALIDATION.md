# C22 start-direct 修改与本地验证

本次在 `codex/acr-query-attention` 的 `ed366ebdb514a83a5504243213a1580bf74a1a48` 上修改现有启动流程，响应用户取消完整服务器预检的明确选择。没有启动正式训练，没有重新给直接入口附加完整 prepare。

## 修改范围

- `autodl_acr.sh start-direct c22`：独立目录 `outputs/c22/launch_direct/`；只生成锁定配方、统一源初始化和权重映射，然后沿用 tmux/supervisor 启动。旧 `launch/` 失败标记、audit 和日志不读取、不修改。
- `train_acr.py`：直接计划无 audit 依赖，记录 `full_preflight=not_run`。通过原 Trainer 的构造器、同一 RNG 位置和原 `model.load` 加载，逐键验证实际 nc=1 模型。启动仅检查原配方、实际映射、空 optimizer 和初始 EMA。
- 正常训练默认 `acr_stats_interval=0`，取消回调自动设置 200 的行为；模型和 EMA 都重置，关闭时不注册逐批统计日志回调。完整 prepare 后的普通 start 仍可用 `ACR_STATS_INTERVAL` 显式开启；直接入口固定关闭。
- status/val/test/pack 选择直接启动记录；pack 明示完整预检未执行。保留原输出、tmux/process 检查、原子锁、退出码、禁止覆盖规则。

`acr.py`、十维公式、0.5 上限、原生 attention、梯度路径、DN mask、模型 YAML 以及公共训练/loss 源码均未修改。本次参数增量为 **0**；ACR 相对 C2 的增量仍是 **1848**，C22 nc=1 总参数仍为 20,084,620。

正式训练读取服务器 C2 原始 109 字段 args，仅允许 model/name/save_dir 三字段变化。200 epochs、640、batch16、workers8、seed42、原 AMP、AdamW、学习率、weight decay 和全部在线增强不变。统一源 SHA256 仍为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。直接入口从该源生成新的 FP32 文件，不能传入已有 smoke 更新模型代替。

## 本地证据

环境：Windows，Python 3.9.25，PyTorch 2.7.1+cu118，RTX2060 6GiB。机器可读结果见 [acr_direct_validation.json](acr_direct_validation.json)。

- 16 项回归通过：原有几何/掩码/梯度测试，加上直接路由、旧失败记录保留、独立证据核对、重复启动锁、统计默认关闭、原生构造对齐、结果包及 supervisor 检查。Bash 语法通过。
- ACR 非零关系头、DN、dropout 下，统计开/关的输出、loss、梯度、一次 AdamW 更新和 RNG 逐位一致；覆盖 CPU、CUDA FP32 和 CUDA AMP。关闭时 `_record_stats` 调用次数为 0。
- 从规定源文件重新生成独立初始化：533 个公共状态精确复制；545 个状态经过 raw、RTDETR(checkpoint)、RTDETR(YAML).load 三种重载核对。
- 实际 `Model.train()` API 在进入 epoch 前停止的夹具核对：109 字段相同，545 个 nc=1 状态及 RNG 与原生 Trainer 相同；536 个状态来自干净初始化，9 个分类状态来自原生 nc=1 构造；338 个 optimizer 参数张量完整覆盖，optimizer 空、EMA updates=0 且状态相同。这部分没有运行原生服务器数据/AMP 自检，报告中已明确标注夹具范围。
- 另取本地真实裂缝数据的一个 batch2，经原生在线增强后在 320 输入上做完整模型 native predict/loss、DN、反向和独立 AdamW 更新。这个本地检查尺寸和 batch 不写回正式配方；不是 batch16/640 服务器 smoke。

| 同一设备/精度下的原生与直接入口对照 | DN split | 输出和 loss 最大差 | 输出和 loss 逐位相等 | 梯度 |
|---|---|---:|---|---|
| CPU FP32 | [200,300] | 0 | 是 | 有限、逐位相等 |
| CUDA FP32 | [200,300] | 0 | 是 | 有限；整网反向不逐位相等 |
| CUDA AMP | [200,300] | 0 | 是 | 有限；整网反向不逐位相等 |

前向/loss 一直采用 `atol=1e-6, rtol=1e-5`，没有放宽。GPU 运行时明确警告 `grid_sampler_2d_backward_cuda` 没有确定性实现。原生→直接梯度最大绝对差为 FP32 `1.5735626e-4`、AMP `0.02734375`；同一原生模型→原生重复控制也分别出现 `1.1444092e-4`、`0.0390625` 差异。报告保留超出上述容差的张量数量与最大差项，**不声称 CUDA 整网梯度精确等价**。受本次开关影响的 ACR 单模块梯度比较则全部逐位一致。

测试开发时先补齐夹具中原生 Trainer 通常附加的 `model.nc`；随后发现 GPU 整网梯度严格相等断言不适用于该原生非确定性算子，因此补做原生重复控制、保留实际差异并明确缩小该项结论，没有修改模型或关闭正式 AMP。最终源文件和新生成的干净初始化 SHA256 均未受测试更新影响。

## 复现范围

这些检查仅供开发者手动运行，`start-direct` 不调用它们：

```bash
export PYTHONPATH="$PWD/ultralytics-main"
export YOLO_AUTOINSTALL=false
python -m unittest discover -s ultralytics-main/tests -p 'test_acr*.py' -v
bash -n tools/autodl_acr.sh
```

默认跳过可选的真实数据集成测试；单独提供本地 `ACR_DIRECT_TEST_SOURCE`、`ACR_DIRECT_TEST_DATA` 后才执行 `test_acr_direct_integration.py`。可用 `ACR_DIRECT_TEST_REPORT` 指定不存在的 JSON 路径保留结果；测试只在临时目录创建干净初始化和隔离模型，退出后删除临时测试文件。

未在 AutoDL 上执行本次同步、tmux 调度、原生 AMP 自检或 batch16/640 正式训练。用户已取消完整服务器预检；服务器可直接按 [ACR_AUTODL.md](ACR_AUTODL.md) 的 `start-direct c22` 启动。正式训练仍执行原生数据加载与 AMP 逻辑，不跳过它们，也不自动调小 batch。
