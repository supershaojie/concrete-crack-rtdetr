# DRA-AIFI 单模块交付

实验别名 `dra_aifi`，分支 `codex/dra-aifi`，不分配 C 编号。新 YAML 从真实 C2 YAML 复制，仅把第 9 层 `AIFI` 改为 `DRAAIFI`。未启用 CSCEF/C17、CBR/C19、SCCA/C24、ACR、RCA、EVC，也未改变骨干卷积、Decoder、DN、Hungarian 匹配或 loss。

方法名暂定 Directional Relation-Aware AIFI（方向关系感知 AIFI）。尚无正式实验成绩，不承诺涨点、首创性或已恢复真实裂缝方向。项目第三阶段“先找到两个单独有效且组合进一步提高的精度模块，再做骨干轻量卷积与实际减参”仅记录，本次不实施。

## 仓库与参考核验

- 当前用户工作树 `D:/MyProjects/Crack_RTDETR` 保持原分支与历史未跟踪文件；独立工作树为 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/dra`。没有在其他任务目录切分支。
- 分支基点 `beedcfa307e250fb2de47587097c51f9c141123b`，选择它是为复用 C26 已有直接启动和完整打包工具；功能基线仍是 C2，未从组合 YAML 删除模块构造新模型。
- 检查父目录、仓库和基点工作树后未发现适用 `AGENTS.md`；主仓库 `.agents` 无文件。已读 README、实验记录、C24 文档、实际 C2 训练 args、初始化脚本、C24 初始化映射和 C26 生命周期/评估/打包代码。
- C2 来源提交 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。实际归档目录为 `D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053`；其 `config/rtdetr-resnet18-lite.yaml` 与当前 C2 YAML 解析内容完全相同，`train_run/args.yaml` 与已核验 109 字段记录相同。
- 实际模块包 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA256 为 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`，与用户附件标识一致。
- 已读取解压文件 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-main/ultralytics/nn/extra_modules/transformer.py` 的 `DynamicPosBias`、`DPB_Attention`、`TransformerEncoderLayer_DPB`。借鉴的是将关系偏置加在 softmax 前，以及几何索引采用非持久 buffer；没有复制其 MLP 偏置、固定 20×20 分组、替代 QKV、卷积 FFN 或归一化实现。
- 已读取本项目 `ultralytics-main/ultralytics/nn/modules/transformer.py` 的 `TransformerEncoderLayer` 与 `AIFI`。DRA 继承其公共参数和 pre/post-norm 路径，复用原生 `nn.MultiheadAttention`，不引入额外依赖或 CUDA 编译。
- 已读取 `tools/init_scca.py` 的公共权重审计方式；DRA 无层号或公共参数嵌套变化，映射是完整 identity 映射，未启用 SCCA。

具体来源散列见 [references.json](references.json)。

## 设计与公式

第 9 层处理 S5 投影后的 `F[B,256,H,W]`；640 输入通常为 20×20。保持 C2 的 8 头、FFN 1024、GELU、dropout、Q/K 的原 sin-cos PE、V、输出投影、残差与 norm。Decoder 仍取 `[19,22,25]`，3 层、300 queries，eval_idx=2；DN 参数为 100 / 0.5 / 1.0。

预测器只预测注意力参数：

`Conv1×1(256→16,bias=False) → SiLU → DWConv3×3(16,pad=1,bias=False) → SiLU → Conv1×1(16→8,bias=True)`。

最终 8 通道按 `(head, component)` 排列成前 4 头每位置二维 `z_hi`。最后卷积权重和 bias 全零，其余正常随机初始化。新增初始化隔离 CPU RNG，使之后公共层初始化不受额外抽样影响。没有第二个零门控。

`d_hi = z_hi / sqrt(1 + sum(z_hi²))`。

这是无向轴的二倍角潜在描述，范数小于 1；没有 atan2、方向标签或额外损失，范数并非经验证的真实方向置信度。

坐标严格按 `i=y*W+x` 展平，x/y 都除以 `max(H,W)`；非方形输入保留比例。原 AIFI 的 PE 生成顺序原样保留，不因新几何顺序而更改基线 PE。

对 `dx=x_j−x_i, dy=y_j−y_i, rho²=dx²+dy²`：

```
e_ij = [(dx²−dy²)/rho², 2 dx dy/rho²]  (i != j)
e_ii = 0
a_hij = d_hi · e_ij
b_hij = d_hj · e_ij
s_hij = −0.25 [logaddexp(−a_hij/0.25, −b_hij/0.25) − log(2)]
B_hij = 0.5 exp(−rho²/(2·0.35²)) s_hij
B_hii = 0; heads 4..7: B = 0
attention = softmax(original_qk_logits + B)
output = original_output_projection(concat(attention @ original_V))
```

实现先把对角分母替为 1，再做除法，避免先产生 NaN；使用稳定 `torch.logaddexp`。由于 `e_ij=e_ji`，第二端评分可用第一端评分的转置得到。B 对称、绝对值不超过 0.5，但最终 attention 不必对称。距离衰减只乘新增偏置，未引入注意力窗口或硬截断。

预测卷积遵循当前 AMP/half dtype；方向归一化、几何和评分显式在 FP32 中计算，再将 mask 转到 MHA 投影计算 dtype。直接调用原 `TransformerEncoderLayer.forward`，把 B 作为浮点 additive mask；每次仅一次原 MHA，原 FFN 和残差顺序不变。

几何缓存只有一组当前尺寸数据，非持久 buffer，不进入 state_dict；尺寸/input dtype/device 作为 key，`.to()`/`.half()`/`.float()` 清空缓存，避免 half 舍入后重复利用。forward 无 quantile、完整 attention 统计、跨 batch 激活缓存或第二条增强特征分支。原生 MHA 仍有标准全局注意力内存开销。

最后预测层为零时 `d=B=0`，数学上回到 AIFI。实测对齐和容差见验证文档；不承诺跨 CPU/CUDA/PyTorch 版本逐 bit 一致。

## 参数与初始化

| 模型 | nc=1、未融合参数量 |
|---|---:|
| C2 | 20,082,772 |
| C2 + DRA-AIFI | 20,087,148 |
| 增量 | 4,376 |

增量为 `256*16 + 16*3*3 + 16*8 + 8 = 4,376`，与设计相符。独立 val 中框架会 fuse，控制台显示的 19,882,092 是融合后模型数值，不是本表的统计口径。

唯一公共初始化源为 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256：

`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。

已读原 `init_rtdetr_r18_lite_imagenet_backbone.py`：其 torchvision ResNet18 ImageNet 权重映射到兼容骨干部分，其他参数是原始初始化，并保存为 half。DRA 使用已经核对散列的同一文件，严格检查 epoch=-1、无 optimizer/EMA/scaler/训练结果，完整核对 533 个公共状态及形状，转为 FP32 后逐键完全加载。缺失、额外、形状异常均拒绝继续，未静默 strict=False。4 个新状态为 predictor 第 0/2 层 weight、第 4 层 weight/bias，其初值范围和 nonzero 计数进入报告。

正式入口保持 C2 的 nc80→nc1 原生训练 API 重建流程。共 537 个 DRA 状态，精确加载 528 个，9 个类别相关 shape skip 逐项核对：DN class embedding、encoder score weight/bias、3 层 decoder score weight/bias。受控 C2 与 DRA 重建后全部公共状态相等。正式初始化在 start-direct 时重新创建，不复用任何测试模型或 optimizer。

## 配方与结果证据

[c2_args.yaml](c2_args.yaml) 是实际 109 字段归档副本。原始文件 SHA256 `ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`。服务器读取实际 C2 args 并逐字段比较值和类型；默认只改变 model/name/save_dir。显式设置 DRA_MAIN 时允许 project/data 路径前缀搬迁，并逐字段解释，所有训练超参数不变。

200 epochs、640、batch16、seed42、AdamW、lr0=0.0005、lrf=0.01、weight_decay=0.0001、warmup=5、cos_lr=True、AMP=True、workers=8、deterministic=True、patience=50、close_mosaic=10；完整在线增强和 loss 配置均来自实际 args。训练 setup 再次核对实际配置和每个新增参数恰好进入 AdamW 一次。

start-direct 只有路径、配置、权重、输出冲突等轻量检查，无 prepare/audit 通过标记。保留原生 AMP 检查（其官方 yolo26n 仅是 AMP 探针，不用于 RT-DETR 训练初始化或配方）。OOM 直接失败并保留日志，不自动减 batch/imgsz 或关闭 AMP。`full_server_preflight=NOT_RUN`。

独立评估使用同一训练 best.pt，val 完成后 test 检查 checkpoint SHA、数据 SHA、源码 SHA 和设置一致。评估口径 `native_c2_sorted_original_mask_v1` 保留 C2 原验证器，包括已知的“排序后使用排序前 mask”行为。导出复制本次预测后调用原生 postprocess，并逐项确认排序和筛选结果一致。不要直接与 C24/C26 的 corrected_sorted_conf_mask_v1 结果混算差值。

逐图压缩 JSONL 保留全部 300 queries 的实际排序、原图像素坐标（不裁剪/舍入）、分数、类别、`used_for_metrics` 和所有 GT；标记准确表达原生 mask 实际使用的子集。数据集不复制到结果包。AP50/AP75/mAP50–95、P/R、10 IoU 阈值 AP 数组和匹配口径来自同次评估；没有额外推理补造证据或新增分组指标。

pack-complete 只收集现存材料，包含 best/last、完整训练输出与可视化、独立 val/test 指标和图、同次预测/GT、日志、配置、源码快照、环境与初始化证据；无 20 MiB 限制。输出 tar.gz、SHA256、inventory、verification，并读取每个成员验证散列。材料缺失时也可导出可用证据，但必须 `complete=false` 并列出 missing_evidence；不会自动补跑。默认主仓库 `downloads/dra`，带时间戳、禁止覆盖。

操作见 [AUTODL.md](AUTODL.md)，实测见 [VALIDATION.md](VALIDATION.md)。
