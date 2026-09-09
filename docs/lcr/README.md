# C2 + LCR-AIFI 单模块实验

别名 `lcr_aifi`，分支 `codex/lcr-aifi`。从真实 C2 YAML 复制，仅 layer9 的 AIFI 改为一个 LCRAIFI。其余 backbone/head/from/repeats/args/scales 完全一致；没有 C17/C19/C24/DRA/EVC/RSC 组合，没有额外检测头、loss、增强、训练技巧或 CUDA 算子。历史基础设施中的其他模块仍保留，但不进入本模型图。

## 来源与现场

C2 源提交 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。工具分支基点 `beedcfa307e250fb2de47587097c51f9c141123b` 用于公共工具复用，不是将 C26 当作 C2。

实际C2归档为 `D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053`。归档模型、上述C2提交和当前C2 YAML解析完全相同。归档args与 [c2_args.yaml](c2_args.yaml) 全部109字段及类型相同。证据及散列见 [references.json](references.json)。

独立外部worktree `D:/MyProjects/Crack_RTDETR-lcr`。主工作树原分支、所有未跟踪文件和其他实验worktree/进程/日志均保留。检查适用父目录与版本树未发现AGENTS.md。

参考包实际路径 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA256 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`；`reference_access=AVAILABLE`。实际阅读transformer.py的ConvolutionalGLU及block.py的ChannelAggregationFFN，解压源文件与zip内成员散列一致。未导入extra_modules。

## 固定首版公式

输入/输出 `[B,256,H,W]`，隐藏维1024，8头；640图像通常为20×20特征。实际H/W在每次forward传给私有FFN，不用sqrt(N)或缓存。

```
U = reshape(fc1(X), B,1024,H,W)
S = DWConv3x3(replicate_pad(U,1))     # stride1, padding0, bias=False
M = avg_pool2d(replicate_pad(S,1),3,stride=1,padding=0)
D = float32(S) - float32(M)
E = D / sqrt(mean_channel(D**2) + 1e-6)
T = gate_out(GELU(gate_in(E)))       # 1024->64->1024, groups16, bias=True
Delta = 0.5 * tanh(T)
A = phi(S) + Delta * (phi(S) - phi(M))
FFN = fc2(original_dropout(contiguous_tokens(A)))
```

phi沿用原GELU。RMS的差分、平方、每位置通道均值、加epsilon和除法全为FP32；E按权重dtype转换并遵循autocast，支持真正half权重。tanh/差异重构用FP32，随后回到主激活dtype。直接用Delta，不先算1+Delta再相减。M/D/E均保留梯度；无detach、nan_to_num、额外norm/dropout/LayerScale或可训练alpha。

实际C2采用 **post-norm**：`X -> MHA(Q/K=X+PE,V=X) -> dropout1 -> +X -> norm1 -> LCR-FFN -> dropout2 -> +norm1输出 -> norm2`。原PE公式及宽高网格顺序、MHA参数全部保留。可编辑完整结构图：[LCRAIFI.mmd](LCRAIFI.mmd)。

pre-norm接口也保留：`N1=norm1(X); Z=X+dropout1(MHA(N1+PE,N1+PE,N1)); Y=Z+dropout2(LCR-FFN(norm2(Z)))`。两种路径均验证非零dropout和RNG对齐，没有第二个完整AIFI。

## 初始化和参数

| 模型，nc1未融合 | 参数量 |
|---|---:|
| C2 | 20,082,772 |
| LCR | 20,101,268 |
| 新增 | 18,496 |

DW为9,216；gate_in为4,160；gate_out为5,120。不要混用独立评估器打印的融合后19,896,212参数量。

DW仅中心权重1，其余0；gate_in Xavier uniform、bias0；gate_out weight/bias0。新增构造使用CPU fork_rng恢复公共层后续初始化随机序列，seed42可复现。初始S=U、Delta=0，数学上恢复原FFN，无参数判零后走旁路的代码。恢复contiguous token布局以保持原dropout掩码元素顺序；数值实测而非宣称跨平台逐bit一致。

唯一公共源为 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，与C2归档相同。原脚本将torchvision ResNet18 ImageNet兼容骨干权重映射到完整模型，其余保持原始初始化，并保存half。本次加载完整源checkpoint并FP32精确保留原half值，没有重新随机化encoder/decoder，也不从其他实验best起步。

源nc80、初始化nc80、原生训练API重建nc1。实测533个公共state逐项identity映射，新增5项：DW weight、两级gate weight/bias。无异常missing/unexpected/shape mismatch。nc80→1精确加载529/538项，9个分类shape适配例外逐一列出；受控C2/LCR的nc1公共state全部相同。完整比对：[initialization_mapping.json](initialization_mapping.json)、[validation.json](validation.json)。

## 配方、评估和复用

[resolved_server_config.yaml](resolved_server_config.yaml) 保存默认服务器109字段完整配方，[server_config_diff.json](server_config_diff.json) 逐项差异默认仅model/name/save_dir。200e、640、batch16、seed42、AdamW、lr0=.0005、lrf=.01、decay=.0001、warmup5、cos_lr、AMP、workers8、deterministic、patience50、close_mosaic10及所有在线增强/loss/划分来自该C2，不用早期150e配方。

服务器启动逐字段核对实际C2 args；LCR_MAIN只允许目录前缀搬迁。新增参数使用相同原生分组、各出现一次。配置、初始化、源码、进程、退出码和哈希全部记录；OOM不自动降batch/改AMP/imgsz，不续训。详细入口见 [AUTODL.md](AUTODL.md)。

独立评估复用C24的 `corrected_sorted_conf_mask_v1` 算法（函数原样复用，不导入实验模型），boxes/conf/classes共用排序和筛选索引。阈值conf=.001、iou=.7、max_det300、imgsz640、batch16、FP32、workers0，保留原生匹配、最终decoder输出和指标。逐图导出全部300 query预测、指标使用标记及全部GT和尺寸，P/R/mAP50/mAP50–95/AP75/10IoU AP、曲线和混淆矩阵来自同次原生评估。

训练best选择仍用原C2训练验证器。历史C2/DRA若采用未修正排序mask口径，不能直接与本入口指标相减宣称提升；正式对比需要使用相同已修正入口重评C2 best。本次未运行完整评估。

## 与参考和C24/DRA的区别

[TransNeXt](https://arxiv.org/abs/2311.17132) 提供邻域上下文通道混合思想；包内ConvolutionalGLU缩放隐藏维、两支路相乘且带额外shortcut。本实现保留1024 Linear FFN，无该shortcut，以局部参照非线性差异重构。

[MogaNet](https://openreview.net/forum?id=XhYWgjqCrV) 的本地ChannelAggregationFFN使用可学习C→1分解与ElementScale。LCR改用空间3×3均值、每位置通道RMS及分组有界Delta，不复制其分解卷积或可学习缩放。OpenReview网页本次遇到浏览器验证，未声称读过不可访问的网页全文。

C24/SCCA改注意力组织；DRA增加方向关系attention bias；LCR仅替换FFN激活计算，原QKV/logits/PE/头数不变。是否涨点、正式延迟和训练稳定性均待实验。本次实测与未验证范围见 [VALIDATION.md](VALIDATION.md)。
