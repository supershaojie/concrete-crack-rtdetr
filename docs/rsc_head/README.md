# C2 + RSC-Head 单模块实验

本分支为 `codex/rsc-head`，实验别名 `rsc_head`，不占用新 C 编号。基线与分支起点均为 C2 源码包提交 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`，不是其他实验的当前 HEAD。主目录、其他 worktree、日志、进程和结果均未修改。本实验不引入 LCR、C19/CBR、C17、C24、DRA、EVC 或组合模块。

## 结构、框来源与梯度

独立 YAML 为 `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-rsc-head.yaml`。与原 C2 YAML 的唯一结构差异是最后一个容器名 `RTDETRDecoder` → `RTDETRDecoderRSC`。输入仍为 `[19,22,25]`，模型层号 26；256 hidden channels、300 个普通 query、三层 Decoder、最后层 eval_idx=2、nc=1。只有 `model.26.dec_score_head.2` 成为 `RSCHead`；前两层分类投影、全部框回归、attention/FFN、Encoder query selection 和 Backbone/AIFI/Neck 保持原样。

`DeformableTransformerDecoderRSC` 是独立子类，复用原层对象及 state_dict 层级。训练的 b1 为 refined_bbox；b2/b3 是原代码 `sigmoid(bbox + inverse_sigmoid(last_refined_bbox))` 辅助梯度重算后、实际进入 dec_bboxes/loss 的框。迭代 refer_bbox 的原 detach 位置保持不变。推理/导出记录三次已经产生的 refined_bbox，仅运行最终分类投影，不重跑前两分类头或 Decoder。history 是 forward 局部变量。训练 DN query 和普通 query 均按原固定索引处理，不聚合、排序或混合轨迹；Q 没有写死为 300。

q_cls 仅进入末层分类线性投影，不写回共享 q，不进入框回归。几何条件在头内 detach；q、原分类参数和新映射保留分类梯度。原生 RTDETRDetectionModel 的 `RTDETRDetectionLoss(nc=self.nc,use_vfl=True)`、VFL、Hungarian 代价、辅助 loss 和 DN 拆分保持不变。detach 仅切断新增条件直接传回历史框的梯度；分类仍更新共享 query，学习后的分类也会影响匹配，不能称作完全不干扰回归。

## 固定公式

三个归一化 cxcywh 框先 detach 并转 FP32，安全宽高仅用于比值/对数：`w_safe=max(w,1e-4)`、`h_safe=max(h,1e-4)`。

对 t=2,3：

```
d_t = [(cx_t-cx_prev)/w_safe_prev, (cy_t-cy_prev)/h_safe_prev,
       log(w_safe_t/w_safe_prev), log(h_safe_t/h_safe_prev)]
u_t = tanh(d_t)
g = [u2(4), u3(4), u2*u3(4), IoU(b1,b2)(1), IoU(b2,b3)(1),
     tanh(log(w_safe_3/h_safe_3)/4)(1), tanh(log(w_safe_3*h_safe_3)/8)(1)]
```

g 为 `[B,Q,16]`。IoU 是同索引对齐普通 IoU；负宽高先非负保护后转 xyxy，面积与交集均按该 xyxy 计算，不裁剪图像边界，不用 1e-4 扩大 IoU 框，union 下限 1e-12，零 union 得 0。不构建 Q×Q 矩阵、不使用 GT 或匹配结果。极细框 xyxy 在 FP32 中有正常端点舍入，不以 nan_to_num 掩盖问题。

```
RMS(q) = q / sqrt(mean(q**2, dim=-1, keepdim=True) + 1e-6)
s = SiLU(sem_proj(RMS(q)))             # 256 -> 16, with bias
t = tanh(geo_proj(g))                 # 16 -> 16, with bias
h = s*t
Delta = 0.5*tanh(mod_proj(h))          # 16 -> 256, with bias
q_cls = q + q*Delta
logits = Linear(q_cls; original W_cls, original b_cls)
```

RMS 仅沿每个 query 的通道轴，无去均值或可学习 affine。原值路径使用原 q。几何、RMS 和条件乘加使用 FP32；Linear 输入转换为对应参数 dtype，允许 autocast 选择算子精度；q_cls 转回 q dtype。输出是 logits，头内不 sigmoid/softmax、不产生独立质量分数或额外 loss，不硬编码修正大小与置信度的正负关系。

RSCHead 继承 nn.Linear，直接保留原 weight/bias 参数对象，不再次初始化它们、不重复注册旧头。sem/geo 使用局部 seed42 的 Xavier uniform、零 bias；mod weight/bias 全零。局部 fork_rng 恢复调用者 RNG，不改变其他公共参数的初始化序列。无零参数检测跳过分支；加载训练 checkpoint 不会再次清零调制。

| 参数 | 数量 |
|---|---:|
| sem_proj | 4,112 |
| geo_proj | 272 |
| mod_proj | 4,352 |
| 合计新增 | 8,736 |
| C2，nc=1，未融合实测 | 20,082,772 |
| RSC，nc=1，未融合实测 | 20,091,508 |

可编辑结构图见 [architecture.mmd](architecture.mmd)。未来若另行研究 C19 组合，最后条件框须使用 CBR 后实际进入 loss 的最终框；本分支仅记录此约束。

## C2 配方与公共初始化

真实配方来自 `D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053/train_run/args.yaml`，原样存于 [c2_args.yaml](c2_args.yaml)。配方和数据档案来源/哈希见 [provenance.json](provenance.json)。该记录与 C17/C19 对照的 C2 同属 200e 在线增强系列：200e、640、batch16、seed42、AdamW、lr0=.0005、lrf=.01、decay=.0001、warmup5、cos_lr=true、AMP=true、close_mosaic=10；workers8、device0、deterministic=true、patience50。

增强保持 hsv=(.015,.5,.35)、degrees5、translate.1、scale.4、shear1.5、perspective.0002、flipud.2、fliplr.5、mosaic.8、mixup.05、cutmix0、copy_paste0、erasing0。完整字段以存档为准，不使用早期 150e 配方。数据原划分 train6048/val1728/test864，配置见 [c2_data.yaml](c2_data.yaml)；不重划。启动时对真实 C2 args 的每个字段和值类型严格检查，同时核对数据路径、划分和类别。允许差异只有 model/name/save_dir；覆盖 RSC_HEAD_MAIN 时 project/data 仅作同一根目录迁移。

统一源为主仓库 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256：

```
fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
```

该 C2 公共初始化文件名说明其预训练来源为 ImageNet backbone，但实验加载文件中**全部**公共状态（包括原 Encoder/Decoder 初始化），不是只加载 backbone，也不是训练 best。源 nc=80，533 个公共状态逐一同名、同形、同值核对，RSC 新增 6 个状态，总计 539。原始半精度存值精确提升 FP32 后装载。nc80→nc1 遵照原训练器重建/同形加载规则，9 个类别维度不兼容项单列：denoising_class_embed.weight；enc_score_head.weight/bias；三层 dec_score_head 的 weight/bias。这些参数按同一 RNG 的 C2 nc1 构造初始化，而非手工挑选某一 COCO 类。其余 530 个状态（含新映射）精确加载；全部 533 个 nc1 公共状态也与同步构造 C2 精确一致。

`build_training_model` 在原训练器当前 RNG 下分别构造 C2 与 RSC，逐项核对包括上述类别适配项。所有参数均使用原 AdamW 规则：bias 不 decay，新映射 weight 正常 decay，每个参数只出现一次且可学习。启动后还保存实际 trainer args 与优化器分组，以检查运行时没有改配方。

## 必要验证与范围

实测结果和容差见 `checks.json`、`initialization_mapping.json`、`ops_checks.json`、`validation.md`。验证用本机 Windows/Python3.9/PyTorch2.7.1+cu118/RTX2060，不能代表 AutoDL 4090/PyTorch2.1.2。

本地有限整网检查为 batch2/imgsz160，含原生随机 DN、loss 和反向；FP32 比较对齐 RNG，TF32 关闭。单头测试覆盖 Q=503、描述符顺序/方向/退化框、query 置换和相互独立、输入不变、梯度边界和调制学到非零后的轨迹作用。另以可比 BN/无 DN 条件核对 train/eval/export 历史语义，独立保留原生 DN/loss 测试；不将该接口对齐当作 DN 验证。

CUDA 原生 grid_sample backward 无确定性实现，按原框架 `warn_only=True` 记录告警，不宣称严格确定性反向。没有关闭正式 AMP 或 DN。兼容性覆盖 CPU/CUDA FP32、CUDA AMP 前反向及优化器步、真正 model.half() 推理、学习后模型与优化器保存重载。真实数据仅检查两张训练样本的 DN/loss 与同次预测/GT 导出，不报告整数据集 AP。

`full_server_preflight`、AutoDL 4090/PyTorch2.1.2 实机验证、正式 batch16 显存、完整训练精度/延迟、完整 val/test 均为 **NOT_RUN**。不保证涨点或发表资格。生命周期测试中的 dispatch/OOM/success 是明确标注的模拟夹具，不能理解为实际训练。

## 参考关系

参考模块 zip 在指定 scratch、项目及相邻资料目录未找到，`reference_access=NOT_AVAILABLE`。没有声称读取 Star_Block 源码，按附件自包含公式实现；仅借鉴附件描述的低维乘法交互，不搬入卷积或外部残差块。

- [Rank-DETR](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html)讨论分类排序与定位质量，包含排序导向结构、loss 和匹配设计。本实验保留 C2 loss/匹配，仅改变最后分类投影。
- [GFLV2](https://arxiv.org/abs/2011.12885)用框分布统计预测定位质量。本 RT-DETR 没有四边离散分布，不移植 DGQP，也没有独立质量输出。
- [Rewrite the Stars](https://arxiv.org/abs/2403.19967)研究逐元素乘法交互。本实验将语义与同 query 修正状态在 16 维相乘，未引入 StarNet。
- [CLSC-DETR 预印本](https://arxiv.org/abs/2608.21457)已有跨层几何支持和分数校准先例。本设计的区别是“同一 query 的有序修正状态＋原分类投影内部的条件化交互”，不宣称跨层信息或质量感知首次提出。

服务器操作见 [AUTODL.md](AUTODL.md)。同步只固定提交、不会启动；仅 `start-direct` 启动正式训练。本次交付不执行正式训练或完整数据集评估。
