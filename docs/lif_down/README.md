# C2 + LIF-Down v1

独立实验 `lif_down`，分支 `codex/lif-down`，直接从真实 C2 提交
`67c3078e54a657fd96d65fee657a75fbb1dae0d6` 建立。
本地 worktree 为 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/lif-down`。
没有启用其他实验模块，没有运行正式 200e 或完整 val/test。

## 来源与公平对照

原始 C2 归档：`D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053`。
`environment/git_state.txt` 明确给出上述 C2 SHA，归档 YAML 与该提交的 C2 YAML 解析内容一致。
C17/C19/C24/LCR/RSC 记录的 C2 SHA 一致；C24/LCR/RSC 的 C2 args 与归档全部 109 字段及类型相同，
C19 记录只有模型与输出身份变化。审计路径、散列及结论见 [provenance.json](provenance.json)。
没有把主工作树 CSCEF-v4 HEAD 当作 C2；未发现适用 AGENTS.md。用户未跟踪文件、其他工作树与输出均保留。

公共初始化：`weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
它是 epoch=-1 的 nc80 初始化，ImageNet 用于 backbone，但加载文件中全部 533 个公共状态，
包括 Neck/Encoder/Decoder 的原始初始化；绝不使用任何实验 best.pt。
源 FP16 存值精确提升到 FP32，同名同形 identity 映射，原下采样 `conv`、`bn` 完整继承。
五个 NEW 权重是 B_proj/P/U_mix/U_dw/O_proj。MISSING/UNEXPECTED/SHAPE_MISMATCH 均为空。

正式训练仍走原 RTDETRTrainer nc80→nc1 重建。只有 9 个类别相关状态保留原生 nc1 随机初始化：
DN class embedding、encoder score weight/bias、3 层 decoder score weight/bias；不挑选 COCO 某类。
新增分支在局部 CPU fork_rng 中 seed42 初始化，离开后恢复 RNG；普通 Conv 构造消耗保持原样。
训练入口还使用原训练器当前 RNG 构造受控 C2，核对全部 533 个 nc1 公共状态完全一致。
nc1 精确加载 529/538 个状态，其余 9 个单列 ALLOWED_CLASS_ADAPTATION。
完整逐项清单见 [initialization_mapping.json](initialization_mapping.json)。

## 唯一修改位置

从原 YAML 复制 `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-lif-down.yaml`，
唯一结构差异是 P3→P4 的 Conv 改为 LIFDown。
`tools/lif_down_topology.py` 从 RTDETRDecoder 的三个输入找到 P3/P4/P5 RepC3，
再沿 P4 RepC3 的 Concat 反查来自 P3 的 stride2 边，代码不写死层号。
当前真实图解析为：

```text
P3: RepC3(19) → LIFDown(20; k3/s2/p1) → Concat(21, Y4=15) → RepC3(22)
P4: RepC3(22) → 原 Conv(23; k3/s2/p1) → Concat(24, Y5=10) → RepC3(25)
Decoder inputs = [19,22,25]
```

Backbone、AIFI、其他 RepC3、Decoder、300 queries、DN、matcher 和 RTDETRDetectionLoss 均为 C2 原实现。
保留原始 Conv 类行为，parser 仅显式注册新模块，无 monkey patch。

## 计算定义

固定 r=16，c1=c2=256；H/W 动态。

```text
Z = B_proj(X)                         # 1x1, 256→16, bias=False, no BN
Z = replicate_pad_right_bottom(Z)     # only if odd; original X unchanged
a,b,c,d = Z[even,even], Z[even,odd], Z[odd,even], Z[odd,odd]
A  = (a+b+c+d)/2
Dh = (a-b+c-d)/2
Dv = (a+b-c-d)/2
Dd = (a-b-c+d)/2
D = stack(Dh,Dv,Dd,dim=2).flatten(1,2)
P(A) = GELU(GroupConv3x3(replicate_pad1(A)))    # 16→48, groups16
E = D-P(A)
U(E) = DWConv3x3(replicate_pad1(GELU(GroupConv1x1(E)))) # 48→16→16, groups16
A' = A+U(E)
T = concat(A',E)                     # 16+48=64
R = O_proj(GELU(T))                  # 64→256, 1x1, bias=False
Y = act(BN(conv(X)+Align(R)))
```

D 的通道排列严格为 `[Dh_0,Dv_0,Dd_0,Dh_1,Dv_1,Dd_1,...]`，
保证 groups16 的每组处理自己的潜在通道。不是 `cat([Dh,Dv,Dd],1)`。
P/U 不加 BN、额外 gate 或 attention。

Align 的几何依据：以输入像素中心为整数坐标，原 k3/s2/p1 输出中心在 2i，
四相位块中心在 2i+0.5，偏移 0.5 个细网格像素，即 0.25 个粗网格单位。
固定重采样在每轴采用 previous 权重 .25、current 权重 .75，先横后纵；左侧/上侧 previous 用 replicate。
因此 Rx=.25·previous_x(R)+.75·R，Rxy=.25·previous_y(Rx)+.75·Rx。
这是固定几何对齐，不是声称非线性 lifting 后能精确恢复连续信号。
模块构造拒绝非 k3/s2/p1/d1 的几何；两路 shape 必须相同。

仅 O_proj.weight 全零。B/P/U 显式 Kaiming uniform(a=sqrt(5)) 正常初始化，使用局部 seed42。
O=0 时真实计算产生 R=0，数学上恢复 C2 Conv；forward 没有零权重捷径，也没有 zero alpha。
首步 O 有梯度而 B/P/U 为零属于预期；O 更新后这些权重均获得有限非零梯度。

由于残差位于 BN 前，通用 Conv-BN 融合不能直接只融合主 Conv。`BaseModel.fuse` 对 LIFDown 保留 BN，
其余普通层照常融合。非零 O 的整网融合、保存重载与真实 half 都有回归验证。

## 参数与验证

| 项目 | nc1 未融合参数 |
|---|---:|
| C2 实测 | 20,082,772 |
| C2 + LIF 实测 | 20,103,876 |
| 增量 | 21,104 |

增量 = B 4,096 + P 432 + U 48+144 + O 16,384 = 21,104。
验证明细见 [validation.md](validation.md)、[checks.json](checks.json)、[ops_checks.json](ops_checks.json)。
本地 smoke 的 batch2/imgsz320 只用于两个真实训练样本的 DN/loss，不改变正式 batch16/imgsz640。

## 正式配方与脚本

[c2_args.yaml](c2_args.yaml) 是权威 C2 109 字段归档，原始 SHA256
`ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`。
正式参数 200e/640/b16/seed42、AdamW、lr0=.0005、lrf=.01、decay=.0001、warmup5、cos_lr、AMP、close_mosaic10，
以及 workers8/device0/deterministic=true/patience50 和全部在线增强字段均继承。
[resolved_formal_config.yaml](resolved_formal_config.yaml) 与 [recipe_diff.json](recipe_diff.json) 显示默认仅 model/name/save_dir 改变。
服务器启动还要重新逐字段及类型检查真实 C2 args 与数据配置；显式 LIF_DOWN_MAIN 搬迁时，project/data 只允许根目录搬迁。

生命周期、评估和同步基础代码复用已审阅的本地 RSC 工具，独立适配到 C2 基点，
不导入任何其他实验模块，也不复制其他实验整套源码。
详见 [AUTODL.md](AUTODL.md)。

SPDConv 仅启发相位折叠实现检查；WTConv2d 仅供高低频表示思想参考。
实际参考路径 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-main/ultralytics/nn/extra_modules/`，
已阅读 block.py 的 SPDConv 和 wtconv2d.py，未移植 extra_modules、pywt 或其实现。
这些参考不构成本项目创新性的证明。v1 是否涨点只能由后续同口径实验结果决定。
