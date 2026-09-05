# C18 GSDR-AIFI-v3 实现与交付

本候选已实现并完成本地合成输入审计，**尚未正式训练，没有 C18 val/test 成绩，也不承诺涨点**。它保留原 AIFI 全局建模路径，仅把 V2 的辅助全局稀疏关系改为逐查询局部采样。固定预算为 128 辅助通道、4 头、每头 4 点、每轴半径 2 个 P5 特征单元；这些数值不是 test 搜索结果。

本地基于 `30f2da6e4bb91dcae5f761bcafd5993e263197da` 创建分支 `exp-rtdetr-r18-lite-gsdr-aifi-v3`，worktree 为 `D:\MyProjects\Crack_RTDETR\outputs\worktrees\gsdr-aifi-v3`。主工作区的 CSCEF 分支及未提交文件没有改动；没有合并其他实验，也没有操作 C17 进程。适用目录内未发现 AGENTS.md。

## 已核对的附件事实

读取的文件位于 `D:\rtdetr跑结果\c16 r18_lite_gsdr_aifi v2`。两份 tar.gz 的实算 SHA256 与各自侧文件一致：

| 结果包 | SHA256 |
| --- | --- |
| `c16_gsdr_aifi_v2_complete_20260906_044540_931282.tar.gz` | `098758e34ccf1bb630b19d92e9ffe6a97ba9e54748fd8238932a38d1312a7569` |
| `c16_v2_val_20260906_051245_957279.tar.gz` | `7bae1ca186af921aac1ccec1a5f919110e9a0e22a37fe55e22fe820f41382e38` |

完整包的 `source/source_commit.txt` 为指定 C16 提交；逐字节读取 `training/weights/best.pt` 得到 `14af3acdbc051d1e8c58f072fefdf98d8b04788b0076a3400d2b0e57486e8da1`。读取 `baseline_c2/training/args.yaml` 的全部 109 字段，与仓库 `ultralytics-main/tests/fixtures/c2_original_args.yaml` 的值及 Python 类型逐项相同。本轮没有重新执行完整包内的 11,356 项逐文件清单校验，未将附件中的既往校验当作本轮执行结果。

七组 `summary.json`、各组 `metrics.json` 和 `activation_probe.json` 的数值与附件一致。七组均为固定 C16 权重、1,728 张 val 图的推理干预，原始/旁路/减半/清零偏移偏置/无偏移/无位置/去空间均值的 mAP50–95 分别是 `0.47722700784978417`、`0.4630698197388673`、`0.4709449969230793`、`0.39190492457696124`、`0.35388997857313687`、`0.47722721693266645`、`0.46307755286162255`。

原始前 128 张 val 的空间常量能量占比为 `0.9999993317760527`，分支/输入 RMS 为 `1.4237226778641343`，偏移分量饱和比例 `0.50`、点截断比例约 `0.34`，内容/位置 logits 的 key 维标准差为 `657.9037771224976`/`0.14061981299892068`，归一化熵 `0.18858344829641283`，最大权重均值 `0.49904295592568815`。这些是每图内部的空间统计，不能断言跨图像相同，也不能推广为全 val 每张图的证明。

从完整包读取的正式 test mAP50–95 为 C2 `0.46963623190802783`、C14 `0.46292202062558846`、C16 `0.46165065799769867`。C16 分支对已训练模型有用，但已测空间变化部分几乎无净贡献。减半、清零偏移或取消偏移在此次干预中均下降；去位置项近乎不变。它们不是重新训练的消融，也不能将 val 与 C2 test 混比。

## 实际源码引用与取舍

先在模块包实际检索了 `deformable`、`reference_points`、`sampling_offsets`、`grid_sample`，然后阅读以下实现。主目录可用，因此没有使用备用 ZIP `D:\7.1 rtdetr改\魔鬼面具_RTDETR\RTDETR-20260623.zip`。

| 实际文件 | 类/函数与阅读内容 | 本轮采用或舍弃 |
| --- | --- | --- |
| `D:\7.1 rtdetr改\魔鬼面具_RTDETR\RTDETR-main\ultralytics\nn\modules\transformer.py` | `MSDeformAttn`：逐查询 offset/weight 投影，heads/levels/points 布局，初始化与 reference 归一化 | 参考逐查询组织；改成单尺度、无 bias、有效域局部坐标。不使用多尺度 decoder、可学习径向 bias 或不受局部域约束的 reference+offset。 |
| `D:\7.1 rtdetr改\魔鬼面具_RTDETR\RTDETR-main\ultralytics\nn\modules\utils.py` | `multi_scale_deformable_attn_pytorch`：batch×head 展开、grid_sample、逐点加权求和 | 参考成熟张量布局；保留 query 轴，删去 level 轴，并把敏感路径放入 FP32。 |
| `D:\7.1 rtdetr改\魔鬼面具_RTDETR\RTDETR-main\ultralytics\nn\extra_modules\attention.py` | `DAttention`：分组采样、共享采样网格、全局 QK、位置项，yx 到 xy，align_corners=True | 用于核对旧式共享网格与本设计的区别；没有整体移植，没有保留 Q/K、相对位置 MLP 或依赖 N-1 的坐标。 |
| `D:\7.1 rtdetr改\魔鬼面具_RTDETR\RTDETR-main\ultralytics\nn\extra_modules\block.py` | `DySample`：像素中心、W/H normalizer、分组 grid_sample | 参考像素中心换算；不采用上采样、pixel shuffle、scope 门控或 border padding。 |
| `D:\MyProjects\Crack_RTDETR\outputs\worktrees\gsdr-aifi-v3\ultralytics-main\ultralytics\nn\modules\transformer.py` | `AIFI` / `TransformerEncoderLayer` 与 `MSDeformAttn` | 前者保持参数名、位置编码、dense attention、FFN、dropout 和两种 norm 顺序；后者核对项目本身的逐查询采样布局。原文件不改。 |
| `D:\MyProjects\Crack_RTDETR\outputs\worktrees\gsdr-aifi-v3\ultralytics-main\ultralytics\nn\modules\utils.py` | `multi_scale_deformable_attn_pytorch` | 核对本项目实际 grid_sample 配置和 batch/head/query/point 张量变换，原文件不改。 |
| `D:\MyProjects\Crack_RTDETR\outputs\worktrees\gsdr-aifi-v3\ultralytics-main\ultralytics\nn\modules\gsdr_aifi_v2.py` | `GSDRAIFIV2` / `SparseDeformableRelationV2` | 复用 AIFI 子类接入顺序、独立 RNG fork 和 PyTorch 2.1 CPU nullcontext 思路；辅助分支整体改造，V2 文件不改。 |

通用依据为 [Deformable DETR 论文](https://arxiv.org/abs/2010.04159)、[官方 MSDeformAttn 源码](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py) 和 [PyTorch 2.1 grid_sample 文档](https://docs.pytorch.org/docs/2.1/generated/torch.nn.functional.grid_sample.html)。参考的是稀疏采样思想和张量/坐标语义，不能据此推出 C18 的训练收益。未加入 CSCEF、Scharr、跨尺度门控、cosine、条形卷积、额外系数或 loss。

## 结构与数值实现

新增 `ultralytics-main/ultralytics/nn/modules/gsdr_aifi_v3.py`，包含 `GSDRAIFIV3(AIFI)` 和 `QueryLocalDeformableRelationV3`。仅新建 V3 YAML 并在两处注册；层 9 之外 YAML 结构、连接、层索引及 decoder 输入均与 C2 一致。

五个 1×1 投影全部 `bias=False`：input `256→128`、value `128→128`、offset `128→32`、weight `128→16`、output `128→256`。input/value 明确使用 Xavier uniform；offset/weight/output 权重全零。逐 token 的通道 LayerNorm 为 `normalized_shape=(128,)`、`elementwise_affine=False`、`eps=1e-5`。没有新增可学习归一化参数或 optimizer bias 组条目。

固定四点锚点 buffer 按 xy 保存 `(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)`；half 能精确保存这些值，atanh 在 FP32 中实时计算。每查询每轴采用 `lo=max(0,p-2)`、`hi=min(N-1,p+2)`、`center=(lo+hi)/2`、`extent=(hi-lo)/2`，然后 `pixel=center+extent*tanh(raw+atanh(anchor))`、`grid=2*(pixel+.5)/N-1`。固定窗口界限不依赖学习结果，没有对最终位置做 clamp。边缘窗口向内缩；singleton 轴得到 pixel=grid=0。

grid 保留 `[B,H*W,heads,points,xy]`，softmax 只沿每个 query/head 的四点轴。value 按连续通道分头，调用 `grid_sample(mode='bilinear', padding_mode='zeros', align_corners=False)` 后逐点求和。没有全局 QK 矩阵、共享 10×10 网格、位置 MLP、offset GroupNorm、output bias、均值去除或 0.5 缩放。

input/output 两端投影正常参与外层 AMP；LayerNorm、value/offset/weight 投影、坐标、softmax、grid_sample 和聚合在禁用 CUDA autocast 的 FP32 路径执行。显式 half 权重经可微 `.float()` 用于内部投影，聚合在 output 投影前恢复参数 dtype。CPU FP32 使用 nullcontext，不构造 CPU FP16 autocast。

pre-norm 时辅助输入为原 norm1 后特征，delta 加在 dense 残差之后、norm2 前；post-norm 时 delta 加在 dense 残差之后、norm1 前，与 V2 接入位置相同。极大 raw 仍可能使 tanh 饱和；局部域约束不代表消除所有梯度饱和。每轴半径是 2，二维 L2 上限可达 `sqrt(8)`，640 输入时每个 P5 单元对应 32 输入像素。

## 初始化、审计与配置锁定

工具由已验证 C16 工具独立派生，未改 trainer/optimizer/EMA/loss/匹配器。干净 C2 初始化 SHA256 固定为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。新增分支在 `torch.random.fork_rng(devices=[])` 内构造，保留所有共同 state_dict 路径和后续分类层 RNG 顺序。

初始化 checkpoint 以 FP32 保存，epoch=-1，best_fitness/optimizer/EMA/updates/scaler/训练指标全为空；拒绝覆盖已有权重。共同状态、分类头、完整嵌套输出、保存/三种重载、nc=1/80 的真实 `RTDETR.train → RTDETRTrainer.get_model` 路径均实际比较。测试夹具只替换 trainer 的数据初始化和循环入口，继承原 get_model，在任何数据读取或 optimizer step 之前停止。正式训练另在第一批之前核对实际模型全部 state hash、完整 args、真实 optimizer 分组、空 optimizer 状态及 EMA updates=0。

`docs/gsdr_aifi_v3_protected.json` 按 C16 起点保护 886 个已有文件，仅两处注册文件例外；使用统一 LF 的内容 hash 避免跨平台换行误报。其他旧模型、旧工具及旧测试保持不变。运行审计和启动计划绑定当前代码 hash、checkpoint hash、Python/PyTorch/CUDA/GPU 和实际导入路径。

完整 109 字段的值和类型对照见 [C18_C2_109_fields.md](C18_C2_109_fields.md)。启动器必须实际读取服务器权威 C2 args，与已核对的原始 fixture 逐字段逐类型比较；**只允许 model/name/save_dir 改变，project 保持原值**，没有 `--project` 接口，也没有缺字段回填。`device` 保留字符串 `'0'`。正式启动要求本机实际 PyTorch **2.1.2** + CUDA 审计通过；本地 2.7.1 的通过报告不能代替它。

锁参计划默认只准备，明确 `--tmux` 或 `--execute` 才启动。已有 run 目录、日志或原子 `.c18.launch.lock` 会阻止重复训练。tmux supervisor 保存 bootstrap.log、console.log、exit_code.json 和 process_exit_code.json；bootstrap/import 失败也有退出码。审计失败直接报原因，不调整 batch/workers/AMP/配方。

## 实际本地结果与未运行范围

完整结果和审计范围见 [C18_LOCAL_AUDIT.md](C18_LOCAL_AUDIT.md)。本机 Python 3.9.25、PyTorch 2.7.1+cu118、RTX 2060；不将其表述为服务器 PyTorch 2.1.2 验证。新增参数 88,064；nc=1 模型 20,170,836，nc=80 为 20,272,272；533 个共同状态 + 5 个新权重 + 1 个 anchors buffer = 539 个状态。

本轮没有运行真实数据训练、真实 val/test、超参数搜索或未来训练权重的 128 图诊断。Linux tmux 实际派发、服务器 2.1.2 兼容性和真实首批之前的 runtime preflight 尚须执行服务器指令；没有远程连接服务器代替用户启动实验。

## AutoDL 启动与诊断

完整可粘贴命令见 [C18_AUTODL.md](C18_AUTODL.md)，实现为正常维护的 `tools/launch_c18_autodl.sh`。顺序是获取代码、创建/核对独立 worktree、初始化、一次兼容性审计、完整计划、`--tmux` 正式启动。不会重装环境、修改全局 editable 路径、等待或终止 C17。启动后读取实际 tmux.json 打印 session、日志和退出状态路径。

之后可只读诊断 C18 已训练权重：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd /root/autodl-tmp/projects/Crack_RTDETR_gsdr_aifi_v3
export PYTHONPATH="$PWD/ultralytics-main"
python tools/diagnose_gsdr_aifi_v3.py \
  --weights /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c18_rtdetr_r18_lite_gsdr_aifi_v3_e200_b16_onlineaug/weights/best.pt \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --device cuda:0 --limit 128 \
  --output outputs/gsdr_aifi_v3/val_probe_128.json
```

该入口只读取排序后的前 128 张 val 图及 checkpoint，不建立 labels cache，不保存修改后权重，不计算 mAP。按 RTDETR val 的 640 stretch/RGB/255 预处理，FP32 单图前向，无预热图计入；报告每图空间常量能量、RMS、逐查询位移、有效域外比例、点权重熵。零 delta 的能量比为 null，不把它当成训练后“不退化”的证据，也不根据阈值调整结构。
