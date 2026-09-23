# 本地验证及服务器待办

本地 Python `D:\miniconda3\envs\rtdetr\python.exe`，3.9.25，torch2.7.1+cu118，RTX2060。导入路径固定本RDL工作树的 `ultralytics-main`；未安装/升级环境。源码母版提交为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。报告生成时新代码尚未提交，因此当时Git HEAD记录为母版；实际新增源码内容SHA另见 `source_audit.json` 和最新 `preflight_local.json`，不把母版HEAD冒充交付SHA。

## 检查结果

| 项目 | 状态 | 实际覆盖 |
|---|---|---|
| 母版源码/CBR/LIF/YAML | PASS | 原结构与模块源码未变，真实节点20/26、head输入19/22/25 |
| 统一源/受控初值 | PASS | 源SHA符合；epoch=-1、nc80；受控公共状态及原模块零初始化审计通过 |
| nc适配/参数 | PASS | 原Trainer仅适配9个类别keys；未融合20,149,765；RDL新增0参数/0buffer |
| 109字段配方/数据指纹 | PASS（本地） | 使用归档C2完整args；本地路径/标签指纹等于成功母版；服务器实际args仍需prepare核对 |
| 数学与梯度 | PASS | 四边16个正负误差案例；4/11示例；空GT、像素下界、保护计数、非法值拒绝、固定目标有限差分 |
| 索引/原L0 | PASS | 两图GT数1/2、0/0、0/1及DN有无；独立标量循环；真实母版criterion各项精确相等，最终匹配仅一次 |
| 零权重更新 | PASS（CPU） | 同状态、同RNG/DN、原AdamW分组及clip，一步独立更新最大差0；未事后复制参数 |
| CUDA真实路径 | PASS | B2、160×192；启用诊断预测精确相等；激活RDL时原L0精确相等；FP32与有界原生AMP梯度有限 |
| 生命周期 | PASS | 调用真实Trainer.get_model；原save_model、加载、resume_training；19号checkpoint恢复到e20、权重.05 |
| EMA/普通推理/融合 | PASS | 形状B×300×5；验证L0-only；融合对照atol=rtol=3e-5；LIF保留BN；融合19,944,965参数 |
| 实际Validator/入口保护 | PASS | 隔离2张合成图B2/160，原Validator实际运行；改动资产、失败预检、外来/strip checkpoint拒绝；正确排序mask案例 |
| 母版现象诊断 | PASS | train/val各128张，seed42，B2/640，FP32 eval，无随机增强，无test |
| 服务器torch2.1.2/原生AMP | SKIPPED | 本机未连接服务器；交付preflight入口 |
| 服务器B16/640容量 | SKIPPED | 需要服务器执行一次隔离前后向，不做optimizer update |
| tmux实际派发、200e正式训练 | SKIPPED | 仅用户显式start；Linux shell通过bash -n，本地没有执行服务器调度 |
| RDL独立val/test与AP提升 | SKIPPED | 尚无RDL正式训练权重；未进行test或额外消融 |

## CUDA差异定位（不放宽所有容差）

最初以“CUDA两次完整反向+AdamW更新必须逐位相等”为条件的检查未通过。原生 `grid_sampler_2d_backward_cuda` 在 deterministic=True/warn_only 下仍提示没有确定性实现。随后仅使用**母版**、两份相同初值、相同batch/RNG及原criterion重复一步：loss均为80.0660400390625，最大梯度差0.00061798095703125，最大参数更新差0.00021503493189811707。没有启用RDL；完整记录见 `native_cuda_behavior.json`。这证明原生完整CUDA反向不适合逐位更新判据，不是top-k身份差异，也未宣称CUDA参数更新精确相同。

因此精确有效更新对照固定在CPU；迁移这两份**独立完成更新**的等值模型到CUDA后，仍对诊断预测和激活RDL的原损失保持精确比较。CUDA只额外要求实际前后向有限性、恢复、推理和融合通过；未靠宽松全局容差抹掉更新差异。

默认GradScaler初值65536在隔离母版上也出现缩放梯度溢出：AMP loss=43.20314025878906且有限，329个参数的缩放梯度非有限。母版 `tools/check_c19_lif_v1.py` 的既有有界检查使用init_scale=128，本检查沿用该值，RDL AMP loss=42.947696685791016且前后向有限。正式Trainer依旧使用原生默认scaler，原生跳步/动态scale照旧，并显式记录溢出。没有关闭AMP、修改正式配方或更换环境。

修复过一个GPU恢复审计问题：重建模型在CPU，读取checkpoint可能在GPU，逐值比较需显式移到相同设备。实际保存/加载/resume复查已通过。早期检查fixture还补齐原Trainer会设置的nc和args.model；这些不改变模型算法。

## 现象诊断

诊断权重SHA256：`24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`，对应本地已归档成功CBR＋LIF母版best.pt。128张样本清单及各图/标签SHA见 `diagnostic_samples.json`；完整统计见 `diagnostic_summary.json`。逐匹配原始边数据位于工作树 `outputs/rdl_v1/diagnosis/pairs.json`，LIGHT保留该文件。

| split | 图像 | M | 超范围边比例 | L_out | L_in | 匹配平均IoU：before→after |
|---|---:|---:|---:|---:|---:|---:|
| train | 128 | 937 | 21.2914% | 0.1149244331 | 0.0939704123 | 0.7557148882 → 0.7893905638 |
| val | 128 | 949 | 21.9178% | 0.1392465312 | 0.0987358906 | 0.7465564519 → 0.7775768885 |

GT短边16–32像素组的超范围比例train37.62%、val36.58%，≥32像素组分别15.99%、17.52%。小于8像素的匹配仅train0个、val1个，不能由此推断普遍规律。a保护次数均0；非零微小误差保护train0、val1。vstar饱和比例在本样本恰好等于超范围比例；完整a/S/eta/|v|分布和分组量级已保存。

观察到稳定的超范围边和可达目标量级，为投入一轮固定配方实验提供现象依据；**它既不证明RDL改善AP，也不能把母版CBR自身的IoU修正增益归因于尚未训练的RDL。**

## 本地实际命令

在独立工作树 `D:\MyProjects\Crack_RTDETR\outputs\worktrees\Crack_RTDETR-rdl_v1` 执行：

```powershell
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/rdl_v1.py prepare --local --main 'D:\MyProjects\Crack_RTDETR' --data 'D:\MyProjects\Crack_RTDETR\configs\crack.yaml' --c2-args docs/c19_lif_v1/c2_args.yaml
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_rdl_v1.py --source 'D:\MyProjects\Crack_RTDETR\weights\rtdetr_r18_lite_imagenet_backbone_init.pt' --device cpu --output outputs/rdl_v1/local_cpu_checks.json
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/rdl_v1.py preflight --local --device cuda:0
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_rdl_v1_ops.py
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_rdl_v1.py --source 'D:\MyProjects\Crack_RTDETR\weights\rtdetr_r18_lite_imagenet_backbone_init.pt' --native-cuda-repeat --output outputs/rdl_v1/native_cuda_behavior.json
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/diagnose_rdl_v1.py --weights 'D:\rtdetr跑结果\C19＋LIF：原 CBR＋LIF-Down v1\training\weights\best.pt' --dataset 'D:\MyProjects\Crack_RTDETR\datasets\crack_det' --device cuda:0 --batch 2 --limit 128
& 'C:\Program Files\Git\bin\bash.exe' -n tools/sync_rdl_v1.sh tools/rdl_v1.sh
```

沙箱身份与创建工作树的身份不同，调用Git时只对本命令设置该工作树的safe.directory，没有改用户全局Git配置。初次prepare内部母版runtime使用环境形式的 `GIT_CONFIG_COUNT=1/GIT_CONFIG_KEY_0=safe.directory/GIT_CONFIG_VALUE_0=本工作树`。本地报告不可作为服务器start授权；服务器必须重新prepare/preflight。
