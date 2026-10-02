# RT-DETR 对比实验公共准备报告

2026-10-02，北京时间。公共软件准备已完成；当前旧划分不满足“未见原图独立泛化”的论文条件。没有训练、GPU 推理、重分数据、修改既有模型/损失或环境安装。

## 交付位置与范围

- 工作树：`D:/MyProjects/Crack_RTDETR/outputs/worktrees/comparison-base`。
- 分支：`bench/comparison-base`；母版基点：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
- 远端：`https://github.com/supershaojie/concrete-crack-rtdetr.git`。建立分支前正常查询确认远端没有同名分支，未覆盖旧准备工作。
- 本报告所在提交与最终 Git 推送回执确定交付身份。提交后可用 `git rev-parse HEAD` 与 `git ls-remote origin refs/heads/bench/comparison-base` 核对；最终回复列出完整本地及远端 SHA。
- 原主工作区保持 `exp-rtdetr-r18-lite-cscef-v4` 和原有未跟踪文件。核对了祖先目录与仓库，未发现适用的 `AGENTS.md`；阅读了 README、数据处理及标注说明。未创建八个模型分支。

新增文件限定于公共目录和所需服务器入口，已有 `.gitignore` 的 `outputs/` 足够，无需修改。按用途列出全部代码文件：

| 文件 | 用途 |
|---|---|
| `benchmarks/comparison/configs/model_registry.yaml` | 八模型顺序、官方上游、计划分支和待固定字段；四组历史锚点 |
| `benchmarks/comparison/configs/protocol.yaml` | 数据身份、跨框架增强、训练预算、初始化与评测规则 |
| `benchmarks/comparison/configs/mother_augmentation.yaml` | 母版配置来源和逐项增强证据模板 |
| `benchmarks/comparison/configs/asset_locations.yaml` | Windows/服务器路径映射，可用 `--run` 覆盖 |
| `benchmarks/comparison/prepare.py` | 一次准备、明确状态、已有证据复用 |
| `benchmarks/comparison/common.py` | 原始哈希和稳定序列化 |
| `benchmarks/comparison/dataset.py` | 实际 split、图像/标签、重复和原图族审计 |
| `benchmarks/comparison/assets.py` | 目录/归档内资产索引和原始小文件副本 |
| `benchmarks/comparison/augmentation.py` | 增强表与可选 CPU 对象构建核验 |
| `benchmarks/comparison/convert_annotations.py` | 使用已审计清单生成 COCO |
| `benchmarks/comparison/evaluation/evaluate.py` | 公共预测入口及高精度旧缓存适配 |
| `benchmarks/comparison/evaluation/native_metrics.py`、`native_provenance.json` | 隔离并原样复用母版指标函数及出处 |
| `benchmarks/comparison/evaluation/README.md` | 公共预测 schema、真实匹配/AP/P/R 规则和数值边界 |
| `benchmarks/comparison/test_preparation.py` | 五项有限正确性测试 |
| `scripts/autodl_prepare_comparison.sh` | 服务器 CPU 准备、独立 tmux、日志及真实退出码 |
| `docs/comparison/SERVER_PREPARE_COMMANDS.md` | 可复制的服务器步骤，全部不含训练 |
| `docs/comparison/evidence/` | 原始 C2/母版 args、紧凑观察快照、增强表和数值诊断；小配置用目录内 `.gitattributes` 保持原始字节 |

本地完整产物在工作树内 `outputs/comparison_prepare/local_final_20261002/`，资产原始小文件与提取的压缩预测缓存位于同级 `local_assets_20261002/`。最终入口显式复用同次工作中完成的全量审计，未反复读取 43 GB 图像。初次开发验证输出保留用于追溯，不覆盖。Git 只收录小证据，不收录数据、权重、完整预测或大结果包。

## 数据事实与论文限制

从真实母版 `training/args.yaml` 的 `data` 定位到归档 `metadata/launch/data_config.yaml`，原始 SHA256 为 `ea2a922586a03526e1b4c8e64c02b2c71dce1a458f1c140f038afa744d68c6aa`。服务器根 `/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det` 显式映射到本地 `D:/MyProjects/Crack_RTDETR/datasets/crack_det`，未改写原 YAML。

| split | 实测图像 | 实测框 | COCO 图像/框 | 历史路径与标签指纹 |
|---|---:|---:|---:|---|
| train | 6048 | 45573 | 6048 / 45573 | 两者完全相同 |
| val | 1728 | 12840 | 1728 / 12840 | 两者完全相同 |
| test | 864 | 6663 | 864 / 6663 | 两者完全相同 |

共 8640 图、65076 框。每个图像内容读取一次，同时 SHA256 和完整解码；原始图像内容约 42.87 GB（十进制）。损坏/无法解码、非法/缺失/空标签、孤立标签、越界警告和精确字节重复均为 0。这里空标签计数为 0；工具及测试仍保留合法负样本，不把空标签判错。

已恢复历史 `tools/c19_lif_v1_data.py` 的算法：POSIX 根相对路径按稳定顺序，以 LF 连接且末尾无 LF，再做 SHA256；标签清单为 `相对标签路径:原始内容SHA256`，同样连接。三组路径/标签哈希与训练归档全部相同。新的完整 manifest 哈希单独命名。新的数据身份额外包含各图像内容哈希，不能拿它冒充旧 `split_paths_sha256`：

`dataset_identity_sha256 = 7a38970243f70e43a9b0a245ad5b7d311c9ad076274638123df9f91d88660c4c`。

原图族依据是实际 `split_manifest.csv` 的显式 `source_id/source_image`，不是从同名文件猜测。清单 SHA256：`0bdbdcb1cb13b82a624bf2d7c0c68408f36ff91d6e4c67f2bd514d38726c8f62`；8640 个唯一图名全部匹配到 1440 个原图族，无缺项或冲突。原清单中 val=864、test=1728，**2592 行 split 字段与实际目录相反**；本轮按实际 YAML/目录和已匹配历史指纹确定 split，只复用其唯一图名到原图的映射，保存矛盾，不修清单、不调换目录。

| 集合对 | 共享原图族 | 相关图像 |
|---|---:|---|
| train–val | 1064 | train 4097；val **1728/1728** |
| train–test | 657 | train 2401；test **864/864** |
| val–test | 444 | val 677；test 559 |

可核实例：`crack_00002` 的 `images/train/crack_000002_original.jpg` 与 `images/val/crack_000002_brightness_contrast_strong.jpg` 同属清单中原图 `datasets/crack_raw/images_renamed/crack_00002.jpg`；`crack_00003` 的 train original 与 test brightness_contrast_strong 同族。完整少量示例保存在 `observed_snapshot.json` 和本地 manifest。

历史生成脚本 `tools/07_make_augfirst_crack_det.py` 先生成增强记录，再对记录随机 shuffle/split，证实该机制；其当前提交所列增强种类与本地六成员族清单不完全一致，因此不能声称仅凭这个脚本已恢复当年生成过程的每个细节。README/旧数据说明写的“先划分再增强”也不能覆盖实际证据。本次当前版本身份由历史路径/标签指纹和实际映射确认。

**结论：旧结果只适用于旧固定划分内部比较，不能证明未见原图泛化。** 字节不重复不等于原图独立。若论文需要独立泛化，应先按原图族划分，仅增强训练集，之后重跑 C2、C2+CBR、C2+LIF、母版和全部拟比较模型；新旧协议分开报告。本轮没有生成新划分或重训。

## 真实母版配置与增强

四组真实 args 均完整保留 109 字段及类型。母版训练文件与 `metadata/launch/actual_train_args.yaml` 原始 SHA256 均为 `31029c8d3bcf0e2dc3cb7107fa7db316f37cd954eba21bf4fc854e2070fc476b`；C2 为 `ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`，均吻合交接参考。原始字节副本为 `evidence/mother_args.yaml`、`evidence/c2_args.yaml`，不是从 test 的 `actual_settings` 重建。

实际配方：200 epochs、patience50、640、batch16、workers8、seed42、AdamW、lr0=0.0005、lrf=0.01、weight_decay=0.0001、cos_lr、warmup5、AMP、deterministic。完整字段以原始 YAML 为准；对比模型优化器允许合理适配，消融才要求完整母版配方不变。

母版 8.4.21 的 dataset、augmentation、trainer、validator 和 metrics 共八个关键源码文件已与成功 run 的 `metadata/source` 做 LF 规范化哈希比较，全部相同。本地既有 `D:/miniconda3/envs/rtdetr/python.exe`（Python3.9.25、PyTorch2.7.1+cu118、NumPy2.0.2）完成 CPU 增强对象构建及关闭前后核验，实际导入本公共工作树源码；没有构造检测模型或执行图像增强抽样。服务器历史环境是 Python3.10.13/PyTorch2.1.2+cu121/NumPy1.26.4，仍需服务器准备入口核查当前状态。

| 增强项 | 实际配置 | 生效与意义 | 关闭 |
|---|---|---|---|
| Mosaic | 0.8 | 四图拼接，p=0.8，使用数据集 buffer 取伙伴 | 达到 epoch index190（第191轮）时 p=0 |
| MixUp | 0.05 | p=0.05，混合比例 Beta(32,32)，第二图走同一预变换 | 同上 |
| 旋转/平移/缩放 | 5° / 0.1 / 0.4 | 角度 U[-5,5]，中心 U[0.4,0.6]，尺度 U[0.6,1.4] | 不随 Mosaic 关闭 |
| shear/perspective | 1.5 / 0.0002 | tan(均匀角度)剪切；均匀透视系数；T×S×R×P×C | 不关闭 |
| HSV | 0.015 / 0.5 / 0.35 | hue 为加法偏移；S/V 为乘法 LUT，不能照搬旧 YOLOv5 hue 语义 | 不关闭 |
| 上下/左右翻转 | 0.2 / 0.5 | 对应 Bernoulli 概率 | 不关闭 |
| CutMix / CopyPaste | 0 / 0 | 未启用；box-only 检测没有分割 CopyPaste | close_mosaic 也会清零 |
| bgr | 0 | RTDETRDataset 未传 hyp.bgr，Format 默认 0，输出 RGB | 无 |
| erasing / auto_augment | 0 / null | 分类路径字段，检测增强链未调用 | 无 |
| multi_scale | 0.0 | 训练输入尺寸抖动未启用，与几何 scale 不同 | 无 |
| Albumentations 隐式项 | 无 args 对应项 | 源码可包含 Blur/MedianBlur/ToGray/CLAHE 各0.01；历史 pip freeze 无此包，本地运行对象 active_transform=false | 历史 import 状态未直接重放，不擅自启用 |

检测框经变换后裁剪，保留宽高>2 px、相对缩放后原框面积比>0.10、长宽比<100；Mosaic 另裁剪并移除零面积框。严格区分这个训练变换行为与标注转换：转换器不裁剪、修复或丢弃非法框。

逐项源码行、配置值、作用、关闭时机、状态和运行对象表在 `configs/mother_augmentation.yaml` 与 `evidence/augmentation_observed.json`。历史 Albumentations 是否曾以非标准方式导入仍未直接验证；本地当前状态不代替服务器历史状态。八模型增强映射均待各自接入时核实，未宣称已完成。

## 四组资产与历史指标

下表全部读自实际结果文件；单位为百分数。它们是历史独立 test 的记录，不是本轮新推理。

| 模型 | P | R | AP50 | AP75 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| C2 | 83.396001 | 80.804442 | 85.816299 | 45.836040 | 46.963623 |
| C2 + CBR | 84.903384 | 83.731191 | 88.518636 | 51.999526 | 50.389242 |
| C2 + LIF | 84.807456 | 82.485367 | 88.034758 | 52.840956 | 50.818825 |
| C2 + CBR + LIF | 86.023937 | 83.535945 | 89.199683 | 54.002362 | 52.200902 |

定位映射和每个原始小文件哈希均在 `asset_index.json`，Git 内紧凑快照保留来源、权重哈希和指标身份：

| 模型 | 本地证据根 | best 本体 | 独立 val / 高精度缓存 | CSV 最大训练 val 的候选 epoch |
|---|---|---|---|---:|
| C2 | `D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_complete_retest_20260904.tar.gz` | best/last 归档成员已流式哈希 | 独立 val 未定位；test 为舍入 JSON | 200 |
| CBR | `D:/rtdetr跑结果/c19 CBR涨2/c19_cbr_test_small_20260907_132852_339107473` | 本地小包缺 best/last；有历史路径/哈希记录 | 独立 val 未定位；test 为舍入 JSON | 186 |
| LIF | `D:/rtdetr跑结果/LIF-Down涨/lif_down_complete_20260910_221150_853145.tar.gz` | best/last 归档成员已流式哈希 | val/test 高精度缓存均完整且哈希吻合 | 189 |
| 母版 | `D:/rtdetr跑结果/C19＋LIF：原 CBR＋LIF-Down v1` | best/last 实际文件已哈希 | val/test 高精度缓存均完整且哈希吻合 | 200 |

四组代码提交均有原始指标 runtime/source_commit 或 CBR `source/git.json` 支持，与登记表锚点一致。母版 best=`24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`；C2 best=`f82c94cfc632339420cf886fc64de4f89981389523a182ad5a699d6d5aed08f3`；LIF best=`5017457bfe114d29a9dcdbc530f72c0390429b5775cb0668725d40ff6e87cf5b`；CBR **仅归档声明** best=`1a5a850c5cb1dadaed34ebff86b20c3f31312f44ed7839149864b3befab19f3d`，未谎称已核实本体。

母版独立 val mAP50–95=52.45427221919051%，LIF=51.85631920696523%。选 best 的源码 fitness 权重为 `[0,0,0,1]`，用训练 val mAP50–95，达到当前 best 时保存；CSV 有舍入，表中 epoch 是最大值候选，未反序列化 checkpoint 证明精确 epoch。原训练 val 沿用旧原生排序/mask 路径，独立评测采用修正版；保留这一区别与旧 best，不用 test 重选。训练与独立 FP32 val 数字略有不同不能静默替换。

公共 ImageNet 骨干初始化实际本地文件已哈希：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。母版受控初始化的归档输出哈希为 `9d583611e4558b873ef07c5a45e5d37c9b2914c7058e27c098d7eb695aad80b4`；两者和母版 best 明确分开。比较模型后续使用各自官方预训练权重并单列预训练数据差异。

母版融合参数19,944,965/GFLOPs58.7、C2融合参数19,877,716/GFLOPs58.3099248（nc=1、640）仅作交接 historical_reference，本轮未重新测量。归档中的母版 unfused 20,149,765 不混入融合口径。资产索引不等于把权重备份到本准备目录或 Git。

## 公共评测与有限验证

COCO 全量转换保持原 split、全部图像和浮点框，三组图像/框数逐项一致，类映射 YOLO0→COCO1，TorchVision0保留为背景。目录和图像列表输入均有实现；既有数据本次是目录。公共预测为每图完整 JSONL（包括空预测图）、原图像素 xyxy、score、category_id，绑定模型代码、权重、数据与评测配置身份。原生 resize/letterbox 逆变换由后续模型适配器负责。

公共 evaluator 复用母版 AP 和匹配函数，不导入训练器、不用 COCOeval 冒充等价。按排序后的 score 严格过滤 >0.001，最多300，IoU十点，单列 AP50/AP75。P/R 为各模型自身平滑最大 F1 工作点。历史 `iou=0.7` 在 RT-DETR 预测路径不生效；未添加 NMS/TTA。详细数值规则见 evaluation README。

五项确定性测试通过：已知框与合法负样本转换、非法/缺失标签的明确处理、排序与过滤对齐、贪心匹配/空预测/身份校验、原样复用函数 AST 检查。三个 CLI 的 help 均正常；Bash 语法检查通过；缺服务器数据正常生成 UNAVAILABLE 状态；重复输出拒绝并保留旧证据。包装器用退出0/7的假 Python 验证真实退出码和日志，重复日志路径拒绝；真实 tmux 行为需服务器现场确认。

已有高精度缓存的 CPU 复算（用独立 COCO GT，**没有模型推理**）：

| 缓存 | 本轮 mAP50–95 (%) | 相对归档差值（百分点） |
|---|---:|---:|
| 母版 test | 52.20090191444802 | 0 |
| 母版 val | 52.45428513356194 | +0.00001291437143 |
| LIF test | 50.81693555208446 | −0.00188903591268 |
| LIF val | 51.85728462729233 | +0.00096542032710 |

母版 test 五项指标完全一致；其余不标为精确复现。最大差异的 LIF test 已进一步定位：独立 YOLO→COCO 与历史 FP32 GT 导出最多相差0.000732421875原图像素，`images/test/crack_001071_brightness_contrast_strong.jpg` 有1个匹配位在 IoU=0.70 边界发生变化；用导出 GT 的数值诊断得到0.5081882459967275，与历史0.5081882458799715只差约1.17e-10。正式公共结果仍使用独立 GT，未以嵌入 GT 替换它；详细诊断在 `evidence/numeric_audit_lif_test.json`。其余极小差异未逐框穷举，保留原始记录。

C2/CBR 旧框和分数有舍入，不能保证精确重算；需要后续统一复评时另行执行，不自动跑 GPU val/test，不覆盖旧指标。

## 服务器补证与下一项

完整命令见 `SERVER_PREPARE_COMMANDS.md`：fetch公共分支→固定SHA→独立工作树→复用既有 Python→新输出目录→独立 `comparison-prepare` tmux。现有同名会话拒绝覆盖。日志管道保存 Python 的 `PIPESTATUS[0]`，完成界面、日志和退出码分开保留。进程成功与数据/论文结论分别展示。

服务器仍需核实当前数据/权重与本地归档是否一致、实际环境导入位置、训练增强对象及 Albumentations 状态，补齐 CBR 权重本体及 C2/CBR 独立 val 资产定位。没有访问 AutoDL，不能把 Windows 验证写成服务器已通过。

下一步接入原始 anchor-based YOLOv5m 前，需要先决定论文继续报告旧固定划分比较，还是另行授权原图族独立协议与重跑；然后固定官方 commit/config/预训练权重 SHA256、隔离环境、逐项校准 HSV（特别是 hue 语义）、几何/Mosaic/MixUp及关闭调度，登记无法等价项，确定模型自身优化器配方，实现原图坐标高精度导出并用公共 evaluator 核验。当前仅登记八个模型，未接入或训练 YOLOv5m。
