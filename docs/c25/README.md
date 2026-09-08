# C25 组合实验交付

C25＝C2＋原版 CSCEFv51（C17）＋原版 SCCAAIFI（C24）。目标是从统一初始化重新联合训练，检验组合效果；本轮没有正式训练或完整 val/test，也没有组合收益结论。

交付分支：`codex/c25-cscef-scca`，从 C24 正式训练源码 `f6e9dfda765046ae7691302cf5ec89d3f76cec5d` 创建。C17 来源 `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`。本地独立工作树 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/c25`。检查了祖先目录、主仓库和工作树，没有适用 AGENTS.md；主仓库未跟踪 bundle/图稿、其他分支/worktree、历史记录全部保留。

## 固定结构与实测参数

复用 `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cscef-scca.yaml`，与 C17 原 YAML 的唯一文本差异是第9层 `AIFI → SCCAAIFI`，保留 `[1024, 8]`，没有再插层。

| 层 | 模块 | 输入 |
|---|---|---|
| 9 | SCCAAIFI | -1 |
| 18 | CSCEFv51 | [17,16] |
| 19 | Concat | [16,18] |
| 27 | 原 RTDETRDecoder | [20,23,26] |

原 Decoder 为3层、300个常规 query，原 DN（num_denoising=100）、匹配、损失不变。未启用 ACR、SALA、CBR；没有改变 Backbone、模块计算或超参数。

| 模型 | 本轮重新构建实测：nc=1、未融合参数 |
|---|---:|
| C2 | 20,082,772 |
| C17 | 20,109,684 |
| C24 | 20,148,312 |
| C25 | 20,175,224 |

C25 比 C2 增加92,452参数，其中 CSCEF 26,912、SCCA 65,540。独立评估的 AutoBackend 会融合模型，控制台融合后的参数量不能替代上表。

`cscef_v51.py`、依赖 `cscef_v5.py` 与 C17 来源逐字节（统一LF）一致，Scharr buffer、原初始化及梯度行为保留。`scca_aifi.py` 与 C24 来源一致，原空间注意力、空间条件Q、输入K/V、潜在通道交互、中心化归一化、有界温度、FP32核心与零输出初始化原样保留。

## 复用与必要补充

- 复用既有 C25 YAML、533项公共权重映射、干净初始化、C2全部109字段核对、直接训练派发、任务/结果目录锁、OOM退出和轻量 `pack`。
- `train_scca.py` 同时检查 SCCA 与 CSCEF 各5个可训练参数张量进入原生 AdamW，且各出现一次；记录逐项nc重建结果。正式启动核对既有 Python3.10 / PyTorch2.1.2+cu121 / RTX4090，不升级环境。实际训练参数同时核对值与类型。
- `scca_results.py` 保持 `corrected_sorted_conf_mask_v1` 指标口径，新增同次推理的流式预测/GT、完整AP数组、实际参数、日志及退出状态。test要求已完成且权重SHA、数据SHA、评估设置和源码commit一致的val。
- `scca_export.py` 新增完整包，包含整个训练目录（含best/last和已有图像）、val/test评估目录、launch证据、源码/补丁、数据配置、环境清单和哈希；流式写包并逐文件读回校验，不受20 MiB限制。缺失证据列在 `metadata/package.json` 及包外 verification.json；包完整性通过不等于训练/评估成功。
- `check_c25_delivery.py` 是可选的本地交付回归，最多接受每split 4张图；临时初始化和nc1 checkpoint结束后删除。它不属于 `start-direct` 前置条件。

## 初始化与完整C2配方

唯一源是主仓库 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256：

```text
fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
```

组合的是结构，不读取C17/C24的best、不拼接训练权重。保留CSCEF/SCCA原始新参数初始化、533个源状态的C17层号偏移映射；保存重载逐键值一致。原生 `RTDETRTrainer.get_model` 的nc80→1重建精确加载536/545，恰好9个分类项重初始化：

```text
model.27.denoising_class_embed.weight
model.27.enc_score_head.weight / bias
model.27.dec_score_head.0.weight / bias
model.27.dec_score_head.1.weight / bias
model.27.dec_score_head.2.weight / bias
```

本地用 [C2归档](../scca/c2_args.yaml) 核对；服务器实际读取主仓库 `runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`，109字段及类型必须完全等于归档。仅model/name/save_dir改变；不改服务器绝对路径。

200e / patience50 / 640 / batch16 / workers8 / device字符串'0' / seed42 / deterministic=true / AMP=true / cache=false / resume=false；AdamW lr0=.0005、lrf=.01、momentum=.937、weight_decay=.0001、cos_lr=true、nbs64；warmup_epochs5、warmup_momentum.8、warmup_bias_lr.1。

在线增强仍为HSV(.015,.5,.35)、degrees5、translate.1、scale.4、shear1.5、perspective.0002、flipud.2、fliplr.5、mosaic.8、mixup.05、close_mosaic10、cutmix0、copy_paste0、auto_augment=null、erasing0。保留augment=false，它不关闭这些训练在线增强。

数据始终使用 `/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`，没有重划分或重新生成增强数据。

## 独立评估与逐图格式

默认val/test：imgsz640、batch16、workers0、half=false、conf.001、iou.7、max_det300、augment=false、rect=false。原训练验证器继续选择best；用户固定模型、配置与checkpoint后手动test，不根据test调参。C2/C17/C24同入口重评不是启动C25的前置条件。

每个评估目录包含 `metrics.json`（受旧mask影响图数、AP50至AP95每.05的 `ap_by_class` 数组、`ap_iou_thresholds`、源码commit、权重/数据SHA、请求及实际设置）、`evaluation.log`、`exit_code.json`、`plots/`、`predictions_gt.jsonl.gz` 和其manifest。

JSONL.GZ每图一行，含相对dataset_root的image路径、原图height/width、0起始类别、完整预测score/bbox和GT。保存全部300常规query，按score降序；`used_for_metrics`标识score>.001的子集，指标仍使用修正后的阈值mask。坐标为连续原图像素 `xyxy`，不取整、不裁剪、不加1；RT-DETR拉伸输入按宽高独立反缩放。推理FP32数值不做小数舍入，坐标转换使用float64。空GT/空预测图也保留；路径多重集与validator的dataset.im_files完全核对才写完成manifest。不二次完整推理，不缓存模型激活；仅当前batch的小检测结果转CPU并逐图写出。

| 实验 | 历史结果 | 本次同入口完整重评 |
|---|---|---|
| C2 | 保留原历史结果，本交付未导入数值 | NOT_RUN |
| C17 | 保留原历史结果，本交付未导入数值 | NOT_RUN |
| C24 | 用户说明已完成正式训练；本交付未导入服务器指标 | NOT_RUN |
| C25 | 尚未正式训练 | NOT_RUN |

历史指标与同入口重评必须分别标注，不能混算组合增益。本地4图结果只证明工具链可运行，不作为实验成绩。

## 完整包

`bash tools/autodl_scca.sh pack-complete c25` 只复制现有证据，不启动训练、val或test。输出到主仓库 `downloads/scca/c25_complete_时间戳.tar.gz`，附 `.sha256`、`.inventory.json` 和 `.verification.json`；包内 `MANIFEST.json` 覆盖除它自身以外每个文件，清单自身由压缩包SHA保护。不覆盖旧包，读取过程中发现文件变化则失败且不出具成功校验。建议训练及评估退出后打包。

完整training目录包括已有权重、args、CSV、曲线、混淆矩阵、labels及train/val batch图；evaluation/val与test完整收集已有评估图与预测/GT；metadata保留初始化映射、参数差异、加载/优化器、训练/评估日志和退出状态、源码快照/补丁、环境清单、数据配置及权重哈希。不会复制整个数据集或环境。旧结果缺少逐图导出时如实列缺失，不通过打包补跑推理。

服务器分步命令见 [AUTODL.md](AUTODL.md)，本轮与复用验证边界见 [VALIDATION.md](VALIDATION.md)。本轮仅交付C25，C19＋C24不在范围内。
