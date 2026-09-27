# ARG v1：原 CBR + 原 LIF-Down + Axis Repair Gain Loss

固定母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec`；实验分支 `exp-rtdetr-r18-lite-arg-v1`。
这是一项尚未正式训练的损失候选。没有涨点保证、首创断言或正式 test 结果。

## 实现与不变项

模型 YAML 原样复用 `ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml`。
LIF 第20层，CBR decoder 第26层读 `[19,22,25]`，3层 decoder、300常规query、36点CBR采样、rho=0.10。
`lif_down.py` 与 `cbr.py` 无 Git 差异；它们的 LF SHA256 必须分别为：

```
26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787
```

新代码只在 `models/rtdetr/arg_loss.py`、`arg_model.py`、`arg_val.py` 和专用操作/检查入口。
`ARGTrainer.get_model` 在同一 RNG 位置调用原生构造与加载，逐状态对照隔离母版，再把实际重建对象专门化为可导入的 `ARGDetectionModel`。
这一 Python 类包装没有新增参数、buffer 或前向计算；nc80→nc1 仍由母版原生加载完成，未另行初始化分类头。
nc1 未融合参数量 20,149,765，552 state keys，345可训练张量；优化器组为130 weight、81 norm、134 bias。

模型的原 predict、CBR 条件梯度、LIF 计算、fusion 排除规则均保持。
ARG 不更改训练期间原生 RTDETRValidator、fitness 或 best 选择。母版 fitness 权重为 `[0,0,0,1]`，即 mAP50–95。
训练终端依然只显示 giou_loss/cls_loss/l1_loss，实际优化 L0 还包括所有 encoder/decoder auxiliary 与 DN 项。

唯一共享源码改动是 `nn/autobackend.py` 的 warmup 输入从 `torch.empty` 改为同 shape/dtype/device 的 `torch.zeros`。
没有改变其他 empty、模型计算或融合规则。隔离母版对照使用相同修复；这是评估可靠性修复，不属于 ARG 的损失贡献。

## 固定公式

`lambda_arg=0.20`，`epsilon_w=0.05`，`eps_num=1e-7`。e 为正常零基 `trainer.epoch`，包括 resume 后恢复的 epoch：

```
r(e) = clip((e - 5) / 15, 0, 1)
```

仅取 CBR 后最终常规匹配正 query 以及当前实际增强 batch 的 GT，归一化 cxcywh 局部关闭 autocast，FP32 可导转成 `b=(l,t,r,d)`、`g=(l*,t*,r*,d*)`。

```
q  = IoU(b.detach(), g)
qx = IoU((g_l,b_t,g_r,b_d), g)    # 修复横轴，残余误差来自纵轴
qy = IoU((b_l,g_t,b_r,g_d), g)    # 修复纵轴，残余误差来自横轴
dx = max(qx-q, 0); dy = max(qy-q, 0)
wx = 2*(0.05+dx)/(0.10+dx+dy)
wy = 2*(0.05+dy)/(0.10+dx+dy)
```

q/qx/qy/dx/dy/wx/wy 全停止梯度；二维 IoU 使用显式虚拟框，union.clamp_min(1e-7)。
不使用在极小面积保护下可能不一致的1D快捷替代。正常尺度有 wx+wy≈2、0<wx,wy<2。

每轴区间 `[u,v]` 和 `[u*,v*]`：

```
I = max(min(v,v*)-max(u,u*), 0)
U = (v-u)+(v*-u*)-I
C = max(v,v*)-min(u,u*)
ell = 1-I/U.clamp_min(1e-7)+(C-U)/C.clamp_min(1e-7)
raw = sum(0.5*(wx*ell_x+wy*ell_y))/max(M,1)
loss_arg = 0.20*r(e)*raw
total = L0 + loss_arg
```

预测框的区间梯度保留，GT 无梯度。没有裁剪、排序、扩大框或长宽比分组/门控/Huber/自动调参。
e≤5 直接调用母版损失，不计算新增几何、零梯度或额外匹配。M=0 为有限零，母版空GT分类监督保留。
非法或非有限输入抛出带匹配行、batch/query/GT索引、图片名及中间量的异常；严格 JSON 用 null 与 nonfinite 标志保存非有限值。

启用时 criterion 显式计算一次最终匹配，同时交给原最终三项和 ARG；aux 仍分别匹配。
DN 使用原 `DETRLoss.forward` 与原 DN 匹配；无DN补零先完成，最后才加入唯一 `loss_arg`。
因此 `_get_loss_aux(postfix="")` 不会触发 ARG，也不会生成 `loss_arg_dn`。
新增标量在原模型 sum 与 Trainer DDP 缩放之前只加一次。

## 配方、初始化与身份

完整权威参数为随仓库交付的 `docs/c19_lif_v1/c2_args.yaml`，不是摘要重建。
prepare 若发现服务器原 C2 args，还会逐字段核对值和类型。原109字段只允许 model/data/project/name/save_dir 的身份/路径变动。
其余包括200 epochs、patience50、B16/640、AdamW、lr0=.0005、lrf=.01、nbs64、seed42、workers8、AMP与全部增强均继承原值。
ARG 参数独立记录，不把通用 box/cls/dfl 参数当作 RT-DETR 的实际 loss gain。L0 gain仍为class/bbox/giou=1/5/2，matcher=2/5/2。

公共初始化来源 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt` 的SHA256必须为：

```
fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
```

prepare 调用未改的 `init_c19_lif_v1.initialize` 组合初始化，输出 `outputs/arg_v1/arg_v1_init.pt`。
从未使用母版 best 或任何其他实验训练权重。初始化存在时核对摘要并复用，拒绝不明文件。
数据仍是 crack_det 原划分：train6048/45573GT、val1728/12840GT、test864/6663GT。
prepare 清点图像/标签、尺寸与逐文件 SHA256；后续绑定重新读取实际内容，数据较大时需等待哈希。

源身份包括完整 Git SHA、LF源码清单、原模块原始字节和LF摘要、解释器、torch/CUDA/GPU、ultralytics与criterion实际导入路径。
不升级依赖。训练期间 `actual_setup.json` 记录实际args、scaler、优化器组和重建参数证明。

## 操作与安全恢复

服务器固定：

| 项 | 路径/名称 |
|---|---|
| 主仓库 | `/root/autodl-tmp/projects/Crack_RTDETR` |
| ARG worktree | `/root/autodl-tmp/projects/Crack_RTDETR-arg_v1` |
| Python | `/root/miniconda3/envs/rtdetr/bin/python` |
| tmux | `arg-v1-training` |
| run | 主仓库的 `runs/c_series/arg_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug` |
| 工程证据 | ARG worktree 的 `outputs/arg_v1/` |

入口 `tools/arg_v1.py --help`，Linux 使用 `bash tools/arg_v1.sh COMMAND`，固定解释器与 PYTHONPATH，设置 `PYTHONUNBUFFERED=1`、`YOLO_AUTOINSTALL=false`。
同步入口 `tools/sync_arg_v1.sh` 要求真实完整SHA，只创建或复用准确匹配的独立worktree，不改主工作区checkout；已有不一致分支/目录直接保留并报错。

依次执行，每项成功后再决定下一项：

1. **prepare**：核对源码/数据/公共源；建立并冻结初始化、完整args与recipe_diff。
2. **preflight**：独立有界检查。不会启动正式训练。**probe** 另提供CPU公式/路由检查，不代替GPU检查。
3. **status**：报告worker PID/命令、dispatch、tmux、epoch、结束状态、日志及best/last/final_eval。
4. **start**：核对干净源码与preflight的SHA/数据/初始化/完整配方绑定；只派发一次独立tmux。
5. **resume**：只接收本实验未正常结束且含有效epoch/optimizer/scaler/EMA的last；保留原训练SHA和schedule。不能从best/诊断权重/其他实验恢复。
6. **val**：训练完成后对原生fitness选中的best做完整独立FP32评估并建立val锁。
7. **test**：要求同一best SHA、数据、eval代码、协议与val锁；完成后不允许重复最终test。
8. **pack**：生成LIGHT包，失败或缺test也可以打包；缺项保留为缺项。

worker脚本使用pipefail和PIPESTATUS，分别记录真实Python与tee退出码。tmux存在不是训练成功。
互斥锁和进程检查只针对本实验，没有全局kill、reset、clean、强推或exist_ok覆盖训练。
训练原生合法早停保留。每epoch最多4个机制样本、2个更新梯度样本；`epochs.jsonl`聚合、`progress.json`存最新状态，不跨步保留计算图。
ARG raw/weighted、L0、定位项比例、匹配数、q/qx/qy、dx/dy、wx/wy、ell、保护触发及关键梯度与scale均单列记录。

## 短检边界和未决条件

服务器整个preflight由外层进程超时约束，默认900秒、最多16个真实B16/640训练micro-batch，无自动循环重试。
CPU数学/路由先行；GPU实际Trainer重建、原AMP检查、AdamW、e=20、原nbs64对应accumulate=4。
只在观察到原生optimizer被调用且CBR参数真实变化、关键梯度有限后确认更新。
保存原生checkpoint，再由新Python进程恢复epoch20、optimizer/EMA/scaler，并执行真实AutoBackend/fuse/零warmup→一批16张val。

原生初始scale先用最多8个micro-batch；若只是overflow导致没有更新，在剩余总边界内重置隔离模型/optimizer/EMA/RNG，以init_scale=128取更新证据。
正式scaler和正式训练e=0完全不变。若fallback成功，整体结果仍可显示PENDING，native_scale适应完成项保持PENDING；
仅当CPU、B16/640 AMP有效更新、ARG机制、真实新进程val与完整resume五项全部PASS，且明确存在scale128证据时，start_eligible才为true。
这表示训练路径已有有界证据，并不表示原生高scale适应全过程已完成。任何其他必需项PENDING/FAIL都会拒绝start。
OOM或真实计算异常保留并退出，不缩batch/分辨率，不靠调容差通过。

AMP离线资源须预先存在于主仓库：`yolo26n.pt`（或weights内同名文件）与 `ultralytics-main/ultralytics/assets/bus.jpg`。
只复制本机已有资源，记录SHA；原生AMP日志必须明确“checks passed”。缺资源、跳过检查或静默关闭AMP均拒绝继续。
这里的YOLO26仅是母版固定AMP自检资源，不是新增YOLO检测实验。

## final_eval失败恢复、独立评估与打包

native训练循环正常结束后，先记 `TRAINING_COMPLETED`，再调用原生final_eval的checkpoint收尾，用校正评估器FP32评估。
如warmup/val失败，异常继续向上抛出，保留exit1及 `FINAL_EVAL_FAILED`。
不要resume重训；确认无活动worker后执行 **val**。val核对结束证据、训练身份和best，重新执行warmup/真实完整val。
成功恢复另写 `evaluation_recovery.json`，不会覆盖原异常、exit.json、training SHA或训练fingerprint。
未来仅允许 `nn/autobackend.py`、`models/rtdetr/arg_val.py` 的评估修复以独立eval SHA评估；其他源码差异拒绝。文档变更不影响训练代码身份。

独立val/test固定：imgsz640、batch16、workers0、FP32、conf=.001、iou=.7、max_det300、augment=false、rect=false、seed42。
采用母版 `corrected_sorted_conf_mask_v1`，先按置信度排序，再以排序后的score生成mask，预测与GT对应关系保持。
报告实际settings、十IoU阈值AP、checkpoint与数据hash、训练及eval完整SHA。历史参照val 52.454272219190514%、test 52.20090191444802%只用于同协议比较，不能依test调参。

LIGHT包含源码/原训练源码快照、配置、数据清单与hash、init/最终权重身份、各检查和失败记录、训练日志/CSV/曲线/epoch统计、独立val/test与锁、manifest大小与SHA256。
默认不含数据集、best.pt/last.pt或巨大权重。`pack --include-predictions`在已有val锁时导出锁定best的val全300原始query及GT，包括query索引、图ID、尺寸与坐标空间；不会按0.25预筛。
test预测只在 `test --include-predictions` 明确请求时保存并独立命名。没有val锁时仍可生成故障包，不伪造预测。

## 验证范围与研究边界

本地检查入口：`check_arg_v1.py`（数学、匹配、梯度、r=0原生更新、CPU新进程resume）、`check_arg_v1_lifecycle.py`（真实单批FP32推理）、`check_arg_v1_ops.py`（控制门禁/失败打包）、`check_arg_v1_setup.py`（本地缩小尺寸的实际Trainer/AMP接线）。
缩小尺寸检查绝不作为正式B16容量PASS。服务器GPU短检、tmux运行、200e或合法早停、完整val/test和实际新增训练开销，在有真实证据前均为PENDING。

只有后续正式结果有效时，才考虑相同系数的等权1D GIoU对照。本轮不追加长训消融。
Bounded IoU、SCALoss、Focal-EIoU是相邻研究，不能据本实现声称首次提出轴向几何或已经证明涨点。
需求中残留的“RMD”操作字样按本实验ARG机制probe处理，没有引入另一损失方案。
