# RDL v1：可达范围分解损失

这是**研究候选**，工程接线通过不等于完整训练有效。唯一母版是 `a0459d6a652cb702699087c88fa39a3e4c4087ec`；分支为 `exp-rtdetr-r18-lite-rdl-v1`。保留原 CBR＋原 LIF-Down，只增加零参数的训练损失。正式训练、独立 val/test 尚未执行。

## 公式、坐标与梯度

最终 decoder 的常规 query 按其最终精修框做原 Hungarian 匹配，得到整个 batch 的 M 个正样本。`before` 是同次 forward 的 CBR 输入框 b0，`tanh_offsets` 是同次计算得到的 v，均保留梯度；GT 来自原 matcher 的 batch 展平索引。边顺序是 **L、R、T、B**，左右向右为正，上下向下为正。

```text
x0 = [cx-w/2, cx+w/2, cy-h/2, cy+h/2]; gs 同样转换 GT
a = sg(0.10 * b0[..., [2,2,3,3]])
ebar = sg(gs-x0)
vstar = sg(clip(ebar/a, -1, 1))
eta = sg(min(1, a/abs(ebar))); ebar=0 时 eta=1
p = [1/W,1/W,1/H,1/H]
S = sg(max(a,p))
o = relu(abs(gs-x0)-a)/S
phi(z) = 0.5*z² (|z|<=1), 否则 |z|-0.5
L_out = sum(phi(o))/(4*max(M,1))
L_in = sum(eta*phi(v-vstar))/(4*max(M,1))
L = L0 + 0.05 * clip((e-5)/15,0,1) * (L_out+L_in)
```

e 是当前 epoch 开始前已完成的 epoch 数：e=0..5 权重为0；e=6 为0.05/15；e=20 为0.05。原 LR warmup 不变。`L_out` 保留 b0 边坐标梯度；`L_in` 保留真实 v 的梯度；a/ebar/vstar/eta/S 和 GT 是固定目标。原 CBR 主路径的宽高缩放、query/P3 条件和梯度都未改变。

新增计算局部禁用 autocast，几何、Huber 和归约均为 FP32。只在 vstar 的分母使用 `a.clamp_min(1e-6)`；eta 的非零误差分母也有1e-6保护，零误差显式为1。a本身和S使用未改写的范围。小于1e-6是数值近似，日志分别统计触发次数。非法宽高、非有限数会报错；不使用 nan_to_num 或输出框 clamp。空匹配返回关联本次 b0/v 图的 FP32 零。H/W 直接来自预处理后图像张量。

## 实际接线及母版核对

- 母版原模型类型仍是 `RTDETRDetectionModel`；先复用 `init_c19_lif_v1` 的严格结构/初始化审计，再添加普通 Python 配置属性。没有改 `__class__`、全局 monkey patch、缓存 hook 或第二次 forward。
- `tasks.py` 仅增加显式 `return_cbr_details` 和按配置启用 criterion 的入口，完整复用原 traversal/save/routing。普通预测、验证及导出不要求 details/GT。
- `loss.py` 增加可选 **final-only** indices 参数。新 criterion 计算一次最终精修框匹配，再将同一 indices 传回原最终主损失；encoder/auxiliary 继续各自的原匹配，DN继续原已知索引。
- dec预测在dim=2拆DN；details在dim=1按同一 `dn_num_split` 拆DN。原 encoder 仍拼在最前，不冒充第四层decoder。仅在所有原主项/aux/DN项完成后加入一个 `loss_rdl`。
- 原 L0：VFL＋5×L1＋2×GIoU，含原encoder/aux/DN。空GT沿用原Focal分支。matcher class/bbox/GIoU=2/5/2；`no_object=0.1` 在原VFL计算中不参与，未重复相乘。原matcher存在非有限cost置零逻辑；RDL激活路径在其前检查相关预测，未改写原matcher。
- YAML核实：LIF节点20，head节点26，输入 `[19,22,25]`，3层decoder、300常规query。原CBR `rho=.10`、36采样点、原条件分支与零初始化保留。`cbr.py`、`lif_down.py`、`transformer.py`、head与YAML均未修改。
- `RDLTrainer.get_model` 执行母版真实重建；原nc80→nc1严格适配9个类别keys，其余逐值相同。附加损失新增参数/缓冲区为0；nc1未融合参数20,149,765。原LIF融合保护保留。
- `on_train_epoch_start` 将实际 `trainer.epoch` 传给模型和已创建criterion；延迟criterion在首次loss调用读取该e。EMA保存RDL版本/配置/e/code；resume核验本实验last、完整配方、代码和epoch，再由原Trainer恢复优化器/scaler/EMA，避免重新ramp。
- 验证loss使用L0，主显示项仍为giou/cls/l1。独立轻量 `rdl_epochs.jsonl` 记录新增原始/加权损失和计数，数值不进入自动求和字典。训练用原val fitness保存best，不改成AP75选模。

## 初始化、数据、配方

统一源 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，epoch=-1、nc80未训练。正式初值独立存储为 `weights/rdl_v1_controlled_init.pt`。已训练母版仅供现象诊断，绝不作为正式初值。

`prepare` 读取全部109个母版C2 args，并与服务器C2实际args逐字段比较；输出逐字段diff。仅改模型、输出身份和已核验数据绝对路径。RDL常数单独写 `loss_config.json`，不塞入Ultralytics args/state_dict。

数据参考来自已归档成功母版 `metadata/launch/dataset_inventory.json`，见 `dataset_reference.json`。本地实测train/val/test为6048/1728/864张，GT45573/12840/6663，全部split路径和标签清单指纹一致。prepare/preflight/start检查这些身份。已有同源增强图跨划分问题沿用并披露，本轮未重新划分、增强或创建holdout。

## 分阶段服务器操作

服务器主项目固定 `/root/autodl-tmp/projects/Crack_RTDETR`；新工作树固定 `/root/autodl-tmp/projects/Crack_RTDETR-rdl_v1`。环境路径 `/root/miniconda3/envs/rtdetr/bin/python` 已从成功母版 `metadata/launch/plan.json` 核验；shell每次显式激活并检查，Python服务器操作要求3.10.13/torch2.1.2+cu121，不自动升级。

交付最终回复给出已填实的完整commit和sync命令。`sync_rdl_v1.sh 完整SHA` 核验origin和远端branch tip后创建独立detached worktree；主项目和其他实验HEAD不变。已有目录必须匹配同一个SHA，否则停止，避免覆盖旧执行证据。

以下各段均独立执行；脚本会定位工作树并加载环境：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh prepare
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh preflight
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh diagnose
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh start
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh resume
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh val
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh test
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rdl_v1/tools/rdl_v1.sh pack
```

prepare、preflight、diagnose互不启动训练。start才派发固定200epoch配方至tmux `rdl-v1-training`；仍保留原patience=50和early-stopping语义。run名 `rdl_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。resume是中断后可选操作，正常完成的/已strip的last不能续训。正式恢复须使用本入口，以便真实Trainer继续启用RDL。

与其他实验可共用GPU；只检查本RDL的session/run，并用本run的文件锁防重入。OOM、非有限预测/总loss或FP32梯度直接退出，保留错误；不自动减batch、改nbs、关AMP、杀其他进程或回滚重试。AMP缩放梯度溢出明确记录epoch/scale和计数，由**原生GradScaler**按原步骤跳过更新并调整scale；这没有修改正式scaler初值或优化器步骤。容量检查沿用母版有限检查的 `GradScaler(init_scale=128)`，是隔离模型一次B16/640 native AMP前后向，**不做optimizer update**，故不代表真实AdamW状态分配后的完整峰值；start仍可能受同时运行进程影响。所有检查不用于公平推理耗时比较。

独立val/test复用 `corrected_sorted_conf_mask_v1`，imgsz640、B16、workers0、device0、FP32、conf.001、IoU.7、max_det300、augment/rect=False、seed42、原额外绘图设置。val只评本run best并记录权重SHA、代码SHA、指标和选择理由；test只读冻结选择，禁止扫描权重。报告十阈值AP、AP50/AP75、mAP50–95、P/R及原最大F1工作点conf，不把训练CSV峰值当独立val。

LIGHT包包含变更源码、身份、配方、检查、诊断及已有结果。保留原始数值；每个log/jsonl最多尾部32KiB，manifest标注字节范围及排除项；结构化失败报告保留完整错误。best/last和大体积预测留在原目录，不写Git，LIGHT要求小于8MiB。

## 科学限制

逐边投影不是最大化IoU；±1目标可能促使原tanh饱和，实验不改变rho/激活。全模型联合训练，新损失仍可通过共享参数影响其他模块。超范围现象只说明有可研究对象，不证明AP收益；是否有效最终依赖完整训练与综合AP。历史成功组合独立val约52.4543%、test约52.2009%仅为同协议参照。GRA/DCC正式结果未确认，不能据此声称失败。首轮不自动添加out/in消融、第二损失、test调参或额外实验矩阵。单seed小幅增益只能作为当前协议下的初步结果。
