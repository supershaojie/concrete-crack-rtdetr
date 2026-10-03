# YOLOv5m COCO + b19 在线增强 pilot

新分支 `bench/yolov5m-coco-b19-pilot` 从实际父提交 `b6d38d6b2d72661d03322257ebc796849eb4d2f3` 派生，包含 scratch 已验证代码 `ea17605d538f28da5fa502ddaaf130f7a9ea8084` 及其后文档。核对远端后未发现更新的 scratch 修复。原 COCO 加载逻辑参考 `a4dcaa0b0b0c9356923c266ba351de67e033c540`。原分支、主项目未提交文件、RT-DETR 母版及 Tunnel_Disease_YOLO26 均保留。

本轮只交付实现、本机必要验证和普通推送，没有 SSH、服务器操作或正式训练。全部 smoke 输出属于合成数据，不能写入论文结果，也不能初始化正式 run。

## 来源与初始化

检测器为原 YOLOv5 v7.0，SHA `915bbf294bb74c859f0b41f1c23bc395014ea679`，原三尺度三锚框 Detect，depth=.67、width=.75。正式初始化使用原始官方 `yolov5m.pt`，42,806,829 bytes，SHA256 `61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56`。不修改官方文件；支持核验后复制既有官方资产。

实际 nc=1 参数量为 20,871,318。475/481 个兼容张量全部迁移，逐个检查实际模型值与来源一致；其余六个张量是三个 Detect 输出卷积的 weight/bias，80 类来源与 1 类目标的形状不兼容。它们保留原生初始化，兼容检测器层没有被人为排除。正式 train.py 在 load_state_dict 后、AMP/AutoAnchor/首次更新前写入 `pretrained_load.json` 和不可覆盖的 `initialization.json`。AutoAnchor 保留原生算法，仅看 train，并记录前后锚框。AMP 检查使用真实模型的私有副本，验证 FP32/FP16 输出并恢复 Python、NumPy、CPU/CUDA RNG，保护 BN、EMA 和 optimizer。

增强适配只取 [Ultralytics v8.4.0 固定 CutMix](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/ultralytics/data/augment.py)，SHA `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`。annotated tag 对象 `6f6158be448c73471c000cf41db5cd9169300ed9` 的 peeled commit 已核实。原码片段、bbox_ioa、Instances.clip、完整 AGPL-3.0 许可和来源文件 SHA 均保留在 adapter 的 vendor/ 与 b19_source.json；v5 源码原 GPL-3.0 许可仍保留。没有导入新版 Ultralytics 检测器、loss、assigner 或 MuSGD。

另一项目的归档 args 和实际启动日志已只读核对并保留。运行名为 `b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42`，项目锚点为 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`。日志显示原训练 Ultralytics **8.4.98**、Python 3.12.3、Torch 2.8.0+cu128；这不是固定增强适配源 v8.4.0，也不是用后续评估版本替代训练版本。版本 banner 不能证明原训练的完整源码 commit；本机未找到命名为 YOLO26_b19_launch_record.md 的文件，实际使用的是已归档 args 与 startup excerpt。

## 参数和实际变换

recipe.json 与 hyp.yaml 是正式配置，最大200轮、patience50、640、物理batch16、train workers8、device0、seed42、deterministic/AMP、SGD/Nesterov、nbs64、lr0=.01、lrf=.1、momentum=.937、weight_decay=.0005、warmup5/.8/.1、cos_lr、close_mosaic10。freeze=null 映射为 v5 的 [0]，表示不冻结任何层；cache=False、rect=False、multi_scale=False、resume=False、save/plots/val=True、save_period=-1、exist_ok=False。warmup 后原生 accumulate=4、有效 weight_decay=.0005；正式实际值由 effective_training.json、first_update.json 和逐轮 augmentation JSON 记录。

| 在线增强项 | 本 pilot 值 |
|---|---:|
| hsv_h / hsv_s / hsv_v | .024 / .84 / .535 |
| degrees / translate / scale | 11 / .17 / .735 |
| shear / perspective | 3.5 / .00055 |
| flipud / fliplr / bgr | 0 / .5 / 0 |
| mosaic / mixup / cutmix / copy_paste | 1 / .135 / .03 / 0 |

原 v7 medium loss 保留 box=.05、cls=.3、obj=.7、cls_pw=1、obj_pw=1、iou_t=.2、anchor_t=4、fl_gamma=0。nl=3、nc=1、imgsz640 的官方有效增益为 box=.05、cls=.00375、obj=.7。正式实际缩放结果仍由原生训练记录确认；没有 YOLO26 box/cls 或无效 dfl 字段。

**预处理核实有一处与提示词假设不同：父 scratch 的 v5 实际仍是等比 resize + LetterBox。** 本 pilot 按明确要求，只在新工作树修正为 train 图像先用 INTER_LINEAR 方形 stretch，再进入 Mosaic/原生 random_perspective。原生 val 继续原流程，公共导出继续冻结 LetterBox。保留父 adapter 的 additive-hue HSV 算子，不迁移 b19 全部预处理。相对实际父 scratch，除了 COCO 初始化与 b19 参数/CutMix，还包含这处 stretch 修正；不能把指标变化归因于单个因素。

实际顺序：train-only stretch → Mosaic 或单图路径 → 原生几何 → MixUp → CutMix → 保留 HSV → 垂直/水平翻转 → v5 标签格式化。两个混合伙伴均来自当前 train dataset，只执行 pre-transform，不调用递归混合。没有隐式 Albumentations。CopyPaste/bgr 为0。

CutMix 保留固定原码检测规则：Beta(1,1)，尝试3个矩形；仅选与现有目标框无交叠的区域；伙伴框保留条件为 patch 对 donor box 的 IOA≥.1，随后裁到 patch 边界并保留离散类别。没有分类软标签或额外尺寸过滤。无可用区域、无 donor 均跳过，因此 **.03 是触发概率，不是实际应用率**。共享同步计数区分 triggered/applied/no_free_area/no_donor。记录范围是 worker 实际执行的变换，包含预取及被丢弃的 batch，不冒称精确的消费样本统计。

200轮下，索引190开始关闭 Mosaic/MixUp/CutMix/CopyPaste，重建 worker 迭代器并丢弃旧预取，保留 stretch/几何/HSV/翻转。续训从最后10轮进入时同样关闭。

## 验证、评测、身份与恢复

训练 val 请求 batch16，覆盖原生翻倍；其余保持原生 rect=True、conf=.001、NMS IoU=.6、max_det300、无TTA。CUDA AMP 下实际使用 FP16 model/input，每次调用将参数、loader batch、实际输入形状和 dtype 写入 native_validation_calls.jsonl。rect 的实际 shape 可以大于请求的 imgsz（例如64 smoke实际96×96），不能把它说成方形公共640评测。每轮保存原生 P/R/AP50/AP75/mAP 原始浮点与百分数；按完整精度 mAP50–95 选 best，ties 保留较晚 epoch，patience 同一指标，test不参与。

自然停止后，同一个 best/EMA 执行公共 val/test：FP32 model/input、方形LetterBox640、batch16、conf=.001、类内NMS IoU=.7、max_det300、无TTA、corrected_sorted_conf_mask_v1。保留空预测图，按实际x/y gain和整数padding逆变换，不额外裁框。summary 保存原始值、百分数、各类 AP、checkpoint/GT/预测身份；中断跑次保留实际已保存轮数并标记 incomplete，不补成200轮。

每个正式 run 新建 UUID，冻结模型/尺度、COCO hash、hyp/b19来源 hash、adapter hash、project commit、数据身份和真实环境路径。仅本 run 的 last 可恢复；检查 UUID、完整选项/hyp、初始审计 hash、optimizer/EMA/scaler/scheduler/RNG、last_opt_step、accumulate、best/last 与 epoch_state。原初始化记录不覆盖，恢复迁移审计另存。保留官方 half 模型/EMA检查点及原生每轮开始清零残余梯度语义；恢复 worker 预取随机流会重启，**不承诺 bitwise 连续训练等价**。

## 本机验证范围

原数据轻量核对为 train6048/45573框、val1728/12840框、test864/6663框，YOLO0→公共1；路径、标签清单和文件大小身份与冻结记录一致。没有全库图片哈希或解码、复制或再增强。原图/标签只读；所有清单、缓存、锁与输出属于新分支/run。

独立 --copies venv 使用本机 RTX2060 6GB，Python3.9.25、Torch2.7.1+cu118、torchvision.22.1+cu118、NumPy1.26.4。新环境叠加固定依赖，Torch/torchvision从已有匹配环境只读继承；sys.executable/sys.prefix与真实 import 路径已核对，未改母版环境。已有 Ultralytics 分发包不会被 v5 进程导入。

已通过：475/481真实迁移值和参数量；243个有限非零梯度及实际 SGD 更新；100种随机种子下 CutMix像素/标签与固定原码及其原始IOA函数完全一致；裁框和两类跳过规则；双worker真实混合与189/190关闭、最后10轮恢复；CUDA batch2/64，两轮，中断后恢复，初始化审计保留，错误身份/选项/完成跑次拒绝；真实合成 best 的FP32/640双split导出与公共评测；空预测 fixture、非方形逆变换；退出码/原始控制字符5项测试与mock tmux会话保护。plots=True生成 labels/train batch 图，补丁仅为 Pillow≥10 绘图兼容和缺字库时用PIL默认字体，无训练中字体下载。

尚未验证：AutoDL实际环境、正式全数据训练/val/test、GPU0两组batch16/640并发容量、真实Linux tmux/POSIX信号。预检失败和绘图适配的早期失败日志保留在本机新输出中，最终通过记录另存。既有 scratch/原COCO/母版结果与表格均未覆盖，母版52.200902%不是指标目标。

完整固定提交服务器命令在同目录 YOLOV5M_COCO_B19_PILOT_SERVER_COMMANDS.md；本机可复核证据为 evidence/yolov5m_coco_b19_pilot_validation.json。
