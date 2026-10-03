# YOLOv8m 官方 COCO + B19 在线增强 pilot 交付记录

实现提交：`16a2e28686b3471ef1b7640b3ba697597335895c`。
实际父提交：`05c6b4c8aeaf04255b7e17f3391d2202e65d9ffe`。
分支：`bench/yolov8m-coco-b19-pilot`；后续文档提交只增加本记录、命令和验证证据。

## 范围和派生关系

从已核实的 scratch 最新提交派生独立工作树，包含代码锚点 `61c386bc722b11208453cab0808d8f6edb15385d` 及后续文档；该锚点至实际父提交没有训练代码改动。保留真实模型 AMP 副本的 RNG 隔离、同 run 身份、原生优化器、SGD 分组及日志/数据隔离修复。参照原 COCO 提交 `74bc7d7168d4fdc199a54d1dc415c497dd0ddfce` 恢复官方加载路径。

公共比较基点 `529c456b9404f1d9ab66d82d2b2f9ec7e0c98545` 与母版锚点 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 均保留。没有合并 main、改动原 scratch/COCO 工作树、其他模型或 Tunnel_Disease_YOLO26。尚未登录服务器或启动正式训练。

这组相对 scratch 同时改变初始化与在线增强，结果不能单独归因于一个因素。既有结果不覆盖，配方不按“让对比指标更低”选择；母版归档 52.200902% 不是本组指标目标。YOLOv8m COCO、旧 scratch 和 RT-DETR ImageNet 骨干的初始化按各自实际情况记录。

## 模型、来源与加载

| 项目 | 锁定值 / 实测 |
|---|---|
| 模型源码 | Ultralytics v8.3.20，`f4d8f7765a490f3920e2d14c592a2967e347f185` |
| 原始 COCO 资产 | [官方 yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt)，52,136,884 bytes |
| COCO SHA256 | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` |
| 兼容补丁 SHA256 | `461f2d4ce4ed99e691517ea316df4ebfb9a434d5f773240e6ffc9a51340ccbe1` |
| 补丁后源码内容 SHA256 | `7588e53839944a7cd34cde919744a7903343f06acb2f24655c94aa397b213dfb` |
| nc=1 未融合参数 | 25,856,899；正常可训练 25,856,883，仅固定 DFL 的 16 元素不训练 |
| 迁移覆盖 | 469/475 state_dict 张量，25,889,517 元素；首次更新前逐值相等 |
| 跳过 | 三个分类输出层各自 weight/bias，共六张量；nc80→nc1 尺寸不同 |
| 检测头 | 官方 M、legacy Conv 分类分支；未换 DWConv、loss、DFL 或 assigner |

本机复用已下载的原始官方权重，复制前后核对 SHA256；检查点文件未修改。正常新 run 走官方 `DetectionTrainer.get_model -> DetectionModel(original COCO YAML, nc=1) -> model.load(weights)`，resume=False。没有加载旧 best/last 或 YOLO26 权重。独立 vendor/source 与 `--copies` venv 的 sys.executable、sys.prefix 和真实 import 路径均已核验。

## B19 参数映射和最小 CutMix 适配

B19 运行线索：`b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42`；原项目锚点 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 已只读核实存在，该锚点 `__version__` 为 8.4.98。指定的 `YOLO26_b19_launch_record.md` 在所给本地项目及其可访问工作树中未找到，部分归档 symlink 目录不可访问。本次明确采用用户文档列出的参数；不宣称完成原始启动记录的独立审计或复现其全部数据流程。

| 项目 | 本 pilot |
|---|---|
| 初始化 / 最大轮数 / 早停 | coco_detection_pretrained / 200 / patience50 |
| 尺寸 / batch / nbs / workers | 640 / 16 / 64 / 8 |
| device / seed / deterministic / AMP | 0 / 42 / true / true |
| cache / rect / multi_scale / freeze / 新 run resume | false / false / false / null / false |
| optimizer / lr0 / lrf / momentum / weight_decay | SGD / 0.01 / 0.01 / 0.937 / 0.0005 |
| warmup_epochs / momentum / bias_lr / cos_lr | 5.0 / 0.8 / 0.1 / true |
| HSV h/s/v | 0.024 / 0.84 / 0.535 |
| degrees / translate / scale / shear / perspective | 11.0 / 0.17 / 0.735 / 3.5 / 0.00055 |
| flipud / fliplr / bgr | 0.0 / 0.5 / 0.0 |
| Mosaic / MixUp / CutMix / CopyPaste | 1.0 / 0.135 / 0.03 / 0.0 |
| close_mosaic | 10；200 最大轮数下从零索引190关闭全部四种混合 |
| box / cls / dfl | 7.5 / 0.5 / 1.5，保留官方 YOLOv8 损失含义与归一化 |
| save / save_period / plots / val / exist_ok | true / -1 / true / true / false |
| 分类/分割额外增强 | erasing=0、auto_augment=null；detect 任务下其余分割/姿态字段 N/A |

没有迁移 MuSGD、batch32、lrf0.003、warmup3 或 patience60。训练继续使用已核验的母版方形 stretch 和 additive MotherHSV；只迁移 B19 在线参数及明确 CutMix。

CutMix 最小摘录来自[固定 v8.4.0 源码](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/ultralytics/data/augment.py)，commit `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`，原文件 SHA256 `e31b60824bb2b1871b2b9a58c0db806a80b3b216f9c300fd28debfa4fe2e4855`。只保留 CutMix 三个方法；名字/导入适配与删去 docstring 之外计算代码不改。摘录 SHA256 `81611e47ab91c530e0bbdedf61f61cd0758ccb83489c68cd3d8a9f89fc54c250`；AGPL-3.0 完整许可随 `LICENSE.cutmix.txt` 保存，SHA256 `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`。

旧版 BaseMixTransform 缺默认伙伴索引，适配器仅从 train 取原始样本；旧版 Instances 在 IOA 前显式转为绝对 xyxy。Beta(1,1)、三个候选裁片、避开主图框、伙伴框至少10%面积被覆盖、裁到粘贴矩形的规则均保留。硬检测类别和框拼接，不使用分类软标签。MixUp 与 CutMix 的伙伴只经过有界 Mosaic/CopyPaste/Affine，不进入完整链再次递归。

实际链：Mosaic → CopyPaste(p0) → RandomPerspective → MixUp → DetectionCutMix → NoAlbumentations → MotherHSV → 垂直/水平翻转 → RGB Format。`cutmix` 与 `initialization_type` 在 adapter 配置中消费，不传入旧版 get_cfg。各 worker 共享 visits/triggered/applied/skipped_geometry 计数。

close_mosaic 重建四种概率为0的数据链；原生 epoch190 条件调用 reset，适配器先关闭旧 iterator 的 worker 再创建新 worker。resume 跨过关闭点时也刷新 worker。其余几何、HSV、翻转保留。

## 优化器、选模与公共评测

原生 SGD 三组保持：83 bias 无 decay、84普通 weight（含固定 DFL，无梯度者不更新）decay0.0005、77 BN weight 无 decay，Nesterov=true。正式 batch16/nbs64 的 warmup 后累积4，有效 decay0.0005。保留原生 warmup、cosine、scaler、unscale、clip_grad_norm(10)、step/update、zero_grad、EMA 顺序，运行时记录实际各组 LR 与累积。没有 auto optimizer、AutoBatch 或 OOM 降配。

训练原生 val 显式 batch16、rect=False、640、conf0.001、NMS IoU0.7、max_det300、无 TTA；按原生全精度 mAP50–95 选 best/早停，相等值选较晚轮次。实际输入 dtype 写入 native_validation.json；本机 smoke 测到 CUDA FP16，服务器尚待实际记录。test 不参与选择。

最终公共导出加载选定 best 保存的 FP32 EMA，FP32 输入，公共 square LetterBox640、配置 batch16、相同 NMS 条件、无 TTA。逆变换使用真实 round 后的 x/y gain 和整数 pad。小样本发现旧边界 clamp 会使 padding-only 框退化为零面积；本 pilot 不追加坐标裁框或删除 NMS 框，保留这类假阳性。这个明确的处理和 rounding 修复写入 postprocessing 身份，既有历史结果不重写。公共接口允许原图坐标空间内有限、正面积且可越界的框。

公共评测只调用已有 `corrected_sorted_conf_mask_v1`，policy SHA256 `bf26c02b5682c982ffc3f19ee621c647a228121567f8de19d3ccbabb67bb12aa`。保存空预测图、P/R/AP50/AP75/mAP50–95 原始小数/百分数、各类 AP、best epoch 和 GT/预测/检查点身份。

每 run 新 UUID；身份包含模型/尺度、COCO 来源、B19 recipe、适配代码、固定上游/补丁、项目提交和轻量数据身份。正式 resume 仅本 run last，比较完整身份、冻结 native 参数、best/last、早停与初始化记录。额外保存/恢复 FP32训练模型、FP32 optimizer/EMA、scaler、scheduler、RNG、增强关闭状态；保留官方兼容 checkpoint 键。仍为原生 epoch 边界恢复，不声称跨 epoch 待累计梯度及预取 worker 批次位级重放。

## 已执行验证与实际限制

1. 官方 COCO SHA/大小、legacy M nc1 参数和全部469匹配张量逐值核验。
2. 真实 RT-DETR 数据轻量 preflight：6048/45573、1728/12840、864/6663，全标签有效、无孤立标签；复用并核对已有 val/test GT。轻量身份 `3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483`，未全库图片哈希/解码。
3. 30项检查中29通过，1项 Windows 下跳过的 Linux信号转发测试。含10集成、5B19几何/实效、4初始化/恢复guard、5公共接口、6 launcher/lifecycle检查（其中1skip）。
4. 实际B19概率链合成256样本：Mosaic应用291（包含伙伴预处理）、MixUp触发/应用30、CutMix触发5、应用1、几何跳过4。配置 CutMix=0.03；应用比例不是配置概率。几何断言覆盖主图冲突、伙伴不足10%覆盖、裁切后框和 hard class。
5. 两个真实 worker 在零索引189/190前后取批：关闭前强制混合概率1以覆盖操作；关闭后新PID，Mosaic/MixUp/CutMix均不执行，HSV保留。固定概率另由第4项验证。
6. 本机 RTX2060 6GB，Torch2.7.1+cu118：真实 COCO 模型 AMP 副本与原模型/BN/optimizer/EMA/RNG隔离；合成64张 train、2 val、2 test，batch2/64、2 epochs，第一轮后中断并同身份恢复。原生 scaler 13次调用中9次溢出跳过、4次成功，243梯度张量，并证实非零 SGD 权重更新；DFL始终固定。
7. 恢复前逐项 exact equality 校验 FP32模型、optimizer momentum/组、EMA、scaler、scheduler及Python/NumPy/Torch CPU/CUDA RNG，初始化文件未覆盖。随后同一 smoke best 完成 FP32/640公共 val/test 各2图导出和公共CPU评测；合成指标均为0，不是论文数据结果。配置 batch16，实际小样本尾批2，不能声称实际16图容量已测。

必要证据：[validation JSON](evidence/yolov8m_coco_b19_validation.json)、[smoke记录](evidence/yolov8m_coco_b19_smoke.json)、[原始训练控制字符日志](evidence/yolov8m_coco_b19_smoke.raw.log)、[增强实效日志](evidence/yolov8m_coco_b19_augmentation.raw.log)。本地验证在提交前运行，smoke的项目commit字段为实际父提交；验证的adapter SHA与交付代码完全一致：`5ff1936f27675f2faecf6f0f7d5c053599a14c593be694f5285f61e8fabb695a`。B19 recipe SHA：`b0fef794bd5e3b662a5f2efbedb57196ca94b62a1d66c0af5a5468c5ed66c4dd`。

尚未验证：服务器真实 batch16/640 AMP 双任务GPU0显存；Linux信号转发；正式完整训练和真实全量公共val/test；B19原始启动记录的独立查验。本机 Python3.9.25 会显示上游 Python>=3.10 的 feature-probe 警告，已跑通所列检查；服务器参考环境Python3.10/Torch2.1.x的实际导入与CUDA检查由bootstrap和运行记录确认。正式训练未开始。

完整后续操作见[固定SHA服务器命令](YOLOV8M_COCO_B19_PILOT_SERVER_COMMANDS.md)。
