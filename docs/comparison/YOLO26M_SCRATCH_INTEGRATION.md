# YOLO26m 随机初始化接入说明

日期：2026-10-03（北京时间）。本次交付代码、隔离验证和服务器命令；正式训练与真实 val/test 全量推理未执行，论文指标保持缺失。

## 代码和初始化

分支 `bench/yolo26m-scratch` 从指定参考 `61c386bc722b11208453cab0808d8f6edb15385d` 创建。公共基点 `529c456b9404f1d9ab66d82d2b2f9ec7e0c98545`；母版锚点 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。实际代码 SHA 见服务器命令文件。

固定官方 Ultralytics **v8.4.0 / f2d3aed634a5b0e4828024718d4a61ab2f83fb19**，不宣称最新版。读取官方 `ultralytics/cfg/models/26/yolo26.yaml`，显式 `scale=m`、`[0.50,1.00,512]`、`nc=1`、`end2end=True`、`reg_max=1`。真实新训练路径为 `setup_model → get_model(weights=None) → DetectionModel`；`pretrained=False`、`resume=False`。合成测试禁止 torch.load/load_state_dict 后仍完成构建和首轮训练，零迁移由执行路径证明。

保留原生 backbone/neck/head、双分支、one-to-one feature detach、分配器、E2ELoss。reg_max=1 对应 Identity，原生 dfl_loss 日志项实际使用 L1，保留 gain=1.5；box=7.5、cls=0.5。本项目固定 SGD，不声称复现官方 COCO MuSGD 配方。既有 RT-DETR 使用 ImageNet 骨干，本模型 random，两者初始化条件不同。

## 输出与评测

训练输出是 one2many/one2one 字典；eval 输出为 `(processed_BNC, branch_dict)`；AutoBackend 可以返回 list 包装。主输出为原生 one-to-one `[B,N,6]`。官方 non_max_suppression 的 end2end 分支提前返回，只执行 score>0.001 和最多300个结果的过滤，**无 IoU NMS**。展开配置仍显示库默认 iou=0.7，它在此路径不生效。适配器不追加 NMS。

训练 val 固定 batch16/workers0；CUDA+AMP 时实际 FP16。best/patience 读取 validator.metrics.box.map 的未舍入 mAP50–95，等值更新。保存官方返回 fitness 供核对（此版本权重为 mAP100%，返回值可能舍入）。test 不参与选择。

最终同一 best 的 val/test 使用 FP32、640、batch16/workers0、无TTA。官方部署 fuse 只作用于推理模型，移除 one-to-many；训练/恢复保留完整结构。固定方形 letterbox，由官方 scale_boxes 逆变换并裁剪一次，直接导出 Results.xyxy 原始浮点值；保留空图片。缺图、类别、身份、非法框明确报错。

公共 CPU 评估复用 corrected_sorted_conf_mask_v1，P/R 是其最大F1工作点。每个 metrics JSON 保存 P/R/AP50/AP75/mAP50–95 的0–1值、百分数、单位。有效预测缓存可独立CPU重算。

## 数据、增强与环境

真实本地轻量核对通过：train 6048图/45573框、val 1728图/12840框、test 864图/6663框；冻结身份 `3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483`。用实际服务器 YAML 和显式本地 data-root 验证；Windows路径只存在本地回执，服务器重新生成自己的绝对清单。

仅读文件清单、图片stat、标签和必要图片头；没有全量图片哈希/解码审计，不标记 AUDITED。train直接使用YOLO标签；只有val/test生成或验证GT。标签/尺寸缓存按模型/run隔离，不触碰原目录 .cache/.npy，不自动修复源图。

训练 square-stretch；Mosaic → CopyPaste(p0) → 几何 → MixUp → 禁用Albumentations → HSV → 垂直/水平翻转。HSV为加性H、乘性S/V、sat_lut[0]=0，与母版及参考scratch一致；核对固定v8.4.0源码后确认其原生HSV公式相同。CutMix=0且不实例化，避免额外随机数抽样；分类auto_augment/erasing不适用。实际参数见 augmentation.json、expanded_train_args.json 与 actual_training_setup.json 中的实际增强树。

计划第191轮（零基190）关闭Mosaic/MixUp，关闭旧worker和预取iterator后重建，其他增强保留；早停不倒推时间。原生几何框过滤保留（宽高>2、面积比>0.1、长宽比<100）；worker Mosaic buffer和新版prefetch_factor=4（参考为2）意味着随机采样轨迹不能逐样本等同。

独立 .envs/.vendor/.runtime 下的 yolo26m-scratch 目录。Linux venv使用 --copies，记录解释器不resolve软链接；核对sys.executable/sys.prefix、导入路径、版本和源码/补丁hash。锁定overlay使用--no-deps，兼容基础环境只读继承，不升级基础Torch/TorchVision。实际环境和pip freeze逐run保存。

初始化预检独立进程。AMP检查使用当前模型deepcopy，保护Python/NumPy/CPU/CUDA RNG、BN、optimizer/EMA/scaler。比较双分支相同候选位置的原始框/logits，避免top-k近同分排序差异；同时要求已处理输出形状一致且有限。正式训练重新seed42并重建模型。

## 续训

last/best原子保存完整未融合FP32训练权重、EMA、FP32 optimizer、scaler、scheduler、E2ELoss、early-stop、部分累积梯度及最近optimizer step、RNG和loader generator。只显式同run恢复；核对UUID、模型/初始化/代码/配方/数据、best/last配对、完成epoch。正式train入口拒绝smoke；原初始化记录不覆盖，每次恢复新建独立记录。

E2ELoss初始o2m=0.8/o2o=0.2，每轮训练批次结束后、val和保存前由官方update一次：`o2m=max(1-updates/(epochs-1),0)*0.7+0.1`，o2o=1-o2m。完整恢复标量和updates，保持官方时机。原生隐式NaN回滚改为明确失败，避免重置调度。

恢复Python/NumPy/CPU/CUDA RNG和loader generator；worker RNG、预取批次、sampler游标、Mosaic内存buffer未序列化，新iterator会重建。不承诺逐位续训；以最后完整保存epoch为界。

## 资源实测

| 单类，640输入 | 参数 | 已计数算子GFLOPs |
|---|---:|---:|
| 完整双分支，未融合 | 21,774,430 | 74.3386104 |
| 官方融合部署，one-to-one | 20,350,223 | 68.0928248 |

真实 `[1,3,640,640]` FP32 CPU前向，候选位置8400（6400/1600/400），top-k上限300。卷积/矩阵乘MAC×2，支持的add/mul按profiler公式。**这是部分算子计数，不是完整总GFLOPs**：BN、激活、softmax、解码、除法、归约、排序没有完整公式，机器报告列出所有未计数算子，完整总量为null。完整训练结构前向不包括backward/loss/optimizer，未照抄80类表或默认为零额外开销。

独占测速单独入口，仅计预分配tensor上的FP32模型前向/原生top-k，不含读图、预处理、传输。未运行论文测速，并行速度不作为独占结果。

## 验证边界

回执：docs/comparison/evidence/yolo26m_scratch_validation.json。已覆盖数据身份、模型与零迁移、双分支detach、CPU有限损失/反向、HSV和真实双worker第191轮边界、坐标/空预测/公共评估、SGD+AMP合成中断恢复、同best FP32导出、错误身份/缺失状态拒绝、tee退出码、同run/session保护、mock tmux window设置顺序。

本地RTX2060/6GB，合成batch2/64、3轮计划、首轮后中断。这不证明正式batch16/640或双模型同GPU容量。真实Linux tmux/信号进程组、服务器环境及正式batch仍待服务器验证。正式训练、真实全量推理、独占测速均未执行。

未连接服务器，未停止训练，未修改其他模型工作树和原数据；权重/环境/大结果全部忽略，不推GitHub。
