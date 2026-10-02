# 官方 YOLOv13l 随机初始化对比接入

日期：2026-10-03（北京时间）。范围为代码、有限验证、普通提交推送和服务器命令。
未登录服务器，未启动正式训练或真实数据全量推理，未停止或修改正在运行的 YOLOv5m/YOLOv8m。
正式结果待用户安排运行后生成。

## 身份与边界

- 用户远端：https://github.com/supershaojie/concrete-crack-rtdetr.git
- 分支：bench/yolov13l-scratch。
- 直接父提交：61c386bc722b11208453cab0808d8f6edb15385d。
- 公共比较基点：529c456b9404f1d9ab66d82d2b2f9ec7e0c98545。
- 母版锚点：a0459d6a652cb702699087c88fa39a3e4c4087ec，RT-DETR-R18-Lite + 原 CBR + 原 LIF-Down v1。
- 官方：[iMoonLab/yolov13 固定提交](https://github.com/iMoonLab/yolov13/commit/73289949533efac82bb5f72ec19b746618656bd2)；
  [论文](https://arxiv.org/abs/2506.17733)。
- 包版本8.3.63只是一个字段；同时核验 remote、完整 commit、补丁、源文件树哈希和实际 import 路径。
- 新 YOLO=random、pretraining_source=null、pretrained_tensors_loaded=0；
  既有 RT-DETR=ImageNet 骨干，历史 COCO YOLO 结果保留原条件。

仅新增本模型目录、启动脚本和本模型文档/回执；母版、公共评测算法、原 YOLOv8m 实现未修改。
数据、vendor、环境、检查点、缓存和大结果均不推 GitHub。

## 实际结构与初始化

从固定官方 ultralytics/cfg/models/v13/yolov13.yaml 解析字典，显式设置 scale=l、nc=1。
官方只有 n/s/l/x，没有 m；l=[1.0,1.0,512]。新 run 从不调用 .pt 构造器。

| 项目 | 实际值 |
|---|---|
| 未融合 / 融合参数 | 27,566,874 / 27,513,562 |
| 输出 stride | 8 / 16 / 32 |
| AAttn / ABlock | 各16 |
| HyperACE / FullPAD_Tunnel | 1 / 7 |
| DSC3k2 / DSConv / A2C2f / DownsampleConv | 6 / 58 / 2 / 1 |
| Detect | 非 legacy，原生深度可分离分类分支，reg_max=16 |

逐层 from 连接、类型、模块数和参数量都有运行断言。
保留官方 l 规则：DSC3k2 的 l 分支；A2C2f residual=True、mlp_ratio=1.5；
HyperACE/DownsampleConv channel_adjust=False。没有沿用 YOLOv8 旧检测头断言。

FullPAD gate=0；ABlock Conv trunc_normal(std=.02)；A2C2f gamma=.01；
HyperACE prototype Xavier；BN weight=1/bias=0、eps=.001/momentum=.03；
Detect 回归 bias=1、单类分类 bias=log(5/(640/stride)^2)；DFL 固定 arange(16)。
构造器 stride 探测产生的原生 BN 状态不重置。程序核验 gate/BN/DFL/检测偏置，
保存 ABlock 统计和源码依据，不把全部张量强行改成同一分布。

构造期间拦截 torch.load、Module.load_state_dict、BaseModel.load、hub 权重和下载函数。
运行 initialization.json 保存原 YAML 哈希、有效配置哈希、scale/nc/seed、参数量、
完整拓扑、源码/补丁/项目/数据身份和展开训练参数。相同常量不被误判为预训练迁移。

## 后端、精度、环境

本次固定作者原生非 Flash：显式矩阵乘法与减最大值的 exp/sum softmax。
日志中的“SDPA”不代表实际代码；没有替换运算、布局、缩放、分区或梯度。
本地 RTX2060（计算能力7.5）无可用作者 Flash，等价性记为未验证。
bootstrap 不安装 Flash wheel。若服务器已有可用 Flash，隔离 check 测小 AAttn
area=1/4 的输出、输入和参数梯度，保存容差及结果；训练仍固定 native。
小模块比较不构成全网络等价证明。

AMP 检查使用实际模型副本、64输入，保护原模型/BN/EMA/优化器和 Python/NumPy/CPU/CUDA RNG。
FP32 对照及独立导出禁用 autocast 和 TF32、要求 USE_FLASH_ATTN=False，
检查内部 mm/bmm/exp/softmax dtype。训练保留 AMP；训练内 val 使用官方 CUDA FP16。

本地：Python3.9.25、Torch2.7.1+cu118、TorchVision0.22.1+cu118、NumPy1.26.4。
服务器环境尚未执行。默认创建 --copies 私有 venv，只读继承兼容基础包，
固定安装少量依赖；不向母环境 pip upgrade。核验实际 executable/prefix，
解释器路径使用 abspath，不沿软链接 resolve 回母环境。冻结 pip freeze 和版本，拒绝漂移。

作者 requirements 的 Torch2.2.2/TorchVision0.17.2 及 Python3.11/Torch/CUDA/ABI 专属
Flash wheel 不整份盲装。需要作者 Torch 组合时用 --fresh-torch 在新的私有目录安装；
ONNX、Gradio、Albumentations 等不作为训练依赖。详见服务器命令。

## 数据、训练与选轮

轻量检查：train6048张/45573框，val1728张/12840框，test864张/6663框。
路径、标签字节和图片大小身份匹配既有冻结回执：
3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483。
不重复全量图片内容哈希/解码，不声称完整审计通过。
val/test读取标签和必要图像头生成GT；train尺寸缓存首次加载建立于本run/cache；
不向共享标签写缓存，不读取/删除共享图片旁的.npy。所有清单仍指向原数据。

正式 recipe 与参考提交除 model 外相等。完整展开值见机器回执。
640、epochs200、patience50、batch16、workers8、seed42、nbs64、SGD、
lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、cos_lr=True、warmup5。
nbs64 是累积基准；稳定阶段 accumulate=4、有效 weight_decay=.0005。
损失、task-aligned assigner、EMA更新和 max_norm=10 梯度裁剪均来自固定官方源码。
val强制batch16/workers0；无AutoBatch/OOM降配/自动恢复。
选轮和patience使用训练val未舍入mAP50–95，相等时更新较新epoch；原生fitness另存，
test不参与。增强详见 [映射文档](YOLOV13L_SCRATCH_AUGMENTATION.md)。

## 检查点、日志和恢复

保存 FP32 训练参数/EMA、优化器、scaler、scheduler、RNG、最佳轮和早停状态，
附原初始化校验值及UUID/模型/数据/配置/代码/环境身份。
checkpoint_pair.json 最后写入last/best哈希；保存中断造成不配对时拒绝恢复。
仅显式恢复本run的last，核对best及最佳epoch；拒绝其他模型、COCO、其他scratch、
smoke和已完成检查点。恢复参数另存，不覆盖原始初始化及展开参数；
保留中断前CSV，再恢复到完整检查点行数。

采用epoch检查点语义；多worker预取和未保存批内累积轨迹不保证逐bit连续等价。
原生日志恢复时“Transferred ... pretrained weights”为通用措辞，此时仅加载本run的核验检查点。
tmux使用实际window ID，remain-on-exit在门闩放行前设置。
stage.py保留回车进度及完整日志、真实子进程/tee退出码和信号；只保护本会话/本run。

## 公共评测与资源边界

导出：同一完成训练best，FP32、640、batch16、workers0、conf=.001、
class-aware NMS IoU=.7、max_det300、max_nms30000、rect=False、无TTA，
取消NMS时间提前截断。官方scale_boxes已还原/裁剪，直接取Results.xyxy，不二次反变换。
全精度压缩JSONL含空预测、稳定image_id、原宽高和类别0→1。
公共 corrected_sorted_conf_mask_v1 未修改，P/R取最大F1点，保存P/R/AP50/AP75/mAP50–95
的0–1原值和百分数。已有缓存可CPU重算。

nc1/640/batch1，Conv/Linear及所有mm/bmm合计91.867129856 GFLOPs（MACs×2），
融合与未融合相同，包含AAttn和HyperACE。
**不是完整FLOPs**：不计bias、BN、激活、exp/softmax、归约、池化、插值、
残差/gate/逐元素、decode、NMS。官方THOP日志也不能冒充完整计数。
speed.py为后续独占硬件FP32融合网络测速入口，排除预处理/NMS/I/O；本次未测速。

## 验证边界

已验证官方构建、零权重拦截、原生初始化、10项CPU集成测试、身份/配方和launcher、
真实CUDA两轮合成训练→中断→恢复→同一best的val/test FP32导出→公共评测。
单张640 AMP前后向832个梯度有限，峰值分配1,551,891,456字节。
固定上游全新检出+补丁重放与锁文件源树哈希一致。

这些权重/指标只属合成smoke。未验证正式batch16/640双模型容量，
未执行服务器Torch2.1.2/0.16.2或作者2.2.2/0.17.2组合。
Windows上真实POSIX信号、tmux、Linux venv软链接待服务器验证；
tmux命令顺序、实际window目标与保护逻辑已用受控替身验证。
机器回执：evidence/yolov13l_scratch_validation.json。
