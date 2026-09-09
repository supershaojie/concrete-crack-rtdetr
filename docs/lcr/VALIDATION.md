# 必要验证实录

本机Windows、Python3.9.25、PyTorch2.7.1+cu118、RTX2060、CPU i7-10750H。不是AutoDL4090/PyTorch2.1.2实测。

## 实际入口

工作目录 `D:/MyProjects/Crack_RTDETR-lcr`，解释器 `D:/miniconda3/envs/rtdetr/python.exe`；YOLO_AUTOINSTALL=false、CUBLAS_WORKSPACE_CONFIG=:4096:8。

```text
python tools/check_lcr.py --source D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt --output outputs/lcr_checks_v2
python tools/check_lcr_data.py --source D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt --data-root D:/MyProjects/Crack_RTDETR/datasets/crack_det --output outputs/lcr_data_v3
python tools/check_lcr_ops.py --output outputs/lcr_ops_v1.json
```

上述最终运行通过。首轮模型测试发现非contiguous token导致非零dropout掩码排列不同；恢复原Linear输出的contiguous布局后按原规格重测通过。真实样本测试两次修正分别为测试脚本解包、补齐原trainer的nc属性，最终第三次通过；没有修改模型公式、DN/loss或正式配方以通过检查。失败输出保留在本机outputs。

## 数值与学习

模块覆盖batch2、20×20/5×7/1×2/1×1，CPU/CUDA、pre/post-norm、train/eval，dropout=.1、seed72对齐。常量场确认replicate边界，输出形状不变、梯度有限。

TF32 matmul/cuDNN关闭、cuDNN benchmark关闭、deterministic=True；deterministic algorithms warn_only=True。原生CUDA grid_sample backward非确定性警告如实保留，不承诺逐bit复现。

整网合成batch2、160×192，保留原生3层decoder、300queries、DN split=[200,300]、Hungarian和loss，训练seed71对齐。

| 模式 | eval最大绝对差 | train输出最大绝对差 | loss最大绝对差 |
|---|---:|---:|---:|
| cpu FP32 | 5.96046448e-07 | 7.46548176e-06 | 0 |
| cuda FP32 | 0 | 0 | 0 |
| cuda AMP | 0 | 0 | 0 |

FP32 atol/rtol=2e-5；AMP atol/rtol=.005。CPU train最大相对误差约0.005921（分母floor1e-8，接近零元素），按atol+rtol*abs(reference)通过，未误报成纯相对误差小于rtol。逐项误差见 [validation.json](validation.json)。

三种整网模式各3步原生loss/AdamW：DW和gate_out首步有限非零梯度，gate_in首步0、之后非零；记录每个新增参数真实更新幅度，全部参数恰好进入原生optimizer一次。gate_in首步weight decay不当作梯度学习。学习后门控输出变化约0.00065747/0.00059760/0.00063920。模型保存重载输出和optimizer tensor状态精确一致。完整model.half()在CPU/CUDA推理为有限FP16 `[2,300,5]`。

完整公共权重逐键映射、参数量、初始化保存重载通过。原生train API的nc80→1重建在epoch循环前主动停止。

独立FP64参考显式遍历复制边界、分组矩阵计算，随机非零DW/门控，FFN最大误差 `1.7683431618e-07`，RMS输入最大误差 `2.71452933553e-06`，并核对有限梯度。

## 真实小样本和工具

本地每split排序前2张图：train2张只做一次原生DN/loss前后向，无optimizer更新；val/test各2张，每split600预测、2个GT，同一debug checkpoint SHA。实际图像/GT哈希及loss见 [real_sample_validation.json](real_sample_validation.json)。CPU、imgsz160、评估batch1/loss batch2、无在线增强是debug设置，正式640/batch16不变。

小样本输出标记evidence_scope=bounded_real_samples，打包审核拒绝将其作为正式full_split。未训练模型零AP仅验证链路，不是精度结果。临时数据/权重清理；文档只保存指标、完整轻量预测/GT流，不上传原图或权重。

模拟生命周期检查无完整预检依赖、重复保护、OOM异常和退出码及各状态；tmux/OOM为mock，日志中Started属于fixture，未真实启动训练。真实归档IO超过20MiB，逐成员、清单及外部SHA校验通过，缺失证据evidence_complete=false，重复包拒绝覆盖。

shell入口bash -n通过；隔离Git固定SHA/detached保护检查见 [sync_validation.json](sync_validation.json)。源文件LF归一化散列见 [validation_source_hashes.json](validation_source_hashes.json)。报告中的commit是运行时尚未提交修改的基点，不冒充最终实现SHA；最终交付SHA以最终回复核验为准。

## NOT_RUN

正式训练、完整真实val/test、full_server_preflight、AutoDL4090/PyTorch2.1.2、真实SSH/tmux投递均NOT_RUN。正式batch16显存、200e稳定性/精度、延迟未测。本模块使用兼容2.1.2的基础API，未将本地较新版本实测称为服务器验证。

## 最终同步验证

`python tools/check_lcr_sync.py --bash "C:/Program Files/Git/bin/bash.exe" --output outputs/lcr_sync_v3.json` 退出0。
真实本地临时Git仓库验证：远程前进仍固定交付SHA、首次detached创建、detached重复同步、脏目标拒绝、不同SHA拒绝、普通目录拒绝、错误origin拒绝；每种情况主工作树HEAD/状态均不变。对真实detached fixture调用启动器verify_delivery通过，没有启动tmux。仅origin get-url身份mock为正确GitHub URL，实际fetch/push全部在临时本地remote。前两次仅修正Windows PATH测试夹具，服务器同步脚本未因此改写。

最终源码与核心数值验证相比仅增加运行来源断言/日志、Git测试夹具路径兼容及文档证据。核心模块/模型图/初始化公式/训练配方未变。源码散列覆盖交付版本，数值与样本报告保留各自真实运行时HEAD。
