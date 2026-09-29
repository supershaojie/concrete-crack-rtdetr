# GNR v1 实际验证记录

本次开发环境为 Windows、Python 3.9.25、PyTorch 2.7.1+cu118、RTX 2060 6GB。
这些是开发端结果，没有在 AutoDL 执行。本次没有启动正式 200 epoch 实验，也没有生成 GNR 正式 val/test 成绩。

| 检查 | 实际结果与证据 |
| --- | --- |
| 母版与实际 VFL | CONTRACT_MATCH；实际 alpha=.25、gamma=1.5；`mother_contract.json` |
| PyTorch 数学与边界 | CPU、CUDA 固定张量通过；实际 VFL autograd 最大绝对误差 2.9802322387695312e-8；`local_validation.json` |
| 真实模型同次前向/关闭等价 | CPU、2 张 160×160 合成输入、真实模型/DN；输出、loss 和所有参数梯度严格相等；无新增参数或 state keys |
| 活跃损失范围 | 回归/encoder/auxiliary/DN 值严格相等；最终正项分类梯度保留；无新增直接框梯度 |
| CUDA AMP 分类路径 | 真正 FP16 Linear 输出、FP32 detached 权重；关闭/保护损失及正 logit 梯度严格相等；6 个负项发生改变，分类头梯度有限非零。固定张量检查，不是完整模型 B16 更新 |
| 临时 checkpoint 恢复 | 调用原 Trainer 恢复 optimizer、启用的 CPU GradScaler、EMA 和 epoch；从零基 epoch19 恢复到20，r=1 |
| 完整 prepare | 公共初始化 SHA 正确；母版九键 nc 适配；全部109字段/类型及 gatefix 精确别名通过；首次全内容数据核验计数完全一致；`local_prepare.json` |
| 母版实际机制诊断 | REVIEW、覆盖完成：64张val、8个原增强B16 train batch；原始报告 `local_diagnose.json` |
| 真实 B16/640 更新预检 | RESOURCE_ERROR，exit1：首个 microbatch 的原 backbone Conv 前向 CUDA OOM；完成microbatch=0，有效更新=0。保留 `local_preflight.json` 与本机原始日志 |
| 运行脚本离线检查 | PASS；只读status、锁竞争/陈旧owner、离线INCOMPLETE包及manifest回读、零干预拒绝、数据元数据失效、Bash语法和Python/tee分别退出；`operations_validation.json` |
| 独立评估基础设施 | 排序后过滤及原query映射、改变阈值后的离线重新匹配、缺索引时复用及损坏产物拒绝，均通过固定fixture检查 |
| AutoDL真实preflight、Linux tmux全流程 | PENDING；必须在目标环境按服务器命令执行 |
| GNR正式训练、完整FP32 val/test、COMPLETE结果包 | PENDING；用户执行start后产生，本次没有代为长训 |

诊断的主指标分母是**全部普通负项的标量梯度强度和**：

| split | 图片/批次 | 改动负项数量 | removed_negative_strength_fraction |
| --- | --- | --- | --- |
| val | 64 / 4 | 4468 | 0.0015294968795558242（约0.153%） |
| train | 128 / 8 | 10449 | 0.003403680033945688（约0.340%） |

最强负项保留检查通过。这里观察到非零且份额较小的干预；没有设置任意收益阈值，也没有证明 mAP 改善或整网梯度冲突消失。所有分组分母、分位数、欠拟合正项比例与样本清单均保存在原始报告中。

真实preflight仍保留母版B16/640/AMP/AdamW/nbs64和训练日程，只将GNR预检上下文设为20。
原生AMP资源检查通过后，首个带梯度的前向在backbone中报OOM，尚未运行到GNR关系计算及optimizer step。
没有减batch、减尺寸、关闭AMP、降低scaler、清理其它实验显存或自动再试。
原始日志位置记在 `local_preflight.json` 的 `folder` 字段；这份失败不会被解释为TECHNICAL_PASS。

`local_diagnose.json` 和 `local_preflight.json` 保留执行当时的真实功能身份；后续运行脚本加固不改写旧证据。
最终功能版本通过固定张量与集成回归，服务器仍须重新prepare/diagnose/preflight，不能复用开发端旧绑定。
验证阶段的Git HEAD是核验提交，功能源码尚未提交，因此报告同时保存当时工作区摘要及逐文件哈希；最终功能提交和纯文档提交关系见 `server_commands.md`。

本机首次多线程CPU梯度累积曾相差2.7940e-9，改为单线程确定性执行后严格相等，没有放宽比较容差。
Windows含apostrophe的路径曾暴露母版权重loader和Git Bash命令行路径处理问题；初始化入口使用同文件相对路径，Bash语法检查从标准输入读取脚本。原母版文件未改。

核查的新入口和实际调用链只复用指定母版初始化、Trainer、IoU/匹配及c19独立评估排序修复。
没有导入其它实验算法或GPU独占helper；原Trainer自己的显存缓存整理保留，它不等待空卡、不清理其它进程。
唯一强制结束子进程的路径是有界诊断/预检超时，只处理该次新建子进程及其后代。
