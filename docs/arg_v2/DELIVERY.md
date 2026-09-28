# ARG v2 交付

已实现 `ARG-v2-q2`，本轮未启动正式训练、完整 val 或 test。
实现为独立 v2 criterion/model/trainer/validator 和工具入口，保留 v1 和母版文件。
服务器操作见 [server_commands.md](server_commands.md)，公式与流程说明见
[README.md](README.md)，本地检查证据摘要见 [local_validation.json](local_validation.json)。

## 提交与路径

- 分支：`exp-rtdetr-r18-lite-arg-v2`。
- 训练代码提交：`5e7fde5e8393107a4f6c9df9d8282e81a7d03b62`。
- 最终代码提交信息：`fix(arg): preserve completed evaluations when lock publication fails`。
- 主实现提交：`37b0ee0693a2a657e3c9c60338be9441c472cea9`，信息为 `feat(arg): add isolated q2-gated v2 and reusable evaluation delivery`。
- 起点：`ef9cb7e05e5557f7dd06c95cf2361998a284adc9`；母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
- 实际 origin：`https://github.com/supershaojie/concrete-crack-rtdetr.git`。
- 本机管理 worktree：`C:/Users/o'v'o/.codex/worktrees/arg-v2/Crack_RTDETR`。
- 本机路径别名：`D:/MyProjects/Crack_RTDETR-arg_v2`，junction 指向同一管理 worktree。
- 服务器独立 worktree：`/root/autodl-tmp/projects/Crack_RTDETR-arg_v2`。
- 服务器 Python：`/root/miniconda3/envs/rtdetr/bin/python`；tmux：`arg-v2-training`。
- 正式 run：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/arg_v2_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。

本文件、命令和验证摘要由后续纯文档提交交付，不在文档中伪造自身 SHA。
最终回复给出该文档提交及远端核对结果。服务器固定 checkout 上述代码提交，
不按远端最新 HEAD 改变 prepare/preflight/training 的绑定。
服务器路径取自本次要求并已核对脚本的配置，现场存在性由同步/prepare 检查。

## 实现与保留证据

只在原 geometry 逐匹配归约前加入 `stopgrad(IoU)^2`：

    raw_v2 = sum(q² * 0.5 * (wx*ell_x + wy*ell_y)) / max(M, 1)
    loss_arg = 0.20 * clip((trainer.epoch-5)/15, 0, 1) * raw_v2

epsilon_w=.05、gamma=2、eps_num=1e-7；原显式轴向修复框及可导一维 GIoU 不变。
最终普通 CBR 框复用原最终匹配，DN/aux/encoder 不新增项，原 L0 完整保留。
q=0 时 ARG 附加值与对应框梯度为零，r=0 直接走母版路径，无门控和归一化或增大 lambda。

LIF/CBR 的 LF SHA256 分别为
`26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7` 和
`d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787`。
公共初始化源 SHA256 为
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
原构建/映射下未融合 nc1 参数 20,149,765、state keys 552，参数集合和形状一致。
109 字段正式配方只更改 model/name/save_dir 的实验身份；原增强和训练设置逐值逐类型保持。

每 epoch 前四个 batch 的采样诊断保存 count/sum/sumsq 和四个 IoU 区间，
按匹配数聚合，空区间 null。记录真实 optimizer 更新数，不添加诊断前向/反传。
prepare 优先复用保存的数据快照，保留 names 字符串键规范和冲突报错。
日常入口不全量 inventory；显式 recheck-data 保存真实差异并使原快照失效。
同路径内容的原地修改不能由轻量元数据检查保证发现，数据必须冻结，疑似改变需显式重验。

final_eval 在原生 strip_optimizer 后，单次正式 FP32 val 同步产生完整 300-query+GT 导出和锁。
finish 复用 val，执行缺少的锁定 test，一次完整打包；重入恢复缺失阶段。
已成功评估却缺材料时要求显式恢复。status/pack 无模型推理和原始数据扫描。
包默认包含全部已有预测和证据，不含权重/数据本体；缺项明确标 analysis_complete=false。

## 本地检查：实际范围

环境为 Windows、Python3.9.25、torch2.7.1+cu118、CUDA11.8、RTX2060 6GiB。
检查在 ef9cb7e 基座加本轮 v2 工作区内容上执行，再提交为上述代码 SHA；
验证摘要记录源文件 LF 哈希与提交内容一致性，不能将测试时的基座 HEAD 冒写为当时已提交。
本地原始记录位于管理 worktree 的 outputs/arg_v2，未提交诊断权重或数据/训练产物。

| 检查 | 结果 | 实际范围 |
| --- | --- | --- |
| 几何/自动微分 | PASS | 7 项数学测试；q² 逐框梯度、冻结权重有限差分、q=0、空/窄/分离框、FP32、统计聚合 |
| 匹配/损失接线 | PASS | DN/无 DN、混合空和全空 GT；原匹配调用与原损失键不变；仅增加 loss_arg |
| 母版一致性 | PASS | 真实公共 init；原生 get_model；B2/160 CPU 同前向、r=0 loss/grad/AdamW 更新逐位一致 |
| ARG-only 梯度 | PASS | CBR offset/query、最终回归、共享 Neck 有限非零；独立最终分类头无直接梯度 |
| 新进程 CPU 恢复 | PASS | epoch20/ramp1、optimizer/EMA/禁用 scaler 状态一致；不代替 CUDA scaler 检查 |
| 数据身份 | PASS | 10 项实际临时 YAML/图片/标签/JSON IO；prepare/binding；缓存、复用、碰撞与真实变化拒绝 |
| 运维边界 | PASS | 6 项服务器控制模拟；实际临时锁、子进程超时、哈希与故障包 IO；未实际 dispatch |
| 导出/finish/恢复 | PASS | 10 项合成评估器/训练完成夹具；真实 final_eval 接线、exporter、锁发布故障、包、重入恢复；未正式 val/test |
| 本机真实 FP32 生命周期 | PASS | 独立进程加载、有限零输入 warmup、1 张真实 val、300-query 导出；融合前后母版输出一致 |
| 原生 Trainer setup | PASS | 本机 B2/160/workers0 诊断，原生重建和离线 AMP check；未执行正式 B16 更新 |
| 保存的数据清单 | PASS | v1 已有 8640 行清单内部哈希/配置/映射/划分计数核对；原始图像/标签未重哈希 |
| 静态和完整配方 | PASS | 14 个 Python 文件、2 个 Bash 脚本、11 个 CLI help、109 字段配方、Git diff 检查 |

初次本机 lifecycle 检查遇到管理路径中的单引号被上游权重加载路径过滤而找不到文件，
失败记录保留在 outputs/arg_v2/local_lifecycle/lifecycle.json。
通过上述 junction 别名在同一 checkout 完成复验，成功记录为
outputs/arg_v2/local_lifecycle_alias/lifecycle.json。未为此改动公共模型/加载器源码。
单图的零指标属于公共初始化诊断，不能作为 ARG v2 的正式结果。

## 待服务器执行

服务器 B16/640 AMP 容量与实际更新、CUDA scale/overflow/显存、新进程实际 FP32 val、
完整 CUDA optimizer/EMA/scaler resume 均为 **PENDING**。本机 setup/单图 val 不替代这些检查。
900 秒/16 micro-batch 的预检保留 v1 必要资格规则；正式原生 scaler 不因隔离诊断改变。
完整训练、全量 val/test、耗时开销和精度收益均为 **PENDING**，本轮没有涨点结论。

按命令页依次执行同步、prepare、preflight、status，再单独 start。
训练后只需 finish；状态/日志、resume、显式导出恢复与纯 pack 命令均已分开列出。
所有恢复保留原始错误和退出码，不把训练完成与 final_eval 成功混为一项。
