# RMD v1 实施交付

从母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 创建独立分支 `exp-rtdetr-r18-lite-rmd-v1`，本地目录为 `D:\MyProjects\Crack_RTDETR-rmd_v1`。主工作区及其他实验均保留。实施源码与最终固定 SHA 的发布命令一同交付；发布记录将在本文件和 server_commands.md 中明确区分实施提交与仅文档的交付提交，避免提交内容自引用自己的 hash。

实现、公式、接线与操作语义见 [README.md](README.md)，完整任务规格见 [SPECIFICATION.md](SPECIFICATION.md)。

已完成：RMD 仅替换所有原 DN decoder 层的正例 L1/GIoU 系数；临时参考框捕获与最终普通匹配复用；真实 Trainer 重建和 epoch/resume；固定门槛、严格启动、有限检查、独立 tmux、末尾评估恢复、val 锁/test、可保留失败证据的 LIGHT 包。

本机环境为 Python 3.9.25 / torch 2.7.1+cu118 / RTX 2060 6GB / 定制 Ultralytics 8.4.21，解释器 `D:\miniconda3\envs\rtdetr\python.exe`。不是历史服务器的 Python 3.10.13 / torch 2.1.2+cu121 / RTX4090；没有自动升级依赖。

本地验证事实：

- 公式、权重边界、同 GT 均值、K=1、r=0、全相等/全零、空 GT、不同图 GT 数、动态 K、padding/负例排除、未匹配 GT、缺失元数据、非法宽高、shuffle 与跨 GT 错配测试通过。
- 最终匹配只做一次，encoder/普通 aux 分别匹配；所有 DN 层只替换框项，普通损失/DN 分类保留；总 loss 无双计数。
- 固定单线程 CPU 的真实网络 r=0 与 K=1，输出、RNG、梯度和一次原生 AdamW 更新逐项完全一致；get_cdn_group 和 decoder 都只调用一次；正常/异常 hook 清理通过。
- nc80 公共初始化 hash 核验通过。实际 nc1 Trainer 重建 552 个 state 项与母版完全相等，543/552 原同形状态加载，9 个原分类适配；20,149,765 可训练参数，新增 0。
- 数据图像/标签逐文件 hash 和划分计数通过：train 6048/45573 GT，val 1728/12840 GT，test 864/6663 GT。
- 新进程 FP32 AutoBackend/fuse/zeros warmup → 16 张真实 val、44 个 GT、4800 个 query 通过；隔离母版使用同一 warmup 修复，实际预测最大绝对差为 0，LIF BN 保留。此项不代表完整 val/test。
- 原生 checkpoint/恢复的小型 CPU 夹具与启动门禁/末尾评估故障/失败打包检查由仓库内测试提供；精确运行结果在发布验证记录中列明。

母版模块保留证据：

| 文件 | 本机原始字节 SHA256 | LF SHA256 |
|---|---|---|
| lif_down.py | `5660d678aa3acd48336f5f2a6cd49edbe5ac365426edb6d80a3bb87e421a81ac` | `26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7` |
| cbr.py | `6d2161de932c7bfa72707ee0e59a6f97d5733d2331ca01e54dd0e341c99e84b2` | `d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787` |

两个原文件、模型 YAML、原 decoder、原 loss 与 Trainer 均无修改。唯一原文件修改是 AutoBackend warmup 的一行有限零输入修复；新增逻辑在实验包装和入口中。

服务器仍需验证 B16/640 AMP 容量、原生 AMP checkpoint 恢复、固定 900 秒/16 个真实训练 micro-batch 边界内的有效更新及已训练母版生效性。本机没有默认历史母版 run 的 best.pt，不能报告 APPLICABILITY_PASS。新增训练开销尚无配对测量。未启动正式长训、未进行完整独立 val 或 test。

下一步按 [server_commands.md](server_commands.md) 分段执行同步、prepare、preflight；任何 PENDING/FAIL/NOT_APPLICABLE 均不会被另一个 PASS 掩盖，start 会拒绝。
