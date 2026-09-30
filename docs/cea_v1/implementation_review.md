# 相对母版的实施审查

基点：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。

| 原文件 | 审查结论 |
|---|---|
| nn/modules/cbr.py | 唯一运算流外扩展为 return_context 和显式 hidden.detach 返回；原表达式、初始化、精度不变；已与 Git 母版实际 forward 比较 |
| nn/modules/lif_down.py | 完全未修改 |
| nn/modules/head.py | 完全未修改；确认普通 query=300、decoder=3，原 DN 输入构造 |
| nn/modules/transformer.py | 完全未修改；确认 return_final_query 对应最后返回框的同一层 query |
| nn/tasks.py | predict 新增默认 false 参数与一个 head.forward_with_context 分支；母版 loss 未修改 |
| models/utils/loss.py | 完全未修改；派生类仅显式分离 final/aux 的匹配参数，DN 用原父类逻辑 |
| models/utils/ops.py | 完全未修改；matcher 增益 2/5/2、独立 alpha/gamma 保持 |
| models/rtdetr/train.py | 完全未修改；正常模型、数据增强、验证器复用 |
| engine/trainer.py | 完全未修改；通过顶层子类/回调绑定 epoch 与日志；原 backward/accumulate/scaler/clip/step/EMA 不替换 |
| nn/autobackend.py | warmup 的 torch.empty 改为同 shape/dtype/device 的 torch.zeros；这是有限输入修复，不修改真实预测或掩盖非有限输出 |

新增实验代码只在 `models/rtdetr/cea.py`、`cea_trainer.py` 和 `tools/*cea_v1*`。
母版独立评估 helper 的排序后 mask 修复作为清晰的小函数实现，并新增显式原 query 索引，
没有导入旧实验启动、GPU 闸门或自动打包行为。

初始化复用原 build/topology/is_added/native_rebuild，重新审计受控映射与 9 个允许分类适配张量。
原 source_contract 的老 CBR 哈希保护未关闭或修改；新 context API 的等价性由独立代码比较测试验证。
原 LIF/CBR 零输出初始化、state_dict 键、unfused/fused 参数量均核对。

服务器生命周期独立实现所需的薄调度：run 范围锁、身份、有界子进程、tmux 留页、退出码、纯离线包。
未调用母版 `start_direct`、其 preflight gate 或 package helper。
原 Trainer OOM 缩批重试通过原计数分支和回调阻止；成功 batch 恢复计数避免重启 epoch。
正式训练完成后保留原 best 选择并 strip optimizer，独立 FP32 评估留给显式 val/test。

功能摘要包含实际被调用的模块和配置，以及母版数据/权重证据；纯 Markdown 不参与摘要。
同 run 的 init/data/code/config/environment 任何变化都使原预检失效。
没有依赖必须存在的模块 ZIP，也没有从其它算法实验取初始化或混入其损失。
