# 有限验证记录

最终模型检查命令：

```
D:/miniconda3/envs/rtdetr/python.exe -u tools/check_rsc_head.py --source D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt --real-dataset D:/MyProjects/Crack_RTDETR/datasets/crack_det --output outputs/rsc_head_checks5
```

退出码 0，checks.json status=PASSED。运行时尚未 commit，runtime.commit 为基线；记录的 source_sha256 将检查绑定到实际代码，而不是假称检查已在未来交付 SHA 执行。

| 检查 | 实测 |
|---|---|
| 参数量 C2 / RSC / 新增 | 20,082,772 / 20,091,508 / 8,736 |
| nc80 公共映射 | 533 项全部精确；新增6项；缺失/未预期/形状异常为0 |
| 原生 nc1 适配 | 530项加载精确，9项按原类别适配重建；533项公共状态与 C2 全部相等 |
| AdamW | 332个参数张量各出现一次：81 norm weight、120 decay weight、131 bias |
| 描述符 scalar oracle | 最大绝对误差 1.4901161193847656e-7；容差 atol2e-6/rtol1e-5 |
| 单头与整网初始化 | CPU/CUDA logits、boxes、Encoder输出及DN loss 最大绝对/相对误差均0；容差 atol1e-6/rtol1e-5 |
| 原生 DN | 合成 batch2，DN200+普通300；两张真实样本 DN198+普通300 |
| query 索引 | Q503 置换等变、修改单query不影响其他query、输入不原地改写 |
| 梯度 | 框条件无梯度；q与原回归有梯度；mod首步有梯度，sem/geo在mod更新后均有非零梯度 |
| 已开启调制 | 冻结q改变一条轨迹，最大logit差 0.00013375282287597656 |
| train/eval/export历史 | 非零回归偏置、BN固定、无DN接口对照，各层历史和实际输出一致；推理不调用前两分类头 |
| CUDA AMP | batch2/160 合成前向、原生DN loss、GradScaler反向及优化器步通过；loss50.11800765991211 |
| 真正 model.half | CUDA torch.float16 [2,300,5] 输出有限，通过 |
| 学习后保存重载 | CPU/CUDA整模型输出最大误差0，所有参数与优化器状态值相等，mod未重置 |
| 两张真实样本 | 原生DN/loss backward通过，loss56.86052703857422；同次600预测与全部GT导出；无整集指标 |

TF32 关闭、cudnn benchmark=false/deterministic=true、torch deterministic warn_only=true。CUDA grid_sample backward 的原生非确定性告警如实保留。AMP 使用兼容 PyTorch2.1.2 的 torch.cuda.amp API；本地2.7.1显示 deprecation 告警，不因此升级依赖。

早期尝试保存在本地 outputs：第一次揭示极小框 xyxy 面积需按端点计算，已修复实现；标量 oracle 的细长框用 y=0 避免把 FP32端点舍入误当算法差错，未放宽容差。后续修复三个测试夹具问题：直接构造模型需补原训练器设置的 nc；接口对照不能切换输入投影BN；优化器需按CPU加载后映射设备的流程比较状态。最终检查没有删 DN、关闭 AMP 或改变算法来通过。

操作工具检查：`check_rsc_head_ops.py` 使用模拟 dispatch/OOM/PID 状态和真实归档 IO，不创建真实 tmux 或训练。覆盖无完整预检标记直接调度、重复启动拒绝、OOM非零退出且batch不变、失联/缺产物状态、超过20MiB归档完整读回、拒绝覆盖、缺失材料不标完整。Bash语法通过。具体结果见 ops_checks.json；真实本地 Git 集成测试共6个场景通过：远程前进后固定旧SHA、同一干净SHA重复同步、带修改拒绝、不同SHA拒绝、非worktree拒绝、错误origin拒绝。真实 detached worktree 的 verify_delivery 也通过，主库既有修改保持原样。仅将fetch目标定向到隔离bare仓库，没有伪造git对象或启动训练。记录见 sync_checks.json。

未执行：AutoDL4090/PyTorch2.1.2、正式batch16显存、完整训练/数据集val/test/吞吐延迟或精度提升验证。full_server_preflight=NOT_RUN。
