# YOLO 随机初始化对比交付

日期：2026-10-03（北京时间）。两组代码及 CPU/CUDA 功能验证完成，尚未进行服务器正式训练。

| 模型 | 新分支 | 实际父提交 | 已验证代码 SHA | 参数量（nc=1，未融合） |
|---|---|---|---|---|
| YOLOv5m | bench/yolov5m-scratch | a4dcaa0b0b0c9356923c266ba351de67e033c540 | ea17605d538f28da5fa502ddaaf130f7a9ea8084 | 20,871,318 |
| YOLOv8m | bench/yolov8m-scratch | 74bc7d7168d4fdc199a54d1dc415c497dd0ddfce | 61c386bc722b11208453cab0808d8f6edb15385d | 25,856,899 |

远端原分支没有锚点后的新提交，原 worktree 没有未提交修改；分别直接派生，完整继承之前的修复。公共准备提交仍为 529c456b9404f1d9ab66d82d2b2f9ec7e0c98545，组合母版仍为 a0459d6a652cb702699087c88fa39a3e4c4087ec。文档后续提交不改变上述训练代码，服务器命令锁定表中的代码 SHA。

正式条件均为 epochs=200、patience=50、imgsz=640、batch=16、workers=8、seed=42、device=0、AMP=True、deterministic=True；SGD、nbs64、warmup5、cos_lr=True、close_mosaic10。YOLOv5 lrf=0.1，YOLOv8 lrf=0.01；其余优化器、损失、增强、锚框、EMA、原生验证设置保持原冻结实现。以原生 val 的完整精度 mAP50–95 选 best，相等值沿用原来的最新轮规则；test 不参与选模或早停。

新 YOLO=random、无外部预训练；旧 YOLO=COCO；既有 RT-DETR=ImageNet backbone，旧结果保持原样。可以比较这些具体训练设置下的表现并补充初始化敏感性研究，不得把整表描述成“全部从零训练”或“预训练条件相同”。不预设结果高低，不筛除真实指标。

本机已完成 CPU 检查及 RTX 2060 6GB 的合成 CUDA smoke：batch2、64像素、两轮，第一轮后中断，再用同一 scratch 检查点续训到第二轮，随后用真实 scratch best 做两张/每 split 的生产 FP32 640 导出与公共评测。Smoke 不进入正式训练。尚未运行 AutoDL 正式训练、完整数据 val/test 或 batch16/640 的双模型显存容量测试；Windows 上未实测真实 tmux/POSIX 信号。本次没有 SSH 或服务器操作。

## 实现与验证

- YOLOv5：weights 为空，真实官方 M YAML，原生三尺度 anchor-based Detect。隔离实际模型 AMP 检查；本地检查点路径保留单引号；检查点原子写入、UUID/初始化/数据/代码身份保护；二进制日志保留回车与真实退出码。
- YOLOv8：显式 random 模式，官方 YAML 字典指定 m，get_model(weights=None)，pretrained=False，保留原 COCO 验证能力和固定 DFL。bootstrap/预检/训练/恢复/导出身份一致。
- 两组均在禁止 torch.load 与 Module.load_state_dict 的条件下完成初始化和首轮原生训练；AMP 前后模型、BN、EMA、优化器、Python/NumPy/CPU/CUDA RNG 保持不变；重复设定同一 seed 后重建一致。
- 两组原生 CUDA smoke 完成 1 轮→中断→恢复至第2轮；旧 COCO 身份、另一 scratch UUID 被拒绝。实际生成的 best 完成 FP32 640 生产导出及公共评测，val/test 各2张合成图。
- YOLOv5 控制测试5项、handoff3项、公共评测5项通过，CPU增强 worker 边界 smoke、官方补丁干净重放通过。YOLOv8 生命周期5项、handoff3项、初始化身份3项、既有集成10项通过；1项 Linux POSIX 信号测试在 Windows 上明确跳过。
- 数据清单与原记录一致，未执行整库内容哈希。两组正式配方除初始化相关字段外与父提交逐项相等；损失/分配器/检测头/EMA/选模规则未替换。

## 服务器执行顺序

先按任一命令文档创建 scratch worktree，inspect 并定向 stop 旧 YOLOv5m、YOLOv8m；helper 保全已核验的完整检查点和真实轮次。然后分别准备两个环境和数据清单，启动两个独立 scratch tmux 会话，即可并行使用 GPU0。不会自动杀旧会话，也不会停止其他实验；yolov5m-view 仅是查看日志。

完整命令：

- [YOLOv5m scratch](https://github.com/supershaojie/concrete-crack-rtdetr/blob/bench/yolov5m-scratch/docs/comparison/YOLOV5M_SCRATCH_SERVER_COMMANDS.md)
- [YOLOv8m scratch](https://github.com/supershaojie/concrete-crack-rtdetr/blob/bench/yolov8m-scratch/docs/comparison/YOLOV8M_SCRATCH_SERVER_COMMANDS.md)

证据在各分支 docs/comparison/evidence/yolov5m_scratch_validation.json 与 yolov8m_scratch_validation.json。原始本地 smoke 和失败诊断日志保留在各 worktree 的 outputs/yolov5m-scratch、outputs/yolov8m-scratch；失败尝试未覆盖。

原训练分支、RT-DETR 源码、数据集和既有结果未覆盖；未修改 Tunnel_Disease_YOLO26，未合并主分支、未强推、未登录服务器。
