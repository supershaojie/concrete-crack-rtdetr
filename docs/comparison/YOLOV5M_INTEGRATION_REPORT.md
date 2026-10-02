# YOLOv5m 接入报告

2026-10-02（北京时间）。**代码接入完成，尚未启动正式训练或完整 val/test 推理。**

## 身份与范围

分支 bench/yolov5m，从公共固定提交 529c456b9404f1d9ab66d82d2b2f9ec7e0c98545 派生。具体交付 SHA 以本报告所属提交和推送回执为准。原主工作树及其他创新分支未改动；本轮无 PR、force push 或公共分支回写。

上游官方 v7.0 已解析并检出到 915bbf294bb74c859f0b41f1c23bc395014ea679。固定 release 的 yolov5m.pt 实际下载 42,806,829 bytes，SHA256：

61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56

来源：https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt 。已验证 depth=.67、width=.75、三层三 anchor 的原始 Detect，COCO 80 类官方方式适配 crack 1 类，475/481 个 state_dict 项加载，六项不匹配均是三尺度预测卷积的 weight/bias。没有调用新版 ultralytics.YOLO，也没有使用 yolov5mu。版本与资产校验在 upstream.lock.json；上游 LICENSE 保留，修改以 GPL-3.0 patch 交付。

## 文件与入口

| 文件/目录 | 作用 |
|---|---|
| benchmarks/comparison/yolov5m/assets.py、upstream.lock.json、upstream.patch、NOTICE.md | 下载/正确资产复用、固定身份、可重放补丁与许可 |
| recipe.json、hyp.yaml、requirements-server.txt | 冻结完整配方与独立环境依赖约定 |
| support.py、data.py | 哈希/环境身份、实际列表标签轻量检查、公共 GT 复用或图片头宽高转换 |
| worker.py、bench_yolov5_runtime.py | 隔离上游导入、训练、实际增强关闭、预训练/anchor/epoch 记录、FP32 导出、资源测量 |
| run.py | prepare/train/export/evaluate/pipeline/summary/resources；明确恢复、锁、日志与退出码 |
| validation.py、test_control.py | 合成模型/worker 切换/坐标/公共指标验证；失败、中断状态及缓存身份测试 |
| scripts/autodl_yolov5m.sh | AutoDL 新 run 与独立 tmux 顺序执行，保留完成 pane |
| docs/comparison/YOLOV5M_SERVER_COMMANDS.md | 实际可用参数、环境和服务器命令 |
| docs/comparison/YOLOV5M_AUGMENTATION.md | 母版证据、增强映射、调度和数值例外 |

公共改动仅更新 model_registry.yaml 的第一模型状态及定位字段；公共评测函数与母版模型/损失均未改动，其余七模型仍只登记。

## 冻结配方与语义

正式训练 640 / batch16 / workers8 / seed42 / max epochs200 / patience50 / AMP / deterministic；SGD lr0=.01、lrf=.1、momentum=.937、weight_decay=.0005、余弦、warmup5。nbs64 表示名义累积基准，实际正常累积4步，不是 batch64。完整展开参数和实际缩放/调度运行时均落盘。

best 和 patience 改用纯训练验证 mAP50–95，保留原生 CSV；同一 best 用于两组最终评测。HSV 改为母版加法 hue，MixUp 改为独立总体概率 .05；第191轮前重建 worker iterator，关闭 Mosaic/MixUp 而保留几何/HSV/翻转。明确禁用历史未证实的隐式 Albumentations。保留 YOLO letterbox、Mosaic 全 train 伙伴抽样及中间框过滤等差异，不声称与 RT-DETR 像素级相同。AutoAnchor 仅读取 train，正式数据 anchors 是否变化等待训练时实测。

最终导出 FP32、640、batch16、workers0、conf=.001、class-aware NMS IoU=.7、max_det300；关闭会静默漏图的 NMS 时间上限。原图坐标按实际 xy resize gain/整数 padding 逆变换，不舍入或额外裁剪，YOLO0→公共1，保留每张图和空预测。统一指标完全复用 corrected_sorted_conf_mask_v1；P/R 是公共最大 F1 工作点，单位0–1。

## 已完成验证

- 轻量核对本地实际目录：train 6048/45573、val 1728/12840、test 864/6663（图片/框）；未读取全部图片内容做 SHA/解码或重扫历史权重。
- 已有公共 val/test COCO 与其小 manifest，经当前路径/稳定 ID/标签字节/图像大小及标注核对成功复用，未重跑正式推理。
- 官方 checkpoint 单类结构/导入隔离验证；两张合成图 64×64 CPU 前向、官方损失、一次反向均通过。
- 真实双 worker InfiniteDataLoader：index189 的 batch 实际执行两次 Mosaic 预变换，index190 的所有 batch 均不执行 Mosaic/MixUp，仍执行 HSV 和 geometry；也验证非 Mosaic 分支会执行 MixUp。
- 非方形奇数宽高的 letterbox 往返最大误差约 1.67e-6 像素；已知框/空预测通过公共 schema 和 evaluator，人工样例 AP=.995 属测试值，不是模型结果。
- 额外仅两张合成训练图/两张合成验证图、64 像素、CPU、1 epoch 的真实训练器检查通过，正常保存未 strip 的 best/last 和实际 epoch 结束记录。临时 Windows 检查脚本 main 保护问题已修正，正式 worker 本身有保护。
- 生产导出器另用两张合成图在 640 FP32 CPU 上联通公共评测，仅替换设备选择；没有使用正式 val/test。
- 复用公共五项测试，新增三项控制/轻量 GT 测试均通过；CLI/help、Python 编译、Bash 语法、全量最终配置打印/冻结校验通过。
- 干净官方提交重放 patch 后，六个修改文件逐个相同。实际脚本用退出0/7的假解释器验证 pipeline/tee 真实退出码与重复日志保护。
- 中断状态的 Python 单测与 Git Bash SIGTERM 回执均核实为 interrupted。Git Bash 中信号组子进程的清理行为未通过 Linux 等价验证，不能宣称 AutoDL 信号清理已测通；服务器入口使用其自身独立进程组、重复信号和有界终止，未操作其他实验。

紧凑实测记录在 evidence/yolov5m_validation.json；原始大输出留在本工作树 outputs/yolov5m/ 并被忽略。本地 smoke 依赖在独立 venv 中补齐，复用已有 torch，没有升级原 rtdetr 环境。

## 服务器待执行

AutoDL 当前依赖/GPU 驱动、真实 tmux/信号清理、CUDA AMP 与实际 batch16 显存、首次正常标签缓存/AutoAnchor、正式训练及完整 val/test 尚未执行。断点恢复已实现版本/配置/环境/数据/checkpoint/epoch 核对与 patience 状态恢复，但没有模拟完整多轮 GPU 中断恢复。资源测量仅交付入口，未提供论文独占硬件速度结论。

服务器 Python 训练进程的真实 CUDA/worker 行为以运行回执为准。继续使用既有固定划分，其原图族事实保留于公共报告；本轮未重新划分、离线增强或改变标签。下一模型 YOLOv8m 未接入。
