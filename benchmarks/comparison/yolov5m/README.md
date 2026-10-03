# YOLOv5m COCO + b19 pilot

本分支的正式配置是 recipe.json/hyp.yaml，输出为 outputs/yolov5m-coco-b19-pilot/。

实现和验证说明见 [pilot 报告](../../../docs/comparison/YOLOV5M_COCO_B19_PILOT.md)，固定提交、独立环境、tmux、评测和恢复命令见 [服务器交付](../../../docs/comparison/YOLOV5M_COCO_B19_PILOT_SERVER_COMMANDS.md)。继承的 scratch/旧COCO 文档是历史归档，不是本分支的启动命令。

`worker.py smoke` 与 `test_coco_b19.py` 只用合成数据，不能作为正式训练或初始化入口。正式使用新命名的 scripts/autodl_yolov5m_coco_b19_pilot.sh；每个新 run 从经 hash 核验的官方 yolov5m.pt 创建。
