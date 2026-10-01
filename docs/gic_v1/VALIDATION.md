# 本地验证与未执行项（2026-10-01）

本地环境：Windows，Python3.9.25、torch2.7.1+cu118、NumPy2.0.2、RTX2060 6GB。服务器预期环境来自母版真实归档：Python3.10、torch2.1.2+cu121、NumPy1.26.4。此次没有连接服务器，服务器当前环境为 **NOT_RUN**；preflight会实查并记录，不安装/升级任何环境。

| 检查 | 状态与边界 |
|---|---|
| 数学/作用域/输出梯度/模型装配/保存重载 | PASS，8项，reports/local_tests.json |
| 配方/母版源码/评估协议/缺导出拒绝重推理/打包/状态/tmux脚本 | PASS，9项，reports/operations_tests.json；tmux实际Linux会话尚未运行 |
| 源初始化SHA、原构建次序及nc1适配 | PASS，真实本地公共源；完整key报告见reports/initialization.json、nc1_loading.json |
| 固定真实train B16/640 FP32 | PASS，loss=91.32498931884766，有限反向和1次更新 |
| AMP，原默认GradScaler初始65536 | 初次尝试失败：前向有限，反向缩放后梯度非有限；原始FAIL记录保留在reports/initial_smoke.json |
| AMP单位缩放数值接线 | PASS，loss=91.37728881835938，有限反向和1次更新；reports/amp_unit_scale.json |
| 母版val固定前64张已导出证据 | PASS，纯离线；reports/mother_val64.json。并非新模型AP评估 |
| 服务器900秒preflight | NOT_RUN，本地通过不替代目标环境检查 |
| 正式200轮训练、正式val/test及真实完整包 | NOT_RUN，用户之后在服务器明确启动 |

实际本地共3次真实批次前后向尝试、2次有效optimizer更新：FP32一次、默认AMP一次失败、仅补做AMP单位缩放一次。没有重复FP32、自动扩展步数、缩批、关闭正式AMP或改变eta/shift。预检入口最终版本固定最多两次尝试（FP32/AMP各一次），两个临时模型分别重新建立公共初始化；正式训练会在新进程恢复原随机种子、原optimizer、原GradScaler、BN和权重。

本地技术冒烟使用固定16张train图像的原stretch/RGB/255预处理，不施加在线随机增强；这用于检查精度、容量与接线，不替代完整增强配方训练。两次有效更新均为临时模型，未保存为正式训练权重。单位缩放通过支持“初次非有限梯度与缩放倍率相关”的判断，不能把原倍率失败改写成原生AMP训练已通过。

固定64张val证据中，区间非零194/213、差量非零168/213、预计梯度减弱168/213。原梯度近零忽略0个。只能验证机制覆盖情况；涨点、泛化、收敛速度均未验证。官方母版FP32 mAP50–95参考val=52.454272%、test=52.200902%，不与训练时AMP验证峰值混用。

独立运行：

```bash
python tools/check_gic_v1.py --report outputs/gic_v1/unit_tests.json
python tools/check_gic_v1_ops.py --report outputs/gic_v1/operations_tests.json
```

测试不新增训练框架。真实Linux tmux、服务器环境、全量正式评估与包流程仍需要服务器实际运行；本地fixture测试不宣称已经验证这些外部状态。
