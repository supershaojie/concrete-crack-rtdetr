# 规范对应与交付范围

|规范内容|实现或证据|
|---|---|
|先附件/仓库/旧组合审计，再生产改动（0–5、33、55–57）|compatibility_audit.md/json；附件与nested source哈希；C20/C25/C26原指标；新worktree从C2创建；用户原目录保留|
|整体 stable reference / main path / firewall（6–7）|ARCHITECTURE.md；triad_compat.py语义图；真实hook和非零路由干预|
|CSCEF-v6机制与位置（8–12）|cscef_v6.py继承原v51内核；refs detach、PAN后P3 sidecar；26,912参数|
|SCCA-v2机制与原forward（13–15）|scca_aifi_v2.py；同参数v1/v2 pre/post完全相同；65,540参数|
|CBR-v2稳定ref、条件隔离、rho（16–20、30）|cbr_v2.py；四输入三core；36点；rho=.075；位移w/h条件额外隔离；45,889参数|
|7 YAML与参数量（21–24、38）|7份独立YAML；parameter_topology_report.json；统一variant registry|
|公共权重/原生分类适配/zero init（25–27）|init_triad_compat.py；7份evidence/*_initialization.json；533公共state全部精确；9分类适配；全部嵌套输出误差0|
|真实梯度、防火墙、诊断（28–29）|check_triad_compat_gradients.py；gradient_firewall.json；diagnose_triad_compat.py；diagnostics.json|
|parser明确新类、旧类回归（31–32、46–47）|仅registry/parser+可选final_query接口；原head/loss/DN/trainer未改；21项模块/路由/流程测试|
|分支策略与不改配方（33–35）|codex/rtdetr-triad-compat；109字段逐类型比较；model/name/save_dir之外不变|
|固定评估，不用test搜索（36–37）|triad_compat_results.py；既有corrected-mask独立入口、seed42/workers0；val同checkpoint后才test；无rho/阈值扫描|
|AutoDL工具与保护（39–45）|autodl/sync/train/results工具；独立tmux/run/lock/token；强制原环境预检；状态/包完整性验证|
|本地构建、shape、DN、AMP/half、save/load、optimizer（46–47）|7份validation；FP32与CUDA AMP各2步synthetic；DN500→498；全参数覆盖；非零重载误差0|
|不加入其他模块/外部依赖（48–50）|仅原C17/C19/C24机制与新版本；模块ZIP只用于机制/工程参考；不依赖该ZIP运行|
|后续分阶段正式实验，不能预报涨点（51–54、61）|AUTODL.md；架构只是待验证候选；未训练、未宣称组合收益或绘制最终论文图|
|源码追溯、commit普通push、固定SHA命令（55–60、62）|source_files_sha256.json与FILES.md；最终回复给出提交完整SHA和远端核验；git提交对象是自身版本权威标识，不在提交内伪造自引用SHA|

可执行微型评估脚本只处理原split各两张复制图。pack验证中的训练状态是明确标记的synthetic夹具；它不能作为训练成功或模型效果证据。所有正式动作仍由用户后续按AUTODL命令触发。
