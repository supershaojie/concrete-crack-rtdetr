# GEO v1 实施交付

本次已实现固定 `GEO-v1-envelope-K3`、必要本地验证、独立分支提交和 push。
未启动正式200轮训练或 test，未进行消融，不声称 AP 涨点。

## 代码身份

- 实际远端：`https://github.com/supershaojie/concrete-crack-rtdetr.git`。
- 实际母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
- 分支：`exp-rtdetr-r18-lite-geo-v1`。
- 功能代码 SHA：`3bc172c6f22470a00abb2ab6a434e22663406e78`，已正常 push；
  服务器训练命令固定此40位 SHA，后续纯文档提交单独保留，不伪造自引用 hash。
- 本地独立 worktree：`C:/Users/o'v'o/.codex/worktrees/geo-v1/Crack_RTDETR`。
- 参考可靠性提交：`ef9cb7e05e5557f7dd06c95cf2361998a284adc9`，只采用非算法可靠性模式。
  GEO 分支直接从母版建立，没有合并 ARG 或其他实验。
- 适用祖先 AGENTS.md 为空、仓库无额外 AGENTS；原主工作区未提交产物和其他 worktree 保留。

详细固定公式、梯度边界、接线路由和创新边界见 [README.md](README.md)。
可逐段复制的实际服务器命令见 [server_commands.md](server_commands.md)。

## 修改文件和接入证据

新算法入口：`ultralytics-main/ultralytics/models/rtdetr/geo_loss.py`。
可导入模型/真实 Trainer：`geo_model.py`；独立 FP32 评估/同次导出：`geo_val.py`。
命令/身份/有界检查/评估打包位于 `tools/geo_v1.py`、`geo_v1_common.py`、
`geo_v1_preflight.py`、`geo_v1_results.py`；shell 入口为 `geo_v1.sh`、`sync_geo_v1.sh`。
检查入口为 `check_geo_v1.py`、`check_geo_v1_ops.py`。

共享代码仅修复 AutoBackend warmup 的有限零输入，以及已存在本地权重路径保留单引号。
原 `nn/tasks.py`、`models/utils/loss.py`、`models/rtdetr/train.py`、训练期 validator、
母版模型 YAML、LIF-Down 和 CBR 保持原源码。没有新增部署 tensor 或推理算子。

核对结果：3层 decoder、300普通 queries、CBR36点/rho=.10；未融合 20,149,765 参数、
552个 state_dict 键。实际参数名称/形状/数量表见 [model_inventory.json](model_inventory.json)。
LIF/CBR 的 LF hash 与提示词参考值一致；同权重构建 RNG、状态值、融合前后推理分别一致。

原 criterion 的 VFL(.25/1.5)、matcher(.25/2.0)、代价2/5/2及全部 L0 项保持。
GEO 只复用最终普通层匹配，原模型先完成普通/DN拆分；所有原层与 DN 损失完成后才加入一次
loss_geo，finally 清理上下文。原实际优化总和已验证包含该项。关闭/ramp0 不执行新几何。
resume/懒创建均同步真实 epoch；val/inference 关闭 GEO；训练期 fitness 与 best 规则不变。

完整109项配方见 [formal_args.yaml](formal_args.yaml)，逐字段比较见
[recipe_diff.json](recipe_diff.json)。服务器实际只改变 model/name/save_dir，106项保持；
正式epochs200/patience50/B16/640/AMP/AdamW及所有原增强均未改变。
公共源 hash、nc80→nc1原生543项精确加载/9项分类适配已经本地核验。
正式 start 不读取任何诊断 checkpoint 或训练状态。

## 真实检查结果

机器为 Windows、Python3.9.25、torch2.7.1+cu118、RTX2060 6GiB。
服务器母版环境与实时路径本地无法核验，命令会明确检查。
完整证据及源码 LF 指纹见 [local_validation.json](local_validation.json)。

|范围|结果|边界|
|---|---|---|
|数学、FP64面积等价/有限差分、边梯度、top-k并列/零项/M分母|PASS|固定样例 raw=.2、weighted=.04，直接边梯度正确|
|独立输出的普通/aux/encoder/DN/logits/未匹配梯度隔离|PASS|同输出所有L0逐项保持；最终matcher调用不增加|
|微小CUDA autocast几何|PASS|输出GEO为FP32且梯度有限；不是B16容量证据|
|真实原生模型重建、关闭/ramp0前向/损失/梯度、推理/融合一致|PASS|合成B2/160 CPU；单线程保证embedding累加逐位可比|
|真实增强batch＋原生AdamW更新|PASS|仅本地B2/160 CPU，1个micro-batch；345组梯度有限、CBR参数真实改变|
|原生checkpoint新进程resume|PASS|start_epoch20、ramp1、optimizer/EMA/scaler、scheduler.last_epoch19恢复；CPU scaler为空状态|
|FP32零输入warmup＋真实一批val＋同次完整query/GT导出|PASS|本地B1/640，1图、300queries、1GT；不是全划分评估|
|身份/锁恢复/离线统计/INCOMPLETE打包等操作检查|PASS|8个临时fixture；不运行正式训练/test|
|CLI主入口/全部子命令新进程help及shell语法|PASS|19项检查，含preflight/finish/pack|
|提交后真实prepare|PASS|复用已确认数据清单，公共初始化/完整配方/源码身份绑定|
|真实pack入口与压缩包逐文件hash核验|PASS|明确输出INCOMPLETE并列出未训练/未评估缺项；无权重本体、无原数据扫描或推理|
|本地调用有界preflight入口|PENDING，符合预期|Windows不能冒充原Linux环境B16/640 AMP资格|
|Linux正式AMP/B16有效更新、真实tmux/full-run resume|PENDING|服务器执行900秒/最多16 micro-batch命令，原GradScaler，无降scale备用方案|
|正式长训/全量FP32 val/test/完整训练结果包/涨点|PENDING|本次未执行，由用户按命令启动与finish|
|DDP|UNVERIFIED|本次仅单GPU0，无新增分布式补偿|

数学/路由共9项，操作fixture共8项。早期检查夹具问题及Windows路径加载错误均保留在
local_validation 的说明中，修复后重验；没有靠改GEO公式、换原损失或改正式超参凑PASS。

本地复查命令（从该独立worktree执行）：

```powershell
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_geo_v1.py --math-only --output outputs/geo_v1/recheck_math
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_geo_v1_ops.py --output outputs/geo_v1/recheck_ops.json
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/geo_v1.py status
```

真实完整模型和生命周期诊断的执行参数、路径、源hash、数值和作用域均在local_validation中。
各命令本身不存在隐式start/test。若需要重跑本地完整模型检查，应使用新的output目录以保留旧证据。

## 训练完成后的行为

原训练期val、mAP50-95 fitness和best保存规则保持；final_eval改为记录合法停止并保留完整
last训练状态。用户执行finish时才进行所需的正式FP32 val/test，并在首次推理同时导出
所有普通queries与GT、原query索引、图像身份/尺寸/坐标定义。两split使用同一个原生best。
结果锁绑定split/checkpoint/data snapshot/评估源码及协议；完整结果复用，缺锁先核对补锁，
部分/失败结果不写成功锁。离线汇总全部AP和曲线、各split原生最佳F1点、val选阈值下两split
TP/FP/FN与P/R/F1，不使用test选阈值，不重复推理补预测或曲线。

finish正常交付一个COMPLETE包；status/pack不推理、不重扫原数据、不重选best。
缺项包明确INCOMPLETE。包包含源码快照、初始化/数据/配方/环境、预检、机制/训练日志、
退出/恢复历史、正式指标曲线、完整预测GT、best/last身份和锁及逐文件hash。
默认不含原始数据和权重本体，manifest不自包含自身hash。完整真实训练结果包仍PENDING。

## 网络与交付

最初两次HTTPS远端查询遇Connection reset；SSH443通路经官方
[GitHub公钥指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)
核验，但本机publickey认证失败；没有关闭主机校验或修改全局SSH配置。
随后对原HTTPS origin的正常git push成功。保留这些真实网络失败记录，不将它们算作训练失败。
最终远端文档提交在交付后另行核对；功能代码与文档SHA区分，服务器命令固定上述功能SHA。
另附可用git fetch读取的代码/文档bundle作网络故障备用，不包含数据、权重、凭据或训练产物。
