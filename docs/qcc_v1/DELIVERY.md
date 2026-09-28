# QCC v1 实施交付

交付结果：**QCC-v1-capK3 已实现，本地必要检查 PASS；正式服务器检查/训练/完整评估 PENDING。没有运行正式训练或 test，没有涨点结论。**

## 身份

| 项目 | 真实身份 |
|---|---|
| 仓库 origin | https://github.com/supershaojie/concrete-crack-rtdetr.git |
| 分支 | `exp-rtdetr-r18-lite-qcc-v1` |
| 实际起点 | `a0459d6a652cb702699087c88fa39a3e4c4087ec`（直接母版，不从 ARG 起训） |
| 算法、接线、评估/生命周期实现提交 | `44a3811721049c01096f3ee985ead8779264f2ea` |
| 最终代码提交（含 prepare 原子快照修复） | `d2b5a27d53278ac72c94fcfac2636f3d808b0cc8` |
| 参考可靠性工具提交 | `ef9cb7e05e5557f7dd06c95cf2361998a284adc9` |
| 本地独立工作区 | `C:/Users/o'v'o/.codex/worktrees/qcc-v1/Crack_RTDETR` |
| 公共初始化 SHA256 | `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e` |

后续交付提交仅添加本文件和 server_commands.md，不改算法或评估代码。最终答复另列实际推送后的交付 SHA；不把代码 SHA 冒充包含本文的提交，不伪造本文的自引用 SHA。服务器同步命令读取已 fetch 的交付 SHA，先证明其在最终代码提交之上且代码路径没有变化，再保存真实 SHA 到 `sync.json`。prepare/preflight/start 绑定实际 checkout SHA、源码哈希、配方、公共初始化和数据快照。

原工作区 `D:/MyProjects/Crack_RTDETR` 仍是用户原分支；未 reset/clean/强推/改 remote，未修改其他实验或历史。

## 修改与接线

| 文件 | 作用 |
|---|---|
| `ultralytics-main/ultralytics/models/rtdetr/qcc_loss.py` | 固定公式、stable top3、原生质量捕获、单份 weighted loss |
| `.../qcc_model.py` | 实际 get_model 绑定、真实 epoch/lazy criterion、原优化流程、有限采样、训练完成与 final_eval 分离 |
| `.../qcc_io.py` | 原子 JSON、严格有限值表示、计数/和/二阶矩聚合 |
| `.../qcc_val.py` | corrected sorted mask、首次完整 streaming 导出、完整性标记、全指标/曲线 |
| `ultralytics-main/ultralytics/nn/autobackend.py` | 唯一公共修改：warmup 改为全零有限输入 |
| `tools/qcc_v1.py` | prepare/preflight/status/start/resume/val/test/finish/pack 生命周期、离线分析 |
| `tools/qcc_v1_common.py` | 母版 109 字段配方、固定数据快照、公共初始化、代码绑定 |
| `tools/qcc_v1_preflight.py` | 总计最多 16 micro-batch / 900 秒，B16/640 AMP、有效更新和新进程恢复门槛 |
| `tools/qcc_v1.sh`, `tools/sync_qcc_v1.sh` | 固定 Python、导入路径、原仓库独立 worktree、有限 fetch |
| `tools/check_qcc_v1*.py` | 数学/梯度/原损失路由、真实模型、流程与新进程检查 |
| `docs/qcc_v1/` | 完整配方、差异、固定公式、原母版证据、本地结果和执行命令 |

路由为原 `RTDETRDetectionModel.loss` 的 DN 拆分 → 原 encoder/decoder 拼接 → `QCCDetectionLoss` 捕获最终普通层原匹配及 VFL 的真实质量 → 原 L0 的全部普通/aux/DN 完成 → 增加一次 `loss_qcc`。不把最终匹配传给整个 super.forward；不生成 `loss_qcc_dn` 或 auxiliary QCC。

QCC 仅对选中组的最终普通 logits 直接回传。q/框/GT/IoU/owner/topK/a/h 都 detached。严格 p<q 决定正例梯度，p=q 或 p>q 的直接梯度为零；q=0 跳过，q=1 不求 logit(1)。局部 FP32、稳定 logsumexp，无 arbitrary q clamp、STE、额外温度或可学习项。作用域 finally 清理，disabled/r=0 无额外几何计算。完整公式及迁移来源表在 README.md。

## 原母版与配方证据

- LIF LF SHA256：`26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`。
- CBR LF SHA256：`d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787`。
- 原 3 decoder 层、300 普通 queries、CBR 36 点/rho=0.10；未融合 nc=1 参数 20,149,765、state_dict 552 tensor keys，与原母版相同。
- VFL 实例 alpha=0.25、gamma=1.5；matcher alpha=0.25、gamma=2.0，class/bbox/giou=2/5/2；原 L0 class=1、bbox=5、giou=2，aux/encoder/DN 保留。
- 原生 nc=80→1 映射 543/552 个状态精确加载，9 个原生分类适配键。公共 source 身份和零初始化 LIF O / CBR offset 的原流程通过初始化工具核验。
- `formal_args.yaml` 是完整 109 字段模板；同时比较 `c2_args.yaml` 与母版 `resolved_formal_config.yaml` 的全部非身份字段。服务器模板只改 model/name/save_dir，实际 prepare 路径差异受限于 model/data/project/name/save_dir。训练/增强/AMP/累积/预算未改。
- native loss/matcher/tasks/train/CBR/LIF 源文件无修改；无其他第三创新的 loss/model/state 引入。
- 同状态 inference 与 head export 模式逐张量相同、参数名称/形状与 state keys 相同。未生成实际 ONNX/TensorRT 工件，未声称该导出后端已测试。

## 已完成检查

详见 `local_validation.json` 与 `native_contract.json`。本机 Python 3.9.25、torch 2.7.1+cu118/CUDA 11.8、RTX 2060 6GB，实际 ultralytics 来自本 QCC 工作区（8.4.21），未升级环境。

| 检查 | 实际结论 |
|---|---|
| 11 个小张量/公式/路由测试 | PASS：含参考 ell=0.7804489351881119、给定梯度、p=q 零梯度、q=0/1、极端 logits、窄框、stable tie、图像隔离、全 GT owner、排除其他正例与更优定位候选、无 GT/unmatched/候选、FP32 autocast、按全部 M 归一化、冻结选择的 FP64 gradcheck |
| 原损失与匹配接线 | PASS：disabled/r=0 原损失和独立输出梯度逐项相同；开启只多一份 QCC；Hungarian 调用次数不增加；DN/aux 规则保持；质量就是原 VFL 输入 |
| 真实模型 B2/160 CPU | PASS：恢复 RNG 后原训练 forward/loss/所有参数梯度/一次原生 AdamW 更新精确一致；CBR 后最终普通框的真实路由、推理/export-mode/state 集合一致 |
| 实际 Trainer._setup_train | PASS：本地诊断 B2/160/workers0，公共 nc80 init 重建 nc1；原生 AMP 检查明确 checks passed。不是 B16 容量结论 |
| 原生 save_model → 新进程 resume | PASS：epoch=19→start_epoch=20、原 optimizer/EMA 状态和 CPU disabled scaler 恢复；CUDA scaler 的正式验证仍待服务器 |
| 新进程 AutoBackend/fuse/FP32 warmup + 真实 val | PASS：全零有限 warmup，B1/640 一图 val，导出全部 300 query/logits/GT。只是诊断，不是正式 full val |
| 7 个流程夹具检查 | PASS：类别键冲突/规范化、清单复用、binding 不 inventory、status/pack 不推理、同锁 val/test 重入复用、缺导出要求显式恢复、中断不写 complete |
| Python 编译、bash -n、git diff --check | PASS |

本地检查命令（均在本 QCC checkout 执行）：

```powershell
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_qcc_v1.py --output 'D:\MyProjects\Crack_RTDETR\outputs\qcc_v1_local\checks'
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_qcc_v1.py --math-only --output 'D:\MyProjects\Crack_RTDETR\outputs\qcc_v1_local\math_final'
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_qcc_v1_ops.py
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_qcc_v1_lifecycle.py --checkpoint 'D:\MyProjects\Crack_RTDETR\outputs\qcc_v1_local\checks\native_save\last.pt' --output 'D:\MyProjects\Crack_RTDETR\outputs\qcc_v1_local\lifecycle'
& 'D:\miniconda3\envs\rtdetr\python.exe' tools/check_qcc_v1_lifecycle.py --setup-only --checkpoint 'D:\MyProjects\Crack_RTDETR\outputs\qcc_v1_local\checks\public_init.pt' --output 'D:\MyProjects\Crack_RTDETR\outputs\qcc_v1_local\setup'
```

以上最后一个命令只初始化 Trainer，不调用 train()。诊断输出目录已经使用过，导出文件默认不覆盖；再次生命周期诊断应选择新的诊断目录。Windows 原生 asset loader 会移除权重路径的单引号，因此诊断权重使用 D 盘输出路径；没有为此改模型或原 loader。CPU bitwise 梯度检查用一个线程消除原 embedding/scatter 并行求和次序差异。

## 待服务器完成

PENDING：真实服务器路径/解释器/原配置和数据快照核验；B16/640 原 AMP 有效更新、scale/overflow/显存/时间；完整 native CUDA optimizer/scaler/epoch 恢复和 B16 真实 val；正式从公共初始化/e=0 开始的训练；完整 FP32 val/test 和最终分析包。DDP 未验证，非本轮单 GPU 门槛。历史 ARG 的诊断没有冒充本实验 PASS。

preflight 严格有界，不启动长训。native scale 与独立低 scale 诊断分列；fallback 只能按 README 记载的参考门槛资格判断，不能写成 native AMP 长训已通过。

## 一次导出/恢复/打包

首次正式 val/test 即保存完整 queries/GT 和实际评估材料，成功后原子加锁。训练 final_eval 采用同一 formal FP32 val 流程，finish 可直接复用；原 epoch AMP val 保持母版行为，只用于原 fitness/best 选择。finish 只补缺失的 val/test，然后离线生成 val 锁定阈值的精确 TP/FP/FN/P/R/F1，产出一个分析包。再次 finish 复用已完成阶段和已有包。

训练完成证据在原 stopper 决策且 checkpoint 保存后落盘；worker 的启动/结束/真实 Python 退出码与 final_eval 独立记录。训练完成后的评估失败使用 val/finish 恢复，不 resume 训练。strip 后 checkpoint 不再含完整 optimizer/scaler，工具明确拒绝训练恢复。原错误不覆盖，恢复事件写新目录。

status/pack 只读已有产物。包含全部已存导出、日志、指标/曲线、完整配置、源码快照、身份/锁、预检和 manifest；权重本体及原图全集排除，但保留权重 SHA/大小/路径/选择 epoch。缺材料只能标记 INCOMPLETE；成功锁缺导出不能偷偷补推理。

完整服务器命令见 [server_commands.md](server_commands.md)。
