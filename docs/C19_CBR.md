# C19：C2 RT-DETR-R18-Lite + CBR 实施与交付

CBR 是待验证的框边细化候选。当前交付不包含正式 200 轮训练或涨点结论，不代表论文新颖性已成立。C19 不加入 CSCEF，不继续 GSDR-AIFI-v4；C17 的 CSCEF、C18 的代码和结果留在原分支/worktree。未来 C20 的组合收益需单独验证。

## 实际基础与参考核对

- 新分支 `exp-rtdetr-r18-lite-cbr`，本地 worktree `D:/MyProjects/Crack_RTDETR/outputs/worktrees/cbr`，直接基于 C2 环境记录提交 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。原工作区未提交内容未动。
- 完整阅读 `D:/rtdetr跑结果/c19 CBR/C19_CBR_Codex_brief.md`。
- 实际读取本地目录 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-main`，并打开同目录 `RTDETR-20260623.zip`，逐文件核对以下内容与 ZIP 字节完全相同。路径可用，因此没有使用或声称读取未知附件落点。
- `ultralytics/nn/extra_modules/block.py:3413`，DySample：参考 `2*u-1`、像素中心、`align_corners=False`、`padding_mode="border"` 坐标约定；没有复制上采样头、自由 offset、pixel shuffle 或整块代码。
- `ultralytics/nn/extra_modules/FreqFusion.py:241`，LocalSimGuidedSampler：查看局部引导、offset 与 sample 实现；没有移植整套 FreqFusion、CARAFE/mmcv 或自由采样偏移。
- `ultralytics/nn/extra_modules/LPRM.py:31`：确认无 mmcv 时插值回退忽略 mask；没有采用，也没有把回退当成等价动态重组。
- `ultralytics/__init__.py`：模块包 8.0.201；当前项目和 C18 环境 8.4.21。无依赖升级、降级或包覆盖。新代码沿用项目 AGPL-3.0 标识，参考事实和哈希见 `cbr_source_trace.json`。
- ZIP 文件名检索未找到 SABL、BorderDet、D-FINE/dfine；这只是本包检索范围，不能等同于全部实现检索。
- 实际读取 C18 小包 `D:/rtdetr跑结果/c18 r18-lite-gsdr-aifi-v3/c18_diagnostic_20260906_222023_846012466.tar.gz`。记录的训练/评估提交为 `10e72ebbbf3af1808c3bfbe9cb10ea9fe2483529`。包中确有 transformer.py；没有 head.py、tasks.py、loss.py 的源码副本。其审计中的 head.py、transformer.py、loss.py 哈希与 C2 核心一致；tasks.py 结合本地 C18 worktree 的导入/注册差异核对。不能声称这些缺失源码均由压缩包直接读取。
- C2 的完整 109 字段 `args.yaml` 从该包 `references/c2/args.yaml` 获取，作为校验 fixture；服务器仍须读取原 C2 run 的权威文件。C18 独立 test mAP50–95=40.5490%、AP75=37.9655%，与实施说明一致。C2 历史 test 数字仅按实施说明引用；本次未重算历史性能。

## 网络与梯度路径

新增 `ultralytics/nn/modules/cbr.py`：`RTDETRDecoderCBR` 继承原 head，保留全部公共参数路径。新增 YAML 仅将最后的 head 类替换为 CBR 子类；三层 Decoder、300 常规查询、原始 AIFI 与 P3/P4/P5 连接均保留。当前 YAML 的 from=[19,22,25] 对应 Neck 的 P3/8、P4/16、P5/32。模块读 **head 输入列表的第 0 项**，没有把 19 硬编码到实现；将来改连接时必须保持 P3/P4/P5 顺序。640 输入的三尺度为 80²、40²、20²。

`DeformableTransformerDecoder.forward` 仅增加可选 `return_final_query=False`；开启时返回对应末次 append 的 query。默认仍返回二元组。原 `last_refined_bbox` 与 `refer_bbox.detach()` 的更新语句原样保留。没有 forward hook 或隐式激活缓存。模型 `predict(..., cbr_diagnostics=True)` 是诊断专用的显式返回；普通训练/推理结构不变。head.py 原文件未修改。

P3 1×1 Conv→64，query Linear→64；四边按沿边 1/4、1/2、3/4 各取内、边、外三点，共36点。L/R 的法向间隔为 w×0.1，T/B 为 h×0.1，边顺序 L,R,T,B，采样维内、边、外。局部证据 `[edge, inside-outside]` 保留符号，线性投影后与 query、边 embedding、宽高投影相加，SiLU 后打分并沿三个位置 softmax。score 无 bias，避免无效常数；query 在非线性之前进入，确实能改变聚合权重。共享64维 MLP 输出四边 t，末线性 weight/bias 零初始化。

残差严格使用实施说明公式：`rho=.1`，位移为 `rho*[w,w,h,h]*tanh(t)`；直接加到原始 xywh，几何和残差算术 FP32。对正宽高下界为原始0.8倍。采样几何和宽高条件 detach；P3、query、原始框主路径保留梯度。只替换最终返回框，不送回先前层 reference。训练末层全部常规/DN查询同样处理，顺序和 dn_meta 不变。

采样超出图像时复制边缘特征；输出框没有额外 clamp、epsilon 或坐标往返。原 RT-DETR validator 的坐标处理原样保留，画图裁剪不等于预测框裁剪。这里的边界是水平检测框，不是裂缝像素轮廓。

grid/双线性采样局部关闭 autocast、FP32 输入，随后恢复投影层合理 dtype；聚合权重与残差几何 FP32，其他网络保留原 AMP。显式 half 只作推理验证。新增45,889参数；nc=1 总20,128,661，C2总20,082,772。

原 Hungarian 与 RT-DETR loss 源码未修改。源码 `models/utils/loss.py` 的实际 gain：class=1、bbox(L1)=5、giou=2、no_object=0.1（另有 mask/dice=1）；matcher cost class=2、bbox=5、giou=2。不能用通用 args 的 box=7.5、cls=0.5 替代这些值。早期层/encoder 辅助框与分类头不新增分支；匹配和质量目标仍会因新末层框变化。空 GT 时原框损失不参与，CBR 梯度为空是预期行为。

## 工具与公平性

| 文件 | 行为 |
|---|---|
| `tools/init_rtdetr_r18_lite_cbr_controlled.py` | SHA 锁定 C2 原始 ImageNet 初始化，构建FP32新文件，公共状态/随机数/重载严格检查，拒绝覆盖和已训练源 |
| `tools/audit_rtdetr_r18_lite_cbr.py` | nc=1/80、完整640逐位等价、真实 train API/optimizer、精度/梯度、回归、耗时与可选真实数据 smoke；失败写具体报告 |
| `tools/smoke_rtdetr_r18_lite_cbr.py` | 原 trainer 构建与增强加载，3次真实优化、DN和重载；独立目录，debug checkpoint epoch=0不能作为受控初始化 |
| `tools/train_rtdetr_r18_lite_cbr.py` | 默认仅生成109字段计划；只有 --tmux/--execute 启动；真实训练回调再次核对全参数与初始化 |
| `tools/cbr_tmux_worker.py` | 保存 bootstrap、子进程退出码，包含早期导入失败 |
| `tools/cbr_results.py` | freeze固定val清单；evaluate同设置C2/C19官方指标及定位诊断；pack白名单小包 |
| `tools/autodl_c19.sh` | 使用已有rtdetr环境的prepare/start/status/val/pack入口 |

初始化源 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。源是 C2 的80类全模型（backbone为ImageNet，其他层原初始化），原存储half值精确提升FP32。正式nc=1由原trainer按seed42重建分类层，审计与C2同流程逐位一致。新增构造使用CPU `fork_rng(devices=[])`隔离。

109字段逐字段、逐类型核对，只允许 model/name/save_dir 改动；project原样保留。包括200轮、batch16、640、workers8、seed42、AdamW、lr0=.0005、decay=.0001、warmup5、AMP及在线增强，均取权威文件。`cbr_protected.json`保护原head/loss/trainer/config等；bus.jpg是从C18已有官方AMP资源复制的小文件，哈希独立按二进制核验，未改变AMP检查实现。

正式启动绑定实际初始化SHA、代码哈希、commit、Python/Torch/CUDA/GPU/import路径、C2args和data哈希，要求服务器2.1.2+CUDA审计和真实AMP smoke通过。原AdamW公共分组不变；新增非bias参数decay=.0001，bias decay=0，保留原bias warmup规则。启动前检查run目录、进程、tmux、退出记录，再以独占文件认领run；保留失败锁和日志，拒绝覆盖或重复启动。缺失退出报告只表示尚未记录退出，不能推断训练成功。

Smoke 的 batch=2/workers=0、debug输出、关闭val/plots/save以及初始GradScaler scale=128是**仅调试差异**，报告逐项列出；正式训练使用原生C2 scaler默认值和完整配置。本地原数据标签目录不能写cache，所以将16张train、4张val真实图片/标签复制到忽略的调试目录，样本来源另存manifest。服务器prepare将使用原完整数据配置。无依赖安装。CUDA grid_sample反向不保证位级确定性，沿用C2 deterministic warn-only行为，不将一次测试等同于统计可复现性。

## 验证记录与限制

本地 Windows / Python3.9.25 / torch2.7.1+cu118 / RTX2060 6GB / ultralytics8.4.21。检查：

- nc=1和80的全部公共权重、9个分类状态、构造后CPU RNG、原生get_model与实际Model.train重建；完整640 FP32输出逐位相同，无放宽初始化容差。
- 横长/竖长框、40×80像素中心、边缘padding、有符号内外差；坐标核验绝对误差1.2e-7（单位坐标约两个FP32 ULP），无统一放宽。
- 默认head与CBR零残差训练输出、公共梯度逐位一致；非零CBR改变实际返回框、query条件改变聚合权重、P3输入和上游梯度可学习；含空GT和可变DN查询。
- CPU FP32、CUDA FP32/AMP前后向；CUDA显式half完整640前向；三步真实数据AMP更新；FP32非零checkpoint重载逐位一致。half/AMP检查有限性、形状与宽高合法性，不将跨dtype输出要求为逐位相同。
- 6项风险回归，包括109字段变更拒绝、保护文件、子进程退出、固定匹配与小包缺项/覆盖/权重排除。工具在4张真实调试val图像上端到端跑通；其指标没有实验效力。

早期审计失败日志保留在本地outputs：夹具缺少nc、空GT不产生框梯度的错误预期、原数据缓存权限、smoke结束后原Model.train试图读取不存在的last、JSON Path序列化、二进制保护哈希。分别修正实际原因；没有调宽权重/输出等价门槛。最终结果摘要见 `cbr_local_validation.json`。本地完整性能基准是同输入、未融合FP32、batch1、3轮预热+10次计时；报告实测数值，不能外推4090或AMP正式训练速度。

**未执行**：服务器PyTorch2.1.2验证、tmux实际正式任务、200轮训练、正式val/test、与C2的实验性能比较。服务器命令会执行缺少的环境审计后才允许启动。不能把本地通过写成服务器成功。

## 评估、小包与后续

`freeze`在训练前固定前16个排序val样本，保存图片/标签SHA。`evaluate`对C2/C19各自best.pt在同一环境、640、FP32、conf=.001、batch16等相同参数下独立官方复验；报告P/R、mAP50、AP75、mAP50–95、完整AP数组、参数、速度及有效args。原生后处理完整继承。full precision JSONL保留官方后处理后的预测/GT，使用gzip避免重复；不采用原生JSON的3位框/5位score舍入。

定位诊断固定conf=.25，以最大总IoU的一对一Hungarian匹配报告GT召回及平均IoU；按**水平框w/h**、原图短边像素分组，不推断真实裂缝方向/宽度。预测分类为背景重叠<.1、定位重叠[.1,.5)、未分配高重叠、已分配高重叠，是诊断分类，不宣称完整TIDE或官方AP。

固定val样本额外运行未融合FP32，保存全部query细化前/后框、分数、同一GT/query匹配、IoU变化、四边绝对误差变化、平均修正和`abs(tanh)>=.95`饱和比例。匹配只由细化前框决定一次；不是细化前后各挑最佳匹配。未融合诊断与融合官方评估分开标识。该行为分析不是重新训练的移除消融。

先看val决定是否继续，固定test要求传入JSON文件`--val-decision`，含`selected_c19_sha256`（已选best.pt SHA）和`reason`（val依据）；工具不会替用户伪造选择理由。测试集不用于扫描rho/阈值/采样预算。C19有效后再考虑C20，不预先承诺组合可加。

`pack`要求训练args/results、初始化/完整audit/plan/实际args/preflight/tmux/退出记录、官方指标及AP/有效参数、checkpoint SHA、预测GT、固定清单诊断、commit与本轮源码；控制台仅末64KB。默认排除weights、图片、整个仓库和旧实验。先打印大小清单，压缩后超过20MB即失败并保留大文件原因。权重留服务器。训练CSV最优值仅曲线参考，最终对比标为独立复验。

## 研究定位

逐边定位与边界特征已有先例：[SABL](https://arxiv.org/abs/1912.04260)采用逐边定位、分桶及偏移；[BorderDet](https://arxiv.org/abs/2007.11056)使用边界特征；[D-FINE](https://arxiv.org/abs/2410.13842)研究DETR细粒度分布回归。已打开原论文摘要页核对基本定位。本方案未复现D-FINE的FGL/自蒸馏，也不将四边预测或双线性采样当原创。待验证假设是轻量RT-DETR中P3内外成对证据和query条件小幅框边细化的作用；完整相近工作比较和论文新颖性仍需后续研究。

服务器操作见 `C19_AUTODL.md`。只有显式 `start` 命令启动正式训练。
