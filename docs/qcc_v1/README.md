# QCC v1

唯一实验：RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + **QCC-v1-capK3**。QCC 是研究候选，本交付没有正式训练或涨点结论。

入口：`bash tools/qcc_v1.sh --help`。正式流程分开执行 `prepare`、`preflight`、`status`、`start`，训练结束后执行一次 `finish`。`preflight` 不调用 `start`，本地检查不运行正式 test。

## 实现边界

- 基座为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`，直接从母版创建分支；没有继承 ARG/RMD 等损失。
- 原 LIF、CBR、decoder、matcher、VFL、原损失模块和训练循环保持原文件。唯一公共源码修复是 AutoBackend warmup 用有限全零输入。
- QCC 的 `get_model` 执行原生 nc=80→1 构建/加载，再使用可导入的 Python 模型类；审计构建位于隔离 RNG 范围内，无新参数或持久 tensor key。
- `RTDETRDetectionModel.loss` 原生拆 DN、拼 encoder。QCC 仅捕获第一项最终普通层 `_get_loss` 的原匹配和 `_get_loss_class` 的真实 `gt_scores`。原 auxiliary、DN 及无 DN 补零全部完成后，才加入一项 `loss_qcc`。
- 捕获内容在 `finally` 清除；关闭和 epoch≤5 完全走原损失路径，不额外分组/排序/耗 RNG。验证和推理无 QCC。
- 仅支持 nc=1；单 GPU。本交付不声明 DDP 已验证。

## 固定公式

`lambda=0.10, K=3, r(e)=clip((e-5)/15,0,1)`，e 是零基真实 trainer epoch，resume 延续。

普通未匹配 query 对本图全部 GT 做原 `bbox_iou(xywh=True, eps=1e-7)`，按最高 IoU 归属。`a=最高IoU-次高IoU`；单 GT 的次高值为 0。排除所有已匹配 queries、a=0、IoU>该 GT 原生匹配质量 q 的候选；按 detached `a*sigmoid(z)` 稳定降序，平分优先原 query 小索引，最多取 3 个。

当 0<q<1，仅在 `sigmoid(z_pos.detach())<q` 时保留正例 logit 梯度；否则使用 detached `log(q)-log1p(-q)`，等号处梯度为 0。q=0 跳过，q=1 直接用原 logit。

组 logits 为 `[z_tilde, z_u+log(a_u), 0]`；目标 `[q,0,...,0,1-q]`；`ell=logsumexp(group)-q*z_tilde+q*log(q)+(1-q)*log(1-q)`。`h=max(a_u*p_u)` detached，`raw=sum(h*ell)/max(M,1)`，M 包括所有最终普通匹配，包括无候选组。只优化 `0.10*r(e)*raw` 一次。

QCC 局部关闭 autocast，用 FP32；q、GT、框、几何、owner/topK、a/h 全断梯度。原 L0 精度和权重不变。没有 padding 假候选，边界不做任意质量裁剪。

## 可靠性工具来源

参考 `ef9cb7e05e5557f7dd06c95cf2361998a284adc9`，迁移的都是运行/审计流程：

| 来源 | QCC 中使用的内容 | 本轮修改 |
|---|---|---|
| `tools/arg_v1_common.py` | 完整配方比较、公共初始化、code binding、原 inventory、类别键规范化 | 新增固定快照复用；后续只核对缓存/配置/顶层目录 mtime；配方额外核对母版 paired args |
| `tools/arg_v1.py` | PID+创建时间锁、tmux、PIPESTATUS、真实 worker 退出码、预检资格、resume/只评估恢复保护 | 新写完整首次导出、幂等 val/test/finish、离线阈值统计、单一完整包；移除可选预测导出及补推理打包 |
| `tools/arg_v1_preflight.py` | ≤16 micro-batch/≤900 秒、原生 scaler 与独立 scale128 诊断、新进程恢复/val | QCC 分类梯度探针；CBR 和最终 bbox head 的 QCC 直接梯度应为空；全参数梯度有限性审计 |
| `arg_model.py` 的运行模式 | 原生 get_model 对照、optimizer post-step hook、AMP 必须明确 PASS | 独立重写 QCC model/trainer；按计数/和/二阶矩聚合；法定 stop 保存后标记训练完成 |
| `arg_val.py` / 母版 `c19_lif_v1_results.py` | `corrected_sorted_conf_mask_v1`，排序和 conf mask 同序 | 独立重写 QCC streaming export、logits/两种坐标、完整性原子标记、十 IoU AP/曲线/统计 |
| `nn/autobackend.py` | `torch.empty`→`torch.zeros` warmup 修复 | 精确迁移一行，无推理算法变化 |
| `sync_arg_v1.sh` / `arg_v1.sh` | 有限 fetch、原仓库 worktree、安全拒绝覆盖、固定解释器 | QCC 专属分支/目录/session |

没有复制 ARG loss，没有 ARG 专属状态/超参，没有 ARG 模型或 checkpoint 导入。

## 数据身份和一次导出

`prepare` 优先读取本实验已存身份；首次可自动复用同服务器 ARG `outputs/arg_v1/prepare.json` + `data_manifest.jsonl.gz`。复用要核对 source、真实路径、split、规范化类别、清单摘要和历史计数，并记录 `reused_from`。这不声称重新核验了原图字节。无可复用身份才做一次图片/标签清单和哈希。

`start/resume/preflight/val/test` 只读准备好的身份和轻量配置；检测顶层目录变化会拒绝并要求显式 `prepare --recheck-data`。原地修改文件内容不能靠这种轻量检查发现，固定快照的操作前提是数据不变；需要时显式 recheck。`status/pack` 不做全量 identity inventory、不实例化检测模型、不构建原始 DataLoader。正常训练/评估读取对应图像和已有标签 cache 属于必要模型流程。

首次正式 val/test 默认写 `*_queries_gt.jsonl.gz`，无 `--include-predictions` 开关。包括稳定相对 image_id、split/原尺寸、完整 GT、全部 300 个原 query 索引、CBR 后归一化 cxcywh 和原图 xyxy、sigmoid 分数/类别、可用的原 logits。原图坐标按 W/H 独立缩放，不裁剪；原始导出不筛选/排序，不人为舍入。正常评估仍按原 corrected conf=0.001 筛选，RT-DETR 不使用 NMS，iou/max_det 参数不新增处理步骤。

gzip 先写 `.partial`，全 split、GT 计数和评估成功后才改名并原子写 `.complete.json`。保存 checkpoint/data/code/实际 args/schema/评估器版本。中断保留 partial 和错误，不发成功锁。成功但缺导出的旧锁必须显式 `val --recover-export` 或 `test --recover-export`；status/pack/普通 finish 不偷偷重跑。

训练原本必要的 final_eval 用同一 FP32 导出/锁流程，finish 优先复用；AMP epoch val 不能作为正式 FP32 锁。finish 顺序固定为完成证据与 best → 完整 val 锁 → 同 best test → 离线汇总 → 一个分析包。重入复用已完成阶段，保留原错误和退出码。

P/R/F1 保留评估器在各 split 最大平滑 F1 点的原值；十个 IoU AP、AP75、PR/P/R/F1 曲线及混淆矩阵均保存。原生 P/R 对应计数为插值后取整，单独注明。离线另从 val 锁定阈值，在导出的全部 queries/GT 上筛选后重做原 IoU-greedy TP 匹配，给出 val/test 的准确整数 TP/FP/FN 与 P/R/F1。测试集不用于选择部署阈值。Precision 不是 Accuracy，不定义 TN。

包默认含全部已存 query/GT、运行/恢复日志、配方/环境、源码快照、报告、曲线、锁和含每文件大小/SHA256 的清单；manifest 不包含自身哈希。不含大权重和原始图像全集，记录权重路径/大小/SHA/epoch/选择规则。不足材料可输出 INCOMPLETE 包，不冒充完成。

## 日志和有界预检

每 epoch 仅前 4 个训练 micro-batch 采集机制统计，按 count/sum/sumsq 合并。空统计 count=0、均值 null，不写 NaN；disabled 日程不为了日志运行分组。optimizer hook 计真实 step，前两次更新另记全梯度有限性、scale/overflow 和参数实际变化；整个进程的 update/attempt/skip 计数也保留。

server preflight 保持 B16/640/AdamW/原 AMP/累积4，单独 e=20 覆盖机制。最多总计16个 micro-batch、900秒，独立进程与阶段心跳，有界超时只终止本次子进程树。先诊断 native 初始 scale，若预算内未有效更新，可按上述参考工具的既有规则隔离重置做 scale128 诊断；分别报告，绝不把 fallback 当作 native scaler 已通过。正式训练 scaler 不变，从公共初始化/e=0 新建状态。

启动必须 math/routing、B16 AMP 有效更新、QCC 机制、新进程真实 val、完整 optimizer/scaler/epoch 恢复为 PASS。native-scale 仅 PENDING 时还必须有独立 scale128 有效更新证据；此门槛沿用参考提交，不降低任何明确失败项。DDP 不是本轮单 GPU 启动门槛。

本机诊断允许明确标注的缩小输入，不代表正式容量通过；具体结果与服务器命令见 DELIVERY.md、server_commands.md。
