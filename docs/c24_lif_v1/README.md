# C24 SCCA + original LIF-Down v1

独立分支 `codex/rtdetr-c24-lif-v1` 从 `0e95bbade3558b0d2b77c5531483c60810391d88` 建立。
只迁入成功 C24 的 `scca_aifi.py`、单模块 YAML、两处注册及 AIFI parser 局部识别；没有 merge C24 历史。
本地工作树：`D:/MyProjects/Crack_RTDETR/outputs/worktrees/c24-lif-v1`。

**工程交付不等于数值门禁全部通过。** 本机 2060 的 640×640 AMP/true-half 融合出现自然候选集合变化，
连续计算和双方 A/B 双集合重放通过原容差，仍为 `REQUIRES_REVIEW`。没有足够可比父对照证据时绝不派发训练。
没有执行正式训练、完整 val/test 或 AutoDL 操作；不存在本组合性能成绩。

## 固定结构与来源

| 模型 | nc1 未融合参数 | 状态 |
|---|---:|---:|
| C2 | 20,082,772 | 533 |
| C24 | 20,148,312 | 538 |
| LIF v1 | 20,103,876 | 538 |
| C24+LIF v1 | 20,169,416 | 543 |

组合 layer9=`SCCAAIFI([1024,8])`，layer20=`LIFDown([256,3,2])`，layer23 仍普通 Conv。
layer26 仍原 RTDETRDecoder，输入 `[19,22,25]`、三尺度、三层、300 regular queries。
拓扑工具从 decoder 和 PAN 连接推导位置，再校验两处差异，不拿预期层号替代结构检查。

两个模块源字节保持原版，LF 固定在 `.gitattributes`：

* SCCA：`67b0d347d5305c16d48bb031cd10fd1d23b7ea447b58b2423166eb68b81dedca`
* LIF：`26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`

`BaseModel.fuse` 的 `not isinstance(m, LIFDown)` 未改；原 BN 前相加及 forward_fuse 保留。
head.py、transformer.py、梯度、loss、DN、matcher、生产 top-k 和两个父工具的默认行为均未修改。
来源包、实际读取参考文件及其 SHA256、109 字段和值类型核对见 `PROVENANCE.json`。

## 初始化和训练

公共源唯一 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，
验证 nc80、epoch=-1、optimizer/EMA/scaler 等旧训练状态为空。
在各自 CPU fork_rng/seed42 的原父构造流程产生初值，逐键映射 533 个公共状态与 10 个新训练状态。
不读取任何父 best.pt。新 buffer 为零。

原生 `RTDETRTrainer.get_model` 实际重建 nc1：534/543 精确加载，9 个分类状态适配；
与同一次 RNG 的原生 C2 nc1 分类初始化逐键一致。保存重载精确。
正式 RecordingTrainer 只调用该原生重建并写审计，训练仍使用原优化器、warmup、AMP 和全部109字段配方。
336 个 requires_grad 张量恰好各覆盖一次；新增10张量均按原 weight 组正常训练。
所有测试更新均在临时副本上，临时权重测试后移除，不能充当正式 init。

## 文件入口

* `init_c24_lif_v1.py`：原公共源、父初值、保存重载、native Trainer 审计。
* `check_c24_lif_v1.py` / `c24_lif_v1_numerics.py`：有限数值、真实候选、损失和入口检查。
* `train_c24_lif_v1.py`：一次 preflight 后才派发；独立锁和真实 PID/start-time/token。
* `sync_c24_lif_v1.sh` / `autodl_c24_lif_v1.sh`：固定 SHA、独立环境和用户操作入口。
* `c24_lif_v1_results.py`：沿用成功父评估的 `corrected_sorted_conf_mask_v1`。
* `c24_lif_v1_pack.py`：标准库故障包≤8,000,000字节；完整分析包默认不包含权重和全量预测。

测试证据见 `VALIDATION.md` 和逐阶段 JSON；服务器命令见 `AUTODL.md`。
