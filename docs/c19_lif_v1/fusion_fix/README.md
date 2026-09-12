# C19 + LIF v1 预检融合工程修复

本修复追加在 `codex/rtdetr-c19-lif-v1`，旧提交为 `c27064ce6b06d0c32d38e644fcff64394a1a5992`。不改变模型、初始化、正式训练配方或评估协议。没有启动正式训练或完整 val/test。

## 故障证据与结论边界

原包 SHA256 `765b4458637e9f5555423f20adbeba120df6ceb7e9b15f6f5e129c98e2fa9791`，6,637,805 bytes，已核对。外层失败报告是 AutoDL Python 3.10.13 / Torch 2.1.2+cu121 / RTX4090 的 CPU FP32 B1/160×192 融合检查，最大绝对误差 0.20000004768371582，相对误差 0.38095247745513916。没有进入 CUDA 或 B16 容量阶段。嵌套源码快照内的历史 PASS 不是此次服务器通过证据。

包内没有 `.pt`、失败输入/RNG、失配 key/索引或 lock owner。开发机不能从这些信息确定原始首个失配张量，也没有复现服务器那次 0.2。**原始根因仍待服务器采集，不能宣布一定是 top-k。** 新入口只读加载旧初始化并验证它与公共初始化严格一致，保持旧 seed321 和六个父退化用例的 RNG 前缀，再保存该环境的实际融合 fixture。旧工作树、初始化、日志、失败报告与共享锁保持原样。

已证实的旧比较器问题：必需 key 缺失被交集静默略过；集合遍历导致错误顺序不确定；整数 ID 转 float 会丢失身份精度；probe 重算 top-k，未核对实际 gather；CPU FP32 对不同候选直接逐行比较；异常发生在返回赋值之前，部分 fuse 报告丢失。

## 修复后的判断

`c19_lif_v1_probe.py` 临时包装实际 encoder top-k，核对作用域、唯一调用、输入 logits、实际 gather 的特征、selected score 和 reference/anchor。返回的 native IDs 与 replay IDs 分别保存。两边重放各自的特征与原生 Decoder/CBR，只固定同一组候选 ID。CPU/CUDA/Python/NumPy RNG 均恢复；正常、安装异常、forward 异常均卸载 hook 和方法包装。生产 forward 没有修改。

`c19_lif_v1_diagnostic.py` 使用明确的 C2/LIF/C19/pair schema。先验证双方完整 schema，再按显式父模型差异表投影比较。候选整数精确比较；shape、语义 dtype、浮点有限性与合法 anchor inf mask 都检查。报告包括计算阶段、设备、精度、key、shape、dtype、最大失配坐标、两端值、超差数量。allclose 仍是 `abs(a-b) <= atol + rtol*abs(b)`；展示相对误差的分母是 `max(abs(b),1e-12)`。AMP hook 边界可产生 FP16/FP32 两种浮点 dtype，此时按浮点语义在 promoted FP32 比较；整数与 FP32/true-half dtype 不放松。

三阶段记录：LIF input/conv/residual/pre-BN/post-BN，P3/P4/P5，input projections/展平特征、encoder 特征/全部分数、anchor/valid mask；自然候选 ID/level/y/x、边界分数差、集合和行序；自然输出、ID 双射对齐、两端固定 ID 重放的 selected features/anchors/reference、逐层 query、raw boxes/scores、CBR 输入/offset/残差/输出、最终 boxes/scores。

| 候选关系 | 通过标准与动作 |
|---|---|
| 同 ID 同序 | 连续算子和全部后续输出满足原容差，`PASSED` |
| 同集合换位 | 连续算子、实际 ID 对齐、两端重放均通过，`PASS_WITH_CANDIDATE_PERMUTATION`；自然行序不一致仍保留 |
| 集合改变 | 保存全差异、公共数、交换 margin 与实测 `2e + 浮点 slack`、同 fixture C2 父模型敏感性；即使算子重放通过，仍为 `REQUIRES_REVIEW / OPERATOR_CHECK_PASSED / NATURAL_SELECTION_DRIFT`，阻止正式启动 |
| 连续算子/对齐/重放超差 | `FAILED_REAL_NUMERICAL_MISMATCH`，保留首个失配节点 |
| schema/trace/shape/证据保存异常 | `FAILED_INCOMPLETE_DIAGNOSTIC`，阻止启动 |

自然输出中的逐行失配记录仅描述未经身份对齐的误差；不能单独解释为算子错误。没有 bbox 排序、Hungarian、交集放行、种子更换、增加容差或删除 CPU/fuse 检查。

每阶段原子更新 `fuse_diagnostic.json` 和外层 `checks.json`。每个唯一输出目录保存 `fixture.pt`（输入、双方 state_dict、RNG）及 `records.pt`（自然/对齐/重放/父模型记录），并记录 SHA256。异常不会在 finally 中变成成功。张量、初始化和权重只在忽略的 outputs/weights 中，不进入 Git。启动失败仍复制部分预检 JSON，保留路径与 SHA。

## 实测范围

本地 Python 3.9.25 / Torch 2.7.1+cu118 / RTX2060 6GB；四 CPU threads，TF32 off。FP32 融合容差仍为 `2e-5/2e-4`；父回归 `2e-6/2e-5`；AMP/true-half `3e-3/3e-2`。

| 模式 | 排序前最大绝对误差 | 分数 e | ID 换位/集合差 | 自然最终框差 | ID 对齐框差 | 重放框差 |
|---|---:|---:|---|---:|---:|---:|
| CPU FP32 | 2.38419e-6 | 7.74860e-7 | 0 / 0 | 2.98023e-8 | 2.98023e-8 | 2.98023e-8 |
| CUDA FP32 | 1.60933e-6 | 5.96046e-7 | 0 / 0 | 3.72529e-9 | 3.72529e-9 | 3.72529e-9 |
| CUDA true-half | 0.00244141 | 0.0009765625 | 218 / 0 | 0.749882 | 1.18464e-6 | 1.16602e-6 |
| CUDA AMP | 0.00269222 | 0.0009765625 | 211 / 0 | 0.791673 | 1.18837e-6 | 1.18464e-6 |

该表来自 `local_full_checks.json`。两个 FP32 模式全部通过，两个半精度模式通过候选换位诊断。半精度自然序首个失配都是 `selected_features`，不是排序前张量；这只证明开发机本例的换位，不能替代 AutoDL CPU 故障定位。进一步的最终工具融合回归与 fixture 重放见 `local_final_fusion.json` / `fixture_replay.json`。

原 C2/LIF/C19 隔离历史构造、eval/train 输出、loss/梯度回归通过；CPU/CUDA 共 12 个父退化用例通过；非零 LIF+非默认 BN+非零 CBR、重复 fuse、非零模型保存重载、AutoBackend 自动 fuse 通过。有限真实 B2/160 三步原生 loss/backward/optimizer/warmup、动态 DN 与 CPU/CUDA/AMP 通过。仅用两张训练图复制出的工程 fixture 验证评估/打包链，没有完整 val/test。

29 个故障/正例验证了 missing key、shape、dtype、整数精确身份、NaN、合法/非法 anchor mask、纯置换、集合改变、真实 gather 与异常卸载、LIF residual 丢失、BN 破坏、encoder score/query/CBR 扰动、仅重放阶段扰动、外层 checks 失败落盘及容量缺失阻断。真实失配分别定位到 `lif_residual`、`lif_post_bn`、`candidate_scores`、`query_layer_0`、`cbr_residual`。具体断言和失配索引见 `fault_tests.json`。旧 compare 的整数 float 精度丢失假通过、原逐行框比较对纯置换的假失败也作为对照运行并记录。

26 个恢复用例通过：只读不改字节；错误 owner/variant/SHA、非 failed 状态、正式 run、worker/plan/训练现场、并发恢复、活跃或身份不明 PID、C19 worker/preflight、精确 tmux、tmux 读取失败均拒绝；成功时保留式归档，两次失败产生两个不同存档，C17 fixture 不受影响。恢复测试的 Git/proc/tmux 为显式 mock，不代表服务器锁已经恢复。原生命周期 mock 与真实归档 IO 回归通过。

**AutoDL NOT_RUN，B16/640 AMP 容量 NOT_RUN，正式训练 NOT_RUN，完整 val/test NOT_RUN。** 本地小图通过不满足启动容量门禁。新 `preflight-only` 使用独立临时初始化完成原有限预检和原 B16 原生 loss/backward；不派发训练。之后显式 `start-direct` 仍创建自己的新正式初始化并重复全部门禁，未把手工预检结果代入正式启动。

验证发生在补丁提交之前，因此报告内 `runtime.commit` 是当时旧 HEAD；实际受测源码由报告及 `audit.json` 中的源码哈希锁定。`local_full_checks.json` 先完成全部有限回归，随后对证据落盘等收尾修改运行了最终四精度融合和 29 个故障/正例。不要把该字段误读为服务器原故障已经复现或通过。

## 不变项与文件

模型目录、head/transformer/tasks、CBR/LIF、初始化工具、公共旧 LIF 工具、sync 脚本、原评估工具、configs 相对旧 SHA 均无 diff。nc1 未融合仍为 20,149,765 参数；533 COMMON + 19 NEW_TRAINABLE + 0 NEW_BUFFER；native nc1 543 项精确加载 + 9 项原分类适配。原 rho=.10、36 点、最终 Neck P3/final_query、采样 detach 和梯度规则、BN 前 LIF 残差及融合保护均保持。

- `lif_down.py` LF SHA256：`26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`
- `cbr.py` LF SHA256：`d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787`

修改：probe/checker、启动失败记录和门禁、shell 的有限预检入口、生命周期 mock fixture。新增：有序比较/三阶段诊断、fusion-only/fixture 重放入口、纯有限预检入口、安全恢复、故障与恢复测试。本目录保留本次审计及验证，不覆盖父目录历史报告。服务器具体操作见 [AUTODL.md](AUTODL.md)。
