# 本地验证

环境：Windows / Python 3.9.25 / PyTorch 2.7.1+cu118 / RTX2060 6GiB。测试运行于独立进程；正式配方仍为seed42、deterministic=True、AMP=True、B16、640、200e。受控FP32诊断关闭TF32、线程4；不会把这些诊断设置带进正式进程。

| 检查 | 实测 |
|---|---|
| 四模型 nc1 参数 | {'c2': 20082772, 'lif': 20103876, 'c17': 20109684, 'c17_lif_v1': 20130788} |
| 原历史C2/C17/LIF独立导入及非零模块输出/输入梯度/参数梯度 | max_abs=0.0 |
| 三组退化，CPU/CUDA，160×192及640×640 | max_abs=0，max_rel=0 |
| CPU FP32非零BN融合，特征+对齐query/raw | max_abs=7.152557373046875e-07, max_rel=0.4777902364730835 |
| CUDA FP32非零BN融合 | max_abs=8.344650268554688e-07, max_rel=0.9517436027526855 |
| CUDA AMP融合 | max_abs=0.00146484375 |
| CUDA true-half融合 | max_abs=0.00146484375，输出确为float16 |
| LIF BN保留，其他BN正常融合 | 69→27个BN |
| 非零参数保存重载 | CPU/CUDA max_abs=0；重复fuse可预期 |
| 真实predict自动fuse入口 | CPU/CUDA通过，见每项误差 |
| 真实val自动fuse入口 | 仅2张train图的隔离fixture，通过；每split 600预测/6 GT |
| 优化器 | 原Trainer.build_optimizer，336张量各出现一次；127 weight/81 norm/128 bias；10个创新权重均weight组 |
| 公共初始化 | 533 COMMON + 10新增weight + 2新增buffer，9项分类重建；missing/unexpected/shape mismatch均空 |
| 109字段配方 | 仅model/name/save_dir不同，类型不变，auto_augment为null |
| 数据 | train6048/45573框，val1728/12840框，test864/6663框；清单和标签指纹已记录 |

预先容差：退化 atol=2e-6/rtol=2e-5；FP32融合 atol=2e-5/rtol=2e-4；AMP与true-half atol=.008/rtol=.04。没有放宽失败测试的容差。近零值max_rel可较大，判定使用每元素绝对+相对容差。

CPU未融合/融合640的自然top-k出现重新排序（详见summary）。先验证三个尺度及排序前encoder score，再以未融合encoder anchor IDs在两个测试副本对齐。只在测试进程临时替换topk返回的索引，保留原分数，测完恢复；正式模型/训练/评估源代码没有此替换。自然逐行差异与query集合交集均记录，不把不同query行当同一框。原父退化测试保持原生排序且精确相等。

真实两张train图resize160，原RT-DETR DN/loss、CPU3步，CUDA FP32 3步+AMP3步有限更新；没有epoch循环。CPU losses=[57.088165283203125, 24.108715057373047, 24.467044830322266]；CUDA losses=[57.094482421875, 24.309080123901367, 24.542070388793945, 24.15387535095215, 23.343557357788086, 23.730545043945312]。两模块人为激活时，10项创新梯度、相关主路梯度均有限且非零；零初始化梯度开启顺序另外检查。模型和AdamW state保存/重载逐张量相等。更新后的smoke对象仅属于outputs，绝不进入正式init。

原warmup规则：nb=378、nw=1890；正式初始化accumulate=4、effective decay=.0001；前3batch按原插值accumulate=1，weight lr从0开始，bias lr从.1开始。数值见summary；没有为创新添加LR/梯度缩放或强制固定warmup accumulate。正式worker还记录前4batch的真实lr/参数范数。

CUDA本轮峰值allocated=737745408 bytes，限B1/640推理及B2/160训练的工程检查，不能代表B16容量。两图评估的时延仅作入口smoke，不能作为正式速度基准。原grid_sample backward不保证逐位确定性，捕获警告保留在checks.json；本地torch2.7的GradScaler弃用警告与AutoDL2.1.2兼容实现并存，不升级依赖。

NOT_RUN：AutoDL Python3.10.13/torch2.1.2+cu121动态执行、真实服务器tmux派发、B16/640 AMP容量、200e训练、完整val/test、正式成功训练包。start-direct先在独立进程执行有限原模块回归/非零融合/DN/loss/optimizer检查及原增强管线B16/640 AMP一次smoke，成功后重新创建未训练init和正式进程。历史C17 Git对象在服务器若不存在，仍用已核验字节fixture独立进程动态回归原模块；本地完整历史模型回归另有记录。

生命周期测试为显式mock：状态、重复启动、OOM不降batch、拒绝缺项pack；约22MB测试包使用真实流式写入、逐文件回读散列与拒绝覆盖，但训练metadata验证为MOCKED。它不是真实结果包。同步测试使用隔离Git fixture，源数据/分支均不会被训练或推送到公网。

同步集成实测通过：远端前进后仍固定旧SHA、detached worktree重入、跟踪修改/错误SHA/非worktree/错误origin拒绝、未跟踪结果保留、主仓库已有修改保留。只有fetch目的地重定向到隔离本地bare仓库。两个Bash脚本语法通过；真实Bash status返回NOT_STARTED，多余variant被拒绝。

数值报告产生于提交前的候选工作树，因此runtime.commit显示LIF父HEAD；同报告的文件SHA256标识实际被测组合代码。最终交付SHA由提交后的git rev-parse与远端ls-remote核验，不用父HEAD冒充。
