# C2 + FSA-Deform v1

本实验直接从真实 C2 `67c3078e54a657fd96d65fee657a75fbb1dae0d6` 分叉，分支 `codex/fsa-deform`。
原工作目录和其他实验/worktree/用户文件均保留。未启动正式 200 epoch 训练或真实数据集完整 val/test。
本轮不制作论文模块图，不创建 v2。

## 唯一结构变化

当前代码位于 `ultralytics-main/ultralytics/`。独立 YAML 是
`cfg/models/rt-detr/rtdetr-resnet18-lite-fsa-deform.yaml`，相对 C2 YAML 只替换最终容器为 `RTDETRDecoderFSA`。
容器先执行原构造器及三层 clone，之后仅替换 `model.26.decoder.layers[1].cross_attn`。

```text
decoder.layers[0].cross_attn = MSDeformAttn
decoder.layers[1].cross_attn = FSADeformAttn
decoder.layers[2].cross_attn = MSDeformAttn
```

实际规格：d_model=256、heads=8、K=4、levels=3、regular queries=300。
实现按实际 feature levels/尺寸广播，不固定 level 数量；独立正式模型审计核对其等于 C2 的 3。
容器遇到非三层 Decoder 直接报错。未改 self-attention、AIFI、backbone、neck、FFN、loss、matcher、DN、query 数和 bbox/score heads。
没有全局 monkey patch，没有引用其他实验模块，没有引入 SFS 模块包或 CUDA/C++ extension，没有升级环境。

## 计算与初始化

`FSADeformAttn(MSDeformAttn)` 保留原 `sampling_offsets`、`attention_weights`、`value_proj`、`output_proj` 和全部 key。
原 offsets、一次 attention softmax、reference-based 中心位置、level×K 加权聚合和 output projection 语义不变。
FSA 改变的是**每个原 sampling location 对应的局部读取值**。

同一 cross-attention query 先做无仿射 RMS：

```text
q32 = query.float()
rms = q32 / sqrt(mean(q32**2, dim=-1, keepdim=True) + 1e-6)
h = SiLU(support_fc1(rms converted to Linear weight dtype))
raw = support_fc2(h).reshape(B, Q, 8, 3)   # tx, ty, a
```

RMS 的平方、平均、epsilon、sqrt、除法均在禁用 autocast 的 FP32 上执行；不减均值，不是 LayerNorm。
Linear 输入按权重 dtype 转换，Linear 自身继续服从调用方 autocast。
`support_fc1=Linear(256,16)`，Xavier uniform、bias=0；`support_fc2=Linear(16,24)`，weight/bias 严格全部为 0。
构造器在 fork_rng 内创建局部扩展，不推进公共模型构造的 RNG 序列。

对当前 reference box `(cx,cy,w,h)`、每个实际 level `(H_l,W_l)`：

```text
bx = clamp(w * W_l / (2*K), 0.25, 1.5)
by = clamp(h * H_l / (2*K), 0.25, 1.5)
rx = clamp(bx * 2 * sigmoid(tx), 0.125, 2.0)
ry = clamp(by * 2 * sigmoid(ty), 0.125, 2.0)
p_support = p + (sx*rx/(sqrt(3)*W_l), sy*ry/(sqrt(3)*H_l))
(sx,sy) = (-1,-1), (-1,+1), (+1,-1), (+1,+1)
v0 = V(p)
v_area = (v1+v2+v3+v4)/4
eta = 0.5*tanh(a)                         # signed, bounded [-0.5,+0.5]
v_new = v0 + eta*(v_area-v0)
```

tx/ty/a 每个 query/head 一组，广播到所有 level、原 point 和 head channel；rx/ry 随 level 改变且彼此独立。
w/h 来自当前 Decoder reference，不来自 GT 或方向标签。中心不额外 clamp，不重归一化越界点。
每 level 一次向量化 `grid_sample` 读取 K×5 个位置，只沿 level 循环。
使用原 `grid=2*location-1`、bilinear、zeros padding、align_corners=False；value_mask 的 True 位置在投影后置零。
支持几何和 FSA helper 的采样/residual 在 FP32 中执行，聚合结果转回 projected value dtype 后进入原 output_proj。
首版因此在 AMP/half 下也有 FP32 的支持读取成本。

zero-init 时半径非零、eta=0，完整执行五点读取并退化为原 attention；没有 zero-init 短路分支，没有额外 alpha。
首步 fc2 的 a 输出行可学习，tx/ty 行和 fc1 首步梯度为 0 属于预期。
一次更新让 eta 路径非零后，tx/ty 和 fc1 获得有限非零梯度。没有为此破坏初始化。

reference 最后维为 2 时显式调用 `super().forward` 中心路径，不运行 support MLP，不补宽高。
四维路径不额外 detach 或改变梯度规则：原 Decoder 在训练中对上一层 refined reference detach，eval 沿用原规则。
DN query、mask、顺序及 metadata 均通过原 head/decoder 传递。

## C2 来源和公共权重

`source_trace.json` 记录了原 C2 归档、参考模块实际路径及文件摘要。
参考路径和历史 zip 均存在于 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/`。
原 C2 transformer/head/utils/validator/YAML 按 LF 标准化与 C2 commit 逐字节相同。
未发现适用 AGENTS.md（祖先路径和 worktree 已检查）。

初始化唯一来源：主仓库 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
它是 epoch=-1、无 optimizer/EMA 的公共初始化 checkpoint，绝不使用实验 best.pt。

公共源 nc=80：533 个状态全部按相同 key、shape、数值映射，之后使用原 RTDETRTrainer 的 nc=1 重建规则。
允许的 nc80→1 例外只有 denoising_class_embed 以及 encoder/三层 decoder 分类 weight/bias，共 9 个状态；
baseline 和 FSA 用相同 seed42 适配，适配后全部 533 个公共状态仍完全一致。
第二层原 attention 的八个 weight/bias 逐项一致；第一、第三层所有状态一致。
新增仅四个 support tensor；MISSING、UNEXPECTED、未允许 SHAPE_MISMATCH 均为空。
详见 `weight_mapping.json` 的 COMMON / NEW / MISSING / UNEXPECTED / SHAPE_MISMATCH / ALLOWED_EXCEPTION。
初始化采用完整字典 strict=True，训练 API 内部加载后另做逐状态审计，不用 strict=False 掩盖缺失。

实测 nc=1 未融合参数：C2 **20,082,772**；FSA **20,087,292**；delta **4,520**。
新增 256×16+16=4,112，16×24+24=408。
本地生成的初始 checkpoint 在 worktree `weights/fsa_deform_controlled_init.pt`（忽略于 Git）；服务器启动时从同源重新构建。
字节哈希可因序列化元数据/路径不同而不同，公共源 SHA256 和逐 tensor 映射必须一致。

## 已完成的有限验证

`validation.json` 是本机 PyTorch 2.7.1+cu118 / Python3.9.25 / RTX2060 的实际结果；
`initialization.json`、`weight_mapping.json`、`recipe_diff.json` 和 `lifecycle_validation.json` 为辅助证据。
报告的 runtime.commit 是测试时提交，working_tree_dirty/source_sha256 明确标识提交前源码；后续文档提交不会改变数学实现。

- 三层类型、唯一插入位置、参数差量和非三层拒绝均通过。
- CPU FP32 eval、CPU train/DN、CUDA FP32 eval：layer1/2 输入输出及最终 bbox/score 的 max_abs_error/max_rel_error 均为 0。
- DN 实际 Q=500，meta、mask、顺序相同；训练 reference 已 detach。
- 独立 x/y 半径、上下双 clamp、四点正负组合、normalized→grid 的 2 倍变换、mask/越界行为、常量/仿射/非线性场通过。
- 非零 support 与独立四次原生 helper 的数值 oracle 对照通过；2D fallback 与原生输出严格相同。
- 首步 a 梯度非零，tx/ty/fc1 为零；第二步 tx/ty/fc1 梯度非零，均有限。
- CUDA FP32/AMP decoder+DN 前后向有限；学到非零 support 后的整网 true model.half() inference 通过。
- state_dict 和整网保存重载保持已学习 support；初始化 checkpoint 通过 RTDETR API 重载逐项精确相同。
- 一步 B2、160×160 合成输入的原生 matcher/loss/整网 backward + 普通 AdamW 更新通过，仅测试入口使用这个小规模。
- 正式入口 mock 验证同名拒绝、真实退出码分类、OOM 原样传播且 batch16 不变、归档读回哈希和大于20MiB文件、不完整包拒绝。
- C2 validator 的历史排序/阈值细节原样保留，专项乱序阈值输入测试结果与 C2 完全一致。

torch deterministic=True 按 C2 使用 warn_only=True；CUDA grid_sample backward 本身不是确定性 kernel，因此不能声称 GPU 训练逐 bit 可复现。
独立 head 的测试在跨设备 deepcopy 后清空原生 anchor cache；整网设备/dtype 迁移继续使用原 RTDETRDetectionModel._apply。

粗测为整网 FP32 eval B1、160×160、5次 warmup、20次计时：结果见 validation.json。
本次最终测量约 C2 38.01ms / FSA 42.13ms，peak allocated 175,631,360 / 175,650,304 bytes；该小输入下全网峰值发生在其他部分，
不能据此认为训练显存开销很小。FSA 局部采样从4增至20个读取位置/level/head/query，读取 tensor 和 backward 保存量明显增加。
桌面 GPU 状态会影响毫秒粗测；这些不是 AutoDL4090、batch16、640 的正式速度或显存结论。
本地数学验证不是检测性能证据，不保证 mAP 提升，不声称恢复已消失特征、四点有四倍有效信息或 SCI 新颖性。

## 训练、评估与打包入口

正式参数逐字段/类型锁定原始109项 C2 args：200e、640、batch16、seed42、AdamW、lr0=.0005、lrf=.01、
weight_decay=.0001、warmup5、cos_lr=True、AMP=True、close_mosaic10、workers8、deterministic=True、patience50，在线增强全保留。
默认路径下仅 model/name/save_dir 三项改变；FSA_MAIN 只用于同一目录树显式迁移，不能拿它变更数据或配方。
实际 trainer args 再次逐项核验，所有参数均进入原 AdamW 参数组一次；无特殊学习率、无冻结。

服务器主仓库默认 `/root/autodl-tmp/projects/Crack_RTDETR`，独立 worktree `/root/autodl-tmp/projects/Crack_RTDETR-fsa-deform`。
`bash tools/sync_fsa_deform.sh FULL_40_CHARACTER_COMMIT` fetch `codex/fsa-deform` 并核对完整远端 SHA、C2 祖先、origin，
新建 detached worktree；已有同一 SHA 的干净 detached worktree 也可重复核验。
已有目录、其他仓库、不同 SHA、脏修改均拒绝覆盖。同步/运行不依赖当前 branch name，不 reset/clean/kill、不创建环境。
首次同步脚本可用 `git show PINNED_COMMIT:tools/sync_fsa_deform.sh` 从 fetch 后的固定提交取得；最终交付消息提供实际完整 SHA 命令。

进入独立 worktree 后：

```bash
bash tools/autodl_fsa_deform.sh start-direct
bash tools/autodl_fsa_deform.sh status
tail -f /root/autodl-tmp/projects/Crack_RTDETR-fsa-deform/outputs/fsa_deform/console.log
```

start-direct 使用现有 rtdetr conda，先记录 Python/PyTorch/CUDA/import路径/FSA源文件/git SHA、model.yaml、
resolved train args、来源和生成权重 SHA256、初始化映射、pip freeze、源码快照及 patch。
检查输出/独立目录、同名 worker/tmux、共享原子 reservation 后在 tmux `fsa_deform-training` 启动。
OOM 或 AMP 检查失败立即失败，不减少 batch/imgsz/query/support points，不关闭AMP。
status 仅输出 NOT_STARTED / DISPATCHED / RUNNING / SUCCESS / FAILED，以 PID start token 和 Python/shell 退出码判定。
服务器完整预检及正式训练尚未运行；start-direct 中环境/路径/配方/映射检查不等于完整服务器训练验证。

正式结果：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/fsa_deform_rtdetr_r18_lite_e200_b16_onlineaug`。
失败保留输出和锁以供诊断，不自动覆盖/重试；如需下一次设计由用户明确决定。

训练成功后依次：

```bash
bash tools/autodl_fsa_deform.sh val
bash tools/autodl_fsa_deform.sh test
bash tools/autodl_fsa_deform.sh pack-complete
```

val/test 使用同一训练 best.pt，并比对 checkpoint SHA256、源码 SHA、冻结 data config。
使用 C2 原 RTDETRValidator：conf=.001（C2 conf=null 在 validator 中的默认）、iou=.7、max_det300、640、batch16、workers8、
独立评估 FP32 half=False、无增强、相同后处理。训练内 validation 继续原 C2 AMP 条件下的 FP16 规则。
历史 C2 postprocess 按未排序 score mask 过滤已排序预测的细节不会在本实验悄然修复；diagnostics 如实标记原生 metric 选择。
导出 Precision、Recall、mAP50、mAP50-95、AP75、完整10个IoU AP、曲线/混淆矩阵、每图全部300最终预测及GT。
诊断副本不改变实际用于 metrics 的原生 selection。

pack-complete 要求成功退出、best/last、args/results/曲线/混淆矩阵/样例、初始化/训练配置/源码/val/test证据齐全，
缺失即拒绝。输出包含 training/、val/、test/、console/、metadata/source/、MANIFEST.json，
归档读回所有字节逐项 SHA256 验证，附 `.tar.gz.sha256`、`.tar.gz.inventory.json`、`.tar.gz.verification.json`。
默认目录 `/root/autodl-tmp/projects/Crack_RTDETR/downloads/fsa_deform`。仅归档实验输出、诊断和源码，不收集数据集、环境目录或凭证。

本地复验（使用现有环境，不安装任何包）：

```bash
python tools/check_fsa_deform.py --source /path/to/main/weights/rtdetr_r18_lite_imagenet_backbone_init.pt
python tools/check_fsa_deform_ops.py --output docs/fsa_deform/lifecycle_validation.json
```

未验证：正式200e稳定性、真实val/test、检测指标、AutoDL4090 batch16/640显存和速度、ONNX/export/其他后端。
等待 v1 正式结果后，再结合 Recall/Precision/严格IoU/稳定性/显存决定保留、针对性v2或停止此机制。
