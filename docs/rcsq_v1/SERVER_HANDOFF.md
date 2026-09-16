# RCS-Q v1 服务器交接

此文件是提交内可重复生成模板；交付目录内 `SERVER_HANDOFF_DELIVERED.md` 已替换完整 SHA。本次没有登录/同步服务器、没有正式训练，也没有最终 test。默认主候选为 **CBR + LIF-Down + RCS-Q** (`cbr_lif_rcsq_v1`)；单模块仅准备。

交付 SHA：`@DELIVERY_SHA@`。基点：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。将完整小交付目录上传为 `@DELIVERY_DIR@`；含 SHA、同步脚本、task.env、manifest、可选精简 bundle，不含权重/数据/参考模块 ZIP。

## 1. 新终端：校验交付并建立独立 worktree

```bash
set -Eeuo pipefail
cd '@DELIVERY_DIR@'
/root/miniconda3/envs/rtdetr/bin/python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path.cwd()
m=json.loads((root/'manifest.json').read_text())
assert m['commit']=='@DELIVERY_SHA@'
for name,row in m['files'].items():
    p=root/name
    assert p.stat().st_size==row['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
print('Delivery manifest verified:',m['commit'])
PY
test "$(tr -d '\r\n' < DELIVERY_SHA.txt)" = '@DELIVERY_SHA@'
bash ./sync_rcsq_v1.sh '@DELIVERY_SHA@' \
  '/root/autodl-tmp/projects/Crack_RTDETR' \
  '/root/autodl-tmp/projects/Crack_RTDETR-rcsq-v1' \
  '@DELIVERY_DIR@/rcsq_v1.bundle'
```

同步先检查固定对象；仅缺失才 fetch 目标分支 `exp-rtdetr-r18-lite-rcsq-v1`。使用 HTTP/1.1、`--progress --no-tags`、`GIT_TERMINAL_PROMPT=0`、lowSpeedLimit=1、lowSpeedTime=60、每次 timeout 120 秒、最多 5 次；失败记录保留。fetch 失败后可以用相同 SHA 的 bundle，先校验必要 base 和 `git bundle verify`。不会 reset/clean/覆盖已有不同 HEAD 或修改过的 worktree；冲突时改用新的专属目录并相应更新 task.env。只有 worktree 创建成功且 HEAD 精确匹配后才创建 weights/outputs。不得预先把权重复制进尚未创建的目标 worktree。

## 2. 验证既有环境、源码来源与固定权重

每个新终端都先 source 交付的 task.env；其中显式激活 `/root/miniconda3` 的既有 `rtdetr` 环境并设置当前 worktree PYTHONPATH。不要安装/升级 torch 或 Ultralytics。

```bash
source '@DELIVERY_DIR@/task.env'
python - <<'PY'
import os,pathlib,sys,torch,ultralytics,hashlib
root=pathlib.Path(os.environ['RCSQ_ROOT']).resolve()
assert pathlib.Path(ultralytics.__file__).resolve()==root/'ultralytics-main/ultralytics/__init__.py'
p=pathlib.Path(os.environ['RCSQ_SOURCE'])
assert hashlib.sha256(p.read_bytes()).hexdigest()=='fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e'
print('python',sys.executable,sys.version)
print('ultralytics',ultralytics.__file__)
print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available())
assert torch.cuda.is_available()
print('GPU',torch.cuda.get_device_name(0))
print('Source SHA256 verified')
PY
python tools/train_rcsq_v1.py amp-resources --main "$RCSQ_MAIN" \
  --report "$RCSQ_ROOT/outputs/rcsq_v1/amp_resources.json"
```

历史服务器为 Python 3.10.13、torch 2.1.2+cu121、RTX 4090，上面实际输出才是本次环境事实。AMP 资源入口沿用原受检资源准备路径，优先使用主仓库现有 bus.jpg/yolo26n.pt；必要时记录官方来源下载。服务器预检会执行原生 AMP 检查，不允许 monkeypatch。

## 3. 在服务器受控生成两个初值

```bash
source '@DELIVERY_DIR@/task.env'
python tools/init_rcsq_v1.py --source "$RCSQ_SOURCE" \
  --init-dir "$RCSQ_INIT" --variant both
```

检查初始化 JSON 的公共状态逐值一致、两个 RCS-Q 相同、58,370 参数、真实 Trainer nc=1 类别适配，以及正式训练更新=0。初值目录已存在时入口拒绝覆盖；检查既有审计后使用新的专属目录，不能修改 JSON 假装通过。无需上传 factors 文件。

## 4. 主候选服务器预检，再生成 plan

以下预检执行真实训练列表的完整组合、B16/640、AMP、原 DN 一次前向和反向，**不做正式 optimizer.step**。保留 GT 数量、实际 DN 查询长度、峰值显存、非确定性警告、FP32/AMP/非零分支融合证据和原生 AMP 检查结果。OOM 应报告并处理环境原因，不能自行降低正式配方。真实数据的 train/val/test 路径和标签指纹要与历史审计相符；仅 train 数据用于容量样本。

```bash
source '@DELIVERY_DIR@/task.env'
python tools/preflight_rcsq_v1.py --source "$RCSQ_SOURCE" \
  --init-dir "$RCSQ_INIT" --output "$RCSQ_PREFLIGHT" \
  --variant cbr_lif_rcsq_v1 --device cuda --data "$RCSQ_DATA" \
  --capacity --main "$RCSQ_MAIN"
python tools/train_rcsq_v1.py plan --variant cbr_lif_rcsq_v1 \
  --source "$RCSQ_SOURCE" --init-dir "$RCSQ_INIT" \
  --c2-args "$RCSQ_ROOT/docs/rcsq_v1/cbr_lif_authoritative_args.yaml" \
  --data "$RCSQ_DATA" --project "$RCSQ_ROOT/runs/rcsq_v1" \
  --output "$RCSQ_PLAN" --main "$RCSQ_MAIN"
```

检查 `server_preflight/checks.json` 顶层必须 PASSED；`LOCAL_PASSED_SERVER_PENDING` 不能启动正式训练。plan 的逐字段差异仅可涉及模型身份/输出位置/已核验环境路径。run 名为 `cbr_lif_rcsq_v1_rtdetr_r18_lite_e200_b16_onlineaug`，`exist_ok=False`。任何代码/源/初值/数据指纹变化都须重新生成专属审计和 plan。

单模块使用同一源和规则完成服务器预检，保存不同目录；不生成/启动额外正式训练。

```bash
source '@DELIVERY_DIR@/task.env'
python tools/preflight_rcsq_v1.py --source "$RCSQ_SOURCE" \
  --init-dir "$RCSQ_INIT" --output "$RCSQ_ROOT/outputs/rcsq_v1/server_preflight_single" \
  --variant rcsq_v1 --device cuda --data "$RCSQ_DATA" --capacity --main "$RCSQ_MAIN"
```

## 5. 单独保留的正式 tmux start 命令

**本次不执行本节。** 待用户决定启动且所有服务器预检通过后，在新终端执行。start 校验预检与完整指纹，并重新运行实时服务器门禁，不接受仅手改报告状态；再次门禁不更新正式权重。现有同名 tmux/session/run 不覆盖，也不停止其他 GPU 任务。

```bash
set -Eeuo pipefail
tmux new-session -d -s rcsq-v1-main \
  "bash -lc 'source @DELIVERY_DIR@/task.env; set -o pipefail; python tools/train_rcsq_v1.py start --plan \"\$RCSQ_PLAN/plan.json\" --preflight \"\$RCSQ_PREFLIGHT/checks.json\" 2>&1 | tee \"\$RCSQ_ROOT/outputs/rcsq_v1/train-console.log\"'"
tmux attach-session -t rcsq-v1-main
```

训练退出后以 `training_result.json` 和实际 epochs/早停/异常状态判断，不能只看进程退出。如果 start 的实时预检因瞬时环境问题失败，且没有建立正式 run、状态为 `PREFLIGHT_FAILED/NOT_STARTED`、原进程已经结束，修复原因后可显式重试：

```bash
source '@DELIVERY_DIR@/task.env'
python tools/train_rcsq_v1.py start --plan "$RCSQ_PLAN/plan.json" \
  --preflight "$RCSQ_PREFLIGHT/checks.json" --retry-preflight
```

入口保留失败目录和预约信息至 `plan/failed_preflight_时间戳`，重新运行完整实时门禁；不清空或覆盖旧证据。有正式 checkpoint 时使用独立 resume，不能用 retry-preflight 重建初值。start/resume 均持有该 run 的内核进程锁，已有训练或恢复进程存活时会拒绝第二个进程。

中断续训加载本任务非零已学习 checkpoint，不运行初始化覆盖它：

```bash
source '@DELIVERY_DIR@/task.env'
python tools/train_rcsq_v1.py resume --plan "$RCSQ_PLAN/plan.json" \
  --checkpoint "$RCSQ_ROOT/runs/rcsq_v1/cbr_lif_rcsq_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/last.pt"
```

## 后续评估与轻量打包

本次最终 test 为 NOT_RUN。下面评估命令仅留给本任务训练结束后执行，入口从 plan 指向的真实 best/EMA 加载，核对完成状态并冻结 checkpoint 与协议；训练/val/test 产物分目录，报告 P/R/AP50/AP75/AP50:95、速度、曲线与混淆矩阵。test 要求相同 checkpoint 的 val 报告。

```bash
source '@DELIVERY_DIR@/task.env'
python tools/eval_rcsq_v1.py --plan "$RCSQ_PLAN/plan.json" --split val \
  --output "$RCSQ_ROOT/outputs/rcsq_v1/evaluation_val" --device 0
python tools/eval_rcsq_v1.py --plan "$RCSQ_PLAN/plan.json" --split test \
  --output "$RCSQ_ROOT/outputs/rcsq_v1/evaluation_test" --device 0 \
  --val-report "$RCSQ_ROOT/outputs/rcsq_v1/evaluation_val/metrics.json"
```

仅打包已存在的证据，不触发训练或评估。预检阶段可以省略 train/val/test；下例为训练和评估结束后的轻量包。默认排除权重、数据集、参考 ZIP、逐图预测和完整日志，输出 manifest 与 SHA256。

```bash
source '@DELIVERY_DIR@/task.env'
python tools/pack_rcsq_v1_light.py \
  --preflight "$RCSQ_PREFLIGHT" --init-dir "$RCSQ_INIT" --plan-dir "$RCSQ_PLAN" \
  --train "$RCSQ_ROOT/runs/rcsq_v1/cbr_lif_rcsq_v1_rtdetr_r18_lite_e200_b16_onlineaug" \
  --val "$RCSQ_ROOT/outputs/rcsq_v1/evaluation_val" \
  --test "$RCSQ_ROOT/outputs/rcsq_v1/evaluation_test" \
  --output "$RCSQ_ROOT/outputs/rcsq_v1/rcsq_v1_LIGHT_completed.tar.gz"
```

## 可选轻量诊断

以下入口读取指定 checkpoint 的 model/EMA 副本，用显式诊断 forward 输出汇总 JSON，区分 DN/normal 的 null 权重、有效点数、原 query/修正范数、out_proj 范数和梯度状态；不保存整批特征或新权重。`--backward` 仅在可丢弃副本上执行原损失反传，optimizer steps 恒为 0。`--data` 只读取两个 train 样本并 resize；省略则是明确标注的合成 fixture。它不替代真实在线增强 B16/640 容量检查。

```bash
source '@DELIVERY_DIR@/task.env'
python tools/diagnose_rcsq_v1.py \
  --checkpoint "$RCSQ_INIT/cbr_lif_rcsq_v1_controlled_init.pt" \
  --output "$RCSQ_ROOT/outputs/rcsq_v1/diagnostic_init.json" \
  --data "$RCSQ_DATA" --device cuda --imgsz 160 --backward
```

`--device` 接受 `cpu` 或 `cuda`；可以省略 `--backward` 只看前向统计。后续需要观察学习后的分支时，将 checkpoint 改为本任务真实 best/last，并使用新的 JSON 路径。该入口不读取 test 样本，不写回 checkpoint，不自动进行诊断优化。

## 提交后重建交付

开发机在已提交、干净的 `exp-rtdetr-r18-lite-rcsq-v1` 工作树运行 `python tools/make_rcsq_v1_delivery.py --output outputs/rcsq_v1/delivery`。精简 bundle 基于当前 HEAD 排除 base 的既有历史；服务器必须已拥有 base，工具验证前提。交付目录 manifest 包含各文件 SHA256。推送远端 SHA 的验证由交付记录单独保存，不把尚未发生的 push 写成成功。
