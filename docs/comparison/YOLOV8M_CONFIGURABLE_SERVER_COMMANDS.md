# YOLOv8m configurable：固定 SHA 与服务器命令

工具分支：`bench/yolov8m-configurable`。实际训练代码 SHA：`a7b3d842df60adf28e5cf35c086c58ef752674f8`；父 SHA：`16a2e28686b3471ef1b7640b3ba697597335895c`（其父 `05c6b4c8aeaf04255b7e17f3391d2202e65d9ffe`）。后续文档/归档提交不充当历史训练 SHA。以下为服务器 Bash 命令；本次没有连接服务器、改变现有 pilot 或启动正式训练。

## 1. 一次性拉取固定 SHA，准备独立工作树

目标路径存在时必须已经是同一干净代码；命令不会覆盖它。主项目、旧 pilot 工作树和它们的 HEAD 保持原状。

~~~bash
set -euo pipefail
REPO=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-configurable
CODE_SHA=a7b3d842df60adf28e5cf35c086c58ef752674f8
PARENT_SHA=16a2e28686b3471ef1b7640b3ba697597335895c

git -C "$REPO" remote -v
test "$(git -C "$REPO" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$REPO" fetch origin bench/yolov8m-configurable
git -C "$REPO" cat-file -e "$CODE_SHA^{commit}"
test "$(git -C "$REPO" show -s --format=%P "$CODE_SHA")" = "$PARENT_SHA"
if [[ -e "$WORK" ]]; then
    test "$(git -C "$WORK" rev-parse HEAD)" = "$CODE_SHA"
    test -z "$(git -C "$WORK" status --porcelain --untracked-files=normal)"
else
    git -C "$REPO" worktree add --detach "$WORK" "$CODE_SHA"
fi
cd "$WORK"
test "$(git rev-parse HEAD)" = "$CODE_SHA"
test -z "$(git status --porcelain --untracked-files=normal)"
~~~

后续候选复用此 SHA/worktree，不因数值配方改代码、切分支、重建环境或下载。下面代码块默认同一 Shell 已初始化变量；重连 SSH 后重设第2节变量并进入 WORK，不重复创建已有候选/run。

## 2. 只读复用环境、源码、COCO 原始权重与数据

~~~bash
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-configurable
PILOT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-coco-b19-pilot
PY="$PILOT/.envs/yolov8m-coco-b19-pilot/bin/python"
SOURCE="$PILOT/.vendor/yolov8m-coco-b19-pilot/ultralytics-v8.3.20"
WEIGHT="$PILOT/.runtime/yolov8m-coco-b19-pilot/assets/yolov8m.pt"
DATA=/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml
DATA_ROOT=/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
ENTRY="$WORK/benchmarks/comparison/yolov8m/run.py"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$WORK"
test -x "$PY"
test -d "$SOURCE"
test -f "$WEIGHT"
test -f "$DATA"
test -d "$DATA_ROOT"
command -v tmux
command -v flock
command -v tee
mkdir -p outputs/yolov8m-configurable-asset-checks
CHECK="outputs/yolov8m-configurable-asset-checks/check_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${RANDOM}.json"
"$PY" "$ENTRY" check --source "$SOURCE" --weights "$WEIGHT" \
    --config benchmarks/comparison/yolov8m/default_config.yaml --output "$CHECK"
~~~

这是导入、完整源码身份、环境与原始 COCO 迁移核验，不训练。上游固定 Ultralytics v8.3.20 / `f4d8f7765a490f3920e2d14c592a2967e347f185`；兼容补丁不变。检测 CutMix 摘录固定 v8.4.0 / `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`，不安装整套新版。原始权重须为 52,136,884 bytes、SHA256 `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`；实际迁移 469/475、nc1 未融合参数 25,856,899。

新启动器直接使用上述默认路径，不调用 bootstrap、pip install 或下载。preflight 自动查找旧 pilot 的 `outputs/yolov8m-coco-b19-pilot/*` 中符合此数据根和正式计数的输入缓存；只复用 manifest、GT、尺寸/标签缓存，不取旧 run 的训练配置。可在 start 时明确指定 `--reuse-run /absolute/pilot/run`，以及必要的 `--python/--source/--weights/--data/--data-root/--public-coco` 路径。这些路径一起冻结，resume/finalize 不接受更换。

复用时核对实际 YAML 路径顺序、图片 stat、标签字节、清单/GT 校验和及正式数据身份；旧尺寸/标签缓存只读复用。无匹配缓存时仅建立本 run 必要的小型缓存，不哈希图片内容、复制图片库或改数据。固定 train6048/45573框、val1728/12840框、test864/6663框，YOLO id0→公共 category_id1。任何核验失败都会报错并保留原因。

## 3. 创建两个候选 YAML

首份完整入口与 pilot 的所有开放字段一致；它是默认起点，没有最优参数声明，不改变当前 pilot。

~~~bash
cd "$WORK"
mkdir -p runtime_configs
test ! -e runtime_configs/v8_cfg_01.yaml
cp benchmarks/comparison/yolov8m/default_config.yaml runtime_configs/v8_cfg_01.yaml
"$PY" - <<'PY'
from pathlib import Path
import yaml
dst = Path("runtime_configs/v8_cfg_02.yaml")
if dst.exists():
    raise FileExistsError(dst)
cfg = yaml.safe_load(Path("benchmarks/comparison/yolov8m/default_config.yaml").read_text())
cfg.update(lr0=0.008, cutmix=0.05)  # 示例，不代表改进或最优
with dst.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
PY
~~~

支持 epochs、patience（0 禁用早停）、close_mosaic（0 不关闭）、workers、nbs、SGD/Adam/AdamW、lr0/lrf/momentum/weight_decay、warmup、cos_lr、原生 box/cls/dfl、HSV、几何、翻转及 Mosaic/MixUp/CutMix 概率。Adam/AdamW 的 momentum 是原生 beta1；beta2=0.999、eps=1e-8 固定。warmup_momentum 仅用于 SGD，Adam/AdamW 中不可改成其他值。optimizer=auto、分类 erasing/auto_augment、额外 beta2/eps/nesterov 等字段不开放。未知/重复字段、错误类型/范围和不支持组合均报错。

正式运行固定 batch16、imgsz640、device0、seed42、AMP、deterministic=true、cache/rect/multi_scale=false。bgr=0、copy_paste=0 保持当前检测实现；非零值不能作为有效检测增强。default_config.yaml 是文档要求的完整首份 YAML，其余固定模型/协议字段由 recipe 和代码补全进 resolved_config.yaml。

## 4. start：每个 run 独立 tmux，训练后仅公共 val

~~~bash
cd "$WORK"
bash scripts/autodl_yolov8m_configurable.sh start \
    --run-id v8_cfg_01 --config "$WORK/runtime_configs/v8_cfg_01.yaml"
bash scripts/autodl_yolov8m_configurable.sh start \
    --run-id v8_cfg_02 --config "$WORK/runtime_configs/v8_cfg_02.yaml"
~~~

两候选共享 GPU0，分别持有自己的 session/run 锁；无 GPU 空闲门槛，OOM 真实失败，不降 batch、换优化器或改分辨率。start 在派发之前独占创建 run，复制原始 YAML，补齐默认值并冻结路径/配置哈希/唯一 UUID。随后轻量 preflight→原生训练→同一 best 的公共 val 导出/评估→调参汇总，**不自动评 test**。源 YAML 改动、移动或删除不影响已启动 run。

新候选从核验通过的原始 COCO 权重开始，加载全部兼容层，只保留正常固定 DFL 投影；不加载旧候选 best/last。保持 pilot 的预处理、HSV、几何、框裁切/过滤和检测 CutMix；CutMix 保留 train-only 配对、触发/应用/几何跳过计数。默认 zero-based epoch190（第191轮开始）关闭增强，修改配置后为 epochs-close_mosaic；训练及真实数据 worker 重建/reset 后同步关闭 Mosaic/MixUp/CutMix，HSV/几何/翻转继续执行。

run-id 允许 1–80 位 ASCII 字母、数字、下划线、连字符，首位为字母/数字；SMOKE_ 前缀保留给内部合成验证。同名 start 拒绝覆盖，活动 run 拒绝重复 resume/finalize。完成窗口保留真实退出状态，续操作在该 run 同一 session 创建新窗口；不杀其他会话。每阶段 command/tee 退出码分别记录。

~~~bash
tmux list-sessions
tmux attach -t comparison-yolov8m-v8_cfg_01
# 脱离：先 Ctrl+b，再 d
tmux attach -t comparison-yolov8m-v8_cfg_02
# Ctrl+b、d 脱离

tmux capture-pane -p -S -100 -t comparison-yolov8m-v8_cfg_01
tmux list-panes -s -t comparison-yolov8m-v8_cfg_01 \
    -F '#{window_index}:#{pane_index} dead=#{pane_dead} exit=#{pane_dead_status} command=#{pane_current_command}'

RUN="$WORK/outputs/yolov8m-configurable/v8_cfg_01"
ls -lt "$RUN/launch"
"$PY" -m json.tool "$RUN/train_status.json"
"$PY" -m json.tool "$RUN/summary.json"
# 日志：$RUN/launch/start_<时间>_<随机数>/train.log
# 同目录 *.exit.json、pipeline_status.json、final_summary.log
# 原生 CSV：$RUN/train/results.csv；best/last：$RUN/train/weights/
~~~

原生每轮验证及 best/早停沿用 pilot 的流程，以未舍入的原生验证 mAP50–95 选 best（相同值取后轮），原生 weighted fitness 另记。真实调用 batch16、rectFalse、conf0.001、iou0.7、max_det300 和输入 dtype 记录于 native_validation.json。保留原生 AMP 精度行为，不声称原生验证恒为 FP32。公共 val 用于统一报告；选参依据同一验证集 mAP50–95 与收敛表现，不预设指标排序。

## 5. 同 run 恢复

仅对已中断且有可恢复 last 的 run 使用。训练已合法完成而公共 val 失败时，resume 跳过训练补 val；仍不评 test。

~~~bash
cd "$WORK"
bash scripts/autodl_yolov8m_configurable.sh resume --run-id v8_cfg_01
tmux attach -t comparison-yolov8m-v8_cfg_01
~~~

无需原候选 YAML。要求原训练 SHA、冻结配置/路径/UUID、数据、源码/补丁、环境一致，只读自己的 last。恢复 FP32 模型/optimizer/EMA、scaler、scheduler、RNG、早停和增强状态；保留 initialization.json 并追加日志。CSV 若晚于 checkpoint，先备份，再恢复 checkpoint 已完成行后追加。raw decay/loss 不从有效值重复缩放，各组 LR/accumulation/effective decay 分别记录。

这是原生 epoch-boundary resume，不承诺逐位重放预取批次或跨 epoch 未提交梯度。完整结束、早停、人为中断分别记录，中断不写成跑满200轮。若 last 已到 epoch/patience 终点但中断导致终态记录缺失，会明确拒绝继续训练，需检查终态证据；不会越过停止规则。参数改变使用新 YAML/new run-id，从 COCO 开始。

## 6. 选定 run 的 finalize：同一 best，补公共 test

按验证集 mAP50–95/收敛表现选定已合法结束的 run 后执行：

~~~bash
cd "$WORK"
bash scripts/autodl_yolov8m_configurable.sh finalize --run-id v8_cfg_01
tmux attach -t comparison-yolov8m-v8_cfg_01
# Ctrl+b、d 脱离
"$PY" -m json.tool "$WORK/outputs/yolov8m-configurable/v8_cfg_01/summary.json"
~~~

finalize 不训练，校验并复用同一 best 的公共 val，再补 test。公共协议固定 FP32 参数/输入、640、batch16、conf0.001、类内 NMS0.7、max_det300、无 TTA；rounded resize gain(x/y)、整数 padding 逆变换、浮点框、完整图像/空预测及 padding-only false positive 处理不变，复用 corrected_sorted_conf_mask_v1，不改 AP。

缓存仅在配置/UUID/代码/环境/checkpoint/GT/协议/内容校验和匹配时复用；不匹配失败并保留证据。summary 分别显示 training_status、public_val_status、final_test_status、tuning_status。调参完成、test 未执行是正常 tuning_completed / final_test_status=not_requested；test 失败不改写合法训练状态，完成 test 后为 finalized。

## 7. 正式配置与结果归档：独立开发/归档工作树

只在选定 run 完成 finalize 后执行。活动训练 WORK 保持原 SHA；只提交小型配置/摘要，不提交大权重、全量预测缓存或日志。

~~~bash
REPO=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-configurable
CODE_SHA=a7b3d842df60adf28e5cf35c086c58ef752674f8
RUN_ID=v8_cfg_01
RUN="$WORK/outputs/yolov8m-configurable/$RUN_ID"
PILOT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-coco-b19-pilot
PY="$PILOT/.envs/yolov8m-coco-b19-pilot/bin/python"
ARCHIVE_BRANCH="codex/yolov8m-configurable-archive-$RUN_ID"
ARCHIVE_WORK="/root/autodl-tmp/projects/Crack_RTDETR-yolov8m-archive-$RUN_ID"

"$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
row = json.loads((run/"summary.json").read_text())
assert row["training_status"] == "completed" and row["status"] == "finalized", row
assert row["public_val_status"] == row["final_test_status"] == "completed", row
PY

test ! -e "$ARCHIVE_WORK"
git -C "$REPO" worktree add -b "$ARCHIVE_BRANCH" "$ARCHIVE_WORK" "$CODE_SHA"
DEST="$ARCHIVE_WORK/docs/comparison/yolov8m_configurable_runs/$RUN_ID"
mkdir -p "$(dirname "$DEST")"
mkdir "$DEST"
for name in user_config.yaml resolved_config.yaml config_identity.json run_id.json \
    identity.json initialization.json actual_training_setup.json native_validation.json \
    train_status.json export_val_status.json evaluate_val_status.json \
    export_test_status.json evaluate_test_status.json summary.json; do
    cp "$RUN/$name" "$DEST/$name"
done
mkdir "$DEST/metrics"
cp "$RUN/metrics/val.json" "$RUN/metrics/test.json" "$DEST/metrics/"

"$PY" - "$RUN" "$DEST" <<'PY'
import json, sys
from pathlib import Path
run, dst = map(Path, sys.argv[1:])
identity = json.loads((run/"identity.json").read_text())
result = {"run_id": identity["run_id"], "run_uuid": identity["run_uuid"],
          "actual_training_sha": identity["model_code_sha"],
          "config_sha256": identity["config_sha256"],
          "source_run": str(run), "archive_is_not_training_sha": True,
          "weights_and_full_logs_committed": False}
(dst/"archive_provenance.json").write_text(json.dumps(result, indent=2)+"\n")
PY

git -C "$ARCHIVE_WORK" add -- "docs/comparison/yolov8m_configurable_runs/$RUN_ID"
git -C "$ARCHIVE_WORK" diff --cached --stat
git -C "$ARCHIVE_WORK" commit -m "docs(bench): archive selected YOLOv8m configurable run $RUN_ID"
git -C "$ARCHIVE_WORK" push -u origin "$ARCHIVE_BRANCH"
~~~

归档保留原 identity/summary 的实际训练 SHA。后续候选仍在原 WORK 用新 YAML/new run-id 复用代码、环境和资产。预处理算法、权重迁移、模型实现或评估逻辑改变属于代码修改。

## 实际验证范围

33 项小型测试通过、1 项 POSIX 信号转发测试因 Windows 跳过：9 项配置/缓存/worker/启动保护，5 项 CutMix/坐标，4 项初始化 guard，10 项现有集成，5 项有效生命周期检查。真实 M 模型在 CPU、batch2/64、8 个合成训练图像上更新、保存、第1轮中断、恢复到第2轮；FP32 模型/optimizer/EMA/scaler/scheduler/RNG 恢复前逐项相等，raw decay0.0007 与 effective0.00056 不重复缩放。同一 best 完成两张合成 val/test 的公共 FP32 导出/评估。

同一干净代码还通过两个 YAML 的实际 prepare/guard 与 Bash start/resume/finalize 派发检查：源 YAML 改动/移动后快照不变，派发不含候选 YAML，session 分离；该派发检查使用 mock tmux/资产路径，不执行训练。真实原始资产由独立 check 和 M-model smoke 核验。服务器资产/实际 tmux、正式200轮与服务器双任务 batch16/640/AMP 显存容量均未在本轮验证。
