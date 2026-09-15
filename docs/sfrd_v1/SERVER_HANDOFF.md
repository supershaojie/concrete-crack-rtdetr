# SFR-D v1 服务器交接

本轮已实现、初始化和本机预检；未自动同步服务器，未启动正式训练、完整 val/test 或消融。首轮 B 槽为 `cbr_lif_sfrd_v1`。A 槽由 NBR-G 任务自己的交接负责，本文件不指定其 HEAD 或入口。

## 固定版本

- 本地 worktree：`D:/MyProjects/Crack_RTDETR/outputs/worktrees/Crack_RTDETR-sfrd-v1`
- 分支：`exp-rtdetr-r18-lite-sfrd-v1`
- origin：`https://github.com/supershaojie/concrete-crack-rtdetr.git`
- base：`a0459d6a652cb702699087c88fa39a3e4c4087ec`
- 本提交的完整 SHA 由此工作树的 `git rev-parse HEAD` 取得。提交后独立 `SERVER_HANDOFF_DELIVERED.md` 副本与最终回复写入真实 40 位 SHA；避免文档提交自身 SHA 的循环。
- 所有权重、谱因子、预检临时文件均不在 git 中。Git 只包含源码、YAML、小型审计和文档。

## 1. 手动同步独立工作树

以下沿用原 `sync_c19_lif_v1.sh` 的共享仓库、固定 SHA、新 worktree 和保留现有目录的方法，按本任务要求加上目标 SHA 已存在检查及有界 fetch；不运行 C19 分支专用同步脚本。

先使用独立交接副本中已填好的 `export SFRD_SHA=...` 行，再执行：

```bash
set -Eeuo pipefail
: "${SFRD_SHA:?Use the exact export line in SERVER_HANDOFF_DELIVERED.md}"
[[ "$SFRD_SHA" =~ ^[0-9a-f]{40}$ ]]
SFRD_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SFRD_WT=/root/autodl-tmp/projects/Crack_RTDETR-sfrd-v1
SFRD_BRANCH=exp-rtdetr-r18-lite-sfrd-v1
test "$(git -C "$SFRD_MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
if ! git -C "$SFRD_MAIN" cat-file -e "$SFRD_SHA^{commit}" 2>/dev/null; then
  for SFRD_ATTEMPT in 1 2 3 4 5; do
    if timeout 120 env GIT_TERMINAL_PROMPT=0 git -C "$SFRD_MAIN" \
      -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head origin \
      "refs/heads/$SFRD_BRANCH:refs/remotes/origin/$SFRD_BRANCH"; then break; fi
  done
fi
git -C "$SFRD_MAIN" cat-file -e "$SFRD_SHA^{commit}"
git -C "$SFRD_MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SFRD_SHA"
git -C "$SFRD_MAIN" cat-file -e "$SFRD_SHA:tools/init_sfrd_v1.py"
if [[ -e "$SFRD_WT" ]]; then
  test -f "$SFRD_WT/.git"
  test "$(git -C "$SFRD_WT" rev-parse HEAD)" = "$SFRD_SHA"
  test "$(git -C "$SFRD_WT" rev-parse --path-format=absolute --git-common-dir)" = "$(git -C "$SFRD_MAIN" rev-parse --path-format=absolute --git-common-dir)"
  test -z "$(git -C "$SFRD_WT" status --porcelain --untracked-files=no)"
else
  git -C "$SFRD_MAIN" worktree add --detach "$SFRD_WT" "$SFRD_SHA"
fi
cd "$SFRD_WT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export PYTHONPATH="$SFRD_WT/ultralytics-main"
export YOLO_AUTOINSTALL=false
```

fetch 5 次失败即停止。既有离线方式是在本地独立 worktree 执行 `git bundle create outputs/sfrd_v1.bundle exp-rtdetr-r18-lite-sfrd-v1`，手动上传到服务器 `$SFRD_MAIN/outputs/sfrd_v1.bundle`，运行 `git bundle verify`、从 bundle fetch 目标分支，再以同一 SHA 创建独立 worktree。源码 bundle 是离线交付物，不能与默认≤5MiB的故障诊断包混淆。禁止 reset/clean、强制 stash/切分支、force push、合并或删除已有 worktree。

## 2. 真实因子与初始化

本机已经生成三份独立 nc80 初值：

```text
weights/sfrd_v1/cbr_lif_sfrd_v1_init.pt
weights/sfrd_v1/cbr_lif_sfr_control_v1_init.pt
weights/sfrd_v1/sfrd_v1_init.pt
weights/artifacts/sfrd_v1/factors.pt
weights/artifacts/sfrd_v1/factors.json
```

请用现有手动文件传输方式，将本机**实际** `factors.pt` 和 `factors.json` 上传到新服务器 worktree 的同一相对目录。两个文件必须一起传输，以 [svd_audit.json](svd_audit.json) 中 `file_sha256` 校验；无需传完整模块包。FP32 A/B 合计1,572,864个元素，约6MiB，独立保存，不放入故障包。符号规范不能保证不同数值库对重根给出同一基，因此优先复用实际缓存。

在源权重、因子文件就绪后生成服务器自己的三组初值及审计。首次执行输出目录必须不存在相应 `_init.pt`；已有文件时保留并核查，重试使用新目录。

```bash
test "$(sha256sum "$SFRD_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" | cut -d' ' -f1)" = fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
test -f weights/artifacts/sfrd_v1/factors.pt
test -f weights/artifacts/sfrd_v1/factors.json
python tools/init_sfrd_v1.py \
  --source "$SFRD_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --cache weights/artifacts/sfrd_v1/factors.pt \
  --output-dir weights/sfrd_v1 \
  --report-dir outputs/sfrd_v1_initialization
```

若实际缓存无法传输，可显式选择全新的 `weights/artifacts/sfrd_v1_server/factors.pt` 路径重算 CPU FP64 SVD并审计；同一次结果仍必须供服务器三变体共同复用，并保留本机/服务器因子差异记录。不能覆盖已交付缓存或宣称不同设备逐位相同。

## 3. 服务器必要预检

先确认 GPU 未被另一任务预检占用；若最终两任务共卡，两个任务各自保持 batch16，并核实总显存可承受。不要中止 NBR-G 或修改其 batch/AMP。

```bash
python tools/preflight_sfrd_v1.py \
  --source "$SFRD_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --cache weights/artifacts/sfrd_v1/factors.pt \
  --initialized weights/sfrd_v1/cbr_lif_sfrd_v1_init.pt \
  --data "$SFRD_MAIN/configs/crack_autodl.yaml" \
  --output outputs/sfrd_v1_server_preflight --cuda --capacity-b16
```

必须 `preflight.json.status=PASSED` 且 capacity 为 CUDA、640、batch16、AMP=true、optimizer_steps=0；若 OOM、数据缺失或其他失败，保留证据，不改配方。小 batch CUDA/CPU 通过不替代容量检查。每次重跑选择新 output 目录，不能把旧失败删掉冒充初次通过。

数据采用已有 `crack_autodl.yaml` 的 nc1、images/train/val/test。工具核对原组合提交的三划分路径和全部标签指纹，禁止重划；只是读取全量路径/标签，不做全量模型评估。

## 4. 审阅配方与之后的正式训练

以下 `plan` 可单独运行，不训练：

```bash
python tools/train_sfrd_v1.py plan \
  --variant cbr_lif_sfrd_v1 \
  --initialized weights/sfrd_v1/cbr_lif_sfrd_v1_init.pt \
  --c2-args "$SFRD_MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml" \
  --data "$SFRD_MAIN/configs/crack_autodl.yaml" \
  --project "$SFRD_MAIN/runs/c_series" \
  --output outputs/sfrd_v1_server_plan
```

本轮不要执行以下正式命令。后续用户明确启动时，在既有 SSH 独立 tmux 会话中执行，保留 console.log：

```bash
python -u tools/train_sfrd_v1.py train \
  --variant cbr_lif_sfrd_v1 \
  --initialized weights/sfrd_v1/cbr_lif_sfrd_v1_init.pt \
  --c2-args "$SFRD_MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml" \
  --data "$SFRD_MAIN/configs/crack_autodl.yaml" \
  --project "$SFRD_MAIN/runs/c_series" \
  --output outputs/sfrd_v1_training \
  --preflight outputs/sfrd_v1_server_preflight/preflight.json
```

沿用成功服务器 Python3.10.13／torch2.1.2+cu121 环境；不自动升级依赖。原生 Trainer 重新从干净初值构建 nc1，先审计公共状态、A/B、零 gate，再创建优化器；进入正式训练前再次核对初值、实际 args、AMP、batch 和所有参数的优化器包含关系。原 OOM 自动降 batch 路径被父项目已有回调禁止。初始化没有优化器、scaler 或 EMA 训练状态，正式训练不复用预检对象。

主训练输出目录为 `$SFRD_MAIN/runs/c_series/cbr_lif_sfrd_v1_rtdetr_r18_lite_e200_b16_onlineaug`，保留 args.yaml、results.csv、weights/best.pt、weights/last.pt。launch、初值审计、优化器审计和退出状态位于 `outputs/sfrd_v1_training`。正常早停允许完成，记录实际轮次和退出原因；不能把早停写成跑满200轮。

### 后续控制与独立消融

三配置与三组初始化已由上述统一命令生成并核验。纯 SFR 与独立 SFR-D 本轮仅要求构建、初始化、640输出和新进程重载；完整训练前需要补充对应 variant 的损失/融合/容量预检。当前完整 preflight 入口明确只接受首轮主组合，不能拿主组合报告冒充另外两组容量证据。

可执行的后续配方审阅命令：

```bash
for SFRD_VARIANT in cbr_lif_sfr_control_v1 sfrd_v1; do
  python tools/train_sfrd_v1.py plan \
    --variant "$SFRD_VARIANT" \
    --initialized "weights/sfrd_v1/${SFRD_VARIANT}_init.pt" \
    --c2-args "$SFRD_MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml" \
    --data "$SFRD_MAIN/configs/crack_autodl.yaml" \
    --project "$SFRD_MAIN/runs/c_series" \
    --output "outputs/${SFRD_VARIANT}_later_plan"
done
```

最终结构确定后，按父项目 sorted-confidence mask、锁定 best SHA、独立 val 后 test 的流程接入评估；当前 C19 专用 results 入口会拒绝 SFR 拓扑，不能直接冒用该入口。本轮不执行评估/打包，不生成论文结构图。

## 5. 本地证据与故障包

本机 Python3.9.25、torch2.7.1+cu118、RTX2060 上最终 `status=PASSED`。CPU/CUDA FP32 初始和非零 gate 整网融合、非零 gate 新进程恢复、原生 Trainer/EMA/AdamW、真实检测 loss/匹配、CUDA AMP 和 640/batch16/AMP 单批均通过。该单批未做 optimizer.step，峰值CUDA分配6,532,433,408 bytes；不能用它替代服务器 Linux 环境及双任务同时运行的显存补检。初始权重与因子在预检后逐文件哈希保持不变。

结果见 [preflight.json](preflight.json)、[complexity.json](complexity.json)、[initialization_audit.json](initialization_audit.json)、[svd_audit.json](svd_audit.json)。完整失败 fixture 只保留在 ignored outputs；源码提交只包含必要小型诊断。默认故障包≤5MiB，仅 JSON、日志尾、必要源码与 manifest，禁止包含原权重、SVD因子、图像、完整runs。
