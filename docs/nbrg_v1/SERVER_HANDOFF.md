# NBR-G v1 服务器交接

**本次没有远程同步、正式训练或最终 test。以下命令由用户之后在服务器执行。**

## 1. 固定交付 SHA，同步到独立工作树

仓库：`https://github.com/supershaojie/concrete-crack-rtdetr.git`。分支：`exp-rtdetr-r18-lite-nbrg-v1`。基点：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。

本地 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/nbrg-v1/outputs/nbrg_delivery/` 提供 `DELIVERY_SHA.txt`、`delivery.json`、两份同步脚本及小型离线 bundle。`DELIVERY_SHA.txt` 是实际完整 40 位交付 HEAD，不是需要猜测的示例值。将此小目录复制至服务器 `/root/autodl-tmp/nbrg_v1_delivery/`。

```bash
export NBRG_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export NBRG_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-nbrg-v1
export NBRG_SHA="$(cat /root/autodl-tmp/nbrg_v1_delivery/DELIVERY_SHA.txt)"
bash /root/autodl-tmp/nbrg_v1_delivery/sync_nbrg_v1.sh \
  "$NBRG_SHA" "$NBRG_MAIN" "$NBRG_WORKTREE"
cd "$NBRG_WORKTREE"
test "$(git rev-parse HEAD)" = "$NBRG_SHA"
```

薄入口复用现有 `sync_c19_lif_v1.sh` 的保护与 worktree 流程，以显式 `nbrg_v1` profile 选择分支。先查本地对象；缺失时只 fetch 该目标分支，`GIT_TERMINAL_PROMPT=0`、HTTP/1.1、lowSpeedLimit=1、lowSpeedTime=60、每次 timeout120 秒、最多 5 次。不会强制 push、覆盖其他 SHA、清理已有 worktree 或改动主检出。

若网络失败且服务器已有基点，按原离线 bundle→同一 worktree 流程导入：

```bash
git -C "$NBRG_MAIN" bundle verify /root/autodl-tmp/nbrg_v1_delivery/nbrg_v1.bundle
git -C "$NBRG_MAIN" fetch /root/autodl-tmp/nbrg_v1_delivery/nbrg_v1.bundle \
  exp-rtdetr-r18-lite-nbrg-v1:refs/remotes/nbrg-bundle/exp-rtdetr-r18-lite-nbrg-v1
bash /root/autodl-tmp/nbrg_v1_delivery/sync_nbrg_v1.sh \
  "$NBRG_SHA" "$NBRG_MAIN" "$NBRG_WORKTREE"
```

## 2. 环境、权重与数据

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export YOLO_AUTOINSTALL=false
export PYTHONPATH="$NBRG_WORKTREE/ultralytics-main"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NBRG_SOURCE="$NBRG_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
export NBRG_DATA="$NBRG_MAIN/configs/crack_autodl.yaml"
export NBRG_C2_ARGS="$NBRG_MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
export NBRG_INIT="$NBRG_WORKTREE/outputs/nbrg_clean_init"
sha256sum "$NBRG_SOURCE"
python -c 'import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())'
```

源权重必须为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。缺失/不同哈希时停止并报告，不替换为 best/last 或其他模型。历史服务器环境为 Python3.10.13、torch2.1.2+cu121；本地通过的是 Windows/Python3.9.25/torch2.7.1+cu118，故服务器预检待执行，不可仅复用 Windows 通过状态。

数据 YAML 必须是 `names: {0: crack}`，train/val/test 分别指向 `images/train`、`images/val`、`images/test`，路径根对应同一份 `datasets/crack_det`。脚本核验原组合路径/标签指纹与 6048/1728/864 图片、45573/12840/6663 框数；只读指纹不代表已执行 test。权威 args 可从原运行 args.yaml 获取；若原路径已归档，可显式将 `NBRG_C2_ARGS` 指向内容一致的 `docs/nbrg_v1/authoritative_c2_args.yaml`，脚本仍逐字段严格核验。

## 3. 干净初始化与有限预检

以下目录要求全新；重复执行时保留旧目录并换新名字。

```bash
python tools/init_nbrg_v1.py --source "$NBRG_SOURCE" --output "$NBRG_INIT"
mkdir -p outputs/nbrg_logs
set -o pipefail
python -u tools/preflight_nbrg_v1.py --mode cpu \
  --source "$NBRG_SOURCE" --initial-dir "$NBRG_INIT" \
  --output outputs/nbrg_cpu_server |& tee outputs/nbrg_logs/cpu_server.log
python -u tools/preflight_nbrg_v1.py --mode cuda \
  --source "$NBRG_SOURCE" --initial-dir "$NBRG_INIT" --data "$NBRG_DATA" \
  --output outputs/nbrg_cuda_server |& tee outputs/nbrg_logs/cuda_server.log
python -u tools/preflight_nbrg_v1.py --mode capacity --variant cbr_lif_nbr_control_v1 \
  --source "$NBRG_SOURCE" --initial-dir "$NBRG_INIT" --data "$NBRG_DATA" \
  --output outputs/nbrg_capacity_control_server |& tee outputs/nbrg_logs/capacity_control_server.log
python -u tools/preflight_nbrg_v1.py --mode capacity --variant cbr_lif_nbrg_v1 \
  --source "$NBRG_SOURCE" --initial-dir "$NBRG_INIT" --data "$NBRG_DATA" \
  --output outputs/nbrg_capacity_candidate_server |& tee outputs/nbrg_logs/capacity_candidate_server.log
```

逐条串行执行，上一条失败先查看对应 `preflight.json` 和日志。不要同时运行多个 GPU 预检。容量模式固定 B16/640/AMP，OOM 标记 PENDING，不降低 batch、不关闭 AMP；源码/数据/初始化或 CUDA 环境变化会使旧证据失效。保存重载与 backward 使用独立检查对象，不回写 `NBRG_INIT`。

第三配置 `nbrg_v1` 已由默认 CPU/CUDA 预检覆盖。后续若要正式独立消融，先额外运行同一容量命令的 `--variant nbrg_v1`，输出独立目录。

## 4. 首轮两组正式命令（本次未执行）

| variant | YAML | 初值 | 正式运行名 |
|---|---|---|---|
| `cbr_lif_nbr_control_v1` | `rtdetr-resnet18-lite-cbr-lif-nbr-control-v1.yaml` | `$NBRG_INIT/cbr_lif_nbr_control_v1.pt` | `cbr_lif_nbr_control_v1_rtdetr_r18_lite_e200_b16_onlineaug` |
| `cbr_lif_nbrg_v1` | `rtdetr-resnet18-lite-cbr-lif-nbrg-v1.yaml` | `$NBRG_INIT/cbr_lif_nbrg_v1.pt` | `cbr_lif_nbrg_v1_rtdetr_r18_lite_e200_b16_onlineaug` |

仅准备计划时，将下面的 `start` 改成 `plan`，使用新的 metadata 目录；plan 不训练，不占用正式运行目录。

在全部对应服务器预检通过、用户决定启动之后执行。推荐沿用现有 tmux 使用方式，在 `tmux new -s nbrg-control` 的终端中运行第一组；完成后再运行第二组，避免单卡争抢。

```bash
set -o pipefail
python -u tools/train_nbrg_v1.py start --variant cbr_lif_nbr_control_v1 \
  --source "$NBRG_SOURCE" --initial-dir "$NBRG_INIT" --c2-args "$NBRG_C2_ARGS" \
  --data "$NBRG_DATA" --project "$NBRG_MAIN/runs/nbrg_v1/train" \
  --metadata "$NBRG_MAIN/runs/nbrg_v1/metadata/cbr_lif_nbr_control_v1" \
  --preflight outputs/nbrg_cpu_server/preflight.json outputs/nbrg_cuda_server/preflight.json \
    outputs/nbrg_capacity_control_server/preflight.json \
  |& tee outputs/nbrg_logs/control_formal.log
```

```bash
set -o pipefail
python -u tools/train_nbrg_v1.py start --variant cbr_lif_nbrg_v1 \
  --source "$NBRG_SOURCE" --initial-dir "$NBRG_INIT" --c2-args "$NBRG_C2_ARGS" \
  --data "$NBRG_DATA" --project "$NBRG_MAIN/runs/nbrg_v1/train" \
  --metadata "$NBRG_MAIN/runs/nbrg_v1/metadata/cbr_lif_nbrg_v1" \
  --preflight outputs/nbrg_cpu_server/preflight.json outputs/nbrg_cuda_server/preflight.json \
    outputs/nbrg_capacity_candidate_server/preflight.json \
  |& tee outputs/nbrg_logs/candidate_formal.log
```

第三组后续入口为同一 `train_nbrg_v1.py` 的 `--variant nbrg_v1`，YAML=`rtdetr-resnet18-lite-nbrg-v1.yaml`，初值=`$NBRG_INIT/nbrg_v1.pt`，独立 metadata、容量报告和运行名 `nbrg_v1_rtdetr_r18_lite_e200_b16_onlineaug`。

正式初值保持 nc=80 源适配路径；实际 Trainer 强制核验 nc=1、公共状态一致、gate 可训练且为零、原配方未漂移，然后构建 AdamW。native AMP 检查资源沿用原组合入口，不改变训练架构。完整 actual args 和 optimizer audit 落在 metadata，`args.yaml`、`results.csv`、best/last 留在各正式训练目录。

训练正常早停保留 patience=50，记录实际 epoch 和原因，不能称为跑满200轮。崩溃、停止及中断查看 `training_state.json`，不能作完成结果。最终 test 输出预留在 `$NBRG_MAIN/runs/nbrg_v1/test/<variant>/`，沿用原评估协议，由后续明确要求再执行；此次无 test 命令。
