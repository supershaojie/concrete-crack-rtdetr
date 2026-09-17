# CSR-P3 服务器命令

此文件的交付版本由 `tools/render_csr_p3_handoff.py` 填入完整提交。
固定 HEAD：`@HEAD@`；基点：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
主组合 `cbr_lif_csr_p3_v1`，首轮只执行该组合。正式训练 NOT_STARTED，test NOT_RUN。
所有段落单独复制；每段都在子 bash 中运行，失败不会关闭交互终端。

## 1. 同步与环境（下一条命令）

优先使用主仓库已有对象；缺失才最多5次有界 fetch。网络仍失败时，先上传交付的
`csr_p3_v1.bundle` 到 `/root/autodl-tmp/csr_p3_v1.bundle`，然后重跑同一段。
该增量包以成功基点为前提，bundle SHA256 为 `@BUNDLE_SHA@`，本机已实际导入验证。
不 clone、不重置已有 worktree。

```bash
bash <<'CSR_SYNC'
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SHA=@HEAD@
BRANCH=exp-rtdetr-r18-lite-csr-p3-v1
BUNDLE=/root/autodl-tmp/csr_p3_v1.bundle
cd "$MAIN"
REMOTE=$(git remote get-url origin)
case "$REMOTE" in
  https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) printf 'Unexpected origin: %s\n' "$REMOTE" >&2; exit 1 ;;
esac
if ! git cat-file -e "$SHA^{commit}" 2>/dev/null; then
  for attempt in 1 2 3 4 5; do
    printf 'Fetch attempt %s/5\n' "$attempt"
    if timeout 120s env GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags origin \
      "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"; then :; fi
    if git cat-file -e "$SHA^{commit}" 2>/dev/null; then break; fi
  done
fi
if ! git cat-file -e "$SHA^{commit}" 2>/dev/null; then
  test -f "$BUNDLE"
  printf '%s  %s\n' '@BUNDLE_SHA@' "$BUNDLE" | sha256sum -c -
  git bundle verify "$BUNDLE"
  git fetch --no-tags "$BUNDLE" "refs/heads/$BRANCH:refs/remotes/csr-bundle/$BRANCH"
fi
git cat-file -e "$SHA^{commit}"
SCRIPT=$(mktemp)
trap 'rm -f -- "$SCRIPT"' EXIT
git show "$SHA:tools/sync_csr_p3.sh" > "$SCRIPT"
bash "$SCRIPT" "$SHA"
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
python - <<'CSR_ENV_PY'
import pathlib, torch, ultralytics
from ultralytics.nn.modules import csr_p3
root=pathlib.Path.cwd()
assert pathlib.Path(ultralytics.__file__).resolve().is_relative_to(root/'ultralytics-main')
print('ultralytics:', ultralytics.__file__)
print('CSR:', csr_p3.__file__)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'PENDING: CUDA absent')
CSR_ENV_PY
CSR_SYNC
```

## 2. 初始化与预检

先复用本机兼容的 `yolo26n.pt` 与 `assets/bus.jpg`；若工具明确提示缺少资源，使用本节后面的资源准备段。
预检仅更新一次性副本，正式目录不创建，不会跑训练轮次或完整 val/test。
下面的初始化/plan/报告均保护已有文件；已成功完成的命令无需重做。失败报告保留，修复后用新的报告文件名重跑，并在 start 指向新的报告。

```bash
bash <<'CSR_PREFLIGHT'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
V=cbr_lif_csr_p3_v1
SOURCE="$CSR_P3_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
INIT="$CSR_P3_WORK/weights/${V}_controlled_init.pt"
LAUNCH="$CSR_P3_WORK/outputs/csr_p3/$V"
python tools/prepare_csr_p3_resources.py --main "$CSR_P3_MAIN" --report outputs/csr_p3/resources.json
python tools/init_csr_p3.py "$V" --source "$SOURCE" --output "$INIT" --report "$LAUNCH/initialization.json"
python tools/train_csr_p3.py plan --variant "$V"
python tools/check_csr_p3_math.py --device cpu --output outputs/csr_p3/math_cpu.json
python tools/check_csr_p3_math.py --device cuda:0 --output outputs/csr_p3/math_cuda.json
python tools/check_csr_p3_runtime.py --complexity --output outputs/csr_p3/runtime_complexity.json
python tools/preflight_csr_p3.py "$V" --mode server --device 0 --cuda-small \
  --source "$SOURCE" --init "$INIT" --data "$CSR_P3_MAIN/configs/crack_autodl.yaml" \
  --recipe "$LAUNCH/train_args.yaml" --amp-check-weights "$CSR_P3_WORK/yolo26n.pt" \
  --report "$LAUNCH/server_preflight.json"
python - <<'CSR_STATUS_PY'
import json,pathlib
p=pathlib.Path('outputs/csr_p3/cbr_lif_csr_p3_v1/server_preflight.json')
r=json.loads(p.read_text())
print('preflight:',r['status'],'capacity:',r.get('server_capacity',{}).get('status'))
assert r['status']=='PASSED', 'Do not start: inspect PENDING/FAILED details'
CSR_STATUS_PY
CSR_PREFLIGHT
```

仅在上述工具提示缺少资源且本机没有兼容副本时执行：

```bash
bash <<'CSR_RESOURCES'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
python tools/prepare_csr_p3_resources.py --main "$CSR_P3_MAIN" --download --report outputs/csr_p3/resources_download.json
CSR_RESOURCES
```

该命令只访问本版本代码指定的官方 Ultralytics 资源 URL，不安装/升级依赖。失败或资源不兼容会停止；原生 AMP 检查不允许跳过。

## 3. 显式正式 start（预检通过后，由用户执行）

200轮、patience50、B16、640、原在线增强和原生AMP保持不变。
启动工具再次核对提交、源码、初值、数据和全部配方身份；已有同名结果目录拒绝另起 name2。

```bash
bash <<'CSR_START'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
python tools/train_csr_p3.py start --variant cbr_lif_csr_p3_v1 \
  --preflight outputs/csr_p3/cbr_lif_csr_p3_v1/server_preflight.json
python tools/train_csr_p3.py status --variant cbr_lif_csr_p3_v1
CSR_START
```

tmux 会话为 `csr-p3-cbr_lif_csr_p3_v1`；实际日志与退出码在 `outputs/csr_p3/cbr_lif_csr_p3_v1/attempt_*/`。
状态区分 COMPLETED_200、EARLY_STOPPED、FAILED、INTERRUPTED，同时保留完成轮数和 final_eval 状态；tmux `[exited]` 不代表成功。
40/80轮只记录与父实验同期 val 的观察，不改学习率/早停规则。

## 4. resume（仅恢复已有未完成训练）

使用该唯一run的 `weights/last.pt` 和原PASSED报告，恢复已学习CSR、原模块、optimizer、EMA、scaler；不重新初始化。
完成200轮但final_eval失败时保留权重并独立验证，不自动重训。

```bash
bash <<'CSR_RESUME'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
python tools/train_csr_p3.py resume --variant cbr_lif_csr_p3_v1 \
  --preflight outputs/csr_p3/cbr_lif_csr_p3_v1/server_preflight.json
python tools/train_csr_p3.py status --variant cbr_lif_csr_p3_v1
CSR_RESUME
```

## 5. 独立 val，再对同一 best/hash 执行 test

以下两段留待训练结束。协议固定 `corrected_sorted_conf_mask_v1`，640/B16/workers0/FP32，conf0.001、iou0.7、max_det300、seed42，无额外NMS或阈值优化。
工具固定选择训练验证所得的唯一 best.pt，并保存完整精度JSON与原生图表。test要求前一份val通过且checkpoint哈希一致，训练中拒绝test。

```bash
bash <<'CSR_VAL'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
python tools/csr_p3_results.py val --variant cbr_lif_csr_p3_v1 --device 0
CSR_VAL
```

```bash
bash <<'CSR_TEST'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
python tools/csr_p3_results.py test --variant cbr_lif_csr_p3_v1 --device 0
CSR_TEST
```

## 6. 轻量证据包

包含训练/验证证据、完整指标、初始化/预检/状态、日志尾、实际源码/配置、提交与SHA256清单。
排除数据、权重、大预测、参考ZIP。默认小于20MiB；超限保留partial并列出大文件，不静默删关键证据。
训练已停止或失败时也可用于故障证据，但缺项会明确列出，不能当完整结果包；活动训练期间拒绝打包。

```bash
bash <<'CSR_PACK'
set -Eeuo pipefail
source /root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1/outputs/csr_p3/environment.sh
python tools/pack_csr_p3.py --variant cbr_lif_csr_p3_v1 \
  --evidence "$CSR_P3_WORK/outputs/csr_p3" \
  --output "$CSR_P3_MAIN/runs/c_series/cbr_lif_csr_p3_v1_light.tar.gz"
CSR_PACK
```

单模块配置与全部工具同样支持 `csr_p3_v1`。首轮默认主组合；若以后选择消融，分别设置variant并完整生成其独立初值、plan、PASSED预检和run，不能共用主组合报告，也不要并行执行两个显存预检。
