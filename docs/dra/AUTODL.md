# AutoDL 操作

本次开发未启动正式训练、真实完整 val/test 或服务器预检。以下 start-direct/val/test 由用户在服务器明确执行。

## 固定提交同步

交付消息给出最终完整 40 位 SHA。将该 SHA 原样填入 `DRA_SHA`，不要使用会移动的分支名代替训练版本。

```bash
set -euo pipefail
DRA_SHA='填写交付消息中的完整40位SHA'
DRA_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
DRA_WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR-dra
case "$(git -C "$DRA_MAIN" remote get-url origin)" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo 'origin 不符合预期，保留现场'; exit 1 ;;
esac
git -C "$DRA_MAIN" fetch origin codex/dra-aifi
DRA_SYNC="$(mktemp /tmp/dra-sync-XXXXXX.sh)"
git -C "$DRA_MAIN" show "$DRA_SHA:tools/sync_dra.sh" > "$DRA_SYNC"
bash "$DRA_SYNC" "$DRA_SHA" "$DRA_MAIN" "$DRA_WORKTREE"
export DRA_MAIN
cd "$DRA_WORKTREE"
test "$(git rev-parse HEAD)" = "$DRA_SHA"
```

同步脚本验证 origin、SHA 属于远端 DRA 分支历史且包含 DRA 实现、路径独立与 common git dir；打印主仓库 HEAD/status。已有目标 worktree 只有在 HEAD 完全相同且干净时接受；不同 HEAD/本地分支会打印差异并停下，不 reset、clean、stash、切换或覆盖。主仓库其他实验修改可保留，新增 worktree 不要求停止它们。

根路径变更时同时覆盖同步参数 `DRA_MAIN/DRA_WORKTREE`，并在随后每个新 shell 中 export 同一 DRA_MAIN。实际 C2 args 和权重须在新主根相同相对路径；data YAML 内数据路径必须指向同一数据集。脚本会记录路径搬迁差异，拒绝训练超参数差异。默认数据配置自身使用绝对数据路径，复制快照后解释不变。

## 直接启动、状态与日志

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-dra
export DRA_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
bash tools/autodl_dra.sh start-direct
bash tools/autodl_dra.sh status
tail -n 100 -F outputs/dra/console.log
# Ctrl-C 只退出 tail。可选查看 tmux：
tmux attach -t dra-aifi-training
# Ctrl-B 再按 D 脱离 tmux，保持后台进程。
```

入口激活现有 conda `rtdetr`，不升级 PyTorch/CUDA。预计服务器是 4090/PyTorch2.1.2，实际版本与 GPU 写入记录；本地通过不能代替服务器通过。

正式 run：主根 `runs/c_series/dra_aifi_rtdetr_r18_lite_e200_b16_onlineaug`。
新初始化：DRA 工作树 `weights/dra_aifi_controlled_init.pt`。
启动元数据：DRA 工作树 `outputs/dra`。
会话：`dra-aifi-training`。完整命令在 plan/launch_state/worker.sh；PID 在 process.json，Python 与 shell 两层退出码分别在 exit_code.json/process_exit_code.txt。

状态包括 not_submitted、initializing、dispatched、worker_starting、training、successfully_exited、failed/failed_worker_missing/failed_dispatch_lost。training 表示完成训练 setup 且进程身份仍匹配，不是收敛或吞吐健康保证；结合日志和 results.csv 判断进度。tmux 存在本身不会被报告为训练正常。

同别名 tmux、run、launch、初始化和主根原子锁防止重复/覆盖。启动失败会保留现场；修复问题后先人工核对并把本实验失败材料移到新的留档目录，再重新启动，不能直接删除其他实验文件。没有自动降低资源配置的恢复路径。

## 独立 val、test 与完整包

训练成功结束后执行：

```bash
bash tools/autodl_dra.sh status
bash tools/autodl_dra.sh val
bash tools/autodl_dra.sh test
bash tools/autodl_dra.sh pack-complete
```

val/test 固定训练生成的 best.pt，test 必须在同 SHA 的独立 val 后执行。默认 imgsz640、batch16、FP32、workers0、conf0.001、iou0.7、max_det300、无推理增强；eval 是项目独立评估口径，训练仍继承 C2 workers8/AMP。结果目录是 `outputs/dra/evaluation_val` 和 `evaluation_test`，同路径已有结果即拒绝覆盖。

评估保持 C2 原生 mask 策略，不能与其他 corrected-mask 策略的结果直接当作同口径对照。metrics.json 含 checkpoint/data/source、实际设置、AP/P/R、匹配与预测导出散列；`predictions_gt.jsonl.gz` 来自本次评估的同一轮推理。

pack-complete 不启动训练或评估，未训练时也可以运行并导出标记不完整的证据。默认输出主根 `downloads/dra/dra_evidence_时间戳.tar.gz`；可用 `--output /明确目录/新名字.tar.gz` 指定目标。必须查看 `.verification.json` 的 `complete` 和 `missing_evidence`，散列正确不等于实验材料齐全。服务器空间要容纳 best/last 和完整输出，包大小不限制 20 MiB。

## 下载与校验

服务器从 pack 输出复制完整包文件名，先校验：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR/downloads/dra
sha256sum -c dra_evidence_实际时间戳.tar.gz.sha256
cat dra_evidence_实际时间戳.tar.gz.verification.json
```

本地 PowerShell（替换 AutoDL SSH 地址/端口和服务器打印的包名）：

```powershell
$DraHost = '你的AutoDL-SSH地址'
$DraPort = 你的SSH端口
$DraPackage = 'dra_evidence_实际时间戳.tar.gz'
$DraDest = 'D:/rtdetr跑结果/DRA-AIFI'
scp -P $DraPort "root@${DraHost}:/root/autodl-tmp/projects/Crack_RTDETR/downloads/dra/${DraPackage}*" $DraDest
$DraExpected = ((Get-Content -LiteralPath "$DraDest/$DraPackage.sha256" -Raw).Trim() -split '\s+')[0]
$DraActual = (Get-FileHash -LiteralPath "$DraDest/$DraPackage" -Algorithm SHA256).Hash.ToLower()
if ($DraActual -ne $DraExpected) { throw 'DRA 文件 SHA256 不一致' }
Get-Content -LiteralPath "$DraDest/$DraPackage.verification.json" -Raw
```

## 可选开发检查

这些小样本检查不属于 start-direct 的前置条件，不应代替或触发完整服务器预检。输出路径必须是全新的；调试模型在临时目录中销毁。

```bash
python tools/check_dra.py \
  --source "$DRA_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --output "outputs/dra_check_$(date +%Y%m%d_%H%M%S)"
python tools/check_dra_tools.py --output "outputs/dra_tools_check_$(date +%Y%m%d_%H%M%S)"
```

check_dra 是合成 batch2、160×192 的三步 optimizer 练习，以及模块 20×20/非方形/单像素检查；check_dra_tools 是每 split 两张合成图片、160/batch1/CPU 的评估工具集成检查，不读取真实数据集。不得拿调试模型作为正式初始化。
