# AutoDL 公共准备命令

只做 CPU 准备，不训练、不推理、不安装环境、不检查 GPU 空闲、不停止其他会话。当前 Windows 执行不代表这些服务器步骤已运行。完整结果见 `PREPARATION_REPORT.md`。

先核实远端，再固定本次 fetch 到的公共提交；所有 Git 操作保留现有主仓库改动。现有主仓库只作为数据和 run 路径来源。

```bash
set -e
repo=/root/autodl-tmp/projects/Crack_RTDETR
git -C "$repo" remote get-url origin
# 应为 https://github.com/supershaojie/concrete-crack-rtdetr.git 或该仓库的 SSH URL。
git -C "$repo" status --short
git -C "$repo" fetch origin bench/comparison-base
prepared_sha=$(git -C "$repo" rev-parse FETCH_HEAD)
git -C "$repo" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$prepared_sha"
printf 'Preparation commit: %s\n' "$prepared_sha"
worktree=/root/autodl-tmp/projects/Crack_RTDETR-comparison-base
if [ -e "$worktree" ]; then
  test "$(git -C "$worktree" rev-parse HEAD)" = "$prepared_sha"
  test -z "$(git -C "$worktree" status --porcelain --untracked-files=no)"
else
  git -C "$repo" worktree add --detach "$worktree" "$prepared_sha"
fi
cd "$worktree"
/root/miniconda3/envs/rtdetr/bin/python benchmarks/comparison/prepare.py --help
```

若已有同名目录的提交不同，上述命令退出；保留该目录，另选新的 `worktree` 路径重新执行。不得 reset/强切旧工作树。可把 `prepared_sha` 与本次交付回复中的完整 SHA 对照。

数据约 43 GB，第一次会逐文件解码并计算内容哈希，采用一个扫描进程，且不生成数据缓存到原目录。通过独立 tmux 保存完成界面：

```bash
export COMPARISON_PYTHON=/root/miniconda3/envs/rtdetr/bin/python
export COMPARISON_OUTPUT="$worktree/outputs/comparison_prepare/$(date -u +%Y%m%dT%H%M%SZ)_server"
export COMPARISON_SESSION=comparison-prepare
bash scripts/autodl_prepare_comparison.sh --tmux \
  --data "$repo/configs/crack_autodl.yaml" \
  --data-root "$repo/datasets/crack_det" \
  --probe-transforms
tmux attach -t "$COMPARISON_SESSION"
```

同名 tmux 会话已存在时入口拒绝启动且保留旧会话；使用新的 `COMPARISON_SESSION=comparison-prepare-另一个名字`。不会 kill 或覆盖。`remain-on-exit` 保留完成界面。不要重复启动正在运行的准备任务。也可省略 `--tmux` 在前台执行。

默认四组资产位置在 `benchmarks/comparison/configs/asset_locations.yaml`。已知路径变更时可追加 `--run mother=/实际训练run --run mother=/实际launch元数据目录`；同一模型第一次 `--run` 替换默认路径，后续同名 `--run` 追加。输入可以是解包目录或历史 tar.gz。不要先复制大权重。本轮只索引、哈希权重，不反序列化模型。服务器缺失的独立 val/test 记录会列入 missing，不会自动补跑。

`--probe-transforms` 只构造增强对象并记录导入模块位置、版本、关闭前后链，不加载图片、不构造检测模型。失败写为 UNAVAILABLE。它证明本次环境，不等于恢复历史运行；历史激活状态仍结合冻结环境和日志证据。

输出包含 `dataset_manifest.json`、三组清单、`coco/{train,val,test}.json`、`conversion_counts.json`、`augmentation_table.json`、`asset_index.json`、原始小文件副本、`summary.json`、`SUMMARY.md` 和 `COMPLETE.txt`。外侧 `${COMPARISON_OUTPUT}.console.log` 保存日志，`${COMPARISON_OUTPUT}.process_exit_code.txt` 保存 **Python** 真实退出码；`tee` 退出码另行显示，不能取代 Python 状态。

```bash
cat "$COMPARISON_OUTPUT.process_exit_code.txt"
cat "$COMPARISON_OUTPUT/COMPLETE.txt"
cat "$COMPARISON_OUTPUT/SUMMARY.md"
```

`0` 表示审计程序正常完成，不代表论文数据条件合格；缺数据、非法标签、同原图族跨集合分别显示为数据/论文状态。参数、依赖、文件操作错误返回 `2`。同原图族重叠时保留旧划分及旧指标，限制为旧固定划分内对比，不能声称未见原图泛化；本命令不创建新划分。

准备输出可供后续 CPU 评测使用。已有高精度缓存时的可选独立步骤（无模型推理）：

```bash
"$COMPARISON_PYTHON" benchmarks/comparison/evaluation/evaluate.py \
  --gt "$COMPARISON_OUTPUT/coco/test.json" \
  --legacy-cache /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/outputs/c19_lif_v1/evaluation_test/predictions_gt.jsonl.gz \
  --legacy-metrics /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/outputs/c19_lif_v1/evaluation_test/metrics.json \
  --output "$COMPARISON_OUTPUT/mother_test_cpu.json"
```

已生成审计可通过 `--dataset-manifest` 和 `--asset-index` 显式复用，输出会标明未重新核实当前源文件；仅在原数据和资产未变时使用。所有输出需新目录，重跑不会覆盖旧证据。本阶段结束后再决定是否接入 YOLOv5m。
