# GNR v1 服务器命令

功能提交：`bc2754423f898d204e90110c263b4c558fc8bb41`。
分支：`exp-rtdetr-r18-lite-gnr-v1`。
已通过 `git ls-remote` 核实功能提交已推送到 `https://github.com/supershaojie/concrete-crack-rtdetr.git`。
本文件由随后单独的 docs-only 提交添加；以下命令固定同步到上述功能提交，因此不依赖本文件自身的提交SHA。
功能工作区中的运行代码、配置、测试及证据与该功能提交完全对应；本文件可在开发端或远端分支阅读。

开发端真实B16/640 preflight遇到CUDA OOM，有效更新为0，不能作为服务器TECHNICAL_PASS。
服务器必须依次prepare、diagnose、preflight；审阅结果后由用户单独执行start。默认GPU0共享，无空卡等待、显存阈值或停止其它实验的步骤。

## 1. 首先执行：有界同步到独立工作区

在服务器终端完整复制下面这一块。只从已核实的Git仓库读取已固定版本的同步脚本。
引导fetch和同步脚本内部fetch各最多3次、单次120秒；失败明确停止。
不切换主目录分支，不reset/clean未知文件，不杀进程。现有GNR目录脏、分支不符、不能快进或仍有本实验Python任务时保留现场并报错。

```bash
/root/miniconda3/envs/rtdetr/bin/python - <<'PY'
from pathlib import Path
import subprocess
main = Path('/root/autodl-tmp/projects/Crack_RTDETR')
sha = 'bc2754423f898d204e90110c263b4c558fc8bb41'
branch = 'exp-rtdetr-r18-lite-gnr-v1'
remote = 'https://github.com/supershaojie/concrete-crack-rtdetr.git'
actual = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=main, text=True, timeout=30).strip()
if actual != remote:
    raise SystemExit('Unexpected origin; repository preserved')
for attempt in range(3):
    try:
        subprocess.run(['git', 'fetch', '--no-tags', 'origin', branch + ':refs/remotes/origin/' + branch], cwd=main, check=True, timeout=120)
        break
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f'Fetch {attempt + 1}/3 failed: {error}', flush=True)
else:
    raise SystemExit('Bounded fetch failed; no worktree changed')
subprocess.run(['git', 'merge-base', '--is-ancestor', sha, 'refs/remotes/origin/' + branch], cwd=main, check=True, timeout=30)
script = subprocess.check_output(['git', 'show', sha + ':tools/sync_gnr_v1.sh'], cwd=main, timeout=30)
subprocess.run(['bash', '-s', '--', sha], input=script, cwd=main, check=True)
PY
```

成功输出 `SYNCED` 和完整SHA。再确认代码身份及真实入口：

```bash
git -C /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1 rev-parse HEAD
git -C /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1 status --short
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py --help
```

HEAD应为上述40字符SHA，status应无输出。服务器需有原rtdetr环境、tmux、原数据、公共初值，以及母版原生AMP检查已使用的yolo26n.pt/bus.jpg；脚本缺资源会明确报告，不自动换环境或改配方。

## 2. Prepare

以下母版文件路径来自成功母版归档。公共源为MAIN/weights下未训练初值，母版best只用于diagnose。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py prepare \
  --main /root/autodl-tmp/projects/Crack_RTDETR \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --mother-args /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml
```

应看到`PREPARED`、`actual_mother_args: CHECKED`和`formal_training_started: false`。
首次核验全部内容并建立快照；同一身份以后复用，不反复完整扫描。若数据元数据变化，会要求显式`prepare --refresh-data`；须先查明变化，正式开始后不再允许重写准备结果。
如果母版文件已经搬迁，应从其真实归档定位后使用入口已提供的`--mother-args`或`--mother-best`，不能拿其它实验或未训练权重替代。

## 3. 有界diagnose与preflight（均不会启动长训）

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py diagnose \
  --mother-best /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt \
  --seconds 900 --val-images 64 --train-batches 8
```

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py preflight --seconds 900 --max-batches 16
```

diagnose使用64张val和最多8个原增强B16 train batch，不做optimizer step、不用test；报告非零干预时为REVIEW，供审阅。
完整诊断全零为ZERO_INTERVENTION，start会拒绝；缺资源/覆盖不足等状态不会伪装成有效机制。
preflight保持B16/640/AMP/AdamW/nbs64，最多900秒或16microbatch；仅GNR预检上下文e=20。
只有实际有限非零参数更新才是TECHNICAL_PASS。预算内无更新为exit2/PENDING；真实OOM为exit1/RESOURCE_ERROR并保留日志，不自动改设置重试。

## 4. 查看诊断、预检和状态

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py status
cat /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/outputs/gnr_v1/diagnose.json
cat /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/outputs/gnr_v1/preflight.json
```

每次diagnose/preflight的完整console.log和诊断Markdown位于相应JSON的`folder`路径。
status只读已有状态，检查真实PID/命令/cwd/创建时间；不会加载模型、扫描原始数据、补测或自动恢复。

## 5. 单独start：用户审阅诊断后执行

仅当前身份有TECHNICAL_PASS，且已审阅对应REVIEW诊断时执行这一块。
新训练从公共受控初值e=0开始，保留200epoch、patience50和完整母版配方；预检状态不会流入正式训练。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py start
```

返回DISPATCHED后worker在`gnr-v1-training`中持续运行。

## 6. 查看页面、断开与重新连接

```bash
tmux attach -t gnr-v1-view
```

按`Ctrl+B`，松开后按`D`离开viewer；这只detach，不结束训练。断开SSH/网页或关闭本地电脑后，服务器worker继续运行。
重新连接服务器后再次执行同一attach命令，也可单独运行status。worker结束后viewer仍保留日志和结果；viewer存在不代表训练还在运行。
不依赖影响其它实验的全局tmux设置，不执行kill-server。

## 7. 只有同一未完成run才resume

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py resume
```

必须是同一身份且未完成的run，有完整last中的optimizer/scaler/EMA/epoch；strip后的完成权重不是真resume来源。
恢复实际epoch和GNR渐入。已完成训练、仅最后验证失败时执行finish，不resume或重新start。

## 8. 训练结束后finish

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py finish
tmux attach -t gnr-v1-view
```

独立`gnr-v1-finish`依次补齐同一val-selected best的FP32完整val/test、离线阈值统计和包。
已有身份、哈希、覆盖及产物齐全的split显示REUSED，不重复推理；缺项只补对应split并保留旧目录。
worker与tee实际退出码先保存，再离线pack；打包自身退出码为外部receipt。
最后显示val/test的P/R/F1/AP50/AP75/mAP50–95、母版百分点差、best、退出码和包路径，较差结果也完整显示。

## 9. 独立val/test/pack重入

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py val
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py test
```

test要求已锁定且完整可信的独立val。缺成功索引但原metrics和全部产物仍可信时，可恢复索引并复用，不因锁文件缺失重跑。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/tools/gnr_v1.py pack
```

pack纯离线，不调用模型/数据扫描/val/test；缺项输出INCOMPLETE并exit2。需要补正式评估及离线统计时使用finish。
不能以INCOMPLETE包当作完整结果交付。

## 10. 固定产物位置

- RUN：`/root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/runs/c_series/gnr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`
- OUT：`/root/autodl-tmp/projects/Crack_RTDETR-gnr_v1/outputs/gnr_v1`
- 常规训练记录：RUN下`args.yaml`、`results.csv`、`weights/best.pt`、`weights/last.pt`。
- 独立完整指标：OUT下`evaluation_val.json`、`evaluation_test.json`，各自`folder`内有全部300query/GT、曲线、原始匹配统计。
- val选阈值并固定用于test的离线统计：OUT下`threshold_analysis.json`，IoU=.5，新过滤集合重新匹配。
- 持续页面日志：OUT下`view.log`；各次worker日志和Python/tee退出码：OUT下`dispatches/<时间>/`。
- 完整包：OUT下`GNR_v1_COMPLETE_<时间>.tar.gz`；缺项包为`GNR_v1_INCOMPLETE_<时间>.tar.gz`。
- 包路径/哈希/manifest回读结果：OUT下`package.json`。

包不包含原始数据图片或best/last权重本体，权重仍留服务器。所有动作都不自动下载或传输大包。
先查看正式结果，再决定是否下载；现有多轮test使用和增强家族跨split的评估局限保持披露。
