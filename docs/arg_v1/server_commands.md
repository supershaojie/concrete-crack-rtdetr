# ARG v1 数据身份修复：已有 worktree 升级命令

本次固定训练代码提交：`ef9cb7e05e5557f7dd06c95cf2361998a284adc9`。
适用旧提交：`e6d6ce11783e57cdd3377f6b5a103665218df065`；旧 prepare 已完成、preflight 因 names 键类型失败、正式训练尚未启动。
分支：`exp-rtdetr-r18-lite-arg-v1`。母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
后续交付文档提交只记录此代码提交的证据和命令；服务器固定到上面的代码 SHA，不执行泛化的 git pull。

这些命令由用户在服务器执行。本轮未连接服务器、未派发正式训练，服务器 GPU 预检仍待执行。依次执行第 1～4 段；正式 start 单列于第 5 段。

## 1. 升级旧 worktree 并保存证据（首条服务器命令）

原 `sync_arg_v1.sh` 同时拒绝不同 HEAD 和不同 `sync.json`，不能直接用新 SHA 调它升级。
下段先检查旧 SHA、仓库关系、干净工作区和无训练状态，在原操作锁内完整备份 `outputs/arg_v1` 并逐文件核验 SHA256，再用 `merge --ff-only` 更新本实验 worktree。主工作区不切分支。
旧 sync/prepare/preflight 活动记录移入备份的 `retired_active/`，随后调用新 SHA 的原同步脚本生成新 sync。旧失败 preflight 子目录、日志、初始化权重与 provenance 均保留；不使用 reset、clean 或删除输出目录。
备份需要约一个旧 `outputs/arg_v1` 的额外磁盘空间，完成前不会改代码或移走旧记录。

完整复制此块。两次 fetch 均有 120 秒上限、30 秒低速超时；失败退出，不自动重试。

```bash
bash <<'ARG_UPGRADE'
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-arg_v1
PY=/root/miniconda3/envs/rtdetr/bin/python
OLD=e6d6ce11783e57cdd3377f6b5a103665218df065
SHA=ef9cb7e05e5557f7dd06c95cf2361998a284adc9
BRANCH=exp-rtdetr-r18-lite-arg-v1
[[ $(git -C "$MAIN" remote get-url origin) == https://github.com/supershaojie/concrete-crack-rtdetr.git ]]
[[ -f "$WORK/.git" ]]
[[ $(cd "$WORK" && realpath "$(git rev-parse --git-common-dir)") == $(cd "$MAIN" && realpath "$(git rev-parse --git-common-dir)") ]]
[[ $(git -C "$WORK" branch --show-current) == "$BRANCH" ]]
[[ $(git -C "$WORK" rev-parse HEAD) == "$OLD" ]]
[[ -z $(git -C "$WORK" status --porcelain --untracked-files=all) ]]
timeout 120s git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
git -C "$MAIN" cat-file -e "$SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$OLD" "$SHA"
git -C "$MAIN" merge-base --is-ancestor "$SHA" FETCH_HEAD
export ARG_V1_MAIN="$MAIN" PYTHONPATH="$WORK/ultralytics-main:$WORK/tools"
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
"$PY" - "$WORK" "$MAIN" "$OLD" "$SHA" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import shutil, subprocess, sys
from arg_v1 import operation_lock, active_workers, has_tmux
from arg_v1_common import ROOT, MAIN, OUT, RUN, INIT, SOURCE, SOURCE_SHA256, read_json, write_json, sha256, require, git

work, main, old, target = sys.argv[1:]
require(ROOT.resolve() == Path(work).resolve() and MAIN.resolve() == Path(main).resolve(), "Wrong experiment paths")
with operation_lock():
    require(git("rev-parse", "HEAD") == old and not git("status", "--porcelain", "--untracked-files=all"), "Old SHA or clean state changed")
    require(not active_workers() and not has_tmux(), "ARG worker/tmux exists; preserve and inspect status")
    require(not RUN.exists() and not (OUT / "training_identity.json").exists(), "Training identity/run exists; this upgrade is pre-training only")
    require(not any((OUT / "dispatches").glob("*")), "Dispatch evidence exists; inspect before upgrading")
    require(not (OUT / "identity_upgrade.json").exists(), "Prior upgrade receipt exists; inspect its phase/backup")
    plan = read_json(OUT / "prepare.json", {})
    require(plan.get("status") == "PASS" and plan["code"]["commit"] == old, "Expected old successful prepare")
    require(read_json(OUT / "sync.json", {}).get("commit") == old, "Expected old sync identity")
    require(sha256(SOURCE) == SOURCE_SHA256 == plan["source_sha256"], "Source initialization changed")
    require(sha256(INIT) == plan["init_sha256"] == read_json(OUT / "initialization.json")["output_sha256"], "Existing init/provenance changed")
    require(plan["args"]["seed"] == 42, "Unexpected original seed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = OUT.parent / ("arg_v1_before_names_fix_" + stamp + "_" + old[:12])
    require(not OUT.is_symlink() and not any(p.is_symlink() for p in OUT.rglob("*")), "Inspect output symlinks before copying")
    shutil.copytree(OUT, backup, ignore=shutil.ignore_patterns("operation.lock"))
    manifest = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "operation.lock":
            rel = p.relative_to(OUT).as_posix()
            digest = sha256(p)
            require(sha256(backup / rel) == digest, "Backup hash mismatch: " + rel)
            manifest[rel] = digest
    write_json(backup / "backup_manifest.json", dict(old_commit=old, target_commit=target, files=manifest))
    receipt = dict(old_commit=old, target_commit=target, backup=str(backup), phase="BACKED_UP")
    write_json(OUT / "identity_upgrade.json", receipt)
    print("Verified evidence backup:", backup, flush=True)
    subprocess.run(["git", "-C", work, "merge", "--ff-only", target], check=True, timeout=60)
    require(git("rev-parse", "HEAD") == target, "Upgrade did not reach exact target")
    retired = backup / "retired_active"
    retired.mkdir()
    for name in ("sync.json", "prepare.json", "preflight.json"):
        p = OUT / name
        if p.exists():
            p.rename(retired / name)
    receipt["phase"] = "UPGRADED_SYNC_REQUIRED"
    write_json(OUT / "identity_upgrade.json", receipt)
    subprocess.run(["bash", str(ROOT / "tools/sync_arg_v1.sh"), target], check=True, timeout=150)
    receipt["phase"] = "UPGRADED_REPREPARE_REQUIRED"
    write_json(OUT / "identity_upgrade.json", receipt)
    print("Upgrade complete; original init retained. Next: section 2 prepare.", flush=True)
PY
ARG_UPGRADE
```

成功后查看 `outputs/arg_v1/identity_upgrade.json` 中的备份路径和阶段。完整备份在 worktree 的 `outputs/arg_v1_before_names_fix_UTC时间_e6d6ce11783e/`，不位于会被重写的活动输出目录内。
若仅末尾 sync 的网络 fetch 失败，且 receipt 为 `UPGRADED_SYNC_REQUIRED`、HEAD 已是新 SHA，可单独重试下面的同步命令，然后继续第 2 段；不要重跑只接受旧 SHA 的整个升级块：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/sync_arg_v1.sh ef9cb7e05e5557f7dd06c95cf2361998a284adc9
```

其他中断或校验失败先保留现场，检查 receipt、HEAD 和备份；不要用强制同步覆盖差异。

## 2. 重新 prepare 并核对原身份（必须等升级和同步成功）

旧 JSON 的 names 已是字符串键，修复后可以比较；本次重新 prepare 是为了让 prepare 中的代码身份、数据清单和源码快照明确绑定新 SHA。
保留的 `arg_v1_init.pt` 与 `initialization.json` 会按原哈希复用，不重新初始化。下面在 prepare 前后核对原公共初始化、seed42 和完整训练参数，并逐项比较旧/新数据身份；任何真实变化都会停止。

```bash
bash <<'ARG_PREPARE'
set -Eeuo pipefail
WORK=/root/autodl-tmp/projects/Crack_RTDETR-arg_v1
export ARG_V1_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
export PYTHONPATH="$WORK/ultralytics-main:$WORK/tools" PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
/root/miniconda3/envs/rtdetr/bin/python - <<'PY'
from pathlib import Path
from arg_v1 import operation_lock
from arg_v1_common import OUT, INIT, SOURCE, SOURCE_SHA256, prepare, git, read_json, write_json, sha256, require

target = "ef9cb7e05e5557f7dd06c95cf2361998a284adc9"
with operation_lock():
    upgrade = read_json(OUT / "identity_upgrade.json", {})
    require(upgrade.get("target_commit") == target and upgrade.get("phase") in ("UPGRADED_SYNC_REQUIRED", "UPGRADED_REPREPARE_REQUIRED", "REPREPARED"), "Missing completed upgrade/backup receipt")
    require(git("rev-parse", "HEAD") == target and read_json(OUT / "sync.json", {}).get("commit") == target, "Run new-SHA sync first")
    backup = Path(upgrade["backup"])
    manifest = read_json(backup / "backup_manifest.json")["files"]
    for name in ("prepare.json", "initialization.json"):
        require(sha256(backup / name) == manifest[name], "Old evidence hash changed: " + name)
    previous = read_json(backup / "prepare.json")
    require(sha256(INIT) == previous["init_sha256"] and sha256(SOURCE) == previous["source_sha256"] == SOURCE_SHA256, "Original initialization changed")
    require(sha256(OUT / "initialization.json") == manifest["initialization.json"], "Original init provenance changed")
    current = prepare()
    require(current["code"]["commit"] == target, "Prepare is still bound to old code")
    for field in ("args", "data", "source_sha256", "init_sha256"):
        require(current[field] == previous[field], "Real identity difference after reprepare: " + field)
    require(current["args"]["seed"] == 42, "Seed changed")
    upgrade.update(phase="REPREPARED", preserved_fields=["args", "data", "source_sha256", "init_sha256"], preflight="MUST_RERUN")
    write_json(OUT / "identity_upgrade.json", upgrade)
    print("Prepare PASS: new code SHA; original data, initialization and full recipe unchanged. Rerun preflight.", flush=True)
PY
ARG_PREPARE
```

须已存在主仓库的 `configs/crack_autodl.yaml`、完整crack_det划分和 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
数据逐文件SHA读取可能需要数分钟，进度按1000图输出。
AMP自检还要求主仓库的 `yolo26n.pt`（或weights同名文件）和 `ultralytics-main/ultralytics/assets/bus.jpg`；缺失时短检报错，不会联网绕过或关闭AMP。

本段生成新 `prepare.json`、`train_args.yaml`、`recipe_diff.json`、数据清单与源码快照。旧文件在完整备份中。若比较失败，不执行 preflight/start；保留新旧记录定位真实差异。

## 3. 一次有界 preflight（必须等prepare成功）

必须重新运行。旧失败 preflight 已备份，不能沿用旧结果或手动改为 PASS；新预检会完整执行 binding 并绑定新 SHA。

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh preflight --seconds 900 --micro-batches 16
```

最多16个实际训练micro-batch、总900秒；诊断e=20，正式训练仍从e=0。
结果在 `outputs/arg_v1/preflight.json`。CUDA B16/640、有效参数更新、ARG-only梯度、新进程val和resume必须分别PASS。
若原生初始scale的有界尝试仅出现overflow、隔离scale128成功，原生适应项仍明确PENDING；其余必需项均PASS后才可能 `start_eligible=true`，正式scaler不改。
缺资源/容量/时间或真实异常时，不把其他PASS替代缺项。不要把preflight和start无条件串联。

可选的CPU机制复核入口（不能替代preflight）：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh probe
```

## 4. 查看状态

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh status
/root/miniconda3/envs/rtdetr/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/preflight.json")
if p.is_file():
    result = json.loads(p.read_text())
    print(json.dumps({"status": result.get("status"), "code_sha": result.get("binding", {}).get("code", {}).get("commit"),
        "start_eligible": result.get("start_eligible", False), "checks": result.get("checks"),
        "error": result.get("error"), "boundary_error": result.get("boundary_error")}, indent=2, ensure_ascii=False))
else:
    print("Preflight NOT_RUN; start is not eligible")
PY
```

检查新 SHA、preflight 每项状态、`start_eligible`、是否已有 worker/tmux/run。失败时也可执行 status；该命令不会启动或恢复训练。

## 5. 正式 start（必须等必需短检通过）

本段独立手动执行：新 preflight 必须绑定 `ef9cb7e05e5557f7dd06c95cf2361998a284adc9`，`start_eligible=true`，且 cpu、cuda_b16_amp、mechanism、new_process_val、resume 分别 PASS。start 入口还会重新核验完整数据、源码、初始化和配方，以及有效更新证据。不要把它接在升级或 preflight 命令后自动执行。

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh start
```

入口固定使用 `/root/miniconda3/envs/rtdetr/bin/python`，设置当前worktree的PYTHONPATH、`PYTHONUNBUFFERED=1`、`YOLO_AUTOINSTALL=false`。
正式worker只在独立tmux `arg-v1-training` 中运行，关掉用户电脑不影响服务器训练。
run固定为 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/arg_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。
保持原200epoch上限与patience50；正常早停不强行补训。

## 6. 日志与进度（start派发后）

```bash
tmux attach -t arg-v1-training
```

按 `Ctrl-b` 后按 `d` 脱离，不结束训练。也可独立执行：

```bash
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/dispatches/*/console.log
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh status
tail -n 3 /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/epochs.jsonl
```

dispatch目录包含worker精确命令、PID/创建时间、开始时间、console.log、exit.json、真实Python和tee退出码。不要把tmux存在当作成功。

## 7. 异常中断时的resume（仅训练尚未正常结束）

只有本实验有效 `last.pt` 含原epoch/optimizer/EMA/scaler且无活动worker时执行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh resume
```

不从best、诊断或另一实验权重恢复。resume继续原epoch与ARG ramp，不重启warmup。
若状态是 `TRAINING_COMPLETED+FINAL_EVAL_FAILED`，跳到下面val恢复评估，不使用resume。

## 8. 训练后独立val / final_eval失败恢复

必须有原生训练结束证据、best和无活动worker。命令只评估原fitness选出的best：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh val
```

完整FP32协议：640/B16/workers0/conf.001/iou.7/max_det300/seed42，无augment/rect。
保存checkpoint hash、训练/eval SHA、数据身份、实际配置、指标、十阈值AP和 `outputs/arg_v1/val_lock.json`。
若在恢复final_eval，会保留原异常及exit1，另写 `evaluation_recovery.json`，不会重写训练fingerprint。

需要在这次val同时导出全300个query和GT，可在本段改用：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh val --include-predictions
```

两种val命令选择一种执行。

## 9. 锁定同一best的最终test（必须等val成功）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh test
```

入口核对同一best hash、val锁、数据与eval源码，不重新选权重或扫阈值；已有最终test锁时拒绝重测。
如需要test预测，在首次执行本段时用 `test --include-predictions`，文件带独立test名称和身份，不混入val。

## 10. LIGHT交付包（失败时也可执行）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh pack
```

输出路径会打印为 `/root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/ARG_v1_LIGHT_时间戳.tar.gz`，目录下可直接下载实际生成的文件。
默认无数据集和best/last权重；manifest含每个成员的大小及SHA256。缺少test仍打包故障证据，缺项不会标PASS。

需要val原始预测与GT的分析包：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh pack --include-predictions
```

若val锁存在但未导出预测，会对锁定best/配置重新执行只读val导出；不会运行test或重新选权重。
