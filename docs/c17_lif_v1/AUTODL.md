# AutoDL

使用最终回复中的真实完整SHA。首次在主仓库只fetch本分支，git show该SHA的sync脚本到mktemp文件，再执行：

```bash
bash tools/sync_c17_lif_v1.sh FULL_SHA [MAIN_REPO] [WORKTREE]
```

服务器默认主仓库 `/root/autodl-tmp/projects/Crack_RTDETR`，独立worktree `/root/autodl-tmp/projects/Crack_RTDETR-c17-lif-v1`。SHA必须属于远端本分支历史并以成功LIF提交为祖先。首次建立detached worktree；同SHA重入仅核验。跟踪修改、错误SHA/仓库/路径冲突均停止；未跟踪结果保留。同步不安装、不初始化、不训练；已训练worktree不checkout/更新源码。支持有限30秒同步lock，不占用训练锁。

仅使用既有环境 `/root/miniconda3` 的rtdetr（Python3.10.13、torch2.1.2+cu121、4090）；tmux worker内显式source conda.sh/activate。实机版本不匹配停止报告。单张4090统一CUDA_VISIBLE_DEVICES=0/device0，不自动选择GPU1；资源不足时保留batch16/imgsz640并等待其他实验，不kill或自动降配。

```bash
(cd /root/autodl-tmp/projects/Crack_RTDETR-c17-lif-v1 && CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c17_lif_v1.sh start-direct)
(cd /root/autodl-tmp/projects/Crack_RTDETR-c17-lif-v1 && bash tools/autodl_c17_lif_v1.sh status)
tail -n 80 -f /root/autodl-tmp/projects/Crack_RTDETR-c17-lif-v1/outputs/c17_lif_v1/console.log
```

start-direct先做有限预检，期间详情为 `outputs/c17_lif_v1/preflight.log`；通过后派发tmux `c17_lif_v1-training`，正式日志才是上述console.log。退出码原样记录；重复run/launch/init/session不会覆盖，也不自动resume/retry。状态区分NOT_STARTED/DISPATCHED/RUNNING/SUCCESS/FAILED；tmux存在不等于成功。

正式run固定为 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c17_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug`。训练SUCCESS后：

```bash
(cd /root/autodl-tmp/projects/Crack_RTDETR-c17-lif-v1 && \
 bash tools/autodl_c17_lif_v1.sh status && \
 bash tools/autodl_c17_lif_v1.sh val && \
 bash tools/autodl_c17_lif_v1.sh test && \
 bash tools/autodl_c17_lif_v1.sh pack-complete)
```

命令不带variant或额外参数；拼错/多参数退出非零。val/test用本run同一best SHA，test要求独立val完成；完整评估固定640/B16/device0/workers0/halfFalse/conf.001/iou.7/max_det300/augmentFalse/rectFalse/plotsTrue/seed42。pack-complete不补跑评估，缺关键证据拒绝完整包。包包含best/last、训练日志/配方、独立val/test曲线/预测GT、源码/来源/映射/预检/数据及checkpoint指纹。输出四个绝对路径：tar.gz、sha256、inventory.json、verification.json，目录 `/root/autodl-tmp/projects/Crack_RTDETR/downloads/c17_lif_v1`。
