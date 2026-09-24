# 服务器分段操作

每段独立执行。交付回复提供固定完整 SHA 的同步命令。以下引导从固定分支提取完整 SHA，再让同步入口核对它；不会切换主工作区。

```bash
(set -euo pipefail
git -C /root/autodl-tmp/projects/Crack_RTDETR fetch origin exp-rtdetr-r18-lite-lcd-v1
lcd_sha=$(git -C /root/autodl-tmp/projects/Crack_RTDETR rev-parse FETCH_HEAD)
git -C /root/autodl-tmp/projects/Crack_RTDETR show "$lcd_sha:tools/sync_lcd_v1.sh" | bash -s -- "$lcd_sha"
)
```

准备：

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py prepare
```

一次有界预检（不启动正式训练）：

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py preflight
```

用户决定正式启动时：

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py start
```

状态（没有 pane 文字也能检查）：

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py status
```

```bash
tmux list-panes -t lcd-v1-training -F '#{pane_pid} #{pane_current_command} #{pane_dead} #{pane_dead_status}'
```

```bash
(set -euo pipefail
lcd_log=$(/root/miniconda3/envs/rtdetr/bin/python -c 'import json; print(json.load(open("/root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/outputs/lcd_v1/dispatch.json"))["log"])')
tail -n 80 -- "$lcd_log"
)
```

仅在中断且有有效 epoch last.pt 时恢复：

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py resume
```

训练成功结束后，依次做一次 val 和锁定 test：

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py val
```

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py test
```

打包已有证据，不隐式启动训练或评估：

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-lcd_v1/tools/lcd_v1.py pack
```
