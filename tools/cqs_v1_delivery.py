"""Render independently copyable server commands using a real full Git SHA."""
from cqs_v1_common import *


def write_commands(sha):
    require(len(sha)==40 and all(v in '0123456789abcdef' for v in sha),'Full real SHA required')
    require(git('rev-parse',sha+'^{commit}')==sha,'Commit not present')
    root='/root/autodl-tmp/projects/Crack_RTDETR-cqs_v1'
    main='/root/autodl-tmp/projects/Crack_RTDETR'
    prefix=(f'PYTHONPATH={root}/ultralytics-main PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false \\\n'
            f'{SERVER_PYTHON} -u {root}/tools/cqs_v1.py ')
    text=f'''# CQS v1 服务器命令

交付完整 SHA：`{sha}`；分支：`{BRANCH}`；母版：`{BASE_SHA}`。
每块独立可复制。正式训练只由用户执行 start/resume 启动；test 只在用户执行该动作时运行。
同步只 fetch 本实验分支并按完整 SHA 核对，不切换或清理主工作区。

同步（首次取得脚本后仍执行正式 sync，写入 delivery.json）：

```bash
bash <<'CQS_SYNC'
set -euo pipefail
git -C {main} fetch --no-tags origin refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}
test "$(git -C {main} rev-parse refs/remotes/origin/{BRANCH})" = {sha}
script=$(mktemp /tmp/cqs-v1-sync.XXXXXX.sh)
git -C {main} show {sha}:tools/sync_cqs_v1.sh > "$script"
CQS_SYNC_FETCHED_SHA={sha} bash "$script" {sha}
CQS_SYNC
```
'''
    descriptions=dict(prepare='准备公共初值、完整母版配方、数据身份与真实 AMP 检查资源',
        preflight='一次隔离的服务器短检：B16/640，最多16 micro-batch／900秒，目标2次有效更新',
        probe='最多64张固定 train 图机会诊断；无母版 best 则明确 PENDING',
        start='验证有效预检并在 cqs-v1-training 中启动正式训练，终端直显并 tee 留日志',
        status='读取本实验 PID／子进程、阶段、日志尾部、真实 epoch 与退出状态',
        resume='仅从本实验有效 last 恢复 optimizer／scaler／EMA／epoch，继续原 run',
        val='训练完成后对本实验 best 独立 FP32 val，锁定 SHA256',
        test='仅在显式执行此块时，对 val 已锁定权重做完整最终 test',
        pack='只打包已有证据，未执行 test 标记 NOT_RUN；输出 FileZilla 绝对路径',
        **{'archive-failed-run':'仅用于无 checkpoint、无 results 数据行且未活跃的初始化失败；重命名保全后再 start'})
    for action,label in descriptions.items():
        text+=f'\n{label}：\n\n```bash\n{prefix}{action}\n```\n'
        if action=='status':
            text+='\n连接训练 tmux：\n\n```bash\ntmux attach-session -t =cqs-v1-training\n```\n\n脱离并保持训练：先按 **Ctrl+B**，松开后按 **D**。\n'
    text+=f'\n日志：`{root}/outputs/cqs_v1/console_<UTC>.log`。正式结果：`{main}/runs/c_series/{RUN_NAME}`。\n'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'server_commands.md').write_text(text,encoding='utf-8')
    return OUT/'server_commands.md'


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sha')
    print(write_commands(parser.parse_args().sha))
