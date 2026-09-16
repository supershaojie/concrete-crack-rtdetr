# 服务器交接约定

本次未登录服务器、未同步、未训练、未 test。使用提交外 `outputs/bscrep_v1_delivery/DELIVERY.md` 中已填入完整 40 位 SHA 的命令，不能用分支名代替锁定 SHA。

该交付文件依次给出：

1. 在服务器历史主仓库核对对象，缺失才从指定新分支 fetch。`GIT_TERMINAL_PROMPT=0`、HTTP/1.1、`--progress --no-tags`、lowSpeedLimit=1、lowSpeedTime=60，单次 timeout120 秒、最多 5 次。
2. 网络失败时，导入同一提交的小型增量 Git bundle。bundle 以成功父提交 a0459d6a652cb702699087c88fa39a3e4c4087ec 为前提；服务器历史成功实验已有此对象。先 `git bundle verify`，缺前提会明确失败。
3. 创建独立 detached worktree 并核对 HEAD，再创建 weights/outputs。不会 checkout 或更新正在运行实验的工作树。
4. 每个新终端都 source 统一 `outputs/bscrep_env.sh`：已有 rtdetr 环境、本工作树 PYTHONPATH、完整 SHA、原主仓库路径。核对实际 ultralytics/BSC import 路径。
5. 用服务器已有固定源权重生成两配置初值，无需上传新预训练或因子。
6. 主组合执行真实 B16/640/native AMP 在线增强预检；输出 checks.json，未通过就停止。单模块可随后以相同门禁预检，不与其他任务并行争用显存。
7. 正式 start 单独列出，通过后用户显式运行 tmux 启动脚本。没有自动启动动作。
8. 后续 resume/val/test/轻量包均有独立命令，每段 `set -Eeuo pipefail` 或 source 同一 env；不依赖其他终端变量。

若服务器显存被其他实验占用，应安排空闲时段运行预检/训练；脚本不会杀进程、降低 batch/imgsz/AMP 或改损失。检查失败目录和初值保留，不自动清理；重新检查应指定新的输出目录，并在 start 的 `--preflight` 中使用新报告。

源权重、数据、参考压缩包、初值和运行输出均不入 Git。服务器预检匹配完整代码/配置/源/初值/数据身份，正式训练更新次数保持 0 直到显式 start。
