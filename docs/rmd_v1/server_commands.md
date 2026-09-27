# RMD v1 服务器命令

固定到经过验证的实施提交的完整命令在发布文档提交中生成。实现入口已经随本源码提交提供：`tools/sync_rmd_v1.sh` 接受 40 位提交 SHA，`tools/rmd_v1.sh --help` 显示全部操作。

执行顺序：同步 → prepare → preflight（必要时 probe）→ status → start → 日志/进度 → 训练结束后 val → 同一 best 的 test → pack。正式 start 由用户执行，预检不会启动长训。路径、公式、门槛和恢复约束见 [README.md](README.md)。
