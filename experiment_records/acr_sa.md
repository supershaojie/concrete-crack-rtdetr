# C22 / C23 ACR-SA 实验登记

日期：2026-09-08。分支：`codex/acr-query-attention`，基于 C17 `0c53a9cf7c8d65530b82f6ce880b3c9d8e6da139`。

检查了仓库现有配置记录、各 worktree、分支和本地结果目录。C21 已用于 SALA；本地 `D:/rtdetr跑结果/c22 ACR-SA` 是空目录，与本任务意图一致，不写入或覆盖该目录。新运行名在本仓库未占用，服务器 prepare/start 再检查并拒绝覆盖。

| 实验 | 处理 |
|---|---|
| C2 / C17 / C19 / C20 / C21 SALA | 已有工作保留，不改分支或历史结果 |
| C22 C2＋ACR-SA | 新正式单模块配置，完成验证后由用户手动 start |
| C23 CSCEFv51＋ACR-SA | 新组合配置，仅预验证；C22 完整 val 支持后再由用户决定 |

本次仅短 smoke 和工具验证，无正式训练、收益结论或 test 选型。统一源初始化和 C2 全量配方锁定，完整说明在 `docs/ACR_SA.md`。
