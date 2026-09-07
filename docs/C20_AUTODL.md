# AutoDL C20 操作

使用现有 `rtdetr` conda 环境，不安装、升级或降级依赖。下面假设 C20 已在 `/root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr` 获取到交付指定提交。最终交付消息给出该提交的完整 SHA 及安全创建 worktree 命令。

## 准备（不启动正式训练）

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr
bash tools/autodl_c20.sh prepare
```

强制现有 PyTorch2.1.2+CUDA、正确代码导入、C2源权重SHA、完整109字段、模块保护、组合接线、nc=1/80权重/RNG/分类头、train API、CPU/CUDA FP32/AMP/half、梯度/重载、optimizer、真实数据三步smoke。失败会返回非零并保留具体错误。只有全部通过才生成可启动计划。准备审计会使用GPU，但不执行200轮训练。

权威输入固定为主仓库 C2 run 的 args.yaml、`weights/rtdetr_r18_lite_imagenet_backbone_init.pt` 和 `configs/crack_autodl.yaml`。训练 project 保持 C2 原值，因此正式输出在主仓库 `runs/c_series/c20_rtdetr_r18_lite_cscef_cbr_e200_b16_onlineaug`。

## 手动启动一次正式训练

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr
bash tools/autodl_c20.sh start
```

会再次检查准备证据和实际训练初始化。拒绝已有run、launch锁、tmux、进程或退出记录；不会覆盖/续跑smoke。失败请先查日志，不删除结果目录或锁来绕过保护。

## 查看

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr
bash tools/autodl_c20.sh status
tail -n 80 -F outputs/c20/launch_c20/console.log outputs/c20/launch_c20/bootstrap.log
```

`Ctrl+C` 只退出 tail。附加训练窗口：

```bash
tmux attach -t c20_c20_rtdetr_r18_lite_cscef_cbr_e200_b16_onlineaug
```

`Ctrl+B` 后按 `D` 脱离，训练继续。训练完成后窗口保留最后输出并等待 Enter；按 Enter 打开交互 shell。真实退出码读取：

```bash
cat /root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr/outputs/c20/launch_c20/exit_code.json
cat /root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr/outputs/c20/launch_c20/process_exit_code.json
```

早期 prepare/start 环境错误位于 `outputs/c20/prepare_*.bootstrap.log` / `start_*.bootstrap.log`；每次prepare的详细审计分别留存 `launch_c20/audit_*.json` 和 `.console.log`。

## 完成后独立 val、test 与小包

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr
bash tools/autodl_c20.sh val
bash tools/autodl_c20.sh test
bash tools/autodl_c20.sh pack
ls -lh outputs/c20/c20_cscef_cbr_small.tar.gz*
cd outputs/c20
sha256sum -c c20_cscef_cbr_small.tar.gz.sha256
```

val/test 使用同一个 best.pt 的明确 split，不重跑 C2/C17/C19。重复评估与 pack 会保护已有目录/文件。小包不含权重/数据图片/全仓库；超过20 MiB时打印大成员并失败。

复制到 AutoDL 常用文件下载目录：

```bash
mkdir -p /root/autodl-tmp/c20_download
cp -n /root/autodl-tmp/projects/Crack_RTDETR_cscef_cbr/outputs/c20/c20_cscef_cbr_small.tar.gz* /root/autodl-tmp/c20_download/
ls -lh /root/autodl-tmp/c20_download/
```

在 AutoDL 的 JupyterLab 文件浏览器进入 `autodl-tmp/c20_download`，下载 `.tar.gz`、`.sha256` 和 `.inventory.json`。本轮用户不提供SSH，因此不编造主机或scp端口。权重继续留服务器原run。
