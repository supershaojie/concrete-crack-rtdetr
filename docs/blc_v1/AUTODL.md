# BLC 服务器入口

完整固定 SHA 在本次交付的 `HANDOFF_FIXED_SHA.md` 中。不要使用分支浮动 HEAD 代替交付 SHA。
仓库主目录 `/root/autodl-tmp/projects/Crack_RTDETR`；独立工作树 `/root/autodl-tmp/projects/Crack_RTDETR-blc-v1`。
`tools/sync_blc.sh FULL_SHA`：先 cat-file，缺少目标才按指定分支 fetch；HTTP/1.1、进度、无 tags、禁止交互、
lowSpeedLimit=1/lowSpeedTime=60；每次 timeout120秒、至多5次。核对 origin/common-dir/完整HEAD/母版祖先关系；
已有不同 HEAD 或改动时退出保留，不 reset/clean，不切换主目录或其他 worktree。

以下每个 `blc_server.sh` 命令都会独立加载 conda.sh、激活现有 rtdetr、设本 worktree PYTHONPATH、验证实际导入路径。
不升级依赖。日志实时 tee，保留 Python 真实退出码。允许已有GPU进程，不设空闲门槛，不kill任何进程。

初始化与有限预检（不启动正式训练）：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1
bash tools/blc_server.sh environment
bash tools/blc_server.sh init --both
bash tools/blc_server.sh init-preflight --both
BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh preflight
bash tools/blc_server.sh plan
```

若任何一步失败，先检查本次时间戳报告，不覆盖报告后重试。preflight 最多总计16个训练batch，包括resume诊断。
如果服务器仍有AMP/half融合PENDING，start会拒绝；需要针对已保存误差继续诊断，不能创建“许可文件”跳过。
原生AMP自检只复用主目录现有 `yolo26n.pt` 与 `bus.jpg`（它们是母版自检依赖，不是YOLO26项目实验）；
缺文件明确退出，不下载或安装模块包。正式权重仍只能来自指定公共RT-DETR初始化。

之后用户显式决定开始主组合时，用以下 tmux 命令。已经存在的 BLC session 仅进入，不发送第二条训练命令。
已有正式 run 不覆盖、不自动添加数字后缀；未完成 checkpoint 用 resume；已完成200轮但final_eval失败也不重开训练。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1
if [ -n "${TMUX:-}" ]; then
  BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh start
elif tmux has-session -t '=blc-v1' 2>/dev/null; then
  tmux attach-session -t '=blc-v1'
else
  tmux new-session -s blc-v1 'cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1; BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh start; rc=$?; echo "BLC start exit=$rc"; exec bash'
fi
```

显式续训（仅本实验真实未完成的正式 last.pt）：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1
if [ -n "${TMUX:-}" ]; then
  BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh resume
elif tmux has-session -t '=blc-v1' 2>/dev/null; then
  tmux attach-session -t '=blc-v1'
else
  tmux new-session -s blc-v1 'cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1; BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh resume; rc=$?; echo "BLC resume exit=$rc"; exec bash'
fi
```

完成训练后，原规则选择 best；独立 val 首次锁定 best SHA，然后才能对相同SHA独立 test。
训练早停同样需要正常结束记录；200轮已结束但 final_eval 失败时，可保留best并独立val，无需从头重训。
参数固定为 corrected_sorted_conf_mask_v1、640/B16/workers0/FP32/conf.001/iou.7/max_det300，原一对一匹配，无额外NMS。
保存P/R最大F1工作点、AP50/AP75/mAP50–95、十IoU AP、图数/GT、完整精度JSON、曲线及混淆矩阵。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1
BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh val
```

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1
BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh test
```

仅打包现有产物；不隐式训练、评估或下载；缺项如实 NOT_RUN/missing，保留旧包：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-blc-v1
BLC_VARIANT=cbr_lif_blc_v1 bash tools/blc_server.sh pack
```

默认只给主组合做容量验收。两份配置都会做结构/初始化验证。本次不运行单模块消融。
以后主组合收益经val审阅后，先显式 `BLC_VARIANT=blc_v1 ... preflight`，随后才可使用
`BLC_VARIANT=blc_v1 bash tools/blc_server.sh start --ablation-after-review`（建议独立tmux session `blc-ablation-v1`）。
临时关分支的输出变化不能替代重新训练的消融；不以中期val与历史最终test比较。

**本次正式训练 NOT_STARTED；最终 test NOT_RUN；未登录服务器。**
