# AutoDL 使用

只用已有 `/root/miniconda3` 的 `rtdetr` 环境。主仓库 `/root/autodl-tmp/projects/Crack_RTDETR`，兼容分支 `codex/rtdetr-triad-compat`，独立 worktree `/root/autodl-tmp/projects/Crack_RTDETR-triad-compat`。本次本地工作没有启动正式训练。

首次同步使用最终回复提供的**字面40位 SHA**执行 `git show SHA:tools/sync_triad_compat.sh | bash -s -- SHA`。脚本 fetch 指定分支、验证固定 SHA 属于该分支历史及 C2 后代，创建 detached worktree 或复用相同 SHA 的干净 worktree。已有修改、不同 SHA、不同仓库或已有 pin 内容冲突时停止。不会 reset/clean、自动切换已有工作树、删除目录或启动训练。同步将记录 `outputs/triad_compat_sync.json`，启动时再次核对。

原 C2 args 必须位于主仓库 `runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`；其全部109字段和类型与审计副本一致，只修改 model/name/save_dir。data/project 原值保留。唯一源权重仍为主仓库 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，按 SHA256 严格检查。不能使用任何 best.pt 作为新模型初始化。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-triad-compat
# Round A
bash tools/autodl_triad_compat.sh cscef_v6 start-direct
bash tools/autodl_triad_compat.sh scca_v2 start-direct

# 状态
bash tools/autodl_triad_compat.sh cscef_v6 status
bash tools/autodl_triad_compat.sh scca_v2 status

# Round B
bash tools/autodl_triad_compat.sh cscef_v6_scca_v2 start-direct
bash tools/autodl_triad_compat.sh cbr_v2 start-direct

# Round C
bash tools/autodl_triad_compat.sh triad_v1 start-direct
# 补全另两个 pair
bash tools/autodl_triad_compat.sh cscef_v6_cbr_v2 start-direct
bash tools/autodl_triad_compat.sh scca_v2_cbr_v2 start-direct
```

各 variant 独立 tmux/session、日志、run、lock 和 state；可以并行启动，资源是否足够仍由实际 GPU 决定。脚本不停止其他实验，不重写 CUDA_VISIBLE_DEVICES 等 GPU 设置，不因 OOM 降 batch。共享 AMP 检查资源只从主仓库已有文件原样复制并以原子方式发布，无 pip/conda 安装或升级。

start-direct 在派发前核对环境/固定 SHA/nc/拓扑/参数量/C2源哈希/公共映射/完整配方/数据路径/不覆盖规则/tmux/同variant进程，并执行所选模型原生 loss/DN、AMP/half 与梯度预检。预检模型均为干净初始化的独立副本，不写回正式初始化。正式 Trainer 回调再次核对 actual args、AMP 和 optimizer；原生默认 GradScaler 不改。

状态区分 NOT_STARTED、DISPATCHED、RUNNING、SUCCESS、FAILED；SUCCESS 需要匹配的 worker token、Python与shell退出0、worker完成证据和训练产物，不根据 tmux/best.pt/results.csv 单独判定。失败锁和日志保留，脚本不会自动删锁重试。

训练 SUCCESS 后执行：

```bash
variant=triad_v1
bash tools/autodl_triad_compat.sh "$variant" val
bash tools/autodl_triad_compat.sh "$variant" test
bash tools/autodl_triad_compat.sh "$variant" pack-complete
```

val/test 均固定选该 run 的 best.pt；test 必须有同 checkpoint SHA256、同代码 SHA、同数据配置、同评估设置的已完成 val。新统一独立入口沿用 C24/C25/C26 的 corrected_sorted_conf_mask_v1，显式 seed42，workers0；640/batch16/FP32/conf=.001/iou=.7/max_det300/augment=False 不变。原训练内验证器和 best 选择保持 C2 原生代码。

pack-complete 只打包已经成功完成的训练/val/test，检查原始 args、预检、源快照、映射、拓扑/梯度证据、曲线、AP75、预测/GT覆盖、checkpoint和文件哈希。按最近 C26 complete 政策包含 best/last，流式输出到主仓库 `downloads/triad_compat/<variant>/`，附 manifest、SHA256、inventory 和 verification；不自动补跑 val/test，不覆盖已有包。
