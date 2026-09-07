# C21 本机验证记录

日期：2026-09-08（Asia/Shanghai）。环境为 Windows、Python 3.9.25、PyTorch 2.7.1+cu118、RTX 2060 6 GiB、仓库内 Ultralytics 8.4.21。结果仅适用于本机；AutoDL 的 PyTorch 2.1.2 必须重新执行 prepare。

## 已通过

| 项目 | 结果 |
|---|---|
| 实际替换 | `model.26.decoder.layers.{0,1,2}.cross_attn` 均为 SALAMSDeformAttn；三个对象和所有注意力参数存储独立 |
| 参数 | 每层完整 214,280，新增 8,680；三层新增 26,040；nc=1 模型 20,108,812 |
| 统一初始化 | 源 SHA256 符合任务；533 公共状态全部精确复制，21 新状态；raw/RTDETR(checkpoint)/YAML.load 三种重载一致 |
| nc=1 / nc=80 | 公共构造/加载状态逐键相等，CPU/CUDA 构造 RNG 相等；nc=1 的 9 个分类状态单独核对；真实 train API 重建相等 |
| 640 batch1 完整输出 | CPU FP32、CUDA FP32、CUDA AMP、CUDA half 均逐位相等，最大绝对差 0 |
| 320×640 非方形完整输出 | 同上四种精度路径逐位相等 |
| 640 batch2 含 DN 训练态输出 | CPU FP32、CUDA FP32、CUDA AMP 与同初始化 C2 逐位相等；控制相同 DN RNG；此项不执行梯度更新 |
| 注意力行为 | 零偏置时权重/坐标/输出逐位不变；非零偏置确实改变尺度总权重且总和为1；层内相对点权重保持；query 与框尺寸都能改变偏置 |
| 边界和接口 | value_mask、单level框广播/逐level框、H/W顺序、尺寸clamp、几何detach/原采样梯度、动态Q；基线2D点保留，SALA拒绝2D点 |
| 真实数据 smoke | 从本地原训练集复制32张真实图像/标签，原生在线增强；独立模型/optimizer，640、batch2，三个 AMP 更新；另有同一真实640批次 FP32 backward |
| 梯度 | 初始 level_head.weight 有限非零，上游新增层初步梯度为0；第三步全部21个新增参数张量都有有限非零梯度 |
| Optimizer | 新增12个weight张量、9个bias张量；完整覆盖、无重复，原参数组不变；weight decay 0.0001，bias不衰减 |
| Checkpoint/EMA | smoke训练后重载 FP32 输出逐位相等，half推理有限；正式初始化hash未变，初始optimizer为空、EMA updates=0且状态精确相等 |
| 工具 | 8项 unittest 全通过；Bash语法和 git diff --check 通过；真实4图val/4图test跑通且校验同权重；打包/重复启动锁/退出码7测试通过 |

最终三批 loss：`56.56080246, 44.67872238, 43.11984634`。DN split 分别为 `[200,300]`、`[200,300]`、`[198,300]`。这些数值仅说明 smoke 可运行，不是训练收敛或效果证据。smoke沿用C2 200轮的配置对象，但覆盖 `_do_train` 只执行三个显式更新，没有进入epoch循环；正式初始化文件与EMA不受其污染。

初始输出头 bias 在各尺度产生共同偏置，其第一步梯度可能因 softmax 平移不变性为零或仅有舍入量；因此第一步重点检查 head.weight，不要求所有新增张量第一步非零。零初始化后续层获得梯度的过程符合设计。

原数据目录的缓存写入曾遇到权限限制，随后在 C21 的 outputs 内复制真实样本并保留 SHA256 清单，所有缓存写入副本。未修改原图像、标签、配置或旧实验权重。最终 audit 完整使用该副本；服务器 prepare 使用服务器真实完整数据配置。

## 精度、环境与尚未验证

原生 YOLO26n AMP 辅助检查在本机因依赖/加载问题跳过；没有将其记为“检查通过”。SALA 的整模型 AMP 对齐、真实损失反向和三个 GradScaler/optimizer 更新已单独实际执行并通过。工具设置 `YOLO_AUTOINSTALL=false`，不自动安装共享依赖。

原有 CUDA `grid_sample` backward 报告非确定性警告，C2 的 `deterministic=True` 使用 warn_only；未改动此规则。逐位相等结论针对受控初始前向，不承诺多个正式训练轨迹逐位一致。

一次本机未融合 FP32、batch1、640随机输入耗时探测（预热3次、计时10次）：C2约32.67ms，C21约32.85ms。该短测受本机负载影响，不能证明速度提升、作为服务器FPS或测得GFLOPs。真实部署延迟仍需服务器充分测量。

本次没有正式200轮训练、完整训练best的val/test、精度收益或消融结论。4图val/test使用的是独立三步smoke权重，只验证工具流程，不是C21实验指标。AutoDL prepare/start/tmux 尚未在服务器执行；未更改现有服务器环境。

## 证据与复现

- [结构化摘要](c21_local_validation.json)
- [109字段完整比较](c21_c2_109_field_comparison.json)：本机初始化路径；服务器prepare重新生成实际服务器路径。
- [逐键完整审计](c21_evidence/audit.json.gz)
- [初始化逐键审计](c21_evidence/initialization.json.gz)
- [最终审计日志](c21_evidence/audit.console.log.gz)
- [真实样本来源和hash](c21_evidence/local_data_manifest.json.gz)
- [证据文件hash清单](c21_evidence/inventory.json)

审计在提交前的工作树执行；完整报告保存逐源文件规范化hash，交付前已核对与最终源码一致。报告中的 git parent 仍是C2，不能当作服务器审计。服务器prepare要求当前干净提交重新审计并逐项绑定，不能拿本机报告直接启动。

```powershell
Set-Location D:\MyProjects\Crack_RTDETR\outputs\worktrees\sala
$env:PYTHONPATH = "$PWD\ultralytics-main"
$env:YOLO_AUTOINSTALL = 'false'
& D:\miniconda3\envs\rtdetr\python.exe -m unittest discover -s ultralytics-main/tests -p test_sala.py -v
& D:\miniconda3\envs\rtdetr\python.exe -m unittest discover -s ultralytics-main/tests -p test_c21_tools.py -v
# 再次完整审计请给 smoke-dir/report 新名字，保留先前验证结果。
& D:\miniconda3\envs\rtdetr\python.exe tools/audit_rtdetr_r18_lite_c21.py `
  --source 'D:\rtdetr跑结果\c2 200e在线\c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053\weights\rtdetr_r18_lite_imagenet_backbone_init.pt' `
  --initialized weights/rtdetr_r18_lite_sala_imagenet_backbone_init.pt `
  --require-torch 2.7.1 --require-cuda `
  --smoke-data outputs/c21/local_data/data.yaml `
  --smoke-dir outputs/c21/local/smoke_recheck --report outputs/c21/local/audit_recheck.json
```
