# 本地有限验证

Windows / Python3.9.25 / PyTorch2.7.1+cu118 / RTX2060 6GiB，已有环境，无安装升级。最终检查记录为 outputs/check_final/checks.json，提交副本为本目录checks.json。检查在提交前运行，runtime.commit显示LIF基点；source_sha256与PROVENANCE标识实际被测组合代码，不把该基点冒充交付SHA。

| 检查 | 结果 |
|---|---|
| 原LIF模块 | Haar逐通道三细节排列、奇数replicate、Align手算、O零启动与激活后梯度通过 |
| 原CBR成功单测 | 原3项unittest直接运行通过；坐标、signed evidence、非零query/P3/box梯度、DN、保存重载 |
| 原box尺度梯度 | 解析检查 d(sum(output))/d(w,h)=1+rho*tanh(0.3)，max_abs=0，几何采样detach保留 |
| 三历史隔离进程 | C2/LIF/C19来自git archive独立导入；构造状态、非零eval与train输出max_abs=0 |
| 历史梯度回归 | 最大差C2=9.31323e-9、LIF=9.31323e-9、C19=1.30385e-8；原生loss分别57.097237/57.134510/57.105350 |
| 三组退化 | 双输出零→C2；CBR非零/LIF零→C19；LIF非零/CBR零→LIF，CPU/CUDA所有中间尺度、raw boxes/scores、query及输出max_abs=max_rel=0 |
| 整网尺寸 | 每组CPU/CUDA均1×3×640×640与1×3×160×192 |
| LIF孤立奇偶 | 80×80、79×80、80×79、79×81、1×1，输出形状与主Conv一致 |
| 非零FP32融合 | 两分支非零；BN mean[-0.3,0.4]、var[0.4,1.8]、weight[0.6,1.4]、bias[-0.2,0.3]；CPU/CUDA最大绝对差1.66893e-6 |
| 保存/重复融合 | 非零模型逐state相等，重载输出、重复fuse输出最大绝对差0；LIF BN保留、其他BN正常减少 |
| 真实自动融合入口 | RTDETR(checkpoint).predict及.val在保存的非零模型上通过；val/test入口各仅用复制的两张train图片，不是正式split评估 |
| 导出 | 每个两图fixture共600预测、6个GT，corrected_sorted_conf_mask_v1，同best SHA，曲线及混淆矩阵生成 |
| 原loss与梯度 | 两真实train样本B2/160，CPU FP32、CUDA FP32、CUDA AMP各3次有限更新；两个模块全部19张量及相关主路梯度有限且非零 |
| 动态DN | 真实get_cdn_group产生总Q=498、496、300，对应GT分组[1,3]、[2,7]、[0,0] |
| final_query | 训练/eval与最后实际Decoder层输出一致；CBR收到槽0同一P3张量；仅最后层box改变，score及前层box不变；原普通Decoder默认二元路径历史回归通过 |
| optimizer | 原Trainer.build_optimizer，345个可训练张量恰好各出现一次；新增13weight+6bias，无新norm；weight decay=1e-4，bias=0 |
| warmup | 直接执行未改Trainer方法内warmup代码块：正式B16/6048图，nb378、nw1890；首3batch accumulate=1；weight lr=0、2.64550e-7、5.29101e-7，bias lr=0.1、0.0999473545、0.0998947090 |
| smoke重载 | 模型及AdamW状态逐tensor精确重载；这些被更新的检查模型排除出正式初始化和结果包 |
| 配方 | 全109字段/类型核对，只有model/name/save_dir变动；数据train6048/45573、val1728/12840、test864/6663 |

容差预先固定：父回归 atol=2e-6/rtol=2e-5；FP32融合2e-5/2e-4；half/AMP融合3e-3/3e-2。接近零的中间激活会产生较大max_rel（CPU0.1376、CUDA0.0475），须结合绝对差和逐元素atol+rtol判断，完整值均记录，没有放宽容差。

初始化模型的半精度候选排序存在离散变化：true-half有218个位置改变、原行序bbox差0.7498824；AMP有211个位置改变、差0.7916726。检查已先比较排序前尺度及candidate_scores，再仅在诊断副本中重放同一候选索引，对齐原query。true-half对齐bbox最大差1.16602e-6、score-logit差0.00146484375、包含query的最大差0.0078125；AMP分别1.18464e-6、0.0009765625、0.00187123。生产代码没有固定排序/索引。真实half参数为float16；原CBR局部FP32使最终输出可能为float32，这是父模块原行为。

沿用原AutoBackend顺序：先FP32 fuse，再half。手动half().fuse()会触发原公共Conv融合函数的Half/Float bias dtype错误；该非原生顺序明确不支持，未修改父模块或公共融合函数掩盖它。

原生loss实测：CPU 57.141304→24.191809→24.688715；CUDA FP32 57.214050→23.964396→24.138121；CUDA AMP 57.270290→24.181843→24.317537。仅为有限工程检查，不能据此比较精度或收敛。smoke GradScaler初值128用于诊断；正式Trainer scaler保持原样。

时延只做未融合synthetic B1/640、1次预热+3次前向：CPU560.46ms/image，CUDA53.15ms/image；CUDA分配峰值649132032字节。不是AutoDL4090性能、训练显存或B16容量证据。

局部检查通过子进程隔离TF32/线程/RNG设置。PyTorch实际发出grid_sampler_2d_backward_cuda非确定性警告，已保留在日志；deterministic=True不保证所有反向逐位一致。未观察到本机replicate backward警告，不推断torch2.1.2上也没有。

ops_checks.json的派发/预检/OOM是明确mock的生命周期控制测试，不是真实服务器进程。约22.9MB测试包写入、逐文件散列读回和拒绝覆盖是真实IO；metadata验证使用fixture。未生成正式成功结果包。sync_checks.json记录单独的真实本地Git/linked-worktree测试；远程地址只在fixture中重定向。

NOT_RUN：AutoDL Linux/Python3.10.13/torch2.1.2+cu121、4090正式B16/640容量、真实tmux训练、200e及完整val/test、正式结果打包。服务器start-direct会重新运行有限回归并要求B16/640 CUDA AMP原loss前后向通过后才派发；B2/160不会被当作容量通过。服务器首轮实际参数组与warmup由只读callbacks另行记录。
