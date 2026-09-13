# 原 C24 共享 P3 截止位证书修复

本修复从 `82adb2265ecc8da32764fc5d5062e556f55f7c88` 在原 `codex/rtdetr-c24-lif-v1` 分支追加。没有访问AutoDL，没有正式训练、完整val/test或新性能结果。当前操作入口见 [AUTODL.md](../AUTODL.md)。旧preflight_fix报告保持为历史。

## 最新 LIGHT 审计

最新文件 c24_lif_v1_LIGHT_20260913_215234_188684.tar.gz，433,147 bytes，SHA256 `f725285bd46204ac5df302cfe330df304527b7af03eafe9a9ecdd277ba103f05`。已核验MANIFEST全部320项和127条结构引用，递归合并并保持列表顺序。见 [latest_light_audit.json](latest_light_audit.json)，读取器为 tools/c24_lif_v1_light.py，不反序列化权重。

服务器22阶段中20个PASSED，仅fusion_cuda_amp/half为REQUIRES_REVIEW，terminal_exception_stage=null。native初始化和非零分支B16/640均通过：真实增强batch，16张图72GT；6次原始loss有限；4次溢出跳过，65536→32768→16384→8192→4096，随后2次连续真实更新；336个参数张量恰好各入组一次、10个新增张量按原weight组；模型/优化器/scaler保存重载EXACT，source_model_unchanged=true。旧记录不作为新SHA的容量缓存，也不代表200轮稳定性。

原C24与组合的融合前完整有序300IDs相等，融合后完整有序300IDs也相等；交换ID两侧分数及逐对margin相等。旧包AMP替换5个、half替换7个。完整encoder_input指纹、整encoder最大扰动不同，不能宣称整个encoder或整个boundary字典相同。旧包缺共享P3动态证书和区域外竞争者证据，始终保留MISSING，未改旧JSON状态。原LIF的完整候选列表不同，不能充当本次豁免。

## 模型与配方保持

[源码审计](source_audit.json) 核验生产nn、初始化、AMP校准、loss、拓扑、评估工具和C2配置相对基点未变。原模块文件字节与各自原提交相等：

- C24原提交 f6e9dfda765046ae7691302cf5ec89d3f76cec5d；SCCA SHA256 `67b0d347d5305c16d48bb031cd10fd1d23b7ea447b58b2423166eb68b81dedca`。
- LIF原提交 0e95bbade3558b0d2b77c5531483c60810391d88；LIF SHA256 `26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`。
- 公共源SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
- nc1组合20,169,416参数、543状态，533公共+10新增，query300；原生Trainer重建534/543加载、9项原分类适配。
- 109字段配方仍200e、640、B16、seed42、AdamW、lr0=.0005、AMP=True，仅原有model/name/save_dir身份差异。

原BN前残差、LIF融合排除、生产top-k、Decoder/loss/matcher/DN和梯度保持；正式GradScaler仍从默认65536开始，不继承诊断scale、优化器或更新后模型。head.py固定LF SHA256为 `c092316743055f02adc84589d843588631ffe705fb0777178e0362d81471f4bc`，另记原始文件哈希；SCCA/LIF provenance始终核对原始字节。

## 同次四路证书与统一门禁

schema升为6。组合A/B和原C24 A/B按原受控公共源及父初值独立重建，使用同一x、seed120和原非零压力设定。在原生gather实际消费top-k的位置采集ID；没有改变生产排序或注入共享激活。双方各自使用自然A集合、B集合在自身融合前后完整重放。

从真实topology、feature shapes和展平顺序推导S：640时P3为80×80、共8400候选；160×192时自动推导20×24。没有固定ID名单或替换数量门槛。固定head源码及实际实例方法，并检查input_proj的1×1卷积/eval BN、enc_output逐token Linear/末通道LayerNorm和独立Linear打分；不能证明逐token独立的未来路径保持REVIEW。

分别在融合前、融合后跨模型核对全部共享prefix参数/BN、selection参数和非零SCCA状态，既保存完整状态集合哈希，也实际逐tensor torch.equal。P3原特征、encoder_input/features/logits、有效mask及合法anchors同时记录哈希和实际零容差比较。同侧必须精确相等，整个encoder哈希仍允许不同。

四路保存全部原始score转FP64的标量列表，验证实际300IDs合法、属于S且同侧父/组合完整有序列表相等。保存k/k+1、全部并列ID/数量、截止band、A_only/B_only及全部交换对margin。每对ma、mb非负，ma+mb不超过实测局部扰动加固定8个FP64 ULP舍入余量；没有FP16 ULP宽免。

两个模型分别使用全部合法区域外候选计算q、gap、e_O和delta_t，严格要求 min(gA,gB)>e_O+delta_t+slack_FP64。validator重算上述事实，不能用VERIFIED字符串代替证明。父operator和双重放独立核对，不递归要求祖父证明。原half完整fingerprint合同保留；新共享P3合同仅在证据完整时适用于AMP及half。

numeric、fusion、aggregate、blocking_summary、require_preflight和worker共用验证器。证书完成后再写fusion最终状态；summary重算融合事实。plan及worker schema一起升为6，source fingerprint包含新增工具。服务器22阶段仍必需，没有绕过B16的缓存。锁缺PID起始时间、归属不同或仍活跃时保留并停止。

## 本地实测和限制

本机RTX2060、PyTorch2.7.1+cu118，与历史服务器4090、2.1.2+cu121分别记录。同侧共享tensor最大差0且hash相同；共享state逐tensor相等。640时区域外2000候选全部合法：

| 模式/模型 | min gap | e_O | delta_t |
|---|---:|---:|---:|
| AMP 组合 | .306640625 | .001708984375 | 0 |
| AMP 原C24 | .30712890625 | .00146484375 | 0 |
| half 组合 | .306640625 | .00146484375 | 0 |
| half 原C24 | .3076171875 | .001220703125 | 0 |

固定FP64 slack为1.7763568394002505e-15。所有交换均满足局部不等式，原C24与组合共享截止事实相同。本机AMP/half各替换8个，cutoff与next均.919921875；不要求与旧服务器替换数量相同。

原融合容差FP32(2e-5,2e-4)、AMP/half(.008,.04)不变。连续张量最大差AMP .0028399229049682617、half .0029296875。自然逐行output_boxes最大差仍约.9373，原始输出完整保存。双方A/B重放通过原容差；组合output_boxes最大差0、output_scores .00048828125。初始化bbox的0误差不外推至训练后，保留更新后非零bbox融合回归。

本地19阶段全部接受：CPU/CUDA FP32为PASSED，AMP/half为ACCEPTED_WITH_PARENT_P3_CUTOFF_WARNING，其余PASSED。小批量native AMP从默认65536回退后连续真实更新2次，保存重载和源模型未污染检查通过。新SHA服务器B16/640、正式训练和完整val/test均NOT_RUN。

初期head换行及预期anchor dtype校验问题保留在outputs/c24_lif_v1_parent_gate_fix/probe和local_preflight报告。原native anchor按encoder输入dtype生成，autocast的log可能输出FP32；校验器已用实际输入dtype重建预期anchor，未改变实际anchor。修正后重新执行，未覆盖失败记录。

## 负例、打包和复现

负例覆盖：只改comparability、缺证书/父/B重放、输入/共享权重/torch.equal改变、未知跨token路径、同top-k但一个P3 logit改变、区域外候选到达cutoff、不被实测扰动支持的margin、完整父ID不同、shape/NaN/重复ID/错误mask、扩大容差、残差/BN错误和状态字符串掩盖失败。真实小形状同集合换位仍通过。合成server-reader夹具只检验启动协议，不表示真实B16容量。

LIGHT采用按内容寻址的结构分片去重，完整证书score不被摘要裁切，每个JSON分片≤64KB。测试逐项验证MANIFEST、引用大小/哈希与列表顺序，完整fusion roundtrip EXACT；读取后的统一门禁仍得到相同警告分类。标准库打包不加载模型、不重跑预检，压缩后硬上限8,000,000 bytes。实际包大小及测试结果见validation_summary.json。

有限复现（不包含服务器真实B16、正式训练或全量评估）：

```powershell
python tools/check_c24_lif_v1.py --source D:/MyProjects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt --output outputs/c24_lif_v1_parent_gate_fix/final_preflight
python tools/check_c24_lif_v1_parent_gate.py --preflight outputs/c24_lif_v1_parent_gate_fix/final_preflight --output outputs/c24_lif_v1_parent_gate_fix/delivery_contracts
python tools/check_c24_lif_v1_preflight_fix.py --output outputs/c24_lif_v1_parent_gate_fix/amp_contracts
```

本地结果不替代服务器新环境证书。一次start-direct会独立生成当前SHA的全部必要证据；任一条件不成立便保持REVIEW/BLOCKED并自动打包，不扩大豁免或换输入。
