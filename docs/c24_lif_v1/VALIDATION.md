# 本地有限验收

本机：Python 3.9 系列（精确值见 checks.json）、torch 2.7.1+cu118、RTX 2060。
所有检查均为合成有限输入或只读数据清单，不做完整 val/test，不运行训练 epoch。
AutoDL torch 2.1.2+cu121/4090 尚未执行，服务器 B16/640 为 NOT_RUN。

## 已通过

* 成功包及 Git 模块哈希逐字核验；全109字段类型和值一致，仅允许3个实验身份字段不同。
* 4模型参数/状态数；533公共状态、10父模块初值、零新增buffer、9分类适配、native Trainer逐键加载。
* C2、原 C24、原 LIF 的历史 Git archive 以独立子进程加载，实际前向精确一致。
  非零原模块的输入/参数梯度精确一致；SCCA 的 pre/post norm 均比较。
* CPU/CUDA 的 640×640 与160×192 三组父退化；LIF奇偶尺寸、Haar分带、Align手算和输出先学习。
* 非默认 LIF BN、非零 SCCA/LIF 的融合；LIF BN状态保留，普通Conv/RepConv正常融合，重复fuse/保存重载。
* 原 RT-DETR loss、真实动态DN、CPU/CUDA backward、CUDA AMP更新；10新增参数和相关主路梯度有限且非零。
* 原 Trainer optimizer 336张量恰好一次，所有norm/bias分类与原规则一致；非零更新的模型和optimizer保存重载精确。
* 一次性更新后的非零 bbox 头另外通过FP32融合检查，避免只依赖零bbox头的退化场景。
* 真正 native EMA validator 分支（单次前向后受控终止）及真实 predictor AutoBackend；CUDA均实际FP16且finite。
* 非法candidate、缺key、NaN、空PASSED、仅A重放、错误父集合等负例继续拒绝；实际错误BN/漏残差也被拦截。
* 异常路径后 torch.topk、实例forward、hooks、RNG、backend和设备环境恢复。

## 数值结论必须保留候选身份

| 同精度融合 | 640×640自然关系 | 160×192自然关系 | 接受 |
|---|---|---|---|
| CPU FP32 | PERMUTATION | IDENTICAL | PASSED，按ID对齐 |
| CUDA FP32 | PERMUTATION | IDENTICAL | PASSED，按ID对齐 |
| CUDA AMP | SET_DRIFT，8个替换 | PERMUTATION | REQUIRES_REVIEW |
| CUDA true-half | SET_DRIFT，8个替换 | PERMUTATION | REQUIRES_REVIEW |

全部模式的连续张量通过预先声明的容差；低精度双方A集合和B集合的完整重放均通过。
记录包含 LIF/P3/P4/P5、encoder输入/特征/logits/有效anchor、实际gather IDs、query/reference、
三个decoder层query、框/logits/分数与最终输出。整数ID从原topk的真实返回值取得，未重算猜测。
完整自然序列、差集、float64边界统计、自然输出差异和双重放小报告保存在逐模式JSON。

FP32融合容差为原LIF的 atol=2e-5/rtol=2e-4；AMP/half使用已声明的 atol=.008/rtol=.04。
没有因本次数值失败而扩大容差。AMP不是model.half，独立评估half=False也不代表EMA验证不用half。

当前两个非零模块均会改变encoder特征，任意C2不是可比截止位父对照。
因此**双重放通过不等于自然集合漂移无害**。实现保留窄例外判据，但没有伪造可比父模型来触发它；
AMP不会获得true-half例外。自然SET_DRIFT没有加入训练成功状态列表。
这属于工程预检状态，不属于检测性能失败，也不预言训练收益。

## 生命周期和包

故障小包测试用131,432,311字节的大JSON/多日志，保留首错、candidate IDs与重放状态并满足8,000,000字节硬上限。
实际生成大小、哈希和删减项数见 `ops_checks.json`。标准库打包不导入torch、不反序列化模型、不重跑预检。
JSON摘要保持可解析；有必要删减的日志均以标注范围的.txt保存。旧文件/大包不自动删除。
PID复用、重复启动、错误token、死owner归档、无关owner保护有有限单元验证。
真实同步脚本在本地独立Git fixture中验证首次/幂等/脏工作树/不同SHA/非工作树保护，证据见 `sync_checks.json`。

服务器 start-direct 会重新执行本机不能替代的CPU/CUDA门禁和一个真实原生增强B16/640 loss/backward/update容量批次。
历史Git动态回归使用已提交的本地证据并核验原模块字节，避免服务器额外fetch父分支或重复导出历史源码。
门禁未通过便不会建立正式训练进程；RESOURCE_BUSY/OOM保留原配方，不降低batch、query、分辨率或关闭AMP。

机制参考仅作解释，不替代实际仓库版本：[top-k](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)、
[浮点精度](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)、[AMP](https://docs.pytorch.org/docs/2.14/amp.html)。
