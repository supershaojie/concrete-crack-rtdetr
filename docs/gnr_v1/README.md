# GNR v1：母版契约核验与阻塞记录

当前状态：**CONTRACT_CONFLICT，GNR 功能尚未实现**。此目录交付母版核验证据，
不代表 TECHNICAL_PASS、可启动正式实验或已有训练结果。

用户实施文档指定母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec`，
第 3.4 节固定 `alpha=0.75, gamma=2`，并要求：

> 首先核实母版确实采用这些 VFL 参数及数学形式，不为了套公式改母版。

同一文档要求只有真实算法契约冲突才报告阻塞。目前确实存在这一冲突：

| 核验项 | 文档 | 指定母版实际行为 |
| --- | --- | --- |
| VFL alpha | 0.75 | 0.25 |
| VFL gamma | 2 | 1.5 |
| nc | 1 | 1（Trainer 按数据集转换后） |
| 最后分类头 | 带 bias 的 Linear | Linear(256, 1)，bias 为 [1] |
| 普通 query | 核实实际数量 | 300 |
| decoder 层数 | 核实实际数量 | 3 |
| 未融合参数量 | 20,149,765 | 20,149,765 |
| nc=80→1 形状变化键 | 9 | 9，逐键形状见 JSON |
| 归档完整配方 | 历史 109 项 | 109 项（此次仅统计字段数） |

调用链已经通过实际模型实例确认：

1. `ultralytics-main/ultralytics/nn/tasks.py` 的
   `RTDETRDetectionModel.init_criterion()` 调用
   `RTDETRDetectionLoss(nc=self.nc, use_vfl=True)`。
2. `RTDETRDetectionLoss` 继承 `DETRLoss.__init__`，该构造函数默认
   `gamma=1.5, alpha=0.25`。
3. `ultralytics-main/ultralytics/models/utils/loss.py` 显式调用
   `VarifocalLoss(gamma, alpha)`，覆盖了 VFL 类自身的 `2.0/0.75` 默认值。
4. 母版训练入口没有另行覆盖上述 VFL 参数。

因此不能将文档的固定导数作为母版的真实负梯度强度，也不能把固定
`0.75*p**2*softplus(z)` 当成原负项做差分替换。gamma 不同，差异并非统一比例。
修改母版 VFL 又会违反原分类监督保护要求。

建议的解决方式是明确授权：保留母版 `alpha=0.25, gamma=1.5`，
GNR 解析导数使用这两个真实参数，其它公式及作用范围保持原文。
这属于对文档固定算法参数的修订，尚未擅自执行。

## 已执行的核验

在 Windows / Python 3.9.25 / PyTorch 2.7.1+cu118 上执行了 CPU 核验：

- 对 9 个关键源文件/YAML 与指定母版 Git 内容逐个比较（统一 CRLF/LF）。
- 实例化 nc=1 和 nc=80 的原模型；调用真实 Trainer 的 `set_model_attributes`
  后从模型构建 criterion，没有替换损失类或参数。
- 检查参数量、分类头、query 数及 9 个分类相关形状变化键。
- 对实际 VFL 负项做 PyTorch autograd，对照包含调制因子导数的解析式。
  logit 为 `[-20,-8,-2,0,2,8,20]`，结果有限；实际参数对应的最大绝对误差
  为 `2.9802322387695312e-08`，预设容差为 `rtol=1e-6, atol=1e-7`。
- 文档参数导数与实际母版导数的最大绝对差为 `0.546917736530304`。
  在 `z=0` 处，实际导数为 `0.09014376997947693`，文档式为 `0.22371509671211243`。

完整机器记录见 `mother_contract.json`，复现入口为：

```bash
python tools/check_gnr_v1_contract.py --output outputs/gnr_v1/mother_contract.json
```

脚本返回 **2** 表示已确认文档与母版冲突。它只构造 CPU 模型与计算示例导数，
不运行训练、不读取数据集、不加载训练 checkpoint，不写入权重。
`cuda_available=true` 仅为环境信息；本次没有执行任何 CUDA/AMP 验证。

## 待解决与未执行

算法契约选择需要用户答复；后续 GNR 权重计算、loss 接入、生命周期入口、
母版现象诊断、CUDA/AMP 真实更新预检、正式训练和独立 val/test 均未执行。
此时不存在 GNR 功能提交，也不存在可用于启动正式训练的服务器入口。

独立分支为 `exp-rtdetr-r18-lite-gnr-v1`，从指定母版建立。
原工作目录、LIF-Down、CBR、匹配、loss、训练入口均未修改；未启动或停止其它实验。
