# 公共预测接口与评测

本入口仅处理导出的预测和独立 GT，用 NumPy、CPU PyTorch 计算指标；不导入 Ultralytics、检测模型或训练器，不安装任何框架。
`native_metrics.py` 中 `smooth / compute_ap / ap_per_class / box_iou / BaseValidator.match_predictions` 函数体逐字取自母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。出处和函数哈希在 `native_provenance.json`，有限测试比较完整 AST。仅隔离依赖，并为 NumPy 1/2 的梯形积分函数选择提供原来 `checks.check_version` 所需的窄接口。禁用绘图分支。授权和许可继承仓库中的 Ultralytics AGPL-3.0 源码。

预测文件为 UTF-8 JSONL，可 gzip。首行必须是：

```json
{"type":"metadata","schema_version":1,"identity":{"model":"模型名","model_code_sha":"40位实际代码SHA","checkpoint_sha256":"64位实际权重SHA256","dataset_identity_sha256":"取自本次COCO的info字段","evaluation_config_sha256":"取自evaluate.POLICY_SHA","postprocessing":"实际原生NMS/top-k/resize与逆变换说明"}}
```

每张图片必须恰好一行，包括零预测图片，记录按 `image_id` 递增排列以固定全局分数并列时的输入顺序。`image_id` 来自 COCO，禁止凭文件顺序或 basename 重新编号。原图像素 `xyxy`，不裁剪或舍入；`category_id=1` 为 crack，YOLO 类别 0 显式映射到 1，TorchVision 背景类 0 不能作为前景。

```json
{"split":"test","image_id":7777,"width":4096,"height":3072,"box_format":"xyxy","coordinate_space":"original_image_pixels","predictions":[{"category_id":1,"bbox":[20.125,30.5,80.75,90.25],"score":0.9100000262260437}]}
```

这些数字只是格式示例。空预测写 `"predictions":[]`。元数据还应增加官方配置、包版本、预训练来源、原生后处理参数和坐标逆变换来源；八个实际导出适配器尚未接入。新框架不得保留隐藏的高置信度过滤后再宣称使用公共 `conf=0.001`。

共同规则：逐图按置信度降序排列，随后用已排序分数生成 `score > 0.001` mask，最多 300 个检测；不加 NMS。相同分数保留导出顺序，历史缓存原有的原生排序次序得到保留。匹配使用母版逐 IoU 贪心实现：IoU 降序，先唯一化预测，再唯一化 GT；按类别隔离，不用 scipy/Hungarian。AP 使用 0.50 至 0.95 共十个阈值、precision envelope、101 点线性插值和梯形积分。全体预测的 AP 排序仍使用源码原有的 `np.argsort(-conf)`，不同 NumPy 版本及分数并列可能产生细微差异，报告保存运行版本。

P/R 是模型自身平滑最大 F1 工作点：1000 点置信度网格、AP50 下 P/R 曲线、均值 F1 经 `smooth(...,0.1)` 选索引。它不是共同固定阈值的 P/R。阈值 `0.7` 是历史后处理参数，在 RT-DETR 路径中未使用；没有把 AP 限制到 0.7。框与 IoU 采用 FP32。没有 GT 的整个 split 返回未定义 AP；有 GT 而全空预测得到零 AP；漏图片记录直接报错。

```bash
python benchmarks/comparison/evaluation/evaluate.py \
  --gt outputs/comparison_prepare/RUN/coco/test.json \
  --predictions /path/to/exported_predictions.jsonl.gz \
  --output outputs/comparison_prepare/RUN/model_test_metrics.json
```

可用的母版/LIF 高精度旧缓存可用 `--legacy-cache .../predictions_gt.jsonl.gz --legacy-metrics .../metrics.json` 代替 `--predictions`。适配器核实缓存 SHA256、split 清单指纹、图片尺寸、类别和全部图片覆盖，GT 始终来自独立 COCO。旧缓存内嵌 GT 不被拿来替代独立标注。原生评测在 640 空间计算，而当前公共入口在原图像素计算；GT 归一化逆变换、FP32 运算、分数并列和 NumPy 版本可能带来数值差异。输出逐项保存实测差值，不以历史参考值作为通过条件。

C2/CBR 原生 `predictions.json` 使用三位小数框和五位小数 score，不能认证精确复算，入口不会把它冒充高精度缓存。需要后续明确授权的统一复评；本轮不运行模型推理。COCO JSON 是交换格式，这里未以 COCOeval 替换母版 AP 算法。
