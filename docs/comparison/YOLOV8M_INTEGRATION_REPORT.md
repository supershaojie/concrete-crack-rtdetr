# YOLOv8m 代码接入交付报告

2026-10-02，北京时间。代码交付，不是实验结果；没有启动正式训练、完整val/test推理或YOLO11m任务。
普通提交及远端SHA以本次最终汇报与Git回执为准；本报告随`bench/yolov8m`交付。

## 来源、隔离与版本选择

从公共固定提交`529c456b9404f1d9ab66d82d2b2f9ec7e0c98545`建立独立工作树：
`C:/Users/o'v'o/.codex/worktrees/bench-yolov8m/Crack_RTDETR`。
远端为`https://github.com/supershaojie/concrete-crack-rtdetr.git`，分支`bench/yolov8m`。
读取了README、公共API/协议、母版args与源码依据；本工作树及祖先未发现额外非空AGENTS约定。
未更改主工作区、YOLOv5m工作树、公共分支远端、母版实现或历史结果。

拟定官方v8.3.0的tag解析为`6e43d1e1e5db72afbf686dee6745669bcb124b0a`。检查发现：该版本`Detect.__init__`
无条件构建DWConv分类分支，而官方COCO YOLOv8m权重是Conv→Conv→输出Conv的原始分支；nc=1重建会改变头结构。
仅修复此问题也需要改动head及parse_model的legacy分发，因此选择官方已包含该修复的**v8.3.20**，不手改模型结构。

实际冻结：

- tag object：`3b6e1b124174107711942e4b85a93fab7c282c6f`。
- 官方commit：`f4d8f7765a490f3920e2d14c592a2967e347f185`。
- COCO检测权重：[官方固定v8.3.0 assets release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt)。权重release与源码tag分别记录。
- 实测大小：52,136,884字节。
- 权重SHA256：`5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`。
- 补丁SHA256：`461f2d4ce4ed99e691517ea316df4ebfb9a434d5f773240e6ffc9a51340ccbe1`。
- 补丁后规范化源码树SHA256：`7588e53839944a7cd34cde919744a7903343f06acb2f24655c94aa397b213dfb`。

小补丁只处理已校验本地checkpoint的`torch.load(weights_only=False)`兼容，以及官方旧下载器错误删除Windows真实路径内单引号的问题。
没有head、loss、DFL或assigner补丁。源码与许可证保留在忽略的官方检出目录，来源/许可见接入目录`NOTICE.md`及`upstream.lock.json`。
正式训练前确定版本，不使用任何验证/测试分数选版本。

替换版本后重新比较了cfg、trainer、YOLO detection trainer、dataset/base/build、augment、predictor和metrics相关源码。
默认配置值没有变化（仅task注释增加obb），augment.py逐字相同；trainer变化主要为CSV时间列、CSV精度及close_mosaic对args取copy。
cache=False不走新添的disk-cache路径。YOLOv8 legacy分发、原生fitness、双倍val batch、关闭条件及NMS/逆变换均按实际v8.3.20重查。

## 实际模型与配方

官方COCO模型实测25,902,640参数；按官方`DetectionModel(cfg,nc=1).load(weights)`后为25,856,899参数（均未融合）。
识别同时检查M深宽系数、三个Detect尺度、reg_max16、原始Conv分类分支、参数量和类数，避免只看文件名。
实际469/475个state_dict张量精确转移，缺少6个是三个类别输出Conv的weight/bias（80→1）；完整key列表在运行的initialization记录中。

冻结配方：200轮上限、patience50、batch16、640、workers8、GPU0、seed42、nbs64、AMP、deterministic；
SGD、lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、cosine、warmup5、warmup momentum .8、bias LR .1。
这是预先确定的项目配方，未声称全部是官方默认；未改YOLOv5m的lrf。
其余字段从固定官方默认展开，含box7.5、cls.5、dfl1.5；运行保存完整展开值、环境、实际优化器/参数组、LR/累积和batch。
warmup时累积从1到4，之后通常4；weight decay缩放后仍.0005。DFL固定投影参数按官方训练器处理，freeze=null不代表改变官方DFL机制。

原生fitness是`.1*AP50+.9*mAP50–95`，本适配的best/patience改用训练val的完整精度mAP50–95，并保留原生fitness。
相等时按官方规则更新best。训练内val固定batch16、workers0、FP16（CUDA AMP）、conf.001/iou.7/max_det300、rectFalse。
每轮仍原生验证；训练结束不重复隐式验证best、不strip优化器，随后由独立FP32入口导出同一best。
母版历史best及其选择过程保持原样；两个模型最终公共指标和原生日志分开。

增强表见`YOLOV8M_AUGMENTATION.md`。本轮对齐加法HSV、训练square stretch及活跃的几何/Mosaic/MixUp/翻转。
cutmix不受该版本支持且母版p=0；erasing/auto_augment不在检测链执行；Albumentations预先关闭。
独立验证/预测的YOLO letterbox与母版RT-DETR stretch单列，未声称逐样本随机轨迹完全一致。

## 数据、缓存和评测

本地真实YAML通过显式数据根映射到`D:/MyProjects/Crack_RTDETR/datasets/crack_det`。只枚举路径、stat、读取小标签，
读取必要val/test图片头；没有全量图像内容哈希、解码审计、原图族重查、重分或标签修改。

| split | 图片 | 对应标签 | 框 |
|---|---:|---:|---:|
| train | 6048 | 6048 | 45573 |
| val | 1728 | 1728 | 12840 |
| test | 864 | 864 | 6663 |

路径与标签清单指纹吻合公共记录。有效公共val/test COCO逐图路径、尺寸、类别、标签核对后复用，ID和浮点框保留。
当前run标记`LIGHT_CHECKED`，使用独立的路径+标签字节+图片大小轻量身份，不伪造`AUDITED`，也不声称重新证明历史图片内容哈希。
公共准备报告中已有的数据划分限制继续适用；本轮未重复审计或改写它。

`IsolatedDataset`直接读原YOLO标签，必要首次train尺寸缓存来自真实图片头。缓存路径包含上游commit、split、清单和根身份，
写到本run的`cache/`，使用OS局部锁与原子替换。既不调用原生会修复JPEG的`verify_image_label`，也不读/写共享根的`.cache`。
还隔离了官方base即使cache=False仍可能读取/删除的邻接`.npy`路径；原图、原标签及其他框架缓存均只读。
损坏头、EXIF方向、非法/缺失/重复标签或改变的输入都会定位报错，不静默剔除。

首次import前设置独立`YOLO_CONFIG_DIR`和`MPLCONFIGDIR`，显式插入官方路径，检查`__file__`、版本、commit、补丁后源码内容。
关闭自动依赖安装和外部训练集成；私有venv不修改共用环境。模型资产、缓存、环境和大结果均被gitignore排除。

最终导出：FP32、640、batch16、workers0、conf.001、NMS IoU.7、max_det300、augmentFalse、rectFalse、seed42、class-aware NMS。
每批最多加载16张原图，原生`Results`原图坐标仅导出一次，保留完整浮点精度、空预测图、GT稳定ID、相对路径及全部身份。
独立评测直接调用公共`evaluate()`，没有另写AP算法。输出P/R/AP50/AP75/mAP50–95的0–1原值及百分比，P/R是各模型最大平滑F1工作点。
已有正确预测缓存直接CPU重算；参数/GFLOPs提供单类640、融合前后、MACs×2入口，不做共享GPU论文测速。

## 有限验证与未验证项

本地使用Python3.9.25、继承Torch2.7.1+cu118/TorchVision0.22.1+cu118，私有NumPy1.26.4、OpenCV4.10.0、pandas2.2.3、seaborn0.13.2、THOP2.0.9。
v8.3.20的metadata要求Python>=3.8；它的3.10功能探测会在本地打印警告。本地实际CPU检查通过，不把它写成AutoDL环境已验证。
服务器历史环境为Python3.10.13/Torch2.1.2+cu121；当前真实环境仍由bootstrap现场核查。

- 官方下载/哈希、源码导入、展开参数、M单类结构、469/475张量转移均通过；bootstrap完整入口通过。
- 10项定向集成测试通过：缓存/原资产不变、负样本、改标签失败、坏JPEG不修复、加法HSV、真实worker关闭、坐标/空预测/公共接口、best规则、val batch及完整NMS、resume早停/worker恢复。
- 首轮合成batch2、64×64、CPU执行了一次M模型loss和backward，损失与243个梯度张量均有限；没有optimizer更新、正式640训练或GPU占用。
- 5项公共原有准备/评测测试通过，含源码AST不变检查；没有改公共AP实现。
- 5项生命周期测试通过，覆盖command=0/7、tee失败、已有日志/输出和仅本模型同名tmux保护。POSIX SIGTERM组传播测试在Windows跳过，待Linux现场验证。
- CLI/help、Python编译、Bash语法通过。完整小补丁从干净固定commit的重放结果记入交付证据。

未验证：当前AutoDL实际GPU/CUDA/AMP、正式batch16/640显存可行性、正式训练/完整val/test、完整GPU断点恢复、真实tmux终端和POSIX信号行为。
不会用未完成的指标填零或宣称有实验结果。输出目录和已有会话受到保护；单一run锁不锁GPU、不等待YOLOv5m。

运行命令和产物说明见`YOLOV8M_SERVER_COMMANDS.md`；紧凑实测记录见`evidence/yolov8m_delivery_validation.json`。
