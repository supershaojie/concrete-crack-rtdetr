# 第四章八模型检测对比与 Grad-CAM++

代码分支：`codex/ch4-visualization-eight-models-v1`。默认同一组 5 张样本为列、模型为行：Ground truth/Original 前置，Faster R-CNN 第一模型行，ours 最后一行。显示阈值全组统一 0.25，JET 全组统一、透明度 0.45，整图先计算再共同裁剪；本轮不接入 D-FINE-M，不运行训练。

本地实现和资料校验与服务器模型验收是两回事。当前本地能验证交接身份、四组完整缓存、统计/几何/绘图/打包；八模型正式恢复、与缓存对齐、验证集梯度和论文案例视觉审阅必须由真实服务器运行产生记录。没有成功 CAM 的单元在草图显示灰色 `CAM PENDING / FAILED`，图底标 DRAFT；不会用原图或其他响应冒充热图。

## 直接复制到服务器

仅 `CH4_INPUT` 需要填写当前交接 ZIP 的真实位置。可以传本次外层 implementation ZIP，也可以传其中 final handoff ZIP 或已解压的交接根目录。所有八组权重、环境、源码和正式缓存路径自动从登记表读取。

```bash
export CH4_INPUT='/请替换为当前实际路径/Chapter4_Visualization_Codex_Implementation_20261015.zip'
export CH4_CODE=/root/autodl-tmp/projects/Crack_RTDETR-ch4-visualization-eight
export CH4_OUT=/root/autodl-tmp/ch4_visualization/eight_models_v1
export CH4_BRANCH=codex/ch4-visualization-eight-models-v1

if [[ -d "$CH4_CODE/.git" ]]; then
  git -C "$CH4_CODE" fetch origin "$CH4_BRANCH"
  git -C "$CH4_CODE" switch "$CH4_BRANCH"
  git -C "$CH4_CODE" pull --ff-only origin "$CH4_BRANCH"
else
  git clone --single-branch --branch "$CH4_BRANCH" https://github.com/supershaojie/concrete-crack-rtdetr.git "$CH4_CODE"
fi
cd "$CH4_CODE"
git status --short --branch
git rev-parse HEAD
bash scripts/autodl_chapter4_visualization.sh setup
bash scripts/autodl_chapter4_visualization.sh prepare --handoff "$CH4_INPUT" --out "$CH4_OUT"
bash scripts/autodl_chapter4_visualization.sh preflight --out "$CH4_OUT" --device cpu
```

`setup` 只创建独立 `.venv-ch4` 并安装排版/分析依赖；不向八个训练环境安装、升级任何包。模型分别使用登记的 Python/vendor；v13 使用原兼容 wrapper 和 native_fp32/ArithmeticAudit。自有 Grad-CAM++ 实现版本冻结在 `cam.py`，不需要在训练环境安装 grad-cam。

CPU preflight 先核 ours 权重和母版，再逐个恢复模型，核对原正式 HEAD、源码内容、权重 SHA、实际导入、选中 model/EMA 字段、保存 dtype、LIF BN/forward、共同 test 图片大小/标签 SHA。复用交接只读 collector，资料写入本次输出。历史 epoch=-1、旧 argv 缺项继续记 pending。CPU preflight 不执行 CAM，不要求空 GPU，不操作已有训练。

先做三张固定真实 test 图片的数值对齐，再执行全流程：

```bash
# 正式缓存只读接入；仅缺失缓存才在本次目录导出。
bash scripts/autodl_chapter4_visualization.sh predictions --out "$CH4_OUT" --device cuda:0
bash scripts/autodl_chapter4_visualization.sh smoke --out "$CH4_OUT" --device cuda:0

# 自动挑选约40张候选、提出5张待审阅样本、验证集取层、40个CAM单元、两张草图和轻量ZIP。
bash scripts/autodl_chapter4_visualization.sh tmux full --out "$CH4_OUT" --device cuda:0
cat "$CH4_OUT/tmux_session.txt"
tmux attach -t "$(cat "$CH4_OUT/tmux_session.txt")"
```

`cuda:0` 可统一换成 `cuda:1` 等实际设备。模型顺次运行，CAM batch=1；没有“空 GPU 才能启动”门槛。tmux 若已存在同名会话会创建数字后缀，保留原会话。按 `Ctrl-b` 再按 `d` 退出界面；断 SSH、关电脑后任务仍继续。

完整启动也可一次执行（`full` 含 preflight、predictions 和 smoke）：

```bash
bash scripts/autodl_chapter4_visualization.sh tmux full --handoff "$CH4_INPUT" --out "$CH4_OUT" --device cuda:0
```

## 进度、失败与续跑

```bash
tail -f "$CH4_OUT/logs/tmux.log"
watch -n 3 cat "$CH4_OUT/progress.json"
bash scripts/autodl_chapter4_visualization.sh status --out "$CH4_OUT"
find "$CH4_OUT/reports" -maxdepth 1 -type f -name '*failure*.json' -print
cat "$CH4_OUT/reports/preflight.json"
cat "$CH4_OUT/reports/prediction_sources.json"
cat "$CH4_OUT/reports/smoke.json"
cat "$CH4_OUT/layer_protocol.json"
cat "$CH4_OUT/cam_index.json"
cat "$CH4_OUT/exit_code.txt"

# 原命令续跑；复用哈希一致的完整预测和成功CAM，不重做已有全test预测。
bash scripts/autodl_chapter4_visualization.sh tmux full --out "$CH4_OUT" --device cuda:0
```

预测阶段是 8 模型进度，各子进程另有逐图进度；CAM 是 8×5=40 单元并分别报告。失败信息包括具体模型/样本、traceback、原权重与源码身份。smoke 或验证集失败会阻止 test CAM；已生成草图、候选和失败证据仍打包。CAM 单元失败继续收集其他单元，最终以失败状态退出，草图明确标记，不能作为论文终稿。

已登记路径变更时，在本次输出之外另建 JSON，仅覆盖实际核实的路径，例如：

```json
{"lcbr_detr":{"test_predictions":"/实际核实的新目录/predictions_gt.jsonl.gz","test_metrics":"/实际核实的新目录/metrics.json"}}
```

随后加 `--overrides /实际路径/server_paths.json`；原登记表不改写，覆盖来源进入新 registry。禁止用路径覆盖换权重身份。ours 缓存未知/缺失时，使用原 `c19_lif_v1_results.evaluate` 在本次目录重导出；其标签/npy 缓存 IO 重定向到本任务 runtime，计算、300查询、CBR最终框、原验证器不改。原 val 回执缺失时先在新目录执行同入口 val。导出与正式缓存复用分别标记，原正式结果保持不变。

## 候选与视觉审阅

全组缓存必须覆盖同一 864 图，且采用文件名 canonical key 连接 GT。候选使用显示点 IoU .50/.75 的同类一对一 TP/FP/FN，保留 GT/预测索引、IoU、score、低于显示阈值与定位/无候选的区别；这不替换原 AP。会同时比较 C2、v26、v13 和其他模型，兼顾 ours 自身质量、优势类型、图像描述差异，避免同一原片的增强副本重复入选。自动特征只是场景代理，不能把“小框”自动写成“低对比度/遮挡”。

自动初选还要求明确优势证据：ours匹配IoU至少.55、得分至少比显示阈值高.05，并与强对照有至少.05的可见定位差距或明确的显示漏检；另一入口是强对照高于显示阈值.05的无GT重叠误检候选减少。FP细分重复匹配、定位不足和无GT重叠，不能把所有FP自动称为背景。默认解释至多3个明确优势GT；误检优势场景则取至多3个ours高IoU真阳性，均在CAM之前固定。数值边界不能代替人工场景确认，初选仍保留待审阅状态。

查看 `previews/contact_*.png`、候选逐图八模型矩阵、`candidates/candidates.json` 和 `statistics/per_image.csv/json`；完整八模型未过滤预测保存在 `candidates/predictions/`。`cases.proposed.json` 永远是待视觉审阅。只有真实浏览原图、GT 与八模型预测并填写场景与依据后才可生成 reviewed 文件。

可以编辑候选初选为 4–6 张，所有列顺序以 cases 文件为准。每例的 `image`、`target_gt_ids`、`advantage_type`、`comparison_statistics`、`view_region`、`display_threshold`、`reason` 都保留。检测和热图默认共同使用该 cases 文件；裁剪区域同列全模型相同。

```bash
# 先复制并编辑所需样本。保留 proposed 状态，直到真正审阅。
cp "$CH4_OUT/cases/cases.proposed.json" "$CH4_OUT/cases/cases.custom.json"
# 使用编辑器填写 cases 和 review.template.json：reviewer、gt_confirmed、scene_tags、notes。
# preview_sha256 保持真实已查看预览的哈希；新增样本先重新运行 candidates 生成预览。
bash scripts/autodl_chapter4_visualization.sh review --out "$CH4_OUT" \
  --cases "$CH4_OUT/cases/cases.custom.json" --review-record "$CH4_OUT/cases/review.template.json"

# 只补新案例 CAM、重新排版；全test预测缓存复用。
bash scripts/autodl_chapter4_visualization.sh cam --out "$CH4_OUT" --device cuda:0 --cases "$CH4_OUT/cases/cases.reviewed.json"
bash scripts/autodl_chapter4_visualization.sh render --out "$CH4_OUT" --cases "$CH4_OUT/cases/cases.reviewed.json"
bash scripts/autodl_chapter4_visualization.sh pack --out "$CH4_OUT"
```

没有图片/只有数值排序时不能执行“已审阅”签认。本轮本地四组缓存无法完成八模型优势筛选，不能凭四模型排序伪造最终5例。

## CAM 协议和数值边界

采用 [Grad-CAM++ 论文](https://arxiv.org/abs/1710.11063) 的常用梯度平方/立方权重实现，与 [pytorch-grad-cam 的 GradCAMPlusPlus](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py) 同类公式。版本 `ch4-gradcampp-gradient-powers-1.0` 与实际源码 SHA 都入缓存键；这里明确使用一阶梯度的平方/立方形式，不声称显式计算完整 Hessian。它对真实 native forward 的裂缝得分求导；GT只做实例对应，不做像素掩膜或定位损失。

实例对应全组采用同一优先序：先一对一匹配显示检测，再匹配低于共同显示阈值的正式检测，最后才为尚未对应的GT寻找原始候选；最终检测通过框/得分映射回可微原候选。这样不会因为一个近GT但得分极低的背景anchor几何更近，就解释错对象。每层对应来源、IoU、原索引及“未形成展示检测”都记录。共同显示阈值进入对应缓存键；无合理对应或无有效梯度仍明确失败。

所有模型固定使用分类路径中 stride 8/16/32 三尺度二维检测特征，逐层 Grad-CAM++ 正响应 min-max、包括未使用尺度为零的等权平均，再全图 min-max；全组一套规则。Faster 取 FPN P3/P4/P5 lateral+top-down 融合后、输出卷积前的特征与 ROI foreground softmax；这些特征同时连到更细的P2，避免细裂缝ROI分配到P2时捕获输出专用分支造成断梯度。v5 取检测卷积前融合特征与 objectness×class，其他 YOLO 取分类读出前特征；v26 取 detach 后真实 one-to-one 分类分支，记录 native top-k 返回的原 anchor 索引；RT 取 decoder input projection 并记录普通 query，ours 用 CBR refined final boxes。未使用/零梯度尺度逐个记录；三尺度全无有效正响应、NaN、断梯度、错误形状均失败。

取层结构规则在 test 选择前固定，再用原 `val.txt` 中按文件名排序的3个非空案例、每例最大GT实例验证，并要求至少一张非正方形图片；不会按 test CAM 美观程度挑层。自动定位 `val.txt` 的依据是固定导出器在已登记 `test.txt` 同目录写出 val/test 清单。缺失则明确 pending，用 `--validation-list /已核实的绝对路径/val.txt` 提供原正式验证清单。

Faster 用原版本 torchvision 的 FrozenBN eps=0、原 FPN/ROI/RPN 参数和 strict best['model'] 重建最终原生图，避免重新下载 COCO 初始化、写原缓存。v5 保持 attempt_load(fuse=True)，保存FP16选中EMA转FP32；v8/v11/v13/v26 原生构造后 strict-load FP32 sidecar，再原生融合。C2取实际model字段；ours记录实际model/EMA及保存dtype、保留 LIF BN/forward 和原 CBR detach/有符号数学。CAM前后核对同图检测，检查参数版本与BN缓冲区不变，无 optimizer.step。

smoke 的输入坐标绝对容差固定 0.002 像素，按原图/input尺寸比例换算；坐标 rtol=1e-6，score atol=2e-5。C2另加入原 xywh 三位小数到角点的最大量化误差约 .001，以及score五位小数的 .000005。使用完整二分匹配处理顺序/重复框，报告最大误差；失败不自动放宽。C2原排序/mask历史语义保留，不覆盖原AP；缓存内全部300分数通过阈值的边界单独记录。

Grad-CAM++ 成功元数据包含权重SHA、参数字段、正式commit与当前工具commit、真实导入、输入几何、目标/GT/候选索引、目标层与BCHW、梯度范围、原生分辨率、归一化、原图SHA、前后检测一致性。保存低分辨率 npz；letterbox先放大到输入、去真实整数pad、再回原图，RT按拉伸还原。跨模型红色不是绝对置信度。类别CAM本身不能证明 CBR 的定位贡献，后者结合精修框/IoU/消融解释。

## 输出、打包与后续模型接口

`figures/` 包含两张总图 PNG/PDF/SVG、真实像素与物理尺寸、字体身份和行列清单。默认 7.2 英寸宽、最大9.5英寸高、450dpi，8.5pt水平行名，白底三线，无密集网格；PDF/SVG框线、文字为矢量。原图比例保留，各单元用共同画布留白。GT青、预测红，原值以两位小数统一显示，低阈值原值仍完整保留。热图不叠检测框。

`selected/` 保存选中原图、完整GT与八模型原生预测JSON，以及 `full_panels/` 的GT/八模型全视野检测图和 `heatmaps/` 的全视野叠加图；原生CAM及元数据按缓存键位于 `cam/`。`pack` 包含总图、候选预览/理由、cases、所选原图/GT/预测、CAM、全test逐图统计、registry/配置/身份、关键日志和本复现文档；权重/环境/全数据/全test预测大缓存留服务器，用路径和SHA定位。包内可再生全视野审阅图/候选预览先缩至最长1600像素，并在MANIFEST记录导出策略、服务器原文件路径和SHA；选中原图、最终总图、预测JSON和原生CAM保留原字节。必要时仅继续压缩这些审阅图。

```bash
bash scripts/autodl_chapter4_visualization.sh pack --out "$CH4_OUT"
cat "$CH4_OUT/pack_receipt.json"
stat -c '%n %s bytes' "$CH4_OUT/chapter4_review.zip"
test "$(stat -c %s "$CH4_OUT/chapter4_review.zip")" -lt 100000000
unzip -t "$CH4_OUT/chapter4_review.zip"
```

打包校验实际字节数严格 `<100000000`，逐项SHA与ZIP CRC；优先压缩可再生候选预览，最终图像分辨率/模型/案例/CAM/失败记录不删。仍超限则报告 `pack_sizes.json` 并保留完整服务器产物，不伪称成功。一个ZIP一次带回完整审阅材料。

将来另有 D-FINE-M 正式交接后，提供新增 record 的真实 `id/model/run_id/checkpoint.sha256/paths/code_commit/adapter_entry`。`adapter_entry` 是固定原源码/环境中的 `module:create_native(request)`，返回与 `worker.Native` 相同的 `identity/torch/forward/cam/unchanged/mid` 接口；forward输出统一真实预测，cam负责本协议的可微原生目标与元数据。登记本身不加载模型：

```bash
bash scripts/autodl_chapter4_visualization.sh add-model --out "$CH4_OUT" --model-record /后续正式交接/dfine_visualization_record.json
# 后续才执行：按登记的独立环境验证，固定 cases 不变；仅新增模型需要新计算。
bash scripts/autodl_chapter4_visualization.sh preflight --out "$CH4_OUT" --model-id dfine_m
bash scripts/autodl_chapter4_visualization.sh predictions --out "$CH4_OUT" --device cuda:0
bash scripts/autodl_chapter4_visualization.sh smoke --out "$CH4_OUT" --device cuda:0 --model-id dfine_m
bash scripts/autodl_chapter4_visualization.sh validate-layers --out "$CH4_OUT" --device cuda:0 --model-id dfine_m
bash scripts/autodl_chapter4_visualization.sh cam --out "$CH4_OUT" --device cuda:0 --model-id dfine_m --cases "$CH4_OUT/cases/cases.reviewed.json"
bash scripts/autodl_chapter4_visualization.sh render --out "$CH4_OUT" --cases "$CH4_OUT/cases/cases.reviewed.json"
bash scripts/autodl_chapter4_visualization.sh pack --out "$CH4_OUT"
```

新增模型明确插在 ours 前，Faster始终第一、ours最后；本轮不提供或运行未经正式交接的 D-FINE 权重/层号/适配器。缓存键隔离模型/权重/源码/预处理/解释目标/样本/目标层协议，改样本只补新案例，排版修改复用原生CAM。

## 本地验收记录

详见 `docs/chapter4_visualization_validation.json`：17项CPU检查通过；960项交接文件SHA通过；本地864图的图片大小与标签SHA全部一致；四组真实完整缓存共622385个框已适配并生成6912行逐图统计。使用真实图片和这四组缓存检查了3240×4275像素、7.2×9.5英寸、450dpi的9×5草图，其余20格明确标示待计算。这些固定图片只用于布局检查，未作为优势案例初选。tmux脚本通过bash语法、入口和带空格/单引号参数、已有会话后缀的模拟检查，没有实际启动服务器tmux。

Grad-CAM++公式、零响应拒绝和缓存重试的本地单元检查，不能当作真实模型反传证明。八模型实际恢复、原生/缓存对齐、验证集有效梯度、40个真实CAM、最终场景审阅和完整服务器结果ZIP字节数仍待上述服务器流程真实产生。检查命令：

```bash
export CH4_TEST_HANDOFF="$("$CH4_CODE/.venv-ch4/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["handoff"])' "$CH4_OUT/run.json")"
"$CH4_CODE/.venv-ch4/bin/python" -m unittest discover -s benchmarks/visualization/chapter4/tests -v
```
