# 相关工作与本地模块包审计

本实验不把论文结果、模块类名或局部响应当成检测有效性证明。

- [Pixel Difference Networks for Efficient Edge Detection, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Su_Pixel_Difference_Networks_for_Efficient_Edge_Detection_ICCV_2021_paper.html)：执行合同给出的像素差分相关工作。本次访问该 CVF 页面返回 403，未补充未经核实的论文细节。
- [Strip R-CNN, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38217)：已核对会议官网，发表日期为 2026-03-14，卷 40(15)，论文包含大型条带骨干与检测头。TCR 的固定五点均值并非其完整 StripNet/Strip Head。
- [Rotation Invariant and Symmetry Aware Pixel Difference Network for Remote Sensing Object Detection, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhan_Rotation_Invariant_and_Symmetry_Aware_Pixel_Difference_Network_for_Remote_CVPR_2026_paper.html)：执行合同给出的 RIS-PiDiNet 相关工作。本次访问该 CVF 页面返回 403；不把 TCR 宣称为其完整几何机制，也不宣称严格旋转不变。

找到本地包 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA256 为 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`。

解包目录的 `ultralytics/nn/extra_modules/block.py` SHA256 为 `e3e3617c89bd7caf350d2601740f2e399847a16a3a659d7b3e40010b88498cd2`，与 ZIP 内同一条目的字节哈希一致。`Strip_Attention` 定义位于 10463 行，forward 在 10471 行，10475 行确为 `# x = self.spatial_gating_unit(x)`。实际执行的是 proj_1→GELU→proj_2→shortcut；仅在构造函数创建 Strip_Block，不能据此称条带算子已进入 forward。该文件只读审计，没有导入或修改它。

TCR 的区别是切向聚合后保留中心相对两侧同向差分的符号；它既不是用 abs/平方后的边缘能量，也不是简单中心减两侧平均。与已有研究的差异尚不构成充分新颖性论证。
