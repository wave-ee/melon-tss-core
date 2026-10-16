# 哈密瓜视觉 TSS 方案

本仓库提供一个用于哈密瓜图像糖度（TSS，总可溶性固形物）估计的视觉方案。
方案结合图像分割、掩膜物理特征、多区域视觉特征和回归预测。

## 项目内容

| 文件 | 用途 |
| --- | --- |
| `melon_tss/segmentation.py` | 方向卷积分支与网纹 clDice |
| `melon_tss/mmseg_integration.py` | DAConv 在 UNet 编码器中的接入 |
| `melon_tss/phenotypes.py` | 标尺尺度、网纹几何和区域颜色特征 |
| `melon_tss/regions.py` | 原图、果实、网纹、果皮和标尺视图 |
| `melon_tss/fusion.py` | ConvNeXt-Tiny 特征、区域注意力和物理特征门控 |
| `melon_tss/regression.py` | Huber/MSE 预测、翻转增强和果实聚合 |
| `melon_tss/best_protocol.py` | 图像到 TSS 的推理流程 |
| `configs/method.json` | 方案使用的参数设置 |
| `configs/mmseg_model.py` | 分割模型图和损失权重 |

## 安装与运行

支持 Python 3.8 及以上版本。参考环境使用 PyTorch 2.4.1、torchvision 0.19.1
和 scikit-learn 1.3.2。

```bash
python -m pip install -e .
python -m examples.core_demo
python -m unittest discover -s tests -v
```

示例可以在 CPU 上运行，不需要图像、模型权重或网络下载。示例使用随机权重
验证数据流，不输出精度指标。

## 输入与特征融合

分割标签为 background=0、net=1、skin=2、ruler=3。物理特征使用原始分辨率的
OpenCV BGR 图像和对齐的标签掩膜。颜色空间采用 OpenCV uint8 BGR/HSV/Lab，
参考标尺长度为 5 cm。缺少有效尺度时，相关特征保留为 NaN，并由训练折的中位数
填补器处理；特征顺序记录在 `physics_feature_names.json` 中。

五种 RGB 视图统一调整为 320 x 320，用于回归。非选中区域填充 RGB 0.5，之后进行
ImageNet 归一化。共享 ConvNeXt-Tiny 编码器为每个视图生成 768 维特征。区域权重
形成注意力平均特征，并与视图均值、样本标准差、五个区域权重及其最大值拼接。
视觉分支映射到 256 维；161 个物理特征经过填补、稳健缩放和 PCA 降至 16 维，
再映射到 64 维。双路 softmax 门控以 `1 + gate` 缩放两个投影后送入回归头。

在训练折数据上拟合填补、缩放、PCA 和目标值统计量。对验证和推理数据使用
`pipeline.transform(...).astype(np.float32)`。模型输出为标准化 TSS，使用对应训练折
的目标均值和标准差还原到原始单位。

## 预测流程

回归流程在标准化 TSS 上使用 Huber 和 MSE 损失。每个预测分支分别计算原视图和
水平翻转视图的结果并取平均，物理特征保持不变。两个分支的结果在 TSS 单位下取
平均，再对同一果实的多张视图取中位数。果实编号用于聚合同一果实的视图，不作为
模型输入特征。

## 分割配置

使用 `configs/mmseg_model.py` 需要兼容的 MMSegmentation 1.2.2、MMCV 和 MMEngine
环境。导入 `melon_tss.mmseg_integration` 会注册 `MelonCoreDAUNet` 和
`MelonCoreNetCLDiceLoss`。数据集、运行环境和训练设置由实际应用配置。

四个 DAConv 分支使用 1x5、5x1、3x3 和 3x3 卷积核；`conv_45` 与 `conv_135` 使用
可学习的 3x3 深度卷积。分割图使用 `DiceLoss` 和网纹 clDice 骨架项，主要损失权重
为 CE=1、Dice=2、net clDice=0.2。

回归流程使用该分割阶段生成的预测掩膜。

## 许可与来源

项目采用 Apache-2.0 许可证，详见 `LICENSE`、`NOTICE` 和 `SOURCES.md`。依赖和数据集
分别适用其自身许可证。本仓库提供哈密瓜视觉 TSS 方案的核心实现。
