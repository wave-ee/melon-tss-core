# Melon Vision TSS

This repository provides a computer-vision workflow for estimating total
soluble solids (TSS) in melon images. It combines segmentation, mask-derived
physical descriptors, multi-region visual features, and regression.

## Components

| File | Purpose |
| --- | --- |
| `melon_tss/segmentation.py` | Directional convolution branches and net clDice |
| `melon_tss/mmseg_integration.py` | DAConv integration in the UNet encoder |
| `melon_tss/phenotypes.py` | Ruler scale, net geometry, and region color features |
| `melon_tss/regions.py` | Aligned RGB, melon, net, skin, and ruler views |
| `melon_tss/fusion.py` | ConvNeXt-Tiny features, region attention, and physical-feature gating |
| `melon_tss/regression.py` | Huber/MSE prediction, flip augmentation, and fruit aggregation |
| `melon_tss/best_protocol.py` | Image-to-TSS inference wiring |
| `configs/method.json` | Method settings used by the workflow |
| `configs/mmseg_model.py` | Segmentation model graph and loss weights |

## Install and run

Python 3.8 or later is supported. The reference environment uses PyTorch
2.4.1, torchvision 0.19.1, and scikit-learn 1.3.2.

```bash
python -m pip install -e .
python -m examples.core_demo
python -m unittest discover -s tests -v
```

The example runs on CPU without images, model weights, or network downloads.
It uses random weights to exercise the data flow and does not report an
accuracy metric.

## Inputs and feature fusion

Segmentation labels are background=0, net=1, skin=2, and ruler=3. Physical
descriptors use an original-resolution OpenCV BGR image and an aligned label
mask. Color spaces are OpenCV uint8 BGR/HSV/Lab. The reference ruler length is
5 cm. Scale-dependent descriptors without a valid scale remain NaN for the
training-fold median imputer; descriptor order is in
`physics_feature_names.json`.

Five RGB views are resized to 320 x 320 for regression. Nonselected pixels are
filled with RGB 0.5 before ImageNet normalization. A shared ConvNeXt-Tiny
encoder produces 768-dimensional features per view. Learned region weights
form an attention average; this is concatenated with the view mean, sample
standard deviation, five weights, and their maximum. The visual projection has
256 units. The 161 physical descriptors are imputed, robust-scaled, reduced to
16 PCA components, and projected to 64 units. A two-way softmax gate scales
these projections by `1 + gate` before the regression head.

Fit imputation, scaling, PCA, and target statistics on training-fold rows.
Use `pipeline.transform(...).astype(np.float32)` for validation and inference
rows. Model outputs are standardized TSS values and are converted back with
the target mean and standard deviation from the corresponding training fold.

## Prediction flow

The regression workflow uses Huber and MSE losses on standardized TSS values.
Each prediction branch averages the original and horizontally flipped region
predictions while keeping physical inputs fixed. The branch outputs are then
averaged in TSS units, followed by a median across views of the same fruit.
Fruit identifiers group views and are not model input features.

## Segmentation configuration

Install a compatible MMSegmentation 1.2.2, MMCV, and MMEngine environment to
use `configs/mmseg_model.py`. Importing `melon_tss.mmseg_integration` registers
`MelonCoreDAUNet` and `MelonCoreNetCLDiceLoss`. Provide dataset, runtime, and
training settings for the surrounding application.

The four DAConv branches use kernel sizes 1x5, 5x1, 3x3, and 3x3. The
`conv_45` and `conv_135` branches are learned 3x3 depthwise filters. The
segmentation graph uses `DiceLoss` together with the net clDice skeleton term;
the main loss weights are CE=1, Dice=2, and net clDice=0.2.

The regression workflow consumes predicted masks from this segmentation stage.

## License and sources

Apache-2.0; see `LICENSE`, `NOTICE`, and `SOURCES.md`. Dependency and dataset
licenses apply separately. This repository contains the core implementation
of a melon vision TSS workflow.
