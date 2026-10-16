# Melon TSS: PADF-EW+FM core implementation

Core implementation of the highest-scoring PADF-EW+FM protocol: DAUNet with
net clDice, mask-derived physical descriptors, five-region ConvNeXt fusion,
two fixed regression branches, and fruit-median aggregation.

This release contains only the final method's core algorithm. It includes model
components, descriptor definitions, fixed method settings and a synthetic
execution example. The complete training/evaluation runner, exact fold
assignments, datasets and trained weights are not included.

## Modules

| File | Core implementation |
| --- | --- |
| `melon_tss/segmentation.py` | Four-branch residual DAConv and net-only clDice |
| `melon_tss/mmseg_integration.py` | DAConv after each UNet encoder stage in the final segmentation graph |
| `melon_tss/phenotypes.py` | Ruler scale, net geometry, region color statistics; 161 ordered inputs |
| `melon_tss/regions.py` | Aligned RGB, melon, net, skin and ruler views with shared augmentation |
| `melon_tss/fusion.py` | Shared ConvNeXt-Tiny, region attention and visual/physical gating |
| `melon_tss/regression.py` | Fixed Huber/MSE branches, flip TTA, equal fusion and fruit aggregation |
| `melon_tss/best_protocol.py` | One-call wiring of the final image-to-TSS inference path |
| `configs/method.json` | Fixed PADF-EW+FM settings |
| `configs/mmseg_model.py` | Segmentation model graph and loss weights |

## Run the core example

Python 3.8 or later; tested with PyTorch 2.4.1, torchvision 0.19.1 and
scikit-learn 1.3.2. From the repository root:

```bash
python -m pip install -e .
python -m examples.core_demo
python -m unittest discover -s tests -v
```

The example runs on CPU without images, weights or network downloads. It uses
random weights and does not report an accuracy.

## Inputs and feature fusion

Segmentation labels are background=0, net=1, skin=2 and ruler=3. Physical
descriptors use an original-resolution OpenCV BGR image and an aligned predicted
label mask. Color spaces are OpenCV uint8 BGR/HSV/Lab. The reference ruler length
is 5 cm. Missing scale-dependent descriptors remain NaN for the training-fold
median imputer; descriptor order is in `physics_feature_names.json`.

Five RGB views are resized to 320 x 320 for regression. Nonselected pixels are
filled with RGB 0.5 before ImageNet normalization. A shared ConvNeXt-Tiny produces
768-dimensional features per view. Learned region weights form an attention
average; this is concatenated with the view mean, sample standard deviation,
five weights and their maximum. The visual projection has 256 units, while
161 physical descriptors are imputed, robust-scaled and reduced to 16 PCA
components, then projected to 64 units. A two-way softmax gate scales these
projections by `1 + gate` before the regression head.

Fit imputation/scaling/PCA and target mean/std using only training-fold rows.
Use `pipeline.transform(...).astype(np.float32)` for other rows. Model outputs
are standardized TSS and must be converted back using the same fold's target
mean/std.

## Regression prediction and reference protocol

The final protocol always trains both branches: Huber uses
`smooth_l1_loss(beta=1.0)` on standardized TSS, and the second uses MSE. Each
branch averages the original and horizontally flipped region predictions;
physical inputs remain fixed. The two outputs are then averaged equally in TSS
units, followed by the median of all available views of each fruit. Fruit
identifiers are used only to group views. `best_protocol.py` wires these steps
without exposing alternative configurations.

The saved reference experiment used 250 images from 146 fruits, with a fixed
target-stratified allocation of complete fruit groups to five folds (50 images
per fold). Validation RMSE selected checkpoints from epoch 1 (an improvement
exceeding 1e-6, or an `isclose` tie with higher R2). These validation predictions
reflect the checkpoint-selection protocol. The reference pooled, image-weighted
out-of-fold R2 is
0.7187329120: each fruit median is assigned back to that fruit's images.
Giving each fruit one vote yields R2=0.6939799550. Pooled R2 and the mean of five
fold R2 values are different summaries. Exact split assignments and the full
runner are not included in this core release.

## Segmentation core

Install a compatible MMSegmentation 1.2.2 / MMCV / MMEngine environment to use
`configs/mmseg_model.py`. Importing `melon_tss.mmseg_integration` registers
`MelonCoreDAUNet` and `MelonCoreNetCLDiceLoss`. The graph specifies the encoder,
FCN heads, preprocessing and loss weights; provide your dataset, runtime and
training schedule separately.

The four DAConv branches retain the saved kernel sizes 1x5, 5x1, 3x3 and 3x3.
The branches named `conv_45` and `conv_135` are learned 3x3 depthwise filters
without fixed diagonal masks. The saved `TopoSegLoss` is a `DiceLoss` alias,
so the released graph names it `DiceLoss`; net clDice supplies the skeleton
term. The skeleton uses 3x3 square erosion and additive accumulation exactly
as in the saved code. The main loss weights are CE=1, Dice=2 and net clDice=0.2.

The regression protocol consumes full-coverage predicted masks from this
segmentation stage. Ground-truth masks are not substituted for predicted masks.

## License and sources

Apache-2.0; see `LICENSE`, `NOTICE` and `SOURCES.md`. Dependency and dataset
licenses apply separately. For a paper's availability statement, describe this
repository as a release of the **PADF-EW+FM core implementation**.
