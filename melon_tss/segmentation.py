# Copyright (c) Open-MMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extracted DAConv and net-only clDice; see SOURCES.md for attribution."""
import torch
from torch import nn
from torch.nn import functional as F


class DirectionAwareConvBlock(nn.Module):
    """Directional depthwise filters used by the local DAUNet checkpoint.

    The block applies four lightweight directional filters to each feature map
    and fuses them with a 1x1 convolution. The parameter names intentionally
    match the external checkpoint keys (`conv_0`, `conv_90`, `conv_45`,
    `conv_135`, `fuse`).
    """

    def __init__(self, channels):
        super().__init__()
        self.conv_0 = nn.Conv2d(
            channels, channels, kernel_size=(1, 5), padding=(0, 2),
            groups=channels)
        self.conv_90 = nn.Conv2d(
            channels, channels, kernel_size=(5, 1), padding=(2, 0),
            groups=channels)
        self.conv_45 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_135 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        directional = torch.cat(
            [self.conv_0(x), self.conv_90(x), self.conv_45(x),
             self.conv_135(x)],
            dim=1)
        return x + self.fuse(directional)

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """软形态学腐蚀：使用 3×3 最小值池化。"""
    if len(img.shape) == 4:
        # (B, C, H, W) → min_pool2d
        p = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    elif len(img.shape) == 5:
        p = -F.max_pool3d(-img, kernel_size=3, stride=1, padding=1)
    else:
        raise ValueError(f'Unsupported tensor dim: {len(img.shape)}')
    return p

def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """软形态学膨胀：使用 3×3 最大值池化。"""
    if len(img.shape) == 4:
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    elif len(img.shape) == 5:
        return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)
    else:
        raise ValueError(f'Unsupported tensor dim: {len(img.shape)}')

def soft_open(img: torch.Tensor) -> torch.Tensor:
    """软形态学开运算：先腐蚀再膨胀。"""
    return soft_dilate(soft_erode(img))

def soft_skeleton(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """软骨架提取（可微分）。

    通过迭代形态学减薄提取骨架：
    skeleton = Σ_i ReLU(erode^i(img) - open(erode^i(img)))

    Args:
        img: 输入掩码 (B, C, H, W)，值域 [0, 1]
        num_iter: 迭代次数，控制骨架细度
    """
    img_eroded = img.clone()
    skeleton = F.relu(img_eroded - soft_open(img_eroded))

    for _ in range(num_iter):
        img_eroded = soft_erode(img_eroded)
        delta = F.relu(img_eroded - soft_open(img_eroded))
        skeleton = skeleton + delta

    return skeleton

class NetCLDiceLoss(nn.Module):
    """clDice focused on the melon net class.

    The net label is a thin, connected structure. This loss only compares the
    predicted probability map of one target class with the binary mask of that
    class, so it strengthens net continuity without letting background dominate
    the skeleton term.
    """

    def __init__(self,
                 target_class: int = 1,
                 num_iter: int = 8,
                 smooth: float = 1.0,
                 loss_weight: float = 1.0,
                 ignore_index: int = 255,
                 loss_name: str = 'loss_net_cldice'):
        super().__init__()
        self.target_class = target_class
        self.num_iter = num_iter
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index: int = 255,
                **kwargs) -> torch.Tensor:
        pred_soft = pred.softmax(dim=1)[:, self.target_class:self.target_class + 1]
        valid = (target != self.ignore_index)
        target_bin = ((target == self.target_class) & valid).float().unsqueeze(1)
        pred_soft = pred_soft * valid.float().unsqueeze(1)

        skel_pred = soft_skeleton(pred_soft, self.num_iter)
        skel_target = soft_skeleton(target_bin, self.num_iter)

        tprec = ((skel_pred * target_bin).sum(dim=(2, 3)) + self.smooth) / (
            skel_pred.sum(dim=(2, 3)) + self.smooth)
        tsens = ((skel_target * pred_soft).sum(dim=(2, 3)) + self.smooth) / (
            skel_target.sum(dim=(2, 3)) + self.smooth)
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
        return self.loss_weight * (1.0 - cl_dice.mean())

    @property
    def loss_name(self):
        return self._loss_name
