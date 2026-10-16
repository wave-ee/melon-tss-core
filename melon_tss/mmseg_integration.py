# Copyright (c) Open-MMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional MMSegmentation 1.2.2 integration; not needed for the core demo."""
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmseg.models.backbones.unet import UNet
from mmseg.registry import MODELS
from .segmentation import DirectionAwareConvBlock, NetCLDiceLoss

@MODELS.register_module(name="MelonCoreDAUNet")
class DAUNet(UNet):
    """Apply DAConv after each upstream UNet encoder stage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.ModuleList(list(self.decoder)[::-1])
        channels = [self.base_channels * 2**i for i in range(self.num_stages)]
        self.dac_blocks = nn.ModuleList(
            [DirectionAwareConvBlock(c) for c in channels])

    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            x = self.dac_blocks[i](x)
            enc_outs.append(x)

        dec_outs = [x]
        for i, dec in enumerate(self.decoder):
            skip_idx = self.num_stages - 2 - i
            x = dec(enc_outs[skip_idx], x)
            dec_outs.append(x)

        return dec_outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'

MODELS.register_module(name="MelonCoreNetCLDiceLoss", module=NetCLDiceLoss)
