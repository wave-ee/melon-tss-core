"""Shared ConvNeXt encoder, region attention, and visual/physical gating."""
import torch
from torch import nn
from torchvision import models as tvm


class FeatureBackbone(nn.Module):
    """ConvNeXt-Tiny features + average pooling (without classifier norm)."""
    def __init__(self, name="convnext_tiny", pretrained=False):
        super().__init__()
        if name != "convnext_tiny":
            raise ValueError("The melon TSS workflow uses a ConvNeXt-Tiny backbone")
        self.name = name
        model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = int(model.classifier[-1].in_features)
        self.kind = "cnn_features"

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class PhysicsAssistedRegionRegressor(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, dropout: float, physics_dim: int):
        super().__init__()
        self.encoder = FeatureBackbone(backbone, pretrained=pretrained)
        d = self.encoder.feature_dim
        gate_hidden = max(64, d // 4)
        self.gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))
        self.visual = nn.Sequential(nn.LayerNorm(d * 3 + 6), nn.Linear(d * 3 + 6, 256), nn.GELU(), nn.Dropout(dropout))
        self.physical = nn.Sequential(nn.LayerNorm(physics_dim), nn.Linear(physics_dim, 64), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.fusion_gate = nn.Sequential(nn.Linear(320, 64), nn.GELU(), nn.Linear(64, 2))
        self.head = nn.Sequential(nn.Linear(320, 128), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 1))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, regions: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        b, r, c, h, w = regions.shape
        feats = self.encoder(regions.reshape(b * r, c, h, w)).reshape(b, r, -1)
        weights = torch.softmax(self.gate(feats).squeeze(-1), dim=1)
        visual_raw = torch.cat([torch.sum(feats * weights.unsqueeze(-1), dim=1), feats.mean(dim=1), feats.std(dim=1), weights, weights.max(dim=1, keepdim=True).values], dim=1)
        visual = self.visual(visual_raw)
        physical = self.physical(physics)
        joined = torch.cat([visual, physical], dim=1)
        gate = torch.softmax(self.fusion_gate(joined), dim=1)
        gated = torch.cat([visual * (1.0 + gate[:, :1]), physical * (1.0 + gate[:, 1:2])], dim=1)
        return self.head(gated).squeeze(1)
