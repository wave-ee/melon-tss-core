"""Five aligned views: RGB, melon, net, skin, ruler."""
from __future__ import annotations
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as TF

def prepare_pair(img: Image.Image, mask: Image.Image, image_size: int = 320, training: bool = False) -> tuple[Image.Image, Image.Image]:
    size = [image_size, image_size]
    img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)
    if not training:
        return img, mask
    if random.random() < 0.5:
        img, mask = TF.hflip(img), TF.hflip(mask)
    if random.random() < 0.25:
        img, mask = TF.vflip(img), TF.vflip(mask)
    if random.random() < 0.35:
        angle = random.uniform(-12, 12)
        translate = [int(random.uniform(-0.04, 0.04) * image_size), int(random.uniform(-0.04, 0.04) * image_size)]
        scale = random.uniform(0.92, 1.08)
        shear = [random.uniform(-4, 4), random.uniform(-4, 4)]
        img = TF.affine(img, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.affine(mask, angle, translate, scale, shear, interpolation=TF.InterpolationMode.NEAREST, fill=0)
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.82, 1.18))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.82, 1.18))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.86, 1.14))
    return img, mask


def build_region_views(image, mask, image_size=320, training=False):
    """PIL RGB image + PIL grayscale predicted labels -> [5,3,H,W]."""
    image, mask = prepare_pair(image.convert("RGB"), mask.convert("L"), image_size, training)
    rgb = TF.to_tensor(image)
    labels = torch.as_tensor(np.array(mask), dtype=torch.long)
    masks = [((labels == 1) | (labels == 2)), labels == 1, labels == 2, labels == 3]
    neutral = torch.full_like(rgb, 0.5)
    views = [rgb]
    for selected in masks:
        selected = selected.float().unsqueeze(0)
        views.append(rgb * selected + neutral * (1 - selected))
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=rgb.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=rgb.dtype).view(3, 1, 1)
    return torch.stack([(view - mean) / std for view in views])
