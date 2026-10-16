"""Run: python -m examples.core_demo (no weights or downloads)."""
import numpy as np
import torch
from PIL import Image
from melon_tss.segmentation import DirectionAwareConvBlock, NetCLDiceLoss
from melon_tss.phenotypes import feature_vector
from melon_tss.regions import build_region_views
from melon_tss.fusion import PhysicsAssistedRegionRegressor
from melon_tss.regression import fit_physics_preprocessor, predict_best_protocol, fuse_huber_mse, fruit_median


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    features = DirectionAwareConvBlock(8)(torch.randn(2, 8, 32, 32))
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    loss = NetCLDiceLoss(loss_weight=0.2)(logits, torch.randint(0, 4, (2, 32, 32)))
    loss.backward()
    labels = logits.detach().argmax(1)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:54, 8:54] = 2
    mask[8:54:4, 8:54] = 1
    mask[56:60, 8:54] = 3
    physical = feature_vector(rgb[:, :, ::-1].copy(), mask)
    views = build_region_views(Image.fromarray(rgb), Image.fromarray(mask), image_size=64)
    # A synthetic training set demonstrates fold-local fitting only.
    training = rng.normal(size=(24, 161))
    pipeline = fit_physics_preprocessor(training)
    physics = torch.tensor(pipeline.transform(physical[None]).astype(np.float32))
    model = PhysicsAssistedRegionRegressor("convnext_tiny", False, 0.25, 16)
    prediction = predict_best_protocol(model, model, views[None], physics, 10.0, 2.0, 10.0, 2.0)
    averaged = fuse_huber_mse([8.0, 10.0, 14.0], [10.0, 12.0, 16.0])
    aggregated = fruit_median(averaged, ["fruit_a", "fruit_a", "fruit_b"])
    print("DAConv:", tuple(features.shape), "mask:", tuple(labels.shape))
    print("Descriptors:", physical.shape, "views:", tuple(views.shape), "prediction:", tuple(prediction.shape))
    print("Fruit aggregation:", aggregated.tolist())
    print("Synthetic execution check passed; random model output is not an accuracy result.")


if __name__ == "__main__":
    main()
