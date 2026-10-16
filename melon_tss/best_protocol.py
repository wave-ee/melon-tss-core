"""One-call wiring for the highest-scoring PADF-EW+FM method.

This module exposes one fixed path. A caller supplies a predicted mask, a
fold-fitted physical pipeline,
and the two trained branch checkpoints.
"""
import numpy as np
import torch
from PIL import Image

from .phenotypes import feature_vector
from .regions import build_region_views
from .regression import predict_best_protocol, fruit_median


IMAGE_SIZE = 320


def prepare_best_inputs(image_rgb, mask, image_bgr, physics_pipeline):
    """Create the exact five-view and 16-D physical inputs for one image.

    ``image_rgb`` is a PIL RGB image, ``mask`` is a PIL grayscale label mask,
    and ``image_bgr`` is the same image in an original-resolution uint8 OpenCV
    BGR array. ``physics_pipeline`` must be fitted on the current training fold.
    """
    regions = build_region_views(
        image_rgb, mask, image_size=IMAGE_SIZE, training=False).unsqueeze(0)
    raw = feature_vector(image_bgr, np.asarray(mask, dtype=np.uint8))
    physical = physics_pipeline.transform(raw.reshape(1, -1)).astype(np.float32)
    return regions, torch.from_numpy(physical)


def predict_best(huber_model, mse_model, regions, physical,
                 huber_target_mean, huber_target_std,
                 mse_target_mean, mse_target_std):
    """Predict TSS with flip TTA and fixed equal Huber/MSE fusion."""
    return predict_best_protocol(
        huber_model, mse_model, regions, physical,
        huber_target_mean, huber_target_std,
        mse_target_mean, mse_target_std)


def aggregate_best(image_predictions, fruit_ids):
    """Apply the final within-fruit median aggregation."""
    return fruit_median(image_predictions, fruit_ids)
