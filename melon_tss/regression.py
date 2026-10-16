"""The fixed PADF-EW+FM regression protocol.

The two branches are both part of the final method: SmoothL1 (Huber, beta=1)
and MSE are trained separately, each uses horizontal-flip TTA, their predictions
are averaged in TSS units, and the result is reduced by fruit-level median.
"""
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def fit_physics_preprocessor(training_features, n_components=16):
    """Fit on training rows only; call pipe.transform on validation/inference."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(10, 90))),
        ("pca", PCA(n_components=n_components, random_state=0)),
    ])
    pipe.fit(training_features)
    return pipe


def standardize_targets(training_targets):
    values = np.asarray(training_targets, dtype=np.float64)
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if std <= 0 or not np.isfinite(values).all():
        raise ValueError("Training targets must be finite and nonconstant")
    return (values - mean) / std, mean, std


def huber_loss(prediction, standardized_target):
    """Loss used by the fixed robust branch (beta=1 on standardized TSS)."""
    return F.smooth_l1_loss(prediction, standardized_target, beta=1.0)


def mse_loss(prediction, standardized_target):
    """Loss used by the fixed MSE branch on standardized TSS."""
    return F.mse_loss(prediction, standardized_target)


@torch.no_grad()
def predict_branch(model, regions, physics, target_mean, target_std):
    """Return one branch's TSS prediction with the required horizontal-flip TTA."""
    model.eval()
    prediction = model(regions, physics)
    prediction = 0.5 * (
        prediction + model(torch.flip(regions, dims=[-1]), physics))
    return prediction * target_std + target_mean


def fuse_huber_mse(huber_tss, mse_tss):
    """Fixed equal-weight fusion of the two final branches in TSS units."""
    huber_tss = np.asarray(huber_tss, dtype=float)
    mse_tss = np.asarray(mse_tss, dtype=float)
    if huber_tss.ndim != 1 or huber_tss.shape != mse_tss.shape:
        raise ValueError("Branch predictions must be aligned 1D arrays in TSS units")
    return 0.5 * (huber_tss + mse_tss)


@torch.no_grad()
def predict_best_protocol(huber_model, mse_model, regions, physics,
                          huber_mean, huber_std, mse_mean, mse_std):
    """Run the complete highest-scoring inference path for one image batch."""
    huber = predict_branch(huber_model, regions, physics, huber_mean, huber_std)
    mse = predict_branch(mse_model, regions, physics, mse_mean, mse_std)
    return 0.5 * (huber + mse)


def fruit_median(image_predictions, fruit_ids):
    """Broadcast each fruit's median to its images (primary image weighting).

    Call on one held-out fold or one inference set with aligned fruit IDs.
    Fruit IDs identify views; they are not model input features.
    """
    values = np.asarray(image_predictions, dtype=float)
    if values.ndim != 1 or len(values) != len(fruit_ids) or not np.isfinite(values).all():
        raise ValueError("Finite 1D predictions and aligned fruit IDs are required")
    groups = {}
    for i, fruit in enumerate(fruit_ids):
        groups.setdefault(fruit, []).append(i)
    output = np.empty_like(values)
    for indices in groups.values():
        output[indices] = np.median(values[indices])
    return output


def metrics(targets, predictions):
    return {"r2": float(r2_score(targets, predictions)),
            "mae": float(mean_absolute_error(targets, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(targets, predictions)))}
