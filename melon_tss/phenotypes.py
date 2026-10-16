"""Image/mask physical descriptors (OpenCV BGR input, label mask 0..3)."""
import json
from importlib.resources import read_text
import cv2
import numpy as np
from skimage.morphology import skeletonize

PHYSICS_COLUMNS = tuple(json.loads(read_text("melon_tss", "physics_feature_names.json", encoding="utf-8")))


def channel_stats(channel, mask):
    values = channel[mask > 0]
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "mode": 0.0,
            "mode_freq": 0.0,
        }
    hist = np.bincount(values.astype(np.uint8), minlength=256)
    mode = int(np.argmax(hist))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "mode": float(mode),
        "mode_freq": float(hist[mode] / values.size),
    }

def largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)

def ruler_pixels_from_mask(ruler_mask):
    ruler_mask = largest_component(ruler_mask)
    contours, _ = cv2.findContours(ruler_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan, 0.0
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return np.nan, 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return float(max(w, h)), float(cv2.contourArea(contour))

def extract_features(image_bgr, mask, ruler_real_cm):
    net_mask = (mask == 1).astype(np.uint8)
    skin_mask = (mask == 2).astype(np.uint8)
    ruler_mask = (mask == 3).astype(np.uint8)
    melon_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)

    net_px = int(net_mask.sum())
    skin_px = int(skin_mask.sum())
    melon_px = int(melon_mask.sum())
    ruler_px, ruler_area_px = ruler_pixels_from_mask(ruler_mask)

    cm_per_px = np.nan
    if np.isfinite(ruler_px) and ruler_px > 0:
        cm_per_px = ruler_real_cm / ruler_px

    area_scale = cm_per_px ** 2 if np.isfinite(cm_per_px) else np.nan
    net_area_cm2 = net_px * area_scale if np.isfinite(area_scale) else np.nan
    skin_area_cm2 = skin_px * area_scale if np.isfinite(area_scale) else np.nan
    melon_area_cm2 = melon_px * area_scale if np.isfinite(area_scale) else np.nan
    ruler_area_cm2 = ruler_area_px * area_scale if np.isfinite(area_scale) else np.nan

    net_density_skin = net_px / skin_px if skin_px > 0 else np.nan
    net_coverage_melon = net_px / melon_px if melon_px > 0 else np.nan

    thickness_px = np.nan
    thickness_std_px = np.nan
    thickness_p90_px = np.nan
    if net_px > 0:
        dist = cv2.distanceTransform(net_mask, cv2.DIST_L2, 5)
        skeleton = skeletonize(net_mask > 0)
        skeleton_dists = dist[skeleton]
        if skeleton_dists.size > 0:
            widths = 2.0 * skeleton_dists
            thickness_px = float(np.mean(widths))
            thickness_std_px = float(np.std(widths))
            thickness_p90_px = float(np.percentile(widths, 90))

    thickness_cm = thickness_px * cm_per_px if np.isfinite(thickness_px) and np.isfinite(cm_per_px) else np.nan
    thickness_mm = thickness_cm * 10.0 if np.isfinite(thickness_cm) else np.nan

    contours, _ = cv2.findContours(melon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    melon_perimeter_px = float(sum(cv2.arcLength(c, True) for c in contours)) if contours else np.nan
    melon_perimeter_cm = melon_perimeter_px * cm_per_px if np.isfinite(melon_perimeter_px) and np.isfinite(cm_per_px) else np.nan
    circularity = (
        4.0 * np.pi * melon_px / (melon_perimeter_px ** 2)
        if np.isfinite(melon_perimeter_px) and melon_perimeter_px > 0
        else np.nan
    )

    h, w = mask.shape
    moments = cv2.moments(melon_mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = w // 2, h // 2
    quadrants = [
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(cx, w)),
        (slice(cy, h), slice(0, cx)),
        (slice(cy, h), slice(cx, w)),
    ]
    quadrant_coverages = []
    for rows, cols in quadrants:
        q_melon = melon_mask[rows, cols].sum()
        q_net = net_mask[rows, cols].sum()
        quadrant_coverages.append(float(q_net / q_melon) if q_melon > 0 else 0.0)

    features = {
        "ruler_length_px": ruler_px,
        "cm_per_px": cm_per_px,
        "pixels_per_cm": 1.0 / cm_per_px if np.isfinite(cm_per_px) and cm_per_px > 0 else np.nan,
        "ruler_area_cm2": ruler_area_cm2,
        "net_px": net_px,
        "skin_px": skin_px,
        "melon_px": melon_px,
        "net_area_cm2": net_area_cm2,
        "skin_area_cm2": skin_area_cm2,
        "melon_area_cm2": melon_area_cm2,
        "net_density_skin": net_density_skin,
        "net_coverage_melon": net_coverage_melon,
        "net_thickness_px": thickness_px,
        "net_thickness_std_px": thickness_std_px,
        "net_thickness_p90_px": thickness_p90_px,
        "net_thickness_mm": thickness_mm,
        "melon_perimeter_cm": melon_perimeter_cm,
        "melon_circularity": circularity,
        "net_uniformity_var": float(np.var(quadrant_coverages)),
        "net_uniformity_range": float(np.max(quadrant_coverages) - np.min(quadrant_coverages)),
    }

    spaces = {
        "bgr": image_bgr,
        "hsv": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB),
    }
    channel_names = {
        "bgr": ("B", "G", "R"),
        "hsv": ("H", "S", "V"),
        "lab": ("L", "A", "B"),
    }
    for space_name, image in spaces.items():
        for channel_idx, channel_name in enumerate(channel_names[space_name]):
            channel = image[:, :, channel_idx]
            for region_name, region_mask in (("net", net_mask), ("skin", skin_mask), ("melon", melon_mask)):
                stats = channel_stats(channel, region_mask)
                for stat_name, value in stats.items():
                    features[f"{region_name}_{space_name}_{channel_name}_{stat_name}"] = value
            net_mean = features[f"net_{space_name}_{channel_name}_mean"]
            skin_mean = features[f"skin_{space_name}_{channel_name}_mean"]
            features[f"net_skin_{space_name}_{channel_name}_mean_diff"] = net_mean - skin_mean

    return features


def feature_vector(image_bgr, mask, ruler_real_cm=5.0):
    """Return the 161 ordered regression inputs, including missing values.

    Extract at the image's original resolution. Resize a predicted label mask
    with nearest-neighbor interpolation before calling this function.
    """
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 OpenCV BGR image")
    if mask.ndim != 2 or mask.shape != image_bgr.shape[:2]:
        raise ValueError("Label mask and image must have the same H,W")
    if not np.isin(mask, [0, 1, 2, 3]).all():
        raise ValueError("Mask labels: background=0, net=1, skin=2, ruler=3")
    if not np.isfinite(ruler_real_cm) or ruler_real_cm <= 0:
        raise ValueError("Ruler length must be a positive finite number")
    values = extract_features(image_bgr, mask, ruler_real_cm)
    return np.asarray([values[name] for name in PHYSICS_COLUMNS], dtype=np.float64)
