from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from .config import AppConfig
from .frames import FrameRepository
from .roi_utils import PreparedRoiPair, get_prepared_roi_pair
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult


TEMPLATE_HEIGHT = 64
TEMPLATE_WIDTH = 256
PHASH_SIZE = 8
PHASH_BITS = PHASH_SIZE * PHASH_SIZE
MIN_COMPONENT_RATIO = 5e-4
CORE_ERODE_MIN = 1
SHIFT_RANGE_DEFAULT = 1
SHIFT_RANGE_MAX = 2
HIGH_HASH_DISTANCE = 6
MED_HASH_DISTANCE = 12
HIGH_IOU_THRESHOLD = 0.75
VERY_HIGH_IOU_THRESHOLD = 0.90
MED_RHO_THRESHOLD = 0.85
HIGH_RHO_THRESHOLD = 0.93
EDGE_DILATION_SCALE = 0.5
EPS = 1e-6


def compute_similarity(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
    prepared: PreparedRoiPair | None = None,
) -> SubtitleSimilarityResult:
    roi_data = prepared or get_prepared_roi_pair(repository, request)
    roi = roi_data.roi
    a = roi_data.frame_a
    b = roi_data.frame_b

    if a.size == 0 or b.size == 0:
        raise ValueError("ROI is empty after clamping to frame bounds.")

    a_norm = a.astype(np.float32) / 255.0
    b_norm = b.astype(np.float32) / 255.0

    mu = request.mu_sub / 255.0
    delta = max(request.delta_y / 255.0, 1.0 / 255.0)

    core_a, template_a, w_est_a, contrast_thr_a = _build_core_and_template(a_norm, mu, delta)
    core_b, template_b, w_est_b, contrast_thr_b = _build_core_and_template(b_norm, mu, delta)

    if not core_a.any() and not core_b.any():
        zero_result = SubtitleSimilarityResult(
            score=0.0,
            confidence=0.0,
            decision="different",
            roi=roi,
            metrics={
                "similarity": 0.0,
                "phash_similarity": 0.0,
                "phash_hamming": float(PHASH_BITS),
                "core_iou": 0.0,
                "edge_ncc": 0.0,
                "w_est_a": 0.0,
                "w_est_b": 0.0,
            },
            details={
                "core_area_a": 0.0,
                "core_area_b": 0.0,
                "contrast_thr_a": float(contrast_thr_a),
                "contrast_thr_b": float(contrast_thr_b),
                "shift_dx": 0.0,
                "shift_dy": 0.0,
                "decision_stage": "no-core",
            },
            delta=(0, 0),
        )
        return zero_result

    hash_a = _phash_bits(template_a)
    hash_b = _phash_bits(template_b)
    hamming = _hamming_distance(hash_a, hash_b)

    shift_range = request.search_radius or config.search_radius or SHIFT_RANGE_DEFAULT
    shift_range = max(SHIFT_RANGE_DEFAULT, min(SHIFT_RANGE_MAX, shift_range))
    iou, (shift_dx, shift_dy), shifted_core_b = _best_iou_with_shift(core_a, core_b, shift_range)

    edge_a = _edge_magnitude(a_norm)
    edge_b = _edge_magnitude(b_norm)
    edge_b_shifted = _shift_float(edge_b, shift_dx, shift_dy, fill=0.0)

    union_mask = np.logical_or(core_a, shifted_core_b)
    stroke_values = [w for w in (w_est_a, w_est_b) if np.isfinite(w) and w > 0]
    median_w = float(np.median(stroke_values)) if stroke_values else 1.0
    dilation_radius = max(1, int(round(max(1.0, median_w * EDGE_DILATION_SCALE)))) if union_mask.any() else 1
    edge_mask = _dilate_mask(union_mask, dilation_radius)
    edge_corr = _masked_ncc(edge_a, edge_b_shifted, edge_mask)

    phash_similarity = float(max(0.0, 1.0 - hamming / PHASH_BITS))
    edge_component = 0.5 * (edge_corr + 1.0)
    score = float(np.clip(0.4 * phash_similarity + 0.35 * iou + 0.25 * edge_component, 0.0, 1.0))

    decision_stage = "different"
    confidence = score
    if hamming <= HIGH_HASH_DISTANCE and iou >= HIGH_IOU_THRESHOLD:
        decision_stage = "same-high"
        confidence = max(confidence, 0.92)
    elif (hamming <= MED_HASH_DISTANCE and edge_corr >= MED_RHO_THRESHOLD) or iou >= VERY_HIGH_IOU_THRESHOLD:
        decision_stage = "same-medium"
        confidence = max(confidence, 0.82)
    elif edge_corr >= HIGH_RHO_THRESHOLD:
        decision_stage = "same-low"
        confidence = max(confidence, 0.68)

    metrics = {
        "similarity": score,
        "phash_similarity": float(phash_similarity),
        "phash_hamming": float(hamming),
        "core_iou": float(iou),
        "edge_ncc": float(edge_corr),
        "w_est_a": float(w_est_a),
        "w_est_b": float(w_est_b),
    }

    details = {
        "core_area_a": float(core_a.mean()),
        "core_area_b": float(core_b.mean()),
        "template_fill_a": float(template_a.mean()),
        "template_fill_b": float(template_b.mean()),
        "contrast_thr_a": float(contrast_thr_a),
        "contrast_thr_b": float(contrast_thr_b),
        "shift_dx": float(shift_dx),
        "shift_dy": float(shift_dy),
        "edge_mask_radius": float(dilation_radius),
        "decision_stage": decision_stage,
    }

    result = SubtitleSimilarityResult(
        score=score,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        decision=decision_stage,
        roi=roi,
        metrics=metrics,
        details=details,
        delta=(shift_dx, shift_dy),
    )
    return result

def _build_core_and_template(image: np.ndarray, mu: float, delta: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    core, template, w_est, contrast_thr, _ = _core_pipeline(image, mu, delta, capture_debug=False)
    return core, template, w_est, contrast_thr


def _core_pipeline(
    image: np.ndarray,
    mu: float,
    delta: float,
    *,
    capture_debug: bool,
) -> Tuple[np.ndarray, np.ndarray, float, float, dict | None]:
    working = np.clip(image, 0.0, 1.0).astype(np.float32)
    debug: dict | None = {"preprocessed": working.copy()} if capture_debug else None

    h, w = working.shape
    if min(h, w) >= 3:
        working = cv2.GaussianBlur(working, (3, 3), 0)

    image_u8 = np.clip(working * 255.0, 0, 255).astype(np.uint8)
    enhanced = _top_hat_enhance(image_u8, mu)
    enhanced_f = enhanced.astype(np.float32) / 255.0
    if np.ptp(enhanced_f) > 1e-3:
        enhanced_f = cv2.normalize(enhanced_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
    if capture_debug:
        debug["enhanced"] = enhanced_f.copy()

    diff = np.abs(enhanced_f - mu)
    m0 = diff <= delta

    contrast = _local_contrast(image_u8, radius=3) / 255.0
    contrast_values = contrast[m0]
    if contrast_values.size:
        contrast_thr = float(np.percentile(contrast_values, 60.0))
    else:
        contrast_thr = float(np.percentile(contrast, 60.0)) if contrast.size else 0.0
    contrast_thr = max(contrast_thr, 0.02)
    mask_seed = np.logical_and(m0, contrast >= contrast_thr)
    if capture_debug:
        debug["mask_seed"] = mask_seed.copy()

    cleaned = _clean_mask(mask_seed)
    if not cleaned.any():
        cleaned = mask_seed

    core = _erode_mask(cleaned, radius=CORE_ERODE_MIN)
    if not core.any():
        core = cleaned

    w_est = _estimate_stroke_width(core)
    adaptive_radius = max(1, int(round(max(1.0, w_est / 3.0))))
    refined = _erode_mask(cleaned, radius=adaptive_radius)
    if refined.any():
        core = refined
        w_est = _estimate_stroke_width(core)

    if capture_debug:
        debug["core"] = core.copy()

    template = _build_template(core, w_est, target_shape=(TEMPLATE_HEIGHT, TEMPLATE_WIDTH))
    return core, template, w_est, contrast_thr, debug


def _top_hat_enhance(image_u8: np.ndarray, mu: float) -> np.ndarray:
    h, w = image_u8.shape
    size = max(3, int(round(min(h, w) / 32)))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if mu >= 0.5:
        hat = cv2.morphologyEx(image_u8, cv2.MORPH_TOPHAT, kernel)
    else:
        hat = cv2.morphologyEx(image_u8, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.addWeighted(image_u8, 0.7, hat, 0.3, 0)
    return enhanced


def _local_contrast(image_u8: np.ndarray, radius: int) -> np.ndarray:
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    local_max = cv2.dilate(image_u8, kernel)
    local_min = cv2.erode(image_u8, kernel)
    return local_max - local_min


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    min_area = max(1, int(mask.size * MIN_COMPONENT_RATIO))
    cleaned = np.zeros_like(mask_u8)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 255
    if cleaned.max() == 0:
        cleaned = mask_u8
    return cleaned > 0


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode((mask.astype(np.uint8)) * 255, kernel)
    return eroded > 0


def _estimate_stroke_width(mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    values = dist[mask]
    if values.size == 0:
        return 1.0
    d95 = float(np.percentile(values, 95.0))
    return max(1.0, 2.0 * d95)


def _build_template(mask: np.ndarray, w_est: float, target_shape: Tuple[int, int]) -> np.ndarray:
    if not mask.any():
        return np.zeros(target_shape, dtype=np.float32)
    radius = max(1, int(round(max(1.0, w_est / 6.0))))
    smoothed = _dilate_mask(mask, radius) if radius > 0 else mask.copy()
    skeleton = _skeletonize(smoothed)
    if not skeleton.any():
        skeleton = mask.copy()
    bbox = _bounding_box(skeleton)
    if bbox is None:
        return np.zeros(target_shape, dtype=np.float32)
    y0, y1, x0, x1 = bbox
    margin = max(2, int(round(w_est)))
    y0 = max(0, y0 - margin)
    y1 = min(mask.shape[0], y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(mask.shape[1], x1 + margin)
    crop = skeleton[y0:y1, x0:x1]
    if not crop.any():
        crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(target_shape, dtype=np.float32)
    inv = np.where(crop, 0, 1).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    t_max = max(1.0, w_est / 2.0)
    normalized = np.clip(dist, 0.0, t_max) / t_max
    template = _resize_template(normalized, target_shape)
    return template


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def _resize_template(patch: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if patch.size == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)
    patch = patch.astype(np.float32)
    h, w = patch.shape
    scale = min(target_h / max(1, h), target_w / max(1, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    img = (mask.astype(np.uint8)) * 255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(img)
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton > 0


def _phash_bits(template: np.ndarray) -> np.ndarray:
    resized = cv2.resize(template.astype(np.float32), (128, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized)
    low_freq = dct[:PHASH_SIZE, :PHASH_SIZE].flatten()
    if low_freq.size <= 1:
        return np.zeros(PHASH_BITS, dtype=np.uint8)
    mean = float(np.mean(low_freq[1:]))
    bits = (low_freq > mean).astype(np.uint8)
    bits[0] = 0
    return bits


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    length = min(len(a), len(b))
    return int(np.count_nonzero(a[:length] != b[:length])) + abs(len(a) - len(b))


def _best_iou_with_shift(a: np.ndarray, b: np.ndarray, radius: int) -> Tuple[float, Tuple[int, int], np.ndarray]:
    best_iou = 0.0
    best_shift = (0, 0)
    best_mask = b.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_mask(b, dx, dy)
            iou = _mask_iou(a, shifted)
            if iou > best_iou:
                best_iou = iou
                best_shift = (dx, dy)
                best_mask = shifted
    return best_iou, best_shift, best_mask


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    h, w = mask.shape
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    width = w - abs(dx)
    height = h - abs(dy)
    if width <= 0 or height <= 0:
        return shifted
    shifted[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = mask[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
    return shifted


def _shift_float(image: np.ndarray, dx: int, dy: int, fill: float = 0.0) -> np.ndarray:
    shifted = np.full_like(image, fill, dtype=np.float32)
    h, w = image.shape
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    width = w - abs(dx)
    height = h - abs(dy)
    if width <= 0 or height <= 0:
        return shifted
    shifted[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = image[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
    return shifted


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    img32 = image.astype(np.float32)
    sobel_x = cv2.Sobel(img32, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img32, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return magnitude


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate((mask.astype(np.uint8)) * 255, kernel)
    return dilated > 0


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    vals_a = a[mask]
    vals_b = b[mask]
    if vals_a.size < 16 or vals_b.size < 16:
        return 0.0
    mean_a = float(vals_a.mean())
    mean_b = float(vals_b.mean())
    da = vals_a - mean_a
    db = vals_b - mean_b
    std_a = float(np.sqrt(max((da * da).mean(), EPS)))
    std_b = float(np.sqrt(max((db * db).mean(), EPS)))
    if std_a < EPS or std_b < EPS:
        return 0.0
    corr = float(np.clip(((da * db).mean()) / (std_a * std_b), -1.0, 1.0))
    return corr


def _clamp_roi(roi: Roi, width: int, height: int) -> Roi:
    if width <= 0 or height <= 0:
        return Roi(x=0, y=0, width=1, height=1)

    w = max(1, min(int(roi.width), width))
    h = max(1, min(int(roi.height), height))

    x = int(min(max(roi.x, 0), width - 1))
    y = int(min(max(roi.y, 0), height - 1))

    if x + w > width:
        x = max(0, width - w)
    if y + h > height:
        y = max(0, height - h)

    return Roi(x=x, y=y, width=w, height=h)
