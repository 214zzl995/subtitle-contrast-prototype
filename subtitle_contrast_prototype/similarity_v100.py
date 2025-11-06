from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ._mobilenet_backend import encode_image, get_mobilenet_backend
from .config import AppConfig
from .frames import FrameRepository
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult

SCORE_THRESHOLD = 0.83


def compute_similarity(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
) -> SubtitleSimilarityResult:
    roi = _clamp_roi(request.roi, config.width, config.height)
    a = repository.load_y_plane(request.frame_a)[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    b = repository.load_y_plane(request.frame_b)[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    if a.size == 0 or b.size == 0:
        raise ValueError("ROI is empty after clamping to frame bounds.")

    radius = int(request.search_radius or config.search_radius or 0)
    backend = get_mobilenet_backend()

    feat_a, disp_a, views_a = _aggregate_features(a, backend)

    best_dx = 0
    best_dy = 0
    best_img = b
    best_similarity = -1.0
    best_disp_b = 0.0
    best_views_b = 0
    second_best = -1.0

    for dx, dy in _shift_candidates(radius):
        if dx == 0 and dy == 0:
            candidate = b
        else:
            candidate = _shift_image(b, dx, dy, fill=int(b.mean()))

        feat_b, disp_b, views_b = _aggregate_features(candidate, backend)
        similarity = float(torch.dot(feat_a, feat_b))
        if similarity > best_similarity:
            second_best = best_similarity
            best_similarity = similarity
            best_dx = dx
            best_dy = dy
            best_img = candidate
            best_disp_b = disp_b
            best_views_b = views_b
        elif similarity > second_best:
            second_best = similarity

    margin = best_similarity - second_best if second_best > -1 else max(0.0, best_similarity)
    base_score = (best_similarity + 1.0) * 0.5
    score = float(np.clip(base_score * (1.0 + 0.1 * max(0.0, margin)), 0.0, 1.0))
    decision = "same" if score >= SCORE_THRESHOLD else "different"

    metrics = {
        "feature_similarity": float(best_similarity),
        "feature_score": score,
        "feature_margin": float(margin),
        "feature_views": float(best_views_b),
        "feature_var_a": float(disp_a),
        "feature_var_b": float(best_disp_b),
    }

    details = {
        "dx": float(best_dx),
        "dy": float(best_dy),
        "roi_requested_x": float(request.roi.x),
        "roi_requested_y": float(request.roi.y),
        "roi_requested_w": float(request.roi.width),
        "roi_requested_h": float(request.roi.height),
        "roi_effective_w": float(roi.width),
        "roi_effective_h": float(roi.height),
        "mean_a": float(a.mean()),
        "mean_b": float(best_img.mean()),
        "std_a": float(a.std()),
        "std_b": float(best_img.std()),
        "search_radius": float(radius),
        "second_best": float(second_best),
        "views_a": float(views_a),
        "backend": "mobilenet_v3_small",
        "weights": backend.weights_source,
    }

    return SubtitleSimilarityResult(
        score=score,
        confidence=score,
        decision=decision,
        roi=roi,
        metrics=metrics,
        details=details,
        delta=(best_dx, best_dy),
    )


def _aggregate_features(image: np.ndarray, backend) -> Tuple[torch.Tensor, float, int]:
    views = _generate_views(image)
    features: List[torch.Tensor] = []
    for view in views:
        features.append(encode_image(backend, _to_pil_image(view)))
    stacked = torch.stack(features)
    mean_feat = F.normalize(stacked.mean(dim=0), dim=-1)
    dispersion = float(stacked.var(dim=0).mean().item())
    return mean_feat, dispersion, len(features)


def _generate_views(image: np.ndarray) -> List[np.ndarray]:
    views = [image]
    h, w = image.shape
    if min(h, w) >= 40:
        scale_areas = [0.7, 0.85]
        for area in scale_areas:
            views.append(_central_crop(image, area))
    if min(h, w) >= 32:
        views.extend(_quadrant_crops(image))
    return views


def _central_crop(image: np.ndarray, area_ratio: float) -> np.ndarray:
    h, w = image.shape
    target_area = h * w * area_ratio
    aspect = w / max(h, 1)
    new_h = int(round((target_area / max(aspect, 1e-6)) ** 0.5))
    new_w = int(round(new_h * aspect))
    new_h = max(8, min(h, new_h))
    new_w = max(8, min(w, new_w))
    y0 = max(0, (h - new_h) // 2)
    x0 = max(0, (w - new_w) // 2)
    return image[y0 : y0 + new_h, x0 : x0 + new_w]


def _quadrant_crops(image: np.ndarray) -> List[np.ndarray]:
    h, w = image.shape
    h2 = h // 2
    w2 = w // 2
    return [
        image[0:h2, 0:w2],
        image[0:h2, w2:w],
        image[h2:h, 0:w2],
        image[h2:h, w2:w],
    ]


def _to_pil_image(image: np.ndarray) -> Image.Image:
    image_u8 = _ensure_uint8(image)
    return Image.fromarray(image_u8, mode="L").convert("RGB")


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8, copy=False)


def _shift_candidates(radius: int) -> Iterable[Tuple[int, int]]:
    yield (0, 0)
    if radius <= 0:
        return
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            yield (dx, dy)


def _shift_image(image: np.ndarray, dx: int, dy: int, fill: int = 0) -> np.ndarray:
    h, w = image.shape
    shifted = np.full_like(image, fill, dtype=image.dtype)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return shifted

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return shifted


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
