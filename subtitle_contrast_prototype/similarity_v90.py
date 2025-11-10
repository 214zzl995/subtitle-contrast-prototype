from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image

from ._clip_backends import encode_image, get_mobileclip2_backend
from .config import AppConfig
from .frames import FrameRepository
from .roi_utils import PreparedRoiPair, get_prepared_roi_pair
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult

SCORE_THRESHOLD = 0.82


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

    radius = int(request.search_radius or config.search_radius or 0)
    backend = get_mobileclip2_backend()

    a_img = _to_pil_image(a)
    feat_a = encode_image(backend, a_img)

    best_dx = 0
    best_dy = 0
    best_img = b
    best_similarity = -1.0

    for dx, dy in _shift_candidates(radius):
        if dx == 0 and dy == 0:
            candidate = b
        else:
            candidate = _shift_image(b, dx, dy, fill=int(b.mean()))
        candidate_img = _to_pil_image(candidate)
        feat_b = encode_image(backend, candidate_img)
        similarity = float(torch.dot(feat_a, feat_b))
        if similarity > best_similarity:
            best_similarity = similarity
            best_dx = dx
            best_dy = dy
            best_img = candidate

    clip_score = (best_similarity + 1.0) * 0.5
    score = float(np.clip(clip_score, 0.0, 1.0))
    decision = "same" if score >= SCORE_THRESHOLD else "different"

    metrics = {
        "clip_similarity": float(best_similarity),
        "clip_score": score,
        "clip_distance": float(1.0 - best_similarity),
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
        "backend": backend.name,
        "pretrained": backend.pretrained_tag,
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
