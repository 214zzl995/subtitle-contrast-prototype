from __future__ import annotations

from typing import Iterable, Tuple

import imagehash
import numpy as np
from PIL import Image

from .config import AppConfig
from .frames import FrameRepository
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult

UPSAMPLE = 2
HASH_SIZE = 12
HIGHFREQ_FACTOR = 4
SCORE_THRESHOLD = 0.82


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
    a_u8 = _normalize_to_uint8(a)
    b_u8 = _normalize_to_uint8(b)

    a_hash = _hash_image(a_u8)

    best_dx = 0
    best_dy = 0
    best_img = b_u8
    best_hash = _hash_image(b_u8)
    best_dist = a_hash - best_hash

    for dx, dy in _shift_candidates(radius):
        if dx == 0 and dy == 0:
            dist = a_hash - best_hash
        else:
            shifted = _shift_image(b_u8, dx, dy, fill=int(b_u8.mean()))
            current_hash = _hash_image(shifted)
            dist = a_hash - current_hash
            if dist < best_dist:
                best_dist = dist
                best_dx = dx
                best_dy = dy
                best_img = shifted
                best_hash = current_hash
        if best_dist <= 0:
            break

    bits = float(a_hash.hash.size)
    normalized_dist = float(best_dist) / bits if bits else 1.0
    score = max(0.0, 1.0 - normalized_dist)
    decision = "same" if score >= SCORE_THRESHOLD else "different"

    metrics = {
        "hash_distance": float(best_dist),
        "hash_bits": bits,
        "normalized_distance": normalized_dist,
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
        "mean_a": float(a_u8.mean()),
        "mean_b": float(best_img.mean()),
        "std_a": float(a_u8.std()),
        "std_b": float(best_img.std()),
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


def _shift_candidates(radius: int) -> Iterable[Tuple[int, int]]:
    yield (0, 0)
    if radius <= 0:
        return
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            yield (dx, dy)


def _hash_image(image_u8: np.ndarray) -> imagehash.ImageHash:
    img = Image.fromarray(image_u8, mode="L")
    if UPSAMPLE > 1:
        width, height = img.size
        img = img.resize((width * UPSAMPLE, height * UPSAMPLE), Image.BICUBIC)
    return imagehash.phash(img, hash_size=HASH_SIZE, highfreq_factor=HIGHFREQ_FACTOR)


def _shift_image(image: np.ndarray, dx: int, dy: int, fill: int = 0) -> np.ndarray:
    h, w = image.shape
    shifted = np.full_like(image, fill, dtype=np.uint8)

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


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 1e-6:
        arr /= max_val
    arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8, copy=False)


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
