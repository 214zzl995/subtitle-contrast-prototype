from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ._clip_backends import encode_image, get_tinyclip_backend
from .config import AppConfig
from .frames import FrameRepository
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult

SCORE_THRESHOLD = 0.80


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
    try:
        backend = get_tinyclip_backend()
    except RuntimeError as exc:
        raise ValueError(
            "TinyCLIP backend unavailable. Install a TinyCLIP checkpoint or set "
            "TINYCLIP_MODEL_NAME/TINYCLIP_PRETRAINED to a valid open_clip entry."
        ) from exc

    feat_a = _blend_features(a, backend)

    best_dx = 0
    best_dy = 0
    best_img = b
    best_similarity = -1.0
    second_best = -1.0

    for dx, dy in _shift_candidates(radius):
        if dx == 0 and dy == 0:
            candidate = b
        else:
            candidate = _shift_image(b, dx, dy, fill=int(b.mean()))

        feat_b = _blend_features(candidate, backend)
        similarity = float(torch.dot(feat_a, feat_b))
        if similarity > best_similarity:
            second_best = best_similarity
            best_similarity = similarity
            best_dx = dx
            best_dy = dy
            best_img = candidate
        elif similarity > second_best:
            second_best = similarity

    margin = best_similarity - second_best if second_best > -1 else max(0.0, best_similarity)
    base_score = (best_similarity + 1.0) * 0.5
    score = float(np.clip(base_score * (1.0 + 0.1 * max(0.0, margin)), 0.0, 1.0))
    decision = "same" if score >= SCORE_THRESHOLD else "different"

    metrics = {
        "clip_similarity": float(best_similarity),
        "clip_score": score,
        "clip_margin": float(margin),
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


def _blend_features(image: np.ndarray, backend) -> torch.Tensor:
    pil_full = _to_pil_image(image)
    feats = [encode_image(backend, pil_full)]

    h, w = image.shape
    if min(h, w) >= 32:
        feats.append(encode_image(backend, _to_pil_image(_resize(image, scale=0.75))))
        feats.append(encode_image(backend, _to_pil_image(_resize(image, scale=0.5))))

    stacked = torch.stack(feats)
    mean_feat = F.normalize(stacked.mean(dim=0), dim=-1)
    return mean_feat


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return image
    h, w = image.shape
    new_h = max(8, int(round(h * scale)))
    new_w = max(8, int(round(w * scale)))
    pil = Image.fromarray(_ensure_uint8(image), mode="L")
    resized = pil.resize((new_w, new_h), Image.BICUBIC)
    return np.array(resized, dtype=np.uint8)


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
