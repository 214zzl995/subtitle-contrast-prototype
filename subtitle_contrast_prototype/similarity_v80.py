from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

import lpips
import numpy as np
import torch

from .config import AppConfig
from .frames import FrameRepository
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult

SCORE_THRESHOLD = 0.75


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

    a_u8 = _ensure_uint8(a)
    b_u8 = _ensure_uint8(b)
    radius = int(request.search_radius or config.search_radius or 0)

    model, device = _get_lpips()
    a_tensor = _to_lpips_tensor(a_u8, device)

    best_dx = 0
    best_dy = 0
    best_img = b_u8
    best_distance = float(_lpips_distance(model, a_tensor, _to_lpips_tensor(b_u8, device)))

    for dx, dy in _shift_candidates(radius):
        if dx == 0 and dy == 0:
            distance = best_distance
        else:
            shifted = _shift_image(b_u8, dx, dy, fill=int(b_u8.mean()))
            distance = _lpips_distance(model, a_tensor, _to_lpips_tensor(shifted, device))
            if distance < best_distance:
                best_distance = distance
                best_dx = dx
                best_dy = dy
                best_img = shifted
        if best_distance <= 0.0:
            break

    score = float(max(0.0, 1.0 - best_distance))
    decision = "same" if score >= SCORE_THRESHOLD else "different"

    metrics = {
        "lpips_distance": float(best_distance),
        "lpips_score": float(score),
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
        "search_radius": float(radius),
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


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8, copy=False)


def _to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor = tensor.repeat(1, 3, 1, 1)  # (1,3,H,W)
    tensor = tensor.mul(2.0).sub(1.0)  # scale to [-1,1]
    return tensor.to(device=device, dtype=torch.float32, non_blocking=False).contiguous()


def _lpips_distance(model: lpips.LPIPS, a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        distance = model(a_tensor, b_tensor)
    return float(distance.squeeze().item())


@lru_cache(maxsize=1)
def _get_lpips() -> Tuple[lpips.LPIPS, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="squeeze")
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, device


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

