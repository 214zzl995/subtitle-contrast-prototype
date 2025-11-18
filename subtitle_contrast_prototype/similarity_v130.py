from __future__ import annotations

import numpy as np

from .config import AppConfig
from .frames import FrameRepository
from .roi_utils import PreparedRoiPair, get_prepared_roi_pair
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult


def compute_similarity(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
    prepared: PreparedRoiPair | None = None,
) -> SubtitleSimilarityResult:
    roi_data = prepared or get_prepared_roi_pair(repository, request)
    roi = roi_data.roi
    frame_a = roi_data.frame_a
    frame_b = roi_data.frame_b

    if frame_a.size == 0 or frame_b.size == 0:
        raise ValueError("ROI is empty after clamping to frame bounds.")

    mu = float(np.clip(request.mu_sub, 0.0, 255.0))
    delta = float(max(request.delta_y, 0.0))
    lower = max(0.0, mu - delta)
    upper = min(255.0, mu + delta)

    count_a = _count_in_range(frame_a, lower, upper)
    count_b = _count_in_range(frame_b, lower, upper)

    similarity = _similarity_from_counts(count_a, count_b)
    decision = "same" if similarity >= 0.5 else "different"

    metrics = {
        "similarity": similarity,
        "count_a": float(count_a),
        "count_b": float(count_b),
    }
    details = {
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "roi_area": float(roi.width * roi.height),
        "count_gap": float(abs(count_a - count_b)),
    }

    return SubtitleSimilarityResult(
        score=similarity,
        confidence=similarity,
        decision=decision,
        roi=roi,
        metrics=metrics,
        details=details,
        delta=(0, 0),
    )


def _count_in_range(values: np.ndarray, lower: float, upper: float) -> int:
    mask = (values >= lower) & (values <= upper)
    return int(mask.sum())


def _similarity_from_counts(count_a: int, count_b: int) -> float:
    if count_a == 0 and count_b == 0:
        return 1.0
    denom = max(count_a, count_b, 1)
    similarity = 1.0 - abs(count_a - count_b) / denom
    return float(np.clip(similarity, 0.0, 1.0))
