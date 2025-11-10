from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

@dataclass(slots=True)
class PreparedRoiPair:
    """ROI metadata with the corresponding crops and optional full frames."""

    roi: Any
    frame_a: np.ndarray
    frame_b: np.ndarray
    frame_a_full: np.ndarray | None = None
    frame_b_full: np.ndarray | None = None


class _SupportsPreparedRoi(Protocol):
    def prepare_roi_pair(self, roi: Any) -> PreparedRoiPair: ...


class _SupportsRequestFrames(Protocol):
    frame_a: str
    frame_b: str
    roi: Any


def clamp_roi_to_bounds(roi: Any, width: int, height: int) -> Any:
    """Clamp ROI to the provided width/height (minimum 1×1)."""
    w = max(1, width)
    h = max(1, height)
    x = int(min(max(roi.x, 0), w - 1))
    y = int(min(max(roi.y, 0), h - 1))
    width_adj = int(max(1, min(roi.width, w - x)))
    height_adj = int(max(1, min(roi.height, h - y)))
    roi_cls = type(roi)
    return roi_cls(x=x, y=y, width=width_adj, height=height_adj)


def build_prepared_roi_pair(frame_a: np.ndarray, frame_b: np.ndarray, roi: Any) -> PreparedRoiPair:
    height = min(frame_a.shape[0], frame_b.shape[0])
    width = min(frame_a.shape[1], frame_b.shape[1])
    clamped = clamp_roi_to_bounds(roi, width, height)
    a_crop = frame_a[clamped.y : clamped.y + clamped.height, clamped.x : clamped.x + clamped.width]
    b_crop = frame_b[clamped.y : clamped.y + clamped.height, clamped.x : clamped.x + clamped.width]
    return PreparedRoiPair(
        roi=clamped,
        frame_a=a_crop,
        frame_b=b_crop,
        frame_a_full=frame_a,
        frame_b_full=frame_b,
    )


def get_prepared_roi_pair(
    repository: FrameRepository | _SupportsPreparedRoi,
    request: _SupportsRequestFrames,
) -> PreparedRoiPair:
    """Return a prepared ROI pair using the repository's shared preprocessing."""
    if hasattr(repository, "prepare_roi_pair"):
        return repository.prepare_roi_pair(request.roi)

    frame_a = repository.load_y_plane(request.frame_a)
    frame_b = repository.load_y_plane(request.frame_b)
    return build_prepared_roi_pair(frame_a, frame_b, request.roi)


# Local import to avoid circular dependency at module import time
from .frames import FrameRepository  # noqa: E402  (import at end of file)
