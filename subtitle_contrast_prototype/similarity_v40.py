from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import AppConfig
from .frames import FrameRepository
from .roi_utils import PreparedRoiPair, get_prepared_roi_pair
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult


# ----------------------------- 常量与参数 -----------------------------

MAX_EDGE_POINTS = 800
GRID_STEP = 2
KEEP_QUANTILE = 0.80
CLIP_PX = 4.0
TIGHT_PX = 1.5
SIGMA_SCALE = 0.03
SIM_THRESHOLD = 0.60
MATCH_THRESHOLD = 0.55


# ----------------------------- 主流程 -----------------------------


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

    mu = float(request.mu_sub) / 255.0
    delta = max(float(request.delta_y) / 255.0, 1.0 / 255.0)

    mask_a = np.abs(a_norm - mu) <= delta
    mask_b = np.abs(b_norm - mu) <= delta

    mask_a = _morph_close(_morph_open(mask_a, (1, 1)), (2, 1))
    mask_b = _morph_close(_morph_open(mask_b, (1, 1)), (2, 1))

    grad_a = _sobel_gradient(a_norm)
    grad_b = _sobel_gradient(b_norm)

    thresh_a = _adaptive_percentile(grad_a, mask_a, 70.0)
    thresh_b = _adaptive_percentile(grad_b, mask_b, 70.0)

    edge_a = (grad_a >= thresh_a) & mask_a
    edge_b = (grad_b >= thresh_b) & mask_b

    points_a = _sample_edge_points(edge_a, MAX_EDGE_POINTS, GRID_STEP)
    points_b = _sample_edge_points(edge_b, MAX_EDGE_POINTS, GRID_STEP)

    dt_edge_a = _distance_transform(edge_a)
    dt_edge_b = _distance_transform(edge_b)

    skeleton_a = _skeletonize(mask_a)
    skeleton_b = _skeletonize(mask_b)
    dt_bg_a = _distance_transform(~mask_a)
    dt_bg_b = _distance_transform(~mask_b)
    w_med_a = _median_stroke_width(dt_bg_a, skeleton_a)
    w_med_b = _median_stroke_width(dt_bg_b, skeleton_b)

    search_radius = request.search_radius or config.search_radius
    best = _search_best_shift(
        points_a,
        points_b,
        dt_edge_a,
        dt_edge_b,
        search_radius,
    )

    diag = math.hypot(roi.width, roi.height)
    sigma = max(SIGMA_SCALE * diag, 1e-6)
    sim_core = math.exp(-((best.cost / sigma) ** 2)) if np.isfinite(best.cost) else 0.0

    stroke_gap = abs(w_med_a - w_med_b)
    stroke_penalty = math.exp(-((stroke_gap / 2.0) ** 2))
    similarity = sim_core * stroke_penalty

    decision = _decision(similarity, best.match_fraction)

    metrics = {
        "similarity": float(np.clip(similarity, 0.0, 1.0)),
        "core_similarity": float(np.clip(sim_core, 0.0, 1.0)),
        "match_fraction": float(np.clip(best.match_fraction, 0.0, 1.0)),
        "stroke_width_penalty": float(np.clip(stroke_penalty, 0.0, 1.0)),
    }

    details = {
        "best_cost_px": float(best.cost),
        "best_shift_x": float(best.dx),
        "best_shift_y": float(best.dy),
        "matched_fraction": float(best.match_fraction),
        "stroke_w_med_a": float(w_med_a),
        "stroke_w_med_b": float(w_med_b),
        "stroke_gap_px": float(stroke_gap),
        "edge_points_a": float(len(points_a)),
        "edge_points_b": float(len(points_b)),
        "search_radius": float(search_radius),
    }

    result = SubtitleSimilarityResult(
        score=float(np.clip(similarity, 0.0, 1.0)),
        confidence=float(np.clip(similarity, 0.0, 1.0)),
        decision="same" if decision else "different",
        roi=roi,
        metrics=metrics,
        details=details,
        delta=(best.dx, best.dy),
    )
    return result


# ----------------------------- 搜索与 Chamfer -----------------------------


@dataclass(slots=True)
class ChamferMatch:
    cost: float
    match_fraction: float
    dx: int
    dy: int


def _search_best_shift(
    points_a: Sequence[Tuple[int, int]],
    points_b: Sequence[Tuple[int, int]],
    dt_a: np.ndarray,
    dt_b: np.ndarray,
    radius: int,
) -> ChamferMatch:
    if len(points_a) == 0 or len(points_b) == 0:
        return ChamferMatch(cost=float("inf"), match_fraction=0.0, dx=0, dy=0)

    height, width = dt_a.shape
    best_cost = float("inf")
    best_match = 0.0
    best_shift = (0, 0)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            c12, f12 = _one_way_partial_chamfer(
                points_a, dt_b, dx, dy, width, height, KEEP_QUANTILE, CLIP_PX, TIGHT_PX
            )
            c21, f21 = _one_way_partial_chamfer(
                points_b, dt_a, -dx, -dy, width, height, KEEP_QUANTILE, CLIP_PX, TIGHT_PX
            )
            cost = 0.5 * (c12 + c21)
            match = 0.5 * (f12 + f21)
            if cost < best_cost or (math.isclose(cost, best_cost) and match > best_match):
                best_cost = cost
                best_match = match
                best_shift = (dx, dy)

    return ChamferMatch(cost=best_cost, match_fraction=best_match, dx=best_shift[0], dy=best_shift[1])


def _one_way_partial_chamfer(
    points: Sequence[Tuple[int, int]],
    dt: np.ndarray,
    dx: int,
    dy: int,
    width: int,
    height: int,
    keep_quantile: float,
    clip_px: float,
    tight_px: float,
) -> Tuple[float, float]:
    if not points:
        return float("inf"), 0.0

    distances: List[float] = []
    tight_hits = 0
    total = 0

    for x, y in points:
        xs = x + dx
        ys = y + dy
        if 0 <= xs < width and 0 <= ys < height:
            d = float(dt[ys, xs])
            d = min(d, clip_px)
            distances.append(d)
            if d <= tight_px:
                tight_hits += 1
            total += 1

    if total == 0 or not distances:
        return float("inf"), 0.0

    distances.sort()
    keep_count = max(1, int(math.floor(len(distances) * keep_quantile)))
    kept = distances[:keep_count]
    cost = float(sum(kept) / len(kept))
    frac = float(tight_hits / total)
    return cost, frac


# ----------------------------- 工具函数 -----------------------------


def _sobel_gradient(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = kernel_x.T
    padded = np.pad(image, 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    gx = np.sum(windows * kernel_x, axis=(-1, -2))
    gy = np.sum(windows * kernel_y, axis=(-1, -2))
    magnitude = np.hypot(gx, gy)
    return magnitude.astype(np.float32)


def _adaptive_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    selection = values[mask]
    if selection.size > 0:
        return float(np.percentile(selection, percentile))
    return float(np.percentile(values, percentile))


def _sample_edge_points(edge: np.ndarray, max_points: int, grid_step: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    height, width = edge.shape
    for y in range(0, height, grid_step):
        y_end = min(y + grid_step, height)
        for x in range(0, width, grid_step):
            x_end = min(x + grid_step, width)
            block = edge[y:y_end, x:x_end]
            if np.any(block):
                rel = np.argwhere(block)
                cy, cx = rel[0]
                points.append((x + int(cx), y + int(cy)))
                if len(points) >= max_points:
                    return points
    return points


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        shape = mask.shape
        return np.full(shape, np.inf, dtype=np.float32)

    mask = mask.astype(bool)
    inf = 1e9
    dist = np.where(mask, 0.0, inf).astype(np.float64)

    for y in range(dist.shape[0]):
        dist[y, :] = _edt_1d(dist[y, :])
    for x in range(dist.shape[1]):
        dist[:, x] = _edt_1d(dist[:, x])

    dist = np.sqrt(dist, out=dist)
    return dist.astype(np.float32)


def _edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    result = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int32)
    z = np.zeros(n + 1, dtype=np.float64)

    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1, n):
        s = _intersection(f, v[k], q)
        while s <= z[k]:
            k -= 1
            s = _intersection(f, v[k], q)
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        diff = q - v[k]
        result[q] = diff * diff + f[v[k]]
    return result


def _intersection(f: np.ndarray, i: int, j: int) -> float:
    if i == j:
        return np.inf
    return ((f[j] + j * j) - (f[i] + i * i)) / (2.0 * (j - i))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    prev = np.zeros_like(binary)
    while True:
        removed = _zhang_suen_step(binary, step=0)
        removed |= _zhang_suen_step(binary, step=1)
        binary[removed] = 0
        if np.array_equal(binary, prev):
            break
        prev = binary.copy()
        if not np.any(removed):
            break
    return binary.astype(bool)


def _zhang_suen_step(image: np.ndarray, step: int) -> np.ndarray:
    height, width = image.shape
    to_remove = np.zeros_like(image, dtype=bool)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if image[y, x] == 0:
                continue
            n = _neighbors(image, x, y)
            transitions = _transitions(n)
            if not (2 <= sum(n) <= 6):
                continue
            if transitions != 1:
                continue
            if step == 0:
                if n[0] * n[2] * n[4] != 0:
                    continue
                if n[2] * n[4] * n[6] != 0:
                    continue
            else:
                if n[0] * n[2] * n[6] != 0:
                    continue
                if n[0] * n[4] * n[6] != 0:
                    continue
            to_remove[y, x] = True
    return to_remove


def _neighbors(image: np.ndarray, x: int, y: int) -> List[int]:
    return [
        image[y - 1, x],
        image[y - 1, x + 1],
        image[y, x + 1],
        image[y + 1, x + 1],
        image[y + 1, x],
        image[y + 1, x - 1],
        image[y, x - 1],
        image[y - 1, x - 1],
    ]


def _transitions(neighbors: Sequence[int]) -> int:
    transitions = 0
    for i in range(len(neighbors)):
        if neighbors[i] == 0 and neighbors[(i + 1) % len(neighbors)] == 1:
            transitions += 1
    return transitions


def _median_stroke_width(distance_map: np.ndarray, skeleton: np.ndarray) -> float:
    if not np.any(skeleton):
        mask = distance_map < np.inf
        if not np.any(mask):
            return 0.0
        return float(np.median(distance_map[mask]) * 2.0)
    distances = distance_map[skeleton]
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite) * 2.0)


def _binary_erosion(mask: np.ndarray, kernel: Tuple[int, int]) -> np.ndarray:
    ky, kx = kernel
    pad_y = ky // 2
    pad_x = kx // 2
    padded = np.pad(mask, ((pad_y, ky - 1 - pad_y), (pad_x, kx - 1 - pad_x)), mode="constant", constant_values=False)
    windows = sliding_window_view(padded, (ky, kx))
    return windows.all(axis=(-1, -2))


def _binary_dilation(mask: np.ndarray, kernel: Tuple[int, int]) -> np.ndarray:
    ky, kx = kernel
    pad_y = ky // 2
    pad_x = kx // 2
    padded = np.pad(mask, ((pad_y, ky - 1 - pad_y), (pad_x, kx - 1 - pad_x)), mode="constant", constant_values=False)
    windows = sliding_window_view(padded, (ky, kx))
    return windows.any(axis=(-1, -2))


def _morph_open(mask: np.ndarray, kernel: Tuple[int, int]) -> np.ndarray:
    return _binary_dilation(_binary_erosion(mask, kernel), kernel)


def _morph_close(mask: np.ndarray, kernel: Tuple[int, int]) -> np.ndarray:
    return _binary_erosion(_binary_dilation(mask, kernel), kernel)


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


def _decision(similarity: float, match_fraction: float) -> bool:
    if similarity < SIM_THRESHOLD:
        return False
    if match_fraction < MATCH_THRESHOLD:
        return False
    return True
