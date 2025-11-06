from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .config import AppConfig
from .frames import FrameRepository
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult


# ----------------------------- 常量配置 -----------------------------

SIM_THRESHOLD = 0.60
MATCH_THRESHOLD = 0.55
SIGMA_SCALE = 0.03
GRID_STEP = 2
MAX_EDGE_POINTS = 800
KEEP_QUANTILE = 0.80
CLIP_PX = 4.0
TIGHT_PX = 1.5
ORB_FEATURES = 1000
ORB_RANSAC_THRESH = 1.5


# ----------------------------- 主流程 -----------------------------


def compute_similarity(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
) -> SubtitleSimilarityResult:
    roi = _clamp_roi(request.roi, config.width, config.height)
    frame_a = repository.load_y_plane(request.frame_a)[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    frame_b = repository.load_y_plane(request.frame_b)[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    if frame_a.size == 0 or frame_b.size == 0:
        raise ValueError("ROI is empty after clamping to frame bounds.")

    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)

    mu = float(request.mu_sub)
    delta = max(float(request.delta_y), 1.0)

    mask_a = _hard_threshold(a, mu, delta)
    mask_b = _hard_threshold(b, mu, delta)

    mask_a = _morph_close(_morph_open(mask_a, (1, 1)), (2, 1))
    mask_b = _morph_close(_morph_open(mask_b, (1, 1)), (2, 1))

    grad_a = _sobel_magnitude(a)
    grad_b = _sobel_magnitude(b)

    thr_a = _adaptive_percentile(grad_a, mask_a, 70.0)
    thr_b = _adaptive_percentile(grad_b, mask_b, 70.0)

    edge_a = (grad_a >= thr_a) & mask_a
    edge_b = (grad_b >= thr_b) & mask_b

    points_a = _sample_edge_points(edge_a, MAX_EDGE_POINTS, GRID_STEP)
    points_b = _sample_edge_points(edge_b, MAX_EDGE_POINTS, GRID_STEP)

    dt_a = _distance_transform_cv(edge_a)
    dt_b = _distance_transform_cv(edge_b)

    skeleton_a = _skeletonize(mask_a)
    skeleton_b = _skeletonize(mask_b)
    dt_bg_a = _distance_transform_bool(~mask_a)
    dt_bg_b = _distance_transform_bool(~mask_b)
    stroke_a = _median_stroke_width(dt_bg_a, skeleton_a)
    stroke_b = _median_stroke_width(dt_bg_b, skeleton_b)

    search_radius = request.search_radius or config.search_radius
    chamfer = _search_best_shift(points_a, points_b, dt_a, dt_b, search_radius)

    diag = math.hypot(roi.width, roi.height)
    sigma = max(SIGMA_SCALE * diag, 1e-6)
    sim_core = math.exp(-((chamfer.cost / sigma) ** 2)) if np.isfinite(chamfer.cost) else 0.0

    stroke_gap = abs(stroke_a - stroke_b)
    stroke_penalty = math.exp(-((stroke_gap / 2.0) ** 2))

    base_similarity = sim_core * stroke_penalty

    shifted_b = _shift_image(b, chamfer.dx, chamfer.dy, fill=mu)
    shifted_mask_b = _shift_mask(mask_b, chamfer.dx, chamfer.dy)

    template_similarity = _compute_template_similarity(
        a, shifted_b, mask_a, shifted_mask_b
    )
    orb_similarity, orb_inlier_ratio, orb_median_residual = _compute_orb_similarity(
        a, shifted_b, mask_a, shifted_mask_b
    )

    similarity_candidates = [base_similarity]
    if template_similarity > 0.0:
        similarity_candidates.append(template_similarity)
    if orb_similarity > 0.0:
        similarity_candidates.append(orb_similarity)
    final_similarity = max(similarity_candidates)

    decision = (
        "same"
        if final_similarity >= SIM_THRESHOLD and chamfer.match_fraction >= MATCH_THRESHOLD
        else "different"
    )

    metrics: Dict[str, float] = {
        "similarity": float(np.clip(final_similarity, 0.0, 1.0)),
        "core_similarity": float(np.clip(sim_core, 0.0, 1.0)),
        "match_fraction": float(np.clip(chamfer.match_fraction, 0.0, 1.0)),
        "stroke_width_penalty": float(np.clip(stroke_penalty, 0.0, 1.0)),
        "template_similarity": float(np.clip(template_similarity, 0.0, 1.0)),
        "orb_similarity": float(np.clip(orb_similarity, 0.0, 1.0)),
    }

    details: Dict[str, float] = {
        "dx": float(chamfer.dx),
        "dy": float(chamfer.dy),
        "best_cost_px": float(chamfer.cost),
        "matched_fraction": float(chamfer.match_fraction),
        "stroke_w_med_a": float(stroke_a),
        "stroke_w_med_b": float(stroke_b),
        "stroke_gap_px": float(stroke_gap),
        "edge_points_a": float(len(points_a)),
        "edge_points_b": float(len(points_b)),
        "template_similarity_raw": float(template_similarity),
        "orb_inlier_ratio": float(orb_inlier_ratio),
        "orb_median_residual": float(orb_median_residual),
        "search_radius": float(search_radius),
    }

    return SubtitleSimilarityResult(
        score=metrics["similarity"],
        confidence=metrics["similarity"],
        decision=decision,
        roi=roi,
        metrics=metrics,
        details=details,
        delta=(chamfer.dx, chamfer.dy),
    )


# ----------------------------- Chamfer 搜索 -----------------------------


@dataclass(slots=True)
class ChamferMatch:
    cost: float
    match_fraction: float
    dx: int
    dy: int


def _search_best_shift(
    points_a: np.ndarray,
    points_b: np.ndarray,
    dt_a: np.ndarray,
    dt_b: np.ndarray,
    radius: int,
) -> ChamferMatch:
    if points_a.size == 0 or points_b.size == 0:
        return ChamferMatch(cost=float("inf"), match_fraction=0.0, dx=0, dy=0)

    h, w = dt_a.shape
    best_cost = float("inf")
    best_match = 0.0
    best_shift = (0, 0)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            c_ab, f_ab = _one_way_chamfer(points_a, dt_b, dx, dy, w, h)
            c_ba, f_ba = _one_way_chamfer(points_b, dt_a, -dx, -dy, w, h)
            cost = 0.5 * (c_ab + c_ba)
            match = 0.5 * (f_ab + f_ba)
            if cost < best_cost or (math.isclose(cost, best_cost) and match > best_match):
                best_cost = cost
                best_match = match
                best_shift = (dx, dy)

    return ChamferMatch(cost=best_cost, match_fraction=best_match, dx=best_shift[0], dy=best_shift[1])


def _one_way_chamfer(
    points: np.ndarray,
    distance_map: np.ndarray,
    dx: int,
    dy: int,
    width: int,
    height: int,
) -> Tuple[float, float]:
    if points.size == 0:
        return float("inf"), 0.0

    xs = points[:, 0] + dx
    ys = points[:, 1] + dy
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size == 0:
        return float("inf"), 0.0

    distances = distance_map[ys, xs]
    distances = np.minimum(distances, CLIP_PX)
    tight_hits = np.mean(distances <= TIGHT_PX)

    if distances.size == 0:
        return float("inf"), 0.0

    keep = max(1, int(math.floor(distances.size * KEEP_QUANTILE)))
    kept = np.partition(distances, keep - 1)[:keep]
    cost = float(np.mean(kept))
    return cost, float(tight_hits)


# ----------------------------- OpenCV 工具函数 -----------------------------


def _hard_threshold(image: np.ndarray, mu: float, delta: float) -> np.ndarray:
    lower = max(mu - delta, 0.0)
    upper = min(mu + delta, 255.0)
    mask = cv2.inRange(image, lower, upper)
    return mask > 0


def _morph_open(mask: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    if kernel_shape == (1, 1):
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    mask_u8 = (mask.astype(np.uint8)) * 255
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return opened > 0


def _morph_close(mask: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    mask_u8 = (mask.astype(np.uint8)) * 255
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return magnitude.astype(np.float32)


def _adaptive_percentile(magnitude: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    values = magnitude[mask]
    if values.size == 0:
        return float(np.percentile(magnitude, percentile))
    return float(np.percentile(values, percentile))


def _sample_edge_points(edge: np.ndarray, max_points: int, step: int) -> np.ndarray:
    points: List[Tuple[int, int]] = []
    height, width = edge.shape
    for y in range(0, height, step):
        y_end = min(y + step, height)
        for x in range(0, width, step):
            x_end = min(x + step, width)
            block = edge[y:y_end, x:x_end]
            if not np.any(block):
                continue
            rel = np.argwhere(block)
            yy, xx = rel[0]
            points.append((x + int(xx), y + int(yy)))
            if len(points) >= max_points:
                return np.asarray(points, dtype=np.int32)
    if not points:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(points, dtype=np.int32)


def _distance_transform_cv(edge: np.ndarray) -> np.ndarray:
    edge_u8 = (edge.astype(np.uint8)) * 255
    dist = cv2.distanceTransform(edge_u8, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


def _shift_image(image: np.ndarray, dx: float, dy: float, fill: float) -> np.ndarray:
    height, width = image.shape
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill),
    )
    return shifted


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = mask.shape
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(
        (mask.astype(np.uint8)) * 255,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted > 0


def _compute_template_similarity(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    valid = mask_a & mask_b
    if not np.any(valid):
        return 0.0
    values_a = image_a[valid]
    values_b = image_b[valid]
    if values_a.size < 16:
        return 0.0
    return _normalized_cross_correlation(values_a.astype(np.float32), values_b.astype(np.float32))


def _compute_orb_similarity(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> Tuple[float, float, float]:
    img_a_u8 = np.ascontiguousarray(np.clip(image_a, 0, 255).astype(np.uint8))
    img_b_u8 = np.ascontiguousarray(np.clip(image_b, 0, 255).astype(np.uint8))
    mask_a_u8 = np.ascontiguousarray((mask_a.astype(np.uint8)) * 255)
    mask_b_u8 = np.ascontiguousarray((mask_b.astype(np.uint8)) * 255)

    if np.count_nonzero(mask_a_u8) < 16 or np.count_nonzero(mask_b_u8) < 16:
        return 0.0, 0.0, float("inf")

    try:
        orb = cv2.ORB_create(ORB_FEATURES)
        kp1, des1 = orb.detectAndCompute(img_a_u8, mask_a_u8)
        kp2, des2 = orb.detectAndCompute(img_b_u8, mask_b_u8)
    except cv2.error:
        return 0.0, 0.0, float("inf")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0.0, 0.0, float("inf")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return 0.0, 0.0, float("inf")

    matches = sorted(matches, key=lambda m: m.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    transform, inliers = cv2.estimateAffine2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ORB_RANSAC_THRESH,
        maxIters=2000,
        confidence=0.99,
    )
    if transform is None or inliers is None:
        return 0.0, 0.0, float("inf")

    inliers_mask = inliers.ravel().astype(bool)
    if not np.any(inliers_mask):
        return 0.0, 0.0, float("inf")

    pts1_in = pts1[inliers_mask]
    pts2_in = pts2[inliers_mask]

    pts1_h = np.concatenate([pts1_in, np.ones((pts1_in.shape[0], 1), dtype=np.float32)], axis=1)
    pts1_transformed = (transform @ pts1_h.T).T
    residuals = np.linalg.norm(pts1_transformed - pts2_in, axis=1)
    median_residual = float(np.median(residuals))

    inlier_ratio = float(np.mean(inliers_mask))
    similarity = float(np.clip(inlier_ratio * math.exp(-((median_residual / 2.0) ** 2)), 0.0, 1.0))
    return similarity, inlier_ratio, median_residual


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_mean = float(a.mean())
    b_mean = float(b.mean())
    a_center = a - a_mean
    b_center = b - b_mean
    denom = float(np.linalg.norm(a_center) * np.linalg.norm(b_center))
    if denom <= 1e-6:
        return 0.0
    return float(np.clip(np.dot(a_center, b_center) / denom, -1.0, 1.0))


# ----------------------------- 辅助工具 -----------------------------


def _distance_transform_bool(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    inf = 1e9
    dist = np.where(mask, 0.0, inf).astype(np.float64)

    for y in range(height):
        dist[y, :] = _edt_1d(dist[y, :])
    for x in range(width):
        dist[:, x] = _edt_1d(dist[:, x])  # type: ignore[index]

    np.sqrt(dist, out=dist)
    return dist.astype(np.float32)


def _edt_1d(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    output = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int32)
    z = np.zeros(n + 1, dtype=np.float64)

    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1, n):
        s = _intersection(values, v[k], q)
        while s <= z[k]:
            k -= 1
            s = _intersection(values, v[k], q)
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        diff = q - v[k]
        output[q] = diff * diff + values[v[k]]
    return output


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
            neighbors = _neighbors(image, x, y)
            transitions = _transitions(neighbors)
            if not (2 <= sum(neighbors) <= 6):
                continue
            if transitions != 1:
                continue
            if step == 0:
                if neighbors[0] * neighbors[2] * neighbors[4] != 0:
                    continue
                if neighbors[2] * neighbors[4] * neighbors[6] != 0:
                    continue
            else:
                if neighbors[0] * neighbors[2] * neighbors[6] != 0:
                    continue
                if neighbors[0] * neighbors[4] * neighbors[6] != 0:
                    continue
            to_remove[y, x] = True
    return to_remove


def _neighbors(image: np.ndarray, x: int, y: int) -> List[int]:
    return [
        int(image[y - 1, x] > 0),
        int(image[y - 1, x + 1] > 0),
        int(image[y, x + 1] > 0),
        int(image[y + 1, x + 1] > 0),
        int(image[y + 1, x] > 0),
        int(image[y + 1, x - 1] > 0),
        int(image[y, x - 1] > 0),
        int(image[y - 1, x - 1] > 0),
    ]


def _transitions(neighbors: Sequence[int]) -> int:
    transitions = 0
    for idx in range(len(neighbors)):
        if neighbors[idx] == 0 and neighbors[(idx + 1) % len(neighbors)] == 1:
            transitions += 1
    return transitions


def _median_stroke_width(distance_map: np.ndarray, skeleton: np.ndarray) -> float:
    if not np.any(skeleton):
        finite = distance_map[np.isfinite(distance_map)]
        if finite.size == 0:
            return 0.0
        return float(np.median(finite) * 2.0)
    values = distance_map[skeleton]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite) * 2.0)


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
