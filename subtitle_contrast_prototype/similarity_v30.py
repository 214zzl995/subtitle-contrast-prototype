from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import AppConfig
from .frames import FrameRepository
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult


# ----------------------------- 配置常量 -----------------------------

_BASELINE_DENSE: Dict[str, float] = {
    "grid_iou": 0.10,
    "grid_tiou": 0.12,
    "ncc_peak": 0.10,
    "peak_psr": 0.05,
    "peak_gap12": 0.05,
    "peak_center": 0.10,
    "peak_concentration": 0.10,
    "sps_similarity": 0.08,
}

_WEIGHTS_DENSE: Dict[str, float] = {
    "grid_iou": 0.18,
    "grid_tiou": 0.16,
    "ncc_peak": 0.14,
    "peak_psr": 0.08,
    "peak_gap12": 0.06,
    "peak_center": 0.05,
    "peak_concentration": 0.05,
    "sps_similarity": 0.28,
}

_BASELINE_SPARSE: Dict[str, float] = {
    "anchors_similarity": 0.10,
    "zncc_columns": 0.08,
    "frcs_similarity": 0.10,
    "sps_similarity": 0.08,
}

_WEIGHTS_SPARSE: Dict[str, float] = {
    "anchors_similarity": 0.40,
    "zncc_columns": 0.22,
    "frcs_similarity": 0.18,
    "sps_similarity": 0.20,
}


# ----------------------------- 主流程 -----------------------------


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

    a_norm = a.astype(np.float32) / 255.0
    b_norm = b.astype(np.float32) / 255.0

    mu = request.mu_sub / 255.0
    delta = max(request.delta_y / 255.0, 1.0 / 255.0)

    b_aligned, gain, bias = _align_brightness(a_norm, b_norm, mu)

    foreground_a = _foreground_likelihood(a_norm, mu, delta)
    foreground_b = _foreground_likelihood(b_aligned, mu, delta)

    bands = _find_text_bands(foreground_a)
    band_mask = _bands_to_mask(foreground_a.shape, bands)
    if not np.any(band_mask):
        bands = [(0, foreground_a.shape[0])]
        band_mask[:, :] = True

    fa_band = foreground_a * band_mask
    fb_band = foreground_b * band_mask

    search_radius = request.search_radius or config.search_radius
    dx_pix, dy_pix = _search_pixel_shift(fa_band, fb_band, band_mask, search_radius)

    b_shifted = _shift_image(b_aligned, dx_pix, dy_pix, fill=float(b_aligned.mean()))
    fb_shifted = _shift_image(foreground_b, dx_pix, dy_pix, fill=0.0)
    fb_band = fb_shifted * band_mask

    grid_a = _grid_ratio_map(foreground_a * band_mask, band_mask, bands, gx=24)
    grid_b = _grid_ratio_map(fb_shifted * band_mask, band_mask, bands, gx=24)

    r_cells = _radius_in_cells(search_radius, a_norm.shape[1], grid_a.shape[1])
    grid_metrics = _grid_metrics(grid_a, grid_b, r_cells)
    sps_sim = _band_spectrum_similarity(fa_band, fb_band, k=16)

    rho = _foreground_density(b_shifted, band_mask, mu, delta)
    entropy = _grid_entropy(grid_b)

    details: Dict[str, float] = {
        "gain": gain,
        "bias": bias,
        "dx": float(dx_pix),
        "dy": float(dy_pix),
        "rho": rho,
        "grid_entropy": entropy,
        "search_radius": float(search_radius),
        "r_cells": float(r_cells),
    }

    if rho >= 0.10 and entropy >= 0.55:
        branch = "dense"
        metrics_raw = _compose_dense_metrics(grid_metrics, sps_sim)
        score, metrics_norm = _fuse_metrics(metrics_raw, _BASELINE_DENSE, _WEIGHTS_DENSE)
        details["branch_dense"] = 1.0
        details.update(_extract_dense_details(grid_metrics))
    else:
        branch = "sparse"
        components_a = _components_in_bands(foreground_a, band_mask)
        components_b = _components_in_bands(fb_shifted, band_mask)
        anchor_sim = _anchors_match(components_a, components_b, width=a_norm.shape[1])
        zncc_sim = _zncc_column_profile(fa_band, fb_band)
        frcs_sim = _frcs_similarity(fa_band, fb_band, band_mask)
        metrics_raw = _compose_sparse_metrics(anchor_sim, zncc_sim, frcs_sim, sps_sim)
        score, metrics_norm = _fuse_metrics(metrics_raw, _BASELINE_SPARSE, _WEIGHTS_SPARSE)
        details["branch_dense"] = 0.0
        details["anchors_raw"] = anchor_sim
        details["zncc_raw"] = zncc_sim
        details["frcs_raw"] = frcs_sim
        details["component_count_a"] = float(len(components_a))
        details["component_count_b"] = float(len(components_b))

    decision = _decision_from_score(score)

    return SubtitleSimilarityResult(
        score=score,
        confidence=score,
        decision=decision,
        roi=roi,
        metrics=metrics_norm,
        details=details,
        delta=(dx_pix, dy_pix),
    )


# ----------------------------- 亮度与前景 -----------------------------


def _align_brightness(ref: np.ndarray, target: np.ndarray, mu_sub: float) -> Tuple[np.ndarray, float, float]:
    def _robust_mean(image: np.ndarray) -> float:
        dist = np.abs(image - mu_sub)
        mask = dist <= 0.08
        if mask.sum() < image.size * 0.01:
            percentile = np.quantile(image, 0.98)
            mask = image >= percentile
        if mask.any():
            return float(image[mask].mean())
        return float(image.mean())

    mu_ref = _robust_mean(ref)
    mu_target = _robust_mean(target)

    gain = np.clip(mu_ref / (mu_target + 1e-6), 0.80, 1.20)
    bias = np.clip(mu_ref - gain * mu_target, -0.15, 0.15)
    aligned = np.clip(gain * target + bias, 0.0, 1.0)
    return aligned.astype(np.float32), float(gain), float(bias)


def _foreground_likelihood(image: np.ndarray, mu_sub: float, delta: float) -> np.ndarray:
    delta_f = float(min(delta, 0.08))
    sigma = max(delta_f / 2.5, 0.01)
    values = image.reshape(-1)
    clip_hi = np.quantile(values, 0.99)
    bg_values = values[values <= clip_hi]
    if bg_values.size < 128:
        bg_values = values
    hist, bin_edges = np.histogram(bg_values, bins=64, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32) + 1e-6
    pdf_bg = hist / hist.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    dist = np.abs(image - mu_sub)
    p_fg = np.exp(-0.5 * (dist / sigma) ** 2) + 1e-6
    p_fg = p_fg.astype(np.float32)

    bin_idx = np.minimum(np.searchsorted(bin_edges, image, side="right") - 1, pdf_bg.size - 1)
    p_bg = pdf_bg[bin_idx]
    llr = np.log(p_fg / p_bg)
    kappa = 3.0
    likelihood = 1.0 / (1.0 + np.exp(-kappa * llr))
    return likelihood.astype(np.float32)


# ----------------------------- 行带与网格 -----------------------------


def _find_text_bands(foreground: np.ndarray, max_bands: int = 2) -> List[Tuple[int, int]]:
    projection = foreground.sum(axis=1)
    if float(projection.sum()) <= 1e-6:
        return [(0, foreground.shape[0])]

    sigma = max(1.5, foreground.shape[0] / 120.0)
    radius = int(np.ceil(3 * sigma))
    kernel = _gaussian_kernel1d(radius * 2 + 1, sigma)
    padded = np.pad(projection, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius]

    threshold = smoothed.max() * 0.3
    if threshold <= 0.0:
        return [(0, foreground.shape[0])]

    segments: List[Tuple[int, int, float]] = []
    in_segment = False
    start = 0
    for idx, value in enumerate(smoothed):
        if value >= threshold and not in_segment:
            start = idx
            in_segment = True
        elif value < threshold and in_segment:
            score = float(smoothed[start:idx].sum())
            segments.append((start, idx, score))
            in_segment = False
    if in_segment:
        score = float(smoothed[start:].sum())
        segments.append((start, len(smoothed), score))

    if not segments:
        return [(0, foreground.shape[0])]

    segments.sort(key=lambda item: item[2], reverse=True)
    selected = segments[:max_bands]
    bands: List[Tuple[int, int]] = []
    height = foreground.shape[0]

    for start, end, _score in sorted(selected, key=lambda item: item[0]):
        center = 0.5 * (start + end)
        width = max(1, end - start)
        stroke = max(2, int(round(width / 2)))
        half = int(np.clip(2.2 * stroke / 2.0, 4, height // 2 if height >= 8 else height))
        y1 = int(max(0, center - half))
        y2 = int(min(height, center + half))
        if y2 <= y1:
            continue
        bands.append((y1, y2))

    if not bands:
        bands = [(0, foreground.shape[0])]

    merged: List[Tuple[int, int]] = []
    for y1, y2 in sorted(bands):
        if not merged:
            merged.append((y1, y2))
            continue
        py1, py2 = merged[-1]
        if y1 <= py2:
            merged[-1] = (py1, max(py2, y2))
        elif y1 - py2 < max(4, (py2 - py1)):
            merged[-1] = (py1, y2)
        else:
            merged.append((y1, y2))

    return merged[:max_bands]


def _bands_to_mask(shape: Tuple[int, int], bands: Sequence[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for y1, y2 in bands:
        mask[y1:y2, :] = True
    return mask


def _grid_ratio_map(
    foreground: np.ndarray,
    band_mask: np.ndarray,
    bands: Sequence[Tuple[int, int]],
    gx: int = 24,
) -> np.ndarray:
    height, width = foreground.shape
    if not bands:
        bands = [(0, height)]
    gy = max(2, 2 * len(bands))

    grid = np.zeros((gy, gx), dtype=np.float32)
    x_edges = np.linspace(0, width, gx + 1, dtype=int)
    row_idx = 0

    for y1, y2 in bands:
        row_edges = np.linspace(y1, y2, 3, dtype=int)
        for row in range(2):
            if row_idx >= gy:
                break
            ry1, ry2 = row_edges[row], row_edges[row + 1]
            ry1 = int(max(0, min(height, ry1)))
            ry2 = int(max(0, min(height, ry2)))
            if ry2 <= ry1:
                row_idx += 1
                continue
            cell_mask_row = band_mask[ry1:ry2, :]
            cell_foreground = foreground[ry1:ry2, :]
            for col in range(gx):
                cx1, cx2 = x_edges[col], x_edges[col + 1]
                if cx2 <= cx1:
                    continue
                mask_cell = cell_mask_row[:, cx1:cx2]
                if not mask_cell.any():
                    continue
                fg_cell = cell_foreground[:, cx1:cx2]
                grid[row_idx, col] = float(fg_cell[mask_cell].mean())
            row_idx += 1

    return grid


def _radius_in_cells(radius_px: int, width_px: int, gx: int) -> int:
    if radius_px <= 0:
        return 0
    cell_width = max(width_px / max(gx, 1), 1.0)
    r_cells = int(np.ceil(radius_px / cell_width))
    return max(1, r_cells)


def _grid_metrics(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    radius: int,
) -> "GridMetrics":
    if radius <= 0:
        radius = 1

    ncc_map = _grid_ncc_map(grid_a, grid_b, radius)
    best_idx = np.unravel_index(int(np.argmax(ncc_map)), ncc_map.shape)
    best_dy = best_idx[0] - radius
    best_dx = best_idx[1] - radius

    grid_b_aligned = _shift_grid(grid_b, best_dx, best_dy, fill=0.0)

    tau = 0.3
    mask_a = grid_a >= tau
    mask_b = grid_b_aligned >= tau

    giou = _grid_iou(mask_a, mask_b)
    tgiou = _grid_tolerant_iou(mask_a, mask_b, expand=1)
    peak_features = _peak_table_features(ncc_map, radius)

    return GridMetrics(
        grid_iou=giou,
        grid_tiou=tgiou,
        best_dx=int(best_dx),
        best_dy=int(best_dy),
        ncc_map=ncc_map,
        peak=peak_features,
    )


@dataclass(slots=True)
class GridPeak:
    value: float
    psr: float
    gap12: float
    center_score: float
    concentration: float
    best_dx: int
    best_dy: int
    peaks: List[Tuple[int, int, float]]


@dataclass(slots=True)
class GridMetrics:
    grid_iou: float
    grid_tiou: float
    best_dx: int
    best_dy: int
    ncc_map: np.ndarray
    peak: GridPeak


def _grid_ncc_map(grid_a: np.ndarray, grid_b: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    ncc = np.zeros((size, size), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_grid(grid_b, dx, dy, fill=0.0)
            ncc[dy + radius, dx + radius] = _normalized_cross_correlation(grid_a, shifted)
    return ncc


def _grid_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter / union)


def _grid_tolerant_iou(mask_a: np.ndarray, mask_b: np.ndarray, expand: int = 1) -> float:
    if expand <= 0:
        return _grid_iou(mask_a, mask_b)
    padded = np.pad(mask_b, expand, mode="constant", constant_values=False)
    windows = sliding_window_view(padded, (2 * expand + 1, 2 * expand + 1))
    dilated = windows.any(axis=(-1, -2))
    union = np.logical_or(mask_a, dilated).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(mask_a, dilated).sum()
    return float(inter / union)


def _peak_table_features(ncc_map: np.ndarray, radius: int, topk: int = 5) -> GridPeak:
    h, w = ncc_map.shape
    flat_idx = np.argsort(ncc_map.reshape(-1))[::-1]
    peaks: List[Tuple[int, int, float]] = []
    taken = np.zeros_like(ncc_map, dtype=bool)

    for idx in flat_idx:
        y = idx // w
        x = idx % w
        if taken[y, x]:
            continue
        value = float(ncc_map[y, x])
        peaks.append((x - radius, y - radius, value))
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)
        x0 = max(0, x - 1)
        x1 = min(w, x + 2)
        taken[y0:y1, x0:x1] = True
        if len(peaks) >= topk:
            break

    if not peaks:
        peaks = [(0, 0, 0.0)]

    best_dx, best_dy, best_value = peaks[0]

    sidelobe_mask = np.ones_like(ncc_map, dtype=bool)
    by = best_dy + radius
    bx = best_dx + radius
    y0 = max(0, by - 1)
    y1 = min(h, by + 2)
    x0 = max(0, bx - 1)
    x1 = min(w, bx + 2)
    sidelobe_mask[y0:y1, x0:x1] = False
    sidelobe = ncc_map[sidelobe_mask]
    if sidelobe.size == 0:
        psr = 0.0
    else:
        mean = float(np.mean(sidelobe))
        std = float(np.std(sidelobe) + 1e-6)
        psr = max(0.0, (best_value - mean) / std)

    gap12 = 0.0
    if len(peaks) >= 2:
        gap12 = max(0.0, best_value - peaks[1][2])

    sigma = max(radius / 1.5, 1.0)
    center_score = float(np.exp(-0.5 * ((best_dx / sigma) ** 2 + (best_dy / sigma) ** 2)))

    positive = np.maximum(ncc_map, 0.0)
    sum_positive = float(positive.sum() + 1e-6)
    sum_topk = float(sum(max(val, 0.0) for _dx, _dy, val in peaks))
    concentration = sum_topk / sum_positive if sum_positive > 0 else 0.0

    return GridPeak(
        value=float(best_value),
        psr=float(psr),
        gap12=float(gap12),
        center_score=center_score,
        concentration=float(np.clip(concentration, 0.0, 1.0)),
        best_dx=int(best_dx),
        best_dy=int(best_dy),
        peaks=peaks,
    )


# ----------------------------- 稀疏锚点 -----------------------------


def _components_in_bands(foreground: np.ndarray, band_mask: np.ndarray, threshold: float = 0.5) -> List[Tuple[float, float]]:
    mask = (foreground >= threshold) & band_mask
    if not mask.any():
        return []

    height, width = mask.shape
    min_area = max(9, (height * width) // 600)

    visited = np.zeros_like(mask, dtype=bool)
    components: List[Tuple[float, float]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            area = len(pixels)
            if area < min_area:
                continue
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            y1, y2 = min(ys), max(ys)
            x1, x2 = min(xs), max(xs)
            height_comp = y2 - y1 + 1
            width_comp = x2 - x1 + 1
            aspect = width_comp / max(height_comp, 1)
            if aspect < 0.2 or aspect > 6.0:
                continue
            band_top = np.argwhere(band_mask[:, x1:x2 + 1].any(axis=1))
            if band_top.size:
                top = band_top.min()
                bottom = band_top.max()
                if y1 <= top or y2 >= bottom:
                    continue
            center_x = float(np.mean(xs))
            components.append((center_x, float(width_comp)))

    components.sort(key=lambda item: item[0])
    return components


def _anchors_match(
    anchors_a: Sequence[Tuple[float, float]],
    anchors_b: Sequence[Tuple[float, float]],
    width: int,
) -> float:
    if not anchors_a or not anchors_b:
        return 0.0

    norm = max(width - 1, 1)
    pos_a = np.array([a[0] for a in anchors_a], dtype=np.float32) / norm
    pos_b = np.array([b[0] for b in anchors_b], dtype=np.float32) / norm
    wid_a = np.array([a[1] for a in anchors_a], dtype=np.float32) / norm
    wid_b = np.array([b[1] for b in anchors_b], dtype=np.float32) / norm

    m, n = pos_a.size, pos_b.size
    window = max(1, int(np.ceil(max(m, n) * 0.1)))
    large = 1e9
    dp = np.full((m + 1, n + 1), large, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, m + 1):
        j_start = max(1, i - window)
        j_end = min(n, i + window)
        for j in range(j_start, j_end + 1):
            cost = 0.7 * abs(pos_a[i - 1] - pos_b[j - 1]) + 0.3 * abs(wid_a[i - 1] - wid_b[j - 1])
            best_prev = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = cost + best_prev

    total_cost = float(dp[m, n])
    if np.isinf(total_cost) or total_cost >= large:
        return 0.0

    avg_cost = total_cost / max(max(m, n), 1)
    score = float(np.exp(-avg_cost / 0.1))
    return np.clip(score, 0.0, 1.0)


def _zncc_column_profile(fa_band: np.ndarray, fb_band: np.ndarray) -> float:
    profile_a = fa_band.sum(axis=0)
    profile_b = fb_band.sum(axis=0)
    return _zncc(profile_a, profile_b)


def _frcs_similarity(fa_band: np.ndarray, fb_band: np.ndarray, band_mask: np.ndarray) -> float:
    ratio_a = _column_ratio(fa_band, band_mask)
    ratio_b = _column_ratio(fb_band, band_mask)
    bits_a = _frcs_bits(ratio_a, threshold=0.2, length=128)
    bits_b = _frcs_bits(ratio_b, threshold=0.2, length=128)
    if bits_a.size == 0 or bits_b.size == 0:
        return 0.0
    hamming = np.count_nonzero(bits_a != bits_b)
    return 1.0 - hamming / bits_a.size


def _column_ratio(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    numerator = values.sum(axis=0)
    denom = mask.sum(axis=0).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, numerator / (denom + 1e-6), 0.0)
    return ratio.astype(np.float32)


def _frcs_bits(profile: np.ndarray, threshold: float, length: int) -> np.ndarray:
    if profile.size == 0:
        return np.zeros(0, dtype=np.uint8)
    binary = (profile >= threshold).astype(np.uint8)
    if profile.size == length:
        return binary
    edges = np.linspace(0, binary.size, length + 1)
    bits = np.zeros(length, dtype=np.uint8)
    for idx in range(length):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end <= start:
            end = min(binary.size, start + 1)
        window = binary[start:end]
        bits[idx] = 1 if window.size and window.mean() >= 0.5 else 0
    return bits


# ----------------------------- 度量与融合 -----------------------------


def _compose_dense_metrics(metrics: GridMetrics, sps: float) -> Dict[str, float]:
    peak = metrics.peak
    return {
        "grid_iou": float(np.clip(metrics.grid_iou, 0.0, 1.0)),
        "grid_tiou": float(np.clip(metrics.grid_tiou, 0.0, 1.0)),
        "ncc_peak": 0.5 * (peak.value + 1.0),
        "peak_psr": 1.0 - np.exp(-max(peak.psr, 0.0) / 4.0),
        "peak_gap12": 1.0 - np.exp(-max(peak.gap12, 0.0) / 0.15),
        "peak_center": float(np.clip(peak.center_score, 0.0, 1.0)),
        "peak_concentration": float(np.clip(peak.concentration, 0.0, 1.0)),
        "sps_similarity": 0.5 * (np.clip(sps, -1.0, 1.0) + 1.0),
    }


def _compose_sparse_metrics(anchor: float, zncc: float, frcs: float, sps: float) -> Dict[str, float]:
    return {
        "anchors_similarity": float(np.clip(anchor, 0.0, 1.0)),
        "zncc_columns": 0.5 * (np.clip(zncc, -1.0, 1.0) + 1.0),
        "frcs_similarity": float(np.clip(frcs, 0.0, 1.0)),
        "sps_similarity": 0.5 * (np.clip(sps, -1.0, 1.0) + 1.0),
    }


def _fuse_metrics(
    metrics: Dict[str, float],
    baselines: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    normalized: Dict[str, float] = {}
    score = 0.0
    for name, value in metrics.items():
        baseline = baselines.get(name, 0.0)
        norm = _baseline_normalize(value, baseline)
        normalized[name] = norm
        weight = weights.get(name, 0.0)
        score += weight * norm
    return float(np.clip(score, 0.0, 1.0)), normalized


def _baseline_normalize(value: float, baseline: float) -> float:
    if value <= baseline:
        return 0.0
    denom = max(1.0 - baseline, 1e-6)
    return float(np.clip((value - baseline) / denom, 0.0, 1.0))


def _extract_dense_details(metrics: GridMetrics) -> Dict[str, float]:
    details: Dict[str, float] = {
        "grid_iou_raw": metrics.grid_iou,
        "grid_tiou_raw": metrics.grid_tiou,
        "grid_peak_dx": float(metrics.peak.best_dx),
        "grid_peak_dy": float(metrics.peak.best_dy),
        "grid_peak_value": metrics.peak.value,
        "grid_peak_psr": metrics.peak.psr,
        "grid_peak_gap": metrics.peak.gap12,
        "grid_peak_center": metrics.peak.center_score,
        "grid_peak_concentration": metrics.peak.concentration,
    }
    return details


# ----------------------------- 基础工具 -----------------------------


def _search_pixel_shift(
    fa: np.ndarray,
    fb: np.ndarray,
    band_mask: np.ndarray,
    radius: int,
) -> Tuple[int, int]:
    best_score = -1.0
    best = (0, 0)
    weight = band_mask.astype(np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_image(fb, dx, dy, fill=0.0)
            score = _weighted_ncc(fa, shifted, weight)
            if score > best_score:
                best_score = score
                best = (dx, dy)
    return best


def _foreground_density(image: np.ndarray, band_mask: np.ndarray, mu_sub: float, delta: float) -> float:
    delta_f = min(delta, 0.08)
    foreground = np.abs(image - mu_sub) <= delta_f
    denom = int(band_mask.sum())
    if denom == 0:
        return 0.0
    return float(np.count_nonzero(foreground & band_mask) / denom)


def _grid_entropy(grid: np.ndarray) -> float:
    total = float(grid.sum())
    if total <= 1e-6:
        return 0.0
    pi = grid / total
    pi = pi[pi > 0]
    entropy = -float(np.sum(pi * np.log(pi)))
    max_entropy = np.log(grid.size)
    if max_entropy <= 1e-6:
        return 0.0
    return float(entropy / max_entropy)


def _band_spectrum_similarity(fa_band: np.ndarray, fb_band: np.ndarray, k: int = 16) -> float:
    row_a = fa_band.sum(axis=1)
    row_b = fb_band.sum(axis=1)
    col_a = fa_band.sum(axis=0)
    col_b = fb_band.sum(axis=0)
    row_sim = _spectral_cosine(row_a, row_b, k)
    col_sim = _spectral_cosine(col_a, col_b, k)
    return 0.5 * (row_sim + col_sim)


def _spectral_cosine(v1: np.ndarray, v2: np.ndarray, k: int) -> float:
    v1 = v1.astype(np.float32) - float(np.mean(v1))
    v2 = v2.astype(np.float32) - float(np.mean(v2))
    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0.0
    spec1 = np.abs(np.fft.rfft(v1))
    spec2 = np.abs(np.fft.rfft(v2))
    spec1 = spec1[1 : 1 + k]
    spec2 = spec2[1 : 1 + k]
    if spec1.size == 0 or spec2.size == 0:
        return 0.0
    return _zncc(spec1, spec2)


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


def _gaussian_kernel1d(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _shift_image(image: np.ndarray, dx: int, dy: int, fill: float) -> np.ndarray:
    h, w = image.shape
    out = np.full_like(image, fill, dtype=np.float32)
    x_src_start = max(0, -dx)
    x_src_end = min(w, w - dx) if dx >= 0 else w
    y_src_start = max(0, -dy)
    y_src_end = min(h, h - dy) if dy >= 0 else h
    if x_src_end <= x_src_start or y_src_end <= y_src_start:
        return out
    x_dst_start = max(0, dx)
    y_dst_start = max(0, dy)
    out[y_dst_start : y_dst_start + (y_src_end - y_src_start), x_dst_start : x_dst_start + (x_src_end - x_src_start)] = image[
        y_src_start:y_src_end, x_src_start:x_src_end
    ]
    return out


def _shift_grid(grid: np.ndarray, dx: int, dy: int, fill: float) -> np.ndarray:
    h, w = grid.shape
    out = np.full_like(grid, fill, dtype=np.float32)
    x_src_start = max(0, -dx)
    x_src_end = min(w, w - dx) if dx >= 0 else w
    y_src_start = max(0, -dy)
    y_src_end = min(h, h - dy) if dy >= 0 else h
    if x_src_end <= x_src_start or y_src_end <= y_src_start:
        return out
    x_dst_start = max(0, dx)
    y_dst_start = max(0, dy)
    out[y_dst_start : y_dst_start + (y_src_end - y_src_start), x_dst_start : x_dst_start + (x_src_end - x_src_start)] = grid[
        y_src_start:y_src_end, x_src_start:x_src_end
    ]
    return out


def _weighted_ncc(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    weight = weight.astype(np.float32)
    s = float(weight.sum())
    if s < 1e-6:
        return 0.0
    a_mean = float((a * weight).sum() / s)
    b_mean = float((b * weight).sum() / s)
    a_centered = a - a_mean
    b_centered = b - b_mean
    a_var = float((weight * a_centered * a_centered).sum() / s)
    b_var = float((weight * b_centered * b_centered).sum() / s)
    if a_var <= 1e-8 or b_var <= 1e-8:
        return 0.0
    num = float((weight * a_centered * b_centered).sum() / s)
    return num / np.sqrt(a_var * b_var)


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    a_centered = a - a_mean
    b_centered = b - b_mean
    denom = float(np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered)))
    if denom <= 1e-8:
        return 0.0
    return float(np.sum(a_centered * b_centered) / denom)


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _decision_from_score(score: float) -> str:
    if score >= 0.70:
        return "same"
    if score <= 0.55:
        return "different"
    return "uncertain"

