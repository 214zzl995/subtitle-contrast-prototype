from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import AppConfig
from .frames import FrameRepository
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult


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

    soft_a = _subtitle_probability(a_norm, mu, delta)
    soft_b = _subtitle_probability(b_aligned, mu, delta)

    edges_a = _edge_strength(a_norm)
    edges_b = _edge_strength(b_aligned)

    prob_a = config.lambda_edge * soft_a + (1.0 - config.lambda_edge) * edges_a
    prob_b = config.lambda_edge * soft_b + (1.0 - config.lambda_edge) * edges_b

    mask_a = _post_process_mask(prob_a)
    mask_b = _post_process_mask(prob_b)

    dx, dy = _coarse_shift(mask_a, mask_b, request.search_radius or config.search_radius)
    mask_b_shifted = _shift_mask(mask_b, dx, dy)
    b_shifted = _shift_image(b_aligned, dx, dy, fill=float(b_aligned.mean()))

    raw_metrics = _compute_metrics(mask_a, mask_b_shifted, a_norm, b_shifted)
    normalized_metrics = _normalize_metrics(raw_metrics)
    weights = _metric_weights()
    score = float(sum(normalized_metrics[name] * weights[name] for name in weights))
    decision = _decision_from_score(score)

    return SubtitleSimilarityResult(
        score=score,
        confidence=score,
        decision=decision,
        roi=roi,
        metrics=normalized_metrics,
        details={
            **raw_metrics,
            "gain": gain,
            "bias": bias,
            "dx": float(dx),
            "dy": float(dy),
        },
        delta=(dx, dy),
    )


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


def _align_brightness(ref: np.ndarray, target: np.ndarray, mu_sub: float) -> Tuple[np.ndarray, float, float]:
    def _robust_mean(image: np.ndarray) -> float:
        distances = np.abs(image - mu_sub)
        mask = distances <= 0.1
        if mask.sum() < image.size * 0.02:
            percentile = np.quantile(image, 0.9)
            mask = image >= percentile
        return float(image[mask].mean()) if mask.any() else float(image.mean())

    mu_ref = _robust_mean(ref)
    mu_target = _robust_mean(target)

    gain = np.clip(mu_ref / (mu_target + 1e-6), 0.75, 1.25)
    bias = np.clip(mu_ref - gain * mu_target, -0.2, 0.2)

    aligned = np.clip(gain * target + bias, 0.0, 1.0)
    return aligned, float(gain), float(bias)


def _subtitle_probability(image: np.ndarray, mu_sub: float, delta: float) -> np.ndarray:
    alpha = 8.0
    dist = np.abs(image - mu_sub)
    score = alpha * (delta - dist) / max(delta, 1e-6)
    return 1.0 / (1.0 + np.exp(-score))


def _edge_strength(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = kernel_x.T
    padded = np.pad(image, 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    gx = np.sum(windows * kernel_x, axis=(-1, -2))
    gy = np.sum(windows * kernel_y, axis=(-1, -2))
    magnitude = np.hypot(gx, gy)
    max_val = magnitude.max()
    return magnitude / max_val if max_val > 1e-6 else np.zeros_like(magnitude)


def _post_process_mask(prob: np.ndarray) -> np.ndarray:
    threshold = _otsu(prob)
    mask = prob >= threshold
    mask = _morph_open(mask, kernel_radius=max(1, prob.shape[0] // 80))
    mask = _morph_close(mask, kernel_radius=max(1, prob.shape[0] // 80))
    mask = _remove_small_components(mask, min_area=max(9, (mask.size // 400)))
    return mask


def _otsu(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    total = values.size
    sum_total = np.dot(hist, (bin_edges[:-1] + bin_edges[1:]) / 2)
    sum_bg = 0.0
    weight_bg = 0.0
    best = 0.5
    max_between = -1.0

    for idx, count in enumerate(hist):
        weight_bg += count
        if weight_bg <= 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            best = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        sum_bg += ((bin_edges[idx] + bin_edges[idx + 1]) / 2) * count

    return float(best)


def _morph_open(mask: np.ndarray, kernel_radius: int) -> np.ndarray:
    return _morph_dilate(_morph_erode(mask, kernel_radius), kernel_radius)


def _morph_close(mask: np.ndarray, kernel_radius: int) -> np.ndarray:
    return _morph_erode(_morph_dilate(mask, kernel_radius), kernel_radius)


def _morph_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    padded = np.pad(mask, radius, mode="constant", constant_values=True)
    windows = sliding_window_view(padded, (size, size))
    return np.all(windows, axis=(-1, -2))


def _morph_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = sliding_window_view(padded, (size, size))
    return np.any(windows, axis=(-1, -2))


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask

    visited = np.zeros(mask.shape, dtype=bool)
    cleaned = mask.copy()
    h, w = mask.shape

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            component = []
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) < min_area:
                for cy, cx in component:
                    cleaned[cy, cx] = False
    return cleaned


def _coarse_shift(mask_a: np.ndarray, mask_b: np.ndarray, radius: int) -> Tuple[int, int]:
    best_score = -1.0
    best = (0, 0)
    mask_a_f = mask_a.astype(np.float32)
    mask_b_f = mask_b.astype(np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_mask(mask_b_f, dx, dy)
            score = float(np.sum(mask_a_f * shifted))
            if score > best_score:
                best_score = score
                best = (dx, dy)
    return best


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = _shift_image(mask.astype(np.float32), dx, dy, fill=0.0)
    return shifted >= 0.5


def _shift_image(image: np.ndarray, dx: int, dy: int, fill: float) -> np.ndarray:
    h, w = image.shape
    output = np.full_like(image, fill)

    x_src_start = max(0, -dx)
    x_src_end = min(w, w - dx) if dx >= 0 else w
    y_src_start = max(0, -dy)
    y_src_end = min(h, h - dy) if dy >= 0 else h

    if x_src_end <= x_src_start or y_src_end <= y_src_start:
        return output

    x_dst_start = max(0, dx)
    y_dst_start = max(0, dy)

    output[
        y_dst_start : y_dst_start + (y_src_end - y_src_start),
        x_dst_start : x_dst_start + (x_src_end - x_src_start),
    ] = image[y_src_start:y_src_end, x_src_start:x_src_end]
    return output


def _compute_metrics(mask_a: np.ndarray, mask_b: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / total) if total > 0 else 0.0
    ssim = _ssim(a, b)
    projection = _projection_correlation(mask_a, mask_b)
    peak, psr = _phase_only_correlation(a, b)

    return {
        "overlap_iou": iou,
        "overlap_dice": dice,
        "structure_ssim": ssim,
        "layout_projection": projection,
        "alignment_peak_raw": peak,
        "alignment_psr_raw": psr,
    }


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    sigma_a = float(np.var(a))
    sigma_b = float(np.var(b))
    covariance = float(np.mean((a - mu_a) * (b - mu_b)))

    numerator = (2 * mu_a * mu_b + c1) * (2 * covariance + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    if denominator <= 0:
        return 0.0
    return float(max(0.0, min(1.0, numerator / denominator)))


def _projection_correlation(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    def _normalize(vector: np.ndarray) -> np.ndarray:
        total = vector.sum()
        if total == 0:
            return np.zeros_like(vector, dtype=np.float32)
        return (vector / total).astype(np.float32)

    proj_a_h = _normalize(mask_a.sum(axis=1))
    proj_b_h = _normalize(mask_b.sum(axis=1))
    proj_a_v = _normalize(mask_a.sum(axis=0))
    proj_b_v = _normalize(mask_b.sum(axis=0))

    corr_h = float(np.dot(proj_a_h, proj_b_h))
    corr_v = float(np.dot(proj_a_v, proj_b_v))
    return 0.5 * (corr_h + corr_v)


def _phase_only_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    eps = 1e-9
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    cross /= np.where(magnitude < eps, 1.0, magnitude)
    corr = np.real(np.fft.ifft2(cross))

    peak = float(np.max(corr))
    flat = corr.flatten()
    idx = int(np.argmax(flat))
    mask = np.ones_like(flat, dtype=bool)
    window = 9
    for offset in range(-window, window + 1):
        pos = idx + offset
        if 0 <= pos < flat.size:
            mask[pos] = False
    sidelobes = flat[mask]
    if sidelobes.size == 0:
        psr = 0.0
    else:
        mean = float(np.mean(sidelobes))
        std = float(np.std(sidelobes) + eps)
        psr = (peak - mean) / std
    return peak, psr


def _metric_weights() -> Dict[str, float]:
    return {
        "overlap_iou": 0.25,
        "overlap_dice": 0.15,
        "structure_ssim": 0.15,
        "layout_projection": 0.15,
        "alignment_peak": 0.15,
        "alignment_psr": 0.15,
    }


def _normalize_metrics(raw: Dict[str, float]) -> Dict[str, float]:
    normalized = {
        "overlap_iou": raw.get("overlap_iou", 0.0),
        "overlap_dice": raw.get("overlap_dice", 0.0),
        "structure_ssim": raw.get("structure_ssim", 0.0),
        "layout_projection": raw.get("layout_projection", 0.0),
        "alignment_peak": _normalize_peak(raw.get("alignment_peak_raw", 0.0)),
        "alignment_psr": _normalize_psr(raw.get("alignment_psr_raw", 0.0)),
    }
    for key, value in normalized.items():
        normalized[key] = float(max(0.0, min(1.0, value)))
    return normalized


def _normalize_peak(value: float) -> float:
    return max(0.0, min(1.0, (value - 0.1) / 0.9))


def _normalize_psr(value: float) -> float:
    if value <= 0:
        return 0.0
    return float(1.0 - np.exp(-value / 10.0))


def _decision_from_score(score: float) -> str:
    if score >= 0.7:
        return "same"
    if score <= 0.55:
        return "different"
    return "uncertain"
