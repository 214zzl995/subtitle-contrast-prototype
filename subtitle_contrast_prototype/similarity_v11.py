from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, Field, validator

from .config import AppConfig
from .frames import FrameRepository


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Roi(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class SubtitleSimilarityRequest(BaseModel):
    frame_a: str
    frame_b: str
    roi: Roi
    mu_sub: float = Field(..., ge=0.0, le=255.0)
    delta_y: float = Field(..., ge=0.0, le=255.0)
    search_radius: int = Field(default=3, ge=0)
    version: str | None = Field(default="v1.1")

    @validator("frame_a", "frame_b")
    def validate_suffix(cls, value: str) -> str:
        if not value.lower().endswith(".yuv"):
            value = f"{value}.yuv"
        return value


@dataclass(slots=True)
class SubtitleSimilarityResult:
    score: float
    confidence: float
    decision: str
    roi: Roi
    metrics: Dict[str, float]
    details: Dict[str, float]
    delta: Tuple[int, int]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


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

    a_hp = _background_suppress(a_norm)
    b_hp = _background_suppress(b_aligned)

    soft_a = _subtitle_probability(a_norm, mu, delta)
    soft_b = _subtitle_probability(b_aligned, mu, delta)

    edges_a = _edge_strength(a_hp)
    edges_b = _edge_strength(b_hp)

    prob_a = config.lambda_edge * soft_a + (1.0 - config.lambda_edge) * edges_a
    prob_b = config.lambda_edge * soft_b + (1.0 - config.lambda_edge) * edges_b

    mask_a = _post_process_mask(prob_a)
    mask_b = _post_process_mask(prob_b)

    dx, dy = _coarse_shift(mask_a, mask_b, request.search_radius or config.search_radius)
    mask_b_shifted = _shift_mask(mask_b, dx, dy)
    b_shifted = _shift_image(b_aligned, dx, dy, fill=float(b_aligned.mean()))
    b_hp_shifted = _shift_image(b_hp, dx, dy, fill=float(b_hp.mean()))
    soft_b_shifted = _shift_image(soft_b, dx, dy, fill=0.0)

    weight_text = np.minimum(soft_a, soft_b_shifted)

    raw_metrics = _compute_metrics(
        mask_a,
        mask_b_shifted,
        a_hp,
        b_hp_shifted,
        weight_text=weight_text,
        a_raw=a_norm,
        b_raw=b_shifted,
    )
    normalized_metrics = _normalize_metrics(raw_metrics)

    bg_a = a_norm[prob_a < 0.3]
    bg_b = b_aligned[prob_b < 0.3]
    bg_gap = float(abs(bg_a.mean() - bg_b.mean())) if (bg_a.size and bg_b.size) else 0.0

    weights = _metric_weights(delta, bg_gap)
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
            "bg_gap": bg_gap,
            "delta_norm": delta,
        },
        delta=(dx, dy),
    )


def _clamp_roi(roi: Roi, width: int, height: int) -> Roi:
    x = min(max(0, roi.x), width - 1)
    y = min(max(0, roi.y), height - 1)
    w = max(1, min(roi.width, width - x))
    h = max(1, min(roi.height, height - y))
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


def _local_mean_std(image: np.ndarray, radius: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    k = radius * 2 + 1
    padded = np.pad(image, radius, mode="reflect")
    windows = sliding_window_view(padded, (k, k))
    mean = windows.mean(axis=(-1, -2))
    mean_sq = (windows * windows).mean(axis=(-1, -2))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var + 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _background_suppress(image: np.ndarray, radius: int | None = None) -> np.ndarray:
    h, w = image.shape
    if radius is None:
        radius = max(3, min(h, w) // 64)
    mean, std = _local_mean_std(image, radius=radius)
    z = (image - mean) / (std + 1e-3)
    hp = 0.5 + 0.5 * np.tanh(z / 2.0)
    return hp.astype(np.float32)


def _hann2d(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    return np.outer(wy, wx)


def _subtitle_probability(image: np.ndarray, mu_sub: float, delta: float) -> np.ndarray:
    delta_eff = max(delta, 1.0 / 255.0)
    alpha = np.clip(8.0 * (0.06 / (delta_eff + 1e-6)), 4.0, 12.0)
    dist = np.abs(image - mu_sub)
    p_mu = 1.0 / (1.0 + np.exp(-alpha * (delta_eff - dist) / delta_eff))

    mean, std = _local_mean_std(image, radius=5)
    z = np.abs(image - mean) / (std + 1e-3)
    p_contrast = np.tanh(z / 2.5)

    w_mu = 0.8 / (1.0 + (delta_eff / 0.06)) + 0.2
    return (w_mu * p_mu + (1.0 - w_mu) * p_contrast).astype(np.float32)


def _ssim_masked(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    w = np.clip(weights.astype(np.float32), 0.0, 1.0)
    sw = float(w.sum())
    if sw < 1e3:
        return 0.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    wa = w * a
    wb = w * b
    mu_a = float(wa.sum() / sw)
    mu_b = float(wb.sum() / sw)
    va = float(((w * (a - mu_a) ** 2).sum()) / sw)
    vb = float(((w * (b - mu_b) ** 2).sum()) / sw)
    cov = float((w * (a - mu_a) * (b - mu_b)).sum() / sw)
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return 0.0 if denominator <= 0 else float(np.clip(numerator / denominator, 0.0, 1.0))


def _phase_only_correlation_weighted(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    eps = 1e-9
    w = np.clip(weights.astype(np.float32), 0.0, 1.0)
    if w.max() <= 0:
        return 0.0, 0.0
    win = _hann2d(a.shape)
    ww = np.sqrt(w) * win
    aa = (a - float(a.mean())) * ww
    bb = (b - float(b.mean())) * ww

    fa = np.fft.fft2(aa)
    fb = np.fft.fft2(bb)
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    cross /= np.where(magnitude < eps, 1.0, magnitude)
    corr = np.real(np.fft.ifft2(cross))

    peak = float(np.max(corr))
    flat = corr.ravel()
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


def _tolerant_iou(mask_a: np.ndarray, mask_b: np.ndarray, radius: int = 1) -> float:
    if radius <= 0:
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(intersection / union) if union > 0 else 0.0
    dilated_a = _morph_dilate(mask_a, radius)
    dilated_b = _morph_dilate(mask_b, radius)
    intersection = np.logical_and(dilated_a, dilated_b).sum()
    union = np.logical_or(dilated_a, dilated_b).sum()
    return float(intersection / union) if union > 0 else 0.0


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


def _compute_metrics(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    a_hp: np.ndarray,
    b_hp: np.ndarray,
    *,
    weight_text: np.ndarray,
    a_raw: np.ndarray,
    b_raw: np.ndarray,
) -> Dict[str, float]:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / total) if total > 0 else 0.0
    tiou = _tolerant_iou(mask_a, mask_b, radius=1)
    ssim = _ssim_masked(a_hp, b_hp, weight_text)
    projection = _projection_correlation(mask_a, mask_b)
    peak_raw, psr_raw = _phase_only_correlation(a_raw, b_raw)
    peak_w, psr_w = _phase_only_correlation_weighted(a_hp, b_hp, weight_text)

    return {
        "overlap_iou": iou,
        "overlap_dice": dice,
        "overlap_tiou": tiou,
        "structure_ssim": ssim,
        "layout_projection": projection,
        "alignment_peak_raw": peak_raw,
        "alignment_psr_raw": psr_raw,
        "alignment_peak": peak_w,
        "alignment_psr": psr_w,
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


def _metric_weights(delta_norm: float, bg_gap: float) -> Dict[str, float]:
    weights = {
        "overlap_iou": 0.23,
        "overlap_dice": 0.12,
        "overlap_tiou": 0.16,
        "structure_ssim": 0.14,
        "layout_projection": 0.15,
        "alignment_peak": 0.10,
        "alignment_psr": 0.10,
    }
    if (delta_norm >= 0.06) or (bg_gap >= 0.08):
        weights.update(
            {
                "overlap_iou": 0.24,
                "overlap_dice": 0.11,
                "overlap_tiou": 0.19,
                "structure_ssim": 0.12,
                "layout_projection": 0.19,
                "alignment_peak": 0.07,
                "alignment_psr": 0.08,
            }
        )
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    return weights


def _normalize_metrics(raw: Dict[str, float]) -> Dict[str, float]:
    normalized = {
        "overlap_iou": raw.get("overlap_iou", 0.0),
        "overlap_dice": raw.get("overlap_dice", 0.0),
        "overlap_tiou": raw.get("overlap_tiou", 0.0),
        "structure_ssim": raw.get("structure_ssim", 0.0),
        "layout_projection": raw.get("layout_projection", 0.0),
        "alignment_peak": _normalize_peak(raw.get("alignment_peak", 0.0)),
        "alignment_psr": _normalize_psr(raw.get("alignment_psr", 0.0)),
    }
    for key, value in normalized.items():
        normalized[key] = float(max(0.0, min(1.0, value)))
    return normalized


def _normalize_peak(value: float) -> float:
    # Empirically, useful peaks live between 0.1 and 1.0
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
