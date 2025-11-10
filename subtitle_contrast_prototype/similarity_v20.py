# ========================= Patch v2 – Robust Similarity =========================
# 仅依赖 numpy，与现有 dataclass/模型/仓库接口兼容

from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .roi_utils import PreparedRoiPair, get_prepared_roi_pair
from .similarity_v11 import SubtitleSimilarityResult

# ------------------------- 基础工具：局部统计/高通/窗口 -------------------------

def _local_mean_std(image: np.ndarray, radius: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    k = radius * 2 + 1
    padded = np.pad(image, radius, mode="reflect")
    win = sliding_window_view(padded, (k, k))
    mean = win.mean(axis=(-1, -2))
    mean2 = (win * win).mean(axis=(-1, -2))
    var = np.maximum(mean2 - mean * mean, 0.0)
    std = np.sqrt(var + 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)

def _background_suppress(image: np.ndarray, radius: int = None) -> np.ndarray:
    h, w = image.shape
    if radius is None:
        radius = max(3, min(h, w) // 64)
    m, s = _local_mean_std(image, radius=radius)
    z = (image - m) / (s + 1e-3)           # 局部 z-score，增强笔画
    hp = 0.5 + 0.5 * np.tanh(z / 2.0)      # 压缩动态范围到 [0,1]
    return hp.astype(np.float32)

def _hann2d(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    return np.outer(wy, wx)

# ------------------------- 字幕概率（ΔY 自适应 + 对比度门控） -------------------------

def _subtitle_probability(image: np.ndarray, mu_sub: float, delta: float) -> np.ndarray:
    # ΔY 自适应（delta≈15/255≈0.06），delta 越大越降低亮度先验强度
    delta_eff = max(delta, 1.0 / 255.0)
    alpha0 = 8.0
    alpha = np.clip(alpha0 * (0.06 / (delta_eff + 1e-6)), 4.0, 12.0)

    dist = np.abs(image - mu_sub)
    p_mu = 1.0 / (1.0 + np.exp(-alpha * (delta_eff - dist) / delta_eff))

    m, s = _local_mean_std(image, radius=5)
    z = np.abs(image - m) / (s + 1e-3)
    p_contrast = np.tanh(z / 2.5)

    # ΔY 大时更依赖对比度（而非绝对亮度）
    w_mu = 0.8 / (1.0 + (delta_eff / 0.06)) + 0.2  # ∈[0.2,1.0)
    return (w_mu * p_mu + (1.0 - w_mu) * p_contrast).astype(np.float32)

# ------------------------- 形态学/掩码（沿用你的实现 + 小改） -------------------------

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
        if weight_bg <= 0: continue
        weight_fg = total - weight_bg
        if weight_fg <= 0: break
        mean_bg = sum_bg / max(weight_bg, 1e-6)
        mean_fg = (sum_total - sum_bg) / max(weight_fg, 1e-6)
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            best = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        sum_bg += ((bin_edges[idx] + bin_edges[idx + 1]) / 2) * count
    return float(best)

def _morph_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return mask
    size = radius * 2 + 1
    padded = np.pad(mask, radius, mode="constant", constant_values=True)
    windows = sliding_window_view(padded, (size, size))
    return np.all(windows, axis=(-1, -2))

def _morph_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return mask
    size = radius * 2 + 1
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = sliding_window_view(padded, (size, size))
    return np.any(windows, axis=(-1, -2))

def _morph_open(mask: np.ndarray, kernel_radius: int) -> np.ndarray:
    return _morph_dilate(_morph_erode(mask, kernel_radius), kernel_radius)

def _morph_close(mask: np.ndarray, kernel_radius: int) -> np.ndarray:
    return _morph_erode(_morph_dilate(mask, kernel_radius), kernel_radius)

def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1: return mask
    visited = np.zeros(mask.shape, dtype=bool)
    cleaned = mask.copy()
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]: continue
            stack = [(y, x)]
            comp = []
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]: continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) < min_area:
                for cy, cx in comp: cleaned[cy, cx] = False
    return cleaned

def _post_process_mask(prob: np.ndarray) -> np.ndarray:
    th = _otsu(prob)
    mask = prob >= th
    kr = max(1, prob.shape[0] // 80)
    mask = _morph_open(mask, kr)
    mask = _morph_close(mask, kr)
    mask = _remove_small_components(mask, max(9, (mask.size // 400)))
    return mask

# ------------------------- 梯度/边缘（在高通图上更稳） -------------------------

def _edge_strength(image_hp: np.ndarray) -> np.ndarray:
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    ky = kx.T
    padded = np.pad(image_hp, 1, mode="edge")
    win = sliding_window_view(padded, (3,3))
    gx = np.sum(win * kx, axis=(-1,-2))
    gy = np.sum(win * ky, axis=(-1,-2))
    mag = np.hypot(gx, gy)
    m = mag.max()
    return mag / m if m > 1e-6 else np.zeros_like(mag)

# ------------------------- 对齐：粗对齐 + 位移搜索的 NCC 地图 -------------------------

def _shift_image(image: np.ndarray, dx: int, dy: int, fill: float) -> np.ndarray:
    h, w = image.shape
    out = np.full_like(image, fill)
    x_src_start = max(0, -dx); x_src_end = min(w, w - dx) if dx >= 0 else w
    y_src_start = max(0, -dy); y_src_end = min(h, h - dy) if dy >= 0 else h
    if x_src_end <= x_src_start or y_src_end <= y_src_start: return out
    x_dst_start = max(0, dx); y_dst_start = max(0, dy)
    out[y_dst_start:y_dst_start+(y_src_end-y_src_start),
        x_dst_start:x_dst_start+(x_src_end-x_src_start)] = image[y_src_start:y_src_end, x_src_start:x_src_end]
    return out

def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return _shift_image(mask.astype(np.float32), dx, dy, fill=0.0) >= 0.5

def _coarse_shift(mask_a: np.ndarray, mask_b: np.ndarray, radius: int) -> Tuple[int,int]:
    best_score, best = -1.0, (0,0)
    a = mask_a.astype(np.float32); b = mask_b.astype(np.float32)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            shifted = _shift_image(b, dx, dy, 0.0)
            score = float(np.sum(a * shifted))
            if score > best_score:
                best_score, best = score, (dx, dy)
    return best

# ------------------------- 峰表：加权 NCC 地图 + Top-K 峰特征 -------------------------

def _weighted_ncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    w = np.clip(w, 0.0, 1.0).astype(np.float32)
    s = float(w.sum())
    if s < 1e-3: return 0.0
    am = float((a * w).sum() / s)
    bm = float((b * w).sum() / s)
    ax = a - am
    bx = b - bm
    av = float((w * ax * ax).sum() / s)
    bv = float((w * bx * bx).sum() / s)
    if av <= 1e-9 or bv <= 1e-9: return 0.0
    num = float((w * ax * bx).sum() / s)
    return num / np.sqrt(av * bv)

def _ncc_map_with_peaks(a_hp: np.ndarray, b_hp: np.ndarray, w_text: np.ndarray, radius: int, topk: int = 5
                       ) -> Tuple[np.ndarray, List[Tuple[int,int,float]], float, float]:
    # 构造 (2r+1)x(2r+1) 的位移 NCC 地图（加权）
    r = radius
    H = 2*r + 1
    ncc = np.zeros((H, H), dtype=np.float32)
    for j, dy in enumerate(range(-r, r+1)):
        for i, dx in enumerate(range(-r, r+1)):
            bs = _shift_image(b_hp, dx, dy, float(b_hp.mean()))
            ws = _shift_image(w_text, dx, dy, 0.0)
            ncc[j, i] = _weighted_ncc(a_hp, bs, ws)

    # 峰表（Top-K 局部极大值）
    peaks: List[Tuple[int,int,float]] = []
    temp = ncc.copy()
    for _ in range(topk):
        idx = int(np.argmax(temp))
        val = float(temp.flat[idx])
        y = idx // temp.shape[1]; x = idx % temp.shape[1]
        dy = y - r; dx = x - r
        peaks.append((dx, dy, val))
        # 非极大值抑制（3x3）
        y0, y1 = max(0, y-1), min(temp.shape[0], y+2)
        x0, x1 = max(0, x-1), min(temp.shape[1], x+2)
        temp[y0:y1, x0:x1] = -np.inf

    # PSR（剔除主峰邻域）
    flat = ncc.ravel()
    idx_max = int(np.argmax(flat))
    mask = np.ones_like(flat, dtype=bool)
    W = 2  # 主峰5x5邻域
    h, w = ncc.shape
    my = idx_max // w; mx = idx_max % w
    for yy in range(max(0, my-W), min(h, my+W+1)):
        for xx in range(max(0, mx-W), min(w, mx+W+1)):
            mask[yy*w + xx] = False
    side = flat[mask]
    if side.size == 0:
        psr = 0.0
    else:
        psr = (float(ncc[my, mx]) - float(np.mean(side))) / (float(np.std(side)) + 1e-6)

    # 峰值集中度（能量集中到 Top-K 的程度）
    top_vals = np.array([p[2] for p in peaks], dtype=np.float32)
    top_energy = float(np.sum(np.maximum(top_vals, 0.0)))
    all_energy = float(np.sum(np.maximum(ncc, 0.0)))
    concentration = top_energy / (all_energy + 1e-6)

    return ncc, peaks, psr, concentration

# ------------------------- 频谱签名（平移不敏感）/ EOH（网格梯度直方图） -------------------------

def _projection_spectrum_signature(mask: np.ndarray, k_bins: int = 16) -> np.ndarray:
    # 行/列投影 -> rFFT 幅度（去 DC）-> 取前 k_bins 合并
    def _spec(v):
        v = v.astype(np.float32)
        mag = np.abs(np.fft.rfft(v))  # len = floor(n/2)+1
        if mag.size > 0: mag[0] = 0.0 # 去 DC
        if mag.size < k_bins:
            pad = np.zeros(k_bins, dtype=np.float32); pad[:mag.size] = mag
            mag = pad
        else:
            mag = mag[:k_bins]
        return mag
    proj_h = mask.sum(axis=1)  # (H,)
    proj_v = mask.sum(axis=0)  # (W,)
    s_h = _spec(proj_h)
    s_v = _spec(proj_v)
    sig = np.concatenate([s_h, s_v], axis=0)
    n = np.linalg.norm(sig) + 1e-6
    return (sig / n).astype(np.float32)

def _eoh_signature(image_hp: np.ndarray, w_text: np.ndarray, grid: Tuple[int,int] = (8,8), bins: int = 8) -> np.ndarray:
    # 计算梯度方向直方图（0..π），按网格统计，权重=幅值*文本权重
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    ky = kx.T
    pad = np.pad(image_hp, 1, mode="edge")
    win = sliding_window_view(pad, (3,3))
    gx = np.sum(win * kx, axis=(-1,-2))
    gy = np.sum(win * ky, axis=(-1,-2))
    mag = np.hypot(gx, gy).astype(np.float32)
    ang = (np.arctan2(np.abs(gy), np.abs(gx)) % np.pi).astype(np.float32)  # 0..π

    H, W = image_hp.shape
    gy_n, gx_n = grid
    cell_h = max(1, H // gy_n)
    cell_w = max(1, W // gx_n)
    hist_all = []

    w = np.clip(w_text, 0.0, 1.0).astype(np.float32)
    weight = mag * (0.5 + 0.5 * w)  # 文本区域更大权重

    for cy in range(gy_n):
        for cx in range(gx_n):
            y0 = cy * cell_h; y1 = H if cy == gy_n - 1 else (y0 + cell_h)
            x0 = cx * cell_w; x1 = W if cx == gx_n - 1 else (x0 + cell_w)
            cell_ang = ang[y0:y1, x0:x1].ravel()
            cell_wt  = weight[y0:y1, x0:x1].ravel()
            # 直方图
            hist, _ = np.histogram(cell_ang, bins=bins, range=(0.0, np.pi), weights=cell_wt)
            hist = hist.astype(np.float32)
            # L2 归一
            hist /= (np.linalg.norm(hist) + 1e-6)
            hist_all.append(hist)
    sig = np.concatenate(hist_all, axis=0)
    sig /= (np.linalg.norm(sig) + 1e-6)
    return sig.astype(np.float32)

def _cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    n = float(np.linalg.norm(x) * np.linalg.norm(y))
    if n <= 1e-9: return 0.0
    return float(np.dot(x, y) / n)

# ------------------------- 容错 IoU、Masked-SSIM、归一/权重 -------------------------

def _tolerant_iou(mask_a: np.ndarray, mask_b: np.ndarray, radius: int = 1) -> float:
    if radius <= 0:
        inter = np.logical_and(mask_a, mask_b).sum()
        uni = np.logical_or(mask_a, mask_b).sum()
        return float(inter / uni) if uni > 0 else 0.0
    a = _morph_dilate(mask_a, radius)
    b = _morph_dilate(mask_b, radius)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni > 0 else 0.0

def _ssim_masked(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    w = np.clip(w.astype(np.float32), 0.0, 1.0)
    sw = float(w.sum())
    if sw < 1e3: return 0.0
    c1 = 0.01**2; c2 = 0.03**2
    wa = w * a; wb = w * b
    mu_a = float(wa.sum() / sw); mu_b = float(wb.sum() / sw)
    va = float(((w * (a - mu_a)**2).sum()) / sw)
    vb = float(((w * (b - mu_b)**2).sum()) / sw)
    cov = float((w * (a - mu_a) * (b - mu_b)).sum() / sw)
    num = (2*mu_a*mu_b + c1) * (2*cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return 0.0 if den <= 0 else float(np.clip(num/den, 0.0, 1.0))

def _baseline_normalize(sim: float, baseline: float) -> float:
    # 去偏：对负样本“随机相似度”进行抵消
    return float(np.clip((sim - baseline) / max(1e-6, 1.0 - baseline), 0.0, 1.0))

# ------------------------- 亮度对齐（沿用你的，略微收紧） -------------------------

def _align_brightness(ref: np.ndarray, tgt: np.ndarray, mu_sub: float) -> Tuple[np.ndarray, float, float]:
    def _robust_mean(im: np.ndarray) -> float:
        dist = np.abs(im - mu_sub)
        mask = dist <= 0.1
        if mask.sum() < im.size * 0.02:
            perc = np.quantile(im, 0.9)
            mask = im >= perc
        return float(im[mask].mean()) if mask.any() else float(im.mean())
    mu_r = _robust_mean(ref); mu_t = _robust_mean(tgt)
    gain = np.clip(mu_r / (mu_t + 1e-6), 0.80, 1.20)   # 稍收紧
    bias = np.clip(mu_r - gain * mu_t, -0.15, 0.15)
    aligned = np.clip(gain * tgt + bias, 0.0, 1.0)
    return aligned, float(gain), float(bias)

# ------------------------- 顶层：计算各特征并融合 -------------------------

def _compute_v2_features(
    a: np.ndarray, b: np.ndarray, mu: float, delta: float, search_radius: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    返回:
      raw_metrics:  原始指标
      norm_metrics: 归一到 [0,1] 的指标（含去偏）
      weights:      动态权重（用于融合）
    """
    # 高通
    a_hp = _background_suppress(a)
    b_hp = _background_suppress(b)

    # 软概率 + 边缘
    soft_a = _subtitle_probability(a, mu, delta)
    soft_b = _subtitle_probability(b, mu, delta)
    edges_a = _edge_strength(a_hp)
    edges_b = _edge_strength(b_hp)
    # 生成概率图：lambda_edge 0.6~0.7 更稳（这里固定到 0.65；如有 config 可外部传入）
    lam = 0.65
    prob_a = lam * soft_a + (1.0 - lam) * edges_a
    prob_b = lam * soft_b + (1.0 - lam) * edges_b
    mask_a = _post_process_mask(prob_a)
    mask_b = _post_process_mask(prob_b)

    # 先做粗平移对齐用于 IoU/投影/权重
    dx, dy = _coarse_shift(mask_a, mask_b, search_radius)
    mask_b_s = _shift_mask(mask_b, dx, dy)
    b_s = _shift_image(b, dx, dy, float(b.mean()))
    b_hp_s = _shift_image(b_hp, dx, dy, float(b_hp.mean()))
    soft_b_s = _shift_image(soft_b, dx, dy, 0.0)

    # 文本权重（软掩码交集）
    w_text = np.minimum(soft_a, soft_b_s)

    # --- 几何与布局 ---
    inter = np.logical_and(mask_a, mask_b_s).sum()
    uni   = np.logical_or(mask_a, mask_b_s).sum()
    total = mask_a.sum() + mask_b_s.sum()
    iou   = float(inter / uni) if uni > 0 else 0.0
    dice  = float(2 * inter / total) if total > 0 else 0.0
    tiou  = _tolerant_iou(mask_a, mask_b_s, radius=1)

    # 投影相关（原始布局）
    def _proj_corr(ma, mb):
        def _norm(v):
            s = v.sum()
            return (v / s).astype(np.float32) if s > 0 else np.zeros_like(v, dtype=np.float32)
        ah = _norm(ma.sum(axis=1)); bh = _norm(mb.sum(axis=1))
        av = _norm(ma.sum(axis=0)); bv = _norm(mb.sum(axis=0))
        return 0.5 * (float(np.dot(ah, bh)) + float(np.dot(av, bv)))
    proj = _proj_corr(mask_a, mask_b_s)

    # --- 平移不敏感的频谱签名 ---
    sps_a = _projection_spectrum_signature(mask_a)
    sps_b = _projection_spectrum_signature(mask_b_s)
    sps_sim = _cosine_sim(sps_a, sps_b)

    # --- EOH/HOG-lite ---
    eoh_a = _eoh_signature(a_hp, w_text, grid=(8,8), bins=8)
    eoh_b = _eoh_signature(b_hp_s, w_text, grid=(8,8), bins=8)
    eoh_sim = _cosine_sim(eoh_a, eoh_b)

    # --- 峰表：NCC 地图 + Top-K 峰特征 ---
    ncc_map, peaks, psr, concentration = _ncc_map_with_peaks(a_hp, b_hp, w_text, radius=search_radius, topk=5)
    top1 = peaks[0][2] if len(peaks) > 0 else 0.0
    top2 = peaks[1][2] if len(peaks) > 1 else 0.0
    gap12 = max(0.0, top1 - top2)  # 峰间隙
    # 峰中心性（最佳峰越靠近 0 位移越好）
    cx, cy, cv = peaks[0] if len(peaks) > 0 else (0,0,0.0)
    center_penalty = np.sqrt(cx*cx + cy*cy) / max(1.0, float(search_radius))
    center_sim = 1.0 - float(np.clip(center_penalty, 0.0, 1.0))

    # --- Masked-SSIM（在文本权重内）
    ssim_m = _ssim_masked(a_hp, b_hp_s, w_text)

    # --- 背景差（用于自适应权重）
    bg_a = a[prob_a < 0.3]; bg_b = b[prob_b < 0.3]
    bg_gap = float(abs(bg_a.mean() - bg_b.mean())) if (bg_a.size and bg_b.size) else 0.0

    raw = {
        "overlap_iou": iou,
        "overlap_dice": dice,
        "overlap_tiou": tiou,
        "layout_projection": proj,
        "sps_similarity": sps_sim,
        "eoh_similarity": eoh_sim,
        "ncc_top1": top1,
        "ncc_gap12": gap12,
        "ncc_psr": psr,
        "ncc_concentration": concentration,
        "ncc_center": center_sim,
        "ssim_masked": ssim_m,
        "dx": float(dx), "dy": float(dy),
        "bg_gap": bg_gap,
    }

    # --- 归一/去偏 ---
    # 基线估计（经验）：不同字幕时的“随机相似度”偏置
    BASE = {
        "layout_projection": 0.10,
        "sps_similarity":   0.08,
        "eoh_similarity":   0.10,
        "ssim_masked":      0.05,
    }
    norm = {
        "overlap_iou": raw["overlap_iou"],
        "overlap_dice": raw["overlap_dice"],
        "overlap_tiou": raw["overlap_tiou"],
        "layout_projection": _baseline_normalize(raw["layout_projection"], BASE["layout_projection"]),
        "sps_similarity":   _baseline_normalize(raw["sps_similarity"],   BASE["sps_similarity"]),
        "eoh_similarity":   _baseline_normalize(raw["eoh_similarity"],   BASE["eoh_similarity"]),
        "ssim_masked":      _baseline_normalize(raw["ssim_masked"],      BASE["ssim_masked"]),
        # NCC 系列：值域在 [-1,1]，映射到 [0,1] 后再做形状约束
        "ncc_top1":         float(np.clip((raw["ncc_top1"] + 1.0) * 0.5, 0.0, 1.0)),
        "ncc_gap12":        float(np.clip(raw["ncc_gap12"], 0.0, 1.0)),  # 差值本身常在 [0,1]
        "ncc_psr":          float(1.0 - np.exp(-max(0.0, raw["ncc_psr"]) / 8.0)),
        "ncc_concentration":float(np.clip(raw["ncc_concentration"], 0.0, 1.0)),
        "ncc_center":       float(np.clip(raw["ncc_center"], 0.0, 1.0)),
    }

    # --- 动态权重（ΔY 大或 bg 差大 → 更依赖形状/投影/峰表） ---
    def _weights(delta_norm: float, bg_gap: float) -> Dict[str, float]:
        w = {
            "overlap_iou": 0.18, "overlap_dice": 0.08, "overlap_tiou": 0.16,
            "layout_projection": 0.12, "sps_similarity": 0.12, "eoh_similarity": 0.12,
            "ssim_masked": 0.06,
            "ncc_top1": 0.05, "ncc_gap12": 0.04, "ncc_psr": 0.04,
            "ncc_concentration": 0.02, "ncc_center": 0.01
        }
        if (delta_norm >= 0.06) or (bg_gap >= 0.08):
            # 加大几何/频谱/布局权重，降低对亮度结构的依赖
            w.update({
                "overlap_iou": 0.19, "overlap_dice": 0.07, "overlap_tiou": 0.19,
                "layout_projection": 0.14, "sps_similarity": 0.15, "eoh_similarity": 0.15,
                "ssim_masked": 0.04,
                "ncc_top1": 0.04, "ncc_gap12": 0.03, "ncc_psr": 0.03,
                "ncc_concentration": 0.01, "ncc_center": 0.01
            })
        s = sum(w.values())
        for k in w: w[k] = w[k] / s
        return w

    weights = _weights(delta, bg_gap)
    return raw, norm, weights

# ------------------------- 对外：compute_similarity（保持签名不变） -------------------------

def compute_similarity(config, repository, request, prepared: PreparedRoiPair | None = None) -> "SubtitleSimilarityResult":
    roi_data = prepared or get_prepared_roi_pair(repository, request)
    roi = roi_data.roi
    a = roi_data.frame_a
    b = roi_data.frame_b
    if a.size == 0 or b.size == 0:
        raise ValueError("ROI is empty after clamping to frame bounds.")

    a_norm = a.astype(np.float32) / 255.0
    b_norm = b.astype(np.float32) / 255.0

    mu = request.mu_sub / 255.0
    delta = max(request.delta_y / 255.0, 1.0 / 255.0)

    # 亮度对齐（v2 略收紧）
    b_aligned, gain, bias = _align_brightness(a_norm, b_norm, mu)

    # v2 特征 + 动态权重融合
    raw, norm, weights = _compute_v2_features(
        a=a_norm, b=b_aligned, mu=mu, delta=delta,
        search_radius=(request.search_radius or getattr(config, "search_radius", 3))
    )

    score = float(sum(norm[k] * weights.get(k, 0.0) for k in norm.keys()))

    # v2 的决策阈值：同保守 0.70 / 0.55；如需抬高负样本裕度，可把下阈升到 0.58
    def _decision_from_score(score: float) -> str:
        if score >= 0.70: return "same"
        if score <= 0.55: return "different"
        return "uncertain"

    decision = _decision_from_score(score)

    # 兼容原返回（metrics 用归一相似度，details 存原值/对齐/亮度信息）
    return SubtitleSimilarityResult(
        score=score,
        confidence=score,
        decision=decision,
        roi=roi,
        metrics=norm,
        details={
            **raw,
            "gain": float(gain), "bias": float(bias),
            "delta_norm": float(delta)
        },
        delta=(int(raw.get("dx", 0)), int(raw.get("dy", 0))),
    )

# ------------------------- 你现有的辅助：ROI clamp（保持不变） -------------------------

def _clamp_roi(roi, width: int, height: int):
    if width <= 0 or height <= 0:
        return type(roi)(x=0, y=0, width=1, height=1)

    w = max(1, min(int(roi.width), width))
    h = max(1, min(int(roi.height), height))

    x = int(min(max(roi.x, 0), width - 1))
    y = int(min(max(roi.y, 0), height - 1))

    if x + w > width:
        x = max(0, width - w)
    if y + h > height:
        y = max(0, height - h)

    return type(roi)(x=x, y=y, width=w, height=h)
# ===============================================================================
