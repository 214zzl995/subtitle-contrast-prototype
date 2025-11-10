from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

# ---- 与工程对齐的类型（保持原接口） -----------------------------------------
from .config import AppConfig
from .frames import FrameRepository
from .roi_utils import PreparedRoiPair, get_prepared_roi_pair
from .similarity_v11 import Roi, SubtitleSimilarityRequest, SubtitleSimilarityResult

# ---- 参数 -------------------------------------------------------------------
UPSAMPLE = 2                  # 2x 双线性
BOX_BG_RADIUS = 7             # 背景盒滤波半径（像素，放大后）
EDGE_P_PCT = 80               # 边阈：梯度分位
MIN_FOREGROUND_RATIO = 0.002  # 前景占比阈
HASH_SAMPLES = 64             # 行轮廓哈希采样
HAMMING_RATIO_TAU = 0.15      # 哈希阈
S_COV_MIN = 0.80              # 骨架覆盖率阈
S_MIN = 0.45                  # 最终分数阈（配合代价归一）
SWT_WIDTH_MIN = 2.0           # 骨架局部宽度窗（像素）
SWT_WIDTH_MAX = 6.0

EDGE_WEIGHT = 0.5
WIDTH_WEIGHT = 0.35
ANGLE_WEIGHT = 0.15

# ---- 主入口 -----------------------------------------------------------------
def compute_similarity(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
    prepared: PreparedRoiPair | None = None,
) -> SubtitleSimilarityResult:
    req_roi = request.roi
    mu = float(request.mu_sub)
    delta = max(float(request.delta_y), 1.0)

    roi_data = prepared or get_prepared_roi_pair(repository, request)
    roi = roi_data.roi
    a_raw = roi_data.frame_a
    b_raw = roi_data.frame_b
    if a_raw.size == 0 or b_raw.size == 0:
        return _different(roi, reason="roi_empty_after_prepare")

    # 1) 预处理：上采样 + 局部归一 + 背景抑制 + 颜色先验
    _, a_top, a_w = _preprocess(a_raw, mu, delta)
    _, b_top, b_w = _preprocess(b_raw, mu, delta)

    # 2) 候选前景：边缘 ∧ 先验；修复/去噪；骨架与背景距变
    a_mask = _foreground(a_top, a_w)
    b_mask = _foreground(b_top, b_w)

    if min(a_mask.sum(), b_mask.sum()) < MIN_FOREGROUND_RATIO * float(a_mask.size):
        return _different(roi, reason="foreground_too_sparse")

    a_mask = _morph_close(a_mask, k=3)
    b_mask = _morph_close(b_mask, k=3)
    a_mask = _fill_holes(a_mask)
    b_mask = _fill_holes(b_mask)
    a_mask = _remove_small_cc(a_mask, min_area=max(16, a_mask.size // 900))
    b_mask = _remove_small_cc(b_mask, min_area=max(16, b_mask.size // 900))
    a_mask = _stroke_width_gate(a_mask)
    b_mask = _stroke_width_gate(b_mask)

    a_skel = _skeletonize(a_mask)
    b_skel = _skeletonize(b_mask)
    a_dt_bg = _edt_bool(~a_mask)
    b_dt_bg = _edt_bool(~b_mask)

    # 3) 基线拉平（旋转）+ 4) 相位相关估平移对齐
    a_flat, ang_a, M_a = _baseline_flatten(a_mask)
    b_flat, ang_b, M_b = _baseline_flatten(b_mask)

    a_skel = _warp_affine_bool(a_skel, M_a)
    b_skel = _warp_affine_bool(b_skel, M_b)
    a_dt_bg = _warp_affine_float(a_dt_bg, M_a)
    b_dt_bg = _warp_affine_float(b_dt_bg, M_b)

    dx, dy = _phase_corr_shift(_normalize01(_distance(a_flat)), _normalize01(_distance(b_flat)))
    b_aligned = _translate_bool(b_flat, dx, dy)
    b_skel = _translate_bool(b_skel, dx, dy)
    b_dt_bg = _translate_float(b_dt_bg, dx, dy)

    # 5) 行轮廓哈希（粗判）
    h_a = _contour_hash(a_flat)
    h_b = _contour_hash(b_aligned)
    bits = len(h_a)
    ham = _hamming(h_a, h_b)
    tau = int(math.ceil(bits * HAMMING_RATIO_TAU))
    if ham > tau:
        return SubtitleSimilarityResult(
            score=0.0, confidence=0.0, decision="different", roi=roi,
            metrics={"hash_bits": float(bits), "hamming": float(ham), "hamming_tau": float(tau),
                     "s_cov": 0.0, "c_bar": float("inf"), "s_final": 0.0},
            details={"reason": "hash_reject", "angle_a_deg": math.degrees(ang_a), "angle_b_deg": math.degrees(ang_b),
                     "dx": float(dx), "dy": float(dy),
                     "roi_requested_x": float(req_roi.x), "roi_requested_y": float(req_roi.y),
                     "roi_requested_w": float(req_roi.width), "roi_requested_h": float(req_roi.height),
                     "roi_effective_w": float(roi.width), "roi_effective_h": float(roi.height)},
            delta=(int(round(dx/UPSAMPLE)), int(round(dy/UPSAMPLE))),
        )

    # 6) 骨架图匹配（精判）
    edges_a = _build_edges(a_skel, a_dt_bg)
    edges_b = _build_edges(b_skel, b_dt_bg)
    if not edges_a or not edges_b:
        return _different(roi, reason="empty_skeleton")

    s_cov, c_bar, matched, matched_len = _compare_edges(edges_a, edges_b)

    # 代价尺度归一，避免 exp(-c_bar) 把分数压成 0
    diag = math.hypot(a_skel.shape[0], a_skel.shape[1])
    c_scaled = c_bar / max(1.0, 0.035 * diag)
    s_final = float(s_cov * math.exp(-c_scaled)) if math.isfinite(c_bar) else 0.0
    decision = "same" if (s_final >= S_MIN and s_cov >= S_COV_MIN) else "different"

    metrics = {"hash_bits": float(bits), "hamming": float(ham), "hamming_tau": float(tau),
               "s_cov": float(np.clip(s_cov, 0, 1)), "c_bar": float(c_bar), "s_final": float(np.clip(s_final, 0, 1))}
    details = {"angle_a_deg": math.degrees(ang_a), "angle_b_deg": math.degrees(ang_b),
               "dx": float(dx), "dy": float(dy),
               "edges_a": float(len(edges_a)), "edges_b": float(len(edges_b)),
               "matched_edges": float(matched), "matched_length": float(matched_len),
               "roi_requested_x": float(req_roi.x), "roi_requested_y": float(req_roi.y),
               "roi_requested_w": float(req_roi.width), "roi_requested_h": float(req_roi.height),
               "roi_effective_w": float(roi.width), "roi_effective_h": float(roi.height)}

    return SubtitleSimilarityResult(
        score=metrics["s_final"], confidence=metrics["s_final"],
        decision=decision, roi=roi, metrics=metrics, details=details,
        delta=(int(round(dx/UPSAMPLE)), int(round(dy/UPSAMPLE))),
    )

def _snap_roi_to_belt(y_plane: np.ndarray, roi: Roi, mu: float, delta: float) -> Roi:
    """在固定 x/width 下，沿 y 搜索字幕带，返回新的 roi。"""
    H, W = y_plane.shape
    x0 = max(0, min(roi.x, W-1))
    x1 = max(x0+1, min(W, roi.x + roi.width))
    band_h = max(2, int(roi.height))

    corridor = y_plane[:, x0:x1].astype(np.uint8)
    _, top, w = _preprocess(corridor, mu, delta)
    energy_line = (top * w).sum(axis=1)  # 垂直能量

    Hup = energy_line.shape[0]
    win = max(2, band_h * UPSAMPLE)
    if Hup <= win:
        y_best_up = 0
    else:
        csum = np.concatenate([[0.0], energy_line.cumsum()])
        window_sum = csum[win:] - csum[:-win]
        y_best_up = int(np.argmax(window_sum))

    y_best = int(round(y_best_up / UPSAMPLE))
    y_best = max(0, min(H - band_h, y_best))
    return Roi(x=roi.x, y=y_best, width=roi.width, height=band_h)

# ---- 预处理 ---------------------------------------------------------------
def _preprocess(y_u8: np.ndarray, mu: float, delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = _resize_bilinear(y_u8, UPSAMPLE).astype(np.float32)

    # 局部归一： (I - mean) / (std + 1)
    mean, std = _mean_std_box(up, r=3)
    eq = _normalize01((up - mean) / (std + 1.0))

    # 背景抑制：顶帽 ≈ I - boxblur(I, r)
    bg = _box_blur(eq, r=BOX_BG_RADIUS)
    top = np.clip(eq - bg, 0.0, 1.0)

    # 文本似然：颜色靠近 mu，且 top-hat 异常高
    mu_map = np.exp(-np.abs(up - mu) / max(delta, 1.0))
    t_mean, t_std = _mean_std_box(top, r=3)
    texture = np.exp(-np.abs(top - t_mean) / (t_std + 1e-3))
    weight = mu_map * texture
    weight /= max(weight.max(), 1e-6)
    return eq, top, weight

def _foreground(top: np.ndarray, weight: np.ndarray) -> np.ndarray:
    gx = _conv2(top, np.array([[1,0,-1],[2,0,-2],[1,0,-1]], np.float32))
    gy = _conv2(top, np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32))
    mag = np.hypot(gx, gy)

    t = np.percentile(mag, EDGE_P_PCT)
    edge = mag >= max(t, 1e-3)

    smooth_w = _box_blur(weight, r=2)
    w_med = float(np.median(smooth_w[edge])) if np.any(edge) else float(np.median(smooth_w))
    th = max(0.15, min(0.35, w_med * 0.9))

    mask = edge & (smooth_w >= th)
    mask = _binary_open(mask, k=3)

    if mask.sum() < 0.0005 * mask.size:  # 兜底再放宽一次
        t2 = np.percentile(mag, 70)
        edge2 = mag >= max(t2, 1e-3)
        th2 = max(0.10, th * 0.6)
        mask = edge2 & (smooth_w >= th2)
        mask = _binary_open(mask, k=3)
    return mask

# ---- 形态学 / 连通域 / 填洞 -----------------------------------------------
def _binary_open(mask: np.ndarray, k: int) -> np.ndarray:
    return _dilate(_erode(mask, k), k)   # 先腐蚀后膨胀

def _morph_close(mask: np.ndarray, k: int) -> np.ndarray:
    return _erode(_dilate(mask, k), k)   # 先膨胀后腐蚀

def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    r = k//2
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-r, r+1):
        y0 = max(0, -dy); y1 = min(h, h-dy)
        for dx in range(-r, r+1):
            x0 = max(0, -dx); x1 = min(w, w-dx)
            out[y0:y1, x0:x1] |= mask[y0+dy:y1+dy, x0+dx:x1+dx]
    return out

def _erode(mask: np.ndarray, k: int) -> np.ndarray:
    r = k//2
    h, w = mask.shape
    out = np.ones_like(mask, dtype=bool)
    for dy in range(-r, r+1):
        y0 = max(0, -dy); y1 = min(h, h-dy)
        for dx in range(-r, r+1):
            x0 = max(0, -dx); x1 = min(w, w-dx)
            tmp = np.zeros_like(mask, dtype=bool)
            tmp[y0:y1, x0:x1] = mask[y0+dy:y1+dy, x0+dx:x1+dx]
            out &= tmp
    return out

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    q: List[Tuple[int,int]] = []
    for x in range(w):
        if not mask[0, x]:   q.append((0, x));   visited[0, x] = True
        if not mask[h-1, x]: q.append((h-1, x)); visited[h-1, x] = True
    for y in range(h):
        if not mask[y, 0]:   q.append((y, 0));   visited[y, 0] = True
        if not mask[y, w-1]: q.append((y, w-1)); visited[y, w-1] = True
    while q:
        y, x = q.pop()
        for ny in (y-1, y, y+1):
            for nx in (x-1, x, x+1):
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and not mask[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
    holes = (~mask) & (~visited)
    return mask | holes

def _remove_small_cc(mask: np.ndarray, min_area: int) -> np.ndarray:
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=np.int32)
    parent: List[int] = []
    size: List[int] = []

    def make_set() -> int:
        parent.append(len(parent)); size.append(0); return len(parent)-1
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def unite(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb: return
        if size[ra] < size[rb]: ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for y in range(h):
        for x in range(w):
            if not mask[y, x]: continue
            neighbors = []
            for ny in (y-1, y):
                for nx in (x-1, x, x+1):
                    if ny == y and nx == x: continue
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] != -1:
                        neighbors.append(labels[ny, nx])
            if not neighbors:
                lbl = make_set(); labels[y, x] = lbl; size[lbl] = 1
            else:
                lbl = neighbors[0]; labels[y, x] = lbl; size[find(lbl)] += 1
                for nb in neighbors[1:]:
                    unite(lbl, nb)

    keep = np.zeros(len(parent), dtype=bool)
    for i in range(len(parent)):
        if size[find(i)] >= min_area:
            keep[i] = True
    out = np.zeros_like(mask, dtype=bool)
    idx = labels >= 0
    out[idx] = keep[labels[idx]]
    return out

# ---- 骨架与宽度门控 ---------------------------------------------------------
def _stroke_width_gate(mask: np.ndarray) -> np.ndarray:
    if not mask.any(): return mask
    dt = _edt_bool(~mask)
    sk = _skeletonize(mask)
    widths = dt[sk] * 2.0
    if widths.size == 0: return mask
    med = float(np.median(widths[np.isfinite(widths)])) if widths.size else 0.0
    if SWT_WIDTH_MIN <= med <= SWT_WIDTH_MAX:
        return mask
    return mask  # 不做硬裁，避免过拟合

# ---- 对齐：基线拉平 + 相位相关平移 ------------------------------------------
def _baseline_flatten(mask: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    h, w = mask.shape
    xs, ys = [], []
    for x in range(w):
        col = mask[:, x]
        idx = np.flatnonzero(col)
        if idx.size:
            xs.append(x); ys.append(int(idx.max()))
    if len(xs) < 2:
        ang = 0.0
    else:
        ang = _robust_slope(np.asarray(xs, np.float32), np.asarray(ys, np.float32))
    M = _rotation_matrix(mask.shape, -ang)
    rot = _warp_affine_bool(mask, M)
    return rot, ang, M

def _phase_corr_shift(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    h, w = A.shape
    wy = np.hanning(h); wx = np.hanning(w)
    WA = A * wy[:, None] * wx[None, :]
    WB = B * wy[:, None] * wx[None, :]
    F1 = np.fft.fft2(WA); F2 = np.fft.fft2(WB)
    R = F1 * np.conj(F2); R /= np.maximum(np.abs(R), 1e-12)
    r = np.fft.ifft2(R).real
    peak = float(r.max())
    if not np.isfinite(peak) or peak < 1e-6:
        return 0.0, 0.0
    y0, x0 = np.unravel_index(np.argmax(r), r.shape)
    if y0 > h//2: y0 -= h
    if x0 > w//2: x0 -= w

    def subpix(arr, y, x):
        y1 = (y-1) % h; y2 = (y+1) % h
        x1 = (x-1) % w; x2 = (x+1) % w
        dy = (arr[y2, x] - arr[y1, x]) / (2*(arr[y2, x] - 2*arr[y, x] + arr[y1, x] + 1e-12))
        dx = (arr[y, x2] - arr[y, x1]) / (2*(arr[y, x2] - 2*arr[y, x] + arr[y, x1] + 1e-12))
        return float(x + dx), float(y + dy)
    x_sub, y_sub = subpix(r, y0, x0)
    return (-(x_sub)), (-(y_sub))  # 把 B 移到 A

# ---- 轮廓哈希 ---------------------------------------------------------------
def _contour_hash(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    top = np.full(w, -1, np.int32)
    bot = np.full(w, -1, np.int32)
    for x in range(w):
        col = mask[:, x]
        idx = np.flatnonzero(col)
        if idx.size:
            top[x] = int(idx.min()); bot[x] = int(idx.max())

    def interp(arr: np.ndarray) -> np.ndarray:
        xs = np.arange(w); valid = arr >= 0
        if valid.sum() < 2: return np.zeros_like(arr, np.float32)
        return np.interp(xs, xs[valid], arr[valid]).astype(np.float32)

    t = interp(top); b = interp(bot); hgt = np.maximum(b - t, 0.0)
    T = _sample_curve(t, HASH_SAMPLES)
    B = _sample_curve(b, HASH_SAMPLES)
    H = _sample_curve(hgt, HASH_SAMPLES)
    return np.concatenate([_diff_bits(T), _diff_bits(B), _diff_bits(H)])

def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    return int(np.count_nonzero(a[:n] ^ b[:n]))

# ---- 骨架图特征与匹配 -------------------------------------------------------
@dataclass(slots=True)
class EdgeFeature:
    points: np.ndarray  # (K,2) yx
    length: float
    width: float
    angle: float

def _build_edges(skel: np.ndarray, dt_bg: np.ndarray) -> List[EdgeFeature]:
    if not np.any(skel): return []
    deg = _degree_map(skel)
    nodes = set(map(tuple, np.argwhere(deg != 2)))
    visited: set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
    edges: List[EdgeFeature] = []

    def add(path: List[Tuple[int,int]]):
        if len(path) < 2: return
        coords = np.asarray(path, np.int32)
        feat = _edge_feature(coords, dt_bg)
        if feat.length >= 2.0: edges.append(feat)

    for node in nodes:
        for nbr in _neighbors_coords(skel, node):
            key = _pair_key(node, nbr)
            if key in visited: continue
            path = [node, nbr]; visited.add(key)
            prev, cur = node, nbr
            while True:
                if tuple(cur) in nodes and tuple(cur) != node: break
                nbrs = [n for n in _neighbors_coords(skel, cur) if n != prev]
                if not nbrs: break
                nxt = nbrs[0]; key = _pair_key(cur, nxt)
                if key in visited: break
                visited.add(key)
                path.append(nxt); prev, cur = cur, nxt
            add(path)

    # 环路补齐
    ring = skel & (deg == 2)
    coords = np.argwhere(ring)
    used = np.zeros(len(coords), bool)
    for i, start in enumerate(map(tuple, coords)):
        if used[i]: continue
        path = [start]; prev = start
        nbrs = _neighbors_coords(skel, start)
        if not nbrs: continue
        cur = nbrs[0]
        while cur != start:
            path.append(cur)
            nxts = [n for n in _neighbors_coords(skel, cur) if n != prev]
            if not nxts: break
            prev, cur = cur, nxts[0]
        add(path)

    return edges

def _edge_feature(points_yx: np.ndarray, dt_bg: np.ndarray) -> EdgeFeature:
    p = points_yx.astype(np.float32)
    if p.shape[0] < 2:
        return EdgeFeature(points=points_yx, length=0.0, width=0.0, angle=0.0)
    dif = np.diff(p, axis=0)
    seg = np.hypot(dif[:,0], dif[:,1])
    length = float(np.sum(seg))
    ys = np.clip(points_yx[:,0], 0, dt_bg.shape[0]-1)
    xs = np.clip(points_yx[:,1], 0, dt_bg.shape[1]-1)
    widths = dt_bg[ys, xs] * 2.0
    widths = widths[np.isfinite(widths)]
    width = float(np.median(widths)) if widths.size else 0.0
    vec = p[-1] - p[0]
    if np.linalg.norm(vec) < 1e-3:
        c = p - p.mean(axis=0)
        if c.shape[0] >= 2:
            _, _, vh = np.linalg.svd(c, full_matrices=False)
            direction = vh[0]; angle = math.atan2(float(direction[0]), float(direction[1]))
        else:
            angle = 0.0
    else:
        angle = math.atan2(float(vec[0]), float(vec[1]))
    return EdgeFeature(points=points_yx, length=length, width=width, angle=angle)

def _compare_edges(A: List[EdgeFeature], B: List[EdgeFeature]) -> Tuple[float, float, int, float]:
    if not A or not B: return 0.0, float("inf"), 0, 0.0
    cost = np.zeros((len(A), len(B)), np.float64)
    for i, ea in enumerate(A):
        for j, eb in enumerate(B):
            geom = _poly_cost(ea.points, eb.points)
            width = abs(math.log((ea.width + 1e-3)/(eb.width + 1e-3)))
            angle = _angle_diff(ea.angle, eb.angle)
            cost[i,j] = EDGE_WEIGHT*geom + WIDTH_WEIGHT*width + ANGLE_WEIGHT*angle
    pad_n = max(len(A), len(B))
    pad = np.full((pad_n, pad_n), cost.max() + 5.0, np.float64)
    pad[:cost.shape[0], :cost.shape[1]] = cost
    matches = _hungarian(pad)

    matched_costs: List[float] = []
    matched_len = 0.0
    for i, j in matches:
        if i < len(A) and j < len(B):
            matched_costs.append(float(cost[i,j]))
            matched_len += min(A[i].length, B[j].length)
    totA = sum(e.length for e in A); totB = sum(e.length for e in B)
    union = totA + totB - matched_len
    if union <= 1e-6: union = max(totA, totB, 1e-6)
    s_cov = float(matched_len / union)
    c_bar = float(np.mean(matched_costs)) if matched_costs else float("inf")
    return s_cov, c_bar, len(matches), matched_len

# ---- 工具集（卷积、滤波、EDT、旋转/平移、哈希辅助等） -------------------------
def _conv2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    ky, kx = k.shape
    r_y, r_x = ky//2, kx//2
    h, w = img.shape
    out = np.zeros_like(img, np.float32)
    for dy in range(-r_y, r_y+1):
        y0 = max(0, -dy); y1 = min(h, h-dy)
        for dx in range(-r_x, r_x+1):
            x0 = max(0, -dx); x1 = min(w, w-dx)
            out[y0:y1, x0:x1] += img[y0+dy:y1+dy, x0+dx:x1+dx] * k[dy+r_y, dx+r_x]
    return out

def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    if r <= 0: return img.copy()
    return _mean_box(img, r)

def _mean_std_box(img: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = _mean_box(img, r)
    mean2 = _mean_box(img*img, r)
    var = np.maximum(mean2 - mean*mean, 0.0)
    std = np.sqrt(var + 1e-6)
    return mean, std

def _mean_box(img: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return img.astype(np.float32)
    img_f = img.astype(np.float64)
    padded = np.pad(img_f, ((r, r), (r, r)), mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    h, w = img.shape
    y0 = np.arange(h); y1 = y0 + 2*r + 1
    x0 = np.arange(w); x1 = x0 + 2*r + 1
    A = integral[y0[:,None], x0[None,:]]
    B = integral[y0[:,None], x1[None,:]]
    C = integral[y1[:,None], x0[None,:]]
    D = integral[y1[:,None], x1[None,:]]
    area = float((2*r+1)*(2*r+1))
    return ((D - B - C + A) / area).astype(np.float32)

def _normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6: return np.zeros_like(x, np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def _resize_bilinear(img: np.ndarray, s: int) -> np.ndarray:
    if s <= 1: return img.copy()
    h, w = img.shape
    H, W = h*s, w*s
    ys = (np.arange(H) + 0.5)/s - 0.5
    xs = (np.arange(W) + 0.5)/s - 0.5
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, h-1); x1 = np.clip(x0 + 1, 0, w-1)
    y0 = np.clip(y0, 0, h-1);     x0 = np.clip(x0, 0, w-1)
    wy = (ys - y0)[..., None]; wx = (xs - x0)[None, ...]
    Ia = img[y0[:,None], x0[None,:]]
    Ib = img[y0[:,None], x1[None,:]]
    Ic = img[y1[:,None], x0[None,:]]
    Id = img[y1[:,None], x1[None,:]]
    top = Ia*(1-wx) + Ib*wx
    bot = Ic*(1-wx) + Id*wx
    out = top*(1-wy) + bot*wy
    return out.astype(img.dtype)

def _edt_bool(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inf = 1e12
    dist = np.where(mask, 0.0, inf).astype(np.float64)
    for y in range(h):
        dist[y,:] = _edt_1d(dist[y,:])
    for x in range(w):
        dist[:,x] = _edt_1d(dist[:,x])
    return np.sqrt(dist).astype(np.float32)

def _edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.size
    v = np.zeros(n, np.int32); z = np.zeros(n+1, np.float64)
    k = 0; v[0] = 0; z[0] = -np.inf; z[1] = np.inf
    def inter(i, j): return ((f[j]+j*j) - (f[i]+i*i)) / (2.0*(j-i))
    for q in range(1, n):
        s = inter(v[k], q)
        while s <= z[k]:
            k -= 1
            s = inter(v[k], q)
        k += 1; v[k] = q; z[k] = s; z[k+1] = np.inf
    k = 0; out = np.empty(n, np.float64)
    for q in range(n):
        while z[k+1] < q: k += 1
        d = q - v[k]; out[q] = d*d + f[v[k]]
    return out

def _distance(mask: np.ndarray) -> np.ndarray:
    return _edt_bool(mask.astype(bool))

def _rotation_matrix(shape: Tuple[int,int], angle_rad: float) -> np.ndarray:
    h, w = shape
    cx, cy = w/2.0, h/2.0
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    M = np.array([[ ca, -sa, cx - ca*cx + sa*cy],
                  [ sa,  ca, cy - sa*cx - ca*cy]], dtype=np.float32)
    return M

def _warp_affine_bool(mask: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    A = M[:, :2]; b = M[:, 2]
    Ai = np.linalg.inv(A)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    XY = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    src = (XY - b) @ Ai.T
    sx = np.rint(src[:,0]).astype(int); sy = np.rint(src[:,1]).astype(int)
    out = np.zeros(h*w, dtype=bool)
    ok = (sx>=0)&(sx<w)&(sy>=0)&(sy<h)
    out[ok] = mask[sy[ok], sx[ok]]
    return out.reshape(h, w)

def _warp_affine_float(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = img.shape
    A = M[:, :2]; b = M[:, 2]
    Ai = np.linalg.inv(A)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    XY = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    src = (XY - b) @ Ai.T
    sx = src[:,0]; sy = src[:,1]
    out = _bilinear_sample(img, sx, sy, h, w)
    return out.reshape(h, w)

def _translate_bool(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, bool)
    x0 = int(np.floor(dx)); y0 = int(np.floor(dy))
    xs = slice(max(0, x0), min(w, w + x0))
    ys = slice(max(0, y0), min(h, h + y0))
    xt = slice(max(0, -x0), min(w, w - x0))
    yt = slice(max(0, -y0), min(h, h - y0))
    out[yt, xt] = mask[ys, xs]
    return out

def _translate_float(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    srcx = xs - dx; srcy = ys - dy
    return _bilinear_sample(img, srcx.ravel(), srcy.ravel(), h, w).reshape(h, w)

def _bilinear_sample(img: np.ndarray, sx: np.ndarray, sy: np.ndarray, h: int, w: int) -> np.ndarray:
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    wx = sx - x0; wy = sy - y0
    x0c = np.clip(x0, 0, w-1); x1c = np.clip(x1, 0, w-1)
    y0c = np.clip(y0, 0, h-1); y1c = np.clip(y1, 0, h-1)
    Ia = img[y0c, x0c]; Ib = img[y0c, x1c]; Ic = img[y1c, x0c]; Id = img[y1c, x1c]
    top = Ia*(1-wx) + Ib*wx; bot = Ic*(1-wy) + Id*wy
    return (top*(1-wy) + bot*wy).astype(np.float32)

def _sample_curve(v: np.ndarray, n: int) -> np.ndarray:
    xs = np.linspace(0, len(v)-1, n)
    return np.interp(xs, np.arange(len(v)), v).astype(np.float32)

def _diff_bits(s: np.ndarray) -> np.ndarray:
    d = np.diff(s)
    if d.size == 0: return np.zeros(1, np.uint8)
    med = float(np.median(d))
    bits = (d >= med).astype(np.uint8)
    return np.concatenate([bits, np.array([np.uint8(bits.mean() >= 0.5)])], 0)

# ---- 杂项（骨架、度数、邻域、匈牙利、角度差、斜率等） -------------------------
def _skeletonize(mask: np.ndarray) -> np.ndarray:
    img = mask.astype(np.uint8).copy()
    prev = np.zeros_like(img)
    while True:
        rem = _zhang_suen(img, 0); rem |= _zhang_suen(img, 1)
        img[rem] = 0
        if np.array_equal(img, prev) or not np.any(rem): break
        prev = img.copy()
    return img.astype(bool)

def _zhang_suen(img: np.ndarray, step: int) -> np.ndarray:
    h, w = img.shape
    rm = np.zeros_like(img, bool)
    for y in range(1, h-1):
        for x in range(1, w-1):
            if img[y,x] == 0: continue
            nb = _neighbors(img, x, y); trans = _transitions(nb)
            s = sum(nb)
            if not (2 <= s <= 6): continue
            if trans != 1: continue
            if step == 0:
                if nb[0]*nb[2]*nb[4] != 0: continue
                if nb[2]*nb[4]*nb[6] != 0: continue
            else:
                if nb[0]*nb[2]*nb[6] != 0: continue
                if nb[0]*nb[4]*nb[6] != 0: continue
            rm[y,x] = True
    return rm

def _neighbors(img: np.ndarray, x: int, y: int) -> List[int]:
    return [int(img[y-1,x]>0), int(img[y-1,x+1]>0), int(img[y,x+1]>0), int(img[y+1,x+1]>0),
            int(img[y+1,x]>0), int(img[y+1,x-1]>0), int(img[y,x-1]>0), int(img[y-1,x-1]>0)]

def _transitions(nb: Sequence[int]) -> int:
    t = 0
    for i in range(len(nb)):
        if nb[i] == 0 and nb[(i+1)%len(nb)] == 1: t += 1
    return t

def _degree_map(skel: np.ndarray) -> np.ndarray:
    h, w = skel.shape
    deg = np.zeros_like(skel, np.uint8)
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y,x]:
                deg[y,x] = sum(_neighbors(skel.astype(np.uint8), x, y))
    return deg

def _neighbors_coords(mask: np.ndarray, p: Tuple[int,int]) -> List[Tuple[int,int]]:
    y, x = p; h, w = mask.shape
    out: List[Tuple[int,int]] = []
    for ny in range(y-1, y+2):
        for nx in range(x-1, x+2):
            if ny==y and nx==x: continue
            if 0<=ny<h and 0<=nx<w and mask[ny,nx]:
                out.append((ny,nx))
    return out

def _pair_key(a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    return (min(a,b), max(a,b))

def _angle_diff(a: float, b: float) -> float:
    d = abs(a-b)
    while d > math.pi: d -= 2*math.pi
    return abs(d)

def _robust_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2: return 0.0
    best_m, best_i = 0.0, -1
    rng = np.random.default_rng(42)
    for _ in range(min(64, xs.size*xs.size)):
        i1 = rng.integers(0, xs.size); i2 = rng.integers(0, xs.size)
        if i1 == i2: continue
        x1, y1, x2, y2 = xs[i1], ys[i1], xs[i2], ys[i2]
        if abs(x2-x1) < 1e-3: continue
        m = float((y2-y1)/(x2-x1))
        res = np.abs(ys - (m*xs + (y1 - m*x1)))
        inl = int(np.sum(res <= 2.0))
        if inl > best_i: best_i, best_m = inl, m
    if best_i <= 0:
        m, _ = np.polyfit(xs, ys, 1); return float(m)
    return best_m

def _poly_cost(pa: np.ndarray, pb: np.ndarray) -> float:
    if pa.shape[0] < 2 or pb.shape[0] < 2: return 1e3
    a = _resample_path(pa, 32); b = _resample_path(pb, 32)
    dab = _mean_min_dist(a, b); dba = _mean_min_dist(b, a)
    return float(0.5*(dab + dba))

def _resample_path(p: np.ndarray, n: int) -> np.ndarray:
    d = np.hypot(np.diff(p[:,0]), np.diff(p[:,1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-6: return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(p)-2)
    t0, t1 = s[idx], s[idx+1]
    r = (t - t0) / np.maximum(t1 - t0, 1e-6)
    q = p[idx]*(1.0 - r[:,None]) + p[idx+1]*(r[:,None])
    return q.astype(np.float32)

def _mean_min_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:,None,:] - b[None,:,:], axis=2)
    return float(np.mean(np.min(d, axis=1)))

def _hungarian(cost: np.ndarray) -> List[Tuple[int,int]]:
    c = cost.copy()
    n = c.shape[0]
    c -= c.min(axis=1, keepdims=True)
    c -= c.min(axis=0, keepdims=True)
    starred = np.zeros_like(c, bool); primed = np.zeros_like(c, bool)
    row_cov = np.zeros(n, bool); col_cov = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if c[i,j] == 0 and not row_cov[i] and not col_cov[j]:
                starred[i,j] = True; row_cov[i] = True; col_cov[j] = True; break
    row_cov[:] = False; col_cov[:] = False

    def cover_star_cols():
        for j in range(n):
            if np.any(starred[:,j]): col_cov[j] = True

    def find_zero():
        for i in range(n):
            if row_cov[i]: continue
            for j in range(n):
                if col_cov[j]: continue
                if c[i,j] == 0 and not primed[i,j]: return i, j
        return -1, -1

    def star_in_row(r): 
        cols = np.where(starred[r])[0]; return int(cols[0]) if cols.size else -1
    def star_in_col(col):
        rows = np.where(starred[:,col])[0]; return int(rows[0]) if rows.size else -1
    def prime_in_row(r):
        cols = np.where(primed[r])[0]; return int(cols[0]) if cols.size else -1
    def augment(path):
        for (i,j) in path:
            starred[i,j] = not starred[i,j]; primed[i,j] = False

    cover_star_cols()
    while True:
        if np.all(col_cov): break
        r, c0 = find_zero()
        while r == -1:
            unc = c[~row_cov][:, ~col_cov]
            if unc.size == 0: break
            m = unc.min()
            c[~row_cov, :] -= m; c[:, col_cov] += m
            r, c0 = find_zero()
        if r == -1: break
        primed[r, c0] = True
        sc = star_in_row(r)
        if sc == -1:
            path = [(r, c0)]
            while True:
                sr = star_in_col(path[-1][1])
                if sr == -1: break
                path.append((sr, path[-1][1]))
                pc = prime_in_row(path[-1][0])
                path.append((path[-1][0], pc))
            augment(path)
            primed[:] = False; row_cov[:] = False; col_cov[:] = False
            cover_star_cols()
        else:
            row_cov[r] = True; col_cov[sc] = False

    res: List[Tuple[int,int]] = []
    for i in range(n):
        cols = np.where(starred[i])[0]
        if cols.size: res.append((i, int(cols[0])))
    return res

def _clamp_roi(roi: Roi, W: int, H: int) -> Roi:
    if W <= 0 or H <= 0: return Roi(x=0, y=0, width=1, height=1)
    w = max(1, min(int(roi.width), W)); h = max(1, min(int(roi.height), H))
    x = int(min(max(roi.x, 0), W-1)); y = int(min(max(roi.y, 0), H-1))
    if x + w > W: x = max(0, W - w)
    if y + h > H: y = max(0, H - h)
    return Roi(x=x, y=y, width=w, height=h)

def _different(roi: Roi, reason: str) -> SubtitleSimilarityResult:
    return SubtitleSimilarityResult(
        score=0.0, confidence=0.0, decision="different", roi=roi,
        metrics={"hash_bits":0.0,"hamming":0.0,"hamming_tau":0.0,"s_cov":0.0,"c_bar":float("inf"),"s_final":0.0},
        details={"reason": reason}, delta=(0,0),
    )
