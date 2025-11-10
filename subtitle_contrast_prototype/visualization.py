from __future__ import annotations

import base64
from typing import Callable, Dict, Tuple

import numpy as np
from PIL import Image

from .config import AppConfig
from .frames import FrameRepository
from .roi_utils import get_prepared_roi_pair
from .similarity_v11 import Roi, SubtitleSimilarityRequest
from .similarity_v120 import _core_pipeline as v120_core_pipeline


Visualizer = Callable[[AppConfig, FrameRepository, SubtitleSimilarityRequest, str], Dict[str, object]]


def render_visualization(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
) -> Dict[str, object]:
    version_raw = getattr(request, "version", None) or "v1.1"
    version = version_raw.lower()
    handler = _VISUALIZERS.get(version, _visualize_default)
    return handler(config, repository, request, version_raw)


def _visualize_v120(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
    version_raw: str,
) -> Dict[str, object]:
    return _visualize_v12_pipeline(
        config,
        repository,
        request,
        version_raw,
        core_pipeline=v120_core_pipeline,
        metadata_extra={"inverse_clamp": True},
    )


def _visualize_v12_pipeline(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
    version_raw: str,
    *,
    core_pipeline: Callable[..., Tuple],
    metadata_extra: Dict[str, object],
) -> Dict[str, object]:
    prepared = get_prepared_roi_pair(repository, request)
    roi = prepared.roi

    mu = request.mu_sub / 255.0
    delta = max(request.delta_y / 255.0, 1.0 / 255.0)

    frames_payload: Dict[str, Dict[str, object]] = {}
    for label, crop in (("frame_a", prepared.frame_a), ("frame_b", prepared.frame_b)):
        base = crop.astype(np.float32) / 255.0
        core, _, w_est, contrast_thr, debug = core_pipeline(base, mu, delta, capture_debug=True)
        debug = debug or {}
        pre_image = debug.get("preprocessed", base)
        mask = debug.get("core", core)

        pre_png = _to_data_url(_encode_gray(pre_image))
        overlay_png = _to_data_url(
            _encode_overlay(
                pre_image,
                mask,
                pre_highlight=(0.0, 200.0, 255.0) if metadata_extra.get("inverse_clamp") else None,
            )
        )

        metadata = {
            "core_area": float(mask.mean()) if mask.size else 0.0,
            "w_est": float(w_est),
            "contrast_threshold": float(contrast_thr),
            **metadata_extra,
        }
        frames_payload[label] = {
            "preprocessed_png": pre_png,
            "overlay_png": overlay_png,
            "metadata": metadata,
        }

    return _build_payload(version_raw, roi, request, frames_payload)


def _visualize_default(
    config: AppConfig,
    repository: FrameRepository,
    request: SubtitleSimilarityRequest,
    version_raw: str,
) -> Dict[str, object]:
    prepared = get_prepared_roi_pair(repository, request)
    roi = prepared.roi

    frames_payload: Dict[str, Dict[str, object]] = {}
    for label, crop in (("frame_a", prepared.frame_a), ("frame_b", prepared.frame_b)):
        base = crop.astype(np.float32) / 255.0
        pre_png = _to_data_url(_encode_gray(base))
        overlay_png = _to_data_url(
            _encode_overlay(base, None, pre_highlight=(0.0, 200.0, 255.0))
        )
        frames_payload[label] = {
            "preprocessed_png": pre_png,
            "overlay_png": overlay_png,
            "metadata": {"note": "default visualization"},
        }
    return _build_payload(version_raw, roi, request, frames_payload)


def _build_payload(
    version_raw: str,
    roi: Roi,
    request: SubtitleSimilarityRequest,
    frames_payload: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    return {
        "version": version_raw,
        "roi": roi.dict(),
        "mu_sub": request.mu_sub,
        "delta_y": request.delta_y,
        "frames": frames_payload,
    }


def _encode_gray(image: np.ndarray) -> bytes:
    arr = _to_uint8(image)
    pil = Image.fromarray(arr, mode="L")
    return _encode_png(pil)


def _encode_overlay(
    image: np.ndarray,
    mask: np.ndarray | None,
    pre_highlight: Tuple[float, float, float] | None = None,
) -> bytes:
    base = _to_uint8(image)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    if pre_highlight is not None:
        highlight = np.array(pre_highlight, dtype=np.float32)
        rgb = np.clip(rgb * 0.35 + highlight * 0.65, 0.0, 255.0)
    if mask is not None and mask.any():
        highlight = np.array([255.0, 64.0, 64.0], dtype=np.float32)
        alpha = 0.65
        mask_bool = mask.astype(bool)
        rgb[mask_bool] = rgb[mask_bool] * (1 - alpha) + highlight * alpha
    rgb_u8 = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(rgb_u8, mode="RGB")
    return _encode_png(pil)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    arr = np.clip(image, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _encode_png(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _to_data_url(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_VISUALIZERS: Dict[str, Visualizer] = {
    "v12.0": _visualize_v120,
}
