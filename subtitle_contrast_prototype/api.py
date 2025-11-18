from __future__ import annotations

import io
import math
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np

from .config import AppConfig, load_config
from .frames import FrameRepository
from .similarity_v10 import compute_similarity as compute_similarity_v10
from .similarity_v11 import (
    compute_similarity as compute_similarity_v11,
    SubtitleSimilarityRequest,
    Roi,
)
from .similarity_v20 import compute_similarity as compute_similarity_v20
from .similarity_v30 import compute_similarity as compute_similarity_v30
from .similarity_v40 import compute_similarity as compute_similarity_v40
from .similarity_v50 import compute_similarity as compute_similarity_v50
from .similarity_v60 import compute_similarity as compute_similarity_v60
from .similarity_v70 import compute_similarity as compute_similarity_v70
from .similarity_v80 import compute_similarity as compute_similarity_v80
from .similarity_v90 import compute_similarity as compute_similarity_v90
from .similarity_v100 import compute_similarity as compute_similarity_v100
from .similarity_v110 import compute_similarity as compute_similarity_v110
from .similarity_v120 import compute_similarity as compute_similarity_v120
from .similarity_v130 import compute_similarity as compute_similarity_v130
from .visualization import render_visualization
from .preprocess import inverse_clamp_uint8
from .roi_utils import PreparedRoiPair, build_prepared_roi_pair


@lru_cache(maxsize=1)
def _load_config() -> AppConfig:
    return load_config()


@lru_cache(maxsize=1)
def _load_repository() -> FrameRepository:
    return FrameRepository(_load_config())


app = FastAPI(
    title="Subtitle Contrast Prototype",
    description="Backend for Y-plane subtitle similarity and frame previews.",
    version="0.2.0",
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def healthcheck() -> dict:
    config = _load_config()
    return {
        "status": "ok",
        "format": config.format,
        "width": config.width,
        "height": config.height,
        "search_radius": config.search_radius,
    }


@app.get("/frames")
def list_frames() -> dict:
    repository = _load_repository()
    frames = repository.list()
    payload = [
        {
            "name": meta.name,
        }
        for meta in frames
    ]
    return {"frames": payload, "count": len(payload)}


@app.get("/frames/{frame_name}/image")
def fetch_frame_image(frame_name: str) -> StreamingResponse:
    repository = _load_repository()
    try:
        png_bytes = repository.render_png(frame_name if frame_name.endswith(".yuv") else f"{frame_name}.yuv")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.post("/compare")
def compare(request: SubtitleSimilarityRequest) -> JSONResponse:
    config = _load_config()
    repository = _load_repository()

    version_raw = getattr(request, "version", None) or "v1.1"
    version = version_raw.lower()
    supported = {"v1.0", "v1.1", "v2.0", "v3.0", "v4.0", "v5.0", "v6.0", "v7.0", "v8.0", "v9.0", "v10.0", "v11.0", "v12.0", "v13.0", "v130"}
    if version not in supported:
        raise HTTPException(status_code=400, detail=f"version must be one of: {', '.join(sorted(supported))}")

    try:
        import time
        start_ns = time.perf_counter_ns()
        pre_repo = _PreprocessedRepository(
            repository,
            request.frame_a,
            request.frame_b,
            request.mu_sub,
            request.delta_y,
        )
        request.roi = _clamp_roi_model(request.roi, pre_repo.common_width, pre_repo.common_height)
        prepared_roi = pre_repo.prepare_roi_pair(request.roi)

        repo = pre_repo
        if version == "v1.0":
            result = compute_similarity_v10(config, repo, request, prepared_roi)
        elif version == "v2.0":
            result = compute_similarity_v20(config, repo, request, prepared_roi)
        elif version == "v3.0":
            result = compute_similarity_v30(config, repo, request, prepared_roi)
        elif version == "v4.0":
            result = compute_similarity_v40(config, repo, request, prepared_roi)
        elif version == "v5.0":
            result = compute_similarity_v50(config, repo, request, prepared_roi)
        elif version == "v6.0":
            result = compute_similarity_v60(config, repo, request, prepared_roi)
        elif version == "v7.0":
            result = compute_similarity_v70(config, repo, request, prepared_roi)
        elif version == "v8.0":
            result = compute_similarity_v80(config, repo, request, prepared_roi)
        elif version == "v9.0":
            result = compute_similarity_v90(config, repo, request, prepared_roi)
        elif version == "v10.0":
            result = compute_similarity_v100(config, repo, request, prepared_roi)
        elif version == "v11.0":
            result = compute_similarity_v110(config, repo, request, prepared_roi)
        elif version == "v12.0":
            result = compute_similarity_v120(config, repo, request, prepared_roi)
        elif version in {"v13.0", "v130"}:
            result = compute_similarity_v130(config, repo, request, prepared_roi)
        else:
            result = compute_similarity_v11(config, repo, request, prepared_roi)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = _json_safe(
        {
            "score": result.score,
            "confidence": result.confidence,
            "decision": result.decision,
            "dx": result.delta[0],
            "dy": result.delta[1],
            "roi": result.roi.dict(),
            "roi_requested": request.roi.dict(),
            "metrics": result.metrics,
            "details": {**result.details, "latency_ms": elapsed_ms},
            "version": version_raw,
            "latency_ms": elapsed_ms,
        }
    )
    return JSONResponse(payload)


@app.post("/visualize")
def visualize(request: SubtitleSimilarityRequest) -> JSONResponse:
    config = _load_config()
    repository = _load_repository()
    pre_repo = _PreprocessedRepository(
        repository,
        request.frame_a,
        request.frame_b,
        request.mu_sub,
        request.delta_y,
    )
    request.roi = _clamp_roi_model(request.roi, pre_repo.common_width, pre_repo.common_height)

    try:
        payload = render_visualization(config, pre_repo, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(payload)


class _PreprocessedRepository:
    def __init__(
        self,
        base: FrameRepository,
        frame_a: str,
        frame_b: str,
        mu_sub: float,
        delta_y: float,
    ) -> None:
        self._base = base
        self._mu = mu_sub
        self._delta = delta_y
        self._frame_a = frame_a
        self._frame_b = frame_b
        self.geometry = base.geometry
        self._cache: Dict[str, np.ndarray] = {}
        self._common_width: int | None = None
        self._common_height: int | None = None
        self._roi_cache: Dict[tuple[int, int, int, int], PreparedRoiPair] = {}
        self._prepare_common_frames()

    @property
    def common_width(self) -> int:
        return int(self._common_width or self.geometry.width)

    @property
    def common_height(self) -> int:
        return int(self._common_height or self.geometry.height)

    def load_y_plane(self, name: str) -> np.ndarray:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        frame = self._base.load_y_plane(name)
        cropped = frame[: self.common_height, : self.common_width]
        processed = inverse_clamp_uint8(cropped, self._mu, self._delta)
        self._cache[name] = processed
        return processed

    def __getattr__(self, item: str):
        return getattr(self._base, item)

    def _prepare_common_frames(self) -> None:
        frame_a = self._base.load_y_plane(self._frame_a)
        frame_b = self._base.load_y_plane(self._frame_b)
        self._common_height = min(frame_a.shape[0], frame_b.shape[0])
        self._common_width = min(frame_a.shape[1], frame_b.shape[1])
        self._cache[self._frame_a] = inverse_clamp_uint8(
            frame_a[: self._common_height, : self._common_width], self._mu, self._delta
        )
        self._cache[self._frame_b] = inverse_clamp_uint8(
            frame_b[: self._common_height, : self._common_width], self._mu, self._delta
        )

    def prepare_roi_pair(self, roi: Roi) -> PreparedRoiPair:
        key = (roi.x, roi.y, roi.width, roi.height)
        cached = self._roi_cache.get(key)
        if cached is not None:
            return cached

        frame_a = self.load_y_plane(self._frame_a)
        frame_b = self.load_y_plane(self._frame_b)
        prepared = build_prepared_roi_pair(frame_a, frame_b, roi)
        self._roi_cache[key] = prepared
        return prepared


def _clamp_roi_model(roi: Roi, width: int, height: int) -> Roi:
    w = max(1, width)
    h = max(1, height)
    x = int(min(max(roi.x, 0), w - 1))
    y = int(min(max(roi.y, 0), h - 1))
    width_adj = int(max(1, min(roi.width, w - x)))
    height_adj = int(max(1, min(roi.height, h - y)))
    return Roi(x=x, y=y, width=width_adj, height=height_adj)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        # Normalize tuples to lists for JSON safety
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else 0.0
    return value
