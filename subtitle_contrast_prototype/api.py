from __future__ import annotations

import io
import math
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np

from .config import AppConfig, load_config
from .frames import FrameRepository
from .similarity_v10 import compute_similarity as compute_similarity_v10
from .similarity_v11 import compute_similarity as compute_similarity_v11, SubtitleSimilarityRequest
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
    supported = {"v1.0", "v1.1", "v2.0", "v3.0", "v4.0", "v5.0", "v6.0", "v7.0", "v8.0", "v9.0", "v10.0", "v11.0"}
    if version not in supported:
        raise HTTPException(status_code=400, detail=f"version must be one of: {', '.join(sorted(supported))}")

    try:
        import time
        start_ns = time.perf_counter_ns()
        if version == "v1.0":
            result = compute_similarity_v10(config, repository, request)
        elif version == "v2.0":
            result = compute_similarity_v20(config, repository, request)
        elif version == "v3.0":
            result = compute_similarity_v30(config, repository, request)
        elif version == "v4.0":
            result = compute_similarity_v40(config, repository, request)
        elif version == "v5.0":
            result = compute_similarity_v50(config, repository, request)
        elif version == "v6.0":
            result = compute_similarity_v60(config, repository, request)
        elif version == "v7.0":
            result = compute_similarity_v70(config, repository, request)
        elif version == "v8.0":
            result = compute_similarity_v80(config, repository, request)
        elif version == "v9.0":
            result = compute_similarity_v90(config, repository, request)
        elif version == "v10.0":
            result = compute_similarity_v100(config, repository, request)
        elif version == "v11.0":
            result = compute_similarity_v110(config, repository, request)
        else:
            result = compute_similarity_v11(config, repository, request)
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
