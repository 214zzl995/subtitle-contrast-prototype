from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .geometry import Geometry, infer_geometry


@dataclass(slots=True)
class AppConfig:
    data_root: Path
    geometry: Geometry
    search_radius: int = 3
    lambda_edge: float = 0.6

    @property
    def width(self) -> int:
        return self.geometry.width

    @property
    def height(self) -> int:
        return self.geometry.height

    @property
    def format(self) -> str:
        return self.geometry.format


def _parse_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _parse_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Environment variable {name} must be a float.") from exc


def _resolve_geometry(data_root: Path) -> Geometry:
    explicit_width = _parse_int("YUV_WIDTH")
    explicit_height = _parse_int("YUV_HEIGHT")
    explicit_format = os.getenv("YUV_FORMAT")

    if explicit_format is not None:
        explicit_format = explicit_format.lower()
        if explicit_format not in {"y_only", "yuv420", "yuv422"}:
            raise ValueError("YUV_FORMAT must be one of: y_only, yuv420, yuv422.")

    if explicit_width and explicit_height and explicit_format:
        return Geometry(format=explicit_format, width=explicit_width, height=explicit_height)

    try:
        first_file = next(p for p in sorted(data_root.iterdir()) if p.suffix.lower() == ".yuv")
    except StopIteration as exc:
        raise FileNotFoundError(f"No .yuv files were found under {data_root}.") from exc

    detected = infer_geometry(first_file)
    width = explicit_width or detected.width
    height = explicit_height or detected.height
    fmt = explicit_format or detected.format
    return Geometry(format=fmt, width=width, height=height)


def load_config() -> AppConfig:
    data_root = Path(os.getenv("DATA_ROOT", "frames/yuv")).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT directory not found: {data_root}")

    geometry = _resolve_geometry(data_root)

    search_radius = _parse_int("YUV_SEARCH_RADIUS") or 3
    if search_radius < 0:
        raise ValueError("YUV_SEARCH_RADIUS must be non-negative.")

    lambda_edge = _parse_float("YUV_EDGE_LAMBDA") or 0.6
    if not 0.0 <= lambda_edge <= 1.0:
        raise ValueError("YUV_EDGE_LAMBDA must lie within [0, 1].")

    return AppConfig(
        data_root=data_root,
        geometry=geometry,
        search_radius=search_radius,
        lambda_edge=lambda_edge,
    )
