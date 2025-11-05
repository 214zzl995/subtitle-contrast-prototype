from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from .config import AppConfig
from .geometry import Geometry, infer_geometry


@dataclass(frozen=True, slots=True)
class FrameMeta:
    name: str
    path: Path


class FrameRepository:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._frames = self._discover_frames(config.data_root)
        if not self._frames:
            raise FileNotFoundError(f"No .yuv files discovered under {config.data_root}")
        self._index: Dict[str, FrameMeta] = {frame.name: frame for frame in self._frames}
        self._geometry_cache: Dict[str, Geometry] = {}
        self._geometry_by_size: Dict[int, Geometry] = {
            _expected_payload_size(config.geometry): config.geometry
        }

    @property
    def geometry(self):
        return self._config.geometry

    def list(self) -> List[FrameMeta]:
        return list(self._frames)

    def get(self, name: str) -> FrameMeta:
        try:
            return self._index[name]
        except KeyError as exc:
            raise FileNotFoundError(f"Frame '{name}' not found.") from exc

    def load_y_plane(self, name: str) -> np.ndarray:
        meta = self.get(name)
        payload = meta.path.read_bytes()
        geometry = self._resolve_geometry(meta, len(payload))
        y_bytes = _extract_y_plane(payload, geometry.format, geometry.width, geometry.height)
        array = np.frombuffer(y_bytes, dtype=np.uint8).reshape((geometry.height, geometry.width))
        return array

    def render_png(self, name: str) -> bytes:
        y_plane = self.load_y_plane(name)
        image = Image.fromarray(y_plane, mode="L")
        buffer = _to_png(image)
        return buffer

    @staticmethod
    def _discover_frames(root: Path) -> List[FrameMeta]:
        frames: List[FrameMeta] = []
        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".yuv":
                frames.append(FrameMeta(name=entry.name, path=entry))
        return frames

    def _resolve_geometry(self, meta: FrameMeta, payload_size: int) -> Geometry:
        if meta.name in self._geometry_cache:
            return self._geometry_cache[meta.name]

        size_cached = self._geometry_by_size.get(payload_size)
        if size_cached:
            self._geometry_cache[meta.name] = size_cached
            return size_cached

        inferred = infer_geometry(meta.path)
        self._geometry_cache[meta.name] = inferred
        self._geometry_by_size[payload_size] = inferred
        return inferred


def _expected_payload_size(geometry: Geometry) -> int:
    pixels = geometry.width * geometry.height
    if geometry.format == "y_only":
        return pixels
    if geometry.format == "yuv420":
        return pixels * 3 // 2
    if geometry.format == "yuv422":
        return pixels * 2
    raise ValueError(f"Unsupported YUV format: {geometry.format}")


def _extract_y_plane(payload: bytes, fmt: str, width: int, height: int) -> bytes:
    pixels = width * height
    if fmt == "y_only":
        if len(payload) != pixels:
            raise ValueError(
                f"Y-only payload length mismatch: expected {pixels}, got {len(payload)}."
            )
        return payload
    if fmt == "yuv420":
        expected = pixels * 3 // 2
        if len(payload) != expected:
            raise ValueError(
                f"YUV420 payload length mismatch: expected {expected}, got {len(payload)}."
            )
        return payload[:pixels]
    if fmt == "yuv422":
        expected = pixels * 2
        if len(payload) != expected:
            raise ValueError(
                f"YUV422 payload length mismatch: expected {expected}, got {len(payload)}."
            )
        return payload[0:expected:2]
    raise ValueError(f"Unsupported YUV format: {fmt}")


def _to_png(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
