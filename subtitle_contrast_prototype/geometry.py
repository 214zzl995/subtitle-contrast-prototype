from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True, slots=True)
class Geometry:
    format: str
    width: int
    height: int


COMMON_WIDTHS: Tuple[int, ...] = (
    4096,
    3840,
    3440,
    3200,
    3072,
    2880,
    2560,
    2464,
    2400,
    2336,
    2304,
    2160,
    2048,
    2000,
    1980,
    1920,
    1880,
    1856,
    1820,
    1760,
    1720,
    1680,
    1600,
    1536,
    1520,
    1500,
    1440,
    1408,
    1366,
    1344,
    1320,
    1280,
    1248,
    1232,
    1216,
    1180,
    1152,
    1120,
    1080,
    1056,
    1024,
    992,
    960,
    928,
    896,
    864,
    854,
    848,
    832,
    800,
    768,
    736,
    704,
    672,
    640,
    608,
    576,
    544,
    528,
    512,
    496,
    480,
    464,
    448,
    432,
    416,
    400,
    384,
    368,
    352,
    336,
    320,
)

ASPECT_TARGETS = (1.0, 4 / 3, 16 / 9, 21 / 9, 2.0)


def infer_geometry(path: Path) -> Geometry:
    raw = path.read_bytes()
    file_size = len(raw)
    if file_size <= 0:
        raise ValueError("YUV file is empty.")

    candidates = list(_iter_candidates(raw))
    if not candidates:
        raise ValueError("Unable to infer YUV geometry from payload.")

    best_score = float("inf")
    best_geometry = None

    for fmt, width, height in candidates:
        score = _score_candidate(raw, fmt, width, height)
        if score < best_score:
            best_score = score
            best_geometry = Geometry(format=fmt, width=width, height=height)

    if best_geometry is None:
        raise ValueError("Failed to determine a plausible YUV geometry.")

    return best_geometry


def _iter_candidates(raw: bytes) -> Iterable[Tuple[str, int, int]]:
    size = len(raw)

    def push(fmt: str, width: int, height: int) -> Iterable[Tuple[str, int, int]]:
        if width <= 0 or height <= 0:
            return
        if height < 64:
            return
        aspect = width / max(height, 1)
        if 0.5 <= aspect <= 5.0:
            yield fmt, width, height

    for width in COMMON_WIDTHS:
        if size % width == 0:
            yield from push("y_only", width, size // width)
        if (size * 2) % (width * 3) == 0:
            yield from push("yuv420", width, (size * 2) // (width * 3))
        if size % (width * 2) == 0:
            yield from push("yuv422", width, size // (width * 2))

    if size % 2 == 0:
        for width in range(256, 4097, 16):
            if size % width == 0:
                yield from push("y_only", width, size // width)
            if size % (width * 2) == 0:
                yield from push("yuv422", width, size // (width * 2))

    if (size * 2) % 3 == 0:
        for width in range(256, 4097, 16):
            if (size * 2) % (width * 3) == 0:
                yield from push("yuv420", width, (size * 2) // (width * 3))


def _score_candidate(raw: bytes, fmt: str, width: int, height: int) -> float:
    fmt_bias = {"y_only": 0.0, "yuv420": 2.0, "yuv422": 4.0}[fmt]
    aspect = width / height
    aspect_penalty = min(abs(aspect - target) for target in ASPECT_TARGETS)

    mean_h, mean_v = _mean_absolute_differences(raw, fmt, width, height)
    if mean_h == 0.0 and mean_v == 0.0:
        texture_penalty = 10.0
    else:
        ratio = mean_h / (mean_v + 1e-6)
        texture_penalty = abs(math.log(max(ratio, 1e-6)))

    alignment_penalty = (height % 16) / 16.0 + (height % 8) / 32.0
    bonus = width * 1e-5  # prefer higher resolutions on ties

    return fmt_bias + texture_penalty * 6.0 + aspect_penalty * 12.0 + alignment_penalty - bonus


def _mean_absolute_differences(raw: bytes, fmt: str, width: int, height: int) -> Tuple[float, float]:
    if fmt == "y_only":
        plane = raw
        expected = width * height
        if len(plane) != expected:
            raise ValueError("Payload size does not match Y-only expectation.")
    elif fmt == "yuv420":
        expected = width * height * 3 // 2
        if len(raw) != expected:
            raise ValueError("Payload size does not match YUV420 expectation.")
        plane = raw[: width * height]
    elif fmt == "yuv422":
        expected = width * height * 2
        if len(raw) != expected:
            raise ValueError("Payload size does not match YUV422 expectation.")
        plane = raw[0:expected:2]
    else:  # pragma: no cover - defensive path
        raise ValueError(f"Unsupported YUV format: {fmt}")

    sum_h = 0
    sum_v = 0
    count_h = 0
    count_v = 0

    for y in range(height):
        row = y * width
        row_end = row + width - 1
        for idx in range(row, row_end):
            sum_h += abs(plane[idx + 1] - plane[idx])
        count_h += max(width - 1, 0)

        if y < height - 1:
            nxt = (y + 1) * width
            for x in range(width):
                sum_v += abs(plane[nxt + x] - plane[row + x])
            count_v += width

    mean_h = float(sum_h) / count_h if count_h else 0.0
    mean_v = float(sum_v) / count_v if count_v else 0.0
    return mean_h, mean_v
