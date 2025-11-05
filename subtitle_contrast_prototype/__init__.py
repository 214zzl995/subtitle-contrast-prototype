"""
Subtitle contrast prototype package.

This module exposes helpers for loading YUV frames and computing
subtitle similarity scores as described in the project README.
"""

from .config import AppConfig, load_config
from .frames import FrameRepository, FrameMeta
from .similarity import (
    SubtitleSimilarityRequest,
    SubtitleSimilarityResult,
    compute_similarity,
)

__all__ = [
    "AppConfig",
    "FrameRepository",
    "FrameMeta",
    "SubtitleSimilarityRequest",
    "SubtitleSimilarityResult",
    "compute_similarity",
    "load_config",
]
