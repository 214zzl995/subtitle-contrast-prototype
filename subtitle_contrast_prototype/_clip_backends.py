from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Optional, Sequence, Tuple

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class ClipBackend:
    name: str
    pretrained_tag: str
    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    device: torch.device


def _resolve_backend_from_env(prefix: str) -> Optional[Tuple[str, str]]:
    name = os.getenv(f"{prefix}_MODEL_NAME")
    pretrained = os.getenv(f"{prefix}_PRETRAINED")
    if name:
        return name, pretrained or ""
    return None


def _list_pretrained_matches(keyword: str) -> Sequence[Tuple[str, str]]:
    keyword_lower = keyword.lower()
    pairs = open_clip.list_pretrained()
    matches = [
        (name, tag)
        for name, tag in pairs
        if keyword_lower in name.lower() or (tag and keyword_lower in tag.lower())
    ]
    return matches


def _create_backend(model_name: str, pretrained_tag: str) -> ClipBackend:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    if pretrained_tag:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return ClipBackend(
        name=model_name,
        pretrained_tag=pretrained_tag,
        model=model,
        preprocess=preprocess,
        device=device,
    )


def _select_backend(keyword: str, env_prefix: str) -> ClipBackend:
    override = _resolve_backend_from_env(env_prefix)
    errors: list[str] = []
    tried: list[Tuple[str, str]] = []

    candidates: Iterable[Tuple[str, str]]
    if override:
        candidates = [override]
    else:
        matches = _list_pretrained_matches(keyword)
        if not matches:
            raise RuntimeError(
                f"No pre-trained checkpoints containing '{keyword}' were found. "
                f"Set environment variables {env_prefix}_MODEL_NAME and "
                f"{env_prefix}_PRETRAINED explicitly to select a checkpoint."
            )
        candidates = matches

    last_exception: Optional[Exception] = None
    for model_name, pretrained_tag in candidates:
        try:
            backend = _create_backend(model_name, pretrained_tag)
            return backend
        except Exception as exc:  # pragma: no cover - defensive path
            errors.append(f"{model_name}:{pretrained_tag or '<none>'} -> {exc}")
            tried.append((model_name, pretrained_tag))
            last_exception = exc
    raise RuntimeError(
        f"Failed to initialise {keyword} backend. Tried: "
        + ", ".join(f"{name}:{tag or '<none>'}" for name, tag in tried)
    ) from last_exception


@lru_cache(maxsize=1)
def get_mobileclip2_backend() -> ClipBackend:
    return _select_backend(keyword="mobileclip", env_prefix="MOBILECLIP2")


@lru_cache(maxsize=1)
def get_tinyclip_backend() -> ClipBackend:
    return _select_backend(keyword="tinyclip", env_prefix="TINYCLIP")


def encode_image(backend: ClipBackend, image: Image.Image) -> torch.Tensor:
    tensor = backend.preprocess(image).unsqueeze(0).to(backend.device)
    with torch.no_grad():
        features = backend.model.encode_image(tensor)
        features = F.normalize(features, dim=-1)
    return features.squeeze(0).cpu()
