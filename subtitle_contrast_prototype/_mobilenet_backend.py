from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass(frozen=True)
class MobilenetBackend:
    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    device: torch.device
    weights_source: str


def _default_preprocess() -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_custom_weights(model: torch.nn.Module, env_var: str = "MOBILENET_V3_SMALL_WEIGHTS") -> Optional[str]:
    weights_path = os.getenv(env_var)
    if not weights_path:
        return None
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Custom MobileNetV3-Small weights not found at {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return str(path)


@lru_cache(maxsize=1)
def get_mobilenet_backend() -> MobilenetBackend:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = _default_preprocess()
    weights_tag = "custom"
    model = mobilenet_v3_small(weights=None)

    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        preprocess = weights.transforms()
        weights_tag = weights.meta.get("id", "imagenet")
    except Exception as exc:  # pragma: no cover - fallback path
        custom_src = _load_custom_weights(model)
        if not custom_src:
            raise RuntimeError(
                "Failed to load pretrained MobileNetV3-Small weights. "
                "Ensure torchvision weights are cached or provide a checkpoint path via "
                "MOBILENET_V3_SMALL_WEIGHTS."
            ) from exc
        weights_tag = custom_src

    feature_extractor = torch.nn.Sequential(model.features, model.avgpool)
    feature_extractor.eval()
    feature_extractor.to(device)
    for param in feature_extractor.parameters():
        param.requires_grad_(False)

    return MobilenetBackend(
        model=feature_extractor,
        preprocess=preprocess,
        device=device,
        weights_source=weights_tag,
    )


def encode_image(backend: MobilenetBackend, image: Image.Image) -> torch.Tensor:
    tensor = backend.preprocess(image).unsqueeze(0).to(backend.device)
    with torch.no_grad():
        feature_map = backend.model(tensor)
        feature_vec = torch.flatten(feature_map, 1)
        feature_vec = F.normalize(feature_vec, dim=-1)
    return feature_vec.squeeze(0).cpu()

