from __future__ import annotations

import numpy as np


def inverse_clamp_uint8(frame: np.ndarray, mu_sub: float, delta_y: float) -> np.ndarray:
    """Clamp pixels outside [mu - delta, mu + delta] to inverted color."""
    if frame.dtype != np.uint8:
        data = frame.astype(np.float32)
    else:
        data = frame.astype(np.float32, copy=False)
    mu = float(np.clip(mu_sub, 0.0, 255.0))
    delta = float(max(delta_y, 0.0))
    low = max(0.0, mu - delta)
    high = min(255.0, mu + delta)
    replacement = float(255.0 - mu)
    mask = (data >= low) & (data <= high)
    processed = np.where(mask, data, replacement)
    processed = np.clip(processed, 0.0, 255.0)
    return processed.astype(np.uint8)
