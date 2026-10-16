"""Etapas auxiliares do pipeline. O RAW é tratado como resposta digital, sem conversão física.

Matriz FSR → aquisição RAW → baseline/tara → filtragem temporal → movimento →
estabilidade → features → modelo de calibração → validação → API.
"""
from __future__ import annotations

import numpy as np


def apply_deadzone(matrix: np.ndarray, deadzone_raw: float) -> np.ndarray:
    """Zera leituras <= deadzone_raw. Usado só no mapa de calor e na postura, nunca na massa."""
    result = matrix.copy()
    result[result <= deadzone_raw] = 0.0
    return result


def window_mean(frames: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    """Filtragem temporal: média dos frames da janela de estabilidade."""
    if not frames:
        return fallback.astype(np.float64)
    return np.mean(np.stack(frames), axis=0).astype(np.float64)
