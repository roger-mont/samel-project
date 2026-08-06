"""Pipeline de pressão — pass-through direto dos valores HID (0-255).

Os bytes recebidos via USB HID já representam intensidade de pressão bruta.
Não há conversão de tensão, resistência ou força.
"""
from __future__ import annotations

import logging

import numpy as np

from config.settings import CalibrationParams

logger = logging.getLogger(__name__)


def compute_force_matrix(
    pressure_matrix: np.ndarray,
    params: CalibrationParams,
) -> np.ndarray:
    """Aplica deadzone sobre a matriz bruta HID (0-255) e retorna como está.

    Valores <= deadzone_threshold são zerados para eliminar ruído de fundo.
    """
    snap = params.snapshot()
    deadzone = snap["deadzone_threshold"]

    result = pressure_matrix.copy()
    result[result <= deadzone] = 0.0

    if float(np.max(result)) > 0:
        logger.info(
            "PRESSAO matrix (sum=%.0f, max=%.0f, pontos_ativos=%d)",
            float(np.sum(result)),
            float(np.max(result)),
            int(np.count_nonzero(result)),
        )

    return result


def compute_total_mass(pressure_matrix: np.ndarray) -> float:
    """Retorna a soma total de pressão como escalar (unidade: pressão bruta).

    O campo 'weight_kg' no frontend passará a representar pressão total acumulada,
    não massa física.
    """
    return float(np.sum(pressure_matrix))


def apply_ema(current: float, previous: float, alpha: float) -> float:
    """Média Móvel Exponencial para suavização da pressão total exibida.

    EMA = α × atual + (1 − α) × anterior
    """
    return alpha * current + (1.0 - alpha) * previous
