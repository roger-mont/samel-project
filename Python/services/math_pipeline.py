"""Pipeline de pressão — aplica calibração e retorna kg por bloco.

Fluxo:
  matriz HID bruta (0-255)
    → deadzone (compute_force_matrix)
    → conversão por bloco via curva polinomial (compute_total_mass)
    → EMA temporal (apply_ema)
"""
from __future__ import annotations

import logging

import numpy as np

from config.settings import CalibrationParams
from services.calibration_store import CalibData

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


def compute_total_mass(
    pressure_matrix: np.ndarray,
    calib: CalibData | None = None,
) -> float:
    """Converte a matriz de pressão em kg usando calibração por bloco.

    Se `calib` for None ou inválido, retorna a soma bruta como fallback.
    """
    if calib is not None and calib.is_valid:
        return calib.matrix_to_kg(pressure_matrix)

    # Fallback — sem calibração carregada
    logger.debug("sem calibração válida — usando soma bruta")
    return float(np.sum(pressure_matrix))


def apply_ema(current: float, previous: float, alpha: float) -> float:
    """Média Móvel Exponencial para suavização do peso total exibido.

    EMA = α × atual + (1 − α) × anterior
    """
    return alpha * current + (1.0 - alpha) * previous
