"""Pipeline físico-matemático de conversão de dados matriciais de pressão em força e massa."""
from __future__ import annotations

import logging
import numpy as np

from core.settings import CalibrationParams
from storage.calibration_store import CalibData, GRAVITY_M_S2

logger = logging.getLogger(__name__)


def compute_force_matrix(
    pressure_matrix: np.ndarray,
    params: CalibrationParams,
) -> np.ndarray:
    """Aplica deadzone sobre a matriz bruta HID (0-255).
    
    Valores <= deadzone_threshold são zerados para eliminar ruído de lençol.
    """
    snap = params.snapshot()
    deadzone = snap["deadzone_threshold"]

    result = pressure_matrix.copy()
    result[result <= deadzone] = 0.0
    return result


def compute_total_force(
    pressure_matrix: np.ndarray,
    calib: CalibData | None = None,
) -> float:
    """Converte a matriz de pressão em força total calibrada (N)."""
    if calib is not None and calib.is_valid:
        return calib.matrix_to_newton(pressure_matrix)
    return float(np.sum(pressure_matrix))


def compute_total_mass(
    pressure_matrix: np.ndarray,
    calib: CalibData | None = None,
) -> float:
    """Converte matriz de pressão em massa estimada P̂ (kg) conforme modelo ativo."""
    if calib is not None and calib.is_valid:
        return calib.predict_mass(pressure_matrix)

    force_n = compute_total_force(pressure_matrix, calib)
    return force_n / GRAVITY_M_S2


def apply_ema(current: float, previous: float, alpha: float) -> float:
    """Média Móvel Exponencial (EMA) para suavização do peso total."""
    return alpha * current + (1.0 - alpha) * previous
