"""Pipeline de pressão — aplica calibração e retorna Newton/kg por bloco.

Fluxo:
  matriz HID bruta (0-255)
    → deadzone (compute_force_matrix)
    → conversão por bloco via curva polinomial → Newton (compute_total_force)
    → divisão por g → kg (compute_total_mass)
    → EMA temporal (apply_ema)
"""
from __future__ import annotations

import logging

import numpy as np

from config.settings import CalibrationParams
from services.calibration_store import CalibData, GRAVITY_M_S2

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


def compute_total_force(
    pressure_matrix: np.ndarray,
    calib: CalibData | None = None,
) -> float:
    """Converte a matriz de pressão em contribuição total calibrada C_total.

    Usa calibração multi-bloco. Se `calib` for None ou inválido,
    retorna a soma bruta como fallback.
    """
    if calib is not None and calib.is_valid:
        return calib.matrix_to_newton(pressure_matrix)

    logger.debug("sem calibração válida — usando soma bruta")
    return float(np.sum(pressure_matrix))


def apply_correction_c(m_fisica: float, calib: CalibData | None = None) -> float:
    """Modelo C (Metodologia §17): Aplica correção linear global m̂ = max(0, a * m_fisica + b)."""
    if m_fisica <= 0.0:
        return 0.0
    if calib is None or calib.correction_c is None:
        return m_fisica
    a, b = calib.correction_c
    return max(0.0, a * m_fisica + b)


def compute_total_mass(
    pressure_matrix: np.ndarray,
    calib: CalibData | None = None,
) -> float:
    """Converte matriz de pressão em massa estimada P̂ (kg) conforme modelo ativo."""
    if calib is not None and calib.is_valid:
        return calib.predict_mass(pressure_matrix)

    force_n = compute_total_force(pressure_matrix, calib)
    return force_n / GRAVITY_M_S2


def compute_model_a(
    pressure_matrix: np.ndarray,
    calib: CalibData | None = None,
) -> float:
    """Modelo A (Metodologia §15): Soma direta de forças calibradas por bloco (massa física).

    F_A = Σ_k F_k(soma_k)  →  m_A = F_A / g
    """
    force_n = compute_total_force(pressure_matrix, calib)
    return force_n / GRAVITY_M_S2


def _trapezoid_1d(y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Calcula a regra trapezoidal 1D de forma compatível com NumPy 1.x e 2.x."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, axis=axis)
    if hasattr(np, "trapz"):
        return getattr(np, "trapz")(y, axis=axis)
    if y.shape[axis] < 2:
        return np.sum(y, axis=axis)
    return np.sum(y, axis=axis) - 0.5 * (np.take(y, 0, axis=axis) + np.take(y, -1, axis=axis))


def compute_model_b(
    pressure_matrix: np.ndarray,
    calib: CalibData | None = None,
) -> float:
    """Modelo B (Metodologia §16): Integração numérica 2D trapezoidal do campo de força.

    F_B ≈ ∬ p(x,y) dA  →  m_B = F_B / g
    """
    if calib is None or not calib.is_valid:
        return compute_total_mass(pressure_matrix, calib)

    force_field = calib.matrix_to_force_field(pressure_matrix)
    if float(np.sum(force_field)) <= 0.0:
        return 0.0

    # Integração trapezoidal 2D: ao longo das colunas e depois das linhas
    f_rows = _trapezoid_1d(force_field, axis=1)
    integrated_n = float(_trapezoid_1d(f_rows, axis=0))

    # Proteção para grades com poucos pontos ativos discretos
    if integrated_n <= 0.0:
        integrated_n = float(np.sum(force_field))

    return integrated_n / GRAVITY_M_S2


def apply_ema(current: float, previous: float, alpha: float) -> float:
    """Média Móvel Exponencial para suavização do peso total exibido.

    EMA = α × atual + (1 − α) × anterior
    """
    return alpha * current + (1.0 - alpha) * previous
