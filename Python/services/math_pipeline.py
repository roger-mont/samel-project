"""Pipeline matemático vetorizado: ADC → Tensão → Resistência → Força → Peso."""
from __future__ import annotations

import logging

import numpy as np

from config.settings import CalibrationParams, GRAVITY

logger = logging.getLogger(__name__)


def compute_force_matrix(
    adc_matrix: np.ndarray,
    params: CalibrationParams,
) -> np.ndarray:
    """Converte matriz ADC bruta em matriz de forças (N).

    Pipeline:
      V_out   = ADC × (VCC / resolução)
      R_FSR   = R_pull × (VCC / V_out − 1)
      C       = 1 / R_FSR
      F(x,y)  = M × C + B
      Deadzone: F[F < threshold] = 0
    """
    snap = params.snapshot()
    vcc = snap["vcc"]
    resolution = snap["adc_resolution"]
    pulldown = snap["pulldown_resistor"]
    factor_m = snap["factor_m"]
    offset_b = snap["offset_b"]
    deadzone = snap["deadzone_threshold"]

    if resolution == 0:
        return np.zeros_like(adc_matrix)

    v_out = adc_matrix * (vcc / resolution)

    # Guard: V_out == 0 → R_FSR infinito → sensor sem pressão
    safe_vout = np.where(v_out > 0, v_out, np.inf)
    r_fsr = pulldown * (vcc / safe_vout - 1.0)

    # Guard: R_FSR <= 0 → divisão por zero na condutância
    safe_rfsr = np.where(r_fsr > 0, r_fsr, np.inf)
    conductance = 1.0 / safe_rfsr

    force = factor_m * conductance + offset_b
    force = np.maximum(force, 0.0)

    _log_pipeline_steps(adc_matrix, v_out, r_fsr, conductance, force, deadzone)

    # Deadzone: zera forças abaixo do limiar
    force[force < deadzone] = 0.0

    return force


def _log_pipeline_steps(
    adc: np.ndarray,
    v_out: np.ndarray,
    r_fsr: np.ndarray,
    conductance: np.ndarray,
    force_pre_deadzone: np.ndarray,
    deadzone: float,
) -> None:
    """Loga etapas intermediarias do pipeline quando ha dados."""
    adc_max = float(np.max(adc))
    if adc_max == 0:
        return

    max_idx = np.unravel_index(np.argmax(adc), adc.shape)
    logger.info(
        "PIPELINE pico em [%d,%d]: "
        "ADC=%d → V_out=%.4fV → R_FSR=%.0fΩ → C=%.2e S → "
        "F=%.2e N (deadzone=%.4f → %s)",
        max_idx[0],
        max_idx[1],
        int(adc[max_idx]),
        float(v_out[max_idx]),
        float(r_fsr[max_idx]),
        float(conductance[max_idx]),
        float(force_pre_deadzone[max_idx]),
        deadzone,
        "PASSA" if float(force_pre_deadzone[max_idx]) >= deadzone else "ZERADO",
    )


def compute_total_mass(force_matrix: np.ndarray) -> float:
    """Soma todas as forças e converte para massa em Kg."""
    total_force = float(np.sum(force_matrix))
    return total_force / GRAVITY


def apply_ema(current: float, previous: float, alpha: float) -> float:
    """Média Móvel Exponencial para suavização do peso exibido.

    EMA = α × atual + (1 − α) × anterior
    """
    return alpha * current + (1.0 - alpha) * previous
