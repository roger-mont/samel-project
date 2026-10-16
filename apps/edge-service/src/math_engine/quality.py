"""Indicador de qualidade da medição (item 10 das orientações).

WEIGHT_VALID só é verdadeiro quando todas as condições necessárias são atendidas.
LOW_SENSITIVITY é informativo: não invalida o peso enquanto a região de baixa
sensibilidade não for definida pela calibração.

SATURATION é só a saturação digital (RAW no limite do conversor). LOW_SENSITIVITY
(dm/dRAW alto: limite metrológico) e LOW_DATA_SUPPORT (célula no trecho entre 0 e o
primeiro nó de um setor em que a menor massa deu RAW 0) são flags separadas e
informativas (orientações de 15/10, itens 3.2 e 4).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class QualityFlags:
    tare_ok: bool
    baseline_ok: bool
    signal_stable: bool
    movement: bool
    saturation: bool
    missing_cells: bool
    out_of_calibration: bool
    low_sensitivity: bool
    model_valid: bool
    low_data_support: bool = False

    @property
    def weight_valid(self) -> bool:
        return (
            self.tare_ok
            and self.baseline_ok
            and self.signal_stable
            and not self.movement
            and not self.saturation
            and not self.missing_cells
            and not self.out_of_calibration
            and self.model_valid
        )

    def as_dict(self) -> dict[str, bool]:
        data = {k.upper(): v for k, v in asdict(self).items()}
        data["WEIGHT_VALID"] = self.weight_valid
        return data


def saturated_cells(raw_matrix: np.ndarray, saturation_raw: float) -> int:
    """Quantidade de células no limite do conversor. Não são interpoladas (item 12)."""
    return int(np.count_nonzero(raw_matrix >= saturation_raw))
