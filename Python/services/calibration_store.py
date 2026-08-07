"""Carrega calibração de bloco e converte matriz de pressão bruta em kg.

Estratégia de escala (um bloco calibrado → 8 blocos):
  A calibração feita com o bloco 1 é aplicada a todos os blocos.
  Isso assume que os sensores são do mesmo lote e têm resposta uniforme.
  Cada bloco tem sua soma bruta convertida individualmente pela curva e somada.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Mapeamento block_id (1-8) → (row_slice, col_slice) na matriz 32×64
# Espelha o algoritmo ButtonShowDeal do C# e o HidFrameReader
BLOCK_REGIONS: dict[int, tuple[slice, slice]] = {
    1: (slice(16, 32), slice(0,  16)),
    2: (slice(16, 32), slice(16, 32)),
    3: (slice(16, 32), slice(32, 48)),
    4: (slice(16, 32), slice(48, 64)),
    5: (slice(0,  16), slice(48, 64)),
    6: (slice(0,  16), slice(32, 48)),
    7: (slice(0,  16), slice(16, 32)),
    8: (slice(0,  16), slice(0,  16)),
}


class CalibData:
    """Dados de calibração carregados do JSON.

    Atributos:
        coefficients: coeficientes do polinômio net_sum → kg
        tare_block_sum: soma bruta do bloco a 0 kg
        rmse_kg: erro quadrático médio do ajuste
        is_valid: True se o arquivo foi carregado com sucesso
    """

    def __init__(
        self,
        coefficients: list[float],
        tare_block_sum: float,
        rmse_kg: float,
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.tare_block_sum = tare_block_sum
        self.rmse_kg = rmse_kg
        self.is_valid = True

    @classmethod
    def null(cls) -> "CalibData":
        """Retorna objeto inválido — usado como fallback quando não há calibração."""
        obj = object.__new__(cls)
        obj.coefficients = np.array([])
        obj.tare_block_sum = 0.0
        obj.rmse_kg = 0.0
        obj.is_valid = False
        return obj

    def block_sum_to_kg(self, raw_sum: float) -> float:
        """Converte a soma bruta de um bloco 16×16 em kg via polinômio calibrado."""
        net = raw_sum - self.tare_block_sum
        kg = float(np.polyval(self.coefficients, max(0.0, net)))
        return max(0.0, kg)

    def matrix_to_kg(self, matrix: np.ndarray) -> float:
        """Converte a matriz 32×64 completa em kg somando todos os blocos.

        Cada bloco é convertido individualmente pela curva calibrada.
        """
        if not self.is_valid:
            return float(np.sum(matrix))  # fallback: soma bruta

        total_kg = 0.0
        for row_sl, col_sl in BLOCK_REGIONS.values():
            block_sum = float(matrix[row_sl, col_sl].sum())
            total_kg += self.block_sum_to_kg(block_sum)

        return total_kg


def load_calibration(path: str | Path) -> CalibData:
    """Carrega calibration.json e retorna CalibData.

    Retorna CalibData.null() se o arquivo não existir ou estiver malformado.
    """
    calib_path = Path(path)
    if not calib_path.exists():
        logger.warning("calibration.json não encontrado em %s — sem calibração", calib_path)
        return CalibData.null()

    try:
        raw = json.loads(calib_path.read_text(encoding="utf-8"))
        calib = CalibData(
            coefficients=raw["coefficients"],
            tare_block_sum=float(raw.get("tare_block_sum", 0.0)),
            rmse_kg=float(raw.get("rmse_kg", 0.0)),
        )
        logger.info(
            "calibração carregada: bloco=%s grau=%s RMSE=%.4f kg",
            raw.get("block_id", "?"),
            raw.get("polynomial_degree", "?"),
            calib.rmse_kg,
        )
        return calib
    except (KeyError, ValueError, json.JSONDecodeError) as err:
        logger.error("erro ao carregar calibration.json: %s", err)
        return CalibData.null()
