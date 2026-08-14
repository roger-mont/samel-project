"""Carrega calibração multi-bloco e converte matriz de pressão bruta em Newton/kg.

Formato do calibration.json (v3):
  {
    "version": 3,
    "blocks": {
      "1": {
        "unit": "newton",
        "coefficients_raw_to_n": [...],
        "tare_block_sum": 0.0,
        "rmse_n": 4.28,
        "rmse_kg": 0.44,
        "calibration_points": [
          { "kg": 0.0, "n": 0.0, "raw_sum": 241.46 }
        ]
      }
    }
  }

Estratégia de fallback por bloco:
  - Bloco com calibração própria  → usa sua curva
  - Bloco sem calibração         → usa a curva do bloco com menor RMSE
  - Nenhum bloco calibrado       → retorna soma bruta
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GRAVITY_M_S2: float = 9.81

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


class _BlockCalib:
    """Calibração de um único bloco 16×16.

    A curva polinomial converte net_sum → Newton.
    A conversão para kg é feita dividindo por g (9.81 m/s²).
    """

    __slots__ = ("coefficients", "tare", "rmse_n", "rmse_kg")

    def __init__(
        self,
        coefficients: list[float],
        tare: float,
        rmse_n: float,
        rmse_kg: float,
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.tare = tare
        self.rmse_n = rmse_n
        self.rmse_kg = rmse_kg

    def sum_to_newton(self, raw_sum: float) -> float:
        """Converte soma bruta do bloco em força (Newton) via polinômio."""
        if raw_sum <= self.tare:
            return 0.0
        net = raw_sum - self.tare
        val = float(np.polyval(self.coefficients, net))
        return max(0.0, val)

    def sum_to_kg(self, raw_sum: float) -> float:
        """Converte soma bruta do bloco em massa (kg) = F / g."""
        return self.sum_to_newton(raw_sum) / GRAVITY_M_S2


class CalibData:
    """Dados de calibração para todos os blocos disponíveis.

    Atributos públicos:
        is_valid  — True se pelo menos 1 bloco foi calibrado
        blocks    — dict[block_id_int → _BlockCalib]
    """

    def __init__(self, blocks: dict[int, _BlockCalib]) -> None:
        self.blocks = blocks
        self.is_valid = len(blocks) > 0
        self._fallback: _BlockCalib | None = self._best_fallback()

    def _best_fallback(self) -> _BlockCalib | None:
        if not self.blocks:
            return None
        return min(self.blocks.values(), key=lambda b: b.rmse_n)

    def _resolve(self, block_id: int) -> _BlockCalib | None:
        return self.blocks.get(block_id, self._fallback)

    def matrix_to_newton(self, matrix: np.ndarray) -> float:
        """Converte a matriz 32×64 completa em força total (Newton)."""
        if not self.is_valid:
            return float(np.sum(matrix))

        total_n = 0.0
        for bid, (row_sl, col_sl) in BLOCK_REGIONS.items():
            block_sum = float(matrix[row_sl, col_sl].sum())
            calb = self._resolve(bid)
            if calb is not None:
                total_n += calb.sum_to_newton(block_sum)
            else:
                total_n += block_sum
        return total_n

    def matrix_to_kg(self, matrix: np.ndarray) -> float:
        """Converte a matriz 32×64 completa em massa (kg) = F_total / g."""
        return self.matrix_to_newton(matrix) / GRAVITY_M_S2

    def matrix_to_force_field(self, matrix: np.ndarray) -> np.ndarray:
        """Retorna matriz 32×64 com força em Newton por pixel (campo 2D)."""
        force_field = np.zeros_like(matrix, dtype=np.float64)
        for bid, (row_sl, col_sl) in BLOCK_REGIONS.items():
            calb = self._resolve(bid)
            if calb is None:
                continue
            block = matrix[row_sl, col_sl]
            block_sum = float(block.sum())
            if block_sum <= 0.0:
                continue
            f_total_block = calb.sum_to_newton(block_sum)
            force_field[row_sl, col_sl] = (block / block_sum) * f_total_block
        return force_field

    @classmethod
    def null(cls) -> "CalibData":
        obj = object.__new__(cls)
        obj.blocks = {}
        obj.is_valid = False
        obj._fallback = None
        return obj


# ---------------------------------------------------------------------------
# Loader — suporta v1, v2 e v3
# ---------------------------------------------------------------------------

def _build_block(bdata: dict) -> _BlockCalib:
    """Constrói _BlockCalib a partir de um dict de bloco (v2 ou v3)."""
    unit = bdata.get("unit", "kg")

    if unit == "newton":
        coeffs = bdata["coefficients_raw_to_n"]
        rmse_n = float(bdata.get("rmse_n", 0.0))
        rmse_kg = float(bdata.get("rmse_kg", rmse_n / GRAVITY_M_S2))
    else:
        # v2 legado: coeficientes em kg → converter para Newton
        coeffs_kg = np.array(bdata["coefficients"])
        coeffs = (coeffs_kg * GRAVITY_M_S2).tolist()
        rmse_kg = float(bdata.get("rmse_kg", 0.0))
        rmse_n = rmse_kg * GRAVITY_M_S2

    return _BlockCalib(
        coefficients=coeffs,
        tare=float(bdata.get("tare_block_sum", 0.0)),
        rmse_n=rmse_n,
        rmse_kg=rmse_kg,
    )


def _parse_blocks(raw: dict) -> dict[int, _BlockCalib]:
    result: dict[int, _BlockCalib] = {}
    for bid_str, bdata in raw.get("blocks", {}).items():
        try:
            result[int(bid_str)] = _build_block(bdata)
        except (KeyError, ValueError) as err:
            logger.warning("bloco %s malformado no calibration.json: %s", bid_str, err)
    return result


def _parse_v1_block(raw: dict) -> dict[int, _BlockCalib]:
    bid = int(raw.get("block_id", 1))
    coeffs_kg = np.array(raw["coefficients"])
    coeffs_n = (coeffs_kg * GRAVITY_M_S2).tolist()
    rmse_kg = float(raw.get("rmse_kg", 0.0))
    return {
        bid: _BlockCalib(
            coefficients=coeffs_n,
            tare=float(raw.get("tare_block_sum", 0.0)),
            rmse_n=rmse_kg * GRAVITY_M_S2,
            rmse_kg=rmse_kg,
        )
    }


def load_calibration(path: str | Path) -> CalibData:
    """Carrega calibration.json e retorna CalibData.

    Aceita formato v1 (bloco único), v2 (multi-bloco, kg) e v3 (multi-bloco, Newton).
    Retorna CalibData.null() se arquivo ausente ou inválido.
    """
    calib_path = Path(path)
    if not calib_path.exists():
        logger.warning("calibration.json não encontrado em %s — peso sem calibração", calib_path)
        return CalibData.null()

    try:
        raw = json.loads(calib_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        logger.error("calibration.json inválido: %s", err)
        return CalibData.null()

    try:
        version = raw.get("version", 1)
        if version in (2, 3):
            blocks = _parse_blocks(raw)
        else:
            blocks = _parse_v1_block(raw)
    except (KeyError, ValueError) as err:
        logger.error("erro ao interpretar calibration.json: %s", err)
        return CalibData.null()

    calib = CalibData(blocks)
    logger.info(
        "calibracao carregada (v%s): %d bloco(s) — IDs %s",
        version,
        len(blocks),
        sorted(blocks.keys()),
    )
    return calib
