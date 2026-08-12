"""Carrega calibração multi-bloco e converte matriz de pressão bruta em kg.

Formato do calibration.json (v2):
  {
    "version": 2,
    "blocks": {
      "1": { "coefficients": [...], "tare_block_sum": 0.0, "rmse_kg": 0.21 },
      "2": { ... },
      ...
    }
  }

Estratégia de fallback por bloco:
  - Bloco com calibração própria  → usa sua curva
  - Bloco sem calibração         → usa a curva do bloco com menor RMSE disponível
  - Nenhum bloco calibrado       → retorna soma bruta
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

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
    """Calibração de um único bloco 16×16."""

    __slots__ = ("coefficients", "tare", "rmse")

    def __init__(self, coefficients: list[float], tare: float, rmse: float) -> None:
        self.coefficients = np.array(coefficients)
        self.tare = tare
        self.rmse = rmse

    def sum_to_kg(self, raw_sum: float) -> float:
        net = max(0.0, raw_sum - self.tare)
        return max(0.0, float(np.polyval(self.coefficients, net)))


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
        return min(self.blocks.values(), key=lambda b: b.rmse)

    def _resolve(self, block_id: int) -> _BlockCalib | None:
        return self.blocks.get(block_id, self._fallback)

    def matrix_to_kg(self, matrix: np.ndarray) -> float:
        """Converte a matriz 32×64 completa em kg somando cada bloco.

        Blocos com calibração própria usam sua curva.
        Blocos sem calibração usam o fallback (bloco com menor RMSE).
        """
        if not self.is_valid:
            return float(np.sum(matrix))

        total = 0.0
        for bid, (row_sl, col_sl) in BLOCK_REGIONS.items():
            block_sum = float(matrix[row_sl, col_sl].sum())
            calb = self._resolve(bid)
            if calb is not None:
                total += calb.sum_to_kg(block_sum)
            else:
                total += block_sum  # fallback bruto (nunca deve ocorrer se is_valid)

        return total

    @classmethod
    def null(cls) -> "CalibData":
        obj = object.__new__(cls)
        obj.blocks = {}
        obj.is_valid = False
        obj._fallback = None
        return obj


# ---------------------------------------------------------------------------
# Loader — suporta formato v1 (legado) e v2 (multi-bloco)
# ---------------------------------------------------------------------------

def _parse_v2_blocks(raw: dict) -> dict[int, _BlockCalib]:
    result: dict[int, _BlockCalib] = {}
    for bid_str, bdata in raw.get("blocks", {}).items():
        try:
            bid = int(bid_str)
            result[bid] = _BlockCalib(
                coefficients=bdata["coefficients"],
                tare=float(bdata.get("tare_block_sum", 0.0)),
                rmse=float(bdata.get("rmse_kg", 0.0)),
            )
        except (KeyError, ValueError) as err:
            logger.warning("bloco %s malformado no calibration.json: %s", bid_str, err)
    return result


def _parse_v1_block(raw: dict) -> dict[int, _BlockCalib]:
    bid = int(raw.get("block_id", 1))
    return {
        bid: _BlockCalib(
            coefficients=raw["coefficients"],
            tare=float(raw.get("tare_block_sum", 0.0)),
            rmse=float(raw.get("rmse_kg", 0.0)),
        )
    }


def load_calibration(path: str | Path) -> CalibData:
    """Carrega calibration.json e retorna CalibData.

    Aceita formato v1 (bloco único) e v2 (multi-bloco).
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
        if raw.get("version") == 2:
            blocks = _parse_v2_blocks(raw)
        else:
            blocks = _parse_v1_block(raw)  # migração silenciosa v1→v2
    except (KeyError, ValueError) as err:
        logger.error("erro ao interpretar calibration.json: %s", err)
        return CalibData.null()

    calib = CalibData(blocks)
    logger.info(
        "calibracao carregada: %d bloco(s) — IDs %s",
        len(blocks),
        sorted(blocks.keys()),
    )
    return calib
