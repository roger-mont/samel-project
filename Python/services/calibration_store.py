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
    1: (slice(0,  16), slice(48, 64)),  # Canto Superior Direito
    2: (slice(0,  16), slice(32, 48)),  # Superior Meio-Direito
    3: (slice(0,  16), slice(16, 32)),  # Superior Meio-Esquerdo
    4: (slice(0,  16), slice(0,  16)),  # Canto Superior Esquerdo
    5: (slice(16, 32), slice(0,  16)),  # Canto Inferior Esquerdo
    6: (slice(16, 32), slice(16, 32)),  # Inferior Meio-Esquerdo
    7: (slice(16, 32), slice(32, 48)),  # Inferior Meio-Direito
    8: (slice(16, 32), slice(48, 64)),  # Canto Inferior Direito
}


class _BlockCalib:
    """Calibração de um único bloco 16×16 com intercepto zero obrigatório."""

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

    def sum_to_contribution(self, raw_sum: float) -> float:
        """Converte soma líquida em contribuição calibrada C_k (passa pela origem)."""
        if raw_sum <= self.tare:
            return 0.0
        net = raw_sum - self.tare
        val = float(np.polyval(self.coefficients, net))
        return max(0.0, val)

    def sum_to_newton(self, raw_sum: float) -> float:
        """Alias para sum_to_contribution mantendo compatibilidade."""
        return self.sum_to_contribution(raw_sum)

    def sum_to_kg(self, raw_sum: float) -> float:
        """Converte soma líquida em massa equivalente (kg)."""
        return self.sum_to_contribution(raw_sum) / GRAVITY_M_S2


class CalibData:
    """Dados de calibração multi-modelo para a maca inteira.

    Suporta:
      - 'multivariate_linear': P̂ = b₀ + Σ bₖ Sₖ (Modelo B - Principal)
      - 'simple_sum': P̂ = a S_T + b (Modelo A - Baseline)
      - 'multivariate_quadratic': P̂ = b₀ + Σ bₖ Sₖ + Σ cₖ Sₖ² (Modelo C)
      - 'block_curves': Curvas por bloco com correção global (Legado)
    """

    def __init__(
        self,
        blocks: dict[int, _BlockCalib] | None = None,
        correction_c: tuple[float, float] | None = None,
        active_model: str = "block_curves",
        model_b_coeffs: dict[str, float] | None = None,
        model_a_coeffs: tuple[float, float] | None = None,
        model_c_coeffs: dict[str, Any] | None = None,
        block_tares: dict[int, float] | None = None,
    ) -> None:
        self.blocks = blocks or {}
        self.correction_c = correction_c
        self.active_model = active_model
        self.model_b_coeffs = model_b_coeffs  # {"b0": float, "b1": float, ... "b8": float}
        self.model_a_coeffs = model_a_coeffs  # (a, b)
        self.model_c_coeffs = model_c_coeffs  # {"b0": float, "linear": [...], "quadratic": [...]}
        self.block_tares = block_tares or {bid: 0.0 for bid in BLOCK_REGIONS}
        self.is_valid = bool(self.blocks or self.model_b_coeffs or self.model_a_coeffs)
        self._fallback: _BlockCalib | None = self._best_fallback()

    def _best_fallback(self) -> _BlockCalib | None:
        if not self.blocks:
            return None
        return min(self.blocks.values(), key=lambda b: b.rmse_n)

    def _resolve(self, block_id: int) -> _BlockCalib | None:
        return self.blocks.get(block_id, self._fallback)

    def extract_block_net_sums(self, matrix: np.ndarray) -> np.ndarray:
        """Extrai vetor líquido [S1, ..., S8] dos 8 blocos."""
        net_sums = np.zeros(8, dtype=np.float64)
        for idx, (bid, (row_sl, col_sl)) in enumerate(sorted(BLOCK_REGIONS.items())):
            raw_s = float(matrix[row_sl, col_sl].sum())
            tare_s = self.block_tares.get(bid, 0.0)
            if calb := self.blocks.get(bid):
                tare_s = calb.tare
            net_sums[idx] = max(0.0, raw_s - tare_s)
        return net_sums

    def predict_mass(self, matrix: np.ndarray) -> float:
        """Calcula o peso estimado (kg) usando o modelo ativo."""
        if not self.is_valid:
            return float(np.sum(matrix))

        net_sums = self.extract_block_net_sums(matrix)
        total_net = float(np.sum(net_sums))

        # Se não há carga na maca após a tara, retorna 0
        if total_net <= 0.0:
            return 0.0

        if self.active_model == "multivariate_linear" and self.model_b_coeffs:
            b0 = self.model_b_coeffs.get("b0", 0.0)
            pred = b0
            for idx in range(8):
                b_k = self.model_b_coeffs.get(f"b{idx + 1}", 0.0)
                pred += b_k * net_sums[idx]
            return max(0.0, float(pred))

        if self.active_model == "simple_sum" and self.model_a_coeffs:
            a, b = self.model_a_coeffs
            return max(0.0, float(a * total_net + b))

        if self.active_model == "multivariate_quadratic" and self.model_c_coeffs:
            b0 = self.model_c_coeffs.get("b0", 0.0)
            lin = self.model_c_coeffs.get("linear", [0.0] * 8)
            quad = self.model_c_coeffs.get("quadratic", [0.0] * 8)
            pred = b0 + np.sum(np.array(lin) * net_sums) + np.sum(np.array(quad) * (net_sums ** 2))
            return max(0.0, float(pred))

        # Modelo Legado (Curvas por bloco + Correção C)
        m_base = self.matrix_to_kg(matrix)
        if self.correction_c is not None:
            a, b = self.correction_c
            return max(0.0, a * m_base + b)
        return m_base

    def matrix_to_newton(self, matrix: np.ndarray) -> float:
        """Converte a matriz 32×64 em contribuição total via blocos."""
        if not self.is_valid:
            return float(np.sum(matrix))

        total_n = 0.0
        for bid, (row_sl, col_sl) in BLOCK_REGIONS.items():
            block_sum = float(matrix[row_sl, col_sl].sum())
            calb = self._resolve(bid)
            if calb is not None:
                total_n += calb.sum_to_contribution(block_sum)
            else:
                total_n += block_sum
        return total_n

    def matrix_to_kg(self, matrix: np.ndarray) -> float:
        """Converte a matriz 32×64 em massa base (kg) = C_total / g."""
        return self.matrix_to_newton(matrix) / GRAVITY_M_S2

    def matrix_to_force_field(self, matrix: np.ndarray) -> np.ndarray:
        """Retorna matriz 32×64 com força distribuída por pixel (campo 2D)."""
        force_field = np.zeros_like(matrix, dtype=np.float64)
        for bid, (row_sl, col_sl) in BLOCK_REGIONS.items():
            calb = self._resolve(bid)
            if calb is None:
                continue
            block = matrix[row_sl, col_sl]
            block_sum = float(block.sum())
            if block_sum <= 0.0:
                continue
            c_block = calb.sum_to_contribution(block_sum)
            force_field[row_sl, col_sl] = (block / block_sum) * c_block
        return force_field

    @classmethod
    def null(cls) -> "CalibData":
        obj = object.__new__(cls)
        obj.blocks = {}
        obj.correction_c = None
        obj.active_model = "block_curves"
        obj.model_b_coeffs = None
        obj.model_a_coeffs = None
        obj.model_c_coeffs = None
        obj.block_tares = {bid: 0.0 for bid in BLOCK_REGIONS}
        obj.is_valid = False
        obj._fallback = None
        return obj


# ---------------------------------------------------------------------------
# Loader — suporta v1, v2, v3 e v4
# ---------------------------------------------------------------------------

def _build_block(bdata: dict) -> _BlockCalib:
    """Constrói _BlockCalib garantindo passagem pela origem."""
    coeffs = bdata.get("coefficients_raw_to_n") or bdata.get("coefficients", [0.0, 0.0])
    rmse_n = float(bdata.get("rmse_n", 0.0))
    rmse_kg = float(bdata.get("rmse_kg", rmse_n / GRAVITY_M_S2))

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
    """Carrega calibration.json e retorna CalibData estruturado."""
    calib_path = Path(path)
    if not calib_path.exists():
        logger.warning("calibration.json não encontrado em %s — usando fallback", calib_path)
        return CalibData.null()

    try:
        raw = json.loads(calib_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        logger.error("calibration.json inválido: %s", err)
        return CalibData.null()

    version = raw.get("version", 1)
    blocks = _parse_blocks(raw) if version in (2, 3, 4) else _parse_v1_block(raw)

    correction_c: tuple[float, float] | None = None
    if "correction_model_c" in raw:
        c_data = raw["correction_model_c"]
        if "a" in c_data and "b" in c_data:
            correction_c = (float(c_data["a"]), float(c_data["b"]))

    active_model = raw.get("active_model", "block_curves")
    model_b = raw.get("models", {}).get("multivariate_linear") or raw.get("model_b_multivariate")
    model_a_dict = raw.get("models", {}).get("simple_sum") or raw.get("model_a_simple_sum")
    model_a = (float(model_a_dict["a"]), float(model_a_dict["b"])) if model_a_dict else None
    model_c = raw.get("models", {}).get("multivariate_quadratic")

    block_tares = {bid: b.tare for bid, b in blocks.items()}

    calib = CalibData(
        blocks=blocks,
        correction_c=correction_c,
        active_model=active_model,
        model_b_coeffs=model_b,
        model_a_coeffs=model_a,
        model_c_coeffs=model_c,
        block_tares=block_tares,
    )
    logger.info("calibracao carregada (v%s, modelo=%s): %d blocos", version, active_model, len(blocks))
    return calib
