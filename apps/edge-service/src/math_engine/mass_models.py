"""Modelos de conversão RAW → massa, substituíveis sem tocar na aquisição (item 13).

Interface conceitual: m̂_ij = f(s_ij; θ). Cada modelo recebe a janela já filtrada
(média RAW e média corrigida) e devolve a estimativa com as flags de calibração.

Modelos disponíveis:
- "block_curves": uma curva por setor 16×16, aplicada célula a célula (massa
  local m̂_ij = f_s(x_ij)) e somada com o ganho do setor (massa total
  M = Σ_s α_s · Σ_{ij∈s} m̂_ij). Formas de curva: pchip, linear_interp (também
  serve para o resultado de uma regressão isotônica), power, exponential, polynomial.
- "ml": modelo treinado por scripts/train_weight_model.py (somas por bloco do RAW).
- "legacy_blocks": calibration.json antigo (polinômios por bloco).
"""
from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from math_engine.ml_weight import MLWeightPredictor
from storage.calibration_store import BLOCK_REGIONS, CalibData, load_calibration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowFrame:
    raw_mean: np.ndarray        # média RAW da janela, sem baseline e sem zona morta
    corrected_mean: np.ndarray  # média de x_ij = max(0, s_ij - B_ij) na janela


@dataclass(frozen=True)
class MassEstimate:
    mass_kg: float | None
    model_valid: bool
    out_of_calibration: bool
    low_sensitivity: bool
    model_kind: str
    model_version: str
    calibration_version: str


class MassModel(ABC):
    kind: str = "none"

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @property
    @abstractmethod
    def calibration_version(self) -> str: ...

    @abstractmethod
    def estimate(self, frame: WindowFrame) -> MassEstimate: ...

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ready": self.is_ready,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
        }

    def _result(self, mass: float | None, valid: bool, out: bool = False, low: bool = False) -> MassEstimate:
        return MassEstimate(mass, valid, out, low, self.kind, self.model_version, self.calibration_version)


# ----------------------------------------------------------------------------
# Curvas f(x; θ)
# ----------------------------------------------------------------------------

def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Derivadas de Fritsch–Carlson: interpolação cúbica monótona por partes."""
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    if len(x) == 2:
        d[:] = delta[0]
        return d
    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] <= 0:
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    d[0] = _pchip_end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d


def _pchip_end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    val = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(val) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(val) > abs(3 * d0):
        return 3 * d0
    return val


def _make_pchip(raw: list[float], mass: list[float]) -> Callable[[np.ndarray], np.ndarray]:
    xk = np.asarray(raw, dtype=np.float64)
    yk = np.asarray(mass, dtype=np.float64)
    if len(xk) < 2 or np.any(np.diff(xk) <= 0):
        raise ValueError("pchip exige ao menos 2 pontos com RAW estritamente crescente")
    dk = _pchip_slopes(xk, yk)

    def f(x: np.ndarray) -> np.ndarray:
        xc = np.clip(x, xk[0], xk[-1])
        i = np.clip(np.searchsorted(xk, xc) - 1, 0, len(xk) - 2)
        h = xk[i + 1] - xk[i]
        t = (xc - xk[i]) / h
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t ** 2 * (3 - 2 * t)
        h11 = t ** 2 * (t - 1)
        return h00 * yk[i] + h10 * h * dk[i] + h01 * yk[i + 1] + h11 * h * dk[i + 1]

    return f


def _make_linear(raw: list[float], mass: list[float]) -> Callable[[np.ndarray], np.ndarray]:
    xk = np.asarray(raw, dtype=np.float64)
    yk = np.asarray(mass, dtype=np.float64)
    if len(xk) < 2 or np.any(np.diff(xk) <= 0):
        raise ValueError("linear_interp exige ao menos 2 pontos com RAW estritamente crescente")
    return lambda x: np.interp(x, xk, yk)


def build_curve(kind: str, params: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Fábrica das formas de curva f(x; θ), com x = RAW corrigido e saída em kg."""
    if kind == "pchip":
        return _make_pchip(params["raw"], params["mass_kg"])
    if kind == "linear_interp":
        return _make_linear(params["raw"], params["mass_kg"])
    if kind == "power":
        a, b = float(params["a"]), float(params["b"])
        return lambda x: a * np.power(np.maximum(x, 0.0), b)
    if kind == "exponential":
        a, b = float(params["a"]), float(params["b"])
        return lambda x: a * (np.exp(b * np.maximum(x, 0.0)) - 1.0)
    if kind == "polynomial":
        coeffs = np.asarray(params["coefficients"], dtype=np.float64)
        return lambda x: np.polyval(coeffs, x)
    raise ValueError(f"Forma de curva desconhecida: {kind}")


# ----------------------------------------------------------------------------
# Curva comum por bloco
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockCurve:
    curve: Callable[[np.ndarray], np.ndarray]
    raw_min_valid: float
    raw_max_valid: float
    mass_min_valid: float
    mass_max_valid: float
    low_sensitivity_raw: float | None   # início da região de baixa sensibilidade (ainda não definido)


class BlockCurveModel(MassModel):
    """Massa local m̂_ij = f_s(x_ij) com a curva do setor s da célula;
    massa total M = Σ_s α_s · Σ_{ij∈s} m̂_ij, com α_s = 1 enquanto não calibrado."""

    kind = "block_curves"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._curves: dict[int, BlockCurve] = {}
        self._gains: dict[int, float] = {}
        self._gains_calibrated = False
        self._raw_json: dict[str, Any] = {}
        self._model_version = "none"
        self._calibration_version = "none"
        self.load()

    def load(self) -> bool:
        self._curves = {}
        self._gains = {}
        self._gains_calibrated = False
        self._raw_json = {}
        if not self._path.exists():
            return False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._raw_json = data
            for bid_str, b in data["blocks"].items():
                self._curves[int(bid_str)] = BlockCurve(
                    curve=build_curve(b["kind"], b["params"]),
                    raw_min_valid=float(b["raw_min_valid"]),
                    raw_max_valid=float(b["raw_max_valid"]),
                    mass_min_valid=float(b["mass_min_valid"]),
                    mass_max_valid=float(b["mass_max_valid"]),
                    low_sensitivity_raw=(
                        float(b["low_sensitivity_raw"]) if b.get("low_sensitivity_raw") is not None else None
                    ),
                )
            aggregation = data.get("aggregation") or {}
            gains = aggregation.get("gains") or {}
            self._gains = {bid: float(gains.get(str(bid), 1.0)) for bid in self._curves}
            if any(g <= 0 for g in self._gains.values()):
                raise ValueError("ganhos α_s devem ser positivos")
            self._gains_calibrated = bool(aggregation.get("calibrated", False))
            self._calibration_version = str(data["calibration_version"])
            self._model_version = str(data.get("model_version", self._calibration_version))
            missing = set(BLOCK_REGIONS) - set(self._curves)
            if missing:
                logger.warning("Curvas por bloco incompletas, faltam os blocos %s", sorted(missing))
            return True
        except Exception as err:
            logger.error("block_curves.json inválido (%s): %s", self._path, err)
            self._curves = {}
            self._gains = {}
            return False

    @property
    def is_ready(self) -> bool:
        return set(self._curves) == set(BLOCK_REGIONS)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def calibration_version(self) -> str:
        return self._calibration_version

    @property
    def path(self) -> Path:
        return self._path

    @property
    def gains(self) -> dict[int, float]:
        """Ganho α_s de cada setor na soma massa local → massa total."""
        return dict(self._gains)

    @property
    def gains_calibrated(self) -> bool:
        return self._gains_calibrated

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "gains": self.gains, "gains_calibrated": self.gains_calibrated}

    def local_mass(self, block_id: int, x: np.ndarray) -> np.ndarray:
        """m̂_ij em kg para as células x do setor; abaixo de raw_min_valid a célula não conta."""
        c = self._curves[block_id]
        return np.where(x >= c.raw_min_valid, np.maximum(c.curve(x), 0.0), 0.0)

    def sector_local_sums(self, corrected: np.ndarray) -> dict[int, float]:
        """Σ m̂_ij por setor (kg), sem o ganho α_s. Base do ajuste dos ganhos."""
        return {bid: float(np.sum(self.local_mass(bid, corrected[rs, cs])))
                for bid, (rs, cs) in sorted(BLOCK_REGIONS.items())}

    def sector_out_of_calibration(self, corrected: np.ndarray) -> dict[int, int]:
        """Células acima de raw_max_valid por setor (a massa delas fica cortada no último nó)."""
        out: dict[int, int] = {}
        for bid, (rs, cs) in sorted(BLOCK_REGIONS.items()):
            c = self._curves[bid]
            x = corrected[rs, cs]
            out[bid] = int(np.count_nonzero((x >= c.raw_min_valid) & (x > c.raw_max_valid)))
        return out

    @property
    def raw_definition(self) -> dict[str, Any]:
        """Conteúdo de block_curves.json (fontes, método, pontos e layout dos setores)."""
        return self._raw_json

    def curve_samples(self, block_id: int, n: int = 160, overshoot: float = 1.3) -> list[list[float]]:
        """Pontos [RAW, massa_g] da curva do setor, de 0 até overshoot × raw_max_valid."""
        c = self._curves[block_id]
        xs = np.linspace(0.0, c.raw_max_valid * overshoot, n)
        ys = np.maximum(c.curve(xs), 0.0) * 1000.0
        return [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]

    def breakdown(self, corrected: np.ndarray, active_blocks: list[int]) -> dict[str, Any]:
        """Extração célula a célula e por setor, com a curva de cada setor."""
        cell_mass = np.zeros_like(corrected, dtype=np.float64)
        out_mask = np.zeros_like(corrected, dtype=bool)
        sectors: list[dict[str, Any]] = []
        for bid, (row_sl, col_sl) in sorted(BLOCK_REGIONS.items()):
            c = self._curves[bid]
            x = corrected[row_sl, col_sl]
            valid = x >= c.raw_min_valid
            m = self.local_mass(bid, x)
            out = valid & (x > c.raw_max_valid)
            cell_mass[row_sl, col_sl] = m
            out_mask[row_sl, col_sl] = out
            local = float(np.sum(m))
            sectors.append({
                "sector": bid,
                "active": bid in active_blocks,
                "raw_sum": round(float(np.sum(x[valid])), 2),
                "active_cells": int(np.count_nonzero(valid)),
                "max_raw": round(float(np.max(x)), 2),
                "local_mass_kg": round(local, 4),
                "gain": self._gains[bid],
                "mass_kg": round(local * self._gains[bid], 4),
                "cells_out_of_calibration": int(np.count_nonzero(out)),
                "cells_low_sensitivity": (int(np.count_nonzero(valid & (x > c.low_sensitivity_raw)))
                                          if c.low_sensitivity_raw is not None else 0),
            })
        total = sum(s["mass_kg"] for s in sectors if s["active"])
        return {
            "sectors": sectors,
            "gains_calibrated": self._gains_calibrated,
            "total_mass_kg": round(total, 4),
            "cell_mass_g": np.round(cell_mass * 1000.0, 1).tolist(),
            "corrected_raw": np.round(corrected, 1).tolist(),
            "out_of_calibration": out_mask.astype(int).tolist(),
        }

    def estimate(self, frame: WindowFrame) -> MassEstimate:
        if not self.is_ready:
            return self._result(None, False)
        total = 0.0
        out_of_calibration = False
        low_sensitivity = False
        for bid, (row_sl, col_sl) in BLOCK_REGIONS.items():
            c = self._curves[bid]
            x = frame.corrected_mean[row_sl, col_sl]
            active = x[x >= c.raw_min_valid]
            if active.size == 0:
                continue
            if np.any(active > c.raw_max_valid):
                out_of_calibration = True
            if c.low_sensitivity_raw is not None and np.any(active > c.low_sensitivity_raw):
                low_sensitivity = True
            total += self._gains[bid] * float(np.sum(np.maximum(c.curve(active), 0.0)))
        return self._result(total, True, out_of_calibration, low_sensitivity)


# ----------------------------------------------------------------------------
# Modelos existentes, adaptados à interface
# ----------------------------------------------------------------------------

def _file_tag(prefix: str, path: Path) -> str:
    if not path.exists():
        return "none"
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:8]
    return f"{prefix}-{digest}"


class MLModel(MassModel):
    """Modelo de ML atual. Recebe a média RAW sem baseline e sem zona morta,
    que é a mesma entrada usada no treino (somas por bloco da matriz crua)."""

    kind = "ml"

    def __init__(self, predictor: MLWeightPredictor | None = None) -> None:
        self.predictor = predictor or MLWeightPredictor()
        self._version = _file_tag("ML", self.predictor.model_path)

    def load(self) -> bool:
        ok = self.predictor.load()
        self._version = _file_tag("ML", self.predictor.model_path)
        return ok

    @property
    def is_ready(self) -> bool:
        return self.predictor.is_loaded

    @property
    def model_version(self) -> str:
        return str(self.predictor.metadata.get("model_version", self._version))

    @property
    def calibration_version(self) -> str:
        return str(self.predictor.metadata.get("calibration_version", self._version))

    def estimate(self, frame: WindowFrame) -> MassEstimate:
        mass = self.predictor.predict(frame.raw_mean)
        if mass is None:
            return self._result(None, False)
        meta = self.predictor.metadata
        lo, hi = meta.get("min_weight_kg"), meta.get("max_weight_kg")
        out = lo is not None and hi is not None and not (float(lo) <= mass <= float(hi))
        return self._result(mass, True, out)


class LegacyBlockModel(MassModel):
    """calibration.json antigo (polinômios por bloco sobre a soma RAW)."""

    kind = "legacy_blocks"

    def __init__(self, path: Path) -> None:
        self._path = path
        self.calib: CalibData = load_calibration(path)
        self._version = _file_tag("CAL", path)

    def load(self) -> bool:
        self.calib = load_calibration(self._path)
        self._version = _file_tag("CAL", self._path)
        return self.calib.is_valid

    @property
    def is_ready(self) -> bool:
        return self.calib.is_valid

    @property
    def model_version(self) -> str:
        return f"{self._version}:{self.calib.active_model}"

    @property
    def calibration_version(self) -> str:
        return self._version

    def estimate(self, frame: WindowFrame) -> MassEstimate:
        if not self.is_ready:
            return self._result(None, False)
        return self._result(float(self.calib.predict_mass(frame.raw_mean)), True)


class ModelRegistry:
    """Escolhe o modelo ativo entre os registrados.

    'auto' segue a prioridade block_curves → ml → legacy_blocks, mas só entre os
    modelos que o worker registrou (hoje, apenas block_curves).
    """

    AUTO_ORDER = ("block_curves", "ml", "legacy_blocks")

    def __init__(self, models: dict[str, MassModel], selection: str = "auto") -> None:
        self.models = models
        self.selection = selection

    def active(self) -> MassModel | None:
        if self.selection != "auto":
            model = self.models.get(self.selection)
            return model if model is not None and model.is_ready else None
        for name in self.AUTO_ORDER:
            model = self.models.get(name)
            if model is not None and model.is_ready:
                return model
        return None

    def reload(self) -> dict[str, bool]:
        return {name: bool(getattr(m, "load")()) for name, m in self.models.items()}

    def describe(self) -> dict[str, Any]:
        active = self.active()
        return {
            "selection": self.selection,
            "active": active.kind if active else None,
            "models": {name: m.describe() for name, m in self.models.items()},
        }
