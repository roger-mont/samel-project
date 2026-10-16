"""Modelos candidatos RAW → massa local, para comparar com a curva em vigor.

Só diagnóstico: nenhum destes modelos publica peso clínico. Cada candidato é
treinado com os mesmos pontos de calibração de célula que estão em
block_curves.json, em dois modos:

- "sector": um ajuste por setor, com os pontos médios do setor (blocks[s].fit);
  o candidato "pchip_isotonica" é a própria curva em vigor;
- "global": um ajuste único com as células de todos os setores (CSVs idênticos
  entram uma vez só), o que equivale a usar uma curva só para a manta.

Todos os candidatos recebem também o ponto (RAW 0, massa 0): célula vazia pesa
zero. As curvas isotônicas já têm esse nó; sem ele, os modelos de ML dão algumas
dezenas de gramas a cada célula com RAW baixo, e a soma de centenas de células
com carga leve infla a massa total.

Depois do ajuste, cada modelo vira uma tabela RAW → massa (passo de 0,1 RAW,
de 0 a raw_max_valid do setor) e a avaliação ao vivo é uma interpolação, então
comparar todos os modelos custa o mesmo que aplicar uma curva.
"""
from __future__ import annotations

import logging
import threading
import warnings
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from math_engine.mass_models import build_curve
from storage.calibration_store import BLOCK_REGIONS

logger = logging.getLogger(__name__)

Curve = Callable[[np.ndarray], np.ndarray]
Fitter = Callable[[np.ndarray, np.ndarray], Curve]

TABLE_STEP_RAW = 0.1


# ----------------------------------------------------------------------------
# Ajuste isotônico (o mesmo do gerador de curvas)
# ----------------------------------------------------------------------------

def isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators: menor ajuste por mínimos quadrados não decrescente."""
    blocks: list[list[float]] = []  # [soma, peso]
    for v in y:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, w = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += w
    out: list[float] = []
    for s, w in blocks:
        out.extend([s / w] * int(w))
    return np.array(out)


def build_knots(masses: np.ndarray, raw_iso: np.ndarray) -> tuple[list[float], list[float]]:
    """Nós RAW → massa estritamente crescentes. RAWs empatados viram um nó com a massa média."""
    grouped: dict[float, list[float]] = defaultdict(list)
    for m, r in zip(masses, raw_iso):
        grouped[round(float(r), 6)].append(float(m))
    raw_knots = [0.0]
    mass_knots = [0.0]
    for r in sorted(grouped):
        if r <= 0.0:
            continue
        raw_knots.append(r)
        mass_knots.append(float(np.mean(grouped[r])))
    return raw_knots, mass_knots


# ----------------------------------------------------------------------------
# Modelos: fit(raw, massa_kg) → f(raw) em kg
# ----------------------------------------------------------------------------

def _clip_range(f: Curve, lo: float, hi: float) -> Curve:
    return lambda x: np.clip(f(np.asarray(x, dtype=float)), lo, hi)


def fit_isotonic(kind: str) -> Fitter:
    def fit(x: np.ndarray, m: np.ndarray) -> Curve:
        order = np.argsort(m, kind="stable")
        rk, mk = build_knots(m[order], isotonic_increasing(x[order]))
        return build_curve(kind, {"raw": rk, "mass_kg": mk})
    return fit


def fit_power(x: np.ndarray, m: np.ndarray) -> Curve:
    ok = (x > 0) & (m > 0)
    b, log_a = np.polyfit(np.log(x[ok]), np.log(m[ok]), 1)
    return _clip_range(lambda v: np.exp(log_a) * np.power(np.maximum(v, 0.0), b), 0.0, 1.5 * m.max())


def fit_exponential(x: np.ndarray, m: np.ndarray) -> Curve:
    from scipy.optimize import curve_fit

    def f(v, a, b):
        return a * (np.exp(b * np.maximum(v, 0.0)) - 1.0)
    p, _ = curve_fit(f, x, m, p0=(m.max() / 10.0, 0.02), maxfev=20000)
    return _clip_range(lambda v: f(v, *p), 0.0, 1.5 * m.max())


def fit_poly(deg: int) -> Fitter:
    def fit(x: np.ndarray, m: np.ndarray) -> Curve:
        c = np.polyfit(x, m, deg)
        return _clip_range(lambda v: np.polyval(c, v), 0.0, 1.5 * m.max())
    return fit


def fit_newton(x: np.ndarray, m: np.ndarray) -> Curve:
    """Polinômio interpolador de Newton por todos os pontos (diferenças divididas)."""
    xs, idx = np.unique(x, return_index=True)
    ys = m[idx].astype(float)
    n = len(xs)
    coef = ys.copy()
    for j in range(1, n):
        coef[j:] = (coef[j:] - coef[j - 1:-1]) / (xs[j:] - xs[:n - j])

    def f(v):
        v = np.asarray(v, dtype=float)
        out = np.full_like(v, coef[-1])
        for k in range(n - 2, -1, -1):
            out = out * (v - xs[k]) + coef[k]
        return out
    return _clip_range(f, 0.0, 1.5 * m.max())


def sklearn_fitter(make: Callable[[], Any]) -> Fitter:
    """Modelo do scikit-learn com entrada padronizada; X pode ter 1 coluna (RAW) ou mais."""
    def fit(X: np.ndarray, m: np.ndarray) -> Curve:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), make())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(np.asarray(X, dtype=float).reshape(len(X), -1), m)
        return lambda v: np.clip(model.predict(np.asarray(v, dtype=float).reshape(len(v), -1)), 0.0, None)
    return fit


def _gpr():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)


def _svr():
    from sklearn.svm import SVR
    return SVR(C=1.0, epsilon=0.005, gamma="scale")


def _knn():
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=2, weights="distance")


def _rf():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300, random_state=0)


def _gb():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0)


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0)


def _mlp():
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(16, 16), solver="lbfgs", alpha=1e-3, max_iter=5000, random_state=0)


CURVE_FITTERS: dict[str, Fitter] = {
    "pchip_isotonica": fit_isotonic("pchip"),
    "linear_isotonica": fit_isotonic("linear_interp"),
    "potencia": fit_power,
    "exponencial": fit_exponential,
    "polinomio_2": fit_poly(2),
    "polinomio_3": fit_poly(3),
    "newton": fit_newton,
}
ML_FITTERS: dict[str, Fitter] = {
    "ml_gpr": sklearn_fitter(_gpr),
    "ml_svr": sklearn_fitter(_svr),
    "ml_knn": sklearn_fitter(_knn),
    "ml_random_forest": sklearn_fitter(_rf),
    "ml_gradient_boosting": sklearn_fitter(_gb),
    "ml_xgboost": sklearn_fitter(_xgb),
    "ml_mlp": sklearn_fitter(_mlp),
}
ALL_FITTERS: dict[str, Fitter] = {**CURVE_FITTERS, **ML_FITTERS}

# Os 8 candidatos da tela (no máximo 8 séries por gráfico). Ficam de fora os que
# erraram mais ou empataram com outro da mesma família em scripts/comparar_modelos_massa.py.
LIVE_MODELS: list[tuple[str, str]] = [
    ("pchip_isotonica", "PCHIP isotônica"),
    ("linear_isotonica", "Linear isotônica"),
    ("exponencial", "Exponencial"),
    ("polinomio_3", "Polinômio grau 3"),
    ("ml_gpr", "ML · GPR"),
    ("ml_random_forest", "ML · Random Forest"),
    ("ml_xgboost", "ML · XGBoost"),
    ("ml_mlp", "ML · Rede neural (MLP)"),
]
MODES = ("sector", "global")


# ----------------------------------------------------------------------------
# Comparação
# ----------------------------------------------------------------------------

class _Table:
    """f(raw) tabelada de 0 a raw_max; acima de raw_max a célula fica no topo, como na curva em vigor."""

    __slots__ = ("raw", "mass", "raw_max")

    def __init__(self, f: Curve, raw_max: float) -> None:
        self.raw = np.arange(0.0, raw_max + TABLE_STEP_RAW, TABLE_STEP_RAW)
        self.raw[-1] = raw_max
        self.mass = np.maximum(np.asarray(f(self.raw), dtype=float), 0.0)
        self.raw_max = raw_max

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.raw, self.mass)


class ModelComparison:
    """Treina os candidatos com os pontos de block_curves.json e aplica todos à mesma janela."""

    def __init__(self, definition: dict[str, Any], gains: dict[int, float] | None = None,
                 models: list[tuple[str, str]] | None = None) -> None:
        self.models = models or LIVE_MODELS
        self.gains = {bid: float((gains or {}).get(bid, 1.0)) for bid in BLOCK_REGIONS}
        self.version = f"{definition.get('calibration_version')}@{definition.get('generated_at')}"
        self.raw_min_valid: dict[int, float] = {}
        self.raw_max_valid: dict[int, float] = {}
        # tables[mode][model][sector]
        self.tables: dict[str, dict[str, dict[int, _Table]]] = {m: {} for m in MODES}
        self.errors: dict[str, str] = {}
        self._fit(definition)

    # ------------------------------------------------------------------ ajuste

    @staticmethod
    def _sector_points(block: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        fit = block.get("fit") or []
        x = np.array([0.0] + [p["raw_mean"] for p in fit], dtype=float)
        m = np.array([0.0] + [p["mass_kg"] for p in fit], dtype=float)
        return x, m

    @staticmethod
    def _global_points(definition: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = [0.0]
        ms: list[float] = [0.0]
        seen: set[str] = set()
        for src in definition.get("sources", []):
            if src.get("sha1") in seen:
                continue
            seen.add(src.get("sha1"))
            for p in src.get("points_up", []):
                if p["mass_kg"] > 0:
                    xs.append(float(p["raw_corrected"]))
                    ms.append(float(p["mass_kg"]))
        return np.array(xs), np.array(ms)

    def _fit(self, definition: dict[str, Any]) -> None:
        blocks = definition["blocks"]
        for bid in BLOCK_REGIONS:
            b = blocks[str(bid)]
            self.raw_min_valid[bid] = float(b["raw_min_valid"])
            self.raw_max_valid[bid] = float(b["raw_max_valid"])
        gx, gm = self._global_points(definition)
        global_raw_max = float(max(self.raw_max_valid.values()))
        if len(gx):
            global_raw_max = float(np.max(isotonic_increasing(gx[np.argsort(gm, kind="stable")])))

        for key, _label in self.models:
            fitter = ALL_FITTERS[key]
            try:
                per_sector: dict[int, _Table] = {}
                for bid in BLOCK_REGIONS:
                    x, m = self._sector_points(blocks[str(bid)])
                    per_sector[bid] = _Table(fitter(x, m), self.raw_max_valid[bid])
                self.tables["sector"][key] = per_sector
                g = _Table(fitter(gx, gm), global_raw_max)
                self.tables["global"][key] = {bid: g for bid in BLOCK_REGIONS}
            except Exception as err:  # biblioteca ausente ou ajuste que não converge
                logger.warning("Modelo candidato %s indisponível: %s", key, err)
                self.errors[key] = str(err)
                self.tables["sector"].pop(key, None)
                self.tables["global"].pop(key, None)

    # --------------------------------------------------------------- avaliação

    def evaluate(self, corrected: np.ndarray, active_blocks: list[int]) -> dict[str, Any]:
        """Massa local por setor e total com α_s, para cada modelo e modo."""
        out: dict[str, Any] = {}
        for mode in MODES:
            rows = []
            for key, label in self.models:
                tables = self.tables[mode].get(key)
                if tables is None:
                    rows.append({"model": key, "label": label, "available": False, "error": self.errors.get(key)})
                    continue
                sectors = {}
                total = 0.0
                for bid, (rs, cs) in sorted(BLOCK_REGIONS.items()):
                    x = corrected[rs, cs]
                    x = x[x >= self.raw_min_valid[bid]]
                    local = float(np.sum(tables[bid](x))) if x.size else 0.0
                    mass = local * self.gains[bid]
                    sectors[str(bid)] = round(mass, 4)
                    if bid in active_blocks:
                        total += mass
                rows.append({"model": key, "label": label, "available": True,
                             "total_mass_kg": round(total, 4), "sectors_kg": sectors})
            out[mode] = rows
        return out

    def curves(self, mode: str, sector: int, n: int = 120) -> list[dict[str, Any]]:
        """Pontos [RAW, massa_g] de cada modelo para um setor, de 0 a raw_max do modelo."""
        result = []
        for key, label in self.models:
            tables = self.tables[mode].get(key)
            if tables is None:
                result.append({"model": key, "label": label, "available": False})
                continue
            t = tables[sector]
            xs = np.linspace(0.0, t.raw_max, n)
            ys = t(xs) * 1000.0
            result.append({"model": key, "label": label, "available": True, "raw_max": round(t.raw_max, 3),
                           "samples_raw_mass_g": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(xs, ys)]})
        return result


class ComparisonCache:
    """Guarda a comparação treinada e refaz só quando block_curves.json muda."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: ModelComparison | None = None

    def get(self, definition: dict[str, Any], gains: dict[int, float]) -> ModelComparison:
        version = f"{definition.get('calibration_version')}@{definition.get('generated_at')}"
        with self._lock:
            v = self._value
            if v is None or v.version != version or v.gains != {b: float(gains.get(b, 1.0)) for b in BLOCK_REGIONS}:
                v = ModelComparison(definition, gains)
                self._value = v
            return v
