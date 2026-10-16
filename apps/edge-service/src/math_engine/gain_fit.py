"""Ajuste dos ganhos α_s da etapa massa local → massa total.

M = Σ_s α_s · L_s, com L_s = Σ_{ij∈s} f_s(x_ij) a massa local do setor s.
Usado pelo script ajustar_ganho_massa_total.py (sessões de experimento gravadas)
e pelo painel de ajuste de ganho da página /calibracao (pontos registrados ao vivo).

Modos:
- "global": um α para a manta inteira, mínimos quadrados sobre a soma local;
- "setor":  um α por setor, puxado para o α global (ridge, λ adimensional).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODES = ("global", "setor")


def fit_gains(L: np.ndarray, M: np.ndarray, mode: str, lam: float) -> np.ndarray:
    """α global (mínimos quadrados) ou por setor com ridge em direção ao α global.

    L: (pontos × setores) massas locais em kg; M: massas conhecidas em kg.
    """
    if mode not in MODES:
        raise ValueError(f"modo deve ser um de {MODES}")
    total = L.sum(axis=1)
    alpha_global = float(total @ M / (total @ total))
    if mode == "global":
        return np.full(L.shape[1], alpha_global)
    # min ||M − Lα||² + λ·s²·||α − α_g||², com s² = escala média de L² para λ ser adimensional
    scale = float(np.mean(np.sum(L ** 2, axis=0))) or 1.0
    A = L.T @ L + lam * scale * np.eye(L.shape[1])
    b = L.T @ M + lam * scale * alpha_global
    alpha = np.linalg.solve(A, b)
    return np.maximum(alpha, 1e-3)


def _stats(err: np.ndarray, M: np.ndarray) -> dict[str, float]:
    return {"rmse_kg": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "bias_kg": round(float(np.mean(err)), 3),
            "mape_pct": round(float(np.mean(np.abs(err) / M)) * 100.0, 1)}


def leave_one_out(L: np.ndarray, M: np.ndarray, mode: str, lam: float) -> list[float | None]:
    """Previsão de cada ponto com α ajustado sem ele (None quando não dá para ajustar)."""
    preds: list[float | None] = []
    for i in range(len(M)):
        keep = np.arange(len(M)) != i
        if keep.sum() < 1 or L[keep].sum() == 0:
            preds.append(None)
            continue
        preds.append(float(L[i] @ fit_gains(L[keep], M[keep], mode, lam)))
    return preds


def fit_report(L: np.ndarray, M: np.ndarray, alpha: np.ndarray, mode: str, lam: float) -> dict[str, Any]:
    """Erro sem ganho, com o ganho ajustado e deixando um ponto de fora."""
    loo = leave_one_out(L, M, mode, lam)
    loo_err = np.array([p - m for p, m in zip(loo, M) if p is not None])
    loo_M = np.array([m for p, m in zip(loo, M) if p is not None])
    return {
        "points": int(len(M)),
        "without_gain": _stats(L.sum(axis=1) - M, M),
        "fitted": _stats(L @ alpha - M, M),
        "leave_one_out": _stats(loo_err, loo_M) if len(loo_err) else None,
    }


def curves_sha1(data: dict[str, Any]) -> str:
    """Impressão digital das curvas: os ganhos só valem para as curvas com que foram ajustados."""
    return hashlib.sha1(json.dumps(data["blocks"], sort_keys=True).encode("utf-8")).hexdigest()


def write_gains(path: Path, gains: dict[int, float] | None, fit: dict[str, Any] | None) -> dict[str, Any]:
    """Grava aggregation em block_curves.json. gains=None volta todos os α para 1 (não calibrado)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = sorted(int(b) for b in data["blocks"])
    if gains is None:
        aggregation = {"method": "sector_gain_sum", "gains": {str(b): 1.0 for b in blocks},
                       "calibrated": False, "fit": None}
    else:
        if any(g <= 0 for g in gains.values()):
            raise ValueError("ganhos α_s devem ser positivos")
        aggregation = {
            "method": "sector_gain_sum",
            "gains": {str(b): round(float(gains.get(b, 1.0)), 5) for b in blocks},
            "calibrated": True,
            "fit": {
                **(fit or {}),
                "fitted_at": datetime.now(timezone.utc).isoformat(),
                "curves_sha1": curves_sha1(data),
                "calibration_version": data.get("calibration_version"),
            },
        }
    data["aggregation"] = aggregation
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return aggregation
