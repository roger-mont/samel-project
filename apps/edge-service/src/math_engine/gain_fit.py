"""Ajuste dos ganhos α_s da etapa massa local → massa total.

M = Σ_s α_s · L_s, com L_s = Σ_{ij∈s} f_s(x_ij) a massa local do setor s.
Usado pelo script ajustar_ganho_massa_total.py (sessões de experimento gravadas)
e pelo painel de ajuste de ganho da página /calibracao (pontos registrados ao vivo).

Modos:
- "global": um α para a manta inteira, mínimos quadrados sobre a soma local;
- "setor":  um α por setor, puxado para o α global (ridge, λ adimensional).

As orientações de 15/10 (item 3.9) pedem α = 1 até os ensaios de carga distribuída
e de posição; o α ajustado antes disso é provisório. position_report() separa erro
espacial de erro de ganho (item 7).
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


POSITIONS = ("centro", "cabeceira", "peseira", "esquerda", "direita")


def position_report(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensaios de carga conhecida e de posição (item 7 das orientações de 15/10).

    Agrupa os pontos pela massa real (0,1 kg) e compara o erro com o α em vigor:
    - repetibilidade: desvio do erro dentro da mesma posição (ciclos 0 → carga → 0);
    - espalhamento entre posições: desvio das médias de cada posição;
    - viés: erro médio do grupo.
    Se a mesma massa varia muito com a posição, o erro é sobretudo espacial; se todas
    as posições dão valores parecidos, mas deslocados da referência, é de ganho global.
    """
    groups: dict[float, list[dict[str, Any]]] = {}
    for p in points:
        groups.setdefault(round(float(p["known_mass_kg"]), 1), []).append(p)
    report = []
    for mass, pts in sorted(groups.items()):
        if len(pts) < 2:
            continue
        errors = np.array([p["current_estimate_kg"] - p["known_mass_kg"] for p in pts])
        by_pos: dict[str, list[float]] = {}
        for p, e in zip(pts, errors):
            by_pos.setdefault(p.get("position") or "sem posição", []).append(float(e))
        positions = {
            pos: {"n": len(v), "mean_error_kg": round(float(np.mean(v)), 3),
                  "std_error_kg": round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else None}
            for pos, v in sorted(by_pos.items())
        }
        repeats = [v for v in by_pos.values() if len(v) > 1]
        repeat_std = (float(np.sqrt(sum(np.var(v, ddof=1) * (len(v) - 1) for v in repeats)
                                    / sum(len(v) - 1 for v in repeats))) if repeats else None)
        named = {k: v for k, v in by_pos.items() if k != "sem posição"}
        spread = float(np.std([np.mean(v) for v in named.values()], ddof=1)) if len(named) > 1 else None
        bias = float(np.mean(errors))
        if spread is None:
            diagnosis = "registre a mesma massa em outras posições"
        elif spread > max(abs(bias), repeat_std or 0.0):
            diagnosis = "espacial: a mesma massa muda com a posição"
        elif abs(bias) > max(spread, repeat_std or 0.0):
            diagnosis = "ganho global: posições parecidas, deslocadas da referência"
        else:
            diagnosis = "inconclusivo: diferenças do tamanho da repetibilidade"
        report.append({
            "known_mass_kg": mass,
            "n": len(pts),
            "bias_kg": round(bias, 3),
            "repeatability_std_kg": None if repeat_std is None else round(repeat_std, 3),
            "position_spread_kg": None if spread is None else round(spread, 3),
            "positions": positions,
            "diagnosis": diagnosis,
        })
    return report


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
