"""Auditoria das curvas de block_curves.json (itens 3.6, 3.8 e 4 das orientações de 15/10).

Não altera as curvas: compara o que foi medido com o que a curva usa e devolve
alertas de qualidade da calibração.

- Regularização isotônica: o patamar medido (raw_mean) fica separado do valor
  usado na curva (raw_isotonic). A diferença mostra quanto a PAVA corrigiu; também
  é recalculada célula a célula e nos dois ramos (UP e DOWN) a partir de 'sources'.
- Região pouco suportada: massa ensaiada que deu RAW 0 (o trecho até o primeiro
  nó só é interpolado a partir de (0,0)).
- Histerese: erro da curva UP nos patamares DOWN (validation).
- Baixa sensibilidade: maior g por RAW entre nós.
- Setor representado por uma célula só.
- CSVs de calibração idênticos (mesmo SHA-1 em 'sources'), como 12X34 e 13X26
  (parecer de 20/10, item 7). Só alerta: a decisão de manter ou refazer o ensaio
  é da equipe.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from math_engine.model_comparison import isotonic_increasing

ISOTONIC_ALERT_RAW = 5.0        # correção acima disto (em RAW) gera alerta
ISOTONIC_ALERT_REL = 0.10       # ou acima desta fração do patamar medido
HYSTERESIS_ALERT_G = 20.0       # |viés UP→DOWN| acima disto gera alerta


def isotonic_correction(masses: list[float], measured: list[float], fitted: list[float] | None = None) -> dict[str, Any]:
    """Quanto a regressão isotônica mudou os patamares medidos (RAW crescente com a massa)."""
    raw = np.asarray(measured, dtype=float)
    iso = np.asarray(fitted, dtype=float) if fitted is not None else isotonic_increasing(raw)
    diff = iso - raw
    rel = np.abs(diff) / np.maximum(np.abs(raw), 1.0)
    changed = [round(float(m), 6) for m, d in zip(masses, diff) if abs(d) > 1e-6]
    max_abs = float(np.max(np.abs(diff))) if len(diff) else 0.0
    max_rel = float(np.max(rel)) if len(rel) else 0.0
    return {
        "points": len(raw),
        "adjusted_masses_kg": changed,
        "max_abs_raw": round(max_abs, 3),
        "max_rel": round(max_rel, 3),
        # o relativo só conta acima de 1 RAW, para não alertar por ruído em patamar quase zero
        "alert": max_abs > ISOTONIC_ALERT_RAW or (max_rel > ISOTONIC_ALERT_REL and max_abs > 1.0),
    }


def _branch(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    pts = sorted((p for p in points if float(p["mass_kg"]) > 0), key=lambda p: float(p["mass_kg"]))
    if len(pts) < 2:
        return None
    return isotonic_correction([float(p["mass_kg"]) for p in pts], [float(p["raw_corrected"]) for p in pts])


def _fmt_g(kg: float) -> str:
    return f"{kg * 1000:.0f} g"


def audit_calibration(definition: dict[str, Any], low_sensitivity_g_per_raw: float = 0.0) -> dict[str, Any]:
    """Auditoria por setor e lista de alertas, em texto curto."""
    sources = definition.get("sources", []) or []
    sectors: dict[str, Any] = {}
    alerts: list[str] = []
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sources:
        if s.get("sha1"):
            by_digest[s["sha1"]].append(s)
    duplicates = [
        {"sha1": digest, "cells": [s.get("cell") for s in group], "sectors": [s.get("block") for s in group],
         "files": [s.get("file") for s in group]}
        for digest, group in by_digest.items() if len(group) > 1
    ]
    for d in duplicates:
        alerts.append(f"CSVs de calibração idênticos (SHA-1 {d['sha1'][:10]}): células "
                      f"{', '.join(map(str, d['cells']))}, setores {', '.join(map(str, d['sectors']))}.")
    for bid_str, b in sorted(definition.get("blocks", {}).items(), key=lambda kv: int(kv[0])):
        fit = b.get("fit", []) or []
        used = (isotonic_correction([f["mass_kg"] for f in fit], [f["raw_mean"] for f in fit],
                                    [f["raw_isotonic"] for f in fit]) if fit else None)
        cells = []
        for s in sources:
            if str(s.get("block")) != bid_str:
                continue
            cells.append({
                "cell": s.get("cell"),
                "up": _branch(s.get("points_up", [])),
                "down": _branch(s.get("points_down_not_used", [])),
            })
        zero = sorted(float(f["mass_kg"]) for f in fit if float(f["mass_kg"]) > 0 and float(f["raw_mean"]) <= 0)
        first_node = next((float(r) for r in b.get("params", {}).get("raw", []) if float(r) > 0), None)
        validation = b.get("validation") or {}
        profile = b.get("sensitivity_g_per_raw", []) or []
        max_g_per_raw = max((float(p["g_per_raw"]) for p in profile), default=None)
        n_cells = len(b.get("cells", []) or [])
        sectors[bid_str] = {
            "cells": b.get("cells", []),
            "source": b.get("source"),
            "isotonic_used_curve": used,
            "isotonic_by_cell": cells,
            "zero_response_masses_kg": zero,
            "low_support_until_raw": first_node if zero else None,
            "hysteresis": {
                "up_curve_on_down_bias_g": validation.get("up_curve_on_down_bias_g"),
                "up_curve_on_down_rmse_g": validation.get("up_curve_on_down_rmse_g"),
            },
            "loo_rmse_g": validation.get("loo_rmse_g"),
            "max_g_per_raw": max_g_per_raw,
        }

        tag = f"Setor {bid_str}"
        if b.get("source") == "fallback_global":
            alerts.append(f"{tag}: sem célula calibrada, usa a curva global.")
        elif n_cells == 1:
            alerts.append(f"{tag}: curva de uma célula só ({b['cells'][0]}); não mostra a variação entre células.")
        if used and used["alert"]:
            alerts.append(f"{tag}: a isotônica corrigiu até {used['max_abs_raw']:.1f} RAW na curva usada.")
        for c in cells:
            for branch in ("up", "down"):
                info = c[branch]
                if info and info["alert"]:
                    alerts.append(f"{tag}: patamares fora de ordem na célula {c['cell']} ({branch.upper()}), "
                                  f"correção de até {info['max_abs_raw']:.1f} RAW.")
        if zero:
            alerts.append(f"{tag}: {', '.join(_fmt_g(m) for m in zero)} sem resposta (RAW 0); "
                          f"até {first_node:.1f} RAW a curva só interpola a partir de (0,0).")
        bias = validation.get("up_curve_on_down_bias_g")
        if bias is not None and abs(float(bias)) > HYSTERESIS_ALERT_G:
            alerts.append(f"{tag}: viés UP→DOWN de {float(bias):+.0f} g por célula (histerese).")
        if low_sensitivity_g_per_raw > 0 and max_g_per_raw is not None and max_g_per_raw > low_sensitivity_g_per_raw:
            alerts.append(f"{tag}: até {max_g_per_raw:.1f} g por RAW no topo da curva (baixa sensibilidade).")
    return {
        "calibration_version": definition.get("calibration_version"),
        "thresholds": {
            "isotonic_alert_raw": ISOTONIC_ALERT_RAW,
            "isotonic_alert_rel": ISOTONIC_ALERT_REL,
            "hysteresis_alert_g": HYSTERESIS_ALERT_G,
            "low_sensitivity_g_per_raw": low_sensitivity_g_per_raw,
        },
        "sectors": sectors,
        "duplicate_sources": duplicates,
        "alerts": alerts,
    }
