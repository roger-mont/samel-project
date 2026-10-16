"""Compara a generalização entre células: global × setor × setor + ganho individual
(orientações de 15/10, itens 6 e 10.5).

Deixa uma célula inteira de fora e estima as massas dela com curvas ajustadas nas
outras células (mesmo método das curvas em vigor: média por massa, isotônica, PCHIP):

- própria: a curva da própria célula, deixando um patamar de fora (erro de interpolação);
- A global: m̂ = f(s), curva de todas as outras células da manta;
- B setor: m̂_ij = f_s(s_ij), curva das outras células do mesmo setor;
- C setor + ganho: m̂_ij = g_ij · f_s(s_ij), com g_ij ajustado na própria célula
  deixando o patamar avaliado de fora.

Leitura: B muito pior que "própria" = a variação entre células domina. C muito
melhor que B = as células têm a mesma forma de curva e diferem no ganho. O modelo D
(dependência espacial / ML) só entra se nenhum destes generalizar.

Fontes: as calibrações de docs/<n>. Célula AxB/ (um ciclo por célula) e, com
--ensaios, os patamares de scripts/exportar_ensaios_celula.py (vários ciclos).

Uso:
    python scripts/comparar_generalizacao_celulas.py [--ensaios ensaios_celula.csv ...]
        [--sem-docs] [--ramo UP|DOWN] [--json resultado.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

EDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_ROOT / "src"))
sys.path.insert(0, str(EDGE_ROOT / "scripts"))

from math_engine.mass_models import build_curve  # noqa: E402
from math_engine.model_comparison import build_knots, isotonic_increasing  # noqa: E402

Curve = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Plateau:
    cell: str
    block: int
    cycle: int | None
    direction: str
    mass_kg: float
    raw: float          # RAW corrigido do patamar


def from_docs(docs: Path, tail_fraction: float) -> list[Plateau]:
    from gerar_curvas_bloco import discover_cells  # noqa: E402 (scripts/ no sys.path)

    out = []
    for c in discover_cells(docs, tail_fraction):
        for direction, branch in (("UP", c.up), ("DOWN", c.down)):
            for mass, p in branch.items():
                out.append(Plateau(c.name, c.sector, None, direction, float(mass), float(p["raw_corrected"])))
    return out


def from_export(path: Path) -> list[Plateau]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(Plateau(r["cell"], int(r["block_id"]),
                               int(r["cycle_id"]) if r.get("cycle_id") else None,
                               (r.get("direction") or "UNKNOWN").upper(),
                               float(r["known_mass_g"]) / 1000.0, float(r["corrected_mean"])))
    return out


def cell_points(plateaus: list[Plateau]) -> dict[str, dict[float, float]]:
    """Por célula: massa → RAW médio dos ciclos."""
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in plateaus:
        acc[p.cell][round(p.mass_kg, 6)].append(p.raw)
    return {cell: {m: float(np.mean(v)) for m, v in sorted(by.items())} for cell, by in acc.items()}


def repeatability(plateaus: list[Plateau]) -> dict[str, float | None]:
    """Por célula: desvio médio do RAW entre ciclos na mesma massa (None com um ciclo só)."""
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in plateaus:
        acc[p.cell][round(p.mass_kg, 6)].append(p.raw)
    out = {}
    for cell, by in acc.items():
        stds = [float(np.std(v, ddof=1)) for v in by.values() if len(v) > 1]
        out[cell] = round(float(np.mean(stds)), 3) if stds else None
    return out


def fit_curve(cells: list[dict[float, float]]) -> tuple[Curve, float] | None:
    """Curva das células dadas: média por massa, isotônica e PCHIP (como gerar_curvas_bloco.py)."""
    by_mass: dict[float, list[float]] = defaultdict(list)
    for pts in cells:
        for m, r in pts.items():
            if m > 0:
                by_mass[m].append(r)
    if len(by_mass) < 2:
        return None
    masses = np.array(sorted(by_mass))
    raw_iso = isotonic_increasing(np.array([np.mean(by_mass[m]) for m in masses]))
    raw_knots, mass_knots = build_knots(masses, raw_iso)
    if len(raw_knots) < 2:
        return None
    return build_curve("pchip", {"raw": raw_knots, "mass_kg": mass_knots}), raw_knots[-1]


def _stats(errors_kg: list[float], n_out: int) -> dict[str, Any] | None:
    if not errors_kg:
        return None
    e = np.array(errors_kg) * 1000.0
    return {"rmse_g": round(float(np.sqrt(np.mean(e ** 2))), 1), "bias_g": round(float(np.mean(e)), 1),
            "max_abs_g": round(float(np.max(np.abs(e))), 1), "n": len(e), "n_above_curve": n_out}


def _predict(curve: tuple[Curve, float], raws: np.ndarray) -> tuple[np.ndarray, int]:
    f, raw_max = curve
    return np.maximum(f(raws), 0.0), int(np.count_nonzero(raws > raw_max))


def evaluate_cell(cell: str, points: dict[str, dict[float, float]], block_of: dict[str, int]) -> dict[str, Any]:
    mine = {m: r for m, r in points[cell].items() if m > 0}
    masses = np.array(list(mine))
    raws = np.array(list(mine.values()))
    others = [c for c in points if c != cell]
    same = [c for c in others if block_of[c] == block_of[cell]]
    result: dict[str, Any] = {"cell": cell, "block": block_of[cell], "points": len(masses),
                              "other_cells_in_sector": len(same)}

    own_err: list[float] = []
    for i in range(len(masses)):
        keep = {m: r for j, (m, r) in enumerate(mine.items()) if j != i}
        curve = fit_curve([keep])
        if curve is not None:
            own_err.append(float(_predict(curve, raws[i:i + 1])[0][0]) - masses[i])
    result["own"] = _stats(own_err, 0)

    curve_a = fit_curve([points[c] for c in others])
    if curve_a is not None:
        pred, n_out = _predict(curve_a, raws)
        result["A_global"] = _stats(list(pred - masses), n_out)

    curve_b = fit_curve([points[c] for c in same]) if same else None
    if curve_b is None:
        result["B_sector"] = None
        result["C_sector_gain"] = None
        return result
    pred, n_out = _predict(curve_b, raws)
    result["B_sector"] = _stats(list(pred - masses), n_out)
    base = pred
    c_err: list[float] = []
    for i in range(len(masses)):
        keep = np.arange(len(masses)) != i
        den = float(base[keep] @ base[keep])
        if den > 0:
            c_err.append(float(base[i] * (base[keep] @ masses[keep]) / den) - masses[i])
    den = float(base @ base)
    result["C_sector_gain"] = _stats(c_err, n_out)
    result["gain_g_ij"] = round(float(base @ masses) / den, 3) if den > 0 else None
    return result


def between_cells(points: dict[str, dict[float, float]], block_of: dict[str, int]) -> list[dict[str, Any]]:
    """Por setor com 2+ células: diferença de RAW entre as células em cada massa em comum."""
    out = []
    for block in sorted(set(block_of.values())):
        cells = sorted(c for c in points if block_of[c] == block)
        if len(cells) < 2:
            continue
        common = sorted(set.intersection(*(set(points[c]) for c in cells)) - {0.0})
        rows = []
        for m in common:
            raws = np.array([points[c][m] for c in cells])
            rows.append({"mass_kg": m, "raw_min": round(float(raws.min()), 2), "raw_max": round(float(raws.max()), 2),
                         "raw_cv": round(float(np.std(raws, ddof=1) / np.mean(raws)), 3) if np.mean(raws) > 0 else None})
        out.append({"block": block, "cells": cells, "by_mass": rows})
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in ("own", "A_global", "B_sector", "C_sector_gain"):
        rmses = [r[key]["rmse_g"] for r in results if r.get(key)]
        out[key] = {"cells": len(rmses), "median_rmse_g": round(float(np.median(rmses)), 1) if rmses else None,
                    "mean_rmse_g": round(float(np.mean(rmses)), 1) if rmses else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=EDGE_ROOT.parent.parent / "docs")
    parser.add_argument("--sem-docs", action="store_true", help="Não usa as calibrações de docs/")
    parser.add_argument("--ensaios", type=Path, nargs="*", default=[], help="CSVs de exportar_ensaios_celula.py")
    parser.add_argument("--ramo", choices=("UP", "DOWN"), default="UP", help="Ramo comparado. Padrão: UP")
    parser.add_argument("--fracao-cauda", type=float, default=0.5)
    parser.add_argument("--json", type=Path, default=None, help="Grava o resultado completo em JSON")
    args = parser.parse_args()

    plateaus: list[Plateau] = []
    if not args.sem_docs:
        plateaus += from_docs(args.docs, args.fracao_cauda)
    for path in args.ensaios:
        plateaus += from_export(path)
    plateaus = [p for p in plateaus if p.direction == args.ramo]
    if not plateaus:
        raise SystemExit("Nenhum patamar para comparar.")

    points = cell_points(plateaus)
    block_of = {p.cell: p.block for p in plateaus}
    results = [evaluate_cell(c, points, block_of) for c in sorted(points, key=lambda c: (block_of[c], c))]
    report = {
        "branch": args.ramo,
        "cells": results,
        "summary": summarize(results),
        "repeatability_raw_std": repeatability(plateaus),
        "between_cells": between_cells(points, block_of),
    }

    fmt = lambda s: f"{s['rmse_g']:>6.1f}" if s else "     —"  # noqa: E731
    print(f"Ramo {args.ramo}: {len(points)} células. RMSE em g por célula, deixando a célula de fora.")
    print(f"{'célula':<8} {'setor':>5} {'própria':>8} {'A glob':>7} {'B setor':>8} {'C s+g':>7} {'g_ij':>6}")
    for r in results:
        g = f"{r['gain_g_ij']:.2f}" if r.get("gain_g_ij") is not None else "—"
        print(f"{r['cell']:<8} {r['block']:>5} {fmt(r['own']):>8} {fmt(r.get('A_global')):>7} "
              f"{fmt(r.get('B_sector')):>8} {fmt(r.get('C_sector_gain')):>7} {g:>6}")
    s = report["summary"]
    print("Mediana do RMSE: " + " · ".join(f"{k} {v['median_rmse_g']} g ({v['cells']} células)"
                                           for k, v in s.items() if v["median_rmse_g"] is not None))
    lonely = [r["cell"] for r in results if r["B_sector"] is None]
    if lonely:
        print(f"Sem outra célula no setor (B e C não avaliados): {', '.join(lonely)}")
    for b in report["between_cells"]:
        worst = max(b["by_mass"], key=lambda x: x["raw_cv"] or 0, default=None)
        if worst:
            print(f"Setor {b['block']} ({', '.join(b['cells'])}): maior variação entre células em "
                  f"{worst['mass_kg'] * 1000:.0f} g, RAW de {worst['raw_min']} a {worst['raw_max']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] {args.json}")


if __name__ == "__main__":
    main()
