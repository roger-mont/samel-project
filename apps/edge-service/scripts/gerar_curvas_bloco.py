"""Gera src/storage/block_curves.json com uma curva RAW → massa local por setor.

Lê as calibrações de bancada de célula em docs/<n>. Célula <a>X<b>/*.csv (formato
Ponto_Idx, Massa_kg, RAW_Amostra, RAW_Medio, Baseline_Bij, RAW_Corrigido, Direcao,
Estavel, Tempo_Estabilizacao_s). O nome da pasta dá a posição da célula: <a> é o
índice na largura da manta (linha da matriz, 0-31) e <b> o índice no comprimento
(coluna da matriz, 0-63), com origem no setor 4 (peseira, lado direito). O setor
sai de BLOCK_REGIONS; docs/Células-Setores.txt, se existir, só serve de conferência.

Etapa 1, RAW → massa local (por setor s):
- valor do patamar = média da cauda estável de max(0, RAW_Amostra − Baseline_Bij)
  (regime estabilizado; o transitório do começo do patamar fica de fora);
- só o ramo de subida (UP) entra na curva clínica; a descida vira uma curva
  separada, guardada só para estudo de histerese (item 9: não misturar ramos);
- setor com mais de uma célula: média das células em cada massa em comum;
- regressão isotônica (RAW não decrescente com a massa) e PCHIP sobre os nós;
- setor sem célula calibrada usa a curva global (todas as células), marcada.

Etapa 2, massa local → massa total: M = Σ_s α_s · Σ_{ij∈s} f_s(x_ij). Este script
grava α_s = 1 (não calibrado); scripts/ajustar_ganho_massa_total.py ajusta α com
experimentos de carga distribuída conhecida. Regerar as curvas zera os ganhos,
porque eles só valem para as curvas com que foram ajustados.

Uso:
    python scripts/gerar_curvas_bloco.py [--fracao-cauda 0.5] [--limiar-g-por-raw 10]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_ROOT / "src"))

from math_engine.mass_models import build_curve  # noqa: E402
from math_engine.model_comparison import build_knots, isotonic_increasing  # noqa: E402
from storage.calibration_store import BLOCK_REGIONS, SECTOR_LAYOUT  # noqa: E402

REPO_ROOT = EDGE_ROOT.parent.parent
DOCS = REPO_ROOT / "docs"
CALIBRATION_VERSION = "MC-2026-002-PROV"
MODEL_VERSION = "block_curves/pchip-isotonic-v2-setor"
BLOCKS = sorted(BLOCK_REGIONS)
_CELL_RE = re.compile(r"lula\s+(\d+)\s*[xX]\s*(\d+)")


@dataclass
class CellCalibration:
    name: str
    row: int
    column: int
    sector: int
    path: Path
    up: dict[float, dict]
    down: dict[float, dict]


def sector_of(row: int, column: int) -> int:
    for bid, (rs, cs) in BLOCK_REGIONS.items():
        if rs.start <= row < rs.stop and cs.start <= column < cs.stop:
            return bid
    raise ValueError(f"Célula {row}x{column} fora da matriz")


def read_points(path: Path, tail_fraction: float) -> tuple[dict[float, dict], dict[float, dict]]:
    """Um registro por patamar (Ponto_Idx), separado em subida e descida."""
    samples: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            samples[int(row["Ponto_Idx"])].append(row)
    up: dict[float, dict] = {}
    down: dict[float, dict] = {}
    for idx in sorted(samples):
        rows = samples[idx]
        first = rows[0]
        baseline = float(first["Baseline_Bij"])
        values = np.maximum(np.array([float(r["RAW_Amostra"]) for r in rows]) - baseline, 0.0)
        tail = values[int(len(values) * (1.0 - tail_fraction)):]
        point = {
            "ponto_idx": idx,
            "mass_kg": round(float(first["Massa_kg"]), 6),
            "raw_corrected": round(float(np.mean(tail)), 4),
            "raw_std_tail": round(float(np.std(tail)), 4),
            "raw_corrected_csv": float(first["RAW_Corrigido"]),
            "baseline": baseline,
            "stabilization_time_s": float(first["Tempo_Estabilizacao_s"]),
            "n": len(rows),
        }
        branch = up if first["Direcao"].strip().upper() == "UP" else down
        if point["mass_kg"] in branch:
            raise ValueError(f"{path.name}: massa {point['mass_kg']} repetida no mesmo ramo")
        branch[point["mass_kg"]] = point
    return up, down


def discover_cells(docs: Path, tail_fraction: float) -> list[CellCalibration]:
    cells: list[CellCalibration] = []
    for folder in sorted(p for p in docs.iterdir() if p.is_dir()):
        match = _CELL_RE.search(unicodedata.normalize("NFC", folder.name))
        csvs = sorted(folder.glob("*.csv"))
        if not match or not csvs:
            continue
        if len(csvs) > 1:
            raise SystemExit(f"{folder.name}: mais de um CSV, deixe só o da calibração")
        row, column = int(match.group(1)), int(match.group(2))
        up, down = read_points(csvs[0], tail_fraction)
        cells.append(CellCalibration(f"{row}x{column}", row, column, sector_of(row, column), csvs[0], up, down))
    return cells


def check_sector_list(cells: list[CellCalibration], docs: Path) -> None:
    """Confere o setor calculado com docs/Células-Setores.txt, se o arquivo existir."""
    listed = next((p for p in docs.glob("*.txt") if "setores" in unicodedata.normalize("NFC", p.name).lower()), None)
    if listed is None:
        return
    expected: dict[str, int] = {}
    for line in listed.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"\s*Setor\s+(\d+)\s*:\s*(.+)", line)
        if m:
            for cell in re.findall(r"(\d+)\s*[xX]\s*(\d+)", m.group(2)):
                expected[f"{int(cell[0])}x{int(cell[1])}"] = int(m.group(1))
    for c in cells:
        if c.name in expected and expected[c.name] != c.sector:
            print(f"[!] {c.name}: setor {c.sector} pela matriz, {expected[c.name]} em {listed.name}")
        elif c.name not in expected:
            print(f"[!] {c.name}: não aparece em {listed.name}")


def branch_fit(branches: list[dict[float, dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Massas em comum, RAW médio das células e o ajuste isotônico."""
    masses = sorted(m for m in set.intersection(*(set(b) for b in branches)) if m > 0)
    m = np.array(masses)
    raw_mean = np.array([np.mean([b[x]["raw_corrected"] for b in branches]) for x in masses])
    return m, raw_mean, isotonic_increasing(raw_mean)


def pooled_fit(cells: list[CellCalibration]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva global: em cada massa, média das células que a mediram (ramo UP)."""
    by_mass: dict[float, list[float]] = defaultdict(list)
    for c in cells:
        for x, p in c.up.items():
            if x > 0:
                by_mass[x].append(p["raw_corrected"])
    masses = sorted(by_mass)
    m = np.array(masses)
    raw_mean = np.array([np.mean(by_mass[x]) for x in masses])
    return m, raw_mean, isotonic_increasing(raw_mean)


def curve_from(m: np.ndarray, raw_iso: np.ndarray):
    raw_knots, mass_knots = build_knots(m, raw_iso)
    return raw_knots, mass_knots, build_curve("pchip", {"raw": raw_knots, "mass_kg": mass_knots})


def rmse_g(errors: list[float]) -> float | None:
    return round(float(np.sqrt(np.mean(np.square(errors)))) * 1000.0, 1) if errors else None


def validate(m: np.ndarray, raw_mean: np.ndarray, down_points: list[tuple[float, float]]) -> dict:
    """Erro deixando um patamar de fora (LOO) e erro da curva UP nos patamares DOWN."""
    loo: list[float] = []
    for i in range(len(m)):
        keep = np.arange(len(m)) != i
        try:
            *_, f = curve_from(m[keep], isotonic_increasing(raw_mean[keep]))
        except ValueError:
            continue
        loo.append(float(f(np.array([raw_mean[i]]))[0]) - m[i])
    *_, f = curve_from(m, isotonic_increasing(raw_mean))
    down = [float(f(np.array([r]))[0]) - x for x, r in down_points if x > 0]
    return {"loo_rmse_g": rmse_g(loo), "up_curve_on_down_rmse_g": rmse_g(down),
            "up_curve_on_down_bias_g": round(float(np.mean(down)) * 1000.0, 1) if down else None}


def sensitivity_profile(raw_knots: list[float], mass_knots: list[float]) -> list[dict]:
    """g por unidade de RAW em cada trecho entre nós (quanto vale 1 RAW de resolução)."""
    out = []
    for (r0, m0), (r1, m1) in zip(zip(raw_knots, mass_knots), zip(raw_knots[1:], mass_knots[1:])):
        out.append({"raw_from": round(r0, 3), "raw_to": round(r1, 3),
                    "g_per_raw": round((m1 - m0) * 1000.0 / (r1 - r0), 2)})
    return out


def low_sensitivity_start(profile: list[dict], limit_g_per_raw: float | None) -> float | None:
    """Primeiro RAW a partir do qual 1 RAW passa a valer mais que o limite (item 11)."""
    if limit_g_per_raw is None:
        return None
    for seg in profile:
        if seg["g_per_raw"] > limit_g_per_raw:
            return seg["raw_from"]
    return None


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    """Caminho relativo ao repositório quando o arquivo está nele."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=DOCS, help="Pasta com as subpastas 'Célula AxB'")
    parser.add_argument("--fracao-cauda", type=float, default=0.5,
                        help="Fração final de cada patamar usada como regime estável. Padrão: 0.5")
    parser.add_argument("--raw-min-valid", type=float, default=1.0,
                        help="RAW corrigido abaixo do qual a célula não conta (ruído). Padrão: 1.0")
    parser.add_argument("--limiar-g-por-raw", type=float, default=None,
                        help="Marca LOW_SENSITIVITY onde 1 RAW vale mais que este número de gramas. "
                             "Sem valor, a região não é definida (item 11).")
    parser.add_argument("--saida", type=Path, default=EDGE_ROOT / "src" / "storage" / "block_curves.json")
    args = parser.parse_args()
    if not 0.0 < args.fracao_cauda <= 1.0:
        raise SystemExit("--fracao-cauda deve estar em (0, 1]")

    cells = discover_cells(args.docs, args.fracao_cauda)
    if not cells:
        raise SystemExit(f"Nenhuma pasta 'Célula AxB' com CSV em {args.docs}")
    check_sector_list(cells, args.docs)
    digests: dict[str, list[str]] = defaultdict(list)
    for c in cells:
        digests[sha1(c.path)].append(c.name)
    for names in digests.values():
        if len(names) > 1:
            print(f"[!] CSVs idênticos: {', '.join(names)}")

    by_sector: dict[int, list[CellCalibration]] = defaultdict(list)
    for c in cells:
        by_sector[c.sector].append(c)

    pooled_m, pooled_raw, pooled_iso = pooled_fit(cells)
    blocks: dict[str, dict] = {}
    for bid in BLOCKS:
        members = by_sector.get(bid, [])
        if members:
            m, raw_mean, raw_iso = branch_fit([c.up for c in members])
            source = "cells"
        else:
            m, raw_mean, raw_iso = pooled_m, pooled_raw, pooled_iso
            source = "fallback_global"
        raw_knots, mass_knots, _ = curve_from(m, raw_iso)
        profile = sensitivity_profile(raw_knots, mass_knots)

        down_curve = None
        down_points: list[tuple[float, float]] = []
        if members and all(c.down for c in members):
            dm, draw_mean, draw_iso = branch_fit([c.down for c in members])
            if len(dm) >= 2:
                drk, dmk, _ = curve_from(dm, draw_iso)
                down_curve = {"kind": "pchip", "params": {"raw": [round(r, 4) for r in drk],
                                                          "mass_kg": [round(x, 6) for x in dmk]}}
                down_points = list(zip(dm.tolist(), draw_mean.tolist()))

        blocks[str(bid)] = {
            "kind": "pchip",
            "params": {"raw": [round(r, 4) for r in raw_knots], "mass_kg": [round(x, 6) for x in mass_knots]},
            "raw_min_valid": args.raw_min_valid,
            "raw_max_valid": round(raw_knots[-1], 4),
            "mass_min_valid": 0.0,
            "mass_max_valid": round(float(m.max()), 6),
            "low_sensitivity_raw": low_sensitivity_start(profile, args.limiar_g_por_raw),
            "source": source,
            "cells": [c.name for c in members],
            "fit": [{"mass_kg": float(x), "raw_mean": round(float(r), 4), "raw_isotonic": round(float(ri), 4)}
                    for x, r, ri in zip(m, raw_mean, raw_iso)],
            "sensitivity_g_per_raw": profile,
            "validation": validate(m, raw_mean, down_points) if members else None,
            "down_curve_not_used": down_curve,
        }

    payload = {
        "calibration_version": CALIBRATION_VERSION,
        "model_version": MODEL_VERSION,
        "provisional": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "apps/edge-service/scripts/gerar_curvas_bloco.py",
        "method": {
            "direction": "UP (a descida fica em down_curve_not_used, só para estudo)",
            "plateau_value": f"média dos últimos {args.fracao_cauda:.0%} de max(0, RAW_Amostra − Baseline_Bij) do patamar",
            "cell_combination": "média das células do setor em cada massa em comum",
            "monotonic_fit": "regressão isotônica (PAVA) do RAW em função da massa",
            "interpolation": "PCHIP sobre os nós RAW → massa",
            "blocks": "uma curva por setor, da(s) célula(s) calibrada(s) no setor; sem célula, curva global",
            "total_mass": "M = Σ_s α_s · Σ_{ij∈s} f_s(x_ij)",
        },
        "notes": [
            "Uma célula por setor (duas no setor 2): a curva representa o setor inteiro sem medir a variação "
            "entre células do mesmo setor. As duas células do setor 2 diferem em cerca de 90 g na mesma leitura.",
            "Faixa calibrada: até mass_max_valid por célula; acima de raw_max_valid a leitura é OUT_OF_CALIBRATION.",
            "Ganhos α_s = 1 até haver experimento de carga distribuída (scripts/ajustar_ganho_massa_total.py).",
            "Calibrações anteriores (calibration.json, weight_model.joblib, data/sessions) não são usadas.",
        ],
        "sources": [
            {
                "file": display_path(c.path),
                "sha1": sha1(c.path),
                "cell": c.name,
                "row": c.row,
                "column": c.column,
                "block": c.sector,
                "points_up": [c.up[x] for x in sorted(c.up)],
                "points_down_not_used": [c.down[x] for x in sorted(c.down, reverse=True)],
            }
            for c in sorted(cells, key=lambda c: (c.sector, c.name))
        ],
        "sector_layout": {
            "cable_sector": 1,
            "head_to_feet": [{"right": r, "left": l} for r, l in SECTOR_LAYOUT],
            "matrix": {str(b): {"rows": [BLOCK_REGIONS[b][0].start, BLOCK_REGIONS[b][0].stop - 1],
                                "cols": [BLOCK_REGIONS[b][1].start, BLOCK_REGIONS[b][1].stop - 1]} for b in BLOCKS},
        },
        "blocks": blocks,
        "aggregation": {
            "method": "sector_gain_sum",
            "gains": {str(b): 1.0 for b in BLOCKS},
            "calibrated": False,
            "fit": None,
        },
    }

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] {args.saida} gravado ({CALIBRATION_VERSION})")
    print(f"{'setor':>5} {'células':<14} {'nós':>3} {'RAW máx':>8} {'massa máx':>9} {'LOO':>7} {'UP→DOWN':>8}")
    for bid in BLOCKS:
        b = blocks[str(bid)]
        v = b["validation"] or {}
        print(f"{bid:>5} {','.join(b['cells']) or b['source']:<14} {len(b['params']['raw']):>3} "
              f"{b['raw_max_valid']:>8.1f} {b['mass_max_valid'] * 1000:>7.0f} g "
              f"{v.get('loo_rmse_g') or 0:>5.0f} g {v.get('up_curve_on_down_rmse_g') or 0:>6.0f} g")


if __name__ == "__main__":
    main()
