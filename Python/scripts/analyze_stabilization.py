"""Estudo da estabilização temporal e repetibilidade conforme Metodologia §12 e §28.

Analisa a dispersão (std, cv%), tempo de resposta e variabilidade entre repetições.

Uso:
    python Python/scripts/analyze_stabilization.py
    python Python/scripts/analyze_stabilization.py --sessions-dir sessions/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


def _parse_session_details(sessions_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not sessions_dir.exists():
        return records

    for csv_file in sorted(sessions_dir.glob("calib_*.csv")):
        try:
            with csv_file.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_kg = row.get("reference_kg")
                    if ref_kg is None or ref_kg == "":
                        continue
                    try:
                        records.append({
                            "session_id": row.get("session_id", ""),
                            "position": row.get("position_tag", "unknown"),
                            "repetition": int(row.get("repetition", 1)),
                            "reference_kg": float(ref_kg),
                            "raw_sum": float(row.get("raw_sum_block", 0.0)),
                            "mean_raw": float(row.get("mean_raw_block", 0.0)),
                            "std_raw": float(row.get("std_raw_block", 0.0)),
                            "cv_pct": float(row.get("cv_pct", 0.0)),
                            "time_s": float(row.get("time_since_load_s", 5.0)),
                            "state": row.get("stability_state", "stable"),
                        })
                    except ValueError:
                        continue
        except OSError as err:
            print(f"[AVISO] Falha ao ler {csv_file.name}: {err}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de Estabilização e Dispersão Temporal")
    parser.add_argument("--sessions-dir", type=str, default="sessions",
                        help="Diretório de sessões CSV (padrão: sessions)")
    args = parser.parse_args()

    sessions_path = Path(args.sessions_dir)
    records = _parse_session_details(sessions_path)

    print(f"\n{'='*75}")
    print("  ANÁLISE DE ESTABILIZAÇÃO TEMPORAL E VARIABILIDADE EXPERIMENTAL")
    print(f"{'='*75}")
    print(f"Diretório: {sessions_path.resolve()}")
    print(f"Amostras analisadas: {len(records)}\n")

    if not records:
        print("[INFO] Nenhuma sessão encontrada em sessions/.")
        sys.exit(0)

    cv_all = np.array([r["cv_pct"] for r in records])
    std_all = np.array([r["std_raw"] for r in records])

    estaveis = sum(1 for r in records if r["cv_pct"] < 8.0)
    instaveis = len(records) - estaveis
    pct_estavel = (estaveis / len(records)) * 100.0

    print(f"DISPERSÃO GERAL DOS ENSAIOS:")
    print(f"  CV% Médio (Coef. de Variação):   {float(np.mean(cv_all)):.2f}%")
    print(f"  CV% Máximo registrado:           {float(np.max(cv_all)):.2f}%")
    print(f"  Desvio Padrão Médio (std):       {float(np.mean(std_all)):.2f} pts de soma")
    print(f"  Taxa de Estabilidade (< 8% CV):  {pct_estavel:.1f}% ({estaveis}/{len(records)} medições)\n")

    # Agrupamento por Carga
    cargas = sorted({r["reference_kg"] for r in records})
    print(f"{'Carga Real (kg)':<18} {'N Amostras':>10} {'Média Sum':>14} {'Std Médio':>12} {'CV% Médio':>10}")
    print("-" * 75)
    for c in cargas:
        sub = [r for r in records if r["reference_kg"] == c]
        sums = [r["raw_sum"] for r in sub]
        stds = [r["std_raw"] for r in sub]
        cvs = [r["cv_pct"] for r in sub]
        print(f"{c:<18.3f} {len(sub):>10} {float(np.mean(sums)):>14.1f} {float(np.mean(stds)):>12.2f} {float(np.mean(cvs)):>9.2f}%")

    print("\n" + "="*75)


if __name__ == "__main__":
    main()
