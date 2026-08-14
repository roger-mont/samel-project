"""Avaliação de desempenho do Modelo A (Soma direta de forças) conforme Metodologia §15 e §26.

Lê os arquivos CSV da pasta sessions/ e calcula:
  - Erro Absoluto (EA) e Erro Percentual (EP)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
  - Coeficiente de Determinação (R²)
  - Estratificação por faixa de peso e por posição (center, right, etc.)

Uso:
    python Python/scripts/evaluate_model_a.py
    python Python/scripts/evaluate_model_a.py --sessions-dir sessions/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


def _parse_session_files(sessions_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not sessions_dir.exists():
        return records

    for csv_file in sorted(sessions_dir.glob("calib_*.csv")):
        try:
            with csv_file.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_kg = row.get("reference_kg")
                    est_kg = row.get("estimated_kg")
                    if not ref_kg or not est_kg:
                        continue
                    try:
                        r_kg = float(ref_kg)
                        e_kg = float(est_kg)
                        if r_kg <= 0:
                            continue
                        records.append({
                            "file": csv_file.name,
                            "position": row.get("position_tag", "unknown"),
                            "repetition": int(row.get("repetition", 1)),
                            "reference_kg": r_kg,
                            "estimated_kg": e_kg,
                            "cv_pct": float(row.get("cv_pct", 0.0)),
                            "net_sum": float(row.get("net_sum_block", 0.0)),
                        })
                    except ValueError:
                        continue
        except OSError as err:
            print(f"[AVISO] Falha ao ler {csv_file.name}: {err}")
    return records


def _compute_metrics(refs: np.ndarray, ests: np.ndarray) -> dict[str, float]:
    if len(refs) == 0:
        return {"n": 0, "mae": 0.0, "rmse": 0.0, "ep_mean": 0.0, "r2": 0.0}

    errors = refs - ests
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / refs) * 100.0

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ep_mean = float(np.mean(pct_errors))

    ss_tot = float(np.sum((refs - np.mean(refs)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "n": len(refs),
        "mae": mae,
        "rmse": rmse,
        "ep_mean": ep_mean,
        "r2": r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação do Modelo A (Soma Direta)")
    parser.add_argument("--sessions-dir", type=str, default="sessions",
                        help="Diretório com arquivos CSV de sessões (padrão: sessions)")
    args = parser.parse_args()

    sessions_path = Path(args.sessions_dir)
    records = _parse_session_files(sessions_path)

    print(f"\n{'='*70}")
    print("  RELATÓRIO METROLÓGICO — MODELO A (Soma Direta de Forças)")
    print(f"{'='*70}")
    print(f"Diretório analisado: {sessions_path.resolve()}")
    print(f"Total de amostras carregadas: {len(records)}\n")

    if not records:
        print("[INFO] Nenhuma medição com peso de referência > 0 encontrada em sessions/.")
        sys.exit(0)

    refs_all = np.array([r["reference_kg"] for r in records])
    ests_all = np.array([r["estimated_kg"] for r in records])
    metrics_global = _compute_metrics(refs_all, ests_all)

    print(f"MÉTRICAS GLOBAIS (N={metrics_global['n']}):")
    print(f"  MAE  (Erro Médio Absoluto): {metrics_global['mae']:.4f} kg")
    print(f"  RMSE (Raiz do Erro Quadr.): {metrics_global['rmse']:.4f} kg")
    print(f"  EP   (Erro Percentual Médio): {metrics_global['ep_mean']:.2f}%")
    print(f"  R²   (Coef. de Determinação): {metrics_global['r2']:.4f}\n")

    # Estratificação por Posição
    positions = sorted({r["position"] for r in records})
    print(f"{'Estratificação por Posição':<30} {'N':>4} {'MAE (kg)':>10} {'RMSE (kg)':>11} {'EP (%)':>8}")
    print("-" * 70)
    for pos in positions:
        sub_refs = np.array([r["reference_kg"] for r in records if r["position"] == pos])
        sub_ests = np.array([r["estimated_kg"] for r in records if r["position"] == pos])
        m = _compute_metrics(sub_refs, sub_ests)
        print(f"{pos:<30} {m['n']:>4} {m['mae']:>10.3f} {m['rmse']:>11.3f} {m['ep_mean']:>8.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
