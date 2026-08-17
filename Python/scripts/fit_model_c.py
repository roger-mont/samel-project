"""Ajuste e validação do Modelo C (Regressão Linear de Correção Global) conforme Metodologia §17 e §24.

Equação:
    m̂ = a × m_fisica + b

Uso:
    python Python/scripts/fit_model_c.py
    python Python/scripts/fit_model_c.py --sessions-dir sessions/ --output calibration.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from services.calibration_store import GRAVITY_M_S2


def _load_validation_points(sessions_dir: Path) -> list[dict]:
    """Carrega todos os pontos de medição com referência e estimativa válida."""
    points: list[dict] = []
    for csv_path in sorted(sessions_dir.glob("calib_*.csv")):
        try:
            with csv_path.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_kg_s = row.get("reference_kg")
                    est_kg_s = row.get("estimated_kg")
                    if not ref_kg_s or not est_kg_s:
                        continue
                    try:
                        r_kg = float(ref_kg_s)
                        e_kg = float(est_kg_s)
                        if r_kg <= 0.0 or e_kg <= 0.0:
                            continue
                        points.append({
                            "session": csv_path.name,
                            "block_id": row.get("block_id", "unknown"),
                            "position": row.get("position_tag", "center"),
                            "reference_kg": r_kg,
                            "estimated_kg": e_kg,
                        })
                    except ValueError:
                        continue
        except OSError as err:
            print(f"[AVISO] Falha ao ler {csv_path.name}: {err}")
    return points


def _stratified_split(
    points: list[dict],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Divide os dados em treino e validação estratificado por faixas de massa (Metodologia §24)."""
    rng = np.random.default_rng(seed)

    # Faixas: Leve (< 3.0 kg), Médio (3.0 a 6.0 kg), Pesado (> 6.0 kg)
    strata: dict[str, list[dict]] = {"light": [], "medium": [], "heavy": []}
    for p in points:
        ref = p["reference_kg"]
        if ref < 3.0:
            strata["light"].append(p)
        elif ref <= 6.0:
            strata["medium"].append(p)
        else:
            strata["heavy"].append(p)

    train_pts: list[dict] = []
    val_pts: list[dict] = []

    for group in strata.values():
        if not group:
            continue
        indices = np.arange(len(group))
        rng.shuffle(indices)
        n_train = max(1, int(round(len(group) * train_ratio)))
        # Se só tiver 1 item no grupo, vai para treino se possível
        if len(group) == 1:
            train_pts.append(group[0])
            continue
        for idx in indices[:n_train]:
            train_pts.append(group[idx])
        for idx in indices[n_train:]:
            val_pts.append(group[idx])

    # Se validação ficou vazia por arredondamento, move pelo menos 1 amostra de treino
    if not val_pts and len(train_pts) > 1:
        val_pts.append(train_pts.pop())

    return train_pts, val_pts


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / y_true) * 100.0

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ep = float(np.mean(pct_errors))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {"mae": mae, "rmse": rmse, "ep": ep, "r2": r2}


def fit_model_c(
    sessions_dir: Path,
    output_json: Path,
    train_ratio: float = 0.7,
) -> bool:
    points = _load_validation_points(sessions_dir)
    if len(points) < 5:
        print(f"[ERRO] Quantidade insuficiente de pontos com 'estimated_kg' (encontrados: {len(points)}, mínimo: 5).")
        return False

    train_pts, val_pts = _stratified_split(points, train_ratio=train_ratio)

    x_train = np.array([p["estimated_kg"] for p in train_pts])
    y_train = np.array([p["reference_kg"] for p in train_pts])

    x_val = np.array([p["estimated_kg"] for p in val_pts])
    y_val = np.array([p["reference_kg"] for p in val_pts])

    # Ajuste linear y = a * x + b
    coeffs = np.polyfit(x_train, y_train, deg=1)
    a = float(coeffs[0])
    b = float(coeffs[1])

    # Métricas no conjunto de Validação
    y_val_pred = np.maximum(0.0, a * x_val + b)
    val_metrics = _calc_metrics(y_val, y_val_pred)

    # Métricas no conjunto de Treino
    y_train_pred = np.maximum(0.0, a * x_train + b)
    train_metrics = _calc_metrics(y_train, y_train_pred)

    # Métricas Globais antes da correção (Baseline)
    x_all = np.array([p["estimated_kg"] for p in points])
    y_all = np.array([p["reference_kg"] for p in points])
    baseline_metrics = _calc_metrics(y_all, x_all)

    # Métricas Globais após correção
    y_all_corrected = np.maximum(0.0, a * x_all + b)
    corrected_all_metrics = _calc_metrics(y_all, y_all_corrected)

    # Atualiza calibration.json
    if not output_json.exists():
        print(f"[ERRO] Arquivo {output_json} não encontrado.")
        return False

    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        print(f"[ERRO] JSON inválido: {err}")
        return False

    data["version"] = 3
    data["correction_model_c"] = {
        "a": round(a, 6),
        "b": round(b, 6),
        "rmse_validation_kg": round(val_metrics["rmse"], 4),
        "mae_validation_kg": round(val_metrics["mae"], 4),
        "r2_validation": round(val_metrics["r2"], 4),
        "trained_on_n_samples": len(train_pts),
        "validated_on_n_samples": len(val_pts),
    }

    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  RESULTADOS DO AJUSTE -- MODELO C (Regressao Linear de Correcao)")
    print("=" * 70)
    print(f"Total de amostras: {len(points)} (Treino: {len(train_pts)}, Validacao: {len(val_pts)})")
    print(f"\nEquacao de Correcao Global:")
    print(f"  m_hat = {a:.6f} * m_fisica + ({b:.6f})")
    print("\n" + "-" * 70)
    print(f"{'Conjunto':<20} {'N':>4} {'MAE (kg)':>12} {'RMSE (kg)':>12} {'EP (%)':>10} {'R2':>8}")
    print("-" * 70)
    print(f"{'Baseline Global (A)':<20} {len(points):>4} {baseline_metrics['mae']:>12.4f} {baseline_metrics['rmse']:>12.4f} {baseline_metrics['ep']:>9.1f}% {baseline_metrics['r2']:>8.4f}")
    print(f"{'Treino (C)':<20} {len(train_pts):>4} {train_metrics['mae']:>12.4f} {train_metrics['rmse']:>12.4f} {train_metrics['ep']:>9.1f}% {train_metrics['r2']:>8.4f}")
    print(f"{'Validacao (C)':<20} {len(val_pts):>4} {val_metrics['mae']:>12.4f} {val_metrics['rmse']:>12.4f} {val_metrics['ep']:>9.1f}% {val_metrics['r2']:>8.4f}")
    print(f"{'Global Corrigido (C)':<20} {len(points):>4} {corrected_all_metrics['mae']:>12.4f} {corrected_all_metrics['rmse']:>12.4f} {corrected_all_metrics['ep']:>9.1f}% {corrected_all_metrics['r2']:>8.4f}")
    print("=" * 70)
    print(f"\n[OK] Parametros do Modelo C salvos com sucesso em: {output_json.resolve()}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajuste do Modelo C de Correção Global")
    parser.add_argument("--sessions-dir", type=str, default="sessions",
                        help="Diretório de sessões CSV (padrão: sessions)")
    parser.add_argument("--output", type=str, default="calibration.json",
                        help="Arquivo calibration.json (padrão: calibration.json)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Proporção de amostras para treino (padrão: 0.7)")
    args = parser.parse_args()

    fit_model_c(Path(args.sessions_dir), Path(args.output), args.train_ratio)


if __name__ == "__main__":
    main()
