"""Módulo de Ajuste e Comparação Metrológica de Modelos de Calibração.

Implementa o protocolo de avaliação comparativa e validação cega conforme
as seções §12-§15, §19-§22 e §30 da Solução Completa de Calibração.

Modelos avaliados:
  - Pipeline Atual: Curvas por bloco + Correção Linear Global
  - Modelo A: Soma simples direta (P̂ = a * S_T + b)
  - Modelo B: Regressão Linear Multivariada direta (P̂ = b₀ + Σ bₖ Sₖ) [Candidato Principal]
  - Modelo C: Multivariado com termos quadráticos (P̂ = b₀ + Σ bₖ Sₖ + Σ cₖ Sₖ²)

Uso:
    python Python/scripts/compare_and_fit_models.py
    python Python/scripts/compare_and_fit_models.py --save-model multivariate_linear
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from services.calibration_store import GRAVITY_M_S2, BLOCK_REGIONS, load_calibration


def load_all_samples(sessions_dir: Path) -> list[dict]:
    """Carrega todas as amostras disponíveis e converte para representação unificada da maca."""
    samples: list[dict] = []

    # 1. Carrega datasets em formato largo se existirem
    for wide_path in sorted(sessions_dir.glob("calib_wide_*.csv")):
        try:
            with wide_path.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_kg = float(row.get("peso_real_kg", 0.0))
                    if ref_kg <= 0:
                        continue
                    b_vec = np.array([float(row.get(f"B{i}", 0.0)) for i in range(1, 9)])
                    samples.append({
                        "session": row.get("session_id", wide_path.name),
                        "position": row.get("posicao", "center"),
                        "reference_kg": ref_kg,
                        "b_vector": b_vec,
                        "total_sum": float(row.get("soma_total", float(np.sum(b_vec)))),
                    })
        except OSError as err:
            print(f"[AVISO] Falha ao ler {wide_path.name}: {err}")

    # 2. Carrega dataset consolidado / CSVs individuais por bloco
    consolidated = sessions_dir / "consolidated_calibration_dataset.csv"
    if consolidated.exists():
        try:
            with consolidated.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_kg = float(row.get("reference_kg", 0.0))
                    if ref_kg <= 0:
                        continue
                    bid = int(row.get("block_id", 0))
                    net_sum = float(row.get("net_sum_block", 0.0))
                    b_vec = np.zeros(8, dtype=np.float64)

                    if bid in range(1, 9):
                        b_vec[bid - 1] = net_sum
                    else:
                        # Full body legado: distribui soma nos 8 blocos
                        b_vec[:] = net_sum / 8.0

                    samples.append({
                        "session": row.get("session_id", row.get("source_file", "unknown")),
                        "position": row.get("position_tag", "center"),
                        "reference_kg": ref_kg,
                        "b_vector": b_vec,
                        "total_sum": float(np.sum(b_vec)),
                    })
        except OSError as err:
            print(f"[AVISO] Falha ao ler {consolidated.name}: {err}")

    return samples


def split_blind_validation(
    samples: list[dict],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Divisão estratificada por faixas de peso para validação cega."""
    rng = np.random.default_rng(seed)
    bins: dict[str, list[dict]] = {"baixo": [], "medio": [], "alto": []}

    for s in samples:
        r = s["reference_kg"]
        if r < 5.0:
            bins["baixo"].append(s)
        elif r <= 50.0:
            bins["medio"].append(s)
        else:
            bins["alto"].append(s)

    train: list[dict] = []
    val: list[dict] = []

    for group in bins.values():
        if not group:
            continue
        indices = np.arange(len(group))
        rng.shuffle(indices)
        n_tr = max(1, int(round(len(group) * train_ratio)))
        for i in indices[:n_tr]:
            train.append(group[i])
        for i in indices[n_tr:]:
            val.append(group[i])

    if not val and len(train) > 1:
        val.append(train.pop())

    return train, val


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE, R² e Erro Máximo conforme §21."""
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "max_err": 0.0, "ep_mean": 0.0}

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / np.maximum(y_true, 1e-6)) * 100.0

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_err = float(np.max(abs_errors))
    ep_mean = float(np.mean(pct_errors))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "max_err": max_err,
        "ep_mean": ep_mean,
    }


def fit_model_a(train_samples: list[dict]) -> tuple[float, float]:
    """Modelo A (Soma simples): P̂ = a * S_T + b."""
    X = np.array([s["total_sum"] for s in train_samples])
    y = np.array([s["reference_kg"] for s in train_samples])
    A = np.column_stack([X, np.ones_like(X)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def predict_model_a(samples: list[dict], a: float, b: float) -> np.ndarray:
    X = np.array([s["total_sum"] for s in samples])
    return np.maximum(0.0, a * X + b)


def fit_model_b(train_samples: list[dict]) -> dict[str, float]:
    """Modelo B (Multivariado Direto): P̂ = b₀ + Σ bₖ Sₖ."""
    X = np.array([s["b_vector"] for s in train_samples])  # (N, 8)
    y = np.array([s["reference_kg"] for s in train_samples])
    A = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    result = {"b0": float(coeffs[0])}
    for i in range(8):
        result[f"b{i + 1}"] = float(coeffs[i + 1])
    return result


def predict_model_b(samples: list[dict], coeffs: dict[str, float]) -> np.ndarray:
    X = np.array([s["b_vector"] for s in samples])
    b0 = coeffs.get("b0", 0.0)
    b_vec = np.array([coeffs.get(f"b{i + 1}", 0.0) for i in range(8)])
    preds = b0 + np.dot(X, b_vec)
    return np.maximum(0.0, preds)


def fit_model_c_quad(train_samples: list[dict]) -> dict[str, Any]:
    """Modelo C (Não linearidade quadrática): P̂ = b₀ + Σ bₖ Sₖ + Σ cₖ Sₖ²."""
    X = np.array([s["b_vector"] for s in train_samples])
    X2 = X ** 2
    y = np.array([s["reference_kg"] for s in train_samples])
    A = np.column_stack([np.ones(len(X)), X, X2])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    return {
        "b0": float(coeffs[0]),
        "linear": [float(c) for c in coeffs[1:9]],
        "quadratic": [float(c) for c in coeffs[9:17]],
    }


def predict_model_c_quad(samples: list[dict], coeffs: dict[str, Any]) -> np.ndarray:
    X = np.array([s["b_vector"] for s in samples])
    b0 = coeffs.get("b0", 0.0)
    lin = np.array(coeffs.get("linear", [0.0] * 8))
    quad = np.array(coeffs.get("quadratic", [0.0] * 8))
    preds = b0 + np.dot(X, lin) + np.dot(X ** 2, quad)
    return np.maximum(0.0, preds)


def predict_legacy_pipeline(samples: list[dict], calib_path: Path) -> np.ndarray:
    """Predição via pipeline existente (curvas por bloco + Modelo C)."""
    calib = load_calibration(calib_path)
    preds = []
    for s in samples:
        b_vec = s["b_vector"]
        # Reconstrói contribuição por bloco
        tot_c = 0.0
        for idx in range(8):
            bid = idx + 1
            if bcalib := calib.blocks.get(bid):
                tot_c += bcalib.sum_to_contribution(b_vec[idx])
            else:
                tot_c += b_vec[idx]
        m_base = tot_c / GRAVITY_M_S2
        if calib.correction_c is not None:
            ca, cb = calib.correction_c
            preds.append(max(0.0, ca * m_base + cb))
        else:
            preds.append(m_base)
    return np.array(preds)


def print_comparison_table(results: list[dict]) -> None:
    """Imprime tabela no formato da Seção §30 do documento técnico."""
    print("\n" + "=" * 90)
    print("  RELATÓRIO COMPARATIVO ENTRE MODELOS DE CALIBRAÇÃO (Metodologia §30)")
    print("=" * 90)
    header = f"{'Modelo':<22} | {'Treino MAE':>10} | {'Val MAE':>10} | {'Val RMSE':>10} | {'R² Val':>8} | {'Erro Máx':>10} | {'Status':<12}"
    print(header)
    print("-" * 90)
    for r in results:
        line = (
            f"{r['nome']:<22} | "
            f"{r['tr_mae']:>10.4f} | "
            f"{r['val_mae']:>10.4f} | "
            f"{r['val_rmse']:>10.4f} | "
            f"{r['val_r2']:>8.4f} | "
            f"{r['max_err']:>10.4f} | "
            f"{r['obs']:<12}"
        )
        print(line)
    print("=" * 90)


def save_active_model(
    output_path: Path,
    model_name: str,
    model_a_coeffs: tuple[float, float],
    model_b_coeffs: dict[str, float],
    model_c_coeffs: dict[str, Any],
    dataset_summary: dict,
) -> None:
    """Atualiza o arquivo calibration.json com o modelo ativo e versionamento."""
    data = {"version": 4, "blocks": {}}
    if output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    data["version"] = 4
    data["active_model"] = model_name
    data["calibrated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    data["dataset_info"] = dataset_summary
    data["models"] = {
        "simple_sum": {"a": model_a_coeffs[0], "b": model_a_coeffs[1]},
        "multivariate_linear": model_b_coeffs,
        "multivariate_quadratic": model_c_coeffs,
    }

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Modelo '{model_name}' salvo com sucesso em: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparação e ajuste de modelos de calibração")
    parser.add_argument("--sessions-dir", type=str, default="sessions", help="Pasta com CSVs")
    parser.add_argument("--calib-json", type=str, default="calibration.json", help="Arquivo calibration.json")
    parser.add_argument("--save-model", type=str, choices=["multivariate_linear", "simple_sum", "multivariate_quadratic"],
                        default=None, help="Salva o modelo escolhido como ativo no calibration.json")
    args = parser.parse_args()

    sessions_path = Path(args.sessions_dir)
    calib_json = Path(args.calib_json)

    samples = load_all_samples(sessions_path)
    if not samples:
        print("[ERRO] Nenhuma amostra de calibração encontrada em sessions/.")
        sys.exit(1)

    print(f"[INFO] {len(samples)} amostras carregadas para análise.")
    train_pts, val_pts = split_blind_validation(samples, train_ratio=0.7, seed=42)
    print(f"[INFO] Divisão cega: {len(train_pts)} treino, {len(val_pts)} validação.")

    y_train = np.array([s["reference_kg"] for s in train_pts])
    y_val = np.array([s["reference_kg"] for s in val_pts])

    # 1. Pipeline Atual (Legado)
    leg_pred_tr = predict_legacy_pipeline(train_pts, calib_json)
    leg_pred_val = predict_legacy_pipeline(val_pts, calib_json)
    m_leg_tr = calculate_metrics(y_train, leg_pred_tr)
    m_leg_val = calculate_metrics(y_val, leg_pred_val)

    # 2. Modelo A (Soma Simples)
    a, b = fit_model_a(train_pts)
    a_pred_tr = predict_model_a(train_pts, a, b)
    a_pred_val = predict_model_a(val_pts, a, b)
    m_a_tr = calculate_metrics(y_train, a_pred_tr)
    m_a_val = calculate_metrics(y_val, a_pred_val)

    # 3. Modelo B (Multivariado Direto)
    b_coeffs = fit_model_b(train_pts)
    b_pred_tr = predict_model_b(train_pts, b_coeffs)
    b_pred_val = predict_model_b(val_pts, b_coeffs)
    m_b_tr = calculate_metrics(y_train, b_pred_tr)
    m_b_val = calculate_metrics(y_val, b_pred_val)

    # 4. Modelo C (Quadrático Multivariado)
    c_coeffs = fit_model_c_quad(train_pts)
    c_pred_tr = predict_model_c_quad(train_pts, c_coeffs)
    c_pred_val = predict_model_c_quad(val_pts, c_coeffs)
    m_c_tr = calculate_metrics(y_train, c_pred_tr)
    m_c_val = calculate_metrics(y_val, c_pred_val)

    results = [
        {"nome": "Pipeline Atual", "tr_mae": m_leg_tr["mae"], "val_mae": m_leg_val["mae"], "val_rmse": m_leg_val["rmse"], "val_r2": m_leg_val["r2"], "max_err": m_leg_val["max_err"], "obs": "Referência"},
        {"nome": "Modelo A (Soma)", "tr_mae": m_a_tr["mae"], "val_mae": m_a_val["mae"], "val_rmse": m_a_val["rmse"], "val_r2": m_a_val["r2"], "max_err": m_a_val["max_err"], "obs": "Baseline"},
        {"nome": "Modelo B (Multivariado)", "tr_mae": m_b_tr["mae"], "val_mae": m_b_val["mae"], "val_rmse": m_b_val["rmse"], "val_r2": m_b_val["r2"], "max_err": m_b_val["max_err"], "obs": "Principal"},
        {"nome": "Modelo C (Quadrático)", "tr_mae": m_c_tr["mae"], "val_mae": m_c_val["mae"], "val_rmse": m_c_val["rmse"], "val_r2": m_c_val["r2"], "max_err": m_c_val["max_err"], "obs": "Não linear"},
    ]

    print_comparison_table(results)

    if args.save_model:
        dataset_summary = {
            "total_samples": len(samples),
            "train_samples": len(train_pts),
            "val_samples": len(val_pts),
            "source_sessions": sorted(list({s["session"] for s in samples})),
        }
        save_active_model(
            output_path=calib_json,
            model_name=args.save_model,
            model_a_coeffs=(a, b),
            model_b_coeffs=b_coeffs,
            model_c_coeffs=c_coeffs,
            dataset_summary=dataset_summary,
        )


if __name__ == "__main__":
    main()
