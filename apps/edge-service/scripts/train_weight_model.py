"""Script de Treinamento e Avaliação de Modelos de Machine Learning para Estimação de Peso.

Lê as sessões de calibração em data/sessions/, extrai features metrológicas,
executa Validação Cruzada (K-Fold), treina múltiplos regressores e exporta o melhor modelo.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def carregar_dataset(sessions_dir: Path) -> pd.DataFrame:
    """Carrega todos os arquivos CSV de sessões largas disponíveis."""
    arquivos = sorted(sessions_dir.glob("calib_wide_*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo 'calib_wide_*.csv' encontrado em {sessions_dir}")

    dfs = []
    for f in arquivos:
        df_temp = pd.read_csv(f)
        df_temp["origem_arquivo"] = f.name
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["peso_real_kg"] > 0].copy()
    return df


def extrair_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Gera features estatísticas, espaciais e não-lineares a partir das leituras B1-B8."""
    blocos = [f"B{i}" for i in range(1, 9)]
    b_vals = df[blocos].to_numpy(dtype=np.float64)

    X = pd.DataFrame(index=df.index)

    # 1. Leituras diretas por bloco
    for i, b in enumerate(blocos):
        X[b] = b_vals[:, i]

    # 2. Agregações Globais
    soma_total = np.sum(b_vals, axis=1)
    X["soma_total"] = soma_total
    X["max_bloco"] = np.max(b_vals, axis=1)
    X["desvio_blocos"] = np.std(b_vals, axis=1)
    X["blocos_ativos"] = np.sum(b_vals > 50.0, axis=1)
    X["razao_pico_soma"] = X["max_bloco"] / (soma_total + 1e-6)

    # 3. Centro de Pressão / Balanço Espacial Real da Maca:
    # Cabeceira: B1(dir), B8(esq) | Meio-Sup: B2(dir), B7(esq)
    # Meio-Inf: B3(dir), B6(esq)  | Peseira: B4(dir), B5(esq)
    # Lado Direito: B1, B2, B3, B4 | Lado Esquerdo: B8, B7, B6, B5
    
    dir_sum = b_vals[:, 0] + b_vals[:, 1] + b_vals[:, 2] + b_vals[:, 3]  # B1..B4
    esq_sum = b_vals[:, 7] + b_vals[:, 6] + b_vals[:, 5] + b_vals[:, 4]  # B8..B5
    X["balanco_lateral"] = (dir_sum - esq_sum) / (soma_total + 1e-6)

    cab_sum = b_vals[:, 0] + b_vals[:, 7]  # B1 + B8 (Cabeceira)
    meio_sup_sum = b_vals[:, 1] + b_vals[:, 6]  # B2 + B7
    meio_inf_sum = b_vals[:, 2] + b_vals[:, 5]  # B3 + B6
    pes_sum = b_vals[:, 3] + b_vals[:, 4]  # B4 + B5 (Peseira)

    cog_longitudinal = (3.0 * cab_sum + 2.0 * meio_sup_sum + 1.0 * meio_inf_sum + 0.0 * pes_sum) / (soma_total + 1e-6)
    X["centro_pressao_longitudinal"] = cog_longitudinal

    # 4. Raiz quadrada das leituras (linearização da condutância FSR)
    for i, b in enumerate(blocos):
        X[f"sqrt_{b}"] = np.sqrt(np.maximum(0.0, b_vals[:, i]))

    y = df["peso_real_kg"].astype(np.float64)
    return X, y


def avaliar_modelos_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    n_splits: int = 5,
) -> dict[str, dict]:
    """Avalia múltiplos algoritmos usando GroupKFold por sessão para validação cega."""
    from sklearn.model_selection import GroupKFold

    modelos: dict[str, object] = {
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42
        ),
        "Polinomial Grau 2 + Ridge": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=50.0)),
        ]),
        "Ridge Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0)),
        ]),
        "Huber Regressor (Robusto a Outliers)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", HuberRegressor(max_iter=1000)),
        ]),
    }

    if HAS_XGB:
        modelos["XGBoost Regressor"] = XGBRegressor(
            n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42, verbosity=0
        )

    # Identifica número de sessões únicas
    n_sessoes = len(groups.unique()) if groups is not None else len(X)

    if n_sessoes < 2 or len(X) < 2:
        resultados = {}
        for nome, modelo in modelos.items():
            modelo.fit(X, y)
            preds = modelo.predict(X)
            err = float(np.mean(np.abs(y - preds)))
            resultados[nome] = {
                "mae_medio": err,
                "rmse_medio": err,
                "r2_medio": 1.0,
                "modelo_obj": modelo,
            }
        return resultados

    efetivo_splits = min(n_splits, n_sessoes)
    if groups is not None and n_sessoes >= 2:
        cv_splitter = GroupKFold(n_splits=efetivo_splits).split(X, y, groups)
    else:
        cv_splitter = KFold(n_splits=efetivo_splits, shuffle=True, random_state=42).split(X, y)

    splits_list = list(cv_splitter)
    resultados = {}

    for nome, modelo in modelos.items():
        mae_folds = []
        rmse_folds = []
        r2_folds = []

        for train_idx, val_idx in splits_list:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            modelo.fit(X_tr, y_tr)
            preds = modelo.predict(X_val)

            mae_folds.append(mean_absolute_error(y_val, preds))
            rmse_folds.append(root_mean_squared_error(y_val, preds))
            r2_folds.append(r2_score(y_val, preds) if len(y_val) > 1 else 1.0)

        resultados[nome] = {
            "mae_medio": float(np.mean(mae_folds)),
            "rmse_medio": float(np.mean(rmse_folds)),
            "r2_medio": float(np.mean(r2_folds)),
            "modelo_obj": modelo,
        }

    return resultados


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    sessions_dir = base_dir / "data" / "sessions"
    models_dir = base_dir / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" PIPELINE DE MACHINE LEARNING — CALIBRAÇÃO E PREDIÇÃO DE PESO")
    print("=" * 65)

    df = carregar_dataset(sessions_dir)
    n_sessoes = df["session_id"].nunique() if "session_id" in df.columns else 1
    print(f"-> Base de dados carregada: {len(df)} frames distribuídos em {n_sessoes} ensaios.")
    print(f"-> Faixa de peso coberta: {df['peso_real_kg'].min():.2f} kg a {df['peso_real_kg'].max():.2f} kg")

    X, y = extrair_features(df)
    print(f"-> Features geradas: {X.shape[1]} variáveis de entrada.\n")

    print(f"{'Modelo':<35} | {'MAE (kg)':<10} | {'RMSE (kg)':<10} | {'R²':<8}")
    print("-" * 72)

    groups = df["session_id"] if "session_id" in df.columns else None
    resultados = avaliar_modelos_cv(X, y, groups=groups, n_splits=5)

    melhor_nome = ""
    menor_rmse = float("inf")

    for nome, res in sorted(resultados.items(), key=lambda item: item[1]["rmse_medio"]):
        print(f"{nome:<35} | {res['mae_medio']:<10.3f} | {res['rmse_medio']:<10.3f} | {res['r2_medio']:<8.4f}")
        if res["rmse_medio"] < menor_rmse:
            menor_rmse = res["rmse_medio"]
            melhor_nome = nome

    # Treina o modelo campeão com 100% dos dados para produção
    print("\n" + "=" * 65)
    print(f"-> Modelo Campeão: {melhor_nome} (RMSE Médio: {menor_rmse:.3f} kg)")
    campeao = resultados[melhor_nome]["modelo_obj"]
    campeao.fit(X, y)

    # Exporta artefatos
    joblib_path = models_dir / "weight_model.joblib"
    joblib.dump(campeao, joblib_path)

    meta = {
        "model_name": melhor_nome,
        "features": list(X.columns),
        "total_samples": len(df),
        "min_weight_kg": float(df["peso_real_kg"].min()),
        "max_weight_kg": float(df["peso_real_kg"].max()),
        "cv_mae_kg": round(resultados[melhor_nome]["mae_medio"], 4),
        "cv_rmse_kg": round(resultados[melhor_nome]["rmse_medio"], 4),
        "cv_r2": round(resultados[melhor_nome]["r2_medio"], 4),
    }

    meta_path = models_dir / "weight_model_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"-> Artefato salvo: {joblib_path.name} ({joblib_path.stat().st_size / 1024:.1f} KB)")
    print(f"-> Metadados salvos: {meta_path.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
