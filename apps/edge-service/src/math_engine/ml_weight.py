"""Módulo de inferência de peso via modelo de Machine Learning treinado."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from math_engine.spatial_features import compute_cop_normalized, compute_block_stats
from storage.calibration_store import BLOCK_REGIONS

logger = logging.getLogger(__name__)


class MLWeightPredictor:
    """Gerencia o carregamento e predição do modelo de Machine Learning no Edge Service."""

    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None) -> None:
        if model_path is None:
            base = Path(__file__).resolve().parent.parent.parent
            model_path = base / "data" / "models" / "weight_model.joblib"
            meta_path = base / "data" / "models" / "weight_model_meta.json"

        self._model_path = model_path
        self._meta_path = meta_path
        self._model: Any = None
        self._meta: dict[str, Any] = {}
        self._loaded = False
        self._feature_names: list[str] = []
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._meta

    def load(self) -> bool:
        """Carrega o modelo joblib e metadados se existirem no disco."""
        if not self._model_path.exists():
            logger.info("Modelo ML não encontrado em %s (usando calibração padrão)", self._model_path)
            self._loaded = False
            return False

        try:
            self._model = joblib.load(self._model_path)
            if self._meta_path and self._meta_path.exists():
                with self._meta_path.open("r", encoding="utf-8") as f:
                    self._meta = json.load(f)
            self._feature_names = self._meta.get("features", [])
            self._loaded = True
            logger.info("Modelo ML '%s' carregado com sucesso!", self._meta.get("model_name", "ML Model"))
            return True
        except Exception as err:
            logger.warning("Erro ao carregar modelo ML: %s", err)
            self._loaded = False
            return False

    def extract_features(self, matrix_32x64: np.ndarray) -> dict[str, float]:
        """Extrai vetor completo de features da matriz 32x64.

        Retorna dicionário com todas as features disponíveis.
        O modelo usará apenas as que reconhece via metadata.
        """
        # 1. Somas por bloco (B1..B8)
        b_vals = np.zeros(8, dtype=np.float64)
        for bid, (r_sl, c_sl) in BLOCK_REGIONS.items():
            if 1 <= bid <= 8:
                b_vals[bid - 1] = float(np.sum(matrix_32x64[r_sl, c_sl]))

        soma_total = float(np.sum(b_vals))

        row: dict[str, float] = {}
        for i in range(8):
            row[f"B{i+1}"] = b_vals[i]

        # 2. Features agregadas (existentes)
        row["soma_total"] = soma_total
        row["max_bloco"] = float(np.max(b_vals))
        row["desvio_blocos"] = float(np.std(b_vals))
        row["blocos_ativos"] = float(np.sum(b_vals > 50.0))
        row["razao_pico_soma"] = row["max_bloco"] / (soma_total + 1e-6)

        # 3. Balanço lateral
        dir_sum = b_vals[0] + b_vals[1] + b_vals[2] + b_vals[3]
        esq_sum = b_vals[7] + b_vals[6] + b_vals[5] + b_vals[4]
        row["balanco_lateral"] = (dir_sum - esq_sum) / (soma_total + 1e-6)

        # 4. Centro de pressão longitudinal (legado — por zonas)
        cab_sum = b_vals[0] + b_vals[7]
        meio_sup_sum = b_vals[1] + b_vals[6]
        meio_inf_sum = b_vals[2] + b_vals[5]
        pes_sum = b_vals[3] + b_vals[4]
        row["centro_pressao_longitudinal"] = (
            (3.0 * cab_sum + 2.0 * meio_sup_sum + 1.0 * meio_inf_sum + 0.0 * pes_sum)
            / (soma_total + 1e-6)
        )

        # 5. Raiz quadrada por bloco
        for i in range(8):
            row[f"sqrt_B{i+1}"] = float(np.sqrt(max(0.0, b_vals[i])))

        # 6. [NOVO] Centro de Pressão 2D real (pixel-a-pixel)
        cop_row_norm, cop_col_norm = compute_cop_normalized(matrix_32x64)
        row["cop_row"] = cop_row_norm
        row["cop_col"] = cop_col_norm

        # 7. [NOVO] Estatísticas intra-bloco
        block_stats = compute_block_stats(matrix_32x64, BLOCK_REGIONS)
        row.update(block_stats)

        return row

    def predict(self, matrix_32x64: np.ndarray) -> float | None:
        """Extrai features da matriz 32x64 e infere o peso em kg."""
        if not self.is_loaded:
            return None

        try:
            row = self.extract_features(matrix_32x64)
            soma_total = row.get("soma_total", 0.0)

            if soma_total < 10.0:
                return 0.0

            # Usa apenas features conhecidas pelo modelo treinado
            if self._feature_names:
                filtered = {k: row.get(k, 0.0) for k in self._feature_names}
            else:
                filtered = row

            df_input = pd.DataFrame([filtered])
            peso_pred = float(self._model.predict(df_input)[0])
            return max(0.0, peso_pred)
        except Exception as err:
            logger.error("Falha na inferência do modelo ML: %s", err)
            return None
