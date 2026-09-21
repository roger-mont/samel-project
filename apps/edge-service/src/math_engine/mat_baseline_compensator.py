"""Compensador de baseline (zero fisico) para manta de sensores FSR."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np

from core.settings import HID_ROWS, HID_COLS, EMPTY_BED_MAX_RAW_SUM

logger = logging.getLogger(__name__)


class MatBaselineCompensator:
    """Subtrai e calibra o baseline matricial B(t) da manta desocupada."""

    def __init__(self, baseline_path: Path | None = None) -> None:
        self._baseline_path = baseline_path
        self._baseline_matrix = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
        self._empty_since_time: float | None = None
        self._load_persisted_baseline()

    @property
    def baseline_matrix(self) -> np.ndarray:
        return self._baseline_matrix.copy()

    @property
    def baseline_mean_raw(self) -> float:
        return float(np.mean(self._baseline_matrix))

    @property
    def baseline_sum_raw(self) -> float:
        return float(np.sum(self._baseline_matrix))

    def correct_frame(self, raw_matrix: np.ndarray) -> np.ndarray:
        """Aplica correcao sc(t) = max(0, s(t) - B(t)) elemento a elemento."""
        if raw_matrix.shape != (HID_ROWS, HID_COLS):
            return np.maximum(0.0, raw_matrix)
        corrected = raw_matrix.astype(np.float64) - self._baseline_matrix
        return np.maximum(0.0, corrected)

    def is_bed_empty(self, raw_matrix: np.ndarray) -> bool:
        """Avalia se a manta nao possui carga externa significativa."""
        total_sum = float(np.sum(raw_matrix))
        if total_sum > EMPTY_BED_MAX_RAW_SUM:
            return False
        max_cell = float(np.max(raw_matrix))
        return max_cell <= 35.0

    def evaluate_auto_rezero(
        self,
        raw_matrix: np.ndarray,
        current_time: float,
        min_empty_seconds: float = 30.0,
    ) -> bool:
        """Verifica se a maca permaneceu vazia por tempo suficiente para auto-zero."""
        if not self.is_bed_empty(raw_matrix):
            self._empty_since_time = None
            return False
        if self._empty_since_time is None:
            self._empty_since_time = current_time
            return False
        elapsed = current_time - self._empty_since_time
        if elapsed < min_empty_seconds:
            return False
        self._empty_since_time = current_time
        return True

    def update_baseline(self, raw_matrix: np.ndarray) -> None:
        """Define novo baseline e persiste em disco de forma idempotente."""
        if raw_matrix.shape != (HID_ROWS, HID_COLS):
            logger.warning("Dimensoes invalidas para baseline: %s", raw_matrix.shape)
            return
        self._baseline_matrix = raw_matrix.astype(np.float64).copy()
        self._persist_baseline()
        logger.info(
            "Baseline atualizado: media=%.2f, soma=%.1f",
            self.baseline_mean_raw,
            self.baseline_sum_raw,
        )

    def _persist_baseline(self) -> None:
        """Salva a matriz de baseline como array JSON no disco."""
        if self._baseline_path is None:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"baseline_raw": self._baseline_matrix.tolist()}
            with self._baseline_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as err:
            logger.error("Falha ao salvar baseline em %s: %s", self._baseline_path, err)

    def _load_persisted_baseline(self) -> None:
        """Carrega matriz de baseline persistida se disponivel."""
        if self._baseline_path is None or not self._baseline_path.exists():
            return
        try:
            with self._baseline_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            matrix_list = data.get("baseline_raw")
            if matrix_list and len(matrix_list) == HID_ROWS:
                self._baseline_matrix = np.array(matrix_list, dtype=np.float64)
                logger.info("Baseline carregado de %s", self._baseline_path)
        except Exception as err:
            logger.warning("Falha ao carregar baseline em %s: %s", self._baseline_path, err)
