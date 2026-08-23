"""Classificador postural e monitor de permanência estática para prevenção de LPP."""
from __future__ import annotations

import time
import numpy as np


class PostureMonitor:
    """Monitor de postura contínuo e gerador de alertas de rotação de decúbito."""

    def __init__(self, tolerance: float = 0.15, timeout_seconds: int = 3600):
        self._tolerance = tolerance
        self._timeout_seconds = timeout_seconds
        self._reference_snapshot: np.ndarray | None = None
        self._static_start: float | None = None
        self._is_alert = False

    @property
    def elapsed_seconds(self) -> float:
        if self._static_start is None:
            return 0.0
        return time.monotonic() - self._static_start

    @property
    def is_alert(self) -> bool:
        return self._is_alert

    def reset(self) -> None:
        self._reference_snapshot = None
        self._static_start = None
        self._is_alert = False

    def update_tolerance(self, tolerance: float) -> None:
        self._tolerance = tolerance

    def update_timeout(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds

    def update(self, force_matrix: np.ndarray) -> bool:
        """Processa novo frame. Retorna True se atingiu o limiar de alerta LPP."""
        total = float(np.sum(force_matrix))
        if total < 1e-6:
            self._reset_timer()
            return False

        normalized = self._normalize(force_matrix)
        if self._reference_snapshot is None:
            self._set_reference(normalized)
            return False

        similarity = self._compute_similarity(normalized, self._reference_snapshot)
        threshold = 1.0 - self._tolerance

        if similarity >= threshold:
            return self._check_timeout()

        self._set_reference(normalized)
        return False

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        total = float(np.sum(matrix))
        if total < 1e-6:
            return np.zeros_like(matrix)
        return matrix / total

    def _compute_similarity(self, current: np.ndarray, reference: np.ndarray) -> float:
        norm_curr = np.linalg.norm(current)
        norm_ref = np.linalg.norm(reference)
        if norm_curr < 1e-9 or norm_ref < 1e-9:
            return 0.0
        cosine = float(np.sum(current * reference)) / (norm_curr * norm_ref)
        return max(0.0, min(1.0, cosine))

    def _set_reference(self, normalized: np.ndarray) -> None:
        self._reference_snapshot = normalized.copy()
        self._static_start = time.monotonic()
        self._is_alert = False

    def _reset_timer(self) -> None:
        self._static_start = None
        self._is_alert = False

    def _check_timeout(self) -> bool:
        if self._static_start is None:
            self._static_start = time.monotonic()
            return False
        if self.elapsed_seconds >= self._timeout_seconds:
            self._is_alert = True
            return True
        return False

    def classify_posture(self, matrix: np.ndarray) -> dict:
        """Classifica postura clínica, assimetria lateral e score de alívio."""
        total_sum = float(np.sum(matrix))
        if total_sum < 1.0:
            return {
                "posture": "Leito Livre",
                "asymmetry_pct": 0.0,
                "asymmetry_label": "Sem Carga",
                "relief_score": 100,
            }

        rows, cols = matrix.shape
        half_rows = rows // 2
        half_cols = cols // 2

        left_sum = float(np.sum(matrix[:half_rows, :]))
        right_sum = float(np.sum(matrix[half_rows:, :]))
        diff = right_sum - left_sum
        asym_pct = (abs(diff) / max(total_sum, 1e-6)) * 100.0

        torso_sum = float(np.sum(matrix[:, half_cols:]))
        legs_sum = float(np.sum(matrix[:, :half_cols]))
        peak_val = float(np.max(matrix))

        if asym_pct > 28.0:
            if diff > 0:
                posture = "Decúbito Lat. Dir."
                asym_label = f"Assimetria: {round(asym_pct)}% Dir"
            else:
                posture = "Decúbito Lat. Esq."
                asym_label = f"Assimetria: {round(asym_pct)}% Esq"
        else:
            if torso_sum / (legs_sum + 0.1) > 3.2:
                posture = "Posição de Fowler"
                asym_label = "Cabeceira Elevada"
            else:
                posture = "Decúbito Dorsal"
                asym_label = "Pressão Simétrica"

        relief_score = max(15, min(99, int(100 - (peak_val * 65) - (asym_pct * 0.25))))

        return {
            "posture": posture,
            "asymmetry_pct": round(asym_pct, 1),
            "asymmetry_label": asym_label,
            "relief_score": relief_score,
        }
