"""Monitoramento de postura estática — alerta após 1 minuto sem mudança."""
from __future__ import annotations

import time

import numpy as np


class PostureMonitor:
    """Detecta se o usuário permanece na mesma posição por tempo prolongado.

    Compara distribuição normalizada de força entre frames.
    Se a correlação permanecer acima de (1 - tolerance) por timeout_seconds
    contínuos, sinaliza alerta.
    """

    def __init__(self, tolerance: float = 0.15, timeout_seconds: int = 60):
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
        """Reseta cronômetro e snapshot de referência."""
        self._reference_snapshot = None
        self._static_start = None
        self._is_alert = False

    def update_tolerance(self, tolerance: float) -> None:
        self._tolerance = tolerance

    def update_timeout(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds

    def update(self, force_matrix: np.ndarray) -> bool:
        """Processa novo frame. Retorna True se alerta ativo."""
        total = float(np.sum(force_matrix))

        # Sensor vazio — sem pressão, reseta
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

        # Posição mudou — reseta e atualiza referência
        self._set_reference(normalized)
        return False

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        total = float(np.sum(matrix))
        if total < 1e-6:
            return np.zeros_like(matrix)
        return matrix / total

    def _compute_similarity(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Correlação via produto interno de distribuições normalizadas."""
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
