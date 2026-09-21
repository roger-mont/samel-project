"""Detector de estabilidade de leitura e maquina de estados para manta FSR."""
from __future__ import annotations

from collections import deque
from enum import Enum
import math
import numpy as np

from core.settings import (
    STABILITY_EPSILON_RAW,
    STABILITY_SIGMA_MAX,
    STABILITY_DEFAULT_TMIN_S,
    MOTION_DMAX_MATRIX,
    EMPTY_BED_MAX_RAW_SUM,
)


class ReadingState(str, Enum):
    """Estados operacionais de aquisicao e pesagem."""

    NO_LOAD = "SEM CARGA"
    MOVING = "CARREGANDO/MOVIMENTO"
    STABILIZING = "ESTABILIZANDO"
    STABLE = "CARGA ESTAVEL"


class ReadingStabilityDetector:
    """Maquina de estados para estabilizacao de leitura e rejeicao de movimento."""

    def __init__(
        self,
        t_min_s: float = STABILITY_DEFAULT_TMIN_S,
        epsilon_s: float = STABILITY_EPSILON_RAW,
        sigma_max: float = STABILITY_SIGMA_MAX,
        d_max_matrix: float = MOTION_DMAX_MATRIX,
        empty_sum_threshold: float = EMPTY_BED_MAX_RAW_SUM,
        window_size_frames: int = 40,
    ) -> None:
        self.t_min_s = t_min_s
        self.epsilon_s = epsilon_s
        self.sigma_max = sigma_max
        self.d_max_matrix = d_max_matrix
        self.empty_sum_threshold = empty_sum_threshold
        self._window_size = window_size_frames

        self._state = ReadingState.NO_LOAD
        self._prev_matrix: np.ndarray | None = None
        self._raw_history: deque[float] = deque(maxlen=window_size_frames)
        self._prev_mean: float | None = None
        self._stable_start_time: float | None = None
        self._last_matrix_delta = 0.0
        self._last_temporal_delta = 0.0
        self._last_sigma = 0.0

    @property
    def current_state(self) -> ReadingState:
        return self._state

    @property
    def matrix_delta(self) -> float:
        return self._last_matrix_delta

    @property
    def temporal_delta(self) -> float:
        return self._last_temporal_delta

    @property
    def dispersion_sigma(self) -> float:
        return self._last_sigma

    def compute_matrix_delta(self, current_matrix: np.ndarray) -> float:
        """Calcula Dmatriz(t) = sum(|s_ij(t) - s_ij(t - Dt)|)."""
        if self._prev_matrix is None:
            self._prev_matrix = current_matrix.copy()
            return 0.0
        delta = float(np.sum(np.abs(current_matrix - self._prev_matrix)))
        self._prev_matrix = current_matrix.copy()
        return delta

    def compute_window_stats(self, raw_sum: float) -> tuple[float, float, float]:
        """Calcula media movel s_bar, desvio padrao sigma e derivada D(t)."""
        self._raw_history.append(raw_sum)
        mean_val = float(np.mean(self._raw_history))
        std_val = float(np.std(self._raw_history, ddof=1)) if len(self._raw_history) > 1 else 0.0

        if self._prev_mean is None:
            temporal_d = 0.0
        else:
            temporal_d = abs(mean_val - self._prev_mean)
        self._prev_mean = mean_val

        return mean_val, std_val, temporal_d

    def update(
        self,
        current_matrix: np.ndarray,
        current_time: float,
    ) -> ReadingState:
        """Processa frame atual e transiciona estado operacional."""
        raw_sum = float(np.sum(current_matrix))
        self._last_matrix_delta = self.compute_matrix_delta(current_matrix)
        _, self._last_sigma, self._last_temporal_delta = self.compute_window_stats(raw_sum)

        if raw_sum <= self.empty_sum_threshold:
            self._transition_to_no_load()
            return self._state

        if self._last_matrix_delta > self.d_max_matrix:
            self._transition_to_moving()
            return self._state

        self._evaluate_stabilization(current_time)
        return self._state

    def _transition_to_no_load(self) -> None:
        """Reseta contadores ao esvaziar a maca."""
        self._state = ReadingState.NO_LOAD
        self._stable_start_time = None

    def _transition_to_moving(self) -> None:
        """Entra em movimento por variacao dinamica na matriz."""
        self._state = ReadingState.MOVING
        self._stable_start_time = None

    def _evaluate_stabilization(self, current_time: float) -> None:
        """Avalia convergencia para CARGA ESTAVEL com histerese temporal."""
        is_converged = (
            self._last_temporal_delta <= self.epsilon_s
            and self._last_sigma <= self.sigma_max
        )

        if not is_converged:
            self._state = ReadingState.STABILIZING
            self._stable_start_time = None
            return

        if self._stable_start_time is None:
            self._stable_start_time = current_time
            self._state = ReadingState.STABILIZING
            return

        elapsed = current_time - self._stable_start_time
        if elapsed >= self.t_min_s:
            self._state = ReadingState.STABLE
        else:
            self._state = ReadingState.STABILIZING

    def get_progress_pct(self, current_time: float) -> float:
        """Retorna percentual de conclusao da janela de estabilizacao."""
        if self._state == ReadingState.STABLE:
            return 100.0
        if self._state != ReadingState.STABILIZING or self._stable_start_time is None:
            return 0.0
        elapsed = current_time - self._stable_start_time
        return min(99.0, max(0.0, (elapsed / self.t_min_s) * 100.0))
