"""Máquina de estados da leitura e detecção de estabilidade e movimento.

SEM_CARGA → CARGA_DETECTADA → ESTABILIZANDO → ESTAVEL → MEDICAO_VALIDA
Movimento significativo: (qualquer estado com carga) → MOVIMENTO → ESTABILIZANDO.

Critérios (itens 5, 6 e 8 das orientações), sobre a soma corrigida S(t) = Σ x_ij(t):
- carga presente quando μ(t) > load_threshold_raw;
- μ(t), σ(t): média e desvio-padrão de S na janela STABILITY_WINDOW;
- D(t) = |μ(t) - μ(t - Δt)|;
- D_M(t) = Σ |x_ij(t) - x_ij(t - Δt)|, índice de movimento = D_M / μ;
- estável quando σ/μ < STD_THRESHOLD e D/μ < DELTA_THRESHOLD durante MIN_STABLE_TIME.

Estável não significa correto: a passagem de ESTAVEL para MEDICAO_VALIDA depende
das flags de qualidade, avaliadas fora daqui (set_validated).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.settings import CalibrationParams


class ReadingState(str, Enum):
    SEM_CARGA = "SEM_CARGA"
    CARGA_DETECTADA = "CARGA_DETECTADA"
    ESTABILIZANDO = "ESTABILIZANDO"
    ESTAVEL = "ESTAVEL"
    MEDICAO_VALIDA = "MEDICAO_VALIDA"
    MOVIMENTO = "MOVIMENTO"


STATE_LABELS: dict[ReadingState, str] = {
    ReadingState.SEM_CARGA: "SEM CARGA",
    ReadingState.CARGA_DETECTADA: "CARGA DETECTADA",
    ReadingState.ESTABILIZANDO: "ESTABILIZANDO",
    ReadingState.ESTAVEL: "SINAL ESTÁVEL — PESO NÃO VALIDADO",
    ReadingState.MEDICAO_VALIDA: "MEDIÇÃO VÁLIDA",
    ReadingState.MOVIMENTO: "PACIENTE EM MOVIMENTO / AGUARDANDO ESTABILIZAÇÃO",
}


@dataclass(frozen=True)
class StabilityMetrics:
    total_raw: float              # S(t), soma corrigida instantânea
    window_mean: float            # μ(t)
    window_std: float             # σ(t)
    delta: float | None           # D(t); None enquanto não há histórico de Δt
    movement_sum: float | None    # D_M(t)
    movement_index: float | None  # D_M(t) / μ(t)
    signal_stable: bool
    movement: bool
    stabilization_time_s: float | None
    stable_progress_pct: float


_EPS = 1e-9


class ReadingStateMachine:
    """Classifica cada frame corrigido em um estado de leitura."""

    def __init__(self, params: CalibrationParams) -> None:
        self._params = params
        self._state = ReadingState.SEM_CARGA
        self._frames: deque[tuple[float, float, np.ndarray]] = deque()
        self._means: deque[tuple[float, float]] = deque()
        self._load_since: float | None = None
        self._t_ref: float | None = None          # início da estabilização em curso (t0)
        self._stable_since: float | None = None   # desde quando o critério está atendido
        self._stabilization_time_s: float | None = None
        self._metrics = StabilityMetrics(0.0, 0.0, 0.0, None, None, None, False, False, None, 0.0)

    @property
    def state(self) -> ReadingState:
        return self._state

    @property
    def metrics(self) -> StabilityMetrics:
        return self._metrics

    @property
    def window_frames(self) -> list[np.ndarray]:
        """Frames corrigidos da janela de estabilidade (base da estimativa filtrada)."""
        window = self._params.snapshot()["stability_window_s"]
        if not self._frames:
            return []
        t_last = self._frames[-1][0]
        return [m for t, _, m in self._frames if t >= t_last - window]

    def reset(self) -> None:
        self._state = ReadingState.SEM_CARGA
        self._frames.clear()
        self._means.clear()
        self._load_since = None
        self._t_ref = None
        self._stable_since = None
        self._stabilization_time_s = None

    def set_validated(self, weight_valid: bool) -> ReadingState:
        """Alterna ESTAVEL ↔ MEDICAO_VALIDA conforme as flags de qualidade."""
        if self._state == ReadingState.ESTAVEL and weight_valid:
            self._state = ReadingState.MEDICAO_VALIDA
        elif self._state == ReadingState.MEDICAO_VALIDA and not weight_valid:
            self._state = ReadingState.ESTAVEL
        return self._state

    # ------------------------------------------------------------------ frame

    def update(self, corrected: np.ndarray, now: float) -> ReadingState:
        p = self._params.snapshot()
        window_s = p["stability_window_s"]
        lag_s = p["stability_lag_s"]

        total = float(np.sum(corrected))
        self._frames.append((now, total, corrected.astype(np.float32)))
        horizon = now - (window_s + lag_s + 0.5)
        while self._frames and self._frames[0][0] < horizon:
            self._frames.popleft()

        in_window = [s for t, s, _ in self._frames if t >= now - window_s]
        mu = float(np.mean(in_window))
        sigma = float(np.std(in_window, ddof=1)) if len(in_window) > 1 else 0.0
        self._means.append((now, mu))
        while self._means and self._means[0][0] < horizon:
            self._means.popleft()

        mu_lag = self._value_at(self._means, now - lag_s)
        frame_lag = self._frame_at(now - lag_s)
        delta = abs(mu - mu_lag) if mu_lag is not None else None
        movement_sum = float(np.sum(np.abs(corrected - frame_lag))) if frame_lag is not None else None
        ref = max(mu, p["load_threshold_raw"], _EPS)
        movement_index = movement_sum / ref if movement_sum is not None else None

        # Carga decidida pela média da janela: um frame ruidoso da manta vazia não dispara
        if mu <= p["load_threshold_raw"]:
            self._enter_no_load()
            return self._publish(total, mu, sigma, delta, movement_sum, movement_index, False, False, p)

        if self._state == ReadingState.SEM_CARGA:
            # t0 = primeiro frame da janela acima do limiar (aplicação da carga)
            t0 = next(
                (t for t, s, _ in self._frames if t >= now - window_s and s > p["load_threshold_raw"]),
                now,
            )
            self._state = ReadingState.CARGA_DETECTADA
            self._load_since = t0
            self._t_ref = t0
            self._stable_since = None
            self._stabilization_time_s = None

        # Só avalia estabilidade e movimento quando toda a janela e o Δt já estão sob carga
        warmed_up = self._load_since is not None and now - self._load_since >= window_s + lag_s
        if not warmed_up or delta is None or movement_index is None:
            self._state = ReadingState.CARGA_DETECTADA
            return self._publish(total, mu, sigma, delta, movement_sum, movement_index, False, False, p)

        movement = movement_index > p["movement_threshold_rel"]
        if movement:
            self._state = ReadingState.MOVIMENTO
            self._stable_since = None
            self._stabilization_time_s = None
            return self._publish(total, mu, sigma, delta, movement_sum, movement_index, False, True, p)

        if self._state == ReadingState.MOVIMENTO:
            self._t_ref = now

        condition = (
            sigma / max(mu, _EPS) < p["std_threshold_rel"]
            and delta / max(mu, _EPS) < p["delta_threshold_rel"]
        )
        if not condition:
            if self._state in (ReadingState.ESTAVEL, ReadingState.MEDICAO_VALIDA):
                self._t_ref = now
                self._stabilization_time_s = None
            self._stable_since = None
            self._state = ReadingState.ESTABILIZANDO
            return self._publish(total, mu, sigma, delta, movement_sum, movement_index, False, False, p)

        if self._stable_since is None:
            self._stable_since = now
        if now - self._stable_since >= p["min_stable_time_s"]:
            if self._state not in (ReadingState.ESTAVEL, ReadingState.MEDICAO_VALIDA):
                self._state = ReadingState.ESTAVEL
                self._stabilization_time_s = now - (self._t_ref if self._t_ref is not None else now)
            return self._publish(total, mu, sigma, delta, movement_sum, movement_index, True, False, p)

        self._state = ReadingState.ESTABILIZANDO
        return self._publish(total, mu, sigma, delta, movement_sum, movement_index, False, False, p)

    # ---------------------------------------------------------------- helpers

    def _enter_no_load(self) -> None:
        self._state = ReadingState.SEM_CARGA
        self._load_since = None
        self._t_ref = None
        self._stable_since = None
        self._stabilization_time_s = None

    def _publish(
        self,
        total: float,
        mu: float,
        sigma: float,
        delta: float | None,
        movement_sum: float | None,
        movement_index: float | None,
        stable: bool,
        movement: bool,
        p: dict,
    ) -> ReadingState:
        progress = 0.0
        if stable:
            progress = 100.0
        elif self._stable_since is not None and self._frames:
            elapsed = self._frames[-1][0] - self._stable_since
            progress = min(99.0, 100.0 * elapsed / max(p["min_stable_time_s"], _EPS))
        self._metrics = StabilityMetrics(
            total_raw=total,
            window_mean=mu,
            window_std=sigma,
            delta=delta,
            movement_sum=movement_sum,
            movement_index=movement_index,
            signal_stable=stable,
            movement=movement,
            stabilization_time_s=self._stabilization_time_s,
            stable_progress_pct=progress,
        )
        return self._state

    @staticmethod
    def _value_at(series: deque[tuple[float, float]], when: float) -> float | None:
        """Último valor registrado até o instante 'when'."""
        found: float | None = None
        for t, v in series:
            if t > when:
                break
            found = v
        return found

    def _frame_at(self, when: float) -> np.ndarray | None:
        found: np.ndarray | None = None
        for t, _, m in self._frames:
            if t > when:
                break
            found = m
        return found
