"""Compensador de creep e drift temporal para sensores FSR.

O creep é um fenômeno físico onde o valor digital dos FSRs sobe gradualmente
quando uma pressão constante é mantida por tempo prolongado. Este módulo
implementa três estratégias de mitigação:

1. Captura de referência estável via mediana dos primeiros N frames.
2. Limitação do drift máximo permitido por segundo.
3. Auto-rezero quando a manta permanece desocupada por tempo configurável.
"""
from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class CreepCompensator:
    """Limita drift temporal e captura referência estável para cada nova carga."""

    def __init__(
        self,
        settle_frames: int = 15,
        max_drift_kg_per_sec: float = 0.3,
        step_threshold_kg: float = 2.0,
    ) -> None:
        self._settle_frames = settle_frames
        self._max_drift_kg_per_sec = max_drift_kg_per_sec
        self._step_threshold_kg = step_threshold_kg

        self._settling_buffer: deque[float] = deque(maxlen=settle_frames)
        self._is_settled = False
        self._reference_kg: float = 0.0
        self._reference_time: float = 0.0
        self._last_output_kg: float = 0.0

    @property
    def is_settled(self) -> bool:
        return self._is_settled

    @property
    def reference_kg(self) -> float:
        return self._reference_kg

    def reset(self) -> None:
        """Reinicia o estado do compensador (ex: quando manta esvazia)."""
        self._settling_buffer.clear()
        self._is_settled = False
        self._reference_kg = 0.0
        self._reference_time = 0.0
        self._last_output_kg = 0.0

    def update(self, raw_mass_kg: float) -> float:
        """Processa nova leitura e retorna valor compensado.

        Fluxo:
        1. Se massa < threshold mínimo, reseta e retorna 0.
        2. Se ainda coletando frames iniciais, acumula na settling_buffer.
        3. Após estabilizar, usa mediana como referência e limita drift.
        """
        now = time.monotonic()

        if raw_mass_kg < 0.5:
            self.reset()
            return 0.0

        if self._is_settled:
            return self._apply_drift_limit(raw_mass_kg, now)

        return self._accumulate_settling(raw_mass_kg, now)

    def _accumulate_settling(self, raw_mass_kg: float, now: float) -> float:
        """Acumula frames iniciais e estabiliza via mediana."""
        self._settling_buffer.append(raw_mass_kg)

        if len(self._settling_buffer) < self._settle_frames:
            return raw_mass_kg

        sorted_buf = sorted(self._settling_buffer)
        median_kg = sorted_buf[len(sorted_buf) // 2]

        self._reference_kg = median_kg
        self._reference_time = now
        self._is_settled = True
        self._last_output_kg = median_kg

        logger.debug(
            "Creep: referência estabilizada em %.2f kg (%d frames)",
            median_kg,
            self._settle_frames,
        )
        return median_kg

    def _apply_drift_limit(self, raw_mass_kg: float, now: float) -> float:
        """Limita quanto o valor pode subir por segundo em relação à referência."""
        delta_from_ref = raw_mass_kg - self._reference_kg

        if abs(delta_from_ref) > self._step_threshold_kg:
            logger.info(
                "Creep: degrau detectado (%.2f kg → %.2f kg) — re-estabilizando",
                self._reference_kg,
                raw_mass_kg,
            )
            self._is_settled = False
            self._settling_buffer.clear()
            self._settling_buffer.append(raw_mass_kg)
            return raw_mass_kg

        elapsed = max(0.001, now - self._reference_time)
        max_drift = self._max_drift_kg_per_sec * elapsed

        if delta_from_ref > max_drift:
            clamped = self._reference_kg + max_drift
            self._last_output_kg = clamped
            return clamped

        self._last_output_kg = raw_mass_kg
        return raw_mass_kg


class AutoReZero:
    """Detecta manta desocupada e dispara re-zero automático."""

    def __init__(
        self,
        empty_threshold_kg: float = 0.5,
        empty_duration_seconds: float = 30.0,
    ) -> None:
        self._empty_threshold = empty_threshold_kg
        self._empty_duration = empty_duration_seconds
        self._empty_start: float | None = None

    def check(self, current_mass_kg: float) -> bool:
        """Retorna True se a manta está vazia por tempo suficiente para re-zero."""
        now = time.monotonic()

        if current_mass_kg > self._empty_threshold:
            self._empty_start = None
            return False

        if self._empty_start is None:
            self._empty_start = now
            return False

        elapsed = now - self._empty_start
        if elapsed >= self._empty_duration:
            self._empty_start = now
            return True

        return False
