"""Massa residual com a maca vazia (item 3.5 das orientações de 15/10).

Depois da tara, com a maca sem carga, acumula a massa total que o modelo publicaria
em cada frame: média, desvio, máximo absoluto e a contribuição de cada setor.
Pequenos vieses por célula somados em até 2.048 células aparecem aqui.

As estatísticas valem para uma tara: uma tara nova (tare_id diferente) recomeça a
contagem. reset() recomeça sem trocar a tara, para o ensaio de zero de 10 minutos.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Running:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    max_abs: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.max_abs = max(self.max_abs, abs(value))

    def as_dict(self, digits: int = 4) -> dict[str, float | None]:
        if self.n == 0:
            return {"mean_kg": None, "std_kg": None, "max_abs_kg": None}
        std = math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0
        return {"mean_kg": round(self.mean, digits), "std_kg": round(std, digits),
                "max_abs_kg": round(self.max_abs, digits)}


@dataclass
class _State:
    tare_id: str | None = None
    started_at: float | None = None
    last_at: float | None = None
    last_kg: float | None = None
    total: _Running = field(default_factory=_Running)
    without_gain: _Running = field(default_factory=_Running)
    sectors: dict[int, _Running] = field(default_factory=dict)


class ResidualMassTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _State()

    def update(self, tare_id: str, now: float, total_kg: float, without_gain_kg: float,
               sector_kg: dict[int, float]) -> None:
        """Um frame de maca vazia. sector_kg é α_s · L_s de cada setor ativo."""
        with self._lock:
            s = self._state
            if s.tare_id != tare_id:
                s = self._state = _State(tare_id=tare_id)
            if s.started_at is None:
                s.started_at = now
            s.last_at = now
            s.last_kg = total_kg
            s.total.add(total_kg)
            s.without_gain.add(without_gain_kg)
            for bid, value in sector_kg.items():
                s.sectors.setdefault(bid, _Running()).add(value)

    def reset(self) -> None:
        with self._lock:
            self._state = _State(tare_id=self._state.tare_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            s = self._state
            return {
                "tare_id": s.tare_id,
                "n_frames": s.total.n,
                "duration_s": round(s.last_at - s.started_at, 1) if s.started_at is not None else 0.0,
                "last_kg": None if s.last_kg is None else round(s.last_kg, 4),
                **s.total.as_dict(),
                "without_gain": s.without_gain.as_dict(),
                "sectors": {str(b): r.as_dict() for b, r in sorted(s.sectors.items())},
            }
