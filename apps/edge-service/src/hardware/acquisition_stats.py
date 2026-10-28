"""Telemetria da aquisição USB por setor (parecer de 20/10/2026, itens 1 a 4).

O leitor mantém um mapa persistente: cada célula guarda o último valor recebido
até chegar um pacote novo do seu bloco. Se pacotes se perdem, um setor pode ficar
congelado e continuar parecendo válido. Aqui se conta, por setor:

- pacotes recebidos e a taxa por segundo;
- atualizações completas: as 256 células do setor recebidas ao menos uma vez
  desde a última atualização completa. A idade do setor é o tempo desde a última;
- varreduras incompletas: uma célula chegou de novo antes de o setor ser coberto
  inteiro. O protocolo não traz número de sequência, então isso é o indício de
  pacote perdido (ou de firmware que só manda as células com pressão);
- o intervalo real entre atualizações completas (p50, p95, máximo), para definir
  o limite de STALE_SECTOR a partir do hardware, e não dos "40 Hz".

Também guarda a hora da última atualização de cada célula, para separar memória
física do sensor (valor atualizado, mas acima da tara) de ausência de atualização.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.settings import HID_COLS, HID_ROWS

SECTOR_CELLS = 256
SECTORS = tuple(range(1, 9))
_RATE_WINDOW_S = 1.0
_INTERVALS_KEPT = 400


@dataclass
class AcquisitionSample:
    """Retrato da aquisição num instante: resumo por setor e idade de cada célula (s)."""

    summary: dict[str, Any]
    cell_age_s: np.ndarray

    def sector_age(self, sector: int) -> float | None:
        info = self.summary["sectors"].get(str(sector))
        return None if info is None else info["age_s"]


class _Sector:
    __slots__ = ("packets", "cells", "complete", "incomplete", "missed", "last_packet_t",
                 "last_complete_t", "seen", "n_seen", "intervals")

    def __init__(self) -> None:
        self.packets = 0
        self.cells = 0
        self.complete = 0
        self.incomplete = 0
        self.missed = 0
        self.last_packet_t: float | None = None
        self.last_complete_t: float | None = None
        self.seen = bytearray(SECTOR_CELLS)
        self.n_seen = 0
        self.intervals: deque[float] = deque(maxlen=_INTERVALS_KEPT)

    def _complete_at(self, now: float) -> None:
        if self.last_complete_t is not None:
            self.intervals.append(now - self.last_complete_t)
        self.last_complete_t = now
        self.complete += 1
        self.seen = bytearray(SECTOR_CELLS)
        self.n_seen = 0

    def add(self, local_cells: list[int], now: float) -> None:
        self.packets += 1
        self.last_packet_t = now
        for idx in local_cells:
            self.cells += 1
            if self.seen[idx]:
                # Célula repetida antes de cobrir o setor: a varredura anterior ficou incompleta
                self.incomplete += 1
                self.missed += SECTOR_CELLS - self.n_seen
                self.seen = bytearray(SECTOR_CELLS)
                self.n_seen = 0
            self.seen[idx] = 1
            self.n_seen += 1
            if self.n_seen == SECTOR_CELLS:
                self._complete_at(now)


def _percentiles_ms(values: deque[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "max": round(float(np.max(arr)), 1),
    }


def _age(now: float, t: float | None) -> float | None:
    return None if t is None else round(max(0.0, now - t), 3)


class AcquisitionStats:
    """Contadores thread-safe alimentados pela thread de leitura e lidos pelo worker/API."""

    def __init__(self, source: str) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._sectors = {s: _Sector() for s in SECTORS}
        self._packets = 0
        self._ignored = 0
        self._read_errors = 0
        self._reconnects = 0
        self._cell_t = np.full((HID_ROWS, HID_COLS), np.nan, dtype=np.float64)
        self._history: deque[tuple[float, int, dict[int, tuple[int, int]]]] = deque()

    # ------------------------------------------------------------ alimentação

    def on_packet(self, sector: int, local_cells: list[int], rows: list[int], cols: list[int], now: float) -> None:
        """Pacote de um setor: índices locais (0-255) e posições na matriz das células recebidas."""
        with self._lock:
            self._packets += 1
            sec = self._sectors.get(sector)
            if sec is None:
                self._ignored += 1
                return
            sec.add(local_cells, now)
            if rows:
                self._cell_t[rows, cols] = now

    def on_full_frame(self, now: float) -> None:
        """Leitor que entrega a manta inteira de uma vez (simulador): todos os setores completos."""
        with self._lock:
            for sec in self._sectors.values():
                sec.packets += 1
                sec.cells += SECTOR_CELLS
                sec.last_packet_t = now
                sec._complete_at(now)
            self._packets += len(self._sectors)
            self._cell_t[:, :] = now

    def on_ignored_packet(self) -> None:
        with self._lock:
            self._packets += 1
            self._ignored += 1

    def on_read_error(self) -> None:
        with self._lock:
            self._read_errors += 1

    def on_reconnect(self) -> None:
        with self._lock:
            self._reconnects += 1

    # ------------------------------------------------------------------ leitura

    def _rates(self, now: float) -> tuple[float | None, dict[int, tuple[float, float]]]:
        """Pacotes/s e atualizações completas/s por setor no último segundo (a partir do histórico)."""
        counts = {s: (sec.packets, sec.complete) for s, sec in self._sectors.items()}
        if not self._history or now - self._history[-1][0] >= 0.1:
            self._history.append((now, self._packets, counts))
        while len(self._history) > 2 and now - self._history[1][0] >= _RATE_WINDOW_S:
            self._history.popleft()
        t0, total0, counts0 = self._history[0]
        dt = now - t0
        if dt < 0.05:
            return None, {}
        per = {s: ((counts[s][0] - counts0[s][0]) / dt, (counts[s][1] - counts0[s][1]) / dt) for s in counts}
        return (self._packets - total0) / dt, per

    def sample(self, now: float | None = None) -> AcquisitionSample:
        now = time.monotonic() if now is None else now
        with self._lock:
            total_rate, rates = self._rates(now)
            sectors: dict[str, Any] = {}
            for s, sec in self._sectors.items():
                pkt_rate, upd_rate = rates.get(s, (None, None))
                sectors[str(s)] = {
                    "packets": sec.packets,
                    "packets_per_s": None if pkt_rate is None else round(pkt_rate, 1),
                    "complete_updates": sec.complete,
                    "updates_per_s": None if upd_rate is None else round(upd_rate, 2),
                    "incomplete_sweeps": sec.incomplete,
                    "cells_missed": sec.missed,
                    "coverage_pct": round(100.0 * sec.n_seen / SECTOR_CELLS, 1),
                    "last_packet_age_s": _age(now, sec.last_packet_t),
                    "age_s": _age(now, sec.last_complete_t),
                    "update_interval_ms": _percentiles_ms(sec.intervals),
                }
            summary = {
                "source": self.source,
                "uptime_s": round(now - self._started, 1),
                "packets_total": self._packets,
                "packets_per_s": None if total_rate is None else round(total_rate, 1),
                "packets_ignored": self._ignored,
                "read_errors": self._read_errors,
                "reconnects": self._reconnects,
                "sectors": sectors,
            }
            ages = np.where(np.isnan(self._cell_t), np.inf, now - self._cell_t)
        return AcquisitionSample(summary=summary, cell_age_s=ages)
