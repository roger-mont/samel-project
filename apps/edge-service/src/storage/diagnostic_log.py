"""Log de diagnóstico contínuo (parecer de 20/10/2026, itens 1 e 5).

Uma linha por frame lógico, no máximo uma a cada `interval_s`, com o peso e tudo o
que é preciso para explicar uma leitura baixa no mesmo instante: L1…L8, RAW corrigido
e células ativas por setor, idade e pacotes de cada setor, tara em vigor e flags.
Diferente do registro de experimento, roda 24/7, para pegar uma ocorrência de
20-25 kg que ninguém estava gravando.

Arquivos: data/diagnostics/diag_AAAAMMDD.csv (UTC), um por dia. Os dias anteriores
são compactados em .csv.gz e apagados depois de `retention_days`.
"""
from __future__ import annotations

import csv
import gzip
import logging
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SECTORS = tuple(range(1, 9))

FIELDS = [
    "timestamp",
    "reference_mass_kg",
    "reference_note",
    "state",
    "weight_status",
    "weight_kg",
    "estimated_mass_kg",
    "last_valid_weight_kg",
    "mass_without_gain_kg",
    *[f"L{s}" for s in SECTORS],
    *[f"raw_corr_{s}" for s in SECTORS],
    *[f"cells_{s}" for s in SECTORS],
    "n_active_cells",
    "n_out_of_calibration",
    "out_of_calibration_by_sector",
    *[f"age_ms_{s}" for s in SECTORS],
    *[f"pkt_s_{s}" for s in SECTORS],
    *[f"packets_{s}" for s in SECTORS],
    *[f"incomplete_{s}" for s in SECTORS],
    "packets_per_s",
    "read_errors",
    "tare_id",
    "tare_time",
    "baseline_sum",
    "STALE_SECTOR",
    "stale_sectors",
    "MISSING_CELLS",
    "OUT_OF_CALIBRATION",
    "MOVEMENT",
    "SIGNAL_STABLE",
    "WEIGHT_VALID",
]


def _ms(value: float | None) -> int | None:
    return None if value is None else int(round(value * 1000))


def build_row(live: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    """Linha do log a partir do estado ao vivo do worker."""
    q = live.get("measurement_quality") or {}
    local = live.get("sector_local_mass_kg") or {}
    raw_corr = live.get("sector_corrected_raw") or {}
    cells = live.get("sector_active_cells") or {}
    acq = live.get("acquisition") or {}
    acq_sectors = acq.get("sectors") or {}
    tare = live.get("tare") or {}
    ooc = live.get("out_of_calibration_by_sector") or {}
    row: dict[str, Any] = {
        "timestamp": live.get("timestamp"),
        "reference_mass_kg": (reference or {}).get("mass_kg"),
        "reference_note": (reference or {}).get("note"),
        "state": live.get("state_id"),
        "weight_status": live.get("weight_status"),
        "weight_kg": live.get("weight_kg"),
        "estimated_mass_kg": live.get("estimated_mass_kg"),
        "last_valid_weight_kg": live.get("last_valid_weight_kg"),
        "mass_without_gain_kg": live.get("mass_without_gain_kg"),
        "n_active_cells": live.get("n_active_cells"),
        "n_out_of_calibration": live.get("n_out_of_calibration"),
        "out_of_calibration_by_sector": ";".join(f"{k}:{v}" for k, v in sorted(ooc.items()) if v) or None,
        "packets_per_s": acq.get("packets_per_s"),
        "read_errors": acq.get("read_errors"),
        "tare_id": tare.get("tare_id"),
        "tare_time": tare.get("timestamp_tare"),
        "baseline_sum": tare.get("baseline_sum"),
        "STALE_SECTOR": q.get("STALE_SECTOR"),
        "stale_sectors": ";".join(str(s) for s in live.get("stale_sectors") or []) or None,
        "MISSING_CELLS": q.get("MISSING_CELLS"),
        "OUT_OF_CALIBRATION": q.get("OUT_OF_CALIBRATION"),
        "MOVEMENT": q.get("MOVEMENT"),
        "SIGNAL_STABLE": q.get("SIGNAL_STABLE"),
        "WEIGHT_VALID": q.get("WEIGHT_VALID"),
    }
    for s in SECTORS:
        k = str(s)
        info = acq_sectors.get(k) or {}
        row[f"L{s}"] = local.get(k)
        row[f"raw_corr_{s}"] = raw_corr.get(k)
        row[f"cells_{s}"] = cells.get(k)
        row[f"age_ms_{s}"] = _ms(info.get("age_s"))
        row[f"pkt_s_{s}"] = info.get("packets_per_s")
        row[f"packets_{s}"] = info.get("packets")
        row[f"incomplete_{s}"] = info.get("incomplete_sweeps")
    return row


class DiagnosticLog:
    """Grava o log em CSV diário; thread-safe e tolerante a falha de disco (só avisa)."""

    def __init__(self, directory: Path, interval_s: float = 0.5, retention_days: int = 7) -> None:
        self.directory = directory
        self.interval_s = float(interval_s)
        self.retention_days = int(retention_days)
        self._lock = threading.Lock()
        self._last_t: float | None = None
        self._day: str | None = None
        self._file: Any = None
        self._writer: csv.DictWriter | None = None
        self._reference: dict[str, Any] | None = None
        self._rows = 0
        self._error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0

    # --------------------------------------------------------------- referência

    def set_reference(self, mass_kg: float | None, note: str | None = None) -> dict[str, Any] | None:
        """Massa de referência sobre a maca (teste decisivo, item 5); None limpa."""
        with self._lock:
            self._reference = None if mass_kg is None else {
                "mass_kg": round(float(mass_kg), 3),
                "note": note or None,
                "since": datetime.now(timezone.utc).isoformat(),
            }
            return self._reference

    @property
    def reference(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._reference) if self._reference else None

    # ------------------------------------------------------------------ escrita

    def maybe_write(self, now: float, live: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._last_t is not None and now - self._last_t < self.interval_s:
            return
        self._last_t = now
        with self._lock:
            try:
                self._ensure_file(datetime.now(timezone.utc))
                assert self._writer is not None
                self._writer.writerow(build_row(live, self._reference))
                self._file.flush()
                self._rows += 1
                self._error = None
            except Exception as err:  # o log nunca derruba a aquisição
                if self._error != str(err):
                    logger.warning("Falha ao gravar o log de diagnóstico: %s", err)
                self._error = str(err)

    def _ensure_file(self, now_utc: datetime) -> None:
        day = now_utc.strftime("%Y%m%d")
        if day == self._day and self._file is not None:
            return
        self._close_file()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"diag_{day}.csv"
        new = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new:
            self._writer.writeheader()
        self._day = day
        self._housekeeping(now_utc)

    def _housekeeping(self, now_utc: datetime) -> None:
        """Compacta os dias anteriores e apaga os mais velhos que a retenção."""
        limit = (now_utc - timedelta(days=self.retention_days)).strftime("%Y%m%d")
        for path in sorted(self.directory.glob("diag_*.csv*")):
            day = path.name[5:13]
            if day < limit:
                path.unlink(missing_ok=True)
            elif path.suffix == ".csv" and day != self._day:
                gz = path.with_suffix(".csv.gz")
                with path.open("rb") as src, gzip.open(gz, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                path.unlink()

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self._writer = None

    def close(self) -> None:
        with self._lock:
            self._close_file()

    def status(self) -> dict[str, Any]:
        with self._lock:
            files = sorted(p.name for p in self.directory.glob("diag_*.csv*")) if self.directory.exists() else []
            return {
                "enabled": self.enabled,
                "directory": str(self.directory),
                "interval_s": self.interval_s,
                "retention_days": self.retention_days,
                "current_file": f"diag_{self._day}.csv" if self._day else None,
                "rows_written": self._rows,
                "files": files,
                "reference": dict(self._reference) if self._reference else None,
                "error": self._error,
            }
