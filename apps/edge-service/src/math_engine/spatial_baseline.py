"""Baseline espacial da manta vazia (tara por célula).

B_ij = média de s_ij(k) durante T_TARE segundos com a maca vazia.
x_ij(t) = max(0, s_ij(t) - B_ij). O RAW original nunca é alterado.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.settings import HID_COLS, HID_ROWS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TareRecord:
    """Metadados de uma tara (item 3 das orientações)."""

    tare_id: str
    timestamp_tare: str
    duration_s: float
    n_frames: int
    baseline_mean: float      # média de B_ij sobre as 2.048 células
    baseline_std: float       # média do desvio-padrão por célula durante a tara
    baseline_sum: float       # soma de B_ij
    configuration_id: str
    model_version: str
    automatic: bool


class SpatialBaseline:
    """Mantém a matriz B_ij, o desvio por célula e a coleta de novas taras."""

    def __init__(self, baseline_path: Path | None = None) -> None:
        self._path = baseline_path
        self._mean = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
        self._std = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
        self._record: TareRecord | None = None

        self._collecting = False
        self._collect_started: float = 0.0
        self._collect_duration: float = 0.0
        self._collect_automatic = False
        self._samples: list[np.ndarray] = []

        self._load()

    # ------------------------------------------------------------------ estado

    @property
    def has_baseline(self) -> bool:
        """TARE_OK: existe uma tara registrada."""
        return self._record is not None

    @property
    def record(self) -> TareRecord | None:
        return self._record

    @property
    def mean_matrix(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std_matrix(self) -> np.ndarray:
        return self._std.copy()

    @property
    def is_collecting(self) -> bool:
        return self._collecting

    @property
    def is_automatic(self) -> bool:
        """A coleta em curso é de uma tara automática."""
        return self._collecting and self._collect_automatic

    def is_baseline_ok(self, max_mean_std_raw: float) -> bool:
        """BASELINE_OK: a tara existe e o ruído médio por célula ficou dentro do limite."""
        return self._record is not None and self._record.baseline_std <= max_mean_std_raw

    # ---------------------------------------------------------------- correção

    def correct(self, raw_matrix: np.ndarray) -> np.ndarray:
        """x_ij(t) = max(0, s_ij(t) - B_ij)."""
        return np.maximum(0.0, raw_matrix.astype(np.float64) - self._mean)

    # ------------------------------------------------------------------- tara

    def start_tare(self, now: float, duration_s: float, automatic: bool = False) -> None:
        """Começa a acumular frames para uma nova tara."""
        self._collecting = True
        self._collect_started = now
        self._collect_duration = max(0.1, float(duration_s))
        self._collect_automatic = automatic
        self._samples = []
        logger.info("Tara %s iniciada (%.1f s)", "automática" if automatic else "manual", duration_s)

    def cancel_tare(self, reason: str) -> None:
        if self._collecting:
            logger.warning("Tara cancelada: %s", reason)
        self._collecting = False
        self._samples = []

    def feed(
        self,
        raw_matrix: np.ndarray,
        now: float,
        configuration_id: str,
        model_version: str,
    ) -> TareRecord | None:
        """Acumula um frame; ao fim de T_TARE grava a nova tara e devolve o registro."""
        if not self._collecting:
            return None
        if raw_matrix.shape != (HID_ROWS, HID_COLS):
            return None

        self._samples.append(raw_matrix.astype(np.float64))
        if now - self._collect_started < self._collect_duration:
            return None

        stack = np.stack(self._samples)
        self._mean = stack.mean(axis=0)
        self._std = stack.std(axis=0)
        self._record = TareRecord(
            tare_id=str(uuid.uuid4()),
            timestamp_tare=datetime.now(timezone.utc).isoformat(),
            duration_s=round(now - self._collect_started, 3),
            n_frames=len(self._samples),
            baseline_mean=float(self._mean.mean()),
            baseline_std=float(self._std.mean()),
            baseline_sum=float(self._mean.sum()),
            configuration_id=configuration_id,
            model_version=model_version,
            automatic=self._collect_automatic,
        )
        self._collecting = False
        self._samples = []
        self._persist()
        logger.info(
            "Tara %s gravada: %d frames, soma=%.1f, desvio médio=%.2f RAW",
            self._record.tare_id,
            self._record.n_frames,
            self._record.baseline_sum,
            self._record.baseline_std,
        )
        return self._record

    # ------------------------------------------------------------ persistência

    def _persist(self) -> None:
        if self._path is None or self._record is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "record": asdict(self._record),
                "baseline_raw": np.round(self._mean, 4).tolist(),
                "baseline_std": np.round(self._std, 4).tolist(),
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self._path)
        except Exception as err:
            logger.error("Falha ao salvar baseline em %s: %s", self._path, err)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            mean = np.array(data["baseline_raw"], dtype=np.float64)
            std = np.array(data["baseline_std"], dtype=np.float64)
            if mean.shape != (HID_ROWS, HID_COLS) or std.shape != (HID_ROWS, HID_COLS):
                raise ValueError(f"dimensões inválidas: {mean.shape}")
            self._mean = mean
            self._std = std
            self._record = TareRecord(**data["record"])
            logger.info("Baseline %s carregado de %s", self._record.tare_id, self._path)
        except Exception as err:
            logger.warning("Baseline em %s ignorado: %s", self._path, err)
