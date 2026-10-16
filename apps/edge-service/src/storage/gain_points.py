"""Pontos do ajuste de ganho registrados na página /calibracao.

Cada ponto guarda a massa real informada e a média do RAW corrigido da manta inteira
(x_ij = max(0, s_ij − B_ij)) no trecho estável em que foi registrado. As massas
locais L_s são recalculadas a cada uso com as curvas em vigor, então os pontos
continuam valendo depois de regerar as curvas.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from core.settings import HID_COLS, HID_ROWS

logger = logging.getLogger(__name__)


class GainPointStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._points: list[dict[str, Any]] = []
        self._load()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._points]

    def corrected(self, point: dict[str, Any]) -> np.ndarray:
        return np.asarray(point["corrected"], dtype=np.float64)

    def add(self, known_mass_kg: float, corrected: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
        if corrected.shape != (HID_ROWS, HID_COLS):
            raise ValueError(f"matriz com dimensões inválidas: {corrected.shape}")
        point = {
            "id": uuid.uuid4().hex[:12],
            "known_mass_kg": float(known_mass_kg),
            **meta,
            "corrected": np.round(corrected, 3).tolist(),
        }
        with self._lock:
            self._points.append(point)
            self._persist()
        return dict(point)

    def delete(self, point_id: str) -> bool:
        with self._lock:
            kept = [p for p in self._points if p["id"] != point_id]
            if len(kept) == len(self._points):
                return False
            self._points = kept
            self._persist()
            return True

    def clear(self) -> int:
        with self._lock:
            n = len(self._points)
            self._points = []
            self._persist()
            return n

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"points": self._points}), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._points = [p for p in data.get("points", [])
                            if np.asarray(p.get("corrected")).shape == (HID_ROWS, HID_COLS)]
        except Exception as err:
            logger.warning("Pontos de ganho em %s ignorados: %s", self._path, err)
