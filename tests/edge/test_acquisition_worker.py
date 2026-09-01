"""Testes automatizados para o AcquisitionWorker (aquisição 24/7 no Edge Service)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pytest

# Garante import do src do edge-service
_EDGE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "apps", "edge-service", "src")
)
if _EDGE_SRC not in sys.path:
    sys.path.insert(0, _EDGE_SRC)

from database.db_local import LocalDatabase
from hardware.serial_reader import BaseFrameReader
from workers.acquisition_worker import AcquisitionWorker


class DummyReader(BaseFrameReader):
    """Leitor de teste com retorno controlado."""

    def __init__(self, value: float = 50.0):
        self._value = value
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self) -> np.ndarray:
        matrix = np.full((32, 64), self._value, dtype=np.float64)
        time.sleep(0.01)
        return matrix

    def close(self) -> None:
        self._connected = False


def test_acquisition_worker_lifecycle_and_broadcast() -> None:
    """Valida inicialização, leitura e broadcast do AcquisitionWorker."""
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "test_acq.db")
    db = LocalDatabase(db_path=db_file)

    broadcasted_frames: list[dict[str, Any]] = []

    def mock_broadcast(data: dict[str, Any]) -> None:
        broadcasted_frames.append(data)

    reader = DummyReader(value=40.0)
    worker = AcquisitionWorker(
        db=db,
        maca_id="MACA-TEST-01",
        reader=reader,
        broadcast_callback=mock_broadcast,
    )

    try:
        worker.start()
        time.sleep(0.2)  # Aguarda alguns ciclos de leitura

        snapshot = worker.get_live_snapshot()
        assert snapshot["status"] == "conectado"
        assert len(snapshot["heatmap"]) == 32
        assert len(broadcasted_frames) > 0

        # Testa ajuste de parâmetros em runtime
        worker.update_param("posture_timeout_seconds", 1800)
        assert worker.params.posture_timeout_seconds == 1800

        # Testa disparo de tara
        worker.trigger_tare()
        assert worker._tare_pending is True

    finally:
        worker.stop()
        shutil.rmtree(temp_dir, ignore_errors=True)
