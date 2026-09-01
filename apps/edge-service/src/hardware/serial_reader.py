"""Drivers de aquisição de frames de pressão (USB HID real, Serial UART e Gerador Simulado)."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np

try:
    import serial
except ImportError:
    serial = None  # type: ignore[assignment]

try:
    import hid
except ImportError:
    hid = None  # type: ignore[assignment]

from core.settings import HID_VID, HID_PID, HID_ROWS, HID_COLS, HID_PACKET_SIZE

logger = logging.getLogger(__name__)


class BaseFrameReader(ABC):
    """Interface abstrata para leitores de matrizes da manta sensora."""

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Retorna matriz 32x64 com intensidades de pressão."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Verifica se o hardware está conectado."""

    @abstractmethod
    def close(self) -> None:
        """Libera recursos do driver."""


class FakeSerialReader(BaseFrameReader):
    """Gerador de dados sintéticos para desenvolvimento e testes sem hardware."""

    def __init__(self, change_interval: float = 30.0):
        self._change_interval = change_interval
        self._last_change = time.monotonic()
        self._center = self._random_center()
        self._connected = True

    def _random_center(self) -> tuple[float, float]:
        rng = np.random.default_rng()
        return (
            rng.uniform(4.0, HID_ROWS - 4.0),
            rng.uniform(8.0, HID_COLS - 8.0),
        )

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self) -> np.ndarray:
        now = time.monotonic()
        if now - self._last_change > self._change_interval:
            self._center = self._random_center()
            self._last_change = now

        rng = np.random.default_rng()
        rows_idx, cols_idx = np.meshgrid(
            np.arange(HID_ROWS, dtype=np.float64),
            np.arange(HID_COLS, dtype=np.float64),
            indexing="ij",
        )

        dist_sq = (rows_idx - self._center[0]) ** 2 + (cols_idx - self._center[1]) ** 2
        sigma = max(HID_ROWS, HID_COLS) / 3.0
        gaussian = 180.0 * np.exp(-dist_sq / (2.0 * sigma**2))
        noise = rng.normal(0, 3, size=(HID_ROWS, HID_COLS))
        matrix = np.clip(gaussian + noise, 0, 255).astype(np.float64)

        time.sleep(0.025)  # 40 Hz
        return matrix

    def close(self) -> None:
        self._connected = False


class HidFrameReader(BaseFrameReader):
    """Leitura de alta fidelidade via USB HID (Manta WangYing VID=6860, PID=6733)."""

    _RENDER_INTERVAL_S: float = 0.025  # 40 Hz estável

    def __init__(self, vid: int = HID_VID, pid: int = HID_PID) -> None:
        if hid is None:
            raise ImportError("hidapi não instalado: pip install hidapi")
        self._vid = vid
        self._pid = pid
        self._device: hid.device | None = None
        self._persistent_map = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
        self._max_seen: int = 0
        self._connect()

    def _connect(self) -> None:
        try:
            device = hid.device()
            device.open(self._vid, self._pid)
            device.set_nonblocking(True)
            self._device = device
            logger.info("USB HID conectado com sucesso: VID=0x%04X PID=0x%04X", self._vid, self._pid)
        except Exception as err:
            logger.error("Falha ao abrir USB HID VID=0x%04X PID=0x%04X: %s", self._vid, self._pid, err)
            self._device = None

    def is_connected(self) -> bool:
        return self._device is not None

    def read_frame(self) -> np.ndarray:
        frame_start = time.monotonic()

        if not self.is_connected():
            self._connect()

        if self.is_connected():
            self._drain_hid_buffer()

        elapsed = time.monotonic() - frame_start
        remaining = self._RENDER_INTERVAL_S - elapsed
        if remaining > 0:
            time.sleep(remaining)

        return self._persistent_map.copy()

    def _drain_hid_buffer(self) -> None:
        try:
            while True:
                raw = self._device.read(HID_PACKET_SIZE)  # type: ignore[union-attr]
                if not raw:
                    break
                self._parse_hid_packet(bytes(raw), self._persistent_map)
        except Exception as err:
            logger.warning("Erro de leitura USB HID: %s", err)
            self._device = None

    def _parse_hid_packet(self, data: bytes, matrix: np.ndarray) -> None:
        block_id = data[0]
        if block_id == 0:
            return

        for i in range(1, len(data) - 1, 3):
            if i + 2 >= len(data):
                break

            x_local = data[i]
            y_local = data[i + 1]
            pressure = data[i + 2]

            if pressure > self._max_seen:
                self._max_seen = pressure

            if x_local == 0 or y_local == 0:
                break

            eff_id = ((block_id + 3) % 8) + 1
            if eff_id < 5:
                x = 16 * ((9 - eff_id) // 5) + (16 - x_local)
                y = 16 * (eff_id - 1) + (16 - y_local)
            elif eff_id < 9:
                x = 16 * ((9 - eff_id) // 5) + (x_local - 1)
                y = 16 * (8 - eff_id) + (y_local - 1)
            else:
                continue

            if 0 <= x < HID_ROWS and 0 <= y < HID_COLS:
                matrix[x, y] = float(pressure)

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
            logger.info("USB HID desconectado")
