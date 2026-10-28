"""Drivers de aquisição de frames de pressão (USB HID real, Serial UART e Gerador Simulado)."""
from __future__ import annotations

import logging
import threading
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
from hardware.acquisition_stats import AcquisitionSample, AcquisitionStats
from storage.calibration_store import BLOCK_REGIONS

logger = logging.getLogger(__name__)

# Setor de cada bloco 16×16 da matriz: _SECTOR_GRID[linha // 16][coluna // 16]
_SECTOR_GRID: list[list[int]] = [[0] * (HID_COLS // 16) for _ in range(HID_ROWS // 16)]
for _bid, (_rows, _cols) in BLOCK_REGIONS.items():
    _SECTOR_GRID[_rows.start // 16][_cols.start // 16] = _bid


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

    def acquisition_sample(self, now: float | None = None) -> AcquisitionSample | None:
        """Telemetria por setor (pacotes, idade, perdas). None quando o leitor não a fornece."""
        return None


class FakeSerialReader(BaseFrameReader):
    """Gerador de dados sintéticos para desenvolvimento e testes sem hardware."""

    def __init__(self, change_interval: float = 30.0):
        self._change_interval = change_interval
        self._last_change = time.monotonic()
        self._center = self._random_center()
        self._connected = True
        self.stats = AcquisitionStats("simulado")

    def _random_center(self) -> tuple[float, float]:
        rng = np.random.default_rng()
        return (
            rng.uniform(4.0, HID_ROWS - 4.0),
            rng.uniform(8.0, HID_COLS - 8.0),
        )

    def is_connected(self) -> bool:
        return self._connected

    def acquisition_sample(self, now: float | None = None) -> AcquisitionSample:
        return self.stats.sample(now)

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
        self.stats.on_full_frame(time.monotonic())
        return matrix

    def close(self) -> None:
        self._connected = False


class HidFrameReader(BaseFrameReader):
    """Leitura de alta fidelidade via USB HID (Manta WangYing VID=6860, PID=6733).

    A leitura USB roda numa thread própria, separada do processamento (parecer de
    20/10, item 3): o consumidor pesado não bloqueia a aquisição e o buffer HID do
    sistema não transborda enquanto o worker calcula. read_frame() só copia o mapa
    persistente no ritmo do consumidor. Cada pacote alimenta a telemetria por setor
    (self.stats), usada na flag STALE_SECTOR.
    """

    _RENDER_INTERVAL_S: float = 0.025  # ritmo do consumidor, não a taxa real da manta
    _READ_TIMEOUT_MS: int = 50
    _RECONNECT_WAIT_S: float = 0.5

    def __init__(self, vid: int = HID_VID, pid: int = HID_PID) -> None:
        if hid is None:
            raise ImportError("hidapi não instalado: pip install hidapi")
        self._vid = vid
        self._pid = pid
        self._device: hid.device | None = None
        self._lock = threading.Lock()
        self._persistent_map = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
        self._max_seen: int = 0
        self._last_frame_t = time.monotonic()
        self.stats = AcquisitionStats("usb_hid")
        self._connect()
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, name="HidRxThread", daemon=True)
        self._rx_thread.start()

    def _connect(self) -> None:
        try:
            device = hid.device()
            device.open(self._vid, self._pid)
            self._device = device
            logger.info("USB HID conectado com sucesso: VID=0x%04X PID=0x%04X", self._vid, self._pid)
        except Exception as err:
            logger.error("Falha ao abrir USB HID VID=0x%04X PID=0x%04X: %s", self._vid, self._pid, err)
            self._device = None

    def is_connected(self) -> bool:
        return self._device is not None

    def acquisition_sample(self, now: float | None = None) -> AcquisitionSample:
        return self.stats.sample(now)

    def read_frame(self) -> np.ndarray:
        remaining = self._RENDER_INTERVAL_S - (time.monotonic() - self._last_frame_t)
        if remaining > 0:
            time.sleep(remaining)
        self._last_frame_t = time.monotonic()
        with self._lock:
            return self._persistent_map.copy()

    def _rx_loop(self) -> None:
        """Lê pacotes sem parar e aplica no mapa persistente; reconecta se a manta cair."""
        while self._running:
            device = self._device
            if device is None:
                time.sleep(self._RECONNECT_WAIT_S)
                if not self._running:
                    break
                self._connect()
                if self._device is not None:
                    self.stats.on_reconnect()
                continue
            try:
                raw = device.read(HID_PACKET_SIZE, self._READ_TIMEOUT_MS)
            except Exception as err:
                logger.warning("Erro de leitura USB HID: %s", err)
                self.stats.on_read_error()
                self._device = None
                continue
            if raw:
                self._handle_packet(bytes(raw), time.monotonic())

    def _handle_packet(self, data: bytes, now: float) -> None:
        with self._lock:
            sector, local_cells, rows, cols = self._parse_hid_packet(data, self._persistent_map)
        if sector is None:
            self.stats.on_ignored_packet()
        else:
            self.stats.on_packet(sector, local_cells, rows, cols, now)

    def _parse_hid_packet(
        self, data: bytes, matrix: np.ndarray,
    ) -> tuple[int | None, list[int], list[int], list[int]]:
        """Aplica o pacote no mapa e devolve o setor e as células recebidas (índice local 0-255 e posição)."""
        block_id = data[0]
        local_cells: list[int] = []
        rows: list[int] = []
        cols: list[int] = []
        if block_id == 0:
            return None, local_cells, rows, cols

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
                if x_local <= 16 and y_local <= 16:
                    local_cells.append((x_local - 1) * 16 + (y_local - 1))
                    rows.append(x)
                    cols.append(y)

        sector = _SECTOR_GRID[rows[0] // 16][cols[0] // 16] if rows else None
        return sector, local_cells, rows, cols

    def close(self) -> None:
        self._running = False
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self._device is not None:
            self._device.close()
            self._device = None
            logger.info("USB HID desconectado")
