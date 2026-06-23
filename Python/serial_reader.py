"""Leitura e parsing de frames da porta Serial COM."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np

try:
    import serial
except ImportError:
    serial = None  # type: ignore[assignment]

from config import GRID_ROWS, GRID_COLS

logger = logging.getLogger(__name__)


class BaseFrameReader(ABC):
    """Interface para leitores de frames da matriz."""

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """Retorna matriz GRID_ROWS x GRID_COLS com valores ADC."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Verifica se a fonte de dados está ativa."""

    @abstractmethod
    def close(self) -> None:
        """Libera recursos."""


class SerialFrameReader(BaseFrameReader):
    """Leitura real via porta Serial COM.

    Protocolo ASCII (CSV por linha):
      Idle:  "A00\\r\\n"
      Dados: "segment_id,adc_0,adc_1,...,adc_N\\r\\n"

    Valores ADC são mapeados row-major na matriz GRID_ROWS x GRID_COLS.
    O primeiro valor de cada linha (segment_id) é descartado.
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        if serial is None:
            raise ImportError("pyserial nao instalado: pip install pyserial")
        self._port_name = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial: serial.Serial | None = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._serial = serial.Serial(
                port=self._port_name,
                baudrate=self._baudrate,
                timeout=self._timeout,
            )
            logger.info("serial conectada: %s @ %d", self._port_name, self._baudrate)
        except serial.SerialException as err:
            logger.error("falha ao conectar %s: %s", self._port_name, err)
            self._serial = None

    def is_connected(self) -> bool:
        if self._serial is None:
            return False
        return self._serial.is_open

    def read_frame(self) -> np.ndarray:
        matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)

        if not self.is_connected():
            self._connect()
            if not self.is_connected():
                return matrix

        try:
            line = self._read_line()
            if line is None:
                return matrix

            stripped = line.strip()

            if not stripped or stripped.upper() == "A00":
                return matrix

            values = self._parse_csv(stripped)
            if values is None:
                return matrix

            logger.info("serial csv (%d valores): %s", len(values), stripped)
            matrix = self._fill_matrix(values)
        except serial.SerialException as err:
            logger.warning("erro de leitura serial: %s", err)
            self._serial = None
        except Exception as err:
            logger.error("erro inesperado na leitura: %s", err)

        return matrix

    def _read_line(self) -> str | None:
        """Le uma linha completa da serial (ate \\n ou timeout)."""
        if self._serial is None:
            return None
        raw_line = self._serial.readline()
        if not raw_line:
            return None
        return raw_line.decode("ascii", errors="ignore")

    def _parse_csv(self, line: str) -> list[int] | None:
        """Converte linha CSV em lista de inteiros."""
        parts = line.split(",")
        try:
            return [int(p.strip()) for p in parts]
        except ValueError:
            logger.warning("csv invalido: %s", line)
            return None

    def _fill_matrix(self, values: list[int]) -> np.ndarray:
        """Preenche matriz row-major, descartando o primeiro valor (segment_id)."""
        matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
        sensor_data = values[1:]
        total_cells = GRID_ROWS * GRID_COLS

        for i, val in enumerate(sensor_data[:total_cells]):
            row = i // GRID_COLS
            col = i % GRID_COLS
            matrix[row, col] = float(val)

        return matrix

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("serial fechada: %s", self._port_name)


class FakeSerialReader(BaseFrameReader):
    """Gerador de dados sintéticos para desenvolvimento sem hardware.

    Simula pressão gaussiana com centro que muda a cada ~30s,
    permitindo testar o alerta de postura estática.
    """

    def __init__(self, change_interval: float = 30.0):
        self._change_interval = change_interval
        self._last_change = time.monotonic()
        self._center = self._random_center()
        self._connected = True

    def _random_center(self) -> tuple[float, float]:
        rng = np.random.default_rng()
        return (
            rng.uniform(0.5, GRID_ROWS - 0.5),
            rng.uniform(0.5, GRID_COLS - 0.5),
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
            np.arange(GRID_ROWS, dtype=np.float64),
            np.arange(GRID_COLS, dtype=np.float64),
            indexing="ij",
        )

        dist_sq = (rows_idx - self._center[0]) ** 2 + (cols_idx - self._center[1]) ** 2
        sigma = max(GRID_ROWS, GRID_COLS) / 2.0
        gaussian = 200.0 * np.exp(-dist_sq / (2.0 * sigma**2))
        noise = rng.normal(0, 5, size=(GRID_ROWS, GRID_COLS))
        matrix = np.clip(gaussian + noise, 0, 255).astype(np.float64)

        time.sleep(0.02)  # simula ~50Hz
        return matrix

    def close(self) -> None:
        self._connected = False
