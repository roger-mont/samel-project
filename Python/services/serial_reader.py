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

try:
    import hid
except ImportError:
    hid = None  # type: ignore[assignment]

from config.settings import GRID_ROWS, GRID_COLS, HID_VID, HID_PID, HID_ROWS, HID_COLS, HID_PACKET_SIZE

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


class HidFrameReader(BaseFrameReader):
    """Leitura via USB HID — protocolo WangYing (VID=6860, PID=6733).

    Estrutura do pacote de 64 bytes:
      [0]      block_id — identifica o quadrante do colchão (1-8).
                          0 = pacote vazio, aborta leitura.
      [1..63]  trincas (x_local, y_local, pressão):
               x_local  : 1-16 (coluna dentro do quadrante)
               y_local  : 1-16 (linha dentro do quadrante)
               pressão  : 0-255 (intensidade FSR)
               Trinca com x=0 ou y=0 marca fim dos dados úteis.

    Algoritmo de mapeamento (replicado do C# ButtonShowDeal):
      Blocos 1-4  → x = 16 * ((9 - block_id) // 5) + (16 - x_local)
                    y = 16 * (block_id - 1)         + (16 - y_local)
      Blocos 5-8  → x = 16 * ((9 - block_id) // 5) + (x_local - 1)
                    y = 16 * (8 - block_id)         + (y_local - 1)

    Anti-flickering — por que o C# não flicka:
      O software original mantém MapV[64,64] como matriz PERSISTENTE e ACUMULADORA.
      Cada pacote HID atualiza APENAS as células do seu bloco específico — nunca
      zera a matriz inteira. Um timer de 25ms (disTimerA) amostra a matriz completa
      e renderiza. Aqui replicamos isso com _persistent_map + drain completo do
      buffer HID a cada ciclo de 25ms.

    Matriz de saída: HID_ROWS × HID_COLS (32 × 64).
    """

    _RENDER_INTERVAL_S: float = 0.025  # 25ms = 40Hz (igual ao disTimerA do C#)

    def __init__(self, vid: int = HID_VID, pid: int = HID_PID) -> None:
        if hid is None:
            raise ImportError("hidapi nao instalado: pip install hidapi")
        self._vid = vid
        self._pid = pid
        self._device: hid.device | None = None
        # Matriz global persistente — nunca zerada entre pacotes (espelha MapV do C#)
        self._persistent_map = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
        self._max_seen: int = 0  # maior valor de pressão já recebido
        self._connect()

    def _connect(self) -> None:
        try:
            device = hid.device()
            device.open(self._vid, self._pid)
            device.set_nonblocking(True)
            self._device = device
            logger.info("HID conectado: VID=0x%04X PID=0x%04X", self._vid, self._pid)
        except OSError as err:
            logger.error("falha ao conectar HID VID=0x%04X PID=0x%04X: %s", self._vid, self._pid, err)
            self._device = None

    def is_connected(self) -> bool:
        return self._device is not None

    def read_frame(self) -> np.ndarray:
        """Drena todos os pacotes HID disponíveis, acumula em _persistent_map e
        throttla em 40Hz — espelha o par HidMsgDeal + disTimerA do C#."""
        frame_start = time.monotonic()

        if not self.is_connected():
            self._connect()

        if self.is_connected():
            self._drain_hid_buffer()

        # Throttle: aguarda o restante dos 25ms para manter 40Hz estável
        elapsed = time.monotonic() - frame_start
        remaining = self._RENDER_INTERVAL_S - elapsed
        if remaining > 0:
            time.sleep(remaining)

        return self._persistent_map.copy()

    def _drain_hid_buffer(self) -> None:
        """Lê e acumula TODOS os pacotes disponíveis no buffer HID agora."""
        try:
            while True:
                raw = self._device.read(HID_PACKET_SIZE)  # type: ignore[union-attr]
                if not raw:
                    break
                self._parse_hid_packet(bytes(raw), self._persistent_map)
        except OSError as err:
            logger.warning("erro de leitura HID: %s", err)
            self._device = None

    def _parse_hid_packet(self, data: bytes, matrix: np.ndarray) -> None:
        """Replica exata do algoritmo ButtonShowDeal do C# original.

        Cada chamada atualiza APENAS as células do block_id recebido.
        Células de outros blocos não são tocadas — acumulação seletiva.
        """
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
            logger.info("HID desconectado")
