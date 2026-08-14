"""Constantes e parâmetros de calibração editáveis em runtime."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, fields


SOFTWARE_VERSION: str = "1.0.0"

GRID_ROWS: int = 16   # espelha HID_ROWS — metade do grid estava vazia
GRID_COLS: int = 16   # espelha HID_COLS

# Constantes do protocolo USB HID (WangYing / colchão inteligente)
# Dimensões reais usadas pelo software original: 32 linhas × 64 colunas
HID_VID: int = 6860   # Vendor ID  (0x1ACC)
HID_PID: int = 6733   # Product ID (0x1A4D)
HID_ROWS: int = 32
HID_COLS: int = 64
HID_PACKET_SIZE: int = 64  # bytes por report HID

# Dimensões físicas reais da manta (para integração 2D - Modelo B)
MANTA_WIDTH_M: float = 0.90    # largura da maca (m)
MANTA_HEIGHT_M: float = 1.80   # comprimento da maca (m)
PIXEL_AREA_M2: float = (MANTA_WIDTH_M / HID_COLS) * (MANTA_HEIGHT_M / HID_ROWS)

API_PORT: int = 8000
API_HOST: str = "0.0.0.0"
TARE_SAMPLE_COUNT: int = 5
TARE_FILE: str = "tare.json"
WS_PUSH_INTERVAL_SECONDS: int = 10

# Critério de estabilidade (Metodologia §13)
STABILITY_EPSILON_KG: float = 0.3       # |Δm| máximo entre frames consecutivos
STABILITY_TMIN_S: float = 10.0          # janela mínima para travar o peso
STABILITY_VARIANCE_KG2: float = 0.25    # variância máxima da janela
STABILITY_DRIFT_KG_S: float = 0.15      # drift máximo em kg/s
STABILITY_WINDOW_SIZE: int = 20         # frames na janela de variância/drift


@dataclass
class CalibrationParams:
    """Parâmetros de comportamento — mutáveis via UI em tempo real.

    Os valores brutos 0-255 do HID são usados diretamente como intensidade
    de pressão. Não há conversão de tensão ou resistência.
    """

    deadzone_threshold: float = 2.0    # ignora sensores com valor <= este limiar
    ema_alpha: float = 0.3              # suavização exponencial do peso total
    posture_tolerance: float = 0.15
    posture_timeout_seconds: int = 60
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def update(self, key: str, value: float) -> None:
        """Atualiza um campo pelo nome — thread-safe."""
        if not hasattr(self, key) or key.startswith("_"):
            raise ValueError(f"parametro desconhecido: {key}")
        with self._lock:
            expected_type = type(getattr(self, key))
            setattr(self, key, expected_type(value))

    def snapshot(self) -> dict:
        """Retorna cópia imutável dos valores atuais."""
        with self._lock:
            return {
                f.name: getattr(self, f.name)
                for f in fields(self)
                if not f.name.startswith("_")
            }
