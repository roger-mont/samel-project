"""Constantes de hardware, física e parâmetros de calibração editáveis em runtime."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, fields


SOFTWARE_VERSION: str = "1.0.0"

# Dimensões da matriz da manta sensora FSR (WangYing)
HID_VID: int = 6860   # Vendor ID  (0x1ACC)
HID_PID: int = 6733   # Product ID (0x1A4D)
HID_ROWS: int = 32
HID_COLS: int = 64
HID_PACKET_SIZE: int = 64  # bytes por report HID

# Dimensões físicas reais da manta para integração de força (Modelo B)
MANTA_WIDTH_M: float = 0.90    # largura da maca (m)
MANTA_HEIGHT_M: float = 1.80   # comprimento da maca (m)
PIXEL_AREA_M2: float = (MANTA_WIDTH_M / HID_COLS) * (MANTA_HEIGHT_M / HID_ROWS)

TARE_SAMPLE_COUNT: int = 5

# Critérios metrológicos de estabilidade e Média Móvel Contínua (60s)
STABILITY_EPSILON_KG: float = 0.5       # |Δm| máximo entre frames consecutivos
STABILITY_TMIN_S: float = 10.0          # janela mínima para progresso de estabilidade
STABILITY_VARIANCE_KG2: float = 0.5    # variância máxima da janela
STABILITY_DRIFT_KG_S: float = 0.8      # drift máximo em kg/s
STABILITY_WINDOW_SIZE: int = 40         # frames na janela de variância/drift
WEIGHT_WINDOW_SECONDS: float = 60.0    # janela de média móvel contínua (1 minuto)
FAST_RESET_THRESHOLD_KG: float = 2.0   # degrau que aciona reset imediato do buffer móvel


@dataclass
class CalibrationParams:
    """Parâmetros de calibração e limiares clínicos mutáveis em runtime."""

    deadzone_threshold: float = 10.0      # ignora leituras brutas <= este limiar
    ema_alpha: float = 0.5               # suavização exponencial do peso total
    posture_tolerance: float = 0.15       # tolerância de assimetria postural
    posture_timeout_seconds: int = 3600  # tempo máximo sem rotação (60 min)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def update(self, key: str, value: float) -> None:
        """Atualiza um parâmetro de forma thread-safe."""
        if not hasattr(self, key) or key.startswith("_"):
            raise ValueError(f"Parâmetro desconhecido: {key}")
        with self._lock:
            expected_type = type(getattr(self, key))
            setattr(self, key, expected_type(value))

    def snapshot(self) -> dict:
        """Retorna snapshot imutável dos parâmetros."""
        with self._lock:
            return {
                f.name: getattr(self, f.name)
                for f in fields(self)
                if not f.name.startswith("_")
            }
