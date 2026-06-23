"""Constantes e parâmetros de calibração editáveis em runtime."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, fields


GRID_ROWS: int = 6
GRID_COLS: int = 3
GRAVITY: float = 9.81

API_PORT: int = 8000
API_HOST: str = "0.0.0.0"
TARE_SAMPLE_COUNT: int = 5
TARE_FILE: str = "tare.json"
WS_PUSH_INTERVAL_SECONDS: int = 10


@dataclass
class CalibrationParams:
    """Parâmetros de calibração — mutáveis via UI em tempo real."""

    vcc: float = 3.3
    adc_resolution: int = 4095
    pulldown_resistor: float = 470_000.00
    factor_m: float = 59778938.84
    offset_b: float = 5.54
    deadzone_threshold: float = 0.5
    ema_alpha: float = 0.3
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
