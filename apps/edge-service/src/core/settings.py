"""Constantes de hardware e parâmetros de aquisição editáveis em runtime."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field, fields
from typing import Any


SOFTWARE_VERSION: str = "2.0.0"

# Dimensões da matriz da manta sensora FSR (WangYing)
HID_VID: int = 6860   # Vendor ID  (0x1ACC)
HID_PID: int = 6733   # Product ID (0x1A4D)
HID_ROWS: int = 32
HID_COLS: int = 64
HID_PACKET_SIZE: int = 64  # bytes por report HID

# Maior valor que o conversor da manta entrega (resposta digital RAW de 8 bits)
RAW_MAX: int = 255


@dataclass
class CalibrationParams:
    """Parâmetros de aquisição, estabilidade e limiares clínicos mutáveis em runtime.

    Os limiares de estabilidade ainda não são constantes definitivas: os valores
    padrão são pontos de partida, a serem ajustados pelos experimentos.
    Limiares "_rel" são frações da soma corrigida da manta (ex.: 0.02 = 2%).
    """

    deadzone_threshold: float = 10.0        # zera leituras <= limiar no mapa de calor e na postura
    posture_tolerance: float = 0.15         # tolerância de assimetria postural
    posture_timeout_seconds: int = 3600     # tempo máximo sem rotação (60 min)

    # Baseline espacial (tara)
    t_tare_s: float = 5.0                   # T_TARE: duração da aquisição da tara
    baseline_std_max_raw: float = 3.0       # desvio médio por célula aceito na tara (BASELINE_OK)
    auto_rezero_empty_s: float = 30.0       # tempo com maca vazia antes da tara automática
    tare_empty_min_s: float = 3.0           # tara manual só com SEM_CARGA contínuo há pelo menos isto

    # Detecção de carga, estabilidade e movimento
    load_threshold_raw: float = 150.0       # soma corrigida acima da qual há carga
    stability_window_s: float = 1.0         # STABILITY_WINDOW: janela de média e desvio
    stability_lag_s: float = 1.0            # Δt de D(t) e de D_M(t)
    std_threshold_rel: float = 0.02         # STD_THRESHOLD: σ/μ máximo
    delta_threshold_rel: float = 0.01       # DELTA_THRESHOLD: D/μ máximo
    min_stable_time_s: float = 15.0         # MIN_STABLE_TIME: tempo contínuo com o critério atendido
    movement_threshold_rel: float = 0.15    # D_mov: D_M/μ acima do qual há movimento
    saturation_raw: float = float(RAW_MAX)  # leitura que marca célula saturada (saturação digital)
    # LOW_SENSITIVITY: célula em que 1 RAW vale mais que isto em gramas (dm/dRAW da curva).
    # Valor provisório, a ser definido nos ensaios; 0 desliga. Só informativo.
    low_sensitivity_g_per_raw: float = 10.0
    # STALE_SECTOR: setor ativo sem atualização completa (256 células) há mais que isto.
    # Valor provisório: medir o intervalo real em GET /api/v1/diagnostics/acquisition
    # (update_interval_ms por setor) e ajustar; 0 desliga.
    stale_sector_s: float = 1.0

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

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Aplica valores vindos do config.json, ignorando chaves desconhecidas."""
        for key, value in overrides.items():
            try:
                self.update(key, value)
            except (ValueError, TypeError):
                continue

    def snapshot(self) -> dict:
        """Retorna snapshot imutável dos parâmetros."""
        with self._lock:
            return {
                f.name: getattr(self, f.name)
                for f in fields(self)
                if not f.name.startswith("_")
            }

    def configuration_id(self, extra: dict[str, Any] | None = None) -> str:
        """Identificador curto e determinístico do conjunto de parâmetros em vigor."""
        payload = json.dumps({**self.snapshot(), **(extra or {})}, sort_keys=True)
        return "CFG-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
