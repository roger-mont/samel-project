"""Ponte Eel para comunicação entre a Interface Desktop e o Edge-Service."""
from __future__ import annotations

import logging
from typing import Any

import eel

from services.edge_client import EdgeClient

logger = logging.getLogger(__name__)


@eel.expose
def get_sensor_data() -> dict[str, Any]:
    """Retorna snapshot ao vivo da telemetria da maca."""
    live = EdgeClient.get_live_telemetry()
    if live:
        return live
    return {
        "heatmap": [],
        "weight_kg": 0.0,
        "force_n": 0.0,
        "static_seconds": 0,
        "is_alert": False,
        "status": "desconectado",
        "is_locked": False,
        "locked_weight_kg": 0.0,
        "stable_progress_pct": 0.0,
        "posture_info": {
            "posture": "Leito Livre",
            "asymmetry_pct": 0.0,
            "asymmetry_label": "Sem Carga",
            "relief_score": 100,
        },
    }


@eel.expose
def trigger_tare() -> dict[str, str]:
    """Dispara ciclo de tara no Edge Service."""
    ok = EdgeClient.trigger_tare()
    return {"status": "ok" if ok else "error"}


@eel.expose
def update_calibration(key: str, value: float) -> dict[str, Any]:
    """Atualiza parâmetro de calibração no Edge Service."""
    ok = EdgeClient.update_calibration_param(key, value)
    params = EdgeClient.get_calibration_params() or {}
    return {"status": "ok" if ok else "error", "params": params}


@eel.expose
def get_calibration() -> dict[str, Any]:
    """Retorna parâmetros de calibração vigentes."""
    return EdgeClient.get_calibration_params() or {
        "deadzone_threshold": 10.0,
        "ema_alpha": 0.5,
        "posture_tolerance": 0.15,
        "posture_timeout_seconds": 3600,
    }


@eel.expose
def get_max_pressure() -> int:
    """Retorna 255 por padrão."""
    return 255


@eel.expose
def get_dashboard_charts(window_hours: int = 4) -> dict[str, Any]:
    """Retorna séries temporais consolidadas dos 3 gráficos (9 pontos em 4h)."""
    return EdgeClient.get_charts_analytics(window_hours) or {}


@eel.expose
def get_dashboard_events(limit: int = 5) -> list[dict[str, Any]]:
    """Retorna lista dos últimos eventos posturais."""
    return EdgeClient.get_recent_events(limit) or []


@eel.expose
def get_dashboard_summary() -> dict[str, Any]:
    """Retorna o resumo diário do dia civil corrente."""
    return EdgeClient.get_daily_summary() or {}


@eel.expose
def get_system_config() -> dict[str, Any]:
    """Retorna as configurações operacionais do sistema."""
    return EdgeClient.get_system_config() or {
        "maca_id": "MACA-EDGE-001",
        "central_api_url": "http://localhost:8000",
        "sync_interval_sec": 60,
    }


@eel.expose
def save_system_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Salva configurações operacionais do sistema."""
    res = EdgeClient.update_system_config(config_data)
    return res or {"status": "error"}
