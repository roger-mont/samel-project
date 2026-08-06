"""Ponte Python↔JS via Eel — funções expostas para o frontend."""
from __future__ import annotations

import logging
import threading
import time

import eel
import numpy as np

from config.settings import CalibrationParams, HID_ROWS, HID_COLS, TARE_SAMPLE_COUNT
from services.serial_reader import BaseFrameReader
from services.math_pipeline import compute_force_matrix, compute_total_mass, apply_ema
from providers.posture_monitor import PostureMonitor
from services.tare_store import load_tare, save_tare

logger = logging.getLogger(__name__)

# Estado global compartilhado entre thread de leitura, Eel e API
_state_lock = threading.Lock()
_current_heatmap: list[list[float]] = []
_current_weight_kg: float = 0.0
_static_seconds: float = 0.0
_is_alert: bool = False
_connection_status: str = "desconectado"
_tare_offset_kg: float = load_tare()
_tare_pending: bool = False
_tare_samples: list[float] = []
_monitor_ref: PostureMonitor | None = None


def start_reading_loop(
    reader: BaseFrameReader,
    params: CalibrationParams,
    monitor: PostureMonitor,
) -> None:
    """Inicia thread daemon que lê frames e processa continuamente."""
    thread = threading.Thread(
        target=_reading_loop,
        args=(reader, params, monitor),
        daemon=True,
        name="serial-reader-loop",
    )
    thread.start()
    logger.info("loop de leitura iniciado")


def _reading_loop(
    reader: BaseFrameReader,
    params: CalibrationParams,
    monitor: PostureMonitor,
) -> None:
    global _current_heatmap, _current_weight_kg
    global _static_seconds, _is_alert, _connection_status
    global _tare_offset_kg, _tare_pending, _tare_samples

    previous_weight = 0.0

    while True:
        try:
            connected = reader.is_connected()
            adc_matrix = reader.read_frame()

            force_matrix = compute_force_matrix(adc_matrix, params)
            raw_mass = compute_total_mass(force_matrix)

            snap = params.snapshot()
            smoothed_mass = apply_ema(raw_mass, previous_weight, snap["ema_alpha"])
            previous_weight = smoothed_mass

            # Peso líquido — leitura de _tare_offset_kg sem lock (GIL-safe em CPython)
            net_mass = max(0.0, smoothed_mass - _tare_offset_kg)

            monitor.update_tolerance(snap["posture_tolerance"])
            monitor.update_timeout(snap["posture_timeout_seconds"])
            alert = monitor.update(force_matrix)

            # Normaliza heatmap para 0-1 (frontend escala para cor)
            max_val = float(np.max(force_matrix))
            normalized = force_matrix / max_val if max_val > 0 else force_matrix

            tare_to_save: float | None = None

            with _state_lock:
                if _tare_pending:
                    _tare_samples.append(smoothed_mass)
                    if len(_tare_samples) >= TARE_SAMPLE_COUNT:
                        _tare_offset_kg = float(np.mean(_tare_samples))
                        tare_to_save = _tare_offset_kg
                        _tare_pending = False
                        _tare_samples = []

                _current_heatmap = normalized.tolist()
                _current_weight_kg = net_mass
                _static_seconds = monitor.elapsed_seconds
                _is_alert = alert
                _connection_status = "conectado" if connected else "desconectado"

            if tare_to_save is not None:
                logger.info("tara concluida: %.4f kg", tare_to_save)
                save_tare(tare_to_save)

        except Exception as err:
            logger.error("erro no loop de leitura: %s", err)
            with _state_lock:
                _connection_status = f"erro: {err}"
            time.sleep(0.5)


@eel.expose
def get_sensor_data() -> dict:
    """Retorna snapshot dos dados atuais para o frontend."""
    with _state_lock:
        return {
            "heatmap": _current_heatmap,
            "weight_kg": round(_current_weight_kg, 2),
            "static_seconds": round(_static_seconds, 1),
            "is_alert": _is_alert,
            "rows": HID_ROWS,
            "cols": HID_COLS,
            "status": _connection_status,
        }


@eel.expose
def update_calibration(key: str, value: float) -> dict:
    """Atualiza parâmetro de calibração pelo nome."""
    return update_calibration_param(key, value)


@eel.expose
def get_calibration() -> dict:
    """Retorna parâmetros de calibração atuais."""
    return get_calibration_snapshot()


# Referência injetada pelo main.py
_calibration_ref: CalibrationParams = CalibrationParams()


def set_calibration_ref(params: CalibrationParams) -> None:
    """Injeta referência dos parâmetros de calibração."""
    global _calibration_ref
    _calibration_ref = params


def set_monitor_ref(monitor: PostureMonitor) -> None:
    """Injeta referência ao PostureMonitor — análogo a set_calibration_ref."""
    global _monitor_ref
    _monitor_ref = monitor


# --- Funções públicas para api/server.py ---


def get_current_snapshot() -> dict:
    """Snapshot do estado atual sem heatmap (consumidor externo)."""
    with _state_lock:
        return {
            "weight_kg": round(_current_weight_kg, 2),
            "is_alert": _is_alert,
            "status": _connection_status,
            "static_seconds": round(_static_seconds, 1),
            "tare_active": _tare_offset_kg > 0.0,
            "tare_pending": _tare_pending,
            "tare_offset_kg": round(_tare_offset_kg, 3),
        }


def get_calibration_snapshot() -> dict:
    """Retorna cópia dos parâmetros de calibração."""
    return _calibration_ref.snapshot()


def update_calibration_param(key: str, value: float) -> dict:
    """Atualiza parâmetro de calibração — usado por Eel e API REST."""
    try:
        _calibration_ref.update(key, value)
        return {"ok": True, "params": _calibration_ref.snapshot()}
    except ValueError as err:
        return {"ok": False, "error": str(err)}


@eel.expose
def tare_start() -> None:
    """Inicia coleta de TARE_SAMPLE_COUNT amostras para calcular o offset."""
    start_tare_sampling()


@eel.expose
def tare_clear() -> None:
    """Remove a tara via frontend."""
    clear_tare()


@eel.expose
def get_tare_status() -> dict:
    """Estado atual da tara — exposto ao frontend."""
    with _state_lock:
        return {
            "offset_kg": round(_tare_offset_kg, 3),
            "active": _tare_offset_kg > 0.0,
            "pending": _tare_pending,
        }


def start_tare_sampling() -> None:
    """Inicia coleta de TARE_SAMPLE_COUNT amostras para calcular o offset."""
    global _tare_pending, _tare_samples
    with _state_lock:
        _tare_samples = []
        _tare_pending = True
    logger.info("tara: coletando %d amostras", TARE_SAMPLE_COUNT)


def clear_tare() -> None:
    """Remove a tara — pressão exibida volta ao bruto. Persiste offset=0."""
    global _tare_offset_kg, _tare_pending, _tare_samples
    with _state_lock:
        _tare_offset_kg = 0.0
        _tare_pending = False
        _tare_samples = []
    save_tare(0.0)
    logger.info("tara removida")


def reset_posture_monitor() -> None:
    """Reseta o PostureMonitor (cronômetro e referência de postura)."""
    if _monitor_ref is None:
        logger.warning("reset_posture_monitor chamado sem monitor_ref injetado")
        return
    _monitor_ref.reset()
