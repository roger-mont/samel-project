"""Ponte Python↔JS via Eel — funções expostas para o frontend."""
from __future__ import annotations

import collections
import logging
import threading
import time

import eel
import numpy as np

from config.settings import (
    CalibrationParams, HID_ROWS, HID_COLS, TARE_SAMPLE_COUNT,
    STABILITY_EPSILON_KG, STABILITY_TMIN_S,
    STABILITY_VARIANCE_KG2, STABILITY_DRIFT_KG_S, STABILITY_WINDOW_SIZE,
)
from services.serial_reader import BaseFrameReader
from services.math_pipeline import compute_force_matrix, compute_total_mass, apply_ema
from services.calibration_store import CalibData, load_calibration, GRAVITY_M_S2
from providers.posture_monitor import PostureMonitor
from services.tare_store import load_tare, save_tare

logger = logging.getLogger(__name__)

# Estado global compartilhado entre thread de leitura, Eel e API
_state_lock = threading.Lock()
_current_heatmap: list[list[float]] = []
_current_weight_kg: float = 0.0
_current_force_n: float = 0.0
_static_seconds: float = 0.0
_is_alert: bool = False
_connection_status: str = "desconectado"
_tare_offset_kg: float = load_tare()
_tare_pending: bool = False
_tare_samples: list[float] = []
_monitor_ref: PostureMonitor | None = None
_reader_ref: BaseFrameReader | None = None

# Weight lock — estabilidade (Metodologia §13)
_weight_locked: bool = False
_locked_weight_kg: float = 0.0
_stable_consecutive_s: float = 0.0


def start_reading_loop(
    reader: BaseFrameReader,
    params: CalibrationParams,
    monitor: PostureMonitor,
    calib_path: str = "calibration.json",
) -> None:
    """Inicia thread daemon que lê frames e processa continuamente."""
    calib = load_calibration(calib_path)
    thread = threading.Thread(
        target=_reading_loop,
        args=(reader, params, monitor, calib),
        daemon=True,
        name="serial-reader-loop",
    )
    thread.start()
    logger.info("loop de leitura iniciado")


def _check_stability(
    weight_history: collections.deque,
    current_kg: float,
    previous_kg: float,
    dt_s: float,
) -> bool:
    """Retorna True se o frame atual é estável conforme todos os critérios.

    Critérios combinados:
      1. |m(t) - m(t-1)| < STABILITY_EPSILON_KG   (Metodologia §13)
      2. variance(janela) < STABILITY_VARIANCE_KG2
      3. drift < STABILITY_DRIFT_KG_S
    """
    # Critério 1 — delta entre frames consecutivos
    delta = abs(current_kg - previous_kg)
    if delta >= STABILITY_EPSILON_KG:
        return False

    # Critérios 2 e 3 — só avaliáveis com janela suficiente
    if len(weight_history) < 3:
        return delta < STABILITY_EPSILON_KG

    window = np.array(weight_history)
    variance = float(np.var(window))
    if variance >= STABILITY_VARIANCE_KG2:
        return False

    # Drift: diferença entre primeiro e último da janela, normalizada pelo tempo
    window_duration_s = len(window) * dt_s
    if window_duration_s > 0:
        drift = abs(float(window[-1] - window[0])) / window_duration_s
        if drift >= STABILITY_DRIFT_KG_S:
            return False

    return True


def _reading_loop(
    reader: BaseFrameReader,
    params: CalibrationParams,
    monitor: PostureMonitor,
    calib: CalibData,
) -> None:
    global _current_heatmap, _current_weight_kg, _current_force_n
    global _static_seconds, _is_alert, _connection_status
    global _tare_offset_kg, _tare_pending, _tare_samples
    global _weight_locked, _locked_weight_kg, _stable_consecutive_s

    previous_weight = 0.0
    weight_history: collections.deque = collections.deque(maxlen=STABILITY_WINDOW_SIZE)
    last_frame_time = time.monotonic()

    while True:
        try:
            frame_start = time.monotonic()
            dt_s = frame_start - last_frame_time
            last_frame_time = frame_start

            connected = reader.is_connected()
            adc_matrix = reader.read_frame()

            force_matrix = compute_force_matrix(adc_matrix, params)
            raw_mass = compute_total_mass(force_matrix, calib)

            snap = params.snapshot()
            smoothed_mass = apply_ema(raw_mass, previous_weight, snap["ema_alpha"])
            previous_weight = smoothed_mass

            # Peso líquido
            net_mass = max(0.0, smoothed_mass - _tare_offset_kg)

            # Força total em Newton (para exibição)
            from services.math_pipeline import compute_total_force
            force_n = compute_total_force(force_matrix, calib)

            # Weight lock — critério de estabilidade combinado
            weight_history.append(net_mass)
            frame_stable = _check_stability(weight_history, net_mass, previous_weight, dt_s)

            if frame_stable and net_mass > 0.5:
                _stable_consecutive_s += dt_s
                if _stable_consecutive_s >= STABILITY_TMIN_S and not _weight_locked:
                    _weight_locked = True
                    _locked_weight_kg = net_mass
                    logger.info("peso travado: %.2f kg (estável por %.1fs)", net_mass, _stable_consecutive_s)
            else:
                if _stable_consecutive_s > 0:
                    _stable_consecutive_s = 0.0
                if net_mass < 0.5:
                    _weight_locked = False
                    _locked_weight_kg = 0.0

            # Detecção de mudança significativa → destrava
            if _weight_locked and abs(net_mass - _locked_weight_kg) > STABILITY_EPSILON_KG * 3:
                _weight_locked = False
                _locked_weight_kg = 0.0
                _stable_consecutive_s = 0.0
                logger.info("peso destravado: variacao significativa detectada")

            monitor.update_tolerance(snap["posture_tolerance"])
            monitor.update_timeout(snap["posture_timeout_seconds"])
            alert = monitor.update(force_matrix)

            # Normaliza heatmap para 0-1
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
                _current_force_n = force_n
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
        progress = min(100.0, _stable_consecutive_s / STABILITY_TMIN_S * 100) if STABILITY_TMIN_S > 0 else 0.0
        return {
            "heatmap": _current_heatmap,
            "weight_kg": round(_current_weight_kg, 2),
            "force_n": round(_current_force_n, 2),
            "static_seconds": round(_static_seconds, 1),
            "is_alert": _is_alert,
            "rows": HID_ROWS,
            "cols": HID_COLS,
            "status": _connection_status,
            "is_locked": _weight_locked,
            "locked_weight_kg": round(_locked_weight_kg, 2),
            "stable_progress_pct": round(progress, 1),
        }


@eel.expose
def update_calibration(key: str, value: float) -> dict:
    """Atualiza parâmetro de calibração pelo nome."""
    return update_calibration_param(key, value)


@eel.expose
def get_calibration() -> dict:
    """Retorna parâmetros de calibração atuais."""
    return get_calibration_snapshot()


@eel.expose
def get_max_pressure() -> int:
    """Retorna o maior valor de pressão já recebido desde o início da sessão."""
    if _reader_ref is not None and hasattr(_reader_ref, "_max_seen"):
        return _reader_ref._max_seen
    return 0


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


def set_reader_ref(reader: BaseFrameReader) -> None:
    """Injeta referência ao reader ativo para expor métricas ao frontend."""
    global _reader_ref
    _reader_ref = reader


# --- Funções públicas para api/server.py ---


def get_current_snapshot() -> dict:
    """Snapshot do estado atual sem heatmap (consumidor externo)."""
    with _state_lock:
        return {
            "weight_kg": round(_current_weight_kg, 2),
            "force_n": round(_current_force_n, 2),
            "is_alert": _is_alert,
            "status": _connection_status,
            "static_seconds": round(_static_seconds, 1),
            "tare_active": _tare_offset_kg > 0.0,
            "tare_pending": _tare_pending,
            "tare_offset_kg": round(_tare_offset_kg, 3),
            "is_locked": _weight_locked,
            "locked_weight_kg": round(_locked_weight_kg, 2),
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
