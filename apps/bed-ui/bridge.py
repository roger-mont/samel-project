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
    WEIGHT_WINDOW_SECONDS, FAST_RESET_THRESHOLD_KG,
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

# Média Móvel Inteligente de 60 segundos
_rolling_history: collections.deque[tuple[float, float]] = collections.deque()
_weight_locked: bool = False
_locked_weight_kg: float = 0.0
_stable_progress_pct: float = 0.0
_current_posture_info: dict = {
    "posture": "Decúbito Dorsal",
    "asymmetry_pct": 0.0,
    "asymmetry_label": "Pressão Simétrica",
    "relief_score": 94,
}


def _resolve_file_path(filename: str) -> str:
    """Busca o arquivo no CWD atual, no diretório do módulo ou na pasta storage."""
    from pathlib import Path
    app_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        app_dir / filename,
        app_dir / "storage" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(app_dir / filename)


def start_reading_loop(
    reader: BaseFrameReader,
    params: CalibrationParams,
    monitor: PostureMonitor,
    calib_path: str = "calibration.json",
) -> None:
    """Inicia thread daemon que lê frames e processa continuamente."""
    resolved_path = _resolve_file_path(calib_path)
    calib = load_calibration(resolved_path)
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

            # Média Móvel Inteligente de 60 segundos com Reset Rápido
            now = time.monotonic()
            if net_mass < 0.5:
                _rolling_history.clear()
                displayed_weight = 0.0
                _weight_locked = False
                _locked_weight_kg = 0.0
                _stable_progress_pct = 0.0
            else:
                if _rolling_history:
                    curr_avg = sum(m for _, m in _rolling_history) / len(_rolling_history)
                    if abs(net_mass - curr_avg) > FAST_RESET_THRESHOLD_KG:
                        # Reset rápido se houver mudança brusca (degrau > 2 kg)
                        _rolling_history.clear()
                        logger.info("degrau detectado (>%.1f kg) — buffer de 60s reiniciado", FAST_RESET_THRESHOLD_KG)

                _rolling_history.append((now, net_mass))
                cutoff = now - WEIGHT_WINDOW_SECONDS
                while _rolling_history and _rolling_history[0][0] < cutoff:
                    _rolling_history.popleft()

                displayed_weight = sum(m for _, m in _rolling_history) / len(_rolling_history)
                window_span_s = _rolling_history[-1][0] - _rolling_history[0][0] if len(_rolling_history) > 1 else 0.0
                _stable_progress_pct = min(100.0, (window_span_s / WEIGHT_WINDOW_SECONDS) * 100.0)
                _weight_locked = _stable_progress_pct >= 95.0
                _locked_weight_kg = displayed_weight

            monitor.update_tolerance(snap["posture_tolerance"])
            monitor.update_timeout(snap["posture_timeout_seconds"])
            alert = monitor.update(force_matrix)
            posture_info = monitor.classify_posture(force_matrix)

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
                _current_weight_kg = displayed_weight
                _current_force_n = force_n
                _static_seconds = monitor.elapsed_seconds
                _is_alert = alert
                _current_posture_info = posture_info
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
            "force_n": round(_current_force_n, 2),
            "static_seconds": round(_static_seconds, 1),
            "is_alert": _is_alert,
            "rows": HID_ROWS,
            "cols": HID_COLS,
            "status": _connection_status,
            "is_locked": _weight_locked,
            "locked_weight_kg": round(_locked_weight_kg, 2),
            "stable_progress_pct": round(_stable_progress_pct, 1),
            "posture_info": _current_posture_info,
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


def _get_latest_calibration_date(calib_path: str = "calibration.json") -> str | None:
    """Extrai a data mais recente de calibração do calibration.json (raiz ou blocos)."""
    try:
        import json
        from pathlib import Path
        resolved = _resolve_file_path(calib_path)
        p = Path(resolved)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        
        dates: list[str] = []
        # 1. Data global do modelo ativo (raiz)
        if data.get("calibrated_at"):
            dates.append(data["calibrated_at"])
        
        # 2. Datas de calibrações individuais dos blocos
        blocks = data.get("blocks", {})
        for b in blocks.values():
            if isinstance(b, dict) and b.get("calibrated_at"):
                dates.append(b["calibrated_at"])
        
        if dates:
            return sorted(dates)[-1]
        return None
    except Exception:
        return None


def get_calibration_snapshot() -> dict:
    """Retorna cópia dos parâmetros de calibração acrescido da data de calibração."""
    snap = _calibration_ref.snapshot()
    calib_date = _get_latest_calibration_date()
    if calib_date:
        snap["calibrated_at"] = calib_date
    return snap


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
