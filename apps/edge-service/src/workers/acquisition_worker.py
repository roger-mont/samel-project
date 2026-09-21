"""Worker de aquisição contínua de hardware (USB HID / Serial), processamento físico e persistência."""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.settings import (
    CalibrationParams, TARE_SAMPLE_COUNT,
    WEIGHT_WINDOW_SECONDS, FAST_RESET_THRESHOLD_KG,
)
from database.db_local import LocalDatabase
from hardware.serial_reader import BaseFrameReader, HidFrameReader, FakeSerialReader
from math_engine.creep_compensator import CreepCompensator, AutoReZero
from math_engine.mat_baseline_compensator import MatBaselineCompensator
from math_engine.ml_weight import MLWeightPredictor
from math_engine.pipeline import (
    compute_force_matrix, compute_total_mass, compute_total_force,
    compute_cop, apply_ema, correct_baseline,
)
from math_engine.reading_stability_detector import ReadingStabilityDetector, ReadingState
from providers.posture_monitor import PostureMonitor
from storage.calibration_store import CalibData, load_calibration
from storage.tare_store import load_tare, save_tare

logger = logging.getLogger(__name__)

_STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
_CALIB_PATH = _STORAGE_DIR / "calibration.json"
_TARE_PATH = _STORAGE_DIR / "tare.json"
_BASELINE_PATH = _STORAGE_DIR / "baseline_raw.json"


def _resolve_display_weight(
    state: ReadingState,
    current_net_mass: float,
    last_valid_weight: float,
) -> tuple[float, float, str, bool]:
    """Determina o peso e o label textual conforme a FSM de estabilidade."""
    if state == ReadingState.NO_LOAD:
        return 0.0, 0.0, "SEM CARGA", False
    if state == ReadingState.MOVING:
        msg = "PACIENTE EM MOVIMENTO — AGUARDANDO ESTABILIZAÇÃO"
        return last_valid_weight, last_valid_weight, msg, False
    if state == ReadingState.STABILIZING:
        return last_valid_weight, last_valid_weight, "ESTABILIZANDO", False
    return current_net_mass, current_net_mass, "MEDIÇÃO ESTÁVEL", True


class AcquisitionWorker:
    """Worker autônomo 24/7 responsável por:
    1. Leitura contínua dos frames da manta sensora a ~40 Hz.
    2. Execução do pipeline matemático e cálculo do peso estável (janela móvel 60s).
    3. Compensação de creep/drift temporal dos FSRs.
    4. Auto-rezero quando manta desocupada por >30s.
    5. Monitoramento postural e cálculo do tempo de permanência de decúbito.
    6. Ingestão periódica a cada 60s no SQLite e registro de eventos imediatos.
    7. Transmissão em tempo real via callbacks/WebSocket.
    """

    def __init__(
        self,
        db: LocalDatabase,
        maca_id: str = "MACA-01",
        reader: BaseFrameReader | None = None,
        broadcast_callback: Callable[[dict[str, Any]], None] | None = None,
        baseline_compensator: MatBaselineCompensator | None = None,
        stability_detector: ReadingStabilityDetector | None = None,
    ) -> None:
        self._db = db
        self._maca_id = maca_id
        self._broadcast_callback = broadcast_callback
        self._is_running = False
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

        # Configurações e Armazenamento
        self.params = CalibrationParams()
        self.calib = load_calibration(_CALIB_PATH)
        self.ml_predictor = MLWeightPredictor()
        self._tare_offset_kg = load_tare(_TARE_PATH)
        self._tare_pending = False
        self._tare_samples: list[float] = []

        # Compensação de Baseline e Estabilidade
        self.baseline_compensator = (
            baseline_compensator
            if baseline_compensator is not None
            else MatBaselineCompensator(baseline_path=_BASELINE_PATH)
        )
        self.stability_detector = (
            stability_detector
            if stability_detector is not None
            else ReadingStabilityDetector(
                t_min_s=self.params.stability_tmin_seconds,
                epsilon_s=self.params.stability_epsilon_raw,
                d_max_matrix=self.params.motion_dmax_matrix,
            )
        )

        # Compensação de Creep e Auto-ReZero
        self._creep = CreepCompensator(
            settle_frames=15,
            max_drift_kg_per_sec=0.3,
            step_threshold_kg=FAST_RESET_THRESHOLD_KG,
        )
        self._auto_rezero = AutoReZero(
            empty_threshold_kg=0.5,
            empty_duration_seconds=30.0,
        )

        # Monitor de Postura
        self.monitor = PostureMonitor(
            tolerance=self.params.posture_tolerance,
            timeout_seconds=self.params.posture_timeout_seconds,
        )

        # Driver de Hardware
        if reader is not None:
            self._reader = reader
        else:
            try:
                self._reader = HidFrameReader()
            except Exception as err:
                logger.warning("USB HID indisponível (%s) — iniciando leitor simulado de fallback", err)
                self._reader = FakeSerialReader()

        # Estado ao vivo (Live State)
        self._live_state: dict[str, Any] = {
            "heatmap": [],
            "weight_kg": 0.0,
            "force_n": 0.0,
            "static_seconds": 0,
            "is_alert": False,
            "status": "iniciando",
            "is_locked": False,
            "locked_weight_kg": 0.0,
            "stable_progress_pct": 0.0,
            "state_id": ReadingState.NO_LOAD.value,
            "state_label": "SEM CARGA",
            "matrix_delta": 0.0,
            "temporal_delta": 0.0,
            "dispersion_sigma": 0.0,
            "cop": {"row": 0.0, "col": 0.0},
            "posture_info": {
                "posture": "Leito Livre",
                "asymmetry_pct": 0.0,
                "asymmetry_label": "Sem Carga",
                "relief_score": 100,
            },
        }

    def start(self) -> None:
        """Inicia a thread de leitura contínua."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._loop, name="AcquisitionThread", daemon=True)
        self._thread.start()
        logger.info("AcquisitionWorker iniciado (Maca: %s)", self._maca_id)

    def stop(self) -> None:
        """Encerra o worker de forma graciosa."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._reader:
            self._reader.close()
        logger.info("AcquisitionWorker finalizado")

    def trigger_tare(self) -> None:
        """Solicita calibração de tara na próxima amostragem."""
        with self._state_lock:
            self._tare_pending = True
            self._tare_samples.clear()
        logger.info("Amostragem de tara iniciada")

    def get_live_snapshot(self) -> dict[str, Any]:
        """Retorna uma cópia thread-safe do estado atual."""
        with self._state_lock:
            return dict(self._live_state)

    def update_param(self, key: str, value: float) -> None:
        """Atualiza parâmetro de calibração em runtime e sincroniza estabilidade."""
        self.params.update(key, value)
        snap = self.params.snapshot()
        self.monitor.update_tolerance(snap["posture_tolerance"])
        self.monitor.update_timeout(int(snap["posture_timeout_seconds"]))
        if key == "stability_tmin_seconds":
            self.stability_detector.t_min_s = float(value)
        elif key == "motion_dmax_matrix":
            self.stability_detector.d_max_matrix = float(value)
        elif key == "stability_epsilon_raw":
            self.stability_detector.epsilon_s = float(value)

    def _loop(self) -> None:
        previous_weight = 0.0
        last_valid_weight = 0.0
        last_telemetry_time = time.monotonic()
        last_recorded_posture: str | None = None
        last_posture_change_time = time.monotonic()

        while self._is_running:
            try:
                now = time.monotonic()
                connected = self._reader.is_connected()
                raw_adc = self._reader.read_frame()

                # 1. FSM de estabilidade da aquisição
                stability_state = self.stability_detector.update(raw_adc, now)

                # 2. Correção de baseline matricial B(t)
                corrected_adc = correct_baseline(raw_adc, self.baseline_compensator)

                # 3. Auto-ReZero de baseline quando maca vazia por >=30s
                if self.baseline_compensator.evaluate_auto_rezero(raw_adc, now, min_empty_seconds=30.0):
                    self.baseline_compensator.update_baseline(raw_adc)
                    self._tare_offset_kg = 0.0
                    self._creep.reset()

                # 4. Pipeline Físico (deadzone, força e estimativa bruta)
                force_matrix = compute_force_matrix(corrected_adc, self.params)
                raw_mass = compute_total_mass(force_matrix, self.calib, self.ml_predictor)
                snap = self.params.snapshot()
                smoothed_mass = apply_ema(raw_mass, previous_weight, snap["ema_alpha"])
                previous_weight = smoothed_mass

                # 5. Compensação de Creep
                compensated_mass = self._creep.update(smoothed_mass)
                net_mass = max(0.0, compensated_mass - self._tare_offset_kg)
                force_n = compute_total_force(force_matrix, self.calib)

                # 6. Centro de Pressão 2D real
                cop_row, cop_col = compute_cop(force_matrix)

                # 7. Decisão de Leitura via FSM (congelamento em movimento)
                displayed_weight, last_valid_weight, state_label, weight_locked = _resolve_display_weight(
                    stability_state, net_mass, last_valid_weight
                )
                locked_weight_kg = displayed_weight if weight_locked else last_valid_weight
                stable_progress_pct = self.stability_detector.get_progress_pct(now)

                # 8. Classificação Postural e Prevenção LPP
                self.monitor.update_tolerance(snap["posture_tolerance"])
                self.monitor.update_timeout(int(snap["posture_timeout_seconds"]))
                alert = self.monitor.update(force_matrix)
                posture_info = self.monitor.classify_posture(force_matrix)

                # 9. Normalização do Heatmap (0-1)
                max_val = float(np.max(force_matrix))
                normalized_map = (force_matrix / max_val if max_val > 0 else force_matrix).tolist()

                # 10. Processamento de Tara manual
                tare_to_save: float | None = None
                with self._state_lock:
                    if self._tare_pending:
                        self._tare_samples.append(smoothed_mass)
                        if len(self._tare_samples) >= TARE_SAMPLE_COUNT:
                            self._tare_offset_kg = float(np.mean(self._tare_samples))
                            tare_to_save = self._tare_offset_kg
                            self._tare_pending = False
                            self._tare_samples.clear()

                    self._live_state = {
                        "maca_id": self._maca_id,
                        "heatmap": normalized_map,
                        "weight_kg": round(displayed_weight, 2),
                        "force_n": round(force_n, 1),
                        "static_seconds": int(self.monitor.elapsed_seconds),
                        "is_alert": alert,
                        "status": "conectado" if connected else "desconectado",
                        "is_locked": weight_locked,
                        "locked_weight_kg": round(locked_weight_kg, 2),
                        "stable_progress_pct": round(stable_progress_pct, 1),
                        "state_id": stability_state.value,
                        "state_label": state_label,
                        "matrix_delta": round(self.stability_detector.matrix_delta, 1),
                        "temporal_delta": round(self.stability_detector.temporal_delta, 1),
                        "dispersion_sigma": round(self.stability_detector.dispersion_sigma, 2),
                        "cop": {"row": round(cop_row, 2), "col": round(cop_col, 2)},
                        "posture_info": posture_info,
                    }

                if tare_to_save is not None:
                    save_tare(tare_to_save, _TARE_PATH)

                # 1. Ingestão periódica a cada 60s no SQLite
                if now - last_telemetry_time >= 60.0 and displayed_weight > 0.5:
                    last_telemetry_time = now
                    asym = posture_info.get("asymmetry_pct", 0.0)
                    self._db.enqueue_telemetry(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        maca_id=self._maca_id,
                        peso_kg=round(displayed_weight, 2),
                        indice_postural=round(asym / 100.0, 3) if asym else 0.0,
                        tempo_estatico_seg=int(self.monitor.elapsed_seconds),
                        status_alerta=alert,
                        payload={"relief_score": posture_info.get("relief_score", 100)},
                    )

                # 2. Registro imediato de eventos de transição postural
                curr_pos = posture_info.get("posture", "Leito Livre")
                if last_recorded_posture is None:
                    last_recorded_posture = curr_pos
                    last_posture_change_time = now
                elif curr_pos != last_recorded_posture and curr_pos != "Leito Livre":
                    dur_seg = int(now - last_posture_change_time)
                    if dur_seg >= 15:
                        self._db.record_posture_event(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            maca_id=self._maca_id,
                            postura_detectada=curr_pos,
                            postura_anterior=last_recorded_posture,
                            duracao_anterior_seg=dur_seg,
                            regiao_pico=posture_info.get("asymmetry_label"),
                            pico_pct=round(max_val, 1),
                            houve_alerta=alert,
                        )
                        last_recorded_posture = curr_pos
                        last_posture_change_time = now

                # 3. Broadcast do estado ao vivo via WebSocket
                if self._broadcast_callback is not None:
                    try:
                        self._broadcast_callback(self._live_state)
                    except Exception as b_err:
                        logger.debug("Falha no broadcast WebSocket: %s", b_err)

            except Exception as err:
                logger.error("Erro no loop do AcquisitionWorker: %s", err)
                with self._state_lock:
                    self._live_state["status"] = f"erro: {err}"
                time.sleep(0.5)

    def _execute_auto_tare(self, current_smoothed_mass: float) -> None:
        """Executa re-zero automático quando a manta está vazia por tempo suficiente."""
        old_tare = self._tare_offset_kg
        self._tare_offset_kg = current_smoothed_mass
        self._creep.reset()
        save_tare(self._tare_offset_kg, _TARE_PATH)
        logger.info(
            "Auto-ReZero executado: tara %.4f → %.4f kg",
            old_tare,
            self._tare_offset_kg,
        )
