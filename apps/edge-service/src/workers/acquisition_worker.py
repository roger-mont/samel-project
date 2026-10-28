"""Worker de aquisição contínua: RAW → baseline → janela → estados → modelo → validação → API."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.config import config_manager
from core.settings import HID_COLS, HID_ROWS, SOFTWARE_VERSION, CalibrationParams
from storage.calibration_store import BLOCK_REGIONS
from database.db_local import LocalDatabase
from hardware.acquisition_stats import AcquisitionSample
from hardware.serial_reader import BaseFrameReader, HidFrameReader, FakeSerialReader
from math_engine.baseline_return import BaselineReturnTracker
from math_engine.calibration_audit import audit_calibration
from math_engine.mass_models import BlockCurveModel, MassEstimate, ModelRegistry, WindowFrame
from math_engine.pipeline import apply_deadzone, window_mean
from math_engine.quality import QualityFlags, saturated_cells
from math_engine.reading_state import STATE_LABELS, ReadingState, ReadingStateMachine
from math_engine.residual_mass import ResidualMassTracker
from math_engine.spatial_baseline import SpatialBaseline
from math_engine.spatial_features import compute_spatial_features
from providers.posture_monitor import PostureMonitor
from storage.diagnostic_log import DiagnosticLog
from storage.experiment_recorder import ExperimentRecorder
from storage.gain_points import GainPointStore

logger = logging.getLogger(__name__)

_EDGE_ROOT = Path(__file__).resolve().parent.parent.parent
_STORAGE_DIR = _EDGE_ROOT / "src" / "storage"
_BLOCK_CURVES_PATH = _STORAGE_DIR / "block_curves.json"
_BASELINE_PATH = _STORAGE_DIR / "baseline_raw.json"
_EXPERIMENTS_DIR = _EDGE_ROOT / "data" / "experiments"
_GAIN_POINTS_PATH = _EDGE_ROOT / "data" / "gain_points.json"
# Trecho final usado para registrar um ponto de ajuste de ganho (só frames estáveis)
GAIN_CAPTURE_S = 5.0
TARE_EVENTS_KEPT = 50

WEIGHT_STATUS: dict[ReadingState, str] = {
    ReadingState.SEM_CARGA: "NO_LOAD",
    ReadingState.CARGA_DETECTADA: "LOAD_DETECTED",
    ReadingState.ESTABILIZANDO: "STABILIZING",
    ReadingState.MOVIMENTO: "MOVEMENT",
    ReadingState.ESTAVEL: "NOT_VALIDATED",
    ReadingState.MEDICAO_VALIDA: "VALID",
}


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


class AcquisitionWorker:
    """Worker autônomo 24/7.

    Separa aquisição, estabilização e conversão em massa: a aquisição e a tara não
    dependem do modelo de calibração, e o peso só é publicado como válido quando a
    máquina de estados e as flags de qualidade permitem.
    """

    def __init__(
        self,
        db: LocalDatabase,
        maca_id: str = "MACA-01",
        reader: BaseFrameReader | None = None,
        broadcast_callback: Callable[[dict[str, Any]], None] | None = None,
        baseline: SpatialBaseline | None = None,
        models: ModelRegistry | None = None,
        recorder: ExperimentRecorder | None = None,
        params: CalibrationParams | None = None,
        active_blocks: list[int] | None = None,
        gain_points: GainPointStore | None = None,
        diagnostics_dir: Path | None = None,
    ) -> None:
        self._db = db
        self._maca_id = maca_id
        self._broadcast_callback = broadcast_callback
        self._is_running = False
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

        self.params = params or CalibrationParams()
        if params is None:
            self.params.apply_overrides(config_manager.get("acquisition", {}) or {})

        # Só as células dos blocos ativos entram na medição; o RAW gravado é sempre a manta inteira
        blocks = active_blocks or config_manager.get("active_blocks") or list(BLOCK_REGIONS)
        self.active_blocks = sorted(int(b) for b in blocks if int(b) in BLOCK_REGIONS)
        self._mask = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)
        for bid in self.active_blocks:
            self._mask[BLOCK_REGIONS[bid]] = 1.0

        self.baseline = baseline or SpatialBaseline(_BASELINE_PATH)
        self.state_machine = ReadingStateMachine(self.params)
        # Só as curvas por bloco geradas dos CSVs de docs/ valem. O modelo de ML
        # (weight_model.joblib) e o calibration.json antigo continuam no disco, mas
        # não são carregados (pedido de 09/10: ignorar as calibrações anteriores).
        self.models = models or ModelRegistry(
            {"block_curves": BlockCurveModel(_BLOCK_CURVES_PATH)},
            selection=str(config_manager.get("mass_model", "auto")),
        )
        self.recorder = recorder or ExperimentRecorder(_EXPERIMENTS_DIR)
        self.gain_points = gain_points or GainPointStore(_GAIN_POINTS_PATH)

        self._tare_requested = False
        self._empty_since: float | None = None
        self._auto_tare_done = False
        # Tara só com a maca comprovadamente vazia (parecer de 20/10, item 5)
        self._no_load_since: float | None = None
        self._tare_gate: dict[str, Any] | None = None
        self.tare_events: deque[dict[str, Any]] = deque(maxlen=TARE_EVENTS_KEPT)
        # Log de diagnóstico contínuo e eventos de tara em disco (só com diagnostics_dir)
        self._diagnostics_dir = diagnostics_dir
        diag_cfg = config_manager.get("diagnostic_log", {}) or {}
        self.diag_log = DiagnosticLog(
            diagnostics_dir,
            interval_s=float(diag_cfg.get("interval_s", 0.5)),
            retention_days=int(diag_cfg.get("retention_days", 7)),
        ) if diagnostics_dir is not None else None
        self.baseline_return = BaselineReturnTracker()
        self._last_acquisition: dict[str, Any] | None = None
        self._raw_window: deque[tuple[float, np.ndarray]] = deque()
        self._full_window: deque[tuple[float, np.ndarray]] = deque()  # corrigido, manta inteira
        self._full_raw_window: deque[tuple[float, np.ndarray]] = deque()  # RAW medido, manta inteira
        self._capture_window: deque[tuple[float, bool, np.ndarray]] = deque()  # (t, estável, corrigido)
        self.residual = ResidualMassTracker()
        self._audit_cache: tuple[tuple[int, float], dict[str, Any]] | None = None
        self._last_valid: dict[str, Any] | None = None
        self._last_telemetry_time = time.monotonic()
        self._last_recorded_posture: str | None = None
        self._last_posture_change_time = time.monotonic()

        self.monitor = PostureMonitor(
            tolerance=self.params.posture_tolerance,
            timeout_seconds=self.params.posture_timeout_seconds,
        )

        if reader is not None:
            self._reader = reader
        else:
            try:
                self._reader = HidFrameReader()
            except Exception as err:
                logger.warning("USB HID indisponível (%s) — iniciando leitor simulado de fallback", err)
                self._reader = FakeSerialReader()

        self._live_state: dict[str, Any] = {
            "maca_id": self._maca_id,
            "status": "iniciando",
            "heatmap": [],
            "weight_kg": None,
            "weight_status": WEIGHT_STATUS[ReadingState.SEM_CARGA],
            "state_id": ReadingState.SEM_CARGA.value,
            "state_label": STATE_LABELS[ReadingState.SEM_CARGA],
            "stable": False,
            "movement": False,
            "is_locked": False,
            "locked_weight_kg": 0.0,
            "last_valid_weight_kg": None,
            "stable_progress_pct": 0.0,
            "static_seconds": 0,
            "is_alert": False,
            "cop": {"row": 0.0, "col": 0.0},
            "posture_info": {
                "posture": "Leito Livre",
                "asymmetry_pct": 0.0,
                "asymmetry_label": "Sem Carga",
                "relief_score": 100,
            },
        }

    # ------------------------------------------------------------- ciclo de vida

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._loop, name="AcquisitionThread", daemon=True)
        self._thread.start()
        logger.info("AcquisitionWorker iniciado (Maca: %s)", self._maca_id)

    def stop(self) -> None:
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._reader:
            self._reader.close()
        self.recorder.stop()
        if self.diag_log is not None:
            self.diag_log.close()
        logger.info("AcquisitionWorker finalizado")

    # ------------------------------------------------------------------ comandos

    def trigger_tare(self) -> dict[str, Any]:
        """Pede uma tara manual de T_TARE segundos (parecer de 20/10, item 5).

        Só é aceita com a maca em SEM_CARGA contínuo há tare_empty_min_s, sem
        movimento, com a manta conectada e sem setor desatualizado. A checagem é
        feita aqui com o último frame e de novo quando o worker começa a coleta;
        durante a coleta, carga, movimento ou setor parado cancelam a tara. Toda
        recusa e cancelamento fica em tare_events com o motivo.
        """
        with self._state_lock:
            gate = self._tare_gate
            reason = self._tare_block_reason(gate) if gate is not None else None
            if reason is None:
                self._tare_requested = True
        if reason is not None:
            self._tare_event("rejeitada", reason, automatic=False)
            return {"accepted": False, "reason": reason}
        logger.info("Tara manual solicitada")
        return {"accepted": True, "reason": None}

    def get_live_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._live_state)

    def update_param(self, key: str, value: float, persist: bool = False) -> None:
        """Atualiza parâmetro em runtime; com persist=True grava em config.json ('acquisition')."""
        self.params.update(key, value)
        snap = self.params.snapshot()
        self.monitor.update_tolerance(snap["posture_tolerance"])
        self.monitor.update_timeout(int(snap["posture_timeout_seconds"]))
        if persist:
            current = dict(config_manager.get("acquisition", {}) or {})
            current[key] = snap[key]
            config_manager.update_and_save({"acquisition": current})

    def extraction(self) -> dict[str, Any] | None:
        """Extração da janela atual célula a célula, com todos os setores pela curva de cada um.

        Usa a média corrigida da manta inteira na janela de estabilidade (a mesma
        filtragem do modelo); setores fora de active_blocks aparecem, mas não somam.
        """
        model = self.models.active()
        if not isinstance(model, BlockCurveModel) or not model.is_ready:
            return None
        corrected = self.window_corrected()
        live = self.get_live_snapshot()
        p = self.params.snapshot()
        return {
            "timestamp": live.get("timestamp"),
            "state_id": live.get("state_id"),
            "state_label": live.get("state_label"),
            "weight_status": live.get("weight_status"),
            "weight_kg": live.get("weight_kg"),
            "estimated_mass_kg": live.get("estimated_mass_kg"),
            "stable_progress_pct": live.get("stable_progress_pct"),
            "measurement_quality": live.get("measurement_quality"),
            "calibration_version": model.calibration_version,
            "active_blocks": self.active_blocks,
            "tare_id": live.get("tare_id"),
            "tare_in_progress": live.get("tare_in_progress"),
            "tare": live.get("tare"),
            "last_valid": live.get("last_valid"),
            "stale_sectors": live.get("stale_sectors", []),
            "stale_sector_s": p["stale_sector_s"],
            "acquisition": live.get("acquisition"),
            "residual_mass_empty": self.residual.snapshot(),
            "low_sensitivity_g_per_raw": p["low_sensitivity_g_per_raw"],
            **model.breakdown(corrected, self.active_blocks, self.window_raw(), p["saturation_raw"],
                              p["low_sensitivity_g_per_raw"]),
        }

    def cell_report(self, include_inactive: bool = False) -> dict[str, Any] | None:
        """Contribuição de cada célula na janela atual, da maior massa local para a menor (item 3.3).

        Manta inteira; setores fora de active_blocks aparecem, mas não somam nos totais.
        """
        model = self.models.active()
        if not isinstance(model, BlockCurveModel) or not model.is_ready:
            return None
        p = self.params.snapshot()
        corrected, raw = self.window_corrected(), self.window_raw()
        cd = model.cell_diagnostics(corrected, raw, p["saturation_raw"], p["low_sensitivity_g_per_raw"])
        summary = model.sector_summary(cd, self.active_blocks)
        live = self.get_live_snapshot()
        return {
            "timestamp": live.get("timestamp"),
            "weight_status": live.get("weight_status"),
            "calibration_version": model.calibration_version,
            "tare_id": live.get("tare_id"),
            "active_blocks": self.active_blocks,
            "low_sensitivity_g_per_raw": p["low_sensitivity_g_per_raw"],
            "saturation_raw": p["saturation_raw"],
            "totals": summary["totals"],
            "sectors": summary["sectors"],
            "cells": model.cell_records(cd, raw, self.baseline.mean_matrix if self.baseline.record else None,
                                        only_active=not include_inactive),
        }

    def calibration_audit(self) -> dict[str, Any] | None:
        """Auditoria das curvas em vigor (isotônica, faixa sem resposta, histerese), com cache."""
        model = self.models.active()
        if not isinstance(model, BlockCurveModel) or not model.is_ready:
            return None
        limit = float(self.params.snapshot()["low_sensitivity_g_per_raw"])
        key = (id(model.raw_definition), limit)
        if self._audit_cache is None or self._audit_cache[0] != key:
            self._audit_cache = (key, audit_calibration(model.raw_definition, limit))
        return self._audit_cache[1]

    def window_corrected(self) -> np.ndarray:
        """Média do RAW corrigido da manta inteira na janela de estabilidade."""
        frames = [m for _, m in list(self._full_window)]
        return window_mean(frames, np.zeros((HID_ROWS, HID_COLS)))

    def window_raw(self) -> np.ndarray:
        """Média do RAW medido (sem baseline) da manta inteira na janela de estabilidade."""
        frames = [m for _, m in list(self._full_raw_window)]
        return window_mean(frames, np.zeros((HID_ROWS, HID_COLS)))

    def stable_capture(self, min_frames: int = 3) -> tuple[np.ndarray, int, float] | None:
        """Média do RAW corrigido da manta inteira nos frames estáveis dos últimos GAIN_CAPTURE_S.

        Só devolve algo se o frame mais recente também for estável. Retorna
        (matriz média, número de frames, duração coberta em s).
        """
        window = list(self._capture_window)
        if not window or not window[-1][1]:
            return None
        frames = [(t, m) for t, ok, m in window if ok]
        if len(frames) < min_frames:
            return None
        mean = np.mean(np.stack([m for _, m in frames]), axis=0).astype(np.float64)
        return mean, len(frames), frames[-1][0] - frames[0][0]

    def configuration_id(self) -> str:
        return self.params.configuration_id(extra={"active_blocks": self.active_blocks})

    def tare_status(self) -> dict[str, Any]:
        record = self.baseline.record
        return {
            "collecting": self.baseline.is_collecting or self._tare_requested,
            "t_tare_s": self.params.t_tare_s,
            "tare_ok": self.baseline.has_baseline,
            "baseline_ok": self.baseline.is_baseline_ok(self.params.baseline_std_max_raw),
            "record": asdict(record) if record else None,
            "tare_empty_min_s": self.params.tare_empty_min_s,
            "last_event": self.tare_events[-1] if self.tare_events else None,
        }

    def acquisition_report(self) -> dict[str, Any]:
        """Telemetria USB por setor do último frame, com os setores parados (itens 1 a 3)."""
        live = self.get_live_snapshot()
        return {
            "timestamp": live.get("timestamp"),
            "stale_sector_s": self.params.stale_sector_s,
            "stale_sectors": live.get("stale_sectors", []),
            "active_blocks": self.active_blocks,
            "acquisition": self._last_acquisition,
            "note": None if self._last_acquisition else "o leitor em uso não informa pacotes por setor",
        }

    # -------------------------------------------------------------------- tara

    def _tare_block_reason(self, gate: dict[str, Any]) -> str | None:
        """Motivo para recusar a tara agora, ou None se a maca está comprovadamente vazia."""
        if not gate["hardware_ok"]:
            return "manta desconectada ou frame incompleto"
        if gate["stale"]:
            return "setor(es) sem atualização: " + ", ".join(str(s) for s in gate["stale"])
        if gate["movement"]:
            return "movimento sobre a maca"
        if self.baseline.has_baseline:
            # Sem tara anterior não há referência de maca vazia; a primeira só exige estabilidade
            if gate["state"] != ReadingState.SEM_CARGA:
                return (f"há carga sobre a maca ({STATE_LABELS[gate['state']]}, "
                        f"soma corrigida {gate['total_raw']:.0f} RAW)")
            need = float(self.params.tare_empty_min_s)
            empty_for = 0.0 if self._no_load_since is None else gate["now"] - self._no_load_since
            if empty_for < need:
                return f"maca vazia há só {empty_for:.1f} s; são precisos {need:g} s sem carga"
        return None

    def _tare_event(self, kind: str, reason: str | None, automatic: bool, record: Any = None) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": kind,
            "automatic": automatic,
            "reason": reason,
            "tare_id": record.tare_id if record else None,
            "baseline_sum": round(record.baseline_sum, 1) if record else None,
            "baseline_std": round(record.baseline_std, 3) if record else None,
        }
        self.tare_events.append(event)
        log = logger.warning if kind in ("rejeitada", "cancelada") else logger.info
        log("Tara %s %s%s", "automática" if automatic else "manual", kind, f": {reason}" if reason else "")
        if self._diagnostics_dir is not None:
            try:
                self._diagnostics_dir.mkdir(parents=True, exist_ok=True)
                with (self._diagnostics_dir / "tare_events.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception as err:
                logger.warning("Falha ao gravar evento de tara: %s", err)

    def _handle_tare(self, gate: dict[str, Any], raw: np.ndarray, configuration_id: str,
                     model_version: str) -> bool:
        """Pedido manual, vigilância da coleta e acúmulo. Devolve True quando uma tara termina."""
        now = gate["now"]
        with self._state_lock:
            requested, self._tare_requested = self._tare_requested, False
        if requested:
            reason = self._tare_block_reason(gate)
            if reason is not None:
                self._tare_event("rejeitada", reason, automatic=False)
            else:
                first = not self.baseline.has_baseline
                self.baseline.start_tare(now, self.params.t_tare_s, automatic=False)
                self._tare_event("iniciada", "primeira tara: sem tara anterior, a ausência de carga "
                                 "não pôde ser verificada" if first else None, automatic=False)
        if not self.baseline.is_collecting:
            return False
        reason = None
        if not gate["hardware_ok"]:
            reason = "manta desconectada durante a tara"
        elif gate["stale"]:
            reason = "setor(es) sem atualização durante a tara: " + ", ".join(str(s) for s in gate["stale"])
        elif gate["movement"]:
            reason = "movimento durante a tara"
        elif gate["state"] != ReadingState.SEM_CARGA and (self.baseline.has_baseline or self.baseline.is_automatic):
            reason = "carga detectada durante a tara"
        automatic = self.baseline.is_automatic
        if reason is not None:
            self.baseline.cancel_tare(reason)
            self._tare_event("cancelada", reason, automatic=automatic)
            return False
        record = self.baseline.feed(raw, now, configuration_id, model_version)
        if record is None:
            return False
        self._tare_event("concluida", None, automatic=automatic, record=record)
        return True

    def _stale_sectors(self, summary: dict[str, Any] | None, limit_s: float) -> list[int]:
        """Setores ativos sem atualização completa há mais que limit_s (STALE_SECTOR, item 2)."""
        if summary is None or limit_s <= 0:
            return []
        stale = []
        for s in self.active_blocks:
            info = summary["sectors"].get(str(s))
            age = info.get("age_s") if info else None
            if age is None or age > limit_s:
                stale.append(s)
        return stale

    # -------------------------------------------------------------------- frame

    def process_frame(
        self,
        raw: np.ndarray,
        now: float,
        connected: bool = True,
        acquisition: AcquisitionSample | None = None,
    ) -> dict[str, Any]:
        """Processa um frame RAW e devolve o estado ao vivo. Não depende de thread.

        `acquisition` é a telemetria do leitor no mesmo instante (idade e pacotes por
        setor). Sem ela, STALE_SECTOR não é avaliada.
        """
        p = self.params.snapshot()
        configuration_id = self.configuration_id()
        model = self.models.active()
        model_version = model.model_version if model else "none"
        missing = raw.shape != (HID_ROWS, HID_COLS)
        if missing:
            raw = np.zeros((HID_ROWS, HID_COLS), dtype=np.float64)

        # Telemetria por setor (parecer de 20/10, itens 1 e 2): setor ativo sem atualização
        # completa dentro do limite vira STALE_SECTOR e impede o peso validado
        acq_summary = acquisition.summary if acquisition is not None else None
        self._last_acquisition = acq_summary
        stale = self._stale_sectors(acq_summary, p["stale_sector_s"])
        hardware_ok = connected and not missing

        # Tara (parecer de 20/10, item 5): pedido manual, vigilância da coleta e acúmulo.
        # Estado e movimento são os do frame anterior, para o frame que fecha a tara já
        # sair corrigido pelo baseline novo.
        prev = self._tare_gate or {"state": ReadingState.SEM_CARGA, "movement": False, "total_raw": 0.0}
        gate = {**prev, "now": now, "hardware_ok": hardware_ok, "stale": stale}
        if self._handle_tare(gate, raw, configuration_id, model_version):
            self.state_machine.reset()
            self._raw_window.clear()
            self._full_window.clear()
            self._full_raw_window.clear()
            self._capture_window.clear()
            self._auto_tare_done = True

        corrected_full = self.baseline.correct(raw)
        corrected = corrected_full * self._mask
        raw_measured = raw * self._mask

        # Filtragem temporal, movimento e estabilidade
        state = self.state_machine.update(corrected, now)
        metrics = self.state_machine.metrics
        self._raw_window.append((now, raw_measured.astype(np.float32)))
        while self._raw_window and self._raw_window[0][0] < now - p["stability_window_s"]:
            self._raw_window.popleft()
        self._full_window.append((now, corrected_full.astype(np.float32)))
        while self._full_window and self._full_window[0][0] < now - p["stability_window_s"]:
            self._full_window.popleft()
        self._full_raw_window.append((now, raw.astype(np.float32)))
        while self._full_raw_window and self._full_raw_window[0][0] < now - p["stability_window_s"]:
            self._full_raw_window.popleft()
        frame = WindowFrame(
            raw_mean=window_mean([m for _, m in self._raw_window], raw_measured),
            corrected_mean=window_mean(self.state_machine.window_frames, corrected),
        )

        # Maca vazia contínua (exigida pela tara manual) e condições deste frame para o próximo pedido
        if state == ReadingState.SEM_CARGA and hardware_ok and not stale and not metrics.movement:
            if self._no_load_since is None:
                self._no_load_since = now
        else:
            self._no_load_since = None
        with self._state_lock:
            self._tare_gate = {"now": now, "state": state, "movement": metrics.movement,
                               "hardware_ok": hardware_ok, "stale": stale, "total_raw": metrics.total_raw}
        self._handle_auto_tare(state, now, p, hardware_ok=hardware_ok and not stale)
        self.baseline_return.update(
            state == ReadingState.SEM_CARGA and hardware_ok, now, corrected_full,
            self.baseline.std_matrix if self.baseline.has_baseline else None,
            acquisition.cell_age_s if acquisition is not None else None, p["stale_sector_s"],
        )

        # Modelo de calibração
        estimate: MassEstimate | None = None
        if state != ReadingState.SEM_CARGA and model is not None:
            estimate = model.estimate(frame)

        # Diagnóstico célula a célula e por setor (orientações de 15/10). Roda também
        # com a maca vazia, para medir a massa residual depois da tara.
        summary: dict[str, Any] | None = None
        if isinstance(model, BlockCurveModel) and model.is_ready:
            cells = model.cell_diagnostics(frame.corrected_mean, raw_measured, p["saturation_raw"],
                                           p["low_sensitivity_g_per_raw"])
            summary = model.sector_summary(cells, self.active_blocks)
        totals = summary["totals"] if summary else {}

        # Validação da medição
        n_saturated = saturated_cells(raw_measured, p["saturation_raw"])
        flags = QualityFlags(
            tare_ok=self.baseline.has_baseline,
            baseline_ok=self.baseline.is_baseline_ok(p["baseline_std_max_raw"]),
            signal_stable=metrics.signal_stable,
            movement=metrics.movement,
            saturation=n_saturated > 0,
            missing_cells=missing or not connected,
            out_of_calibration=bool(estimate and estimate.out_of_calibration),
            low_sensitivity=bool(estimate and (estimate.low_sensitivity or totals.get("n_low_sensitivity", 0) > 0)),
            model_valid=bool(estimate and estimate.model_valid and estimate.mass_kg is not None),
            low_data_support=bool(estimate and totals.get("n_low_support", 0) > 0),
            stale_sector=bool(stale),
        )
        state = self.state_machine.set_validated(flags.weight_valid)
        stable_now = state in (ReadingState.ESTAVEL, ReadingState.MEDICAO_VALIDA) and not metrics.movement
        self._capture_window.append((now, stable_now, corrected_full.astype(np.float32)))
        while self._capture_window and self._capture_window[0][0] < now - GAIN_CAPTURE_S:
            self._capture_window.popleft()
        weight_status = WEIGHT_STATUS[state]
        timestamp = datetime.now(timezone.utc).isoformat()

        weight_kg: float | None = None
        if state == ReadingState.MEDICAO_VALIDA and estimate is not None and estimate.mass_kg is not None:
            weight_kg = estimate.mass_kg
            self._last_valid = {
                "weight_kg": round(weight_kg, 2),
                "timestamp": timestamp,
                "stabilization_time_s": _round(metrics.stabilization_time_s, 2),
                "calibration_version": estimate.calibration_version,
            }
        elif state == ReadingState.SEM_CARGA:
            self._last_valid = None

        features = compute_spatial_features(corrected)
        # Mapa de calor e postura com a manta inteira, para posicionar a carga
        display = apply_deadzone(corrected_full, p["deadzone_threshold"])

        # Postura e prevenção de LPP
        self.monitor.update_tolerance(p["posture_tolerance"])
        self.monitor.update_timeout(int(p["posture_timeout_seconds"]))
        alert = self.monitor.update(display)
        posture_info = self.monitor.classify_posture(display)
        max_val = float(np.max(display))
        normalized_map = (display / max_val if max_val > 0 else display).tolist()

        tare_record = self.baseline.record
        if (state == ReadingState.SEM_CARGA and summary is not None and tare_record is not None
                and not self.baseline.is_collecting and hardware_ok and not stale):
            self.residual.update(
                tare_record.tare_id, now, totals["mass_kg"], totals["mass_without_gain_kg"],
                {s["sector"]: s["mass_kg"] for s in summary["sectors"] if s["active"]},
            )
        audit = self.calibration_audit()
        last_valid_kg = self._last_valid["weight_kg"] if self._last_valid else None
        # RAW corrigido agregado e células ativas por setor, na mesma janela de L1…L8
        if summary is not None:
            sector_raw = {str(s["sector"]): s["raw_sum"] for s in summary["sectors"]}
            sector_cells = {str(s["sector"]): s["active_cells"] for s in summary["sectors"]}
        else:
            sector_raw = {str(b): round(float(np.sum(frame.corrected_mean[r])), 2) for b, r in BLOCK_REGIONS.items()}
            sector_cells = {str(b): int(np.count_nonzero(frame.corrected_mean[r] > 0)) for b, r in BLOCK_REGIONS.items()}
        live = {
            "maca_id": self._maca_id,
            "timestamp": timestamp,
            "status": "conectado" if connected else "desconectado",
            "software_version": SOFTWARE_VERSION,
            # Item 17 das orientações
            "weight_kg": _round(weight_kg, 2),
            "weight_status": weight_status,
            "stable": metrics.signal_stable,
            "stabilization_time_s": _round(metrics.stabilization_time_s, 2),
            "movement": metrics.movement,
            "measurement_quality": flags.as_dict(),
            "calibration_version": estimate.calibration_version if estimate else (model.calibration_version if model else None),
            "model_version": model_version,
            "model_kind": model.kind if model else None,
            "configuration_id": configuration_id,
            "active_blocks": self.active_blocks,
            "tare_id": tare_record.tare_id if tare_record else None,
            "tare_in_progress": self.baseline.is_collecting,
            "last_valid": self._last_valid,
            "last_valid_weight_kg": last_valid_kg,
            "tare": {
                "tare_id": tare_record.tare_id,
                "timestamp_tare": tare_record.timestamp_tare,
                "baseline_sum": round(tare_record.baseline_sum, 1),
                "automatic": tare_record.automatic,
            } if tare_record else None,
            # Estimativa sem validação, só para análise experimental
            "estimated_mass_kg": _round(estimate.mass_kg if estimate else None, 2),
            # Estado e métricas
            "state_id": state.value,
            "state_label": STATE_LABELS[state],
            "stable_progress_pct": round(metrics.stable_progress_pct, 1),
            "is_locked": weight_status == "VALID",
            "locked_weight_kg": last_valid_kg or 0.0,
            "total_raw": round(metrics.total_raw, 1),
            "raw_mean": round(metrics.window_mean, 1),
            "raw_std": round(metrics.window_std, 2),
            "delta": _round(metrics.delta, 2),
            "movement_index": _round(metrics.movement_index, 4),
            "active_cells": features.active_cells,
            "center_x": round(features.center_x, 2),
            "center_y": round(features.center_y, 2),
            "spatial_dispersion": round(features.spatial_dispersion, 2),
            "saturated_cells": n_saturated,
            # Diagnóstico da calibração (orientações de 15/10, item 8). L_s sem o ganho α_s;
            # com a maca vazia valem como massa residual.
            "n_digital_saturation": n_saturated,
            "sector_local_mass_kg": {str(s["sector"]): s["local_mass_kg"] for s in summary["sectors"]} if summary else None,
            "sector_mass_kg": {str(s["sector"]): s["mass_kg"] for s in summary["sectors"]} if summary else None,
            "mass_without_gain_kg": totals.get("mass_without_gain_kg"),
            "n_active_cells": totals.get("n_active_cells"),
            "n_out_of_calibration": totals.get("n_out_of_calibration"),
            "out_of_calibration_by_sector": (
                {str(s["sector"]): s["cells_out_of_calibration"] for s in summary["sectors"]} if summary else None),
            "out_of_calibration_mass_pct": totals.get("out_of_calibration_mass_pct"),
            "n_low_sensitivity": totals.get("n_low_sensitivity"),
            "n_low_support": totals.get("n_low_support"),
            "hysteresis_up_minus_down_kg": totals.get("hysteresis_up_minus_down_kg"),
            "sector_corrected_raw": sector_raw,
            "sector_active_cells": sector_cells,
            "residual_mass_empty": self.residual.snapshot(),
            "calibration_alerts": audit["alerts"] if audit else [],
            # Aquisição por setor (parecer de 20/10, itens 1 a 3)
            "stale_sectors": stale,
            "acquisition": acq_summary,
            # Postura e interface
            "heatmap": normalized_map,
            "cop": {"row": round(features.center_y, 2), "col": round(features.center_x, 2)},
            "static_seconds": int(self.monitor.elapsed_seconds),
            "is_alert": alert,
            "posture_info": posture_info,
        }

        with self._state_lock:
            self._live_state = live
        if self.diag_log is not None:
            self.diag_log.maybe_write(now, live)

        if self.recorder.active:
            self.recorder.record(
                raw,
                {
                    "timestamp": timestamp,
                    "configuration_id": configuration_id,
                    "tare_id": live["tare_id"],
                    "state": state.value,
                    "stability_flag": metrics.signal_stable,
                    "movement_flag": metrics.movement,
                    "saturation_flag": flags.saturation,
                    "saturated_cells": n_saturated,
                    "raw_sum": float(np.sum(raw)),
                    "total_raw": metrics.total_raw,
                    "raw_mean": metrics.window_mean,
                    "raw_std": metrics.window_std,
                    "delta": metrics.delta,
                    "movement_index": metrics.movement_index,
                    "active_cells": features.active_cells,
                    "center_x": features.center_x,
                    "center_y": features.center_y,
                    "spatial_dispersion": features.spatial_dispersion,
                    "stabilization_time_s": metrics.stabilization_time_s,
                    "calibration_version": live["calibration_version"],
                    "model_version": model_version,
                    "estimated_mass": estimate.mass_kg if estimate else None,
                    "prediction_valid": flags.weight_valid,
                    "weight_status": weight_status,
                    "stale_sectors": ";".join(str(b) for b in stale) or None,
                    "max_sector_age_s": max(
                        (i["age_s"] for i in acq_summary["sectors"].values() if i["age_s"] is not None),
                        default=None) if acq_summary else None,
                },
                self.baseline.mean_matrix if tare_record else None,
                self.baseline.std_matrix if tare_record else None,
            )

        self._persist_events(now, live, posture_info, alert, max_val)
        return live

    # ----------------------------------------------------------------- auxiliares

    def _handle_auto_tare(self, state: ReadingState, now: float, p: dict[str, Any], hardware_ok: bool) -> None:
        """Tara automática uma vez por período de maca vazia.

        Frames zerados de manta desconectada e setores sem atualização não contam como
        maca vazia. O cancelamento por carga, movimento ou setor parado fica em _handle_tare.
        """
        if not hardware_ok:
            self._empty_since = None
            return
        if state != ReadingState.SEM_CARGA:
            self._empty_since = None
            self._auto_tare_done = False
            return
        if self._auto_tare_done or self.baseline.is_collecting:
            return
        if self._empty_since is None:
            self._empty_since = now
            return
        if now - self._empty_since >= p["auto_rezero_empty_s"]:
            self.baseline.start_tare(now, p["t_tare_s"], automatic=True)
            self._tare_event("iniciada", None, automatic=True)
            self._auto_tare_done = True

    def _persist_events(
        self,
        now: float,
        live: dict[str, Any],
        posture_info: dict[str, Any],
        alert: bool,
        max_val: float,
    ) -> None:
        # Telemetria a cada 60 s, só com peso validado
        if now - self._last_telemetry_time >= 60.0 and live["weight_status"] == "VALID":
            self._last_telemetry_time = now
            asym = posture_info.get("asymmetry_pct", 0.0)
            self._db.enqueue_telemetry(
                timestamp=live["timestamp"],
                maca_id=self._maca_id,
                peso_kg=live["weight_kg"],
                indice_postural=round(asym / 100.0, 3) if asym else 0.0,
                tempo_estatico_seg=int(self.monitor.elapsed_seconds),
                status_alerta=alert,
                payload={
                    "relief_score": posture_info.get("relief_score", 100),
                    "weight_status": live["weight_status"],
                    "stabilization_time_s": live["stabilization_time_s"],
                    "calibration_version": live["calibration_version"],
                    "configuration_id": live["configuration_id"],
                },
            )

        # Eventos de transição postural
        curr_pos = posture_info.get("posture", "Leito Livre")
        if self._last_recorded_posture is None:
            self._last_recorded_posture = curr_pos
            self._last_posture_change_time = now
        elif curr_pos != self._last_recorded_posture and curr_pos != "Leito Livre":
            dur_seg = int(now - self._last_posture_change_time)
            if dur_seg >= 15:
                self._db.record_posture_event(
                    timestamp=live["timestamp"],
                    maca_id=self._maca_id,
                    postura_detectada=curr_pos,
                    postura_anterior=self._last_recorded_posture,
                    duracao_anterior_seg=dur_seg,
                    regiao_pico=posture_info.get("asymmetry_label"),
                    pico_pct=round(max_val, 1),
                    houve_alerta=alert,
                )
                self._last_recorded_posture = curr_pos
                self._last_posture_change_time = now

    def _loop(self) -> None:
        while self._is_running:
            try:
                connected = self._reader.is_connected()
                raw = self._reader.read_frame()
                now = time.monotonic()
                live = self.process_frame(raw, now, connected, self._reader.acquisition_sample(now))
                if self._broadcast_callback is not None:
                    try:
                        self._broadcast_callback(live)
                    except Exception as b_err:
                        logger.debug("Falha no broadcast WebSocket: %s", b_err)
            except Exception as err:
                logger.exception("Erro no loop do AcquisitionWorker: %s", err)
                with self._state_lock:
                    self._live_state["status"] = f"erro: {err}"
                time.sleep(0.5)
