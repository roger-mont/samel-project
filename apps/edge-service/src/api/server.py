"""Servidor FastAPI Edge para Maca Hospitalar Inteligente.

Fornece:
- Endpoints REST para ingestão, analytics (4h em 30min, eventos, resumo de hoje) e calibração/tara.
- 'Via expressa' WebSocket (/ws/telemetry) para streaming em tempo real do mapa de calor e KPIs.
- Gerenciamento dos daemons em background:
  - AcquisitionWorker: Leitura 24/7 de hardware USB HID / Serial, física, postura e persistência local.
  - SyncWorker: Daemon Store-and-Forward para sincronização com o servidor central.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastapi.responses import FileResponse

from core.config import LOG_LEVEL, MACA_ID, config_manager
from database.db_local import LocalDatabase
from math_engine.gain_fit import (
    MODES as GAIN_MODES, POSITIONS, curves_sha1, fit_gains, fit_report, leave_one_out, position_report, write_gains,
)
from math_engine.mass_models import BlockCurveModel
from math_engine.model_comparison import MODES, ComparisonCache, ModelComparison
from workers.sync_worker import SyncWorker
from workers.acquisition_worker import AcquisitionWorker

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("edge_server")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_comparison_cache = ComparisonCache()


# -----------------------------------------------------------------------------
# Modelos Pydantic (Clean Code & Validação)
# -----------------------------------------------------------------------------

class SystemConfigPayload(BaseModel):
    maca_id: str | None = Field(default=None, description="Identificador único da maca")
    central_api_url: str | None = Field(default=None, description="URL do servidor central do hospital")
    edge_api_token: str | None = Field(default=None, description="Token de autenticação")
    sync_interval_sec: float | None = Field(default=None, ge=5.0, le=3600.0, description="Intervalo de sync em segundos")
    sync_batch_size: int | None = Field(default=None, ge=1, le=500, description="Tamanho do lote de envio")
    retention_days: int | None = Field(default=None, ge=1, le=90, description="Dias de retenção no SQLite local")
    log_level: str | None = Field(default=None, description="Nível de log (DEBUG, INFO, WARNING, ERROR)")


class TelemetryPayload(BaseModel):
    timestamp: str | None = Field(default=None, description="ISO8601 UTC timestamp")
    maca_id: str = Field(default=MACA_ID, description="Identificador único da maca")
    peso_kg: float | None = Field(default=None, ge=0.0, le=500.0)
    indice_postural: float | None = Field(default=None)
    tempo_estatico_seg: int = Field(default=0, ge=0)
    status_alerta: bool = Field(default=False)
    payload: dict[str, Any] | None = Field(
        default=None, description="Dados adicionais como mapa de pressão"
    )


class PostureEventPayload(BaseModel):
    timestamp: str | None = Field(default=None, description="ISO8601 UTC timestamp")
    maca_id: str = Field(default=MACA_ID, description="Identificador único da maca")
    sessao_id: str | None = Field(default=None)
    postura_detectada: str = Field(
        ..., description="Ex: Decúbito Dorsal, Decúbito Lateral Direito"
    )
    postura_anterior: str | None = Field(default=None)
    duracao_anterior_seg: int = Field(default=0, ge=0)
    regiao_pico: str | None = Field(default=None)
    pico_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    area_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    indice_dist: float | None = Field(default=None)
    houve_alerta: bool = Field(default=False)


class CalibrationUpdatePayload(BaseModel):
    param_name: str = Field(..., description="Nome do parâmetro (ver GET /api/v1/calibration/params)")
    value: float = Field(..., description="Novo valor do parâmetro")
    persist: bool = Field(default=False, description="Grava o valor em config.json (bloco 'acquisition')")


class GainPointPayload(BaseModel):
    known_mass_kg: float = Field(..., gt=0.0, le=500.0, description="Massa real sobre a maca, em kg")
    note: str | None = Field(default=None, max_length=80, description="Ex.: placa no centro, setores 2 e 7")
    position: str | None = Field(default=None, description="Posição da carga: " + " | ".join(POSITIONS))


class GainFitPayload(BaseModel):
    mode: str = Field(default="global", description="global | setor")
    lam: float = Field(default=1.0, ge=0.0, le=1000.0, description="λ do modo setor (puxa cada α para o global)")


class ExperimentStartPayload(BaseModel):
    operator: str | None = Field(default=None, description="Quem conduz o ensaio")
    description: str | None = Field(default=None, description="Objetivo e montagem do ensaio")
    contact_area: str | None = Field(default=None, description="Ex.: penetrador de 27 mm (moeda de R$ 1)")


class ExperimentMarkPayload(BaseModel):
    cycle_id: int | None = Field(default=None, ge=0)
    known_mass_g: float | None = Field(default=None, ge=0.0)
    loading_direction: str | None = Field(default=None, description="UP | DOWN | UNKNOWN")
    target_cell: str | None = Field(default=None, description="Célula ensaiada, 'linha x coluna' (ex.: 12x34); '' limpa")


# -----------------------------------------------------------------------------
# Gerenciador de Conexões WebSocket ("Via Expressa")
# -----------------------------------------------------------------------------

class ConnectionManager:
    """Gerencia conexões ativas de WebSocket para broadcast de telemetria e alertas."""

    def __init__(self) -> None:
        self._active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._active_connections.add(websocket)
        logger.info(
            "Cliente WebSocket conectado. Total ativos: %d",
            len(self._active_connections),
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._active_connections.discard(websocket)
        logger.info(
            "Cliente WebSocket desconectado. Total ativos: %d",
            len(self._active_connections),
        )

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Transmite dados para todos os clientes conectados sem bloquear."""
        if not self._active_connections:
            return

        dead_connections: list[WebSocket] = []
        payload_str = json.dumps(message)

        async with self._lock:
            connections = list(self._active_connections)

        for connection in connections:
            try:
                await connection.send_text(payload_str)
            except Exception:
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                for dead in dead_connections:
                    self._active_connections.discard(dead)

    async def emit_alert(self, alert_data: dict[str, Any]) -> None:
        """Disparo de emergência de alta prioridade para o WebSocket."""
        packet = {
            "type": "EMERGENCY_ALERT",
            "priority": "HIGH",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": alert_data,
        }
        await self.broadcast(packet)


ws_manager = ConnectionManager()
_main_loop: asyncio.AbstractEventLoop | None = None


def _threadsafe_ws_broadcast(data: dict[str, Any]) -> None:
    """Callback seguro para o AcquisitionWorker disparar broadcasts no loop assíncrono."""
    global _main_loop
    if _main_loop is not None and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(data), _main_loop)


# -----------------------------------------------------------------------------
# Ciclo de Vida da Aplicação (FastAPI Lifespan)
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Inicializa banco SQLite, AcquisitionWorker e SyncWorker."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("Iniciando Módulo Edge FastAPI Nativo 24/7...")

    db = LocalDatabase()
    sync_worker = SyncWorker(db=db)
    acquisition_worker = AcquisitionWorker(
        db=db,
        maca_id=MACA_ID,
        broadcast_callback=_threadsafe_ws_broadcast,
    )

    app.state.db = db
    app.state.sync_worker = sync_worker
    app.state.acquisition_worker = acquisition_worker

    sync_worker.start()
    acquisition_worker.start()

    yield

    logger.info("Encerrando Módulo Edge FastAPI...")
    acquisition_worker.stop()
    await sync_worker.stop()


app = FastAPI(
    title="Módulo Edge - Maca Hospitalar Inteligente",
    description="Serviço local de alta resiliência com aquisição contínua, Store-and-Forward e WebSocket.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_db(app_instance: FastAPI) -> LocalDatabase:
    return app_instance.state.db


def _get_worker(app_instance: FastAPI) -> SyncWorker:
    return app_instance.state.sync_worker


def _get_acq(app_instance: FastAPI) -> AcquisitionWorker:
    return app_instance.state.acquisition_worker


# -----------------------------------------------------------------------------
# Endpoints de Telemetria ao Vivo e Controle de Calibração
# -----------------------------------------------------------------------------

@app.get("/api/v1/telemetry/live", tags=["Telemetria"])
def get_live_telemetry(request: Request) -> dict[str, Any]:
    """Retorna snapshot instantâneo do estado da maca."""
    acq = _get_acq(request.app)
    return acq.get_live_snapshot()


@app.post("/api/v1/calibration/tare", tags=["Calibração"])
def trigger_tare(request: Request) -> dict[str, Any]:
    """Inicia uma tara de T_TARE segundos. A maca precisa estar vazia."""
    acq = _get_acq(request.app)
    acq.trigger_tare()
    return {"status": "ok", "message": "Amostragem de tara iniciada", "tare": acq.tare_status()}


@app.get("/api/v1/calibration/tare", tags=["Calibração"])
def get_tare_status(request: Request) -> dict[str, Any]:
    """Tara em vigor (tare_id, horário, média e desvio do baseline) e se há coleta em curso."""
    return _get_acq(request.app).tare_status()


@app.get("/api/v1/calibration/params", tags=["Calibração"])
def get_calibration_params(request: Request) -> dict[str, Any]:
    """Retorna os parâmetros de aquisição e estabilidade em vigor."""
    acq = _get_acq(request.app)
    return {**acq.params.snapshot(), "configuration_id": acq.configuration_id()}


@app.post("/api/v1/calibration/params", tags=["Calibração"])
def update_calibration_param(payload: CalibrationUpdatePayload, request: Request) -> dict[str, Any]:
    """Atualiza parâmetro em runtime (e opcionalmente em config.json)."""
    acq = _get_acq(request.app)
    try:
        acq.update_param(payload.param_name, payload.value, persist=payload.persist)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return {"status": "ok", "params": acq.params.snapshot(), "configuration_id": acq.configuration_id()}


# -----------------------------------------------------------------------------
# Endpoints de Experimentos (gravação do RAW completo)
# -----------------------------------------------------------------------------

@app.get("/api/v1/experiments", tags=["Experimentos"])
def get_experiment_status(request: Request) -> dict[str, Any]:
    return _get_acq(request.app).recorder.status()


@app.post("/api/v1/experiments/start", tags=["Experimentos"])
def start_experiment(payload: ExperimentStartPayload, request: Request) -> dict[str, Any]:
    """Começa a gravar o RAW de todos os frames, com parâmetros, modelo e tara em vigor."""
    acq = _get_acq(request.app)
    model = acq.models.active()
    meta = {
        **payload.model_dump(),
        "maca_id": acq._maca_id,
        "configuration_id": acq.configuration_id(),
        "params": acq.params.snapshot(),
        "model": acq.models.describe(),
        "model_version": model.model_version if model else None,
        "calibration_version": model.calibration_version if model else None,
        "tare": acq.tare_status()["record"],
    }
    try:
        return acq.recorder.start(meta)
    except RuntimeError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err


@app.post("/api/v1/experiments/mark", tags=["Experimentos"])
def mark_experiment(payload: ExperimentMarkPayload, request: Request) -> dict[str, Any]:
    """Registra, a partir de agora, ciclo, massa conhecida, sentido de carga e célula ensaiada."""
    try:
        return _get_acq(request.app).recorder.mark(
            cycle_id=payload.cycle_id,
            known_mass_g=payload.known_mass_g,
            loading_direction=payload.loading_direction,
            target_cell=payload.target_cell,
        )
    except RuntimeError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.post("/api/v1/experiments/stop", tags=["Experimentos"])
def stop_experiment(request: Request) -> dict[str, Any]:
    return _get_acq(request.app).recorder.stop()


# -----------------------------------------------------------------------------
# Endpoints de Configurações do Sistema
# -----------------------------------------------------------------------------

@app.get("/api/v1/system/config", tags=["Configuração"])
def get_system_configuration() -> dict[str, Any]:
    """Retorna o snapshot das configurações operacionais vigentes."""
    return config_manager.snapshot()


@app.post("/api/v1/system/config", tags=["Configuração"])
def update_system_configuration(payload: SystemConfigPayload, request: Request) -> dict[str, Any]:
    """Atualiza e persiste configurações do sistema (MACA_ID, URLs, intervalos)."""
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    new_snapshot = config_manager.update_and_save(updates)

    # Sincroniza workers em runtime
    if "maca_id" in updates:
        new_id = str(updates["maca_id"])
        acq = _get_acq(request.app)
        if acq:
            acq._maca_id = new_id
            with acq._state_lock:
                acq._live_state["maca_id"] = new_id

    return {"status": "ok", "config": new_snapshot}


# -----------------------------------------------------------------------------
# Endpoints de Ingestão REST (Store-and-Forward)
# -----------------------------------------------------------------------------

@app.post(
    "/api/v1/telemetry",
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestão"],
    summary="Enfileira telemetria para persistência local e sync remoto",
)
async def ingest_telemetry(
    payload: TelemetryPayload,
    request: Request,
) -> dict[str, Any]:
    db = _get_db(request.app)
    record_id = db.enqueue_telemetry(
        timestamp=payload.timestamp or datetime.now(timezone.utc).isoformat(),
        maca_id=payload.maca_id,
        peso_kg=payload.peso_kg,
        indice_postural=payload.indice_postural,
        tempo_estatico_seg=payload.tempo_estatico_seg,
        status_alerta=payload.status_alerta,
        payload=payload.payload,
    )

    # Broadcast na via expressa WebSocket
    await ws_manager.broadcast(
        {
            "type": "TELEMETRY_HOT_DATA",
            "maca_id": payload.maca_id,
            "peso_kg": payload.peso_kg,
            "indice_postural": payload.indice_postural,
            "tempo_estatico_seg": payload.tempo_estatico_seg,
            "status_alerta": payload.status_alerta,
        }
    )

    if payload.status_alerta:
        await ws_manager.emit_alert(
            {
                "maca_id": payload.maca_id,
                "motivo": "Tempo estático excessivo - risco de LPP",
                "tempo_estatico_seg": payload.tempo_estatico_seg,
            }
        )

    return {"status": "enqueued", "id": record_id, "table": "telemetry_queue"}


@app.post(
    "/api/v1/events/posture",
    status_code=status.HTTP_201_CREATED,
    tags=["Eventos"],
    summary="Registra evento de alteração postural para auditoria clínica",
)
async def record_event(
    payload: PostureEventPayload,
    request: Request,
) -> dict[str, Any]:
    db = _get_db(request.app)
    event_id = db.record_posture_event(
        timestamp=payload.timestamp or datetime.now(timezone.utc).isoformat(),
        maca_id=payload.maca_id,
        sessao_id=payload.sessao_id,
        postura_detectada=payload.postura_detectada,
        postura_anterior=payload.postura_anterior,
        duracao_anterior_seg=payload.duracao_anterior_seg,
        regiao_pico=payload.regiao_pico,
        pico_pct=payload.pico_pct,
        area_pct=payload.area_pct,
        indice_dist=payload.indice_dist,
        houve_alerta=payload.houve_alerta,
    )

    return {"status": "recorded", "id": event_id, "table": "posture_events"}


# -----------------------------------------------------------------------------
# Endpoints de Analítica para o Dashboard
# -----------------------------------------------------------------------------

@app.get("/api/v1/analytics/charts", tags=["Analytics"])
def get_charts_analytics(window_hours: int = 4, request: Request = None) -> dict[str, Any]:
    """Retorna séries temporais consolidadas em 9 blocos de 30min para os 3 gráficos da UI."""
    db = _get_db(request.app)
    return db.get_charts_analytics(window_hours=window_hours)


@app.get("/api/v1/analytics/events", tags=["Analytics"])
def get_recent_events(limit: int = 5, request: Request = None) -> list[dict[str, Any]]:
    """Retorna os últimos eventos posturais com descrições amigáveis."""
    db = _get_db(request.app)
    return db.get_recent_events(limit=limit)


@app.get("/api/v1/analytics/daily-summary", tags=["Analytics"])
def get_daily_summary(request: Request = None) -> dict[str, Any]:
    """Retorna o resumo diário do dia civil corrente (a partir das 00:00)."""
    db = _get_db(request.app)
    return db.get_daily_summary()


# -----------------------------------------------------------------------------
# WebSocket - Streaming em Tempo Real ("Via Expressa")
# -----------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket) -> None:
    """Canal bidirecional de streaming para a UI ou monitores externos."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type") or msg.get("action")
                if msg_type == "PING":
                    await websocket.send_text(json.dumps({"type": "PONG", "timestamp": datetime.now(timezone.utc).isoformat()}))
                elif msg_type == "tare":
                    acq = _get_acq(websocket.app)
                    acq.trigger_tare()
                    await websocket.send_text(json.dumps({"type": "ACK", "action": "tare"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as ex:
        logger.warning("Exceção na conexão WebSocket: %s", ex)
        await ws_manager.disconnect(websocket)


# -----------------------------------------------------------------------------
# Healthcheck e Status do Sistema
# -----------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
@app.get("/api/v1/health", tags=["Sistema"])
def health_check(request: Request) -> dict[str, Any]:
    """Endpoint de verificação de integridade operacional do Edge Service."""
    worker = _get_worker(request.app)
    acq = _get_acq(request.app)
    db = _get_db(request.app)

    db_healthy = True
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception:
        db_healthy = False

    return {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database_connected": db_healthy,
        "acquisition_running": acq._is_running if acq else False,
        "hardware_connected": acq._reader.is_connected() if acq and acq._reader else False,
        "sync_worker_running": worker.is_running if worker else False,
        "circuit_breaker_state": worker.circuit_breaker.state.value if worker else "CLOSED",
    }


@app.get("/api/v1/status", tags=["Sistema"])
def get_system_status(request: Request) -> dict[str, Any]:
    """Retorna detalhes do status operacional, filas e Circuit Breaker."""
    worker = _get_worker(request.app)
    acq = _get_acq(request.app)
    db = _get_db(request.app)
    return {
        "maca_id": MACA_ID,
        "database": {"connected": True, "path": db.db_path},
        "sync_worker": {
            "is_running": worker.is_running if worker else False,
            "circuit_breaker": worker.circuit_breaker.state.value if worker else "CLOSED",
        },
        "circuit_breaker": worker.circuit_breaker.state.value if worker else "CLOSED",
        "sync_worker_active": worker.is_running if worker else False,
        "acquisition_active": acq._is_running if acq else False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/calibration/model", tags=["Calibração"])
def get_model_info(request: Request) -> dict[str, Any]:
    """Modelos de massa disponíveis e qual está ativo."""
    acq = _get_acq(request.app)
    return {"maca_id": MACA_ID, **acq.models.describe()}


@app.get("/api/v1/calibration/curves", tags=["Calibração"])
def get_calibration_curves(request: Request) -> dict[str, Any]:
    """Curvas em vigor por setor: fontes, pontos medidos, ajuste isotônico, curva amostrada e ganhos α_s."""
    acq = _get_acq(request.app)
    model = acq.models.models.get("block_curves")
    if not isinstance(model, BlockCurveModel) or not model.is_ready:
        raise HTTPException(status_code=404, detail="block_curves.json ausente ou incompleto")
    audit = acq.calibration_audit()
    definition = model.raw_definition
    blocks = definition["blocks"]
    sources = definition.get("sources", [])
    sectors = {}
    for bid_str, curve in blocks.items():
        sectors[bid_str] = {
            "source": curve.get("source", "cells"),
            "cells": curve.get("cells", []),
            "sources": [s for s in sources if str(s.get("block")) == bid_str],
            "fit": curve.get("fit", definition.get("fit", [])),
            "knots": curve["params"],
            "raw_min_valid": curve["raw_min_valid"],
            "raw_max_valid": curve["raw_max_valid"],
            "mass_max_valid_kg": curve["mass_max_valid"],
            "low_sensitivity_raw": curve.get("low_sensitivity_raw"),
            "sensitivity_g_per_raw": curve.get("sensitivity_g_per_raw", []),
            "validation": curve.get("validation"),
            "audit": (audit or {}).get("sectors", {}).get(bid_str),
            "down_curve": curve.get("down_curve_not_used"),
            "gain": model.gains.get(int(bid_str), 1.0),
            "samples_raw_mass_g": model.curve_samples(int(bid_str)),
        }
    curve_keys = ("kind", "params", "raw_min_valid", "raw_max_valid", "mass_max_valid")
    return {
        "calibration_version": definition["calibration_version"],
        "model_version": definition.get("model_version"),
        "provisional": definition.get("provisional", False),
        "method": definition.get("method"),
        "notes": definition.get("notes", []),
        "sector_layout": definition.get("sector_layout"),
        "aggregation": definition.get("aggregation"),
        "gains_calibrated": model.gains_calibrated,
        "identical_in_all_sectors": len({json.dumps({k: b.get(k) for k in curve_keys}, sort_keys=True)
                                         for b in blocks.values()}) == 1,
        "calibration_alerts": (audit or {}).get("alerts", []),
        "sectors": sectors,
    }


@app.get("/api/v1/calibration/extraction", tags=["Calibração"])
def get_calibration_extraction(request: Request) -> dict[str, Any]:
    """Extração da janela atual por setor e por célula, com a curva de cada setor."""
    data = _get_acq(request.app).extraction()
    if data is None:
        raise HTTPException(status_code=409, detail="Modelo ativo não é a curva por bloco")
    return data


def _comparison(request: Request) -> tuple[AcquisitionWorker, ModelComparison]:
    acq = _get_acq(request.app)
    model = acq.models.models.get("block_curves")
    if not isinstance(model, BlockCurveModel) or not model.is_ready:
        raise HTTPException(status_code=404, detail="block_curves.json ausente ou incompleto")
    return acq, _comparison_cache.get(model.raw_definition, model.gains)


@app.get("/api/v1/calibration/compare", tags=["Calibração"])
def get_model_comparison(request: Request) -> dict[str, Any]:
    """Massa da janela atual por todos os modelos candidatos (curvas e ML), só para diagnóstico.

    O peso clínico continua vindo só do modelo em vigor. Os candidatos são treinados com
    os pontos de block_curves.json por setor ("sector") e com todas as células ("global").
    """
    acq, comparison = _comparison(request)
    live = acq.get_live_snapshot()
    return {
        "timestamp": live.get("timestamp"),
        "weight_status": live.get("weight_status"),
        "weight_kg": live.get("weight_kg"),
        "estimated_mass_kg": live.get("estimated_mass_kg"),
        "stable_progress_pct": live.get("stable_progress_pct"),
        "active_blocks": acq.active_blocks,
        "calibration_version": comparison.version,
        "models": comparison.evaluate(acq.window_corrected(), acq.active_blocks),
    }


@app.get("/api/v1/calibration/compare/curves", tags=["Calibração"])
def get_model_comparison_curves(request: Request, sector: int = 2, mode: str = "sector") -> dict[str, Any]:
    """Curvas RAW → massa local de todos os modelos candidatos para um setor."""
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode deve ser um de {MODES}")
    _, comparison = _comparison(request)
    if sector not in comparison.raw_max_valid:
        raise HTTPException(status_code=422, detail="setor deve estar entre 1 e 8")
    return {"sector": sector, "mode": mode, "models": comparison.curves(mode, sector)}


# -----------------------------------------------------------------------------
# Ajuste de ganho α_s (massa local → massa total) pela página /calibracao
# -----------------------------------------------------------------------------

def _block_model(acq: AcquisitionWorker) -> BlockCurveModel:
    model = acq.models.models.get("block_curves")
    if not isinstance(model, BlockCurveModel) or not model.is_ready:
        raise HTTPException(status_code=404, detail="block_curves.json ausente ou incompleto")
    return model


def _gain_points(acq: AcquisitionWorker, model: BlockCurveModel) -> list[dict[str, Any]]:
    """Pontos registrados com as massas locais recalculadas pelas curvas em vigor.

    Setores fora da medição entram com massa local zero, como na massa total.
    """
    gains = model.gains
    out = []
    for p in acq.gain_points.list():
        corrected = acq.gain_points.corrected(p)
        local = model.sector_local_sums(corrected)
        local = {b: (v if b in acq.active_blocks else 0.0) for b, v in local.items()}
        outside = model.sector_out_of_calibration(corrected)
        current = sum(gains[b] * v for b, v in local.items())
        view = {k: v for k, v in p.items() if k != "corrected"}
        view.update({
            "local_kg": {str(b): round(v, 4) for b, v in local.items()},
            "local_total_kg": round(sum(local.values()), 4),
            "current_estimate_kg": round(current, 4),
            "current_error_kg": round(current - p["known_mass_kg"], 4),
            "cells_out_of_calibration": sum(outside[b] for b in acq.active_blocks),
        })
        out.append(view)
    return out


def _gain_fit(acq: AcquisitionWorker, model: BlockCurveModel, payload: GainFitPayload) -> dict[str, Any]:
    if payload.mode not in GAIN_MODES:
        raise HTTPException(status_code=422, detail=f"mode deve ser um de {GAIN_MODES}")
    points = _gain_points(acq, model)
    if len(points) < 2:
        raise HTTPException(status_code=400, detail="Registre pelo menos 2 pontos com massa conhecida antes de calcular.")
    blocks = sorted(model.gains)
    L = np.array([[p["local_kg"][str(b)] for b in blocks] for p in points])
    M = np.array([p["known_mass_kg"] for p in points])
    empty = [i + 1 for i, row in enumerate(L) if row.sum() <= 0]
    if empty:
        raise HTTPException(status_code=400, detail=f"Ponto(s) {empty} com massa local zero: confira a tara e apague esses pontos.")
    alpha = fit_gains(L, M, payload.mode, payload.lam)
    loo = leave_one_out(L, M, payload.mode, payload.lam)
    warnings = []
    if payload.mode == "setor" and len(points) < len(acq.active_blocks):
        warnings.append(f"{len(points)} pontos para {len(acq.active_blocks)} ganhos: os α por setor ficam perto do α global.")
    if any(p["cells_out_of_calibration"] for p in points):
        warnings.append("Há pontos com células acima da faixa calibrada; a massa delas fica cortada e puxa o α para cima.")
    total = L.sum(axis=1)
    return {
        "mode": payload.mode,
        "lam": payload.lam if payload.mode == "setor" else None,
        "alpha": {str(b): round(float(a), 5) for b, a in zip(blocks, alpha)},
        "alpha_global": round(float(total @ M / (total @ total)), 5),
        "current_gains": {str(b): g for b, g in model.gains.items()},
        "report": fit_report(L, M, alpha, payload.mode, payload.lam),
        "points": [{"id": p["id"], "known_mass_kg": p["known_mass_kg"], "without_gain_kg": round(float(s), 4),
                    "current_kg": p["current_estimate_kg"], "fitted_kg": round(float(f), 4),
                    "leave_one_out_kg": None if o is None else round(o, 4)}
                   for p, s, f, o in zip(points, total, L @ alpha, loo)],
        "warnings": warnings,
    }


def _gain_state(acq: AcquisitionWorker, model: BlockCurveModel) -> dict[str, Any]:
    aggregation = model.raw_definition.get("aggregation") or {}
    fit = aggregation.get("fit") or None
    points = _gain_points(acq, model)
    return {
        "gains": {str(b): g for b, g in model.gains.items()},
        "calibrated": model.gains_calibrated,
        # Item 3.9: até os ensaios de carga distribuída e de posição, o α é provisório
        "provisional": True,
        "fit": fit,
        "fit_matches_curves": bool(fit) and fit.get("curves_sha1") == curves_sha1(model.raw_definition),
        "calibration_version": model.calibration_version,
        "active_blocks": acq.active_blocks,
        "tare_id": acq.baseline.record.tare_id if acq.baseline.record else None,
        "positions": list(POSITIONS),
        "points": points,
        "position_report": position_report(points),
    }


@app.get("/api/v1/calibration/gain", tags=["Calibração"])
def get_gain_state(request: Request) -> dict[str, Any]:
    """Ganhos α_s em vigor e os pontos registrados para o ajuste (massas locais pelas curvas atuais)."""
    acq = _get_acq(request.app)
    return _gain_state(acq, _block_model(acq))


@app.post("/api/v1/calibration/gain/points", tags=["Calibração"])
def add_gain_point(payload: GainPointPayload, request: Request) -> dict[str, Any]:
    """Registra a massa real e a média estável dos últimos segundos da manta inteira."""
    acq = _get_acq(request.app)
    model = _block_model(acq)
    if acq.baseline.record is None:
        raise HTTPException(status_code=409, detail="Faça a tara com a maca vazia antes de registrar pontos.")
    capture = acq.stable_capture()
    if capture is None:
        raise HTTPException(status_code=409, detail="A leitura ainda não está estável. Espere o estado ficar Estável ou Válido.")
    position = (payload.position or "").strip().lower() or None
    if position is not None and position not in POSITIONS:
        raise HTTPException(status_code=422, detail=f"position deve ser um de {POSITIONS}")
    corrected, frames, seconds = capture
    live = acq.get_live_snapshot()
    acq.gain_points.add(payload.known_mass_kg, corrected, {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (payload.note or "").strip() or None,
        "position": position,
        "frames": frames,
        "seconds": round(seconds, 2),
        "tare_id": acq.baseline.record.tare_id,
        "calibration_version": model.calibration_version,
        "estimated_mass_kg": live.get("estimated_mass_kg"),
    })
    return _gain_state(acq, model)


@app.delete("/api/v1/calibration/gain/points/{point_id}", tags=["Calibração"])
def delete_gain_point(point_id: str, request: Request) -> dict[str, Any]:
    acq = _get_acq(request.app)
    if not acq.gain_points.delete(point_id):
        raise HTTPException(status_code=404, detail="Ponto não encontrado")
    return _gain_state(acq, _block_model(acq))


@app.delete("/api/v1/calibration/gain/points", tags=["Calibração"])
def clear_gain_points(request: Request) -> dict[str, Any]:
    acq = _get_acq(request.app)
    acq.gain_points.clear()
    return _gain_state(acq, _block_model(acq))


@app.post("/api/v1/calibration/gain/fit", tags=["Calibração"])
def preview_gain_fit(payload: GainFitPayload, request: Request) -> dict[str, Any]:
    """Calcula os α_s com os pontos registrados, sem gravar."""
    acq = _get_acq(request.app)
    return _gain_fit(acq, _block_model(acq), payload)


@app.post("/api/v1/calibration/gain/apply", tags=["Calibração"])
def apply_gain_fit(payload: GainFitPayload, request: Request) -> dict[str, Any]:
    """Calcula os α_s, grava em block_curves.json e recarrega o modelo."""
    acq = _get_acq(request.app)
    model = _block_model(acq)
    result = _gain_fit(acq, model, payload)
    write_gains(model.path, {int(b): a for b, a in result["alpha"].items()}, {
        "source": "painel /calibracao",
        "mode": result["mode"],
        "lambda": result["lam"],
        "point_ids": [p["id"] for p in result["points"]],
        **result["report"],
    })
    acq.models.reload()
    return _gain_state(acq, _block_model(acq))


@app.post("/api/v1/calibration/gain/reset", tags=["Calibração"])
def reset_gains(request: Request) -> dict[str, Any]:
    """Volta todos os α_s para 1 (não calibrado). Os pontos registrados continuam."""
    acq = _get_acq(request.app)
    write_gains(_block_model(acq).path, None, None)
    acq.models.reload()
    return _gain_state(acq, _block_model(acq))


# -----------------------------------------------------------------------------
# Diagnóstico da calibração (orientações de 15/10)
# -----------------------------------------------------------------------------

@app.get("/api/v1/diagnostics/cells", tags=["Diagnóstico"])
def get_cell_diagnostics(request: Request, include_inactive: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Contribuição de cada célula na janela atual: RAW, baseline, RAW corrigido, massa local,
    dm/dRAW, situação na curva e flags, da maior massa local para a menor (item 3.3)."""
    data = _get_acq(request.app).cell_report(include_inactive=include_inactive)
    if data is None:
        raise HTTPException(status_code=409, detail="Modelo ativo não é a curva por bloco")
    if limit is not None and limit >= 0:
        data["cells"] = data["cells"][:limit]
    return data


@app.get("/api/v1/diagnostics/residual", tags=["Diagnóstico"])
def get_residual_mass(request: Request) -> dict[str, Any]:
    """Massa que o modelo produz com a maca vazia desde a tara (ou desde o último reset) (item 3.5)."""
    return _get_acq(request.app).residual.snapshot()


@app.post("/api/v1/diagnostics/residual/reset", tags=["Diagnóstico"])
def reset_residual_mass(request: Request) -> dict[str, Any]:
    """Recomeça a contagem da massa residual sem fazer tara nova (ensaio de zero de 10 min)."""
    acq = _get_acq(request.app)
    acq.residual.reset()
    return acq.residual.snapshot()


@app.get("/api/v1/diagnostics/calibration", tags=["Diagnóstico"])
def get_calibration_audit(request: Request) -> dict[str, Any]:
    """Auditoria das curvas: correção isotônica por setor e por célula (UP e DOWN), massas sem
    resposta, histerese, baixa sensibilidade e setores de uma célula só (itens 3.6, 3.8 e 4)."""
    audit = _get_acq(request.app).calibration_audit()
    if audit is None:
        raise HTTPException(status_code=409, detail="Modelo ativo não é a curva por bloco")
    return audit


@app.get("/calibracao", include_in_schema=False)
def calibration_page() -> FileResponse:
    """Página de diagnóstico da calibração (curva + extração ao vivo)."""
    return FileResponse(_STATIC_DIR / "calibracao.html", media_type="text/html; charset=utf-8")


@app.post("/api/v1/calibration/reload-model", tags=["Calibração"])
def reload_model(request: Request) -> dict[str, Any]:
    """Recarrega as curvas por bloco (block_curves.json) sem reiniciar o serviço."""
    acq = _get_acq(request.app)
    loaded = acq.models.reload()
    return {"status": "ok", "loaded": loaded, **acq.models.describe()}


