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
from typing import Any, AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import LOG_LEVEL, MACA_ID, config_manager
from database.db_local import LocalDatabase
from workers.sync_worker import SyncWorker
from workers.acquisition_worker import AcquisitionWorker

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("edge_server")


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
    param_name: str = Field(..., description="Nome do parâmetro (deadzone_threshold, ema_alpha, posture_tolerance, posture_timeout_seconds)")
    value: float = Field(..., description="Novo valor do parâmetro")


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
def trigger_tare(request: Request) -> dict[str, str]:
    """Dispara ciclo de amostragem e compensação de tara."""
    acq = _get_acq(request.app)
    acq.trigger_tare()
    return {"status": "ok", "message": "Amostragem de tara iniciada"}


@app.get("/api/v1/calibration/params", tags=["Calibração"])
def get_calibration_params(request: Request) -> dict[str, Any]:
    """Retorna os parâmetros de calibração em vigor."""
    acq = _get_acq(request.app)
    return acq.params.snapshot()


@app.post("/api/v1/calibration/params", tags=["Calibração"])
def update_calibration_param(payload: CalibrationUpdatePayload, request: Request) -> dict[str, Any]:
    """Atualiza parâmetro de calibração em runtime."""
    acq = _get_acq(request.app)
    acq.update_param(payload.param_name, payload.value)
    return {"status": "ok", "params": acq.params.snapshot()}


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
