"""Servidor FastAPI Edge para Maca Hospitalar Inteligente.

Fornece endpoints REST e uma 'via expressa' WebSocket (/ws/telemetry) para
streaming em tempo real de dados quentes e broadcast imediato de alertas clínicos.

Mudança vs. versão anterior:
- Imports atualizados: ``database.db_local`` e ``workers.sync_worker`` ao invés de
  ``services.db_local`` e ``services.sync_worker``.
- Instâncias de ``LocalDatabase`` e ``SyncWorker`` são criadas no lifespan e injetadas
  via ``app.state``, eliminando singletons globais e facilitando testes.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import LOG_LEVEL
from database.db_local import LocalDatabase
from workers.sync_worker import SyncWorker

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("edge_server")


# -----------------------------------------------------------------------------
# Modelos Pydantic (Clean Code & Validação)
# -----------------------------------------------------------------------------

class TelemetryPayload(BaseModel):
    timestamp: str | None = Field(default=None, description="ISO8601 UTC timestamp")
    maca_id: str = Field(..., description="Identificador único da maca")
    peso_kg: float | None = Field(default=None, ge=0.0, le=500.0)
    indice_postural: float | None = Field(default=None)
    tempo_estatico_seg: int = Field(default=0, ge=0)
    status_alerta: bool = Field(default=False)
    payload: dict[str, Any] | None = Field(
        default=None, description="Dados adicionais como mapa de pressão"
    )


class PostureEventPayload(BaseModel):
    timestamp: str | None = Field(default=None, description="ISO8601 UTC timestamp")
    maca_id: str = Field(..., description="Identificador único da maca")
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


# -----------------------------------------------------------------------------
# Ciclo de Vida da Aplicação (FastAPI Lifespan)
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Inicializa banco SQLite e inicia o daemon de sincronização em background."""
    logger.info("Iniciando Módulo Edge FastAPI...")

    db = LocalDatabase()
    worker = SyncWorker(db=db)

    app.state.db = db
    app.state.sync_worker = worker

    worker.start()
    yield
    logger.info("Encerrando Módulo Edge FastAPI...")
    await worker.stop()


app = FastAPI(
    title="Módulo Edge - Maca Hospitalar Inteligente",
    description="Serviço local de alta resiliência com Store-and-Forward e WebSocket.",
    version="1.0.0",
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


# -----------------------------------------------------------------------------
# Endpoints REST e WebSocket
# -----------------------------------------------------------------------------

@app.get("/health", tags=["Monitoramento"])
async def health_check() -> dict[str, Any]:
    """Endpoint básico de verificação de integridade para Docker/K8s."""
    return {
        "status": "healthy",
        "service": "edge-maca-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/status", tags=["Monitoramento"])
async def get_edge_status() -> dict[str, Any]:
    """Status detalhado do módulo edge: pendências na fila, worker e circuit breaker."""
    db = _get_db(app)
    worker = _get_worker(app)
    unsynced_tel = db.count_unsynced_telemetry()
    unsynced_evt = db.count_unsynced_events()

    return {
        "maca_id": worker.maca_id,
        "database": {
            "path": db.db_path,
            "unsynced_telemetry": unsynced_tel,
            "unsynced_events": unsynced_evt,
            "total_pending_sync": unsynced_tel + unsynced_evt,
        },
        "sync_worker": {
            "running": worker.is_running,
            "batch_size": worker.batch_size,
            "timeout_seconds": worker.timeout_seconds,
            "circuit_breaker_state": worker.circuit_breaker.state.value,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/api/v1/telemetry",
    status_code=status.HTTP_201_CREATED,
    tags=["Telemetria"],
)
async def ingest_telemetry(data: TelemetryPayload) -> dict[str, Any]:
    """Recebe dados de telemetria, grava no SQLite e faz broadcast no WebSocket."""
    db = _get_db(app)
    ts = data.timestamp or datetime.now(timezone.utc).isoformat()
    record_id = db.enqueue_telemetry(
        timestamp=ts,
        maca_id=data.maca_id,
        peso_kg=data.peso_kg,
        indice_postural=data.indice_postural,
        tempo_estatico_seg=data.tempo_estatico_seg,
        status_alerta=data.status_alerta,
        payload=data.payload,
    )

    ws_packet = {
        "type": "TELEMETRY_HOT_DATA",
        "record_id": record_id,
        "timestamp": ts,
        "maca_id": data.maca_id,
        "peso_kg": data.peso_kg,
        "indice_postural": data.indice_postural,
        "tempo_estatico_seg": data.tempo_estatico_seg,
        "status_alerta": data.status_alerta,
        "payload": data.payload,
    }
    await ws_manager.broadcast(ws_packet)

    if data.status_alerta:
        await ws_manager.emit_alert({
            "event": "STATIC_POSTURE_LIMIT_EXCEEDED",
            "maca_id": data.maca_id,
            "tempo_estatico_seg": data.tempo_estatico_seg,
            "mensagem": "Atenção: Paciente estático por tempo prolongado. Risco de LPP.",
        })

    return {"status": "enqueued", "id": record_id}


@app.post(
    "/api/v1/events/posture",
    status_code=status.HTTP_201_CREATED,
    tags=["Eventos"],
)
async def record_event(data: PostureEventPayload) -> dict[str, Any]:
    """Registra evento clínico de mudança postural e emite alerta se aplicável."""
    db = _get_db(app)
    ts = data.timestamp or datetime.now(timezone.utc).isoformat()
    event_id = db.record_posture_event(
        timestamp=ts,
        maca_id=data.maca_id,
        postura_detectada=data.postura_detectada,
        sessao_id=data.sessao_id,
        postura_anterior=data.postura_anterior,
        duracao_anterior_seg=data.duracao_anterior_seg,
        regiao_pico=data.regiao_pico,
        pico_pct=data.pico_pct,
        area_pct=data.area_pct,
        indice_dist=data.indice_dist,
        houve_alerta=data.houve_alerta,
    )

    event_packet = {
        "type": "POSTURE_EVENT",
        "event_id": event_id,
        "timestamp": ts,
        "maca_id": data.maca_id,
        "postura_detectada": data.postura_detectada,
        "postura_anterior": data.postura_anterior,
        "houve_alerta": data.houve_alerta,
    }
    await ws_manager.broadcast(event_packet)

    if data.houve_alerta:
        await ws_manager.emit_alert({
            "event": "POSTURE_ALERT",
            "maca_id": data.maca_id,
            "regiao_pico": data.regiao_pico,
            "pico_intensidade_pct": data.pico_pct,
            "mensagem": f"Pico de pressão detectado: {data.regiao_pico} ({data.pico_pct}%)",
        })

    return {"status": "recorded", "event_id": event_id}


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket) -> None:
    """Via expressa: WebSocket para transmissão contínua de dados quentes e alertas."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "PING":
                    await websocket.send_text(
                        json.dumps({
                            "type": "PONG",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    )
            except Exception:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as ex:
        logger.warning("Erro na conexão WebSocket: %s", ex)
        await ws_manager.disconnect(websocket)
