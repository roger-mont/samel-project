"""Servidor FastAPI — REST + WebSocket para consumidores externos."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import bridge
from config.settings import TARE_SAMPLE_COUNT, WS_PUSH_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Walter Sensor API",
    version="1.0.0",
    description="API de integração — rede de IA hospitalar",
)


# --- Schemas ---


class CalibrationUpdate(BaseModel):
    value: float


# --- Meta ---


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


# --- Sensor ---


@app.get("/sensor", tags=["sensor"])
def get_sensor() -> dict:
    """Snapshot: peso líquido, alerta de postura, status e tara."""
    return bridge.get_current_snapshot()


@app.get("/sensor/tare", tags=["sensor"])
def get_tare() -> dict:
    """Estado atual da tara (offset, ativa, pendente)."""
    return bridge.get_tare_status()


@app.post("/sensor/tare", tags=["sensor"])
def start_tare() -> dict:
    """Inicia coleta de amostras para tara da maca.

    O offset é calculado como média de TARE_SAMPLE_COUNT frames EMA
    e persistido em disco automaticamente ao concluir.
    Consulte GET /sensor/tare para verificar quando pending=false.
    """
    bridge.start_tare_sampling()
    return {
        "ok": True,
        "status": "sampling",
        "samples_needed": TARE_SAMPLE_COUNT,
        "message": f"coletando {TARE_SAMPLE_COUNT} amostras...",
    }


@app.delete("/sensor/tare", tags=["sensor"])
def clear_tare() -> dict:
    """Remove a tara — peso volta ao valor bruto. Persiste offset=0."""
    bridge.clear_tare()
    return {"ok": True, "offset_kg": 0.0}


# --- Calibração ---


@app.get("/calibration", tags=["calibration"])
def get_calibration() -> dict:
    """Parâmetros de calibração atuais."""
    return bridge.get_calibration_snapshot()


@app.put("/calibration/{key}", tags=["calibration"])
def update_calibration(key: str, body: CalibrationUpdate) -> dict:
    """Atualiza um parâmetro de calibração pelo nome."""
    result = bridge.update_calibration_param(key, body.value)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# --- Monitor ---


@app.post("/monitor/reset", tags=["monitor"])
def reset_monitor() -> dict:
    """Reseta o PostureMonitor (cronômetro e referência de postura)."""
    bridge.reset_posture_monitor()
    return {"ok": True}


# --- WebSocket ---


@app.websocket("/ws/sensor")
async def ws_sensor(websocket: WebSocket) -> None:
    """Push a cada WS_PUSH_INTERVAL_SECONDS: peso, alerta, status, tara."""
    await websocket.accept()
    logger.info("WS cliente conectado: %s", websocket.client)
    try:
        while True:
            payload = bridge.get_current_snapshot()
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            await websocket.send_json(payload)
            await asyncio.sleep(WS_PUSH_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        logger.info("WS cliente desconectado: %s", websocket.client)
    except Exception as err:
        logger.error("WS erro: %s", err)


# --- Startup ---


def start(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Inicia uvicorn em modo bloqueante — chamar em daemon thread."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
