"""Suíte de testes automatizados para o Módulo Edge.

Cobre persistência SQLite anti-corrupção (WAL/FIFO), daemon Store-and-Forward
com Gatilho Duplo, Circuit Breaker e servidor FastAPI/WebSocket.

Mudança vs. versão anterior:
- Imports atualizados para nova estrutura: ``api.server``, ``database.db_local``,
  ``workers.sync_worker`` ao invés de ``services.db_local``, ``services.sync_worker``.
- Fixture ``sys_path_setup`` garante que ``apps/edge-service/src`` esteja no ``sys.path``
  para que os testes rodem independente do diretório de trabalho.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from typing import Generator

import pytest

# Garante que src/ do edge-service esteja no sys.path para resolução de imports
_EDGE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "apps", "edge-service", "src")
)
if _EDGE_SRC not in sys.path:
    sys.path.insert(0, _EDGE_SRC)

from fastapi.testclient import TestClient

from api.server import app
from database.db_local import LocalDatabase
from workers.sync_worker import CircuitBreaker, CircuitState, SyncWorker


@pytest.fixture
def temp_db() -> Generator[LocalDatabase, None, None]:
    """Cria um banco SQLite temporário isolado para testes."""
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "test_edge.db")
    db = LocalDatabase(db_path=db_file, max_records=10)
    yield db
    shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 1. Testes do SQLite Anti-Corrupção e Retenção FIFO
# -----------------------------------------------------------------------------

def test_sqlite_wal_pragmas_and_tables(temp_db: LocalDatabase) -> None:
    """Valida se o SQLite opera em modo WAL, synchronous=NORMAL e cria tabelas."""
    with temp_db.get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync_mode = conn.execute("PRAGMA synchronous;").fetchone()[0]

        assert journal_mode.upper() == "WAL", f"Esperado WAL, obtido {journal_mode}"
        assert sync_mode in (1, "1", "NORMAL"), f"Esperado NORMAL (1), obtido {sync_mode}"

        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "telemetry_queue" in tables
        assert "posture_events" in tables
        assert "system_audit" in tables


def test_sqlite_fifo_buffer_circular_retention(temp_db: LocalDatabase) -> None:
    """Valida a política de buffer circular FIFO descartando os mais antigos."""
    for i in range(15):
        ts = f"2026-08-22T12:{i:02d}:00Z"
        temp_db.enqueue_telemetry(
            timestamp=ts,
            maca_id="MACA-01",
            peso_kg=70.0 + i,
            tempo_estatico_seg=i * 10,
        )

    with temp_db.get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM telemetry_queue").fetchone()[0]
        assert total == 10, f"Deveria reter exatamente 10 registros, mas tem {total}"

        oldest = conn.execute("SELECT MIN(peso_kg) FROM telemetry_queue").fetchone()[0]
        assert oldest == 75.0, f"O mais antigo deveria ter peso 75.0 (i=5), obtido {oldest}"


# -----------------------------------------------------------------------------
# 2. Testes do Circuit Breaker
# -----------------------------------------------------------------------------

def test_circuit_breaker_lifecycle() -> None:
    """Valida transição entre CLOSED, OPEN, HALF_OPEN e CLOSED."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=0.1)

    assert cb.state == CircuitState.CLOSED
    assert cb.can_attempt() is True

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_attempt() is False

    time.sleep(0.15)
    assert cb.can_attempt() is True
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


# -----------------------------------------------------------------------------
# 3. Testes do SyncWorker (Store-and-Forward & Gatilho Duplo)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_worker_triggers_and_ordering(temp_db: LocalDatabase) -> None:
    """Valida ordenação cronológica e gatilhos de sincronização."""
    t1 = "2026-08-22T10:00:00Z"
    t2 = "2026-08-22T10:01:00Z"
    t3 = "2026-08-22T10:02:00Z"

    temp_db.enqueue_telemetry(timestamp=t2, maca_id="MACA-01", peso_kg=71.0)
    temp_db.enqueue_telemetry(timestamp=t1, maca_id="MACA-01", peso_kg=70.0)
    temp_db.enqueue_telemetry(timestamp=t3, maca_id="MACA-01", peso_kg=72.0)

    unsynced = temp_db.get_unsynced_telemetry(limit=10)
    timestamps = [item["timestamp"] for item in unsynced]
    assert timestamps == [t1, t2, t3], f"Registros fora de ordem cronológica: {timestamps}"

    worker = SyncWorker(
        db=temp_db,
        central_url="http://mock-server:8000",
        batch_size=2,
        timeout_seconds=1.0,
    )

    async def mock_send_telemetry(batch: list) -> bool:
        temp_db.mark_telemetry_synced([item["id"] for item in batch])
        return True

    worker._send_telemetry_batch = mock_send_telemetry  # type: ignore

    synced = await worker.check_and_sync()
    assert synced is True
    assert temp_db.count_unsynced_telemetry() == 1

    await asyncio.sleep(1.1)
    synced2 = await worker.check_and_sync()
    assert synced2 is True
    assert temp_db.count_unsynced_telemetry() == 0


# -----------------------------------------------------------------------------
# 4. Testes do Servidor FastAPI e WebSocket
# -----------------------------------------------------------------------------

def test_api_health_and_status() -> None:
    """Testa endpoints /health e /api/v1/status."""
    with TestClient(app) as client:
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        res_status = client.get("/api/v1/status")
        assert res_status.status_code == 200
        data = res_status.json()
        assert "maca_id" in data
        assert "database" in data
        assert "sync_worker" in data


def test_api_ingest_telemetry_and_events() -> None:
    """Testa ingestão REST de telemetria e eventos posturais."""
    with TestClient(app) as client:
        tel_data = {
            "maca_id": "MACA-01",
            "peso_kg": 74.5,
            "indice_postural": 0.85,
            "tempo_estatico_seg": 360,
            "status_alerta": False,
            "payload": {"matriz_resumo": [10, 20, 30]},
        }
        res_tel = client.post("/api/v1/telemetry", json=tel_data)
        assert res_tel.status_code == 201
        assert res_tel.json()["status"] == "enqueued"

        event_data = {
            "maca_id": "MACA-01",
            "postura_detectada": "Decúbito Lateral Direito",
            "postura_anterior": "Decúbito Dorsal",
            "duracao_anterior_seg": 1800,
            "regiao_pico": "Região Trocantérica Direita",
            "pico_pct": 82.0,
            "houve_alerta": True,
        }
        res_evt = client.post("/api/v1/events/posture", json=event_data)
        assert res_evt.status_code == 201
        assert res_evt.json()["status"] == "recorded"


def test_websocket_stream() -> None:
    """Testa a via expressa WebSocket conectando, respondendo PING e recebendo broadcasts."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry") as websocket:
            websocket.send_text('{"type": "PING"}')
            pong_msg = websocket.receive_json()
            assert pong_msg["type"] == "PONG"

            client.post(
                "/api/v1/telemetry",
                json={"maca_id": "MACA-01", "peso_kg": 76.0, "status_alerta": True},
            )

            hot_data = websocket.receive_json()
            assert hot_data["type"] == "TELEMETRY_HOT_DATA"
            assert hot_data["maca_id"] == "MACA-01"

            alert_data = websocket.receive_json()
            assert alert_data["type"] == "EMERGENCY_ALERT"
            assert alert_data["priority"] == "HIGH"
