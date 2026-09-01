"""Suíte de testes automatizados para os endpoints e agregações de Analítica do Módulo Edge.

Valida:
- Agrupamento das últimas 4h em 9 blocos de 30 minutos (peso, índice postural e tempo estático).
- Formatação dos eventos recentes.
- Agregação estrita do dia civil corrente (a partir das 00:00).
- Endpoints REST FastAPI.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Garante inclusão de apps/edge-service/src no sys.path
_EDGE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "apps", "edge-service", "src")
)
if _EDGE_SRC not in sys.path:
    sys.path.insert(0, _EDGE_SRC)

from api.server import app
from database.db_local import LocalDatabase


@pytest.fixture
def temp_db() -> Generator[LocalDatabase, None, None]:
    """Cria um banco SQLite temporário isolado."""
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "test_analytics.db")
    db = LocalDatabase(db_path=db_file, max_records=1000)
    yield db
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_charts_analytics_9_points(temp_db: LocalDatabase) -> None:
    """Valida se o cálculo gera 9 pontos de 30 minutos em uma janela de 4 horas."""
    now = datetime.now(timezone.utc)

    # Inserir dados em diferentes janelas de 30 minutos
    for offset_minutes in [210, 150, 90, 30, 2]:
        ts = (now - timedelta(minutes=offset_minutes)).isoformat()
        temp_db.enqueue_telemetry(
            timestamp=ts,
            maca_id="MACA-01",
            peso_kg=65.0,
            indice_postural=0.05,
            tempo_estatico_seg=offset_minutes * 60,
            status_alerta=False,
        )

    res = temp_db.get_charts_analytics(window_hours=4)
    assert "labels" in res
    assert len(res["labels"]) == 9
    assert len(res["weight_series"]) == 9
    assert len(res["posture_series"]) == 9
    assert len(res["time_series"]) == 9

    # O último ponto ("Agora") deve ter o valor do registro recente (2 minutos atrás)
    assert res["weight_series"][-1] == 65.0


def test_daily_summary_counts_strictly_today(temp_db: LocalDatabase) -> None:
    """Valida se o resumo diário contabiliza estritamente os eventos do dia civil atual."""
    now = datetime.now()
    today_ts = now.strftime("%Y-%m-%dT10:00:00")
    yesterday_ts = (now - timedelta(days=1)).strftime("%Y-%m-%dT10:00:00")

    # Inserir 2 eventos de ontem
    temp_db.record_posture_event(
        timestamp=yesterday_ts,
        maca_id="MACA-01",
        postura_detectada="Decúbito Lateral Esquerdo",
        duracao_anterior_seg=3600,
        houve_alerta=False,
    )
    temp_db.record_posture_event(
        timestamp=yesterday_ts,
        maca_id="MACA-01",
        postura_detectada="Decúbito Dorsal",
        duracao_anterior_seg=3600,
        houve_alerta=False,
    )

    # Inserir 3 eventos de hoje
    for i in range(3):
        temp_db.record_posture_event(
            timestamp=today_ts,
            maca_id="MACA-01",
            postura_detectada=f"Decúbito Posição {i}",
            duracao_anterior_seg=1800,
            houve_alerta=(i == 2),
        )

    # Inserir 10 minutos de telemetria hoje (1 com alerta)
    for i in range(10):
        temp_db.enqueue_telemetry(
            timestamp=today_ts,
            maca_id="MACA-01",
            peso_kg=64.0,
            status_alerta=(i == 0),
        )

    summary = temp_db.get_daily_summary()
    assert summary["total_rotations_today"] == 3, f"Esperado 3 eventos de hoje, obtido {summary['total_rotations_today']}"
    assert summary["avg_posture_time_min"] == 30.0
    assert summary["total_alerts_today"] == 1
    assert summary["relief_score_pct"] == 90.0  # 10 min com 1 alerta = 90%


def test_analytics_endpoints_api() -> None:
    """Valida os 3 endpoints de analítica via FastAPI TestClient."""
    with TestClient(app) as client:
        # Inserir dado básico
        client.post(
            "/api/v1/telemetry",
            json={"maca_id": "MACA-01", "peso_kg": 68.0, "status_alerta": False},
        )
        client.post(
            "/api/v1/events/posture",
            json={"maca_id": "MACA-01", "postura_detectada": "Decúbito Dorsal", "duracao_anterior_seg": 1200},
        )

        res_charts = client.get("/api/v1/analytics/charts?window_hours=4")
        assert res_charts.status_code == 200
        assert len(res_charts.json()["labels"]) == 9

        res_events = client.get("/api/v1/analytics/events?limit=5")
        assert res_events.status_code == 200
        assert isinstance(res_events.json(), list)

        res_summary = client.get("/api/v1/analytics/daily-summary")
        assert res_summary.status_code == 200
        assert "total_rotations_today" in res_summary.json()
        assert "relief_score_pct" in res_summary.json()
