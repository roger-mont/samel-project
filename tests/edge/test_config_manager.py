"""Testes automatizados para o ConfigManager e endpoints de configuração do sistema."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from fastapi.testclient import TestClient

# Garante import do src do edge-service
_EDGE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "apps", "edge-service", "src")
)
if _EDGE_SRC not in sys.path:
    sys.path.insert(0, _EDGE_SRC)

from api.server import app
from core.config import ConfigManager


def test_config_manager_cascade_and_save() -> None:
    """Testa leitura em cascata e salvamento atômico em arquivo JSON temporário."""
    temp_dir = tempfile.mkdtemp()
    cfg_file = os.path.join(temp_dir, "test_config.json")

    # Cria arquivo inicial
    initial_data = {
        "maca_id": "MACA-TEST-99",
        "central_api_url": "http://10.0.0.1:8000",
        "sync_interval_sec": 45.0,
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(initial_data, f)

    # Inicia ConfigManager apontando via variável de ambiente
    os.environ["SAMEL_CONFIG_PATH"] = cfg_file
    try:
        mgr = ConfigManager()
        snap = mgr.snapshot()
        assert snap["maca_id"] == "MACA-TEST-99"
        assert snap["central_api_url"] == "http://10.0.0.1:8000"
        assert snap["sync_interval_sec"] == 45.0

        # Atualiza e salva
        updated = mgr.update_and_save({"maca_id": "MACA-TEST-100", "sync_interval_sec": 30.0})
        assert updated["maca_id"] == "MACA-TEST-100"
        assert updated["sync_interval_sec"] == 30.0

        # Valida que o arquivo no disco foi atualizado
        with open(cfg_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        assert disk_data["maca_id"] == "MACA-TEST-100"
        assert disk_data["sync_interval_sec"] == 30.0

    finally:
        os.environ.pop("SAMEL_CONFIG_PATH", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_api_system_config_endpoints() -> None:
    """Testa endpoints GET e POST /api/v1/system/config."""
    with TestClient(app) as client:
        # 1. GET
        res_get = client.get("/api/v1/system/config")
        assert res_get.status_code == 200
        data_get = res_get.json()
        assert "maca_id" in data_get
        assert "central_api_url" in data_get

        # 2. POST
        res_post = client.post(
            "/api/v1/system/config",
            json={"maca_id": "MACA-LEITO-500", "sync_interval_sec": 90.0},
        )
        assert res_post.status_code == 200
        data_post = res_post.json()
        assert data_post["status"] == "ok"
        assert data_post["config"]["maca_id"] == "MACA-LEITO-500"
        assert data_post["config"]["sync_interval_sec"] == 90.0
