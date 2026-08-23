"""Configurações centralizadas do Edge Service.

Carrega variáveis de ambiente para parametrizar o serviço sem hardcoding.
Centraliza todas as constantes configuráveis do módulo edge em um único ponto.
"""
from __future__ import annotations

import os


# ── Identificação da Maca ───────────────────────────────────────────────────
MACA_ID: str = os.getenv("MACA_ID", "MACA-EDGE-001")

# ── Banco de Dados SQLite Local ─────────────────────────────────────────────
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/edge_maca.db")
MAX_RETENTION_RECORDS: int = int(os.getenv("MAX_RETENTION_RECORDS", "50000"))
RETENTION_DAYS: int = int(os.getenv("RETENTION_DAYS", "7"))

# ── Sincronização Store-and-Forward ─────────────────────────────────────────
CENTRAL_API_URL: str = os.getenv("CENTRAL_API_URL", "http://central-server:8000")
EDGE_API_TOKEN: str = os.getenv("EDGE_API_TOKEN", "edge-default-token")
SYNC_BATCH_SIZE: int = int(os.getenv("SYNC_BATCH_SIZE", "50"))
SYNC_INTERVAL_SEC: float = float(os.getenv("SYNC_INTERVAL_SEC", "60"))

# ── Circuit Breaker ─────────────────────────────────────────────────────────
CB_FAILURE_THRESHOLD: int = int(os.getenv("CB_FAILURE_THRESHOLD", "4"))
CB_RECOVERY_TIMEOUT_SEC: float = float(os.getenv("CB_RECOVERY_TIMEOUT_SEC", "30.0"))

# ── Servidor FastAPI ────────────────────────────────────────────────────────
SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
