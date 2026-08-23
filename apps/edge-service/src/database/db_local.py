"""Módulo de Persistência Local (SQLite Anti-Corrupção) para Módulo Edge.

Opera em modo WAL (Write-Ahead Logging) com synchronous=NORMAL e busy_timeout
para suportar quedas bruscas de energia sem corrupção de dados. Implementa política
de retenção com Buffer Circular (FIFO) para evitar esgotamento de disco em períodos offline.

Mudança vs. versão anterior:
- Imports de constantes agora vêm de ``core.config`` ao invés de ler ``os.getenv``
  localmente, centralizando toda configuração em um único ponto.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Generator, Sequence

from core.config import MAX_RETENTION_RECORDS, RETENTION_DAYS, SQLITE_DB_PATH

logger = logging.getLogger(__name__)


class LocalDatabase:
    """Gerenciador do banco de dados SQLite local com suporte a WAL e resiliência."""

    def __init__(
        self,
        db_path: str = SQLITE_DB_PATH,
        max_records: int = MAX_RETENTION_RECORDS,
    ) -> None:
        self.db_path = db_path
        self.max_records = max_records
        self._ensure_db_dir()
        self.init_db()

    def _ensure_db_dir(self) -> None:
        """Garante a existência do diretório onde o arquivo .db reside."""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    @contextlib.contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Cria e configura uma conexão SQLite com pragmas de segurança e tolerância a falhas."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA temp_store = MEMORY;")
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Executa a rotina de DDL inicializando as tabelas se não existirem."""
        with self.get_connection() as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        maca_id TEXT NOT NULL,
                        peso_kg REAL,
                        indice_postural REAL,
                        tempo_estatico_seg INTEGER DEFAULT 0,
                        status_alerta INTEGER DEFAULT 0,
                        payload_json TEXT,
                        synced INTEGER DEFAULT 0,
                        synced_at TEXT,
                        retry_count INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT (datetime('now', 'utc'))
                    );
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_telemetry_synced_time 
                    ON telemetry_queue (synced, timestamp ASC);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_telemetry_created_at 
                    ON telemetry_queue (created_at);
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS posture_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        sessao_id TEXT,
                        maca_id TEXT NOT NULL,
                        postura_anterior TEXT,
                        postura_detectada TEXT NOT NULL,
                        duracao_postura_anterior_seg INTEGER DEFAULT 0,
                        regiao_pico_pressao TEXT,
                        pico_intensidade_pct REAL,
                        area_contato_pct REAL,
                        indice_distribuicao REAL,
                        houve_alerta INTEGER DEFAULT 0,
                        synced INTEGER DEFAULT 0,
                        synced_at TEXT,
                        retry_count INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT (datetime('now', 'utc'))
                    );
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_posture_synced_time 
                    ON posture_events (synced, timestamp ASC);
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT (datetime('now', 'utc')),
                        event_type TEXT NOT NULL,
                        details TEXT
                    );
                """)

            logger.info("SQLite Edge inicializado em modo WAL: %s", self.db_path)

    # -------------------------------------------------------------------------
    # Ingestão de Dados
    # -------------------------------------------------------------------------

    def enqueue_telemetry(
        self,
        timestamp: str,
        maca_id: str,
        peso_kg: float | None = None,
        indice_postural: float | None = None,
        tempo_estatico_seg: int = 0,
        status_alerta: bool = False,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Enfileira um registro de telemetria e aplica a política de retenção FIFO."""
        payload_str = json.dumps(payload) if payload is not None else None
        alerta_int = 1 if status_alerta else 0

        with self.get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO telemetry_queue (
                        timestamp, maca_id, peso_kg, indice_postural,
                        tempo_estatico_seg, status_alerta, payload_json, synced
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (timestamp, maca_id, peso_kg, indice_postural,
                     tempo_estatico_seg, alerta_int, payload_str),
                )
                inserted_id = cursor.lastrowid or 0

            self._enforce_retention_policy(conn, "telemetry_queue")
            return inserted_id

    def record_posture_event(
        self,
        timestamp: str,
        maca_id: str,
        postura_detectada: str,
        sessao_id: str | None = None,
        postura_anterior: str | None = None,
        duracao_anterior_seg: int = 0,
        regiao_pico: str | None = None,
        pico_pct: float | None = None,
        area_pct: float | None = None,
        indice_dist: float | None = None,
        houve_alerta: bool = False,
    ) -> int:
        """Grava um evento de transição ou alerta postural na fila de sincronização."""
        with self.get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO posture_events (
                        timestamp, sessao_id, maca_id, postura_anterior,
                        postura_detectada, duracao_postura_anterior_seg,
                        regiao_pico_pressao, pico_intensidade_pct,
                        area_contato_pct, indice_distribuicao,
                        houve_alerta, synced
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (timestamp, sessao_id, maca_id, postura_anterior,
                     postura_detectada, duracao_anterior_seg, regiao_pico,
                     pico_pct, area_pct, indice_dist,
                     1 if houve_alerta else 0),
                )
                inserted_id = cursor.lastrowid or 0

            self._enforce_retention_policy(conn, "posture_events")
            return inserted_id

    # -------------------------------------------------------------------------
    # Operações de Sincronização (Store-and-Forward)
    # -------------------------------------------------------------------------

    def get_unsynced_telemetry(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recupera lote de telemetrias pendentes estritamente ordenado por timestamp ASC."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, maca_id, peso_kg, indice_postural,
                       tempo_estatico_seg, status_alerta, payload_json, retry_count
                FROM telemetry_queue
                WHERE synced = 0
                ORDER BY timestamp ASC, id ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                if item.get("payload_json"):
                    try:
                        item["payload"] = json.loads(item["payload_json"])
                    except Exception:
                        item["payload"] = None
                else:
                    item["payload"] = None
                results.append(item)
            return results

    def count_unsynced_telemetry(self) -> int:
        """Retorna o total de registros de telemetria pendentes de envio."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) AS total FROM telemetry_queue WHERE synced = 0"
            )
            row = cursor.fetchone()
            return int(row["total"]) if row else 0

    def get_unsynced_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recupera lote de eventos posturais pendentes ordenado por timestamp ASC."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, sessao_id, maca_id, postura_anterior,
                       postura_detectada, duracao_postura_anterior_seg,
                       regiao_pico_pressao, pico_intensidade_pct,
                       area_contato_pct, indice_distribuicao, houve_alerta,
                       retry_count
                FROM posture_events
                WHERE synced = 0
                ORDER BY timestamp ASC, id ASC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]

    def count_unsynced_events(self) -> int:
        """Retorna o total de eventos posturais pendentes de envio."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) AS total FROM posture_events WHERE synced = 0"
            )
            row = cursor.fetchone()
            return int(row["total"]) if row else 0

    def mark_telemetry_synced(self, record_ids: Sequence[int]) -> None:
        """Marca registros de telemetria como sincronizados (synced = 1) com timestamp UTC."""
        if not record_ids:
            return
        now_utc = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in record_ids)
        with self.get_connection() as conn:
            with conn:
                conn.execute(
                    f"""
                    UPDATE telemetry_queue
                    SET synced = 1, synced_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [now_utc, *record_ids],
                )

    def mark_events_synced(self, event_ids: Sequence[int]) -> None:
        """Marca eventos posturais como sincronizados (synced = 1) com timestamp UTC."""
        if not event_ids:
            return
        now_utc = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in event_ids)
        with self.get_connection() as conn:
            with conn:
                conn.execute(
                    f"""
                    UPDATE posture_events
                    SET synced = 1, synced_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [now_utc, *event_ids],
                )

    def increment_retry_count(self, table_name: str, record_ids: Sequence[int]) -> None:
        """Incrementa o contador de retentativas para registros com falha de transmissão."""
        if not record_ids or table_name not in ("telemetry_queue", "posture_events"):
            return
        placeholders = ",".join("?" for _ in record_ids)
        with self.get_connection() as conn:
            with conn:
                conn.execute(
                    f"""
                    UPDATE {table_name}
                    SET retry_count = retry_count + 1
                    WHERE id IN ({placeholders})
                    """,
                    list(record_ids),
                )

    # -------------------------------------------------------------------------
    # Política de Retenção e Buffer Circular (FIFO)
    # -------------------------------------------------------------------------

    def _enforce_retention_policy(
        self, conn: sqlite3.Connection, table_name: str
    ) -> None:
        """Aplica regras de retenção: expurgo por idade (> 7 dias) e Buffer Circular FIFO."""
        try:
            with conn:
                conn.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE synced = 1 
                      AND created_at < datetime('now', '-{RETENTION_DAYS} days')
                    """
                )

                cursor = conn.execute(
                    f"SELECT COUNT(*) AS total FROM {table_name}"
                )
                row = cursor.fetchone()
                total_count = int(row["total"]) if row else 0

                if total_count > self.max_records:
                    excess = total_count - self.max_records
                    conn.execute(
                        f"""
                        DELETE FROM {table_name}
                        WHERE id IN (
                            SELECT id FROM {table_name}
                            ORDER BY id ASC
                            LIMIT ?
                        )
                        """,
                        (excess,),
                    )
                    logger.warning(
                        "FIFO ativado em %s (%d > %d). %d registros antigos expurgados.",
                        table_name, total_count, self.max_records, excess,
                    )
        except Exception as ex:
            logger.error("Erro na política de retenção de %s: %s", table_name, ex)
