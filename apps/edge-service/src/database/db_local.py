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

    # -------------------------------------------------------------------------
    # Métodos de Analítica e Agregação para o Dashboard
    # -------------------------------------------------------------------------

    def get_charts_analytics(self, window_hours: int = 4) -> dict[str, Any]:
        """Calcula séries temporais de 9 pontos (4h em blocos de 30 min) para os 3 gráficos da UI."""
        labels = ["-4h", "-3.5h", "-3h", "-2.5h", "-2h", "-1.5h", "-1h", "-30m", "Agora"]
        now_ts = datetime.now(timezone.utc).timestamp()
        
        # 9 pontos com intervalos de 30 minutos (1800s cada)
        # O ponto 0 é [-4.0h, -3.5h], o ponto 7 é [-0.5h, 0.0h], o ponto 8 é o momento atual ("Agora")
        weights: list[float | None] = []
        posture_indices: list[float | None] = []
        static_times_min: list[float | None] = []

        with self.get_connection() as conn:
            for i in range(9):
                if i < 8:
                    # Blocos históricos de 30 minutos
                    offset_start_s = (8 - i) * 1800
                    offset_end_s = (7 - i) * 1800
                    t_start_iso = datetime.fromtimestamp(now_ts - offset_start_s, timezone.utc).isoformat()
                    t_end_iso = datetime.fromtimestamp(now_ts - offset_end_s, timezone.utc).isoformat()

                    cursor = conn.execute(
                        """
                        SELECT 
                            AVG(peso_kg) AS peso_avg,
                            AVG(indice_postural) AS indice_avg,
                            MAX(tempo_estatico_seg) AS max_tempo_seg
                        FROM telemetry_queue
                        WHERE timestamp >= ? AND timestamp < ?
                        """,
                        (t_start_iso, t_end_iso),
                    )
                else:
                    # Ponto 8: "Agora" (últimos 5 minutos ou último registro conhecido)
                    t_recent_iso = datetime.fromtimestamp(now_ts - 300, timezone.utc).isoformat()
                    cursor = conn.execute(
                        """
                        SELECT 
                            peso_kg AS peso_avg,
                            indice_postural AS indice_avg,
                            tempo_estatico_seg AS max_tempo_seg
                        FROM telemetry_queue
                        ORDER BY timestamp DESC, id DESC
                        LIMIT 1
                        """,
                    )

                row = cursor.fetchone()
                if row and row["peso_avg"] is not None:
                    weights.append(round(float(row["peso_avg"]), 2))
                    posture_indices.append(round(float(row["indice_avg"] or 0.0), 3))
                    max_tempo = float(row["max_tempo_seg"] or 0.0)
                    static_times_min.append(round(max_tempo / 60.0, 1))
                else:
                    weights.append(None)
                    posture_indices.append(None)
                    static_times_min.append(None)

        return {
            "labels": labels,
            "window_hours": window_hours,
            "weight_series": weights,
            "posture_series": posture_indices,
            "time_series": static_times_min,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_recent_events(self, limit: int = 5) -> list[dict[str, Any]]:
        """Recupera os últimos N eventos posturais formatados para o painel de histórico."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    id, timestamp, postura_anterior, postura_detectada,
                    duracao_postura_anterior_seg, regiao_pico_pressao,
                    pico_intensidade_pct, houve_alerta
                FROM posture_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            events = []
            for r in rows:
                item = dict(r)
                # Formatar horário legível (HH:MM)
                try:
                    dt = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                    time_str = dt.strftime("%H:%M")
                except Exception:
                    time_str = "--:--"
                
                # Montar descrição amigável para a enfermagem
                if item["houve_alerta"]:
                    desc = f"Alerta de tempo estático excedido ({round(item['duracao_postura_anterior_seg']/60)} min). Risco de LPP."
                    level = "warning"
                elif item["postura_anterior"]:
                    min_dur = round(item["duracao_postura_anterior_seg"] / 60)
                    desc = f"Mudança de {item['postura_anterior']} para {item['postura_detectada']} ({min_dur} min anterior)."
                    level = "info"
                else:
                    desc = f"Postura detectada: {item['postura_detectada']}."
                    level = "normal"

                events.append({
                    "id": item["id"],
                    "time": time_str,
                    "timestamp": item["timestamp"],
                    "description": desc,
                    "level": level,
                    "posture": item["postura_detectada"],
                    "alert": bool(item["houve_alerta"]),
                })
            return events

    def get_daily_summary(self, target_date: str | None = None) -> dict[str, Any]:
        """Calcula as métricas do dia civil atual (das 00:00 às 23:59 de hoje)."""
        with self.get_connection() as conn:
            # 1. Total de rotações e tempo médio no dia de hoje
            cursor_evt = conn.execute(
                """
                SELECT 
                    COUNT(*) AS total_rotacoes,
                    AVG(duracao_postura_anterior_seg) AS tempo_medio_seg
                FROM posture_events
                WHERE date(timestamp, 'localtime') = date('now', 'localtime')
                """
            )
            row_evt = cursor_evt.fetchone()
            total_rotacoes = int(row_evt["total_rotacoes"] or 0) if row_evt else 0
            tempo_medio_min = round((float(row_evt["tempo_medio_seg"] or 0.0) / 60.0), 1) if row_evt else 0.0

            # 2. Score de conformidade de alívio e total de alertas hoje
            cursor_tel = conn.execute(
                """
                SELECT 
                    COUNT(*) AS total_minutos,
                    SUM(status_alerta) AS total_alertas
                FROM telemetry_queue
                WHERE date(timestamp, 'localtime') = date('now', 'localtime')
                """
            )
            row_tel = cursor_tel.fetchone()
            total_minutos = int(row_tel["total_minutos"] or 0) if row_tel else 0
            total_alertas = int(row_tel["total_alertas"] or 0) if row_tel else 0

            # Se não houver minutos hoje, score padrão é 100%
            if total_minutos > 0:
                score_alivio = round(100.0 - (total_alertas * 100.0 / total_minutos), 1)
                score_alivio = max(0.0, min(100.0, score_alivio))
            else:
                score_alivio = 100.0

            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_rotations_today": total_rotacoes,
                "avg_posture_time_min": tempo_medio_min,
                "relief_score_pct": score_alivio,
                "total_alerts_today": total_alertas,
                "total_minutes_monitored": total_minutos,
            }

