"""Daemon de Sincronização Store-and-Forward para o Módulo Edge.

Implementa a lógica de Gatilho Duplo (Threshold de 50 registros ou Timeout de 60s),
envio estritamente ordenado por timestamp ASC, confirmação atômica pós HTTP 200 OK
e Circuit Breaker para tolerância a quedas de Wi-Fi e indisponibilidade da API Central.

Mudança vs. versão anterior:
- Imports atualizados: ``database.db_local`` ao invés de ``services.db_local``.
- Constantes de configuração vêm de ``core.config`` ao invés de ler ``os.getenv``
  individualmente em cada parâmetro.
- Não cria instância singleton global; a instância é gerenciada pelo lifespan do FastAPI.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any

import httpx

from core.config import (
    CB_FAILURE_THRESHOLD,
    CB_RECOVERY_TIMEOUT_SEC,
    CENTRAL_API_URL,
    EDGE_API_TOKEN,
    MACA_ID,
    SYNC_BATCH_SIZE,
    SYNC_INTERVAL_SEC,
)
from database.db_local import LocalDatabase

logger = logging.getLogger(__name__)


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit Breaker simples para evitar sobrecarga de rede durante quedas de conexão."""

    def __init__(
        self,
        failure_threshold: int = CB_FAILURE_THRESHOLD,
        recovery_timeout_seconds: float = CB_RECOVERY_TIMEOUT_SEC,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0

    def can_attempt(self) -> bool:
        """Verifica se a tentativa de transmissão é permitida pelo estado atual do circuito."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            now = time.monotonic()
            if now - self.last_failure_time >= self.recovery_timeout_seconds:
                logger.info("Circuit Breaker transitando para HALF_OPEN: testando reconexão.")
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        return True

    def record_success(self) -> None:
        """Registra sucesso na comunicação, restaurando o circuito para CLOSED."""
        if self.state != CircuitState.CLOSED:
            logger.info("Circuit Breaker reestabelecido para CLOSED: rede central operacional.")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_failure(self) -> None:
        """Registra falha de rede/servidor e abre o circuito se o limite for atingido."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit Breaker ABERTO (falhas: %d). Pausando envios por %.1fs.",
                self.failure_count,
                self.recovery_timeout_seconds,
            )


class SyncWorker:
    """Worker assíncrono responsável pela sincronização Store-and-Forward."""

    def __init__(
        self,
        db: LocalDatabase,
        central_url: str = CENTRAL_API_URL,
        api_token: str = EDGE_API_TOKEN,
        batch_size: int = SYNC_BATCH_SIZE,
        timeout_seconds: float = SYNC_INTERVAL_SEC,
        maca_id: str = MACA_ID,
    ) -> None:
        self.db = db
        self.central_url = central_url.rstrip("/")
        self.api_token = api_token
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.maca_id = maca_id

        self.circuit_breaker = CircuitBreaker()
        self._last_sync_time: float = time.monotonic()
        self._is_running: bool = False
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        """Inicia o loop assíncrono do worker em segundo plano."""
        if self._is_running:
            return
        self._is_running = True
        self._last_sync_time = time.monotonic()
        self._task = asyncio.create_task(self._run_loop(), name="SyncWorkerTask")
        logger.info(
            "SyncWorker iniciado [Maca: %s, Batch: %d, Interval: %.1fs, Central: %s]",
            self.maca_id, self.batch_size, self.timeout_seconds, self.central_url,
        )

    async def stop(self) -> None:
        """Interrompe graciosamente o worker de sincronização."""
        if not self._is_running:
            return
        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SyncWorker encerrado com sucesso.")

    async def _run_loop(self) -> None:
        """Loop principal verificando os gatilhos duplo (threshold de 50 ou timeout de 60s)."""
        while self._is_running:
            try:
                await self.check_and_sync()
            except asyncio.CancelledError:
                break
            except Exception as ex:
                logger.error("Erro inesperado no loop do SyncWorker: %s", ex, exc_info=True)

            await asyncio.sleep(2.0)

    async def check_and_sync(self) -> bool:
        """Avalia os critérios de Gatilho Duplo e executa sincronização se atendidos."""
        now = time.monotonic()
        unsynced_telemetry_count = self.db.count_unsynced_telemetry()
        unsynced_events_count = self.db.count_unsynced_events()
        total_unsynced = unsynced_telemetry_count + unsynced_events_count

        if total_unsynced == 0:
            return False

        trigger_threshold = total_unsynced >= self.batch_size
        trigger_timeout = (
            (now - self._last_sync_time) >= self.timeout_seconds and total_unsynced > 0
        )

        if not (trigger_threshold or trigger_timeout):
            return False

        trigger_reason = (
            "THRESHOLD (>= 50)" if trigger_threshold else f"TIMEOUT ({self.timeout_seconds}s)"
        )
        logger.info(
            "Sincronização acionada via %s. Pendências: %d tel, %d evt.",
            trigger_reason, unsynced_telemetry_count, unsynced_events_count,
        )

        return await self.perform_sync_cycle()

    async def perform_sync_cycle(self) -> bool:
        """Executa um ciclo completo de transmissão de dados pendentes com Circuit Breaker."""
        if not self.circuit_breaker.can_attempt():
            logger.debug("Sincronização ignorada: Circuit Breaker está ABERTO.")
            return False

        synced_any = False

        events_batch = self.db.get_unsynced_events(limit=self.batch_size)
        if events_batch:
            success = await self._send_events_batch(events_batch)
            if success:
                synced_any = True
            else:
                return False

        telemetry_batch = self.db.get_unsynced_telemetry(limit=self.batch_size)
        if telemetry_batch:
            success = await self._send_telemetry_batch(telemetry_batch)
            if success:
                synced_any = True
            else:
                return False

        if synced_any:
            self._last_sync_time = time.monotonic()
            self.circuit_breaker.record_success()

        return synced_any

    async def _send_telemetry_batch(self, batch: list[dict[str, Any]]) -> bool:
        """Envia lote de telemetria para a API Central e atualiza o banco local pós HTTP 200."""
        record_ids = [item["id"] for item in batch]
        endpoint = f"{self.central_url}/api/v1/sync/telemetry"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "X-Maca-ID": self.maca_id,
            "Content-Type": "application/json",
        }
        payload = {"maca_id": self.maca_id, "batch_size": len(batch), "records": batch}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

            if response.status_code in (200, 201):
                self.db.mark_telemetry_synced(record_ids)
                logger.info("Lote de %d telemetrias sincronizado (HTTP %d).", len(batch), response.status_code)
                return True

            logger.warning("Falha ao sincronizar telemetria: HTTP %d", response.status_code)
            self.db.increment_retry_count("telemetry_queue", record_ids)
            self.circuit_breaker.record_failure()
            return False

        except (httpx.RequestError, httpx.TimeoutException) as ex:
            logger.warning("Falha de rede na API Central (%s): %s", endpoint, ex)
            self.db.increment_retry_count("telemetry_queue", record_ids)
            self.circuit_breaker.record_failure()
            return False

    async def _send_events_batch(self, batch: list[dict[str, Any]]) -> bool:
        """Envia lote de eventos posturais para a API Central e atualiza pós HTTP 200."""
        event_ids = [item["id"] for item in batch]
        endpoint = f"{self.central_url}/api/v1/sync/events"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "X-Maca-ID": self.maca_id,
            "Content-Type": "application/json",
        }
        payload = {"maca_id": self.maca_id, "batch_size": len(batch), "events": batch}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

            if response.status_code in (200, 201):
                self.db.mark_events_synced(event_ids)
                logger.info("Lote de %d eventos sincronizado (HTTP %d).", len(batch), response.status_code)
                return True

            logger.warning("Falha ao sincronizar eventos: HTTP %d", response.status_code)
            self.db.increment_retry_count("posture_events", event_ids)
            self.circuit_breaker.record_failure()
            return False

        except (httpx.RequestError, httpx.TimeoutException) as ex:
            logger.warning("Falha de rede ao enviar eventos (%s): %s", endpoint, ex)
            self.db.increment_retry_count("posture_events", event_ids)
            self.circuit_breaker.record_failure()
            return False
