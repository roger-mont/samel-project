"""Cliente HTTP assíncrono para comunicação entre o Bed-UI e o Edge-Service."""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

EDGE_BASE_URL: str = os.getenv("EDGE_API_URL", "http://localhost:8000").rstrip("/")
MACA_ID: str = os.getenv("MACA_ID", "MACA-UTI-001")


def _http_request(method: str, path: str, data: dict[str, Any] | None = None, timeout: float = 2.0) -> Any:
    """Executa requisição HTTP síncrona leve usando a biblioteca padrão."""
    url = f"{EDGE_BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body_bytes = json.dumps(data).encode("utf-8") if data is not None else None

    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status in (200, 201):
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception) as ex:
        logger.debug("Edge service indisponível em %s: %s", path, ex)
        return None
    return None


class EdgeClient:
    """Cliente de integração da maca com o serviço Edge."""

    @staticmethod
    def get_live_telemetry() -> dict[str, Any] | None:
        """Obtém snapshot em tempo real do estado da maca."""
        return _http_request("GET", "/api/v1/telemetry/live", timeout=1.0)

    @staticmethod
    def trigger_tare() -> bool:
        """Dispara amostragem de tara no serviço Edge."""
        res = _http_request("POST", "/api/v1/calibration/tare", data={}, timeout=2.0)
        return res is not None and res.get("status") == "ok"

    @staticmethod
    def get_calibration_params() -> dict[str, Any] | None:
        """Obtém parâmetros de calibração vigentes."""
        return _http_request("GET", "/api/v1/calibration/params", timeout=2.0)

    @staticmethod
    def update_calibration_param(param_name: str, value: float) -> bool:
        """Atualiza parâmetro de calibração no Edge."""
        data = {"param_name": param_name, "value": value}
        res = _http_request("POST", "/api/v1/calibration/params", data=data, timeout=2.0)
        return res is not None and res.get("status") == "ok"

    @staticmethod
    def get_system_config() -> dict[str, Any] | None:
        """Obtém configurações operacionais do sistema (MACA_ID, URLs, etc)."""
        return _http_request("GET", "/api/v1/system/config", timeout=2.0)

    @staticmethod
    def update_system_config(config_data: dict[str, Any]) -> dict[str, Any] | None:
        """Atualiza configurações operacionais do sistema no Edge Service."""
        return _http_request("POST", "/api/v1/system/config", data=config_data, timeout=2.0)

    @staticmethod
    def send_telemetry(
        peso_kg: float | None,
        indice_postural: float | None,
        tempo_estatico_seg: int,
        status_alerta: bool,
        payload: dict[str, Any] | None = None,
        maca_id: str = MACA_ID,
    ) -> bool:
        """Envia pacote de telemetria consolidada de 1 minuto para o Edge SQLite."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "maca_id": maca_id,
            "peso_kg": peso_kg,
            "indice_postural": indice_postural,
            "tempo_estatico_seg": tempo_estatico_seg,
            "status_alerta": status_alerta,
            "payload": payload,
        }
        res = _http_request("POST", "/api/v1/telemetry", data=data, timeout=2.0)
        return res is not None and res.get("status") == "enqueued"

    @staticmethod
    def send_posture_event(
        postura_detectada: str,
        postura_anterior: str | None = None,
        duracao_anterior_seg: int = 0,
        regiao_pico: str | None = None,
        pico_pct: float | None = None,
        area_pct: float | None = None,
        indice_dist: float | None = None,
        houve_alerta: bool = False,
        maca_id: str = MACA_ID,
    ) -> bool:
        """Envia evento de transição postural ou alerta imediato."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "maca_id": maca_id,
            "postura_detectada": postura_detectada,
            "postura_anterior": postura_anterior,
            "duracao_anterior_seg": duracao_anterior_seg,
            "regiao_pico": regiao_pico,
            "pico_pct": pico_pct,
            "area_pct": area_pct,
            "indice_dist": indice_dist,
            "houve_alerta": houve_alerta,
        }
        res = _http_request("POST", "/api/v1/events/posture", data=data, timeout=2.0)
        return res is not None and res.get("status") == "recorded"

    @staticmethod
    def get_charts_analytics(window_hours: int = 4) -> dict[str, Any] | None:
        """Consulta as séries temporais consolidadas dos 3 gráficos de 4h (blocos de 30min)."""
        return _http_request("GET", f"/api/v1/analytics/charts?window_hours={window_hours}", timeout=2.0)

    @staticmethod
    def get_recent_events(limit: int = 5) -> list[dict[str, Any]] | None:
        """Consulta os últimos eventos posturais formatados."""
        return _http_request("GET", f"/api/v1/analytics/events?limit={limit}", timeout=2.0)

    @staticmethod
    def get_daily_summary() -> dict[str, Any] | None:
        """Consulta o resumo diário do dia civil corrente."""
        return _http_request("GET", "/api/v1/analytics/daily-summary", timeout=2.0)
