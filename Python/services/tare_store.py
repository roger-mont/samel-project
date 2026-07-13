"""Persistência da tara em disco — load/save JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import TARE_FILE

logger = logging.getLogger(__name__)


def load_tare() -> float:
    """Carrega offset de tara do arquivo. Retorna 0.0 se inexistente ou inválido."""
    path = Path(TARE_FILE)
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("offset_kg", 0.0))
    except Exception as err:
        logger.error("falha ao carregar tara: %s", err)
        return 0.0


def save_tare(offset_kg: float) -> None:
    """Persiste offset de tara no arquivo JSON."""
    path = Path(TARE_FILE)
    try:
        path.write_text(
            json.dumps({"offset_kg": offset_kg}, indent=2),
            encoding="utf-8",
        )
        logger.info("tara salva: %.4f kg → %s", offset_kg, path)
    except Exception as err:
        logger.error("falha ao salvar tara: %s", err)
