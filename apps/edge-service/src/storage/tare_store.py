"""Persistência do offset de tara da maca hospitalar."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TARE_FILE = Path(__file__).resolve().parent / "tare.json"


def load_tare(path: str | Path = _DEFAULT_TARE_FILE) -> float:
    """Carrega o offset de tara em kg. Retorna 0.0 caso o arquivo não exista."""
    tare_path = Path(path)
    if not tare_path.exists():
        return 0.0
    try:
        data = json.loads(tare_path.read_text(encoding="utf-8"))
        return float(data.get("offset_kg", 0.0))
    except Exception as err:
        logger.warning("Falha ao ler tara de %s: %s", tare_path, err)
        return 0.0


def save_tare(offset_kg: float, path: str | Path = _DEFAULT_TARE_FILE) -> None:
    """Salva o offset de tara em kg de forma atômica."""
    tare_path = Path(path)
    try:
        tare_path.parent.mkdir(parents=True, exist_ok=True)
        tare_path.write_text(json.dumps({"offset_kg": round(offset_kg, 4)}, indent=2), encoding="utf-8")
        logger.info("Tara salva: %.4f kg em %s", offset_kg, tare_path)
    except Exception as err:
        logger.error("Falha ao salvar tara em %s: %s", tare_path, err)
