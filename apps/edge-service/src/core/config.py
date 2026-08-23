"""Configurações centralizadas e dinâmicas do Edge Service com suporte a busca em cascata e persistência."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tenta carregar python-dotenv se disponível
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_EDGE_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_PATH = str(_EDGE_ROOT / "data" / "edge_maca.db")


def _get_potential_config_paths() -> list[Path]:
    """Retorna lista ordenada de locais onde o config.json pode estar localizado."""
    paths: list[Path] = []
    
    # 1. Variável de ambiente explícita
    if env_path := os.getenv("SAMEL_CONFIG_PATH"):
        paths.append(Path(env_path))

    # 2. Pasta local do edge-service
    paths.append(_EDGE_ROOT / "config.json")
    paths.append(Path.cwd() / "config.json")

    # 3. Padrão de Produção Windows (%ProgramData%\Samel\config.json)
    if program_data := os.getenv("ProgramData"):
        paths.append(Path(program_data) / "Samel" / "config.json")
    else:
        paths.append(Path("C:/ProgramData/Samel/config.json"))

    return paths


class ConfigManager:
    """Gerenciador centralizado de configurações com suporte a leitura em cascata e salvamento em runtime."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_file_path: Path | None = None
        self._data: dict[str, Any] = {
            "maca_id": "MACA-EDGE-001",
            "sqlite_db_path": _DEFAULT_DB_PATH,
            "max_retention_records": 50000,
            "retention_days": 7,
            "central_api_url": "http://central-server:8000",
            "edge_api_token": "edge-default-token",
            "sync_batch_size": 50,
            "sync_interval_sec": 60.0,
            "cb_failure_threshold": 4,
            "cb_recovery_timeout_sec": 30.0,
            "server_host": "0.0.0.0",
            "server_port": 8000,
            "log_level": "INFO",
        }
        self.reload()

    def reload(self) -> None:
        """Recarrega configurações buscando em cascata de arquivos JSON e variáveis de ambiente."""
        with self._lock:
            # 1. Busca em arquivos config.json
            for path in _get_potential_config_paths():
                if path.is_file():
                    try:
                        content = json.loads(path.read_text(encoding="utf-8"))
                        self._data.update(content)
                        self._active_file_path = path
                        logger.info("Configurações carregadas com sucesso de: %s", path)
                        break
                    except Exception as err:
                        logger.warning("Falha ao ler %s: %s", path, err)

            # 2. Sobrescreve com variáveis de ambiente (se existirem)
            self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Aplica overrides de variáveis de ambiente com cast correto de tipos."""
        env_map = {
            "MACA_ID": ("maca_id", str),
            "SQLITE_DB_PATH": ("sqlite_db_path", str),
            "MAX_RETENTION_RECORDS": ("max_retention_records", int),
            "RETENTION_DAYS": ("retention_days", int),
            "CENTRAL_API_URL": ("central_api_url", str),
            "EDGE_API_TOKEN": ("edge_api_token", str),
            "SYNC_BATCH_SIZE": ("sync_batch_size", int),
            "SYNC_INTERVAL_SEC": ("sync_interval_sec", float),
            "CB_FAILURE_THRESHOLD": ("cb_failure_threshold", int),
            "CB_RECOVERY_TIMEOUT_SEC": ("cb_recovery_timeout_sec", float),
            "SERVER_HOST": ("server_host", str),
            "SERVER_PORT": ("server_port", int),
            "LOG_LEVEL": ("log_level", str),
        }
        for env_key, (data_key, cast_func) in env_map.items():
            if val := os.getenv(env_key):
                try:
                    self._data[data_key] = cast_func(val)
                except Exception:
                    pass

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def update_and_save(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Atualiza valores em memória e persiste no arquivo config.json ativo."""
        with self._lock:
            # Filtra chaves válidas e faz cast
            for k, v in updates.items():
                k_lower = k.lower()
                if k_lower in self._data:
                    current_type = type(self._data[k_lower])
                    try:
                        self._data[k_lower] = current_type(v)
                    except Exception:
                        self._data[k_lower] = v

            # Determina o arquivo de destino
            target_path = self._active_file_path or (_EDGE_ROOT / "config.json")
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
                self._active_file_path = target_path
                logger.info("Configurações persistidas com sucesso em: %s", target_path)
            except Exception as err:
                logger.error("Erro ao persistir configurações em %s: %s", target_path, err)

            return dict(self._data)


# Instância global gerenciada
config_manager = ConfigManager()

# Atalhos globais para compatibilidade com imports existentes
MACA_ID: str = str(config_manager.get("maca_id", "MACA-EDGE-001"))
SQLITE_DB_PATH: str = str(config_manager.get("sqlite_db_path", _DEFAULT_DB_PATH))
MAX_RETENTION_RECORDS: int = int(config_manager.get("max_retention_records", 50000))
RETENTION_DAYS: int = int(config_manager.get("retention_days", 7))
CENTRAL_API_URL: str = str(config_manager.get("central_api_url", "http://central-server:8000"))
EDGE_API_TOKEN: str = str(config_manager.get("edge_api_token", "edge-default-token"))
SYNC_BATCH_SIZE: int = int(config_manager.get("sync_batch_size", 50))
SYNC_INTERVAL_SEC: float = float(config_manager.get("sync_interval_sec", 60.0))
CB_FAILURE_THRESHOLD: int = int(config_manager.get("cb_failure_threshold", 4))
CB_RECOVERY_TIMEOUT_SEC: float = float(config_manager.get("cb_recovery_timeout_sec", 30.0))
SERVER_HOST: str = str(config_manager.get("server_host", "0.0.0.0"))
SERVER_PORT: int = int(config_manager.get("server_port", 8000))
LOG_LEVEL: str = str(config_manager.get("log_level", "INFO"))
