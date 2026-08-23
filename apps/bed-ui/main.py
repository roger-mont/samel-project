"""Entrypoint da Interface Desktop (Totem / Visualizador de Leito)."""
from __future__ import annotations

import argparse
import logging
import os
import eel

import bridge  # Registra as funções @eel.expose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Samel Maca Inteligente — Interface Desktop",
    )
    parser.add_argument("--port", type=int, default=8080, help="Porta local do Eel Webview (default: 8080)")
    parser.add_argument("--kiosk", action="store_true", help="Inicia em tela cheia (Modo Kiosk)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    eel.init(web_dir)

    cmdline_args = ["--disable-gpu"]
    if args.kiosk:
        cmdline_args.append("--kiosk")

    logger.info("Iniciando Interface Desktop Samel — http://localhost:%d", args.port)
    try:
        eel.start(
            "index.html",
            size=(1400, 900),
            port=args.port,
            mode="chrome",
            cmdline_args=cmdline_args,
        )
    except EnvironmentError:
        logger.warning("Chrome não encontrado — iniciando navegador padrão")
        eel.start("index.html", size=(1400, 900), port=args.port, mode="default")


if __name__ == "__main__":
    main()
