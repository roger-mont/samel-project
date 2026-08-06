"""Entrypoint — inicializa Eel, serial reader e orquestra threads."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

import eel

import api.server as api_server
from config.settings import CalibrationParams, API_PORT
from services.serial_reader import SerialFrameReader, FakeSerialReader, HidFrameReader
from providers.posture_monitor import PostureMonitor
from bridge import start_reading_loop, set_calibration_ref, set_monitor_ref

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FSR Matrix — leitura e visualizacao de sensores de pressao",
    )
    parser.add_argument("--port", type=str, default="COM4", help="porta serial (ex: COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="baudrate da serial")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="usa dados sinteticos ao inves do hardware real",
    )
    parser.add_argument(
        "--hid",
        action="store_true",
        help="usa protocolo USB HID (colchao WangYing) ao inves da serial CSV",
    )
    return parser.parse_args()


def _start_api_server(port: int) -> None:
    """Inicia o servidor FastAPI em thread daemon separada."""
    thread = threading.Thread(
        target=api_server.start,
        kwargs={"port": port},
        daemon=True,
        name="fastapi-uvicorn",
    )
    thread.start()
    logger.info("API iniciada — http://localhost:%d/docs", port)


def main() -> None:
    args = parse_args()

    params = CalibrationParams()
    set_calibration_ref(params)
    monitor = PostureMonitor(
        tolerance=params.posture_tolerance,
        timeout_seconds=params.posture_timeout_seconds,
    )
    set_monitor_ref(monitor)

    if args.simulate:
        logger.info("modo simulacao ativo — dados sinteticos")
        reader = FakeSerialReader(change_interval=30.0)
    elif args.hid:
        logger.info("modo USB HID ativo — protocolo WangYing")
        reader = HidFrameReader()
    else:
        logger.info("conectando serial: %s @ %d", args.port, args.baud)
        reader = SerialFrameReader(port=args.port, baudrate=args.baud)

    start_reading_loop(reader, params, monitor)
    _start_api_server(API_PORT)

    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
    eel.init(web_dir)

    logger.info("iniciando webview — http://localhost:8080")
    try:
        eel.start(
            "index.html",
            size=(1400, 900),
            port=8080,
            mode="chrome",
            cmdline_args=["--disable-gpu"],
        )
    except EnvironmentError:
        logger.warning("chrome nao encontrado — tentando modo padrao do sistema")
        eel.start("index.html", size=(1400, 900), port=8080, mode="default")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
