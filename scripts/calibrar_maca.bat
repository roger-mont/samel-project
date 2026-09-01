@echo off
REM ===========================================================================
REM Samel Maca Inteligente — Assistente de Coleta de Calibração
REM ===========================================================================
TITLE Samel - Calibracao de Peso
COLOR 0A
cls

cd /d "%~dp0\.."

if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual nao encontrado em .\venv
    echo Execute a instalacao de dependencias antes de continuar.
    pause
    exit /b 1
)

echo ===========================================================================
echo         SAMEL MACA INTELIGENTE - ASSISTENTE DE CALIBRACAO
echo ===========================================================================
echo.
echo Inicializando assistente de calibracao de blocos e Machine Learning...
echo.

.\venv\Scripts\python.exe apps/edge-service/scripts/coletar_calibracao.py

echo.
echo Pressione qualquer tecla para sair...
pause >nul
