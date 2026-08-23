@echo off
REM ===========================================================================
REM Samel Maca Inteligente — Inicializador do Edge Service (Segundo Plano / Dev)
REM ===========================================================================
TITLE Samel Edge Service (Backend 24/7)

cd /d "%~dp0\..\apps\edge-service\src"

if exist "..\..\..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\..\venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [Samel Edge] Iniciando servidor FastAPI e AcquisitionWorker na porta 8000...
%PYTHON_EXE% -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --log-level info
pause
