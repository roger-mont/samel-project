@echo off
REM ===========================================================================
REM Samel Maca Inteligente — Inicializador da Interface Desktop (Totem)
REM ===========================================================================
TITLE Samel Maca UI (Visualizador Desktop)

cd /d "%~dp0\..\apps\bed-ui"

if exist "..\..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [Samel UI] Iniciando interface gráfica...
%PYTHON_EXE% main.py
pause
