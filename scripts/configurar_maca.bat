@echo off
REM ===========================================================================
REM Samel Maca Inteligente — Assistente Interativo de Configuração de Leito
REM ===========================================================================
TITLE Samel - Assistente de Configuracao de Leito
COLOR 0B
cls

echo ===========================================================================
echo         SAMEL MACA INTELIGENTE - CONFIGURACAO DE LEITO / PRODUCAO
echo ===========================================================================
echo Este utilitario configura a identificacao desta maca e o servidor central.
echo.

set "CONFIG_FILE=%~dp0\..\apps\edge-service\config.json"

set /p MACA_ID_IN="Digite o identificador desta maca [Padrao: MACA-UTI-01]: "
if "%MACA_ID_IN%"=="" set "MACA_ID_IN=MACA-UTI-01"

set /p CENTRAL_URL_IN="Digite a URL do Servidor Central [Padrao: http://192.168.1.100:8000]: "
if "%CENTRAL_URL_IN%"=="" set "CENTRAL_URL_IN=http://192.168.1.100:8000"

set /p SYNC_SEC_IN="Intervalo de sincronizacao em segundos [Padrao: 60]: "
if "%SYNC_SEC_IN%"=="" set "SYNC_SEC_IN=60"

echo.
echo Gravando configuracoes em %CONFIG_FILE%...

(
echo {
echo   "maca_id": "%MACA_ID_IN%",
echo   "central_api_url": "%CENTRAL_URL_IN%",
echo   "edge_api_token": "samel_secret_token_123",
echo   "sync_interval_sec": %SYNC_SEC_IN%,
echo   "sync_batch_size": 50,
echo   "retention_days": 7,
echo   "max_retention_records": 50000,
echo   "log_level": "INFO"
echo }
) > "%CONFIG_FILE%"

echo.
echo [SUCESSO] Configuracao gravada com exito!
echo.
echo Maca ID:       %MACA_ID_IN%
echo Central URL:   %CENTRAL_URL_IN%
echo Sincronizacao: %SYNC_SEC_IN%s
echo.
echo Pressione qualquer tecla para sair...
pause >nul
