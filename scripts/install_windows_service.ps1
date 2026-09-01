<#
.SYNOPSIS
    Script de automação para registrar o Samel Edge Service como Serviço do Windows.
.DESCRIPTION
    Registra o backend FastAPI (com leitura de hardware e Store-and-Forward)
    para iniciar automaticamente no boot do Windows antes mesmo do logon.
.NOTES
    Requer privilégios de Administrador.
    Utiliza NSSM (Non-Sucking Service Manager) para gerenciamento robusto com auto-restart.
#>

param(
    [string]$ServiceName = "SamelEdgeService",
    [string]$DisplayName = "Samel Maca Inteligente - Edge Service",
    [string]$Description = "Serviço de borda 24/7 para aquisição USB HID, persistência local e Store-and-Forward da Maca Samel."
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir\.."
$EdgeSrcDir = "$ProjectRoot\apps\edge-service\src"
$VenvPython = "$ProjectRoot\venv\Scripts\python.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Instalador de Serviço do Windows — Samel Edge Service  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Verifica se está rodando como Administrador
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "Este script deve ser executado como Administrador no PowerShell."
    exit 1
}

# Verifica executável Python
if (-not (Test-Path $VenvPython)) {
    $VenvPython = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
}
if (-not $VenvPython) {
    Write-Error "Interpretador Python não encontrado."
    exit 1
}

Write-Host "[OK] Interpretador Python: $VenvPython" -ForegroundColor Green
Write-Host "[OK] Diretório de Origem: $EdgeSrcDir" -ForegroundColor Green

# Se NSSM estiver disponível, utiliza para criar o serviço com auto-restart
$nssmCmd = Get-Command nssm -ErrorAction SilentlyContinue
if ($nssmCmd) {
    Write-Host "Registrando serviço via NSSM..." -ForegroundColor Yellow
    nssm stop $ServiceName 2>$null
    nssm remove $ServiceName confirm 2>$null

    nssm install $ServiceName $VenvPython "-m uvicorn api.server:app --host 0.0.0.0 --port 8000"
    nssm set $ServiceName AppDirectory $EdgeSrcDir
    nssm set $ServiceName DisplayName $DisplayName
    nssm set $ServiceName Description $Description
    nssm set $ServiceName Start SERVICE_AUTO_START
    nssm set $ServiceName AppRestartDelay 3000
    nssm start $ServiceName
    Write-Host "[SUCESSO] Serviço '$ServiceName' instalado e iniciado com auto-restart!" -ForegroundColor Green
} else {
    Write-Host "NSSM não encontrado no PATH. Registrando via sc.exe (Windows Service Control)..." -ForegroundColor Yellow
    
    # Cria wrapper .bat se necessário
    $batPath = "$ScriptDir\run_edge_service.bat"
    
    sc.exe stop $ServiceName 2>$null
    sc.exe delete $ServiceName 2>$null
    
    # Cria serviço básico
    Write-Host "Dica: Para produção com auto-restart profissional, baixe o nssm (https://nssm.cc) e coloque no PATH." -ForegroundColor Gray
    Write-Host "Comando manual via NSSM:" -ForegroundColor White
    Write-Host "  nssm install $ServiceName `"$VenvPython`" `"-m uvicorn api.server:app --host 0.0.0.0 --port 8000`"" -ForegroundColor Cyan
    Write-Host "  nssm set $ServiceName AppDirectory `"$EdgeSrcDir`"" -ForegroundColor Cyan
}

Write-Host "Concluído!" -ForegroundColor Green
