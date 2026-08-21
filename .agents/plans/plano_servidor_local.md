# Plano de Implementação: Servidor Local (Edge & Central UTI)

> **Projeto:** Maca Inteligente — Sistema de Monitoramento e Prevenção de LPP  
> **Escopo:** Arquitetura do Servidor Local de Leito e Hub do Posto de Enfermagem  
> **Versão:** 1.0.0 | **Data:** 2026-08-21  

---

## 1. Visão Geral da Arquitetura

O sistema adota o padrão **Edge + Sync (Store-and-Forward)** para garantir disponibilidade 100% (mesmo com queda de Wi-Fi/Rede) e integridade dos dados clínicos em caso de corte repentino de energia.

```
┌─────────────────────────────────────────────────────────────┐
│                       DISPOSITIVO MACA (EDGE)               │
│  [Manta 32x64] ──(USB HID)──► [Python Service / Pipeline]   │
│                                      │                      │
│                           ┌──────────┴──────────┐           │
│                           ▼                     ▼           │
│                  [SQLite Local (WAL)]   [UI Eel Local]      │
│                           │                                 │
│                  [Sync Daemon (Async)]                      │
└───────────────────────────┼─────────────────────────────────┘
                            │ (mTLS / REST / WebSocket)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 POSTO DE ENFERMAGEM / SERVIDOR UTI          │
│             [FastAPI Ingestion Hub / Central Multi-Leito]   │
│                           │                                 │
│                           ▼                                 │
│            [PostgreSQL + TimescaleDB Hospitalar]            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes do Servidor Local

### 2.1 Módulo Edge (Na Maca / Totem)
* **Motor de Processamento:** Python 3 + NumPy vetorizado.
* **Armazenamento de Contingência (Store-and-Forward):** SQLite local operando em modo WAL (`PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;`).
* **API de Integração:** FastAPI local (`Python/api/server.py`) expondo endpoints REST e stream WebSocket para telemetria em tempo real.
* **Daemon de Sincronização (`sync_worker.py`):**
  * Consome a fila de eventos do SQLite local.
  * Agrupa em batches de 50 registros.
  * Transmite para o servidor central com autenticação por token (JWT/API Key).
  * Marca registros como `synced = 1` apenas após `HTTP 200 OK` do servidor central.

### 2.2 Hub Central da UTI (Servidor Local no Posto de Enfermagem)
* **Função:** Centralizar o status de todas as macas (ex: Macas 01 a 20 da UTI A) em uma única tela de monitoramento no posto de enfermagem.
* **Stack:** Docker Compose com FastAPI + PostgreSQL + Dashboard Central Web.

---

## 3. Estratégia de Resiliência e Tolerância a Quedas de Energia

1. **Escrita Atômica (Zero Corrupção):**
   * SQLite com WAL garante que transações com commit resistam a desligamentos bruscos.
   * Na reinicialização da maca, o SQLite faz auto-recovery transparente em milissegundos.
2. **Buffer Offline Infinito (Store-and-Forward):**
   * Se o Wi-Fi hospitalar oscilar ou cair, a maca continua gravando no SQLite local sem travar ou gerar latência para o usuário.
   * Assim que o link de rede restabelece, o daemon descarrega o histórico acumulado no servidor central.
3. **Watchdog de Serviço (Systemd / Docker):**
   * Configuração `Restart=always` e `RestartSec=3s` no serviço do sistema operacional para recuperação automática de processos.

---

## 4. Roteiro de Implementação Passo a Passo

### Fase 1: Camada de Persistência Local (Edge SQLite)
- [ ] Criar módulo `Python/services/db_local.py` gerenciando a conexão SQLite com WAL.
- [ ] Implementar tabelas `posture_events`, `tare_history` e `daily_kpi`.
- [ ] Conectar o `PostureMonitor` para gravar transições posturais e tempo estático automaticamente.

### Fase 2: Daemon de Sincronização (Store-and-Forward)
- [ ] Implementar `Python/services/sync_worker.py` executando em thread de segundo plano.
- [ ] Adicionar controle de estado (`synced_at`, tentativas de retry, backoff exponencial).
- [ ] Criar rota de envio em lote (`POST /api/v1/sync/events`).

### Fase 3: Hub do Posto de Enfermagem (Servidor Local UTI)
- [ ] Criar projeto do Servidor Central FastAPI com recepção multi-maca.
- [ ] Implementar WebSocket Broadcast para alertar a equipe de enfermagem em tempo real sobre alertas de tempo estático (LPP).
- [ ] Configuração do Docker Compose (`backend` + `db_postgres` + `nginx`).

### Fase 4: Validação e Testes de Campo
- [ ] Teste de corte de energia física sob escrita contínua.
- [ ] Teste de desconexão de cabo de rede por 4 horas seguido de religamento.
- [ ] Validação de carga com 20 macas simuladas transmitindo concorrentemente.
