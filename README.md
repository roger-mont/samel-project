# Maca Hospitalar Inteligente — Sistema de Monitoramento e Prevenção de LPP

Repositório estruturado em módulos desacoplados para o ecossistema hospitalar:

```text
samel-project/
├── apps/
│   ├── edge-service/               # 🔌 Backend Edge 24/7 (FastAPI + Aquisição USB HID + SQLite WAL + Store-and-Forward)
│   │   ├── config.json.example     # Template documentado de configuração
│   │   ├── .env.example            # Template de variáveis de ambiente
│   │   ├── requirements.txt
│   │   ├── data/                   # Volume persistente do SQLite (edge_maca.db)
│   │   └── src/
│   │       ├── api/                # FastAPI, Analytics REST, Config REST e WebSocket (/ws/telemetry)
│   │       ├── core/               # ConfigManager com busca em cascata e persistência
│   │       ├── database/           # SQLite anti-corrupção e FIFO 7 dias
│   │       ├── hardware/           # Drivers USB HID real (WangYing), Serial e Simulador
│   │       ├── math_engine/        # Pipeline físico-matemático (NumPy, EMA e Força)
│   │       ├── providers/          # Monitor de postura e alerta de escaras (LPP)
│   │       ├── storage/            # Persistência de calibração polinomial e tara
│   │       └── workers/            # Daemons em background (AcquisitionWorker & SyncWorker)
│   │
│   └── bed-ui/                     # 🖥️ Aplicação Desktop/Totem Local (Eel UI + Chart.js + Canvas)
│       ├── main.py                 # Entrypoint da interface
│       ├── bridge.py               # Ponte Eel <-> EdgeClient
│       ├── requirements.txt        # Dependências leves da UI
│       ├── static/                 # Frontend HTML5/CSS/JS (Modal de Configurações, Canvas & Chart.js)
│       └── services/               # EdgeClient (HTTP + WebSocket)
│
├── scripts/                        # 🛠️ Scripts de Operação e Instalação
│   ├── run_edge_service.bat        # Inicializa o Edge Service localmente
│   ├── run_ui.bat                  # Inicializa a Interface Desktop
│   ├── configurar_maca.bat         # Assistente interativo de configuração de leito
│   └── install_windows_service.ps1 # Registra o Edge Service como Serviço do Windows
│
├── tests/                          # 🧪 Suíte de Testes Automatizados
│   └── edge/                       # Testes de aquisição, analytics, persistência e configuração
│
└── .agents/                        # Planos arquiteturais, referências e documentação de engenharia
```

---

## ⚙️ Como Configurar a Maca (Produção & Desenvolvimento)

O sistema possui um **Gestor de Configuração em Cascata** que busca os parâmetros na seguinte ordem de prioridade:
1. Variáveis de ambiente do processo/sistema.
2. `apps/edge-service/config.json` (ou caminho definido em `SAMEL_CONFIG_PATH`).
3. `C:\ProgramData\Samel\config.json` (Padrão do Serviço do Windows em Produção).
4. Arquivo `.env`.
5. Valores padrão seguros em código.

### 3 Formas Simples de Configurar:

#### Opção 1: Pela Interface Visual da Maca (Sem fechar a aplicação)
1. Na tela do totem, clique no botão **⚙️ Configurações** no canto superior direito.
2. Ajuste o **Identificador da Maca** (ex: `MACA-UTI-05`) ou a **URL do Servidor Central**.
3. Clique em **"Salvar e Aplicar"**. A API salvará no `config.json` e atualizará o serviço instantaneamente sem reiniciar.

#### Opção 2: Pelo Assistente Rápido de Terminal
Execute com um duplo clique:
```bash
.\scripts\configurar_maca.bat
```
O assistente fará perguntas simples e gravará o arquivo `config.json` formatado.

#### Opção 3: Edição Direta do Arquivo
Abra `apps/edge-service/config.json` ou `C:\ProgramData\Samel\config.json` no Bloco de Notas:
```json
{
  "maca_id": "MACA-UTI-01",
  "central_api_url": "http://192.168.1.100:8000",
  "edge_api_token": "samel_secret_token_123",
  "sync_interval_sec": 60,
  "sync_batch_size": 50,
  "retention_days": 7
}
```

---

## 🚀 Como Executar

### 1. Inicializar o Serviço de Borda (Edge Service 24/7)
```bash
# Via script batch
.\scripts\run_edge_service.bat
```
- **Documentação Swagger:** `http://localhost:8000/docs`
- **Healthcheck Operacional:** `http://localhost:8000/health`
- **Via Expressa WebSocket:** `ws://localhost:8000/ws/telemetry`

---

### 2. Inicializar a Interface Desktop (Bed-UI)
```bash
# Via script batch
.\scripts\run_ui.bat
```

---

### 3. Instalação em Produção como Serviço do Windows
No computador/totem da maca hospitalar, execute o PowerShell como Administrador:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows_service.ps1
```

---

### 4. Execução dos Testes Automatizados
```bash
pytest tests/edge/ -v
```
