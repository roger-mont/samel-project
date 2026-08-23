# Maca Hospitalar Inteligente — Sistema de Monitoramento e Prevenção de LPP

Repositório estruturado em módulos desacoplados para o ecossistema hospitalar:

```text
samel-project/
├── apps/
│   ├── edge-service/               # 🔌 Backend Edge (Docker: FastAPI + SQLite WAL + Store-and-Forward)
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── requirements.txt
│   │   ├── data/                   # Volume persistente do SQLite (.gitkeep)
│   │   └── src/
│   │       ├── api/                # FastAPI e WebSocket (/ws/telemetry)
│   │       ├── core/               # Configurações centralizadas
│   │       ├── database/           # SQLite anti-corrupção e FIFO
│   │       └── workers/            # Daemon Store-and-Forward (Gatilho Duplo)
│   │
│   └── bed-ui/                     # 🖥️ Aplicação Desktop/Totem Local (Eel UI + Hardware)
│       ├── main.py                 # Entrypoint da interface
│       ├── bridge.py               # Ponte Python <-> JS
│       ├── requirements.txt        # Dependências de UI e drivers
│       ├── static/                 # Frontend HTML/CSS/JS
│       ├── config/                 # Parâmetros e calibração
│       ├── providers/              # Monitores de postura
│       ├── services/               # Leitura serial, matemática e stores
│       └── storage/                # Calibração e histórico de sessões
│
├── tests/                          # 🧪 Suíte de Testes Automatizados
│   └── edge/                       # Testes do serviço edge
│
└── .agents/                        # Planos arquiteturais e documentação de engenharia
```

---

## 🚀 Como Executar

### 1. Backend Edge (Docker)
```bash
cd apps/edge-service
docker compose up -d --build
```
- Healthcheck: `http://localhost:8000/health`
- WebSocket de Telemetria: `ws://localhost:8000/ws/telemetry`

### 2. Interface Local da Maca (Desktop Eel)
```bash
cd apps/bed-ui
pip install -r requirements.txt
python main.py --simulate   # Modo simulação
# ou
python main.py --hid        # Modo USB HID real (colchão WangYing)
```

### 3. Execução dos Testes
```bash
pytest tests/edge/ -v
```
