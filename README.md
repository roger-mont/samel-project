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

### 4. Calibração, tara e experimentos

- **Peso só quando válido:** `GET /api/v1/telemetry/live` devolve `weight_kg` preenchido apenas com `weight_status = VALID`. Os outros estados são `NO_LOAD`, `LOAD_DETECTED`, `STABILIZING`, `MOVEMENT` e `NOT_VALIDATED`. As flags ficam em `measurement_quality`, e o último peso válido vem em `last_valid`.
- **Tara (baseline por célula):** `POST /api/v1/calibration/tare` com a maca vazia. Ela dura `t_tare_s` segundos (padrão 5). A tara em vigor aparece em `GET /api/v1/calibration/tare`.
- **Parâmetros de estabilidade:** `GET/POST /api/v1/calibration/params`. Com `"persist": true`, o valor é gravado em `config.json`, no bloco `acquisition`.
- **Curva de calibração (RAW → massa local):** só valem as calibrações de célula de `docs/<n>. Célula AxB/` (A = linha da matriz, 0-31; B = coluna, 0-63; origem no setor 4). Cada setor ganha a curva da(s) sua(s) célula(s): isotônica + PCHIP, ramo de subida. Para regerar `src/storage/block_curves.json`:
  ```bash
  python apps/edge-service/scripts/gerar_curvas_bloco.py
  ```
- **Massa local → massa total:** `M = Σ_s α_s · Σ_{ij∈s} f_s(x_ij)`. Os ganhos α_s ficam em 1 até serem ajustados com cargas de peso conhecido. Regerar as curvas volta os ganhos para 1.
  - **Pela tela (recomendado):** card "Ajuste de ganho α" de `/calibracao`. Faça a tara, registre pontos (peso real + carga estável), calcule e aplique. Os pontos ficam em `data/gain_points.json` e o α é gravado em `block_curves.json` só ao clicar em "Aplicar ganhos".
  - **Por experimentos gravados:** com `known_mass_g` marcado em `/experiments/mark`; o script só grava com `--gravar`:
    ```bash
    python apps/edge-service/scripts/ajustar_ganho_massa_total.py apps/edge-service/data/experiments/<sessão> [--modo setor] [--gravar]
    ```
- **Comparação de modelos (curvas e ML):** `python apps/edge-service/scripts/comparar_modelos_massa.py`.
- **Tela de calibração ao vivo:** `http://localhost:8000/calibracao`. Mostra as 8 curvas juntas, cada setor com seus pontos, as curvas dos modelos candidatos (curvas e ML, ajuste por setor ou global) e a massa da leitura atual por todos os modelos. Para testar com carga conhecida, informe o peso real e use "Registrar leitura"; o resumo dá o erro de cada modelo e baixa um CSV. Atalhos: `?view=all|sector|models&setor=1..8&modo=sector|global`. A primeira leitura da comparação treina os modelos (alguns segundos).
- **Experimentos (RAW completo):** use `POST /api/v1/experiments/start`, depois `/mark` (ciclo, `known_mass_g`, `loading_direction` e, nos ensaios de célula, `target_cell` como `"12x34"`) e por fim `/stop`. Os dados ficam em `apps/edge-service/data/experiments/` e não vão para o git.
- **Diagnóstico da calibração (orientações de 15/10, `docs/Orientacoes_Roger_Math_Engine_Maca_Inteligente.pdf`):**
  - `GET /api/v1/telemetry/live` também traz L1…L8 (`sector_local_mass_kg`, sem α), `mass_without_gain_kg`, `n_active_cells`, `n_out_of_calibration` (e por setor), `out_of_calibration_mass_pct`, `n_low_sensitivity`, `n_low_support`, `n_digital_saturation`, `hysteresis_up_minus_down_kg`, `residual_mass_empty` e `calibration_alerts`.
  - `SATURATION` continua sendo só RAW ≥ 255. `LOW_SENSITIVITY` agora vem do dm/dRAW da curva (parâmetro `low_sensitivity_g_per_raw`, padrão provisório 10 g por RAW; 0 desliga) e `LOW_DATA_SUPPORT` marca células abaixo do primeiro nó nos setores em que a menor massa deu RAW 0. As duas são só informativas.
  - `GET /api/v1/diagnostics/cells?limit=N` lista cada célula ativa: RAW, baseline, RAW corrigido, massa local (UP e DOWN), dm/dRAW, situação na curva e flags.
  - `GET /api/v1/diagnostics/residual` dá a massa que o modelo produz com a maca vazia desde a tara (média, desvio, máximo e por setor). Para o ensaio de zero: tara, `POST /api/v1/diagnostics/residual/reset` e 10 minutos sem carga.
  - `GET /api/v1/diagnostics/calibration` audita as curvas: correção isotônica (medido × regularizado, por setor e por célula, UP e DOWN), massas sem resposta, histerese, baixa sensibilidade e setores com uma célula só.
  - Na tela `/calibracao`, o card "Diagnóstico da calibração" mostra tudo isso ao vivo, e os pontos de carga ganharam a posição (centro, cabeceira, peseira, esquerda, direita), com o resumo de repetição e posição (erro espacial × erro de ganho).
  - Ensaios por célula: `python apps/edge-service/scripts/exportar_ensaios_celula.py data/experiments/<sessão> --saida ensaios.csv` gera um patamar por célula, ciclo, sentido e massa. `python apps/edge-service/scripts/comparar_generalizacao_celulas.py [--ensaios ensaios.csv]` compara, deixando cada célula de fora, a curva global (A), a do setor (B) e a do setor com ganho individual (C).

- **Diagnóstico da aquisição (parecer de 20/10, `docs/Parecer_Linha_Tiago_Acoes_Roger_Maca_Inteligente.pdf`):** as curvas e o PCHIP ficam congelados; primeiro se prova que os 2.048 dados chegam íntegros e que a tara é válida.
  - A leitura USB roda numa thread própria (`HidRxThread`), separada do processamento, e conta por setor: pacotes, pacotes/s, atualizações completas (as 256 células recebidas), idade da última, intervalo real entre atualizações (p50/p95/máx) e varreduras incompletas (célula repetida antes de cobrir o setor; o protocolo não tem número de sequência, então isso é o indício de pacote perdido). `GET /api/v1/diagnostics/acquisition` mostra tudo.
  - `STALE_SECTOR`: setor ativo sem atualização completa há mais que `stale_sector_s` (padrão provisório 1 s; 0 desliga) invalida o peso. Ajuste o limite pelo `update_interval_ms` medido na manta real.
  - Tara: `POST /api/v1/calibration/tare` devolve 409 com o motivo se a maca não estiver em `SEM_CARGA` contínuo há `tare_empty_min_s` (padrão 3 s), se houver movimento, setor parado ou manta desconectada. Carga, movimento ou setor parado durante a coleta cancelam a tara. A primeira tara (sem tara anterior) só exige estabilidade. Histórico em `GET /api/v1/calibration/tare/events` e em `data/diagnostics/tare_events.jsonl`.
  - Retorno ao baseline: `GET /api/v1/diagnostics/baseline-return` separa, por setor, células acima da tara que recebem valores novos (memória/creep) das que não recebem pacote (falha de atualização). Começa sozinho quando a maca volta a vazia; `POST /api/v1/diagnostics/baseline-return/start` marca "carga retirada agora".
  - Log contínuo: `data/diagnostics/diag_AAAAMMDD.csv`, uma linha a cada 0,5 s com peso validado, peso atual, último válido, L1…L8, RAW corrigido e células por setor, idade e pacotes por setor, tara (`tare_id`, horário, `baseline_sum`) e flags. Dias anteriores viram `.csv.gz` e saem depois de 7 dias (`diagnostic_log` em `config.json`). Para o teste decisivo, `POST /api/v1/diagnostics/reference` grava a massa de referência em cada linha.
  - Interface: o totem e a tela `/calibracao` mostram separados o peso validado, o peso atual (não validado), o status e o último peso válido com o horário. A tela tem o card "Diagnóstico da aquisição" com tudo isso ao vivo.
  - `GET /api/v1/diagnostics/calibration` também acusa CSVs de calibração idênticos (`duplicate_sources`), como os de 12X34 e 13X26.

### 5. Execução dos Testes Automatizados
```bash
pytest tests/edge/ -v
```
