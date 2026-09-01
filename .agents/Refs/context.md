# Contexto e Arquitetura do Projeto: Manta de Sensores FSR (WangYing)

Este documento resume as especificações reais de hardware, a modelagem físico-matemática, os requisitos clínicos e as decisões arquiteturais do sistema de monitoramento de pressão, peso e prevenção de Lesões por Pressão (LPP) em maca hospitalar.

---

## 1. Topologia de Hardware e Captação

O sistema utiliza uma manta hospitalar com matriz de Force Sensing Resistors (FSR) conectada via USB HID.

*   **Matriz Global:** 32 linhas × 64 colunas (2.048 sensores/células).
*   **Setores Físicos:** 8 blocos de 16×16 interligados.
*   **Interface USB:** Protocolo HID proprietário (VID: `0x1ACC` / 6860, PID: `0x1A4D` / 6733).
*   **Transmissão:** Pacotes de 64 bytes contendo coordenadas locais e intensidade de pressão pré-processada pelo microcontrolador na faixa de **0 a 255** a ~40 Hz (25 ms por frame).
*   **Unidade Mínima de Calibração Física:** Bloco 16×16 (acesso por slice matricial `BLOCK_REGIONS`).

---

## 2. Modelagem Matemática (Pipeline Numérico)

Como o protocolo HID entrega intensidades 0–255 diretamente, o pipeline aplica processamento vetorizado com **NumPy/SciPy**:

### Pipeline Físico-Matemático

1.  **Filtragem de Ruído de Fundo (Deadzone):**
    Sensores com leitura abaixo de `deadzone_threshold` são zerados para eliminar ruídos elétricos e peso do lençol:
    $$S_{net}(x,y) = \begin{cases} S(x,y), & \text{se } S(x,y) > \text{limiar} \\ 0, & \text{caso contrário} \end{cases}$$

2.  **Calibração Empírica por Bloco (Sinal → Newton):**
    Cada bloco $k$ possui uma curva polinomial empírica que converte a soma líquida de leituras do bloco em força ($N$):
    $$F_k = \text{polyval}(P_k, \sum S_{net, k} - \text{tara}_k)$$

3.  **Força Total e Massa Equivalente:**
    A força total sobre a manta é somada e convertida em massa ($kg$) dividindo pela aceleração da gravidade ($g = 9,81 \text{ m/s}^2$):
    $$F_{total} = \sum_{k=1}^{8} F_k \quad (\text{em Newtons})$$
    $$m_{física} = \frac{F_{total}}{9,81} \quad (\text{em kg})$$

4.  **Suavização Temporal (EMA):**
    $$m_{EMA}(t) = \alpha \cdot m_{física}(t) + (1 - \alpha) \cdot m_{EMA}(t-1)$$

5.  **Média Móvel Inteligente de 60s com Reset Rápido:**
    Para garantir precisão metrológica, o peso é acumulado em janela deslizante de 60 segundos. Se um degrau $> 2,0 \text{ kg}$ for detectado (ex: entrada/saída de paciente ou colocação de equipamento), o buffer é zerado instantaneamente para estabilização imediata.

6.  **Critério de Estabilidade e Weight Lock (Metodologia §13):**
    O peso é considerado estável e travado quando preenche simultaneamente:
    *   $|m(t) - m(t-\Delta t)| < \epsilon$ ($\epsilon = 0,3 \text{ kg}$)
    *   $\text{variância}(\text{janela}) < 0,25 \text{ kg}^2$
    *   $\text{drift} < 0,15 \text{ kg/s}$
    *   Condição mantida ininterruptamente por $T_{min} \ge 10 \text{ s}$.

---

## 3. Gestão Centralizada de Configuração para Produção

Para garantir fácil manutenção no hospital, o sistema adota um **ConfigManager com busca em cascata**:

1.  **Prioridade 1: Variáveis de Ambiente do SO / Processo.**
2.  **Prioridade 2: `config.json` local ou apontado por `SAMEL_CONFIG_PATH`.**
3.  **Prioridade 3: `C:\ProgramData\Samel\config.json` (Padrão de Produção Windows).**
4.  **Prioridade 4: `.env`.**
5.  **Prioridade 5: Valores padrão seguros em código.**

### Modos de Configuração Disponíveis:
*   **Via Tela do Totem (Modal Técnico):** A equipe médica/técnica altera o ID do leito e IP do servidor direto na interface via `POST /api/v1/system/config`.
*   **Via Assistente Interativo:** Script [`scripts/configurar_maca.bat`](file:///d:/PrOJETOS/samel-project/scripts/configurar_maca.bat).
*   **Via Arquivo Texto:** Edição no Bloco de Notas em `config.json`.

---

## 4. Arquitetura de Software (Aquisição 24/7 no Edge Service)

O sistema opera em **dois módulos desacoplados nativamente no Windows**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        MACA HOSPITALAR (WINDOWS)                       │
│                                                                        │
│  ┌─────────────────────────┐          ┌─────────────────────────────┐  │
│  │   apps/bed-ui           │          │ apps/edge-service           │  │
│  │   (Visualizador Kiosk)  │WebSocket │ (Serviço do Windows 24/7)   │  │
│  │  - Interface HTML5/JS   │◀────────▶│ - Leitura USB HID WangYing  │  │
│  │  - Renderiza Heatmap    │  REST    │ - Pipeline NumPy e Física   │  │
│  │  - Modal de Configuração│          │ - SQLite WAL (FIFO 7 dias)  │  │
│  │  - Pode abrir e fechar  │          │ - Store-and-Forward Daemon  │  │
│  └─────────────────────────┘          └──────────────┬──────────────┘  │
└──────────────────────────────────────────────────────┼─────────────────┘
                                                       │ Store-and-Forward
                                                       ▼ (Batches a cada 60s)
                                     ┌───────────────────────────────────┐
                                     │  Servidor Central / UTI / Nuvem   │
                                     │  (PostgreSQL / TimescaleDB)       │
                                     └───────────────────────────────────┘
```