# Contexto e Arquitetura do Projeto: Manta de Sensores FSR (WangYing)

Este documento resume as especificações reais de hardware, a modelagem físico-matemática e os requisitos de software do sistema de monitoramento de pressão e peso em maca hospitalar.

---

## 1. Topologia de Hardware e Captação

O sistema utiliza uma manta hospitalar com matriz de Force Sensing Resistors (FSR) conectada via USB HID.

*   **Matriz Global:** 32 linhas × 64 colunas (2.048 sensores/células).
*   **Setores Físicos:** 8 blocos de 16×16 interligados.
*   **Interface USB:** Protocolo HID proprietário (VID: `0x1ACC` / 6860, PID: `0x1A4D` / 6733).
*   **Transmissão:** Pacotes de 64 bytes contendo coordenadas locais e intensidade de pressão pré-processada pelo microcontrolador na faixa de **0 a 255** a ~40 Hz (25 ms por frame).
*   **Unidade Mínima de Calibração Física:** Bloco 16×16 (acesso por slice matricial `BLOCK_REGIONS`).

---

## 2. Modelagem Matemática (Backend Python)

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

5.  **Critério de Estabilidade e Weight Lock (Metodologia §13):**
    O peso é considerado estável e travado quando preenche simultaneamente:
    *   $|m(t) - m(t-\Delta t)| < \epsilon$ ($\epsilon = 0,3 \text{ kg}$)
    *   $\text{variância}(\text{janela}) < 0,25 \text{ kg}^2$
    *   $\text{drift} < 0,15 \text{ kg/s}$
    *   Condição mantida ininterruptamente por $T_{min} \ge 10 \text{ s}$.

---

## 3. Regras de Negócio e Monitoramento Clínico

*   **Tara Dinâmica:** Capacidade de zerar a leitura base com colchão e lençol instalados (persistido em `tare.json`).
*   **Monitoramento de Postura Estática (Alerta de Escaras):** O sistema monitora continuamente o centro de pressão e distribuição da carga. Se o paciente permanecer na mesma postura por período superior ao timeout configurado (padrão: 60 s), dispara um alerta visual de redistribuição de pressão.
*   **Log Metrológico de Calibração:** Registro estruturado em CSV de cada ensaio experimental conforme a Metodologia §11 (pasta `sessions/`).

---

## 4. Arquitetura de Software

*   **Backend:** Python 3.10+ (`/Python`).
*   **Comunicação Frontend:** Ponte bidirecional via **Eel (Webview / Chromium)** com API REST e WebSocket para integração com sistemas externos.
*   **Componentes da UI:**
    *   Mapa de calor (*Heatmap*) 32×64 renderizado via HTML5 Canvas com interpolação espectral.
    *   Card de Peso com exibição de Leitura Atual, Força em Newtons e Peso Estável Travado (destaque verde).
    *   Barra de progresso de estabilização em tempo real.
    *   Cronômetro de postura com barra de timeout e alerta modal de redistribuição.
    *   Painel de ajuste de parâmetros em runtime (Deadzone, EMA Alpha, Tolerância e Timeout de Postura).
    *   Controles de Tara (aplicar/remover).