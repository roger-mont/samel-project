# Documentação Matemática — Sistema Palmilex

> **Projeto:** Matriz de Sensores FSR (Force Sensing Resistors)  
> **Plataforma de Hardware:** ESP32-C3 Super Mini + Multiplexador CD74HC4067  
> **Plataforma de Software:** Python 3 · NumPy · Eel (Webview)  
> **Última revisão:** 2026-07-13

---

## Sumário

1. [Arquitetura de Hardware e Captação](#1-arquitetura-de-hardware-e-captação)
2. [Pipeline Matemático de Conversão](#2-pipeline-matemático-de-conversão)
   - [2.1 Tensão de Saída (V_out)](#21-tensão-de-saída)
   - [2.2 Resistência do Sensor (R_FSR)](#22-resistência-do-sensor)
   - [2.3 Condutância (C)](#23-condutância)
   - [2.4 Força em Newtons (F)](#24-força-em-newtons)
   - [2.5 Filtragem de Ruído — Zona Morta](#25-filtragem-de-ruído--zona-morta)
   - [2.6 Massa Total (Kg)](#26-massa-total)
3. [Algoritmo de Suavização — Filtro EMA](#3-algoritmo-de-suavização--filtro-ema)
4. [Referência Rápida de Parâmetros](#4-referência-rápida-de-parâmetros)
5. [Rastreabilidade de Código-Fonte](#5-rastreabilidade-de-código-fonte)

---

## 1. Arquitetura de Hardware e Captação

### 1.1 Topologia do Circuito

O sistema utiliza uma **matriz passiva de sensores FSR** organizada em linhas e colunas. A leitura de cada célula é obtida por **varredura ativa (polling)**: o microcontrolador ESP32-C3 energiza uma linha por vez e percorre as colunas sequencialmente através do multiplexador analógico de 16 canais CD74HC4067.

O circuito de leitura de cada célula forma um **divisor de tensão resistivo** com a seguinte configuração:

```
 VCC (3.3V)
   │
 ┌─┴─┐
 │FSR │  ← Sensor de força (resistência variável)
 └─┬─┘
   │
   ├──── Pino ADC (leitura de V_out)
   │
 ┌─┴─┐
 │10kΩ│  ← Resistor fixo de pull-down
 └─┬─┘
   │
  GND
```

- **Polo positivo (VCC):** Sensor FSR. Sua resistência diminui proporcionalmente à força aplicada.
- **Polo negativo (GND):** Resistor fixo de pull-down ($R_{pull} = 10\,k\Omega$).
- **Ponto de leitura:** O nó intermediário entre o FSR e o resistor fixo é conectado ao pino ADC do ESP32-C3.

### 1.2 Taxa de Varredura e Transmissão

| Parâmetro                  | Valor                          |
| :------------------------- | :----------------------------- |
| Resolução do ADC           | 12 bits (0 – 4095)            |
| Tensão de referência (VCC) | 3.3 V                          |
| Taxa de varredura efetiva  | 50 – 100 Hz (frames completos) |
| Interface de transmissão   | Serial/USB @ 115200 baud       |
| Formato do frame           | Pacote binário (segment_id + triplas x, y, valor) |

O microcontrolador envia **um frame completo da matriz** por ciclo de varredura. O PC recebe exclusivamente os **valores brutos (raw) do ADC** — nenhum processamento matemático ocorre no firmware. Todo o pipeline de conversão é executado no software Python.

---

## 2. Pipeline Matemático de Conversão

O pipeline transforma os valores inteiros brutos do ADC em massa real (Kg) através de uma cadeia determinística de cinco transformações algébricas e uma etapa de filtragem. Todas as operações são executadas de forma **vetorizada** sobre a matriz inteira usando NumPy, eliminando loops explícitos.

> **Arquivo-fonte:** `Python/services/math_pipeline.py`

```
ADC (int) → V_out (V) → R_FSR (Ω) → C (S) → F (N) → deadzone → Massa (Kg)
```

### 2.1 Tensão de Saída

O valor inteiro do ADC de 12 bits é convertido para a tensão real presente no pino de leitura:

$$V_{out} = Valor_{ADC} \times \frac{V_{CC}}{Resolução_{ADC}}$$

Onde:
- $Valor_{ADC}$ — leitura inteira do conversor (0 a 4095)
- $V_{CC}$ — tensão de alimentação do divisor (3.3 V)
- $Resolução_{ADC}$ — valor máximo do conversor (4095 para 12 bits)

**Exemplo numérico:** Para $Valor_{ADC} = 2048$:

$$V_{out} = 2048 \times \frac{3.3}{4095} \approx 1.649\,V$$

**Implementação:**

```python
v_out = adc_matrix * (vcc / resolution)
```

**Caso-limite:** Se a resolução for zero, a função retorna uma matriz nula para evitar divisão por zero.

---

### 2.2 Resistência do Sensor

Com a tensão no ponto intermediário do divisor conhecida, isolamos a resistência do FSR aplicando a **equação do divisor de tensão**:

$$V_{out} = V_{CC} \times \frac{R_{pull}}{R_{FSR} + R_{pull}}$$

Resolvendo para $R_{FSR}$:

$$R_{FSR} = R_{pull} \times \left(\frac{V_{CC}}{V_{out}} - 1\right)$$

Onde:
- $R_{pull}$ — resistor fixo de pull-down (10 kΩ)
- $V_{CC}$ — tensão de alimentação (3.3 V)
- $V_{out}$ — tensão calculada no passo anterior

**Comportamento físico:**
- **Sem pressão:** $R_{FSR} \to \infty$ (circuito aberto), $V_{out} \to 0$.
- **Pressão máxima:** $R_{FSR} \to 0$, $V_{out} \to V_{CC}$.

**Implementação:**

```python
# Guard: V_out == 0 → R_FSR infinito → sensor sem pressão
safe_vout = np.where(v_out > 0, v_out, np.inf)
r_fsr = pulldown * (vcc / safe_vout - 1.0)
```

A cláusula de guarda (`np.inf`) garante que divisões por zero sejam tratadas como resistência infinita (ausência de pressão), propagando corretamente o valor zero nas etapas seguintes.

---

### 2.3 Condutância

A condutância ($C$) é o **inverso da resistência**:

$$C = \frac{1}{R_{FSR}}$$

**Por que usamos condutância em vez de resistência?**

A resposta dos sensores FSR apresenta uma **relação altamente não-linear entre resistência e força** — a curva $R_{FSR} \times F$ é uma hipérbole. Entretanto, a relação entre **condutância e força é aproximadamente linear** dentro da faixa operacional do sensor. Essa linearidade permite a aplicação direta de uma regressão linear simples ($y = mx + b$) para a calibração, eliminando a necessidade de ajustes polinomiais ou logarítmicos complexos.

```
  R (Ω)                         C (S)
   │                              │
   │╲                             │        ╱
   │ ╲                            │      ╱
   │  ╲                           │    ╱
   │    ╲                         │  ╱
   │      ╲___________            │╱___________
   └──────────────── F (N)        └──────────────── F (N)
     Hiperbólica (não-linear)       Linear (calibrável)
```

**Implementação:**

```python
# Guard: R_FSR <= 0 → divisão por zero na condutância
safe_rfsr = np.where(r_fsr > 0, r_fsr, np.inf)
conductance = 1.0 / safe_rfsr
```

---

### 2.4 Força em Newtons

A força exercida em cada ponto $(x, y)$ da matriz é obtida por **regressão linear** sobre a condutância:

$$F(x,y) = m \times C + b$$

Onde:
- $m$ — fator de escala (slope), determinado empiricamente durante a calibração
- $b$ — offset (intercept), compensa o viés residual do sensor
- $C$ — condutância calculada no passo anterior

**Procedimento de calibração:** Os coeficientes $m$ e $b$ são obtidos empiricamente:

1. Aplicar massas conhecidas sobre o sensor (ex: 0 kg, 1 kg, 2 kg, 5 kg).
2. Registrar os valores de condutância correspondentes.
3. Executar uma regressão linear ($F = mC + b$) sobre os pares (condutância, força esperada).
4. Inserir os coeficientes resultantes nos parâmetros `factor_m` e `offset_b` do sistema.

**Implementação:**

```python
force = factor_m * conductance + offset_b
force = np.maximum(force, 0.0)  # Garante que forças negativas sejam zeradas
```

A operação `np.maximum(force, 0.0)` atua como um retificador, eliminando valores negativos que podem surgir do offset $b$ quando a condutância é muito baixa.

---

### 2.5 Filtragem de Ruído — Zona Morta

Em matrizes passivas de sensores, ocorre um fenômeno chamado **crosstalk** (corrente de fuga entre linhas/colunas adjacentes). Esse efeito gera leituras fantasmas de baixa amplitude em células que não estão sendo efetivamente pressionadas. Para eliminar esse ruído, aplica-se uma **zona morta (deadzone)**:

$$F(x,y) =
\begin{cases}
F(x,y), & \text{se } F(x,y) \geq T_{deadzone} \\
0, & \text{se } F(x,y) < T_{deadzone}
\end{cases}$$

Onde:
- $T_{deadzone}$ — limiar mínimo de força (valor padrão: $0.05\,N$)

Qualquer célula cuja força esteja abaixo do limiar é considerada **sem contato real** e zerada.

**Implementação:**

```python
force[force < deadzone] = 0.0
```

> **Nota para manutenção:** O valor de `deadzone_threshold` pode ser ajustado em tempo real pela interface gráfica. Valores muito altos cortam pressões legítimas; valores muito baixos deixam ruído residual visível no heatmap.

---

### 2.6 Massa Total

A massa total sobre a matriz é obtida pelo **somatório bidimensional** de todas as forças individuais, convertido de Newtons para quilogramas pela segunda lei de Newton:

$$M_{total} = \frac{\displaystyle\sum_{x=0}^{rows-1} \sum_{y=0}^{cols-1} F(x,y)}{g}$$

Onde:
- $F(x,y)$ — força em Newtons na célula $(x,y)$, já filtrada pela zona morta
- $g$ — aceleração gravitacional ($9.81\,m/s^2$)

**Implementação:**

```python
def compute_total_mass(force_matrix: np.ndarray) -> float:
    total_force = float(np.sum(force_matrix))
    return total_force / GRAVITY  # GRAVITY = 9.81
```

---

## 3. Algoritmo de Suavização — Filtro EMA

### 3.1 Problema

O pipeline matemático é executado a cada frame recebido (50–100 vezes por segundo). Mesmo com a zona morta, a leitura instantânea de massa exibe **variações frame-a-frame** (tremor/jitter) causadas por:

- Ruído elétrico intrínseco do ADC.
- Micro-oscilações mecânicas da superfície sob carga.
- Diferenças de timing entre ciclos de varredura.

Exibir o valor bruto diretamente na interface gráfica resulta em uma leitura numérica instável e ilegível para o usuário.

### 3.2 Solução — Média Móvel Exponencial (EMA)

O filtro de **Exponential Moving Average** suaviza a série temporal de peso, dando maior relevância às amostras recentes e atenuando progressivamente as anteriores. Diferente de uma média móvel simples (SMA), o EMA:

- **Não requer buffer de amostras** — utiliza apenas o valor atual e o anterior.
- **Responde mais rápido** a mudanças reais de carga.
- **Ocupa $O(1)$ de memória** — independe do tamanho da janela.

### 3.3 Formulação Matemática

$$EMA_t = \alpha \times X_t + (1 - \alpha) \times EMA_{t-1}$$

Onde:
- $EMA_t$ — valor suavizado no instante $t$
- $X_t$ — valor bruto (massa calculada) no instante $t$
- $EMA_{t-1}$ — valor suavizado no instante anterior
- $\alpha$ — constante de suavização ($0 < \alpha \leq 1$)

### 3.4 O Papel da Constante $\alpha$

A constante $\alpha$ controla o **compromisso entre responsividade e estabilidade**:

| $\alpha$         | Comportamento                                                             |
| :--------------- | :------------------------------------------------------------------------ |
| $\alpha \to 1.0$ | O filtro segue quase instantaneamente o sinal bruto (sem suavização).     |
| $\alpha \to 0.0$ | O filtro reage muito lentamente — alta inércia, alta estabilidade visual. |
| $\alpha = 0.3$   | **Valor padrão do sistema.** Balanço entre resposta rápida e leitura estável. |

**Interpretação prática:** Com $\alpha = 0.3$, cada nova amostra contribui com 30% do peso final, enquanto o histórico acumulado contribui com 70%. O resultado é uma curva que segue tendências reais de forma suave, sem oscilar a cada frame.

### 3.5 Implementação

```python
def apply_ema(current: float, previous: float, alpha: float) -> float:
    """Média Móvel Exponencial para suavização do peso exibido."""
    return alpha * current + (1.0 - alpha) * previous
```

**Invocação no loop principal** (`bridge.py`):

```python
smoothed_mass = apply_ema(raw_mass, previous_weight, snap["ema_alpha"])
previous_weight = smoothed_mass
```

O valor `previous_weight` é mantido como estado persistente entre iterações da thread de leitura, garantindo continuidade temporal do filtro.

---

## 4. Referência Rápida de Parâmetros

Tabela consolidada de todos os parâmetros configuráveis do pipeline, com valores padrão e unidades:

| Parâmetro             | Símbolo        | Valor Padrão | Unidade | Editável em Runtime |
| :-------------------- | :------------- | :----------- | :------ | :------------------ |
| Tensão de alimentação | $V_{CC}$       | 3.3          | V       | Sim                 |
| Resolução do ADC      | —              | 4095         | —       | Sim                 |
| Resistor de pull-down | $R_{pull}$     | 10 000       | Ω       | Sim                 |
| Fator de calibração   | $m$            | 1.0          | N·S     | Sim                 |
| Offset de calibração  | $b$            | 0.0          | N       | Sim                 |
| Limiar de zona morta  | $T_{deadzone}$ | 0.05         | N       | Sim                 |
| Constante EMA         | $\alpha$       | 0.3          | —       | Sim                 |
| Aceleração gravitac.  | $g$            | 9.81         | m/s²    | Não (constante)     |

---

## 5. Rastreabilidade de Código-Fonte

Mapeamento entre cada etapa do pipeline e o arquivo/função correspondente no repositório:

| Etapa do Pipeline         | Arquivo                          | Função                           |
| :------------------------ | :------------------------------- | :------------------------------- |
| Leitura Serial / Parsing  | `services/serial_reader.py`      | `SerialFrameReader.read_frame()` |
| ADC → Tensão → Força      | `services/math_pipeline.py`      | `compute_force_matrix()`         |
| Somatório → Massa (Kg)    | `services/math_pipeline.py`      | `compute_total_mass()`           |
| Filtro EMA                | `services/math_pipeline.py`      | `apply_ema()`                    |
| Orquestração do loop      | `bridge.py`                      | `_reading_loop()`                |
| Parâmetros de calibração  | `config/settings.py`             | `CalibrationParams`              |
| Monitor de postura        | `providers/posture_monitor.py`   | `PostureMonitor`                 |
| Persistência de tara      | `services/tare_store.py`         | `load_tare()` / `save_tare()`    |
| API REST + WebSocket      | `api/server.py`                  | FastAPI app                      |
| Entrypoint                | `main.py`                        | `main()`                         |
