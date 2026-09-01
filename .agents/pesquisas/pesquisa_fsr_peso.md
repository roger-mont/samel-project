# Pesquisa: Extração de Peso a partir de Mantas de Sensores FSR

## Resumo Executivo

Foram identificados **dois estudos-chave** e uma base ampla de literatura técnica que abordam diretamente a estimação de peso usando matrizes de sensores FSR/piezoresistivos. Ambos enfrentam limitações semelhantes às do projeto SAMEL (saída digital pré-processada, sem acesso direto à condutância, múltiplos setores/zonas na manta).

---

## 📄 Estudo 1 — DNN para Calibração de Manta Piezoresistiva (Mais relevante)

**Título:** *Ground Force Precision Calibration Method for Customized Piezoresistance Sensing Flexible Force Measurement Mat*

| Campo | Detalhe |
|---|---|
| **Autores** | Seo, Jeong-Woo; Kim, Hyeonjong; Kim, Jaeuk U; Do, Jun-Hyeong; Ko, Junghyuk |
| **Publicação** | Sensors (MDPI), vol. 24, nº 7, 2363 — 2024 |
| **DOI** | [10.3390/s24072363](https://doi.org/10.3390/s24072363) |
| **PDF** | [eScholarship (open access)](https://escholarship.org/content/qt4b9367wm/qt4b9367wm.pdf) |

### Problema Abordado

> Medições **não-normalizadas** causadas por **limitações físicas no hardware e no processamento de sinal** — ou seja, o sistema entrega valores já processados, não a resistência bruta de cada sensor.

> [!IMPORTANT]
> Este paper lida exatamente com o contratempo da manta SAMEL: os valores já chegam processados pelo hardware, sem acesso à condutância/resistência individual de cada FSR.

### Metodologia

1. **Manta piezoresistiva de grande área** (force plate flexível)
2. **Coleta de dados:** valores medidos em cada ponto da matriz + cálculo do **Centro de Pressão (CoP)** como feature adicional
3. **Modelo:** Regressão DNN (Deep Neural Network)
   - **Inputs:** vetor de valores digitais de cada sensor + coordenadas do CoP
   - **Output:** peso predito (kg)
4. **Calibração:** pesos conhecidos aplicados em diferentes posições da manta

### Modelo Matemático

```
Input  → [valor_sensor₁, valor_sensor₂, ..., valor_sensorₙ, CoP_x, CoP_y]
DNN    → camadas densas com ativação ReLU
Output → peso_predito (regressão contínua)
```

- A abordagem **substitui** a equação clássica (derivar a razão resistência/peso de cada sensor individualmente)
- O DNN aprende a compensar não-linearidades, variações entre sensores e limitações do ADC/hardware

### Resultados

| Métrica | Valor |
|---|---|
| Erro médio mínimo | **0.06%** |
| Erro médio máximo | **3.334%** |

### Como Replicar

1. Coletar leituras da manta com **pesos conhecidos** em diferentes posições
2. Calcular o CoP (centro de pressão) a partir da distribuição dos valores
3. Treinar uma DNN de regressão usando os valores brutos digitais + CoP como entrada e o peso real como target
4. Validar com conjunto de teste separado

---

## 📄 Estudo 2 — Smart Mat com FSR 16×8

**Título:** *Prediction of Body Weight of a Person Lying on a Smart Mat in Nonrestraint and Unconsciousness Conditions*

| Campo | Detalhe |
|---|---|
| **Autores** | Tae-Hwan Kim, Youn-Sik Hong |
| **Publicação** | Sensors (MDPI), vol. 20, nº 12, 3485 — 2020 |
| **DOI** | [10.3390/s20123485](https://doi.org/10.3390/s20123485) |

### Problema Abordado

Estimar peso corporal de pacientes idosos deitados em um smart mat, sem restrição de posição (decúbito dorsal, lateral, etc.), usando uma grade de 128 FSRs (16×8).

### Metodologia — Três Métodos de Extração de Features

| Método | Descrição |
|---|---|
| **Segmentação** | Divide a manta em zonas/setores e extrai features agregadas por zona (soma, média, máximo de cada setor) |
| **Soma cumulativa média** | Calcula a soma acumulada média dos valores de pressão de toda a manta |
| **Serialização** | Converte a matriz 2D inteira em um vetor 1D sequencial (flatten) e alimenta diretamente no modelo |

### Modelos Comparados

| Modelo | Descrição |
|---|---|
| **Regressão Linear** | Baseline |
| **DNN** | Rede neural profunda de regressão |
| **CNN** | Rede convolucional tratando a manta como "imagem" |
| **Random Forest** | Ensemble de árvores de decisão |

### Resultado Principal

| Melhor Combinação | MAE |
|---|---|
| **Serialização + DNN** | **±4.6 kg** (peso médio dos participantes: 72.9 kg → ~6.3% de erro) |

> [!NOTE]
> A abordagem de **segmentação** é particularmente relevante para a manta SAMEL com 8 setores — pode-se extrair features por setor e concatenar.

### Amostragem

- Participantes humanos em diferentes posturas sobre a manta
- Ground-truth: balança calibrada
- Múltiplas leituras por postura para reduzir variância

### Como Replicar

1. Para cada aquisição, capturar o frame completo (32×64 no caso SAMEL)
2. **Serialização:** achatar a matriz em vetor de 2048 valores
3. Treinar DNN de regressão com peso real como target
4. Alternativa: segmentar por setor (8 vetores de 256 valores) e agregar

---

## 📐 Métodos Matemáticos Consolidados

### 1. Soma Espacial (Baseline ingênuo)

$$W \approx \alpha \cdot \sum_{i=1}^{N} \sum_{j=1}^{M} V_{i,j} + \beta$$

Onde $V_{i,j}$ é o valor digital do sensor na posição $(i,j)$, e $\alpha$, $\beta$ são coeficientes de regressão linear.

**Limitação:** assume linearidade uniforme entre todos os sensores. Erro típico: 10–25%.

### 2. Regressão Polinomial

$$W \approx \beta_0 + \beta_1 S + \beta_2 S^2 + \beta_3 S^3$$

Onde $S = \sum V_{i,j}$ (soma dos valores). Ajuste por **mínimos quadrados** no dataset de calibração.

**Quando usar:** quando não se tem poder computacional para DNN e a distribuição de carga é razoavelmente uniforme.

### 3. DNN de Regressão (Estado da arte)

```
Input layer  → N sensores (+CoP opcionalmente)
Hidden layers → 2-4 camadas densas, 64-256 neurônios, ReLU
Output layer → 1 neurônio (peso predito), ativação linear
Loss         → MSE (Mean Squared Error)
Optimizer    → Adam
```

**Quando usar:** quando há dados de calibração suficientes (>50 amostras com pesos variados e posições variadas). Compensa automaticamente não-linearidade, variação entre sensores e limitações de hardware.

### 4. Centro de Pressão (Feature auxiliar)

$$CoP_x = \frac{\sum_{i,j} x_j \cdot V_{i,j}}{\sum_{i,j} V_{i,j}}, \quad CoP_y = \frac{\sum_{i,j} y_i \cdot V_{i,j}}{\sum_{i,j} V_{i,j}}$$

Incluir o CoP como feature adicional melhora a predição porque informa ao modelo **onde** o peso está concentrado.

---

## ⚠️ Análise dos Contratempos vs. Manta SAMEL

### Contratempo 1: Manta não informa condutância, somente valor digital já processado

| Aspecto | Situação na literatura |
|---|---|
| **Seo et al. (2024)** | ✅ Enfrentou exatamente isso. Criou método DNN justamente para contornar a falta de acesso à resistência bruta. |
| **Kim & Hong (2020)** | ✅ Trabalha com saída digital do sistema. A serialização dos valores brutos digitais foi o melhor método. |
| **Literatura geral** | A maioria assume acesso à condutância (1/R). Quando não disponível, a recomendação é tratar os valores digitais como **proxy proporcional monotônico** da força e usar regressão/ML. |

> [!TIP]
> **Implicação para SAMEL:** Você não precisa da condutância. Use os valores digitais diretamente como features. O DNN aprende a função de mapeamento implicitamente, incluindo não-linearidades do ADC e do hardware.

### Contratempo 2: Manta formada por 8 setores (matrizes 16×16), totalizando 32×64

| Aspecto | Situação na literatura |
|---|---|
| **Kim & Hong (2020)** | Utilizou segmentação por zonas como um dos métodos. Aplicável à estrutura de 8 setores. |
| **Seo et al. (2024)** | Manta de grande área — o princípio é o mesmo independente da topologia. |

> [!WARNING]
> **Atenção com junções entre setores:** se os 8 setores são lidos por ASICs/ADCs independentes, pode haver **offset e ganho diferentes entre setores**. Recomendação: incluir um passo de normalização por setor ou treinar o modelo com dados que cubram carga em diferentes setores para que o DNN aprenda a compensar.

### Abordagem sugerida para a estrutura 8-setores:

```
Setor 1 (16×16) → flatten → vetor de 256 valores
Setor 2 (16×16) → flatten → vetor de 256 valores
...
Setor 8 (16×16) → flatten → vetor de 256 valores

Concatenar → vetor de 2048 valores + CoP_x + CoP_y = 2050 features

→ DNN regressão → peso predito
```

---

## 📊 Tabela Comparativa dos Estudos

| Critério | Seo et al. (2024) | Kim & Hong (2020) |
|---|---|---|
| **Tipo de sensor** | Piezoresistivo (FSR) | FSR |
| **Topologia** | Manta de grande área | 16×8 grid (128 sensores) |
| **Acesso à resistência** | ❌ Valores processados | Valores via ADC |
| **Modelo principal** | DNN regressão | Serialização + DNN |
| **Features extras** | CoP (centro de pressão) | Segmentação, soma cumulativa |
| **Erro reportado** | 0.06% – 3.33% | ±4.6 kg (~6.3%) |
| **Aplicação** | Force plate biomecânica | Monitoramento de pacientes |
| **Open Access** | ✅ (eScholarship/CC BY 4.0) | ✅ (MDPI Sensors) |

---

## 🔗 Referências para Leitura Adicional

1. **Seo et al. (2024)** — *Ground Force Precision Calibration Method for Customized Piezoresistance Sensing Flexible Force Measurement Mat* — [DOI](https://doi.org/10.3390/s24072363) | [PDF](https://escholarship.org/content/qt4b9367wm/qt4b9367wm.pdf)
2. **Kim & Hong (2020)** — *Prediction of Body Weight of a Person Lying on a Smart Mat in Nonrestraint and Unconsciousness Conditions* — [DOI](https://doi.org/10.3390/s20123485)
3. *The Effect of Biomechanical Variables on Force Sensitive Resistor Error* — buscar no IEEE Xplore
4. *Calibration of Force Sensing Resistors for Static and Dynamic Applications* — buscar no Google Scholar
5. *Compensating for Hysteresis and Creep in FSR Sensor Arrays* — buscar no PubMed

---

## 💡 Recomendação para o Projeto SAMEL

Com base na literatura, a abordagem mais promissora para o contexto da manta SAMEL é:

1. **Coletar dataset de calibração** com pesos conhecidos em múltiplas posições na manta
2. **Subtrair baseline** (leitura sem carga) de cada frame
3. **Calcular CoP** como feature auxiliar
4. **Serializar** a matriz 32×64 em vetor de 2048 valores
5. **Treinar DNN de regressão** (ou iniciar com regressão polinomial como baseline)
6. **Validar** contra balança calibrada

> [!CAUTION]
> Não tente derivar condutância a partir dos valores digitais sem conhecer o circuito de condicionamento (divisor de tensão, resistor fixo, resolução do ADC). Os estudos mostram que trabalhar diretamente com os valores digitais e um modelo data-driven é mais robusto e produz menor erro.
