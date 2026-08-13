# Fase 2 — Integração 2D e Modelos de Correção

> **Status:** Aguardando Fase 1  
> **Pré-requisito:** Fase 1 concluída + ao menos 3 sessões de calibração em CSV cobrindo ≥ 3 posições e faixa de 0–20 kg  
> **Objetivo:** Refinar o cálculo de peso substituindo a soma direta por integração numérica 2D e adicionar modelos de correção estatística para reduzir erros sistemáticos.

---

## Contexto

Com o pipeline da Fase 1 em Newton e os dados de calibração estruturados em CSV, a Fase 2 implementa os modelos A, B, C e D da Metodologia. O objetivo é comparar cada modelo nos dados de validação e selecionar o melhor antes da Fase 3.

A ordem de implementação é deliberada: começar com o modelo mais simples (A — soma direta) como baseline físico, depois adicionar complexidade apenas quando ela produzir ganho mensurável nos dados de validação.

---

## Sprint 1 — Modelo A: Soma Direta de Forças Calibradas (Baseline Físico)

### Definição (Metodologia §15)
```
F_A = Σ_k  F_k_newton(raw_sum_k)
m_A = F_A / 9.81
```

### Status atual
O sistema **já implementa** essencialmente o Modelo A, mas em kg ao invés de Newtons. Após o Sprint 2 da Fase 1 (conversão para Newton), o Modelo A estará formalmente implementado.

### O que falta para o Modelo A ser o baseline oficial

1. **Nomear explicitamente no código:** `compute_model_a(matrix, calib) → float` retornando kg
2. **Calcular e logar as métricas no CSV:** EA, EP, RMSE para cada ponto de referência conhecido
3. **Validar com dados de posições não treinadas** (usando os CSVs de ensaio de posição do Sprint 4 da Fase 1)

### Script de análise (offline, não no sistema embarcado)
**`Python/scripts/evaluate_model_a.py`**

```python
# Lê o CSV de sessão, para cada linha com reference_kg:
#   EA  = |reference_kg - estimated_kg|
#   EP  = EA / reference_kg * 100
# Estratifica por: faixa de massa, position_tag, repetição
# Exporta relatório de métricas em CSV e gráfico residuos
```

### Métricas esperadas (conforme Metodologia §26)
- EA (Erro Absoluto) médio por faixa de massa
- EP (Erro Percentual) médio
- MAE global
- RMSE global
- R² do ajuste
- Gráfico de resíduos `e_i = m_real - m_estimada` vs. `m_real`

---

## Sprint 2 — Estudo da Estabilização Temporal (Metodologia §12)

### Objetivo
Quantificar quanto tempo o sensor leva para estabilizar após aplicação da carga. Necessário para definir o `T_min` correto do weight lock e potencialmente implementar predição antecipada (Fase 3).

### Pontos de análise obrigatórios (Metodologia §12)
```
t = 0, 5, 10, 15, 20, 30, 45, 60, 90, 120 s
```

### O que o sistema já tem
O `session_logger.py` (Fase 1) registra `time_since_load_s`. Se o logger estiver rodando continuamente durante o experimento, os dados temporais já estarão no CSV.

### Script de análise temporal
**`Python/scripts/analyze_stabilization.py`**

```python
# Para cada ensaio (session_id + reference_kg + repetition):
#   Plota m_estimada vs. time_since_load_s
#   Detecta T_estab = primeiro t onde |m(t) - m_final| < ε para ≥ 10s
#   Calcula: T_estab médio, desvio padrão, dependência com massa
```

### O que medir e reportar
| Métrica | Definição |
|---------|-----------|
| T_estab | Tempo até `\|m(t) - m_final\| < ε` por 10s consecutivos |
| T_estab_mean | Média sobre todas as repetições |
| T_estab_std | Desvio padrão |
| Histerese_delta_kg | `m(carga crescente) - m(carga decrescente)` para mesma referência |
| Drift_após_estab | `m(t=120s) - m(t=T_estab)` por unidade de tempo |

### Resultado esperado
Confirmar se T_min = 10s (usado na Fase 1) é adequado ou precisa ser ajustado. Se o sistema leva 40s para estabilizar sistematicamente, o weight lock precisa de `STABILITY_TMIN_S = 40`.

---

## Sprint 3 — Modelo B: Integração Numérica 2D Trapezoidal

### Definição (Metodologia §4.4 e §16)
```
F_B ≈ Σ_i Σ_j  F_ij × ΔA_ij   (integração trapezoidal 2D)
m_B = F_B / 9.81
```

### Como converter o pixel para força

Com o Modelo A, temos `F_k_newton` para cada bloco (soma de todos os pixels do bloco). Para distribuir por pixel:

```python
# F_pixel_ij = F_bloco_k / N_pixels_ativos_k
# N_pixels_ativos_k = count(matrix[row_sl, col_sl] > deadzone)
# Se N_ativos = 0, F_pixel = 0
```

Essa distribuição é uma aproximação — assume uniformidade dentro do bloco. Isso é uma limitação conhecida da granularidade de calibração por bloco.

### Área por pixel `ΔA`

Necessita das dimensões físicas reais da manta:
```
ΔA = (largura_manta_m / 64) × (altura_manta_m / 32)
```

> **Decisão de design:** as dimensões físicas da manta devem ser medidas e registradas em `settings.py` como `MANTA_WIDTH_M` e `MANTA_HEIGHT_M`. Valores de referência típicos: `0.90m × 1.80m` (maca hospitalar padrão).

### Implementação

**`Python/services/math_pipeline.py`** — nova função:

```python
def compute_model_b(
    force_matrix_n: np.ndarray,   # matriz 32×64 em Newton/pixel (já distribuída)
    pixel_area_m2: float,
) -> float:
    """Integração trapezoidal 2D do campo de força.

    Usa scipy.integrate.trapezoid aninhado.
    Retorna F_total em Newton.
    """
    from scipy.integrate import trapezoid
    dx = np.sqrt(pixel_area_m2)  # simplificação: pixels quadrados
    f_rows = trapezoid(force_matrix_n, dx=dx, axis=1)
    return float(trapezoid(f_rows, dx=dx))
```

**`Python/services/calibration_store.py`** — novo método:

```python
def matrix_to_force_field(self, matrix: np.ndarray) -> np.ndarray:
    """Retorna matriz 32×64 com força em Newton por pixel."""
    force_field = np.zeros_like(matrix, dtype=np.float64)
    for bid, (row_sl, col_sl) in BLOCK_REGIONS.items():
        calib = self._resolve(bid)
        if calib is None:
            continue
        block = matrix[row_sl, col_sl]
        n_ativos = int(np.count_nonzero(block))
        if n_ativos == 0:
            continue
        f_total_block = calib.sum_to_newton(float(block.sum()))
        force_field[row_sl, col_sl] = block / block.sum() * f_total_block
    return force_field
```

### Comparação Modelo A vs. Modelo B

Após implementar ambos, rodar `evaluate_model_ab.py` nos dados de validação:

| Métrica | Modelo A | Modelo B | Δ |
|---------|----------|----------|---|
| MAE (kg) | ? | ? | ? |
| RMSE (kg) | ? | ? | ? |
| R² | ? | ? | ? |

> **Hipótese:** O Modelo B produzirá ganho real apenas se houver cargas muito concentradas em bordas ou cantos. Para carga central bem distribuída, A ≈ B.

---

## Sprint 4 — Modelo C: Regressão Linear de Correção (Metodologia §17)

### Definição
```
m̂ = a × m_fisica + b
```

Corrige ganho global (`a`) e offset sistemático (`b`) do sistema.

### Quando usar
Quando a análise dos resíduos do Modelo A ou B mostrar:
- Erro sistemático proporcional à massa (indica `a ≠ 1`)
- Offset constante em toda a faixa (indica `b ≠ 0`)

### Como ajustar

```python
# Python/scripts/fit_model_c.py
from sklearn.linear_model import LinearRegression

# X: m_fisica dos dados de TREINAMENTO (70%)
# y: m_real dos dados de TREINAMENTO
model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)
a, b = model.coef_[0], model.intercept_

# Avaliar em dados de VALIDAÇÃO (30%)
m_hat_val = model.predict(X_val.reshape(-1, 1))
rmse_c = np.sqrt(mean_squared_error(y_val, m_hat_val))
```

### Split treino/validação (Metodologia §24)

```
70% dos pontos → ajuste de a, b
30% dos pontos → avaliação das métricas
```

O split deve ser **estratificado por faixa de massa** — não aleatório simples — para garantir que todas as faixas estejam representadas em ambos os conjuntos.

### Implementação no pipeline

Se Modelo C for selecionado, os coeficientes `a` e `b` são salvos no `calibration.json`:

```json
{
  "version": 3,
  "correction_model_c": {
    "a": 0.983,
    "b": -0.42,
    "rmse_validation_kg": 0.21,
    "trained_on_n_samples": 45,
    "validated_on_n_samples": 19
  },
  "blocks": { ... }
}
```

E aplicado no `math_pipeline.py`:

```python
def apply_correction_c(m_fisica: float, calib: CalibData) -> float:
    if calib.correction_c is None:
        return m_fisica
    a, b = calib.correction_c
    return max(0.0, a * m_fisica + b)
```

---

## Sprint 5 — Modelo D: Regressão Multivariada por Features de Bloco (Metodologia §18)

### Definição
```
m̂ = b0 + b1·S1 + b2·S2 + ... + b8·S8
```

Onde `S_k` é a soma bruta (ou net_sum) de cada bloco.

### Quando usar
- Apenas se o Modelo C apresentar resíduos sistemáticos com dependência espacial (ex: peso subestimado quando carga está no bloco 5 mas não no bloco 1)
- Se a análise de posição mostrar erro > 1.5 kg dependendo de onde a carga está

### Como ajustar

```python
# Python/scripts/fit_model_d.py
# X: [S1, S2, S3, S4, S5, S6, S7, S8] por amostra (8 features)
# y: m_real

from sklearn.linear_model import Ridge  # Ridge para evitar overfitting

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)
# alpha = parâmetro de regularização, ajustar via cross-validation
```

### Risco de overfitting
Com apenas 8 features e potencialmente poucos dados, o Modelo D pode sobreajustar. Mitigações:
- Usar regularização Ridge (L2)
- Avaliar obrigatoriamente no conjunto de validação
- Só adotar D se `RMSE_D_val < RMSE_C_val - 0.1 kg` (ganho mínimo de 100g)

---

## Sprint 6 — Compensação de Crosstalk (Metodologia §21)

### O que é crosstalk
Quando uma carga é aplicada em um sensor, sensores vizinhos respondem levemente (por flexão mecânica da manta, capacitância parasita, ou correntes de fuga).

### Como medir
1. Aplicar carga localizada em 1 bloco isolado (ex: bloco 1, 5 kg)
2. Medir a resposta dos outros 7 blocos (deveria ser zero)
3. Repetir para cada bloco como "fonte"

### Resultado esperado
Matriz de crosstalk `C` (8×8):
```
C_kl = S_l_observado / S_k_esperado   (com l ≠ k, carga apenas em bloco k)
```

### Implementação (se crosstalk for significativo)
```python
F_corr = C_inv @ F_medido   # onde C_inv é a inversa da matriz de crosstalk
```

> **Hipótese:** Para a manta FSR em colchão hospitalar, o crosstalk entre blocos não-adjacentes deve ser < 2–3%. O estudo deve confirmar se a compensação vale a complexidade adicionada.

---

## Sprint 7 — Análise de Histerese (Metodologia §22)

### O que medir
Para cada massa de referência, comparar a leitura durante carregamento crescente vs. decrescente:

```
Histerese_delta_kg = m_estimada(crescente) - m_estimada(decrescente)
para a mesma m_referencia
```

### Dados necessários
Os ciclos crescente e decrescente foram coletados no Sprint 4 da Fase 1 (pontos 2→5 e 9→11 da tabela de ensaios).

### Script de análise
**`Python/scripts/analyze_hysteresis.py`**

```python
# Para cada massa (5, 10, 15 kg):
#   crescente = pontos com sequence_type == "ascending"
#   decrescente = pontos com sequence_type == "descending"
#   Δ = mean(crescente) - mean(decrescente)
#   H% = Δ / m_real × 100
```

### Ação conforme resultado
| Δ | Ação |
|----|------|
| < 0.3 kg | Ignorar — dentro da incerteza |
| 0.3–1.0 kg | Documentar; aplicar fator de correção médio |
| > 1.0 kg | Implementar curvas separadas de carga/descarga ou flag ao usuário |

---

## Critério de Conclusão da Fase 2

- [ ] Modelo A avaliado com métricas em dados de validação (CSV)
- [ ] Estabilização temporal caracterizada — `T_estab` médio documentado
- [ ] Weight lock `T_min` ajustado com base nos dados reais
- [ ] Modelo B implementado e comparado ao Modelo A
- [ ] Split 70/30 treino/validação implementado nos scripts de análise
- [ ] Modelo C ajustado e avaliado no conjunto de validação
- [ ] Modelo D avaliado apenas se Modelo C não atingir a meta de precisão
- [ ] Crosstalk medido e documentado (mesmo que zero)
- [ ] Histerese quantificada para as massas de referência disponíveis
- [ ] Modelo escolhido para a Fase 3 documentado com justificativa e métricas
