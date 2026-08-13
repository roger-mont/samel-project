# Fase 3 — Validação Formal e Seleção do Modelo Final

> **Status:** Aguardando Fase 2  
> **Pré-requisito:** Fase 2 concluída + dataset com ≥ 5 repetições por condição cobrindo 0–100 kg + modelo candidato selecionado na Fase 2  
> **Objetivo:** Validar formalmente o modelo selecionado com dados nunca vistos no ajuste, quantificar todos os fenômenos de erro, e integrar o algoritmo final ao sistema embarcado com documentação completa.

---

## Contexto

A Fase 3 é o encerramento formal da modelagem. A Metodologia §35 define que a fase só termina quando: o protocolo está documentado e reproduzível, as curvas individuais estão registradas, o método de integração está validado, as métricas estão calculadas em dados não usados no ajuste, e existe procedimento documentado de tara, calibração e recalibração.

O objetivo não é atingir perfeição — é saber quantificar e documentar os limites do sistema.

---

## Sprint 1 — Coleta do Dataset Completo de Validação

### Objetivo
Coletar dados suficientes para o split treino/validação/teste (Metodologia §24) e cobrindo todas as condições do protocolo.

### Volume mínimo necessário

| Condição | Repetições | Massas | Total de pontos |
|----------|-----------|--------|-----------------|
| Central, horizontal | 5 | 0, 10, 20, 30, 40, 50, 60, 80, 100 kg | 45 |
| Superior, horizontal | 3 | 10, 30, 50 kg | 9 |
| Inferior, horizontal | 3 | 10, 30, 50 kg | 9 |
| Esquerda, horizontal | 3 | 10, 30, 50 kg | 9 |
| Direita, horizontal | 3 | 10, 30, 50 kg | 9 |
| Histerese (crescente) | 3 | 0, 20, 40, 60, 80, 100 kg | 18 |
| Histerese (decrescente) | 3 | 100, 80, 60, 40, 20, 0 kg | 18 |
| Drift (carga fixa 120s) | 2 | 30 kg, 60 kg | 4 |
| **TOTAL** | | | **~121 pontos** |

### Massas de validação (NUNCA usadas no ajuste da Fase 2)
Conforme Metodologia §25: se o ajuste usou 0, 10, 20, 30, 40, 50, 60, 80, 100 kg, usar como validação: **15, 25, 35, 55, 70, 90 kg**.

> **Regra crítica:** As massas de validação devem ser separadas e rotuladas como `validation_set=true` no CSV antes de qualquer análise de modelo.

---

## Sprint 2 — Split Treino / Validação / Teste (Metodologia §24)

### Proporções
```
70% → Treinamento (ajuste de parâmetros do modelo)
20% → Validação   (seleção de modelo, ajuste de hiperparâmetros)
10% → Teste       (avaliação final do modelo selecionado — não participar de NENHUMA decisão antes)
```

### Regras do split
1. Estratificado por faixa de massa (não aleatório simples)
2. Estratificado por posição — cada posição deve estar em todos os splits
3. Histerese e drift ficam apenas no conjunto de teste
4. As massas de validação separadas (15, 25, 35, ... kg) vão automaticamente para o conjunto de teste

### Implementação
**`Python/scripts/split_dataset.py`**

```python
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("sessions/calib_completo.csv")

# Estratificar por faixa de massa (bins de 20 kg)
df["mass_bin"] = pd.cut(df["reference_kg"], bins=[0, 20, 40, 60, 80, 120])

# Split estratificado
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
for train_idx, val_test_idx in sss.split(df, df["mass_bin"]):
    train_df = df.iloc[train_idx]
    val_test_df = df.iloc[val_test_idx]

# Segundo split: 20/10 a partir do val_test
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
for val_idx, test_idx in sss2.split(val_test_df, val_test_df["mass_bin"]):
    val_df = val_test_df.iloc[val_idx]
    test_df = val_test_df.iloc[test_idx]

# Salvar splits
train_df.to_csv("sessions/split_train.csv", index=False)
val_df.to_csv("sessions/split_val.csv", index=False)
test_df.to_csv("sessions/split_test.csv", index=False)
```

---

## Sprint 3 — Avaliação Final dos Modelos Candidatos

### Modelos a comparar (vem da Fase 2)

| Modelo | Descrição | Implementado em |
|--------|-----------|-----------------|
| A | Soma direta de forças calibradas por bloco | `compute_model_a()` |
| B | Integração trapezoidal 2D | `compute_model_b()` |
| C | Regressão linear sobre m_fisica: `m̂ = a·m + b` | `apply_correction_c()` |
| D | Regressão multivariada por somas de bloco | `apply_correction_d()` |

### Métricas obrigatórias (Metodologia §26)

Para cada modelo, calcular em `split_val.csv`:

| Métrica | Fórmula | Meta |
|---------|---------|------|
| EA médio | `mean(\|m_real - m̂\|)` | < 2 kg |
| EP médio | `mean(\|m_real - m̂\| / m_real × 100)` | < 5% |
| MAE | `(1/N) Σ \|m_i - m̂_i\|` | < 1.5 kg |
| RMSE | `√[(1/N) Σ (m_i - m̂_i)²]` | < 2 kg |
| R² | `1 - SS_res/SS_tot` | > 0.95 |

### Estratificação das métricas (Metodologia §26)
Calcular as métricas **separadamente** por:
- Faixa de massa: [0–20], [20–40], [40–60], [60–80], [80–120] kg
- Posição: center, upper, lower, left, right
- Repetição: para verificar consistência

### Script de avaliação
**`Python/scripts/evaluate_all_models.py`**

```python
# Para cada modelo:
for model_name in ["A", "B", "C", "D"]:
    m_hat = predict_model(model_name, val_df)
    metrics = compute_metrics(val_df["reference_kg"], m_hat)
    plot_residuals(val_df["reference_kg"], m_hat, title=f"Modelo {model_name}")
    print(f"{model_name}: MAE={metrics.mae:.3f} kg  RMSE={metrics.rmse:.3f} kg  R²={metrics.r2:.4f}")
```

### Análise de resíduos (Metodologia §27)

Para o modelo finalista, gerar obrigatoriamente:

1. **Resíduo vs. massa real** — verificar heteroscedasticidade
2. **Resíduo vs. posição** — verificar dependência espacial
3. **Resíduo vs. repetição** — verificar deriva temporal
4. **Histograma dos resíduos** — verificar distribuição normal
5. **Mapa de calor de resíduos por bloco** — verificar sensores problemáticos

---

## Sprint 4 — Critério de Seleção do Modelo Final (Metodologia §31)

### Tabela de critérios (ordenados por prioridade)

| # | Critério | Método de avaliação |
|---|---------|---------------------|
| 1 | Menor RMSE no conjunto de validação | Métrica direta do Sprint 3 |
| 2 | Boa repetibilidade | CV das repetições < 3% por condição |
| 3 | Resíduos sem tendência sistemática | Análise visual + teste de Durbin-Watson |
| 4 | Robusto à posição da carga | RMSE por posição < 2× RMSE global |
| 5 | Robusto à faixa de massa | RMSE por faixa < 2× RMSE global |
| 6 | Histerese dentro de limites | Δ histerese < 1 kg em toda a faixa |
| 7 | Drift dentro de limites | Drift < 0.5 kg em 120s de carga fixa |
| 8 | Baixa complexidade computacional | Tempo de execução < 5ms por frame |
| 9 | Facilidade de recalibração | Procedimento documentado, executável por não-engenheiro |

### Regra de desempate
> Entre modelos com desempenho estatisticamente equivalente (diferença de RMSE < 0.1 kg), escolher o de menor complexidade. (Metodologia §31)

### Teste de significância estatística
Se dois modelos estiverem próximos, aplicar teste de Wilcoxon nos resíduos para verificar se a diferença é estatisticamente significativa (p < 0.05).

---

## Sprint 5 — Avaliação Final no Conjunto de Teste

### Regra absoluta
> O conjunto de teste (`split_test.csv`) **nunca pode ser consultado antes da seleção final do modelo**. Consultar antes invalida a avaliação.

### Procedimento

1. Confirmar que o modelo foi selecionado com base apenas em `split_train.csv` + `split_val.csv`
2. Executar o modelo selecionado em `split_test.csv` — apenas uma vez
3. Registrar as métricas finais — estas são os números definitivos do sistema
4. Se as métricas de teste forem muito piores que as de validação (RMSE cresce > 50%), suspeitar de overfitting e revisitar o Sprint 4

### Resultado esperado do relatório final

```
MODELO SELECIONADO: [A / B / C / D]
Dataset de teste: N=XX pontos

Métricas globais:
  MAE  = X.XX kg
  RMSE = X.XX kg
  R²   = 0.XX

Por faixa de massa:
  [0–20  kg]: MAE=X.XX  RMSE=X.XX
  [20–40 kg]: MAE=X.XX  RMSE=X.XX
  [40–60 kg]: MAE=X.XX  RMSE=X.XX
  [60–80 kg]: MAE=X.XX  RMSE=X.XX
  [80–120kg]: MAE=X.XX  RMSE=X.XX

Histerese máxima: X.XX kg (em XX kg de carga)
Drift em 120s:    X.XX kg
```

---

## Sprint 6 — Avaliação de Inclinação da Maca (Metodologia §29)

> **Pré-requisito:** Consolidar calibração horizontal antes de testar inclinação.

### Ângulos a testar
```
0°, 15°, 30°, 45°
```

### O que medir
Para cada ângulo, com a mesma massa de referência (ex: 30 kg central):
- Campo de pressão muda? (comparar heatmap)
- T_estab muda?
- Erro do modelo aumenta?
- Os coeficientes de calibração precisam ser diferentes por ângulo?

### Implementação no sistema
Se inclinação afetar significativamente o peso (> 2 kg de erro):
- Adicionar campo `inclination_deg` na UI (entrada manual conforme decisão do /grill-me)
- Carregar conjunto de coeficientes de calibração específico por ângulo
- Salvar por ângulo no `calibration.json`:

```json
{
  "inclination_profiles": {
    "0": { "blocks": { "1": {...}, ... } },
    "15": { "blocks": { "1": {...}, ... } },
    "30": { "blocks": { "1": {...}, ... } }
  }
}
```

---

## Sprint 7 — Integração do Modelo Final ao Sistema Embarcado

### Objetivo
Incorporar o algoritmo selecionado ao pipeline do `bridge.py` de forma limpa e parametrizável.

### Arquitetura final do pipeline (Metodologia §33)

```
S_ij(t) [0-255]
    → tara + deadzone
    → CalibData.matrix_to_force_field()     # campo 2D em Newton
    → compute_model_X()                     # modelo selecionado (A/B/C/D)
    → apply_correction()                    # correção C/D se aplicável
    → / 9.81                                # m_fisica em kg
    → EMA temporal                          # suavização
    → weight_lock                           # critério §13
    → m̂_final exibido na UI
```

### Verificação de performance computacional
O pipeline completo deve executar em < 5ms para garantir 40Hz sem atraso. Medir com:

```python
import time
t0 = time.perf_counter()
# ... pipeline completo ...
dt_ms = (time.perf_counter() - t0) * 1000
assert dt_ms < 5.0, f"Pipeline lento: {dt_ms:.1f}ms"
```

---

## Sprint 8 — Documentação Final e Procedimentos de Recalibração

### Documentos a produzir

#### `docs/calibracao_procedimento.md`
Passo a passo para operador não-técnico recalibrar o sistema:
1. Condições ambientais necessárias
2. Equipamentos necessários (pesos certificados, placa rígida)
3. Procedimento de tara
4. Como executar `calibrate_block.py`
5. Como interpretar o RMSE e decidir se a calibração está boa
6. Como salvar e ativar a nova calibração
7. Critério para decidir quando recalibrar (ex: drift > 2 kg após 3 meses)

#### `docs/modelo_final.md`
- Modelo selecionado com justificativa
- Coeficientes do modelo com unidades
- Métricas de precisão (do conjunto de teste)
- Limitações conhecidas
- Condições de validade (faixa de massa, posições, inclinação)

#### Campos de metadados no `calibration.json` (versão final)
```json
{
  "version": 3,
  "created_at": "2026-08-XX",
  "software_version": "X.X.X",
  "manta_width_m": 0.90,
  "manta_height_m": 1.80,
  "selected_model": "C",
  "model_metrics": {
    "mae_kg": 0.87,
    "rmse_kg": 1.12,
    "r2": 0.978,
    "test_set_n": 24
  },
  "correction_model_c": { "a": 0.983, "b": -0.42 },
  "blocks": { ... }
}
```

---

## Critério de Conclusão da Fase 3 (Metodologia §35)

- [ ] Protocolo experimental documentado e reproduzível (`docs/calibracao_procedimento.md`)
- [ ] Curvas de calibração individuais por bloco registradas (`calibration.json` v3)
- [ ] Método de integração implementado e validado no conjunto de teste
- [ ] Métricas de erro calculadas em dados **não utilizados no ajuste** (`split_test.csv`)
- [ ] Histerese quantificada para toda a faixa operacional
- [ ] Drift após estabilização medido e documentado
- [ ] Efeito de posição quantificado e o modelo é robusto a ele (RMSE por posição < 2× global)
- [ ] Estabilização temporal caracterizada — `T_estab` documentado por faixa de massa
- [ ] Inclinação avaliada e comportamento documentado
- [ ] Algoritmo final executando em < 5ms por frame no hardware alvo
- [ ] Modelo final selecionado com justificativa publicada em `docs/modelo_final.md`
- [ ] Procedimento de recalibração documentado para operador não-técnico
- [ ] Todos os scripts de análise versionados e reproduzíveis (`Python/scripts/`)
