# Fase 1 — Infraestrutura de Dados e Pipeline Base

> **Status:** A implementar  
> **Pré-requisito:** Nenhum  
> **Objetivo:** Preparar o sistema para coletar dados estruturados de calibração e ajustar o pipeline de peso para ser fisicamente coerente com a Metodologia.

---

## Contexto

A Fase 1 não altera a precisão do peso — ela prepara o terreno. Sem logging estruturado, as Fases 2 e 3 são impossíveis porque não teremos dados para ajustar nem validar os modelos avançados. Sem a conversão para Newtons, o Modelo B (integração 2D) não pode ser implementado corretamente.

---

## Sprint 1 — Session Logger CSV

### Objetivo
Registrar cada ponto de calibração em CSV com todos os campos exigidos pela Metodologia §11.

### Arquivo a criar
**`Python/services/session_logger.py`**

### Campos obrigatórios por linha (conforme Metodologia §11)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `timestamp_iso` | string | ISO 8601 com fuso (`2026-08-13T17:30:00-04:00`) |
| `session_id` | string | UUID v4 gerado uma vez por execução |
| `software_version` | string | Versão do software Python |
| `block_id` | int | Bloco avaliado (1–8, `null` para frame completo) |
| `reference_kg` | float | Massa real de referência (`null` se desconhecida) |
| `reference_n` | float | `reference_kg × 9.81` em Newtons |
| `position_tag` | string | `"center"` / `"upper"` / `"lower"` / `"left"` / `"right"` / `"full_body"` |
| `repetition` | int | Número da repetição no ensaio |
| `raw_sum_block` | float | Soma bruta do bloco-alvo |
| `net_sum_block` | float | `raw_sum_block - tare_block_sum` |
| `mean_raw_block` | float | Média da soma durante o período de coleta |
| `std_raw_block` | float | Desvio padrão durante a coleta |
| `cv_pct` | float | `(std / mean) × 100` |
| `estimated_kg` | float | Peso estimado pelo modelo atual |
| `estimated_n` | float | `estimated_kg × 9.81` |
| `stability_state` | string | `"transient"` ou `"stable"` |
| `time_since_load_s` | float | Segundos desde aplicação da carga |
| `tare_block_sum` | float | Valor de tara usado neste ponto |

### Comportamento esperado

```python
# Durante calibração (calibrate_block.py) — linha por ponto de peso:
logger.log_calibration_point(
    session_id=session_id,
    block_id=block_id,
    reference_kg=kg,
    position_tag=position_tag,
    repetition=repetition,
    raw_sum=raw_sum,
    net_sum=net_sum,
    mean=mean,
    std=std,
    cv_pct=cv_pct,
    estimated_kg=estimated_kg,
)
```

### Estrutura de arquivos gerados

```
Projeto Walter/
  sessions/
    calib_20260813_bloco1_central.csv
    calib_20260813_bloco2_central.csv
    calib_20260814_bloco1_histerese.csv
```

### Regras de implementação

- Arquivo CSV aberto em modo `append` — nunca sobrescreve dados existentes.
- Cabeçalho escrito apenas se o arquivo for novo (verificar com `Path.stat().st_size == 0`).
- `session_id` gerado com `uuid.uuid4()` uma vez por execução.
- Erros de I/O são logados e nunca derrubam o sistema principal.
- O módulo é injetado no `calibrate_block.py` como dependência opcional.

### Integração com `calibrate_block.py`

No início da sessão, o CLI pergunta:
```
Posição da carga (center/upper/lower/left/right/full_body): center
Número desta repetição: 1
```

Esses valores são salvos em todas as linhas da sessão.

---

## Sprint 2 — Conversão do Pipeline para Newtons

### Objetivo
A curva de calibração passa a produzir **Newtons**. A conversão `F/g = m` é um passo explícito e auditável no pipeline.

### Motivação
A Metodologia §4.4–4.5 exige:
```
F_total = Σ F_ij    (em Newtons)
m_física = F_total / g
```

Hoje o sistema vai direto de `raw_sum → kg`, pulando a grandeza de força. Isso impede a integração 2D (Fase 2) e os coeficientes da curva não têm interpretação física.

### Mudanças necessárias

#### `calibrate_block.py`
- O ajuste polinomial passa a usar `n_values = kg_values × 9.81` como variável dependente
- Curva ajustada: `net_sum → N`
- RMSE reportado em Newtons **e** convertido para kg no relatório final
- Salvar `"unit": "newton"` e `"coefficients_raw_to_n"` no JSON

#### `calibration_store.py`
- `_BlockCalib.sum_to_newton(raw_sum) → float`: novo método primário
- `_BlockCalib.sum_to_kg(raw_sum) → float`: chama `sum_to_newton / 9.81`
- `CalibData.matrix_to_newton(matrix) → float`: somatório em N
- `CalibData.matrix_to_kg(matrix) → float`: `matrix_to_newton / 9.81`

#### `math_pipeline.py`
- `compute_total_force(matrix, calib) → float`: retorna F_total em N
- `compute_total_mass(matrix, calib) → float`: `compute_total_force(matrix, calib) / 9.81`
- Log no terminal exibe: `F_total=XXX.X N  m=XX.XX kg`

#### `calibration.json` — novo schema (v3)
```json
{
  "version": 3,
  "blocks": {
    "1": {
      "unit": "newton",
      "coefficients_raw_to_n": [2.29e-06, 0.00687, 4.08],
      "tare_block_sum": 241.46,
      "rmse_n": 4.28,
      "rmse_kg": 0.44,
      "calibration_points": [
        { "kg": 0.0,  "n": 0.0,   "raw_sum": 241.46 },
        { "kg": 0.88, "n": 8.63,  "raw_sum": 715.1  },
        { "kg": 1.19, "n": 11.68, "raw_sum": 1168.5 }
      ]
    }
  }
}
```

### Script de migração
**`Python/scripts/migrate_calib_v2_to_v3.py`**

```python
# Converte calibration.json de v2 (kg) para v3 (newton) sem perda de dados
for block in data["blocks"].values():
    for pt in block["calibration_points"]:
        pt["n"] = pt["kg"] * 9.81
    # Re-ajusta curva com pontos em Newton
    net_sums = np.array([p["raw_sum"] for p in pts]) - block["tare_block_sum"]
    n_vals   = np.array([p["n"] for p in pts])
    block["coefficients_raw_to_n"] = np.polyfit(net_sums, n_vals, deg=2).tolist()
    block["unit"] = "newton"
data["version"] = 3
```

---

## Sprint 3 — Weight Lock alinhado ao critério §13 da Metodologia

### Critério da Metodologia §13
```
|y(t) - y(t - Δt)| < ε   satisfeito durante T_min consecutivo
```

### Critério atual (existente no sistema)
```python
variance(últimos N frames) < VARIANCE_THRESHOLD
AND drift_por_segundo < DRIFT_THRESHOLD
```

### Implementação proposta — critérios combinados

```
Frame é "estável" se TODOS abaixo:
  1. |m(t) - m(t-1)| < STABILITY_EPSILON_KG         ← Metodologia §13
  2. variance(últimos N frames) < STABILITY_VARIANCE  ← critério atual
  3. |m(t) - m(t - N)| / (N/fps) < STABILITY_DRIFT   ← critério atual

Weight lock dispara após STABILITY_TMIN_S consecutivos estáveis.
```

### Novos parâmetros em `settings.py`
```python
STABILITY_EPSILON_KG: float = 0.3       # Δm máximo entre frames (Metodologia §13)
STABILITY_TMIN_S: float = 10.0          # janela mínima para travar
STABILITY_VARIANCE_KG2: float = 0.25    # variância máxima da janela
STABILITY_DRIFT_KG_S: float = 0.15      # drift máximo em kg/s
```

### Novos campos no estado do `bridge.py`
```python
_stable_consecutive_s: float = 0.0   # segundos consecutivos estáveis
_weight_locked: bool = False          # peso travado ou não
_locked_weight_kg: float = 0.0        # valor travado
```

### Novos campos em `get_sensor_data()`
```python
{
  ...,
  "locked_weight_kg": _locked_weight_kg,
  "is_locked": _weight_locked,
  "stable_progress_pct": min(100, _stable_consecutive_s / STABILITY_TMIN_S * 100),
}
```

### UI — mudanças visuais (`index.html` + `app.js`)

- **Peso em tempo real:** número menor, cinza, etiqueta "Leitura atual"
- **Peso travado:** número grande, verde, etiqueta "PESO ESTÁVEL ✓" — visível apenas quando `is_locked = true`
- **Barra de progresso de estabilidade:** 0% → 100% em direção a `STABILITY_TMIN_S`
- **Reset do lock:** ao detectar `is_locked = false` após ter travado, piscar brevemente antes de sumir

---

## Sprint 4 — Protocolo Experimental Mínimo

### Checklist pré-sessão (Metodologia §6 e §7)
```
[ ] Registrar versão do software
[ ] Aguardar 5 min de aquecimento eletrônico
[ ] Verificar comunicação de todos os 8 blocos
[ ] Maca na posição horizontal (0°)
[ ] Executar tara com manta + colchão instalados
[ ] Confirmar que baseline tem cv% < 5% por 10s
[ ] Iniciar somente então a sequência de cargas
```

### Sequência mínima de ensaios
| Ordem | Massa (kg) | Posição | Reps | Tipo |
|-------|-----------|---------|------|------|
| 1 | 0 (tara) | — | 1 | baseline |
| 2 | 5 | central | 3 | calibração crescente |
| 3 | 10 | central | 3 | calibração crescente |
| 4 | 15 | central | 3 | calibração crescente |
| 5 | 20 | central | 3 | calibração crescente |
| 6 | 10 | superior | 3 | efeito de posição |
| 7 | 10 | inferior | 3 | efeito de posição |
| 8 | 10 | esquerda | 3 | efeito de posição |
| 9 | 15 | central | 1 | histerese (descida) |
| 10 | 10 | central | 1 | histerese (descida) |
| 11 | 5 | central | 1 | histerese (descida) |
| 12 | 0 | — | 1 | verificação de zero / drift |

### Procedimento por ponto de carga (Metodologia §10)
1. Posicionar carga na posição definida
2. Aguardar `is_locked = true` na UI (ou 30s no máximo)
3. Confirmar que o logger CSV registrou o ponto
4. Remover carga e aguardar retorno ao zero (< 5% do baseline)
5. Aguardar 15s antes da próxima carga

---

## Sprint 5 — Atualização do `context.md`

### Edições necessárias

| Campo | Atual (incorreto) | Correto |
|-------|-------------------|---------|
| Topologia | "protótipo 3×3 → 16×16" | "WangYing USB HID, 32×64, 8 blocos de 16×16" |
| Dados brutos | "ADC 0–1023 (10-bit)" | "valores 0–255 pré-processados pelo firmware HID" |
| Circuito | "pull-down 10kΩ, VCC" | Remover — não acessível pelo software |
| Pipeline | Fórmulas V_out, R_FSR, C | "calibração empírica por bloco: raw_sum → N via polinômio grau 2" |
| Taxa | "50 a 100Hz" | "40Hz (25ms/frame)" |
| Painel UI | "VCC, Resolução ADC, R_fixo, M, B" | "limiar de deadzone, EMA alpha, epsilon de estabilidade, TMIN" |

---

## Critério de Conclusão da Fase 1

- [ ] `session_logger.py` implementado e integrado ao `calibrate_block.py`
- [ ] Pelo menos 1 sessão de calibração com CSV gerado e validado manualmente
- [ ] Pipeline interno usando Newtons (divisão por 9.81 explícita)
- [ ] `calibration.json` atualizado para v3 com campo `"unit": "newton"`
- [ ] Weight lock com critério `|Δm| < ε` por `T_min` implementado e testado
- [ ] UI exibindo peso travado vs. transitório com barra de progresso
- [ ] `context.md` atualizado com dados corretos do WangYing
- [ ] Protocolo experimental realizado ao menos 1x e CSV resultante analisado
